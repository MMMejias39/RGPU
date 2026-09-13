//! Simulação de vetor de estado quântico em GPU.
//!
//! # O que é o vetor de estado
//!
//! Um registrador de `n` qubits é um vetor de `2ⁿ` amplitudes complexas. Não há
//! atalho: a representação é exponencial, e é por isso que simular qubits num
//! computador clássico **custa mais** que a conta clássica equivalente, nunca
//! menos.
//!
//! O que existe de aproveitável é o inverso: o laço interno de um simulador é
//! contração tensorial, exatamente o que o [`rtensor`] já faz bem.
//!
//! | Qubits | Memória (`complex64`) |
//! |---:|---:|
//! | 20 | 8,4 MB |
//! | 25 | 268 MB |
//! | 28 | 2,1 GB |
//! | 29 | 4,3 GB |
//! | 30 | 8,6 GB |
//!
//! Cada qubit dobra tudo. Numa placa de 8 GB o teto é **29 qubits**.
//!
//! # Por que isto pertence ao RGPU
//!
//! Aplicar uma porta a um vetor de estado lê duas amplitudes, multiplica por
//! uma matriz minúscula e escreve duas de volta: **0,875 flop por byte**. É
//! carga limitada por banda de memória — o regime **oposto** ao do GEMM, que
//! mede 4,3 TFLOP/s contra um teto aritmético de 15 a 18.
//!
//! Isso torna a simulação quântica o teste natural para técnicas que no GEMM
//! não renderam. A precisão mista, por exemplo, foi medida como inútil lá
//! justamente por não ser banda o gargalo; aqui deveria render.
//!
//! E há espaço aberto: os simuladores existentes reportam tempo, nunca joules.

use bytemuck::{Pod, Zeroable};
use rtensor::gpu::Gpu;

/// Amplitude complexa: parte real e imaginária.
pub type Complexo = (f32, f32);

/// Porta de um qubit: matriz 2×2 complexa, em ordem de linha.
#[derive(Clone, Copy, Debug)]
pub struct Porta1 {
    pub u00: Complexo,
    pub u01: Complexo,
    pub u10: Complexo,
    pub u11: Complexo,
}

impl Porta1 {
    /// Hadamard: leva |0⟩ e |1⟩ a superposições iguais.
    pub fn hadamard() -> Porta1 {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        Porta1 {
            u00: (s, 0.0),
            u01: (s, 0.0),
            u10: (s, 0.0),
            u11: (-s, 0.0),
        }
    }

    /// Pauli-X, a negação quântica.
    pub fn x() -> Porta1 {
        Porta1 {
            u00: (0.0, 0.0),
            u01: (1.0, 0.0),
            u10: (1.0, 0.0),
            u11: (0.0, 0.0),
        }
    }

    /// Pauli-Z: inverte a fase de |1⟩.
    pub fn z() -> Porta1 {
        Porta1 {
            u00: (1.0, 0.0),
            u01: (0.0, 0.0),
            u10: (0.0, 0.0),
            u11: (-1.0, 0.0),
        }
    }

    /// Rotação em torno de Y por `theta`.
    pub fn ry(theta: f32) -> Porta1 {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        Porta1 {
            u00: (c, 0.0),
            u01: (-s, 0.0),
            u10: (s, 0.0),
            u11: (c, 0.0),
        }
    }

    /// Deslocamento de fase por `theta` em |1⟩.
    pub fn fase(theta: f32) -> Porta1 {
        Porta1 {
            u00: (1.0, 0.0),
            u01: (0.0, 0.0),
            u10: (0.0, 0.0),
            u11: (theta.cos(), theta.sin()),
        }
    }

    /// Aplica a porta na CPU — a referência contra a qual a GPU é conferida.
    pub fn aplicar_cpu(&self, psi: &mut [Complexo], qubit: usize) {
        let mascara = (1usize << qubit) - 1;
        for p in 0..psi.len() / 2 {
            let i0 = ((p >> qubit) << (qubit + 1)) | (p & mascara);
            let i1 = i0 | (1 << qubit);
            let (a0, a1) = (psi[i0], psi[i1]);
            psi[i0] = soma(mul(self.u00, a0), mul(self.u01, a1));
            psi[i1] = soma(mul(self.u10, a0), mul(self.u11, a1));
        }
    }
}

fn mul(a: Complexo, b: Complexo) -> Complexo {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn soma(a: Complexo, b: Complexo) -> Complexo {
    (a.0 + b.0, a.1 + b.1)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PortaP {
    pares: u32,
    qubit: u32,
    gx: u32,
    pad: u32,
    /// `(Re u00, Im u00, Re u01, Im u01)`.
    u0: [f32; 4],
    /// `(Re u10, Im u10, Re u11, Im u11)`.
    u1: [f32; 4],
}

/// Precisão de armazenamento das amplitudes.
///
/// A aritmética da porta é sempre `f32`; o que muda é como as amplitudes ficam
/// guardadas entre uma porta e a seguinte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precisao {
    /// `complex64`: dois `f32`, 8 bytes por amplitude.
    F32,
    /// `complex32`: dois `f16` empacotados numa palavra, 4 bytes por amplitude.
    ///
    /// Metade do tráfego numa carga que mede 83% da banda da placa, e **dois
    /// qubits a mais** dentro do limite de 2 GB por binding. O custo é
    /// precisão: `f16` guarda ~3 dígitos decimais, e o erro se acumula a cada
    /// porta — `tests/porta1.rs` mede quanto.
    F16,
}

impl Precisao {
    pub fn bytes_por_amplitude(&self) -> usize {
        match self {
            Precisao::F32 => 8,
            Precisao::F16 => 4,
        }
    }

    /// Teto de qubits imposto pelo limite de binding do WebGPU.
    ///
    /// O limite é 2 GB **menos 4 bytes** — `2147483644`. Um estado de 29 qubits
    /// em meia precisão ocupa exatamente 2 GB e falha por esses 4 bytes; o de 28
    /// qubits em precisão simples, pelo mesmo motivo. Medido, não estimado.
    ///
    /// Meia precisão compra **um** qubit, não dois: a conta dobra, mas o teto
    /// também é uma potência de dois.
    pub fn max_qubits(&self) -> usize {
        match self {
            Precisao::F32 => 27,
            Precisao::F16 => 28,
        }
    }
}

/// Kernel de porta de um qubit.
///
/// Cada thread cuida de **um par** de amplitudes: os índices que diferem apenas
/// no bit do qubit alvo. São `2ⁿ⁻¹` pares, e o par `p` mapeia para
///
/// ```text
/// i₀ = (p >> q) << (q+1) | (p & (2^q − 1))     i₁ = i₀ | 2^q
/// ```
///
/// que percorre todos os índices sem repetição nem lacuna.
///
/// Lê 2 amplitudes e escreve 2: 32 bytes por par, contra ~28 flops. Limitado
/// por banda, como toda simulação de vetor de estado.
const PORTA1: &str = r#"
struct P {
    pares: u32, qubit: u32, gx: u32, pad: u32,
    u0: vec4<f32>,
    u1: vec4<f32>,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> psi: array<vec2<f32>>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn porta1(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let par = (w.y * p.gx + w.x) * 256u + l.x;
    if (par >= p.pares) { return; }

    let mascara = (1u << p.qubit) - 1u;
    let i0 = ((par >> p.qubit) << (p.qubit + 1u)) | (par & mascara);
    let i1 = i0 | (1u << p.qubit);

    let a0 = psi[i0];
    let a1 = psi[i1];
    psi[i0] = cmul(p.u0.xy, a0) + cmul(p.u0.zw, a1);
    psi[i1] = cmul(p.u1.xy, a0) + cmul(p.u1.zw, a1);
}
"#;

/// Mesma porta, com as amplitudes guardadas em meia precisão.
///
/// As duas metades de cada palavra são a parte real e a imaginária. A aritmética
/// acontece em `f32`, como no kernel de precisão simples — só o armazenamento
/// muda, e com ele o tráfego de memória, que é o gargalo aqui.
const PORTA1_F16: &str = r#"
struct P {
    pares: u32, qubit: u32, gx: u32, pad: u32,
    u0: vec4<f32>,
    u1: vec4<f32>,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> psi: array<u32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn porta1(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let par = (w.y * p.gx + w.x) * 256u + l.x;
    if (par >= p.pares) { return; }

    let mascara = (1u << p.qubit) - 1u;
    let i0 = ((par >> p.qubit) << (p.qubit + 1u)) | (par & mascara);
    let i1 = i0 | (1u << p.qubit);

    let a0 = unpack2x16float(psi[i0]);
    let a1 = unpack2x16float(psi[i1]);
    psi[i0] = pack2x16float(cmul(p.u0.xy, a0) + cmul(p.u0.zw, a1));
    psi[i1] = pack2x16float(cmul(p.u1.xy, a0) + cmul(p.u1.zw, a1));
}
"#;

/// Converte meia precisão para `f32`, incluindo subnormais.
///
/// Escrito à mão para manter a crate sem dependências: a `half` faria isto, mas
/// são quinze linhas.
fn f16_para_f32(h: u16) -> f32 {
    let sinal = ((h >> 15) & 1) as u32;
    let expo = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;

    let bits = if expo == 0 {
        if frac == 0 {
            sinal << 31
        } else {
            // Subnormal: normaliza deslocando até o bit implícito aparecer.
            let mut e: i32 = -1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sinal << 31) | (((113 + e) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if expo == 31 {
        (sinal << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sinal << 31) | ((expo + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

/// Vetor de estado de `n` qubits, residente na GPU.
pub struct Estado {
    qubits: usize,
    precisao: Precisao,
    buf: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
}

impl Estado {
    /// Teto de qubits em precisão simples, imposto pelo limite de 2 GB por
    /// binding. Em meia precisão são 29 — ver [`Precisao::max_qubits`].
    pub const MAX_QUBITS: usize = 27;

    /// Cria o estado `|0…0⟩` em precisão simples.
    pub fn novo(gpu: &Gpu, qubits: usize) -> Result<Estado, String> {
        Estado::novo_com(gpu, qubits, Precisao::F32)
    }

    /// Cria o estado `|0…0⟩`: amplitude 1 no índice 0, zero no resto.
    pub fn novo_com(gpu: &Gpu, qubits: usize, precisao: Precisao) -> Result<Estado, String> {
        if qubits == 0 {
            return Err("um estado precisa de ao menos um qubit".into());
        }
        if qubits > precisao.max_qubits() {
            return Err(format!(
                "{qubits} qubits em {precisao:?} exigem {} GB num único binding; o WebGPU \
                 limita a 2 GB, o que dá {} qubits. Passar disso exige repartir o estado \
                 entre buffers.",
                (1usize << qubits) * precisao.bytes_por_amplitude() / (1 << 30),
                precisao.max_qubits()
            ));
        }
        let amplitudes = 1usize << qubits;
        let palavras = amplitudes * precisao.bytes_por_amplitude() / 4;
        let buf = gpu.buffer_bruto(palavras, "psi");

        // |0…0⟩: amplitude 1 no índice 0. Em meia precisão, 1,0 é 0x3C00 na
        // metade baixa e 0 na alta — não é preciso converter nada.
        let mut inicial = vec![0u32; palavras];
        inicial[0] = match precisao {
            Precisao::F32 => 1.0f32.to_bits(),
            Precisao::F16 => 0x3C00,
        };
        gpu.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&inicial));

        let fonte = match precisao {
            Precisao::F32 => PORTA1,
            Precisao::F16 => PORTA1_F16,
        };
        let modulo = gpu.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("porta1"),
            source: wgpu::ShaderSource::Wgsl(fonte.into()),
        });
        let pipeline = gpu
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("porta1"),
                layout: None,
                module: &modulo,
                entry_point: Some("porta1"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Estado { qubits, precisao, buf, pipeline })
    }

    pub fn qubits(&self) -> usize {
        self.qubits
    }

    pub fn precisao(&self) -> Precisao {
        self.precisao
    }

    pub fn amplitudes(&self) -> usize {
        1 << self.qubits
    }

    /// Bytes ocupados pelo vetor de estado.
    pub fn bytes(&self) -> usize {
        self.amplitudes() * self.precisao.bytes_por_amplitude()
    }

    /// Grava a aplicação de uma porta. Não sincroniza.
    pub fn aplicar(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, porta: &Porta1, qubit: usize) {
        assert!(qubit < self.qubits, "qubit {qubit} fora de {}", self.qubits);
        let pares = self.amplitudes() / 2;
        let grupos = pares.div_ceil(256);
        // Cada dimensão da grade vai até 65535; dobra-se em 2-D.
        let (gx, gy) = if grupos <= 32768 {
            (grupos as u32, 1u32)
        } else {
            (32768, grupos.div_ceil(32768) as u32)
        };

        let params = PortaP {
            pares: pares as u32,
            qubit: qubit as u32,
            gx,
            pad: 0,
            u0: [porta.u00.0, porta.u00.1, porta.u01.0, porta.u01.1],
            u1: [porta.u10.0, porta.u10.1, porta.u11.0, porta.u11.1],
        };
        let uniforme = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("porta"),
            size: std::mem::size_of::<PortaP>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue()
            .write_buffer(&uniforme, 0, bytemuck::bytes_of(&params));

        let bind = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforme.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf.as_entire_binding() },
            ],
        });

        let mut passe = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("porta1"),
            timestamp_writes: None,
        });
        passe.set_pipeline(&self.pipeline);
        passe.set_bind_group(0, &bind, &[]);
        passe.dispatch_workgroups(gx, gy, 1);
    }

    /// Baixa as amplitudes para a CPU, convertendo de meia precisão quando for
    /// o caso. Sincroniza.
    pub fn baixar(&self, gpu: &Gpu) -> Vec<Complexo> {
        let bytes = self.bytes() as u64;
        let leitura = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("leitura"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.encoder();
        enc.copy_buffer_to_buffer(&self.buf, 0, &leitura, 0, bytes);
        gpu.queue().submit(Some(enc.finish()));

        let slice = leitura.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        rx.recv().expect("canal").expect("map");

        let vista = slice.get_mapped_range().expect("range");
        let saida = match self.precisao {
            Precisao::F32 => {
                let cru: Vec<f32> = bytemuck::cast_slice(&vista[..]).to_vec();
                cru.chunks_exact(2).map(|c| (c[0], c[1])).collect()
            }
            Precisao::F16 => {
                let cru: Vec<u32> = bytemuck::cast_slice(&vista[..]).to_vec();
                cru.iter()
                    .map(|w| {
                        (
                            f16_para_f32(*w as u16),
                            f16_para_f32((*w >> 16) as u16),
                        )
                    })
                    .collect()
            }
        };
        drop(vista);
        saida
    }
}

/// Estado `|0…0⟩` na CPU, para referência.
pub fn estado_inicial_cpu(qubits: usize) -> Vec<Complexo> {
    let mut v = vec![(0.0, 0.0); 1 << qubits];
    v[0] = (1.0, 0.0);
    v
}
