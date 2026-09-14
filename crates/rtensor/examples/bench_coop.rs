//! GEMM cooperativo: os tensor cores da placa, via matriz cooperativa
//! experimental do wgpu, medidos contra o kernel escalar do repositório.
//!
//! # Desenho
//!
//! A mesma estrutura do GEMM atual — ladrilho de saída 64×64 por workgroup,
//! `K` em passos de 16, estagiagem dos painéis em memória de workgroup, L2 na
//! forma de percurso em linha — com o laço interno trocado por uma instrução:
//!
//! - workgroup de **256 threads = 8 subgrupos** (o subgrupo da NVIDIA é 32, e
//!   o naga emite as matrizes com escopo `Subgroup`);
//! - os 4×4 = 16 ladrilhos cooperativos de 16×16 do ladrilho de saída ficam
//!   dois por subgrupo: o subgrupo `s` cobre a linha de blocos `s % 4` e as
//!   colunas de blocos `{ s/4, s/4 + 2 }`;
//! - A e B em `f16` (as únicas configurações que a placa anuncia), acumulador
//!   em `f32`;
//! - cada subgrupo carrega seu `C` uma vez, acumula `K/16` passos com
//!   `coopMultiplyAdd` e devolve uma vez;
//! - **buffer duplo**: as leituras globais do ladrilho `t+1` são emitidas
//!   antes do `coopMultiplyAdd` sobre o ladrilho `t`, com dois ladrilhos `sa`
//!   e `sb` alternados — o mesmo padrão de `gemm::MM_FAST`;
//! - **rasterização L2**: o despacho é linear (`(m/64)·(n/64)` workgroups em
//!   1-D) e o índice de bloco é reconstruído com o mesmo `bloco()` do kernel
//!   escalar, com `grupo = 8`.
//!
//! # O que decide
//!
//! O kernel escalar anda a **4.386 GFLOP/s** em 4096³ (`bench_gemm`), com o
//! teto de ~5 TFLOP/s imposto pelo canal da memória compartilhada a 1 flop
//! por byte. O cooperativo troca o laço interno por `coopMultiplyAdd` — se os
//! tensor cores pagam os dois preços conhecidos (`unsafe` e operandos `f16`),
//! aparece aqui. O primeiro corte (sem buffer duplo nem rasterização L2) já
//! mediu 5.756–5.988 GFLOP/s em 4096³; com as duas otimizações que faltavam,
//! esta versão mede o teto do desenho comum.
//!
//! # Ressalvas de método
//!
//! - Tempo por relógio de parede em volta de submissão e espera — para cargas
//!   de dezenas de milissegundos o desvio é pequeno; o instrumento fino é o
//!   perfilamento por marcas do dispositivo (`examples/perfil.rs`).
//! - A conferência numérica compara um bloco amostral contra referência em
//!   f64 sobre os **operandos já arredondados para `f16`** — o erro medido é
//!   o da acumulação, não o da conversão.
//! - As conversões `f16` no hospedeiro são as mesmas de `probe_coop_f16.rs`,
//!   conferidas lá contra valores fixados.

use wgpu::util::DeviceExt;

const FONTE: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;

struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, p0: u32, p1: u32, p2: u32 };

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

// Dois ladrilhos, alternados — igual ao buffer duplo do kernel escalar.
var<workgroup> sa: array<f16, 2048>;
var<workgroup> sb: array<f16, 2048>;
// Dois ladrilhos C 16×16 f32 por subgrupo — a saída do kernel passa por aqui
// porque o naga 30.0.1 entra em pânico no caminho de índice do SPIR-V quando
// `coopLoad`/`coopStore` recebem ponteiro dinâmico para buffer de
// armazenamento; para memória de workgroup o mesmo padrão compila. As cópias
// de entrada e saída dos slots são operações planas, que não tocam no bug.
var<workgroup> sc: array<f32, 4096>;

// Ordem de percurso dos blocos com consciência de L2 — mesma lógica de
// `gemm::MM_FAST::bloco`, portada aqui. `grupo = 1` reproduz o percurso em
// linha.
fn bloco(pid: u32, nm: u32, nn: u32) -> vec2<u32> {
    let grupo = max(dims.grupo, 1u);
    let por_grupo = grupo * nn;
    let gid = pid / por_grupo;
    let m0 = gid * grupo;
    let tam = max(min(nm - m0, grupo), 1u);
    return vec2<u32>(m0 + (pid % tam), (pid % por_grupo) / tam);
}

fn carregar_a(m0: u32, k0: u32, ca: u32, ra: u32, K: u32) -> vec4<f16> {
    return vec4<f16>(
        a[(m0 + ra) * K + k0 + ca],
        a[(m0 + ra + 16u) * K + k0 + ca],
        a[(m0 + ra + 32u) * K + k0 + ca],
        a[(m0 + ra + 48u) * K + k0 + ca],
    );
}

fn guardar_a(buf: u32, ca: u32, ra: u32, v: vec4<f16>) {
    sa[buf * 1024u + ra * 16u + ca] = v.x;
    sa[buf * 1024u + (ra + 16u) * 16u + ca] = v.y;
    sa[buf * 1024u + (ra + 32u) * 16u + ca] = v.z;
    sa[buf * 1024u + (ra + 48u) * 16u + ca] = v.w;
}

fn carregar_b(n0: u32, k0: u32, rb: u32, cb: u32, N: u32) -> vec4<f16> {
    return vec4<f16>(
        b[(k0 + rb) * N + n0 + cb],
        b[(k0 + rb) * N + n0 + cb + 1u],
        b[(k0 + rb) * N + n0 + cb + 2u],
        b[(k0 + rb) * N + n0 + cb + 3u],
    );
}

fn guardar_b(buf: u32, rb: u32, cb: u32, v: vec4<f16>) {
    sb[buf * 1024u + rb * 64u + cb] = v.x;
    sb[buf * 1024u + rb * 64u + cb + 1u] = v.y;
    sb[buf * 1024u + rb * 64u + cb + 2u] = v.z;
    sb[buf * 1024u + rb * 64u + cb + 3u] = v.w;
}

@compute @workgroup_size(256, 1, 1)
fn mm(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) l: vec3<u32>,
) {
    let M = dims.m;
    let N = dims.n;
    let K = dims.k;
    let nm = dims.grid_x;
    let nn = N / 64u;

    let bid = bloco(wg.x, nm, nn);
    let m0 = bid.x * 64u;
    let n0 = bid.y * 64u;
    let s = l.x / 32u; // subgrupo, 0..8
    let lane = l.x % 32u;
    let linha = s % 4u; // bloco de 16 linhas deste subgrupo
    let col0 = (s / 4u) * 2u; // primeiro dos dois blocos de 16 colunas

    // Divisões e deslocamentos calculados uma vez; o laço de K fica só com somas.
    let base_ma = linha * 256u;       // ladrilho de A na estagiagem
    let base_mb0 = col0 * 16u;        // ladrilho de B 0
    let base_mb1 = (col0 + 1u) * 16u; // ladrilho de B 1
    let slot = s * 512u;              // dois ladrilhos C deste subgrupo
    let base_c = (m0 + linha * 16u) * N + n0 + col0 * 16u; // ladrilho de C em c

    let ca = l.x % 16u;    // coluna dentro do painel A
    let ra = l.x / 16u;    // linha inicial (de 4 em 4)
    let rb = l.x / 16u;    // linha do painel B, 0..15
    let cb = (l.x % 16u) * 4u; // quatro colunas contíguas do painel B

    // Zera os slots de C deste subgrupo e carrega os acumuladores —
    // C = A·B, sem termo inicial.
    for (var i = lane; i < 512u; i = i + 32u) {
        sc[slot + i] = 0.0;
    }
    workgroupBarrier();
    var mc0 = coopLoadT<coop_mat16x16<f32, C>>(&sc[slot], 16u);
    var mc1 = coopLoadT<coop_mat16x16<f32, C>>(&sc[slot + 256u], 16u);

    // Carga do primeiro ladrilho, buffer 0.
    guardar_a(0u, ca, ra, carregar_a(m0, 0u, ca, ra, K));
    guardar_b(0u, rb, cb, carregar_b(n0, 0u, rb, cb, N));
    workgroupBarrier();

    let passos = K / 16u;
    var cur = 0u;
    for (var t = 0u; t < passos; t = t + 1u) {
        let tem_proximo = t + 1u < passos;
        var prox_a: vec4<f16>;
        var prox_b: vec4<f16>;
        if (tem_proximo) {
            // Emitidas antes do `coopMultiplyAdd`: a latência da memória
            // global corre por baixo do cálculo do ladrilho atual.
            prox_a = carregar_a(m0, (t + 1u) * 16u, ca, ra, K);
            prox_b = carregar_b(n0, (t + 1u) * 16u, rb, cb, N);
        }

        let ma = coopLoadT<coop_mat16x16<f16, A>>(&sa[cur * 1024u + base_ma], 16u);
        let mb0 = coopLoadT<coop_mat16x16<f16, B>>(&sb[cur * 1024u + base_mb0], 64u);
        let mb1 = coopLoadT<coop_mat16x16<f16, B>>(&sb[cur * 1024u + base_mb1], 64u);
        mc0 = coopMultiplyAdd(ma, mb0, mc0);
        mc1 = coopMultiplyAdd(ma, mb1, mc1);

        if (tem_proximo) {
            guardar_a(1u - cur, ca, ra, prox_a);
            guardar_b(1u - cur, rb, cb, prox_b);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }

    // Devolve: store cooperativo no slot (base fixa), barreira, cópia plana
    // para o buffer de saída.
    coopStoreT(mc0, &sc[slot], 16u);
    coopStoreT(mc1, &sc[slot + 256u], 16u);
    workgroupBarrier();
    for (var i = lane; i < 256u; i = i + 32u) {
        c[base_c + (i / 16u) * N + i % 16u] = sc[slot + i];
    }
    for (var i = lane; i < 256u; i = i + 32u) {
        c[base_c + 16u + (i / 16u) * N + i % 16u] = sc[slot + 256u + i];
    }
}
"#;

/// O primeiro corte, sem buffer duplo nem rasterização L2 — preservado aqui
/// só para a comparação lado a lado, no mesmo processo, contra `FONTE`. Mede
/// **5.756–5.988 GFLOP/s em 4096³** no README; ver ali para a origem.
const FONTE_ANTIGA: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // M, N, K, —

var<workgroup> sa: array<f16, 1024>;
var<workgroup> sb: array<f16, 1024>;
var<workgroup> sc: array<f32, 4096>;

@compute @workgroup_size(256, 1, 1)
fn mm(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) l: vec3<u32>,
) {
    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    let m0 = wg.x * 64u;
    let n0 = wg.y * 64u;
    let s = l.x / 32u;
    let lane = l.x % 32u;
    let linha = s % 4u;
    let col0 = (s / 4u) * 2u;

    let base_ma = linha * 256u;
    let base_mb0 = col0 * 16u;
    let base_mb1 = (col0 + 1u) * 16u;
    let slot = s * 512u;
    let base_c = (m0 + linha * 16u) * N + n0 + col0 * 16u;

    let ca = l.x % 16u;
    let ra = l.x / 16u;
    let rb = l.x / 16u;
    let cb = (l.x % 16u) * 4u;

    for (var i = lane; i < 512u; i = i + 32u) {
        sc[slot + i] = 0.0;
    }
    workgroupBarrier();
    var mc0 = coopLoadT<coop_mat16x16<f32, C>>(&sc[slot], 16u);
    var mc1 = coopLoadT<coop_mat16x16<f32, C>>(&sc[slot + 256u], 16u);

    let passos = K / 16u;
    for (var t = 0u; t < passos; t = t + 1u) {
        let k0 = t * 16u;
        for (var q = 0u; q < 64u; q = q + 16u) {
            let r = ra + q;
            sa[r * 16u + ca] = a[(m0 + r) * K + k0 + ca];
        }
        for (var j = 0u; j < 4u; j = j + 1u) {
            sb[rb * 64u + cb + j] = b[(k0 + rb) * N + n0 + cb + j];
        }
        workgroupBarrier();

        let ma = coopLoadT<coop_mat16x16<f16, A>>(&sa[base_ma], 16u);
        let mb0 = coopLoadT<coop_mat16x16<f16, B>>(&sb[base_mb0], 64u);
        let mb1 = coopLoadT<coop_mat16x16<f16, B>>(&sb[base_mb1], 64u);
        mc0 = coopMultiplyAdd(ma, mb0, mc0);
        mc1 = coopMultiplyAdd(ma, mb1, mc1);
        workgroupBarrier();
    }

    coopStoreT(mc0, &sc[slot], 16u);
    coopStoreT(mc1, &sc[slot + 256u], 16u);
    workgroupBarrier();
    for (var i = lane; i < 256u; i = i + 32u) {
        c[base_c + (i / 16u) * N + i % 16u] = sc[slot + i];
    }
    for (var i = lane; i < 256u; i = i + 32u) {
        c[base_c + 16u + (i / 16u) * N + i % 16u] = sc[slot + 256u + i];
    }
}
"#;

/// `f32 → f16`, a mesma de `probe_coop_f16.rs` (arredondamento para o par).
fn f32_para_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sinal = ((bits >> 16) & 0x8000) as u16;
    let expo = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if expo == 255 {
        return if mant == 0 { sinal | 0x7c00 } else { sinal | 0x7e00 };
    }
    let e = expo - 127;
    if e > 15 {
        return sinal | 0x7c00;
    }
    if e >= -14 {
        let exp16 = (e + 15) as u32;
        let mut mant16 = mant >> 13;
        let resto = mant & 0x1fff;
        if resto > 0x1000 || (resto == 0x1000 && (mant16 & 1) == 1) {
            mant16 += 1;
        }
        if mant16 >> 10 != 0 {
            if exp16 == 30 {
                return sinal | 0x7c00;
            }
            return sinal | (((exp16 + 1) << 10) as u16);
        }
        return sinal | ((exp16 << 10) as u16) | (mant16 as u16);
    }
    let shift = (-e - 1) as u32;
    let x = if shift >= 32 {
        0
    } else {
        let valor = 0x0080_0000 + mant;
        let base = valor >> shift;
        let resto = valor & ((1u32 << shift) - 1);
        let meio = 1u32 << (shift - 1);
        if resto > meio || (resto == meio && (base & 1) == 1) { base + 1 } else { base }
    };
    sinal | (x as u16)
}

/// LCG determinístico em [-1, 1): sem dependência e reproduzível.
struct Aleatorio(u64);
impl Aleatorio {
    fn proximo(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn main() {
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ad = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .find(|a| {
            let i = a.get_info();
            i.device_type == wgpu::DeviceType::DiscreteGpu && i.backend == wgpu::Backend::Vulkan
        })
        .expect("sem GPU discreta");
    println!("adaptador: {}", ad.get_info().name);

    let props = ad.cooperative_matrix_properties();
    let config = props.iter().find(|p| {
        p.m_size == 16 && p.n_size == 16 && p.k_size == 16
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F32
    });
    let Some(_) = config else {
        println!("a placa não anuncia 16×16×16 com AB f16 e CR f32 — nada a medir");
        return;
    };

    let f = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16;
    let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: f,
        // O mesmo consentimento das sondas: `enabled()` é `unsafe fn`, e este
        // benchmark mede o que essa concessão compra.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        required_limits: ad.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .expect("dispositivo");

    // As duas variantes, no mesmo processo e no mesmo dispositivo — evita a
    // deriva de clock entre execuções separadas, já documentada em
    // RESULTADOS-NEGATIVOS.md.
    println!("{:>10} {:>7} | {:>9} | {:>5}", "kernel", "N", "GFLOP/s", "ms");
    let mut base: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    for (nome, fonte, tem_grupo) in [("antigo", FONTE_ANTIGA, false), ("novo", FONTE, true)] {
        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(nome),
            source: wgpu::ShaderSource::Wgsl(fonte.into()),
        });
        let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &m,
            entry_point: Some("mm"),
            compilation_options: Default::default(),
            cache: None,
        });

        for &n in &[2048u32, 4096] {
            let k = n;
            let m = n;

            // A e B aleatórios em [-1, 1], já arredondados para f16.
            let mut r = Aleatorio(0x20140815);
            let a: Vec<u16> = (0..(m * k)).map(|_| f32_para_f16(r.proximo())).collect();
            let b: Vec<u16> = (0..(k * n)).map(|_| f32_para_f16(r.proximo())).collect();
            let c: Vec<f32> = vec![0.0; (m * n) as usize];

            let ba = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&a),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let bb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&b),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let bc = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&c),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let dims = if tem_grupo {
                let grid_x = m / 64;
                let grupo = 8u32; // mesmo padrão do kernel escalar em produção
                dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[m, n, k, grid_x, grupo, 0u32, 0u32, 0u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            } else {
                dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[m, n, k, 0u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            };
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &p.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ba.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
                ],
            });

            let leitura = dev.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (m * n * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Despacho 2-D no antigo (sem rasterização), 1-D linear no novo.
            let gx = m / 64;
            let gy = n / 64;
            let rodar = |enc: &mut wgpu::CommandEncoder| {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&p);
                cp.set_bind_group(0, &bg, &[]);
                if tem_grupo {
                    cp.dispatch_workgroups(gx * gy, 1, 1);
                } else {
                    cp.dispatch_workgroups(gx, gy, 1);
                }
            };

            // Aquecimento.
            for _ in 0..2 {
                let mut enc = dev.create_command_encoder(&Default::default());
                rodar(&mut enc);
                q.submit(Some(enc.finish()));
                dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }

            // Medição: 8 execuções, mediana.
            let mut tempos: Vec<f64> = Vec::new();
            for _ in 0..8 {
                let mut enc = dev.create_command_encoder(&Default::default());
                rodar(&mut enc);
                enc.copy_buffer_to_buffer(&bc, 0, &leitura, 0, (m * n * 4) as u64);
                let inicio = std::time::Instant::now();
                q.submit(Some(enc.finish()));
                dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                tempos.push(inicio.elapsed().as_secs_f64() * 1e3);
            }
            tempos.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let ms = tempos[tempos.len() / 2];
            let gflops = 2.0 * (m as f64) * (n as f64) * (k as f64) / (ms / 1e3) / 1e9;

            // Conferência numérica: um bloco 32×32 amostral, referência em f64
            // sobre os operandos já em f16.
            let slice = leitura.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let vista = slice.get_mapped_range().unwrap();
            let obtido: Vec<f32> = bytemuck::cast_slice(&vista[..]).to_vec();
            drop(vista);

            let i0 = ((m as usize) / 3) & !31;
            let j0 = ((n as usize) / 5) & !31;
            let mut erro = 0.0f64;
            let mut norma = 0.0f64;
            for i in i0..i0 + 32 {
                for j in j0..j0 + 32 {
                    let mut e = 0.0f64;
                    for t in 0..k as usize {
                        e += f16_para_f32_ref(a[i * k as usize + t]) as f64
                            * f16_para_f32_ref(b[t * n as usize + j]) as f64;
                    }
                    erro = erro.max((obtido[i * n as usize + j] as f64 - e).abs());
                    norma = norma.max(e.abs());
                }
            }

            let ganho = if let Some(&b0) = base.get(&n) {
                format!("{:+.1}%", 100.0 * (gflops / b0 - 1.0))
            } else {
                base.insert(n, gflops);
                "(base)".to_string()
            };
            println!(
                "{:>10} {:>7} | {:>9.0} | {:>5.1} | {ganho:>8} | erro bloco: {erro:.2e} (norma {norma:.1})",
                nome, n, gflops, ms
            );
        }
    }
}

/// `f16 → f32` para a conferência, a mesma do `rqubit` com o ajuste dos
/// subnormais de lá.
fn f16_para_f32_ref(h: u16) -> f32 {
    let sinal = ((h >> 15) & 1) as u32;
    let expo = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;

    let bits = if expo == 0 {
        if frac == 0 {
            sinal << 31
        } else {
            let mut e: i32 = 0;
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
