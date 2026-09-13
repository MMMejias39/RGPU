//! Sonda da matriz cooperativa nas configurações que o adaptador **anuncia**.
//!
//! # Por que esta sonda existe
//!
//! `probe_coop.rs` estabeleceu o diagnóstico: a sonda original testava
//! `8×8 f32`, configuração que **não está na lista** que o adaptador anuncia
//! (RTX 4070 Laptop anuncia só combinações com `f16` em A e B). Configuração
//! fora da lista é comportamento indefinido — e foi isso que produziu os
//! zeros. Esta sonda repete o produto nas configurações anunciadas:
//!
//! - `16×16×16`, AB `f16`, C `f32` — precisão mista, a forma que um GEMM
//!   misto usaria;
//! - `16×16×16`, AB `f16`, C `f16` — acumulador em meia precisão.
//!
//! # Escada de diagnóstico
//!
//! Todas as variantes partem de `C` preenchido com a marca de vida 1234 e o
//! esperado é `A·B + 1234`. Se o resultado for:
//!
//! - `A·B + 1234` — a cadeia inteira funciona;
//! - `A·B` puro — a **carga de C** devolve zero (o sintoma original);
//! - só a marca — a multiplicação não somou nada, ou o `coopStore` não
//!   escreveu;
//! - tudo zero — a cadeia está morta.
//!
//! Os dados são inteiros pequenos, de modo que produtos e somas parciais são
//! exatos tanto em `f16` quanto em `f32`: o erro medido isola a cadeia, não a
//! precisão.
//!
//! # Referência de CPU
//!
//! A conversão `f16 → f32` é a mesma do `rqubit` (escrita à mão), com o
//! ajuste dos subnormais documentado lá; `f32 → f16` é escrita aqui com
//! arredondamento para o par mais próximo e conferida contra valores fixados
//! no início da execução.

use wgpu::util::DeviceExt;

const MARCA: f32 = 1234.0;

/// `f32 → f16` (u16), arredondamento para o par mais próximo. Escrita à mão
/// pelo mesmo motivo da `f16_para_f32` do `rqubit`: não introduzir
/// dependência.
fn f32_para_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sinal = ((bits >> 16) & 0x8000) as u16;
    let expo = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if expo == 255 {
        // NaN ou infinito.
        return if mant == 0 { sinal | 0x7c00 } else { sinal | 0x7e00 };
    }
    let e = expo - 127; // expoente desviado: viés 127 → viés 15
    if e > 15 {
        return sinal | 0x7c00; // transbordo → infinito
    }
    if e >= -14 {
        let exp16 = (e + 15) as u32;
        let mut mant16 = mant >> 13;
        let resto = mant & 0x1fff;
        if resto > 0x1000 || (resto == 0x1000 && (mant16 & 1) == 1) {
            mant16 += 1;
        }
        if mant16 >> 10 != 0 {
            // A arredondação transbordou a mantissa: sobe o expoente.
            if exp16 == 30 {
                return sinal | 0x7c00; // 65520 → infinito
            }
            return sinal | (((exp16 + 1) << 10) as u16);
        }
        return sinal | ((exp16 << 10) as u16) | (mant16 as u16);
    }
    // Subnormal: x = arredonda(2²³ + mant) >> (−e−1), em unidades de 2⁻²⁴.
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

/// `f16 → f32`, a mesma do `rqubit` com o ajuste dos subnormais de lá.
fn f16_para_f32(h: u16) -> f32 {
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

/// Conferência das conversões contra valores fixados antes de qualquer
/// medição: uma conversão errada invalidaria toda a referência.
fn conferir_conversoes() {
    let casos: [(f32, u16); 12] = [
        (1.0, 0x3c00),
        (2.0, 0x4000),
        (0.5, 0x3800),
        (4.0, 0x4400),
        (1234.0, 0x64d2),
        (2049.0, 0x6800), // meio-termo: vai para o par, 2048
        (3.14159, 0x4248),
        (65504.0, 0x7bff), // maior finito
        (65520.0, 0x7c00), // meio-termo: transborda → infinito
        (2f32.powi(-24), 0x0001), // maior subnormal
        (2f32.powi(-25), 0x0000), // meio-termo: vai para o par, zero
        (-0.5, 0xb800),
    ];
    for (v, esperado) in casos {
        let obtido = f32_para_f16(v);
        assert_eq!(obtido, esperado, "f32_para_f16({v}) = 0x{obtido:04x}, esperado 0x{esperado:04x}");
        // Idempotência, não ida-e-volta: um valor que arredonda (2049 → 2048)
        // não volta a ser ele mesmo — decodificar e recodificar devolve os
        // mesmos bits, e é isso que se exige.
        assert_eq!(f32_para_f16(f16_para_f32(esperado)), esperado, "recodificação de 0x{esperado:04x}");
    }
}

/// WGSL do produto `C = A·B + C` num ladrilho 16×16, AB em `f16`.
///
/// `estagio` escolhe entre leitura direta do buffer de armazenamento e o
/// caminho que um GEMM real usaria: carregar os ladrilhos em memória de
/// workgroup antes de multiplicar. `total` é o número de threads do
/// workgroup, para o laço de estagiagem e de devolução.
fn shader_16(cr_f32: bool, estagio: bool, wg: &str, total: u32) -> String {
    let ct = if cr_f32 { "f32" } else { "f16" };
    let declaracoes = if estagio {
        format!(
            "var<workgroup> sa: array<f16, 256>;\n\
             var<workgroup> sb: array<f16, 256>;\n\
             var<workgroup> sc: array<{ct}, 256>;\n"
        )
    } else {
        String::new()
    };
    let corpo = if estagio {
        format!(
            "    let li = l.x + l.y * 16u;\n\
             \n\
             for (var i = li; i < 256u; i += {total}u) {{\n\
             \x20   sa[i] = a[i];\n\
             \x20   sb[i] = b[i];\n\
             \x20   sc[i] = {MARCA}.0;\n\
             }}\n\
             workgroupBarrier();\n\
             let ma = coopLoadT<coop_mat16x16<f16, A>>(&sa[0], 16u);\n\
             let mb = coopLoadT<coop_mat16x16<f16, B>>(&sb[0], 16u);\n\
             var mc = coopLoadT<coop_mat16x16<{ct}, C>>(&sc[0], 16u);\n\
             mc = coopMultiplyAdd(ma, mb, mc);\n\
             coopStoreT(mc, &sc[0], 16u);\n\
             workgroupBarrier();\n\
             for (var i = li; i < 256u; i += {total}u) {{\n\
             \x20   c[i] = sc[i];\n\
             }}"
        )
    } else {
        format!(
            "    let ma = coopLoadT<coop_mat16x16<f16, A>>(&a[0], 16u);\n\
             let mb = coopLoadT<coop_mat16x16<f16, B>>(&b[0], 16u);\n\
             var mc = coopLoadT<coop_mat16x16<{ct}, C>>(&c[0], 16u);\n\
             mc = coopMultiplyAdd(ma, mb, mc);\n\
             coopStoreT(mc, &c[0], 16u);"
        )
    };
    format!(
        "enable f16;\n\
         enable wgpu_cooperative_matrix;\n\
         \n\
         @group(0) @binding(0) var<storage, read> a: array<f16>;\n\
         @group(0) @binding(1) var<storage, read> b: array<f16>;\n\
         @group(0) @binding(2) var<storage, read_write> c: array<{ct}>;\n\
         \n\
         {declaracoes}\n\
         @compute @workgroup_size({wg})\n\
         fn mm(@builtin(local_invocation_id) l: vec3<u32>) {{\n\
         {corpo}\n\
         }}"
    )
}

/// Carga: `A[i][j] = ((i+j) mod 4) + 1`, `B[i][j] = (3i+j) mod 5` — inteiros
/// pequenos e dependentes da posição, de modo que um produto transposto ou
/// deslocado não pode imitar o resultado. `C = A·B` tem valores inteiros até
/// 256: exatos em `f16` e em `f32`.
fn dados_16() -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..256).map(|i| (((i / 16 + i % 16) % 4) + 1) as f32).collect();
    let b: Vec<f32> = (0..256).map(|i| (((i / 16) * 3 + i % 16) % 5) as f32).collect();
    (a, b)
}

fn esperado_16(a: &[f32], b: &[f32]) -> Vec<f64> {
    let mut e = vec![0.0f64; 256];
    for i in 0..16 {
        for j in 0..16 {
            for k in 0..16 {
                e[i * 16 + j] += a[i * 16 + k] as f64 * b[k * 16 + j] as f64;
            }
        }
    }
    e
}

fn main() {
    conferir_conversoes();
    println!("conversões f16/f32 conferidas contra valores fixados\n");

    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ad = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .find(|a| {
            let i = a.get_info();
            i.device_type == wgpu::DeviceType::DiscreteGpu && i.backend == wgpu::Backend::Vulkan
        })
        .expect("sem GPU discreta");

    let props = ad.cooperative_matrix_properties();
    let config_f32 = props.iter().find(|p| {
        p.m_size == 16 && p.n_size == 16 && p.k_size == 16 && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F32
    });
    let config_f16 = props.iter().find(|p| {
        p.m_size == 16 && p.n_size == 16 && p.k_size == 16 && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F16
    });
    if config_f32.is_none() && config_f16.is_none() {
        println!("nenhuma configuração 16×16×16 com AB f16 anunciada; nada a medir");
        for p in &props {
            println!("  {}×{}×{}  AB: {:?}  CR: {:?}", p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type);
        }
        return;
    }

    let f = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16;
    if !ad.features().contains(f) {
        eprintln!("adaptador sem as features necessárias: {:?} vs {:?}", ad.features(), f);
        return;
    }

    let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: f,
        // Mesma concessão de `probe_coop.rs`: `enabled()` é `unsafe fn`, e esta
        // sonda mede o preço dela antes de o projeto decidir se aceita.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        required_limits: ad.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .expect("dispositivo");

    let (a, b) = dados_16();
    let produto = esperado_16(&a, &b);
    let cheio: Vec<f64> = produto.iter().map(|x| x + MARCA as f64).collect();
    let sem_marca = produto;

    // A e B como padrões de bits f16.
    let a16: Vec<u16> = a.iter().map(|x| f32_para_f16(*x)).collect();
    let b16: Vec<u16> = b.iter().map(|x| f32_para_f16(*x)).collect();

    let buf_bytes = |bytes: &[u8], escrita: bool| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | if escrita { wgpu::BufferUsages::COPY_SRC } else { wgpu::BufferUsages::empty() },
        })
    };

    // (nome, cr em f32?, estagiagem, wg, total de threads)
    let variantes = [
        ("armaz, wg32", true, false, "32, 1, 1", 32),
        ("armaz, wg256", true, false, "16, 16, 1", 256),
        ("workgroup, wg32", true, true, "32, 1, 1", 32),
        ("workgroup, wg256", true, true, "16, 16, 1", 256),
        ("armaz f16, wg32", false, false, "32, 1, 1", 32),
        ("armaz f16, wg256", false, false, "16, 16, 1", 256),
        ("workgroup f16, wg32", false, true, "32, 1, 1", 32),
        ("workgroup f16, wg256", false, true, "16, 16, 1", 256),
    ];

    println!(
        "{:>19} | {:>9} | {}",
        "variante", "erro máx.", "veredito"
    );
    println!("{}", "-".repeat(100));

    for (nome, cr_f32, estagio, wg, total) in variantes {
        let fonte = shader_16(cr_f32, estagio, wg, total);

        let ba = buf_bytes(bytemuck::cast_slice(&a16), false);
        let bb = buf_bytes(bytemuck::cast_slice(&b16), false);
        // C parte da marca de vida — na variante com estagiagem, a marca é
        // escrita em `sc` dentro do kernel, e o buffer de saída parte de zero.
        let bc = if cr_f32 {
            buf_bytes(bytemuck::cast_slice(&vec![MARCA; 256]), true)
        } else {
            let m: Vec<u16> = vec![f32_para_f16(MARCA); 256];
            buf_bytes(bytemuck::cast_slice(&m), true)
        };

        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(nome),
            source: wgpu::ShaderSource::Wgsl(fonte.into()),
        });
        let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(nome),
            layout: None,
            module: &m,
            entry_point: Some("mm"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ba.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bc.as_entire_binding() },
            ],
        });

        let tamanho = if cr_f32 { 1024 } else { 512 }; // 256 × 4 ou 256 × 2 bytes
        let leitura = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: tamanho,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&p);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        enc.copy_buffer_to_buffer(&bc, 0, &leitura, 0, tamanho);
        q.submit(Some(enc.finish()));

        let slice = leitura.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let vista = slice.get_mapped_range().unwrap();
        let obtido: Vec<f64> = if cr_f32 {
            bytemuck::cast_slice::<u8, f32>(&vista[..]).iter().map(|x| *x as f64).collect()
        } else {
            bytemuck::cast_slice::<u8, u16>(&vista[..]).iter().map(|x| f16_para_f32(*x) as f64).collect()
        };
        drop(vista);

        let erro = obtido
            .iter()
            .zip(&cheio)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        let casar = |r: &[f64]| {
            obtido
                .iter()
                .zip(r)
                .all(|(x, y)| (x - y).abs() < 1e-3)
        };
        let veredito = if casar(&cheio) {
            "CORRETO — a cadeia inteira funciona"
        } else if casar(&sem_marca) {
            "carga de C devolve zero (o sintoma original)"
        } else if casar(&vec![MARCA as f64; 256]) {
            "só a marca: a multiplicação não somou, ou o store não escreveu"
        } else if casar(&vec![0.0; 256]) {
            "tudo zero"
        } else {
            "valor errado, sem casar com nenhum estágio conhecido"
        };
        println!("{nome:>19} | {erro:9.3e} | {veredito}");
    }
}
