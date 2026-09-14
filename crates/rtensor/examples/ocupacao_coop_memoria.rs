//! A ocupação cai conforme a memória de workgroup usada sobe? Varia só o
//! preenchimento — a aritmética de `coopMultiplyAdd` é idêntica à de
//! `ocupacao_coop.rs` — para isolar o efeito do orçamento de memória.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example ocupacao_coop_memoria
//!
//! # Por que
//!
//! `ocupacao_coop.rs` mediu o teto puro de `coopMultiplyAdd` (~36–45 TFLOP/s)
//! usando só 2 KB de memória de workgroup por grupo. O GEMM cooperativo real
//! (`bench_coop.rs`) usa **24 KB** — 4 KB de `sa`, 4 KB de `sb` e **16 KB de
//! `sc`**, o quique por bug do naga — e mede só ~7,1 TFLOP/s, ~20% do teto.
//! A hipótese: o orçamento de 48 KB por SM permite menos workgroups
//! residentes com 24 KB do que com 2 KB, e menos workgroups residentes
//! significa menos warps para esconder latência.
//!
//! # Como
//!
//! Um array de preenchimento `var<workgroup> pad: array<f32, N>` é somado ao
//! orçamento de cada workgroup, sem participar da aritmética cronometrada —
//! só é tocado uma vez, fora do laço medido, para o compilador não descartá-lo.
//! `N` varre de 0 a 10.240 floats (0 a 40 KB), cobrindo a faixa entre o
//! orçamento da sonda pura e o do kernel real.

use std::time::Instant;

use wgpu::util::DeviceExt;

const ITERS: u32 = 4096;
const WORKGROUPS: u32 = 2048;
const THREADS: u32 = 256;

/// `pad_floats` cresce a memória de workgroup em `pad_floats * 4` bytes,
/// além dos 2 KB fixos de `sa`+`sb`+`sc` que a aritmética usa de verdade.
fn fonte(pad_floats: u32) -> String {
    let decl_pad = if pad_floats > 0 {
        format!("var<workgroup> pad: array<f32, {pad_floats}>;\n")
    } else {
        String::new()
    };
    let toca_pad = if pad_floats > 0 {
        format!(
            "    for (var i = tid; i < {pad_floats}u; i = i + {THREADS}u) {{ pad[i] = f32(i); }}\n"
        )
    } else {
        String::new()
    };
    format!(
        r#"
enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read_write> saida: array<f32>;

var<workgroup> sa: array<f16, 256>;
var<workgroup> sb: array<f16, 256>;
var<workgroup> sc: array<f32, 512>;
{decl_pad}
@compute @workgroup_size({THREADS})
fn probe(@builtin(local_invocation_index) tid: u32) {{
    for (var i = tid; i < 256u; i = i + {THREADS}u) {{
        sa[i] = f16(f32(i) * 0.001);
        sb[i] = f16(f32(i) * 0.002);
    }}
    for (var i = tid; i < 512u; i = i + {THREADS}u) {{
        sc[i] = 0.0;
    }}
{toca_pad}    workgroupBarrier();

    let ma = coopLoadT<coop_mat16x16<f16, A>>(&sa[0], 16u);
    let mb = coopLoadT<coop_mat16x16<f16, B>>(&sb[0], 16u);
    var mc0 = coopLoadT<coop_mat16x16<f32, C>>(&sc[0], 16u);
    var mc1 = coopLoadT<coop_mat16x16<f32, C>>(&sc[256], 16u);

    for (var t = 0u; t < {ITERS}u; t = t + 1u) {{
        mc0 = coopMultiplyAdd(ma, mb, mc0);
        mc1 = coopMultiplyAdd(ma, mb, mc1);
    }}

    workgroupBarrier();
    coopStoreT(mc0, &sc[0], 16u);
    coopStoreT(mc1, &sc[256], 16u);
    workgroupBarrier();
    saida[tid % 1024u] = sc[tid % 512u];
}}
"#
    )
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
    println!("{}\n", ad.get_info().name);

    let props = ad.cooperative_matrix_properties();
    let config = props.iter().find(|p| {
        p.m_size == 16 && p.n_size == 16 && p.k_size == 16
            && p.ab_type == wgpu::CooperativeScalarType::F16
            && p.cr_type == wgpu::CooperativeScalarType::F32
    });
    if config.is_none() {
        println!("a placa não anuncia 16×16×16 com AB f16 e CR f32 — nada a medir");
        return;
    }

    let f = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16;
    let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: f,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        required_limits: ad.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .expect("dispositivo");

    let saida = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0.0f32; 1024]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let subgrupos = (WORKGROUPS * THREADS / 32) as f64;
    let flop_fixo = subgrupos * ITERS as f64 * 2.0 * 2.0 * 16.0 * 16.0 * 16.0;

    println!("{:>14} {:>9} {:>10} {:>12}", "pad (KB)", "total KB", "ms", "GFLOP/s");
    let mut melhor = 0.0f64;
    let mut linhas = Vec::new();
    // 2 KB fixos (sa+sb+sc) + pad, cobrindo até passar dos 24 KB do kernel real.
    for pad_floats in [0u32, 512, 1536, 2560, 3584, 4608, 5632, 6656, 7680, 8704, 9728] {
        let modulo = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(fonte(pad_floats).into()),
        });
        let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &modulo,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: saida.as_entire_binding() }],
        });

        let rodar = || {
            let mut enc = dev.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&p);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(WORKGROUPS, 1, 1);
            }
            q.submit(Some(enc.finish()));
            dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        rodar();
        let t0 = Instant::now();
        const REPS: u32 = 5;
        for _ in 0..REPS {
            rodar();
        }
        let dt = t0.elapsed().as_secs_f64() / REPS as f64;
        let gflops = flop_fixo / dt / 1e9;
        melhor = melhor.max(gflops);

        let total_kb = 2.0 + (pad_floats as f64 * 4.0) / 1024.0;
        linhas.push((pad_floats, total_kb, dt, gflops));
    }

    for (pad_floats, total_kb, dt, gflops) in linhas {
        let pad_kb = (pad_floats as f64 * 4.0) / 1024.0;
        let barra = "█".repeat((gflops / melhor * 30.0).round() as usize);
        println!(
            "{:>14.1} {:>9.1} {:>10.2} {:>12.1} {barra}",
            pad_kb, total_kb, dt * 1e3, gflops
        );
    }
    println!("\npara comparar: o GEMM cooperativo real usa 24 KB por workgroup e mede ~7.150 GFLOP/s");
}
