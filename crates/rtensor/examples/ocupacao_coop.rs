//! Teto puro de `coopMultiplyAdd`, isolado de qualquer leitura de memória —
//! o equivalente, para tensor cores, do que `ocupacao.rs` já faz para FMA
//! escalar.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example ocupacao_coop
//!
//! # Por que medir isso
//!
//! O GEMM cooperativo com buffer duplo e rasterização L2 mede
//! **7.087–7.204 GFLOP/s em 4096³** (`bench_coop.rs`, RESULTADOS-NEGATIVOS.md).
//! A pergunta que decide o próximo passo: esse número está perto do teto do
//! que os tensor cores desta placa sustentam, ou ainda sobra ocupação/vazão de
//! instrução na mesa — como sobrava no kernel escalar, onde a aritmética pura
//! sustentava 15 a 18 TFLOP/s contra um GEMM de ~4?
//!
//! # Como
//!
//! Cada subgrupo carrega **uma vez** um par de ladrilhos 16×16 em `f16` e um
//! acumulador zerado, depois repete `coopMultiplyAdd` sobre os mesmos operandos
//! `ITERS` vezes — sem nenhuma leitura de memória global ou de workgroup
//! dentro do laço. Dois acumuladores por subgrupo, como no kernel real
//! (`mc0`, `mc1`), para reproduzir a mesma pressão de registradores.

use std::time::Instant;

use wgpu::util::DeviceExt;

const ITERS: u32 = 4096;
const WORKGROUPS: u32 = 2048;
const THREADS: u32 = 256; // 8 subgrupos por workgroup

fn fonte() -> String {
    format!(
        r#"
enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read_write> saida: array<f32>;

var<workgroup> sa: array<f16, 256>;
var<workgroup> sb: array<f16, 256>;
var<workgroup> sc: array<f32, 512>;

@compute @workgroup_size({THREADS})
fn probe(@builtin(local_invocation_index) tid: u32) {{
    for (var i = tid; i < 256u; i = i + {THREADS}u) {{
        sa[i] = f16(f32(i) * 0.001);
        sb[i] = f16(f32(i) * 0.002);
    }}
    for (var i = tid; i < 512u; i = i + {THREADS}u) {{
        sc[i] = 0.0;
    }}
    workgroupBarrier();

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
    println!("{}", ad.get_info().name);

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

    let modulo = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(fonte().into()),
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

    let subgrupos = (WORKGROUPS * THREADS / 32) as f64;
    // 2 coopMultiplyAdd por iteração, 2·16³ flops cada (multiplica-acumula).
    let flop = subgrupos * ITERS as f64 * 2.0 * 2.0 * 16.0 * 16.0 * 16.0;
    let gflops = flop / dt / 1e9;

    println!("\n{ITERS} iterações × 2 coopMultiplyAdd/subgrupo × {subgrupos:.0} subgrupos");
    println!("ms: {:.2}  |  teto puro de coopMultiplyAdd: {:.1} GFLOP/s ({:.2} TFLOP/s)", dt * 1e3, gflops, gflops / 1e3);
    println!("\npara comparar: GEMM cooperativo real (bench_coop.rs, 4096³) mede 7.087–7.204 GFLOP/s");
}
