//! Compara a leitura via memória compartilhada (o gargalo medido em
//! `banda_compartilhada.rs`) com a leitura via `subgroupShuffle` — feature
//! `SUBGROUP` do wgpu, **estável**, sem `unsafe` e sem exigir `f16`.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example sonda_subgrupo
//!
//! # O método
//!
//! Mesmo padrão do GEMM: 256 threads por workgroup, `TK` passos por ladrilho,
//! `ITERS` ladrilhos, 32 flops por passo por thread (16 FMAs). A variante
//! `compartilhada` é a `fma` de `banda_compartilhada.rs`, repetida aqui para
//! comparação direta na mesma execução. A variante `subgrupo` carrega, uma
//! única vez por thread, um par de vetores em registradores e usa
//! `subgroupShuffle` para obter os valores dos outros lanes a cada passo —
//! **nenhuma leitura de memória compartilhada no laço interno**.
//!
//! Se `subgrupo` sustentar mais GFLOP/s que `compartilhada` (teto medido:
//! ~3.975, ~78% do canal de 5,10 TB/s), o gargalo de banda da memória
//! compartilhada é contornável sem `unsafe` e sem tensor cores.

use std::time::Instant;

use wgpu::util::DeviceExt;

const WORKGROUPS: u32 = 2048;
const ITERS: u32 = 256;
const TK: u32 = 16;

fn fonte_compartilhada() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> saida: array<f32>;

var<workgroup> sa: array<f32, 1040>;
var<workgroup> sb: array<f32, 1040>;

@compute @workgroup_size(16, 16)
fn probe(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {{
    let tid = l.y * 16u + l.x;
    for (var i = tid; i < 1040u; i = i + 256u) {{
        sa[i] = f32(i) * 0.001;
        sb[i] = f32(i) * 0.002;
    }}
    workgroupBarrier();

    var acc = array<vec4<f32>, 4>();
    for (var t = 0u; t < {ITERS}u; t = t + 1u) {{
        for (var kk = 0u; kk < {TK}u; kk = kk + 1u) {{
            let ab = kk * 65u + l.y * 4u;
            let bb = kk * 65u + l.x * 4u;
            let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
            let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
            acc[0] = acc[0] + av.x * bv;
            acc[1] = acc[1] + av.y * bv;
            acc[2] = acc[2] + av.z * bv;
            acc[3] = acc[3] + av.w * bv;
        }}
    }}
    let s = acc[0] + acc[1] + acc[2] + acc[3];
    saida[(w.x * 256u + tid) % 1024u] = s.x + s.y + s.z + s.w;
}}
"#
    )
}

fn fonte_subgrupo() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> saida: array<f32>;

// `subgroup_invocation_id` exige workgroup 1-D no naga 30.0.1.
@compute @workgroup_size(256)
fn probe(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) w: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {{
    // Carga única por thread — nada disto se repete no laço interno.
    let base_a = f32(tid) * 0.001;
    let base_b = f32(tid) * 0.002;
    let meu_a = vec4<f32>(base_a, base_a + 1.0, base_a + 2.0, base_a + 3.0);
    let meu_b = vec4<f32>(base_b, base_b + 1.0, base_b + 2.0, base_b + 3.0);

    var acc = array<vec4<f32>, 4>();
    for (var t = 0u; t < {ITERS}u; t = t + 1u) {{
        for (var kk = 0u; kk < {TK}u; kk = kk + 1u) {{
            let origem_a = (lane + kk) % 32u;
            let origem_b = (lane + kk * 3u) % 32u;
            let av = subgroupShuffle(meu_a, origem_a);
            let bv = subgroupShuffle(meu_b, origem_b);
            acc[0] = acc[0] + av.x * bv;
            acc[1] = acc[1] + av.y * bv;
            acc[2] = acc[2] + av.z * bv;
            acc[3] = acc[3] + av.w * bv;
        }}
    }}
    let s = acc[0] + acc[1] + acc[2] + acc[3];
    saida[(w.x * 256u + tid) % 1024u] = s.x + s.y + s.z + s.w;
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
    let info = ad.get_info();
    println!("{}", info.name);
    println!(
        "subgroup_min={} subgroup_max={}\n",
        info.subgroup_min_size, info.subgroup_max_size
    );

    if !ad.features().contains(wgpu::Features::SUBGROUP) {
        panic!("placa sem SUBGROUP");
    }

    let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::SUBGROUP,
        required_limits: ad.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .expect("dispositivo");

    let buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0.0f32; 1024]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let threads = (WORKGROUPS * 256) as u64;
    let flops = threads * ITERS as u64 * TK as u64 * 32;

    println!("{:>14} {:>10} {:>12}", "modo", "ms", "GFLOP/s");
    for (nome, fonte) in [("compartilhada", fonte_compartilhada()), ("subgrupo", fonte_subgrupo())] {
        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(fonte.into()),
        });
        let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &m,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        });
        let rodar = || {
            let mut e = dev.create_command_encoder(&Default::default());
            {
                let mut c = e.begin_compute_pass(&Default::default());
                c.set_pipeline(&p);
                c.set_bind_group(0, &bg, &[]);
                c.dispatch_workgroups(WORKGROUPS, 1, 1);
            }
            q.submit(Some(e.finish()));
            dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        rodar();
        let t0 = Instant::now();
        for _ in 0..5 {
            rodar();
        }
        let dt = t0.elapsed().as_secs_f64() / 5.0;
        println!("{:>14} {:>10.2} {:>12.1}", nome, dt * 1e3, flops as f64 / dt / 1e9);
    }
}
