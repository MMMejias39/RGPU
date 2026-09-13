//! Quanto a memória compartilhada entrega, no padrão exato do nosso GEMM.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example banda_compartilhada
//!
//! # O método
//!
//! Três variantes leem **exatamente os mesmos bytes** da memória compartilhada,
//! no mesmo padrão de endereços do laço interno do GEMM, e diferem só no que
//! fazem com eles:
//!
//! - `fma` — as 16 multiplicações-acumulações por passo, como no GEMM real
//! - `soma` — apenas soma os valores lidos: mesmo tráfego, 1/8 da aritmética
//! - `broadcast` — todas as threads leem o mesmo endereço, o caso sem conflito
//!
//! Se `soma` for muito mais rápida que `fma`, o limite é aritmética. Se as duas
//! empatarem, o limite são as leituras. E `broadcast` dá o teto do canal.

use std::time::Instant;

use wgpu::util::DeviceExt;

const WORKGROUPS: u32 = 2048;
const ITERS: u32 = 256;
const TK: u32 = 16;
/// Bytes lidos por thread a cada passo `kk`: 4 floats de `A` e 4 de `B`.
const BYTES_POR_KK: u64 = 32;

fn fonte(modo: &str) -> String {
    let corpo = match modo {
        "fma" => "
            acc[0] = acc[0] + av.x * bv;
            acc[1] = acc[1] + av.y * bv;
            acc[2] = acc[2] + av.z * bv;
            acc[3] = acc[3] + av.w * bv;",
        "soma" => "
            acc[0] = acc[0] + av;
            acc[1] = acc[1] + bv;",
        _ => "
            acc[0] = acc[0] + av;
            acc[1] = acc[1] + bv;",
    };
    // O broadcast usa o mesmo endereço para todas as threads do warp.
    let (ia, ib) = if modo == "broadcast" {
        ("kk * 65u", "kk * 65u")
    } else {
        ("kk * 65u + l.y * 4u", "kk * 65u + l.x * 4u")
    };
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
            let ab = {ia};
            let bb = {ib};
            let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
            let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);{corpo}
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
    println!("{}\n", ad.get_info().name);

    let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
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
    let bytes = threads * ITERS as u64 * TK as u64 * BYTES_POR_KK;
    // No modo `fma` são 32 flops por passo `kk` e por thread.
    let flops = threads * ITERS as u64 * TK as u64 * 32;

    println!("{:>12} {:>10} {:>12} {:>12}", "modo", "ms", "TB/s (lido)", "GFLOP/s");
    for modo in ["fma", "soma", "broadcast"] {
        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(fonte(modo).into()),
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

        let gflops = if modo == "fma" {
            format!("{:.1}", flops as f64 / dt / 1e9)
        } else {
            "—".into()
        };
        println!(
            "{:>12} {:>10.2} {:>12.2} {:>12}",
            modo,
            dt * 1e3,
            bytes as f64 / dt / 1e12,
            gflops
        );
    }
}
