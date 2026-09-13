//! f32 contra f16 empacotado: a aritmética dobra?
use std::time::Instant;
use wgpu::util::DeviceExt;

const FMA: u32 = 65536;
const WG: u32 = 2048;

fn fonte(n: u32, f16: bool) -> String {
    let iters = FMA / n;
    let (pre, tipo, escalar, um, meio) = if f16 {
        ("enable f16;\n", "vec2<f16>", "f16", "vec2<f16>(1.0001h, 1.0001h)", "vec2<f16>(0.5h, 0.5h)")
    } else {
        ("", "vec2<f32>", "f32", "vec2<f32>(1.0001, 1.0001)", "vec2<f32>(0.5, 0.5)")
    };
    let decl: String = (0..n)
        .map(|i| format!("    var a{i} = {tipo}({escalar}(f32(g.x + {i}u) * 0.0001));\n"))
        .collect();
    let corpo: String = (0..n).map(|i| format!("            a{i} = a{i} * x + m;\n")).collect();
    let soma: String = (0..n).map(|i| format!(" + f32(a{i}.x) + f32(a{i}.y)")).collect();
    format!(r#"{pre}
@group(0) @binding(0) var<storage, read_write> saida: array<f32>;

@compute @workgroup_size(256)
fn probe(@builtin(global_invocation_id) g: vec3<u32>) {{
{decl}    let x = {um};
    let m = {meio};
    for (var t = 0u; t < {iters}u; t = t + 1u) {{
{corpo}    }}
    saida[g.x % 1024u] = 0.0{soma};
}}
"#)
}

fn main() {
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ad = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu
                && a.get_info().backend == wgpu::Backend::Vulkan)
        .expect("sem GPU discreta");
    let tem_f16 = ad.features().contains(wgpu::Features::SHADER_F16);
    println!("{}  |  SHADER_F16: {}", ad.get_info().name, tem_f16);

    let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: if tem_f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() },
        required_limits: ad.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        ..Default::default()
    })).expect("dispositivo");

    let buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&[0.0f32; 1024]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    println!("\n{:>16} {:>12} {:>10} {:>12}", "tipo", "pares/thread", "ms", "GFLOP/s");
    for f16 in [false, true] {
        if f16 && !tem_f16 { continue; }
        for n in [8u32, 16, 32, 48] {
            let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None, source: wgpu::ShaderSource::Wgsl(fonte(n, f16).into()),
            });
            let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None, layout: None, module: &m, entry_point: Some("probe"),
                compilation_options: Default::default(), cache: None,
            });
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &p.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
            });
            let rodar = || {
                let mut e = dev.create_command_encoder(&Default::default());
                { let mut c = e.begin_compute_pass(&Default::default());
                  c.set_pipeline(&p); c.set_bind_group(0, &bg, &[]); c.dispatch_workgroups(WG, 1, 1); }
                q.submit(Some(e.finish()));
                dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            };
            rodar();
            let t0 = Instant::now();
            for _ in 0..5 { rodar(); }
            let dt = t0.elapsed().as_secs_f64() / 5.0;
            // Cada vec2 faz 2 FMAs = 4 flops.
            let flop = 4.0 * FMA as f64 * (WG * 256) as f64;
            println!("{:>16} {:>12} {:>10.2} {:>12.1}",
                if f16 { "f16 empacotado" } else { "f32 (vec2)" }, n, dt * 1e3, flop / dt / 1e9);
        }
    }
}
