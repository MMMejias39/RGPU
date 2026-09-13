//! Quantos acumuladores cabem numa thread antes de a ocupação desabar?
//!
//! Uso: cargo run -p rtensor --release --features gpu --example ocupacao
//!
//! # Por que medir isso
//!
//! Aumentar o bloco por thread num GEMM eleva a intensidade aritmética — cada
//! valor lido da memória compartilhada é reusado mais vezes. Mas cada elemento
//! do bloco é um acumulador vivo, e acumuladores são registradores. Passado o
//! orçamento do SM, cabem menos warps simultâneos, e menos warps significa
//! menos latência escondida.
//!
//! São duas forças opostas: mais ILP por thread contra menos threads por SM. A
//! sonda mede onde uma vence a outra, em vez de supor.
//!
//! # Como
//!
//! O kernel mantém `N` acumuladores vivos e faz `65536/N` iterações sobre eles,
//! de modo que **o trabalho total não depende de `N`**. Só muda como ele se
//! distribui entre registradores e tempo. `N` entra no código como literal —
//! não como uniforme — para que o laço interno seja desenrolado e os
//! acumuladores fiquem mesmo em registradores, e não derramem para memória
//! local.

use std::time::Instant;

use wgpu::util::DeviceExt;

const FMA_POR_THREAD: u32 = 65536;
const WORKGROUPS: u32 = 2048;

/// Gera o kernel com os acumuladores em **variáveis escalares distintas**,
/// e o corpo escrito por extenso. Não há array nem índice variável, então não
/// há como derramar: o que se mede aqui é ocupação, não derramamento.
fn fonte_desenrolada(n_acc: u32, tam_grupo: u32) -> String {
    let iters = FMA_POR_THREAD / n_acc;
    let decl: String = (0..n_acc)
        .map(|i| format!("    var a{i} = f32(g.x + {i}u) * 0.0001;\n"))
        .collect();
    let corpo: String = (0..n_acc)
        .map(|i| format!("            a{i} = a{i} * x + 0.5;\n"))
        .collect();
    let soma: String = (0..n_acc).map(|i| format!(" + a{i}")).collect();
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> saida: array<f32>;

@compute @workgroup_size({tam_grupo})
fn probe(@builtin(global_invocation_id) g: vec3<u32>) {{
{decl}    let x = 1.0000001;
    for (var t = 0u; t < {iters}u; t = t + 1u) {{
{corpo}    }}
    saida[g.x % 1024u] = 0.0{soma};
}}
"#
    )
}

/// Gera o kernel com os acumuladores num array percorrido por laço — a forma
/// que o compilador pode decidir não desenrolar, derramando para memória local.
fn fonte(n_acc: u32, tam_grupo: u32) -> String {
    let iters = FMA_POR_THREAD / n_acc;
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> saida: array<f32>;

@compute @workgroup_size({tam_grupo})
fn probe(@builtin(global_invocation_id) g: vec3<u32>) {{
    var acc: array<f32, {n_acc}>;
    for (var i = 0u; i < {n_acc}u; i = i + 1u) {{
        acc[i] = f32(g.x + i) * 0.0001;
    }}
    let x = 1.0000001;
    for (var t = 0u; t < {iters}u; t = t + 1u) {{
        for (var i = 0u; i < {n_acc}u; i = i + 1u) {{
            acc[i] = acc[i] * x + 0.5;
        }}
    }}
    var s = 0.0;
    for (var i = 0u; i < {n_acc}u; i = i + 1u) {{
        s = s + acc[i];
    }}
    // Escrita obrigatória: sem ela o compilador elimina o laço inteiro.
    saida[g.x % 1024u] = s;
}}
"#
    )
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adaptadores = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let adapter = adaptadores
        .into_iter()
        .find(|a| {
            let i = a.get_info();
            i.device_type == wgpu::DeviceType::DiscreteGpu && i.backend == wgpu::Backend::Vulkan
        })
        .expect("GPU discreta não encontrada");

    let info = adapter.get_info();
    println!("{} ({:?})", info.name, info.backend);

    let limites = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ocupacao"),
        required_features: wgpu::Features::empty(),
        required_limits: limites,
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .expect("falha ao abrir o dispositivo");

    let saida = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("saida"),
        contents: bytemuck::cast_slice(&[0.0f32; 1024]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    println!(
        "\ntrabalho fixo: {} FMAs por thread, {} workgroups\n",
        FMA_POR_THREAD, WORKGROUPS
    );
    println!(
        "{:>22} {:>7} {:>11} {:>12} {:>12} {:>10}",
        "forma", "grupo", "acumulad.", "ms", "GFLOP/s", "relativo"
    );

    for (rotulo, desenrolado) in [("array + laço", false), ("escalares por extenso", true)] {
      for tam_grupo in [128u32, 256] {
        let mut melhor = 0.0f64;
        let mut linhas = Vec::new();

        for n_acc in [4u32, 8, 16, 24, 32, 48, 64, 96] {
            let codigo = if desenrolado {
                fonte_desenrolada(n_acc, tam_grupo)
            } else {
                fonte(n_acc, tam_grupo)
            };
            let modulo = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(codigo.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &modulo,
                entry_point: Some("probe"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: saida.as_entire_binding(),
                }],
            });

            let mut rodar = || {
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&pipeline);
                    p.set_bind_group(0, &bind, &[]);
                    p.dispatch_workgroups(WORKGROUPS, 1, 1);
                }
                queue.submit(Some(enc.finish()));
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            };

            rodar(); // aquece
            let inicio = Instant::now();
            const REPS: u32 = 5;
            for _ in 0..REPS {
                rodar();
            }
            let dt = inicio.elapsed().as_secs_f64() / REPS as f64;

            let threads = (WORKGROUPS * tam_grupo) as f64;
            let flop = 2.0 * FMA_POR_THREAD as f64 * threads;
            let gflops = flop / dt / 1e9;
            melhor = melhor.max(gflops);
            linhas.push((n_acc, dt, gflops));
        }

        for (n_acc, dt, gflops) in linhas {
            let barra = "█".repeat((gflops / melhor * 20.0).round() as usize);
            println!(
                "{:>22} {:>7} {:>11} {:>12.2} {:>12.1} {:>9.0}% {barra}",
                rotulo,
                tam_grupo,
                n_acc,
                dt * 1e3,
                gflops,
                100.0 * gflops / melhor
            );
        }
        println!();
      }
    }

    println!(
        "Ada Lovelace: 65.536 registradores e 1.536 threads por SM.\n\
         Ocupação plena exige ≤ 42 registradores por thread."
    );
}
