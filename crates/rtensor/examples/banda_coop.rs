//! Quanto do tempo do GEMM cooperativo é mover bytes da memória global, e
//! quanto é os tensor cores? Reaproveita o padrão de acesso exato do kernel
//! real (`bench_coop.rs`) — mesma rasterização L2, mesmo buffer duplo, mesmo
//! despacho — e remove só `coopLoadT`/`coopMultiplyAdd`/`coopStoreT`.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example banda_coop
//!
//! # Por que
//!
//! `ocupacao_coop.rs` e `ocupacao_coop_load.rs` já refutaram duas hipóteses:
//! nem o orçamento de memória de workgroup (`sc`, 24 KB) nem `coopLoadT`
//! repetido explicam por que o GEMM real (~7,15 TFLOP/s) fica tão longe do
//! teto puro de `coopMultiplyAdd` (~36–45 TFLOP/s). A diferença que sobra: as
//! duas sondas reusam os mesmos dados 4096 vezes; o kernel real busca um
//! ladrilho **novo** de `A` e `B` da memória global a cada um dos 256 passos
//! de `K` (em 4096³). Esta sonda isola exatamente essa parte.
//!
//! Se o tempo sem tensor cores ficar perto do tempo com eles, a banda global
//! é o gargalo — os tensor cores já estão essencialmente escondidos atrás da
//! espera por dados. Se cair bem abaixo, sobra outra explicação.

use std::time::Instant;

use wgpu::util::DeviceExt;

fn fonte() -> &'static str {
    r#"
enable f16;

struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, p0: u32, p1: u32, p2: u32 };

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> sa: array<f16, 2048>;
var<workgroup> sb: array<f16, 2048>;

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
    @builtin(local_invocation_index) tid: u32,
) {
    let M = dims.m;
    let N = dims.n;
    let K = dims.k;
    let nm = dims.grid_x;
    let nn = N / 64u;

    let bid = bloco(wg.x, nm, nn);
    let m0 = bid.x * 64u;
    let n0 = bid.y * 64u;

    let ca = tid % 16u;
    let ra = tid / 16u;
    let rb = tid / 16u;
    let cb = (tid % 16u) * 4u;

    guardar_a(0u, ca, ra, carregar_a(m0, 0u, ca, ra, K));
    guardar_b(0u, rb, cb, carregar_b(n0, 0u, rb, cb, N));
    workgroupBarrier();

    // Sem tensor cores: só um acumulador escalar, para que o compilador não
    // elimine as leituras/escritas de memória — e nada mais.
    var acc: f32 = 0.0;

    let passos = K / 16u;
    var cur = 0u;
    for (var t = 0u; t < passos; t = t + 1u) {
        let tem_proximo = t + 1u < passos;
        var prox_a: vec4<f16>;
        var prox_b: vec4<f16>;
        if (tem_proximo) {
            prox_a = carregar_a(m0, (t + 1u) * 16u, ca, ra, K);
            prox_b = carregar_b(n0, (t + 1u) * 16u, rb, cb, N);
        }

        acc = acc + f32(sa[cur * 1024u + tid % 16u]) + f32(sb[cur * 1024u + tid % 16u]);

        if (tem_proximo) {
            guardar_a(1u - cur, ca, ra, prox_a);
            guardar_b(1u - cur, rb, cb, prox_b);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }

    if (tid == 0u) {
        c[(wg.x % (M * N / 4096u))] = acc;
    }
}
"#
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
        required_features: wgpu::Features::SHADER_F16,
        required_limits: ad.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .expect("dispositivo");

    let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(fonte().into()),
    });
    let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &m,
        entry_point: Some("mm"),
        compilation_options: Default::default(),
        cache: None,
    });

    println!("{:>7} {:>10} {:>14} {:>10}", "N", "ms", "GB/s (nominal)", "GFLOP/s-eq");
    for &n in &[2048u32, 4096] {
        let (mm_, k) = (n, n);
        let a: Vec<u16> = vec![0x3c00; (mm_ * k) as usize]; // 1.0 em f16, conteúdo não importa
        let b: Vec<u16> = vec![0x3c00; (k * n) as usize];
        let c: Vec<f32> = vec![0.0; 4096];

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
            usage: wgpu::BufferUsages::STORAGE,
        });
        let grid_x = mm_ / 64;
        let grupo = 8u32;
        let dims = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[mm_, n, k, grid_x, grupo, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
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

        let rodar = || {
            let mut enc = dev.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&p);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups((mm_ / 64) * (n / 64), 1, 1);
            }
            q.submit(Some(enc.finish()));
            dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        rodar();
        let t0 = Instant::now();
        const REPS: u32 = 8;
        let mut tempos = Vec::new();
        for _ in 0..REPS {
            let ti = Instant::now();
            rodar();
            tempos.push(ti.elapsed().as_secs_f64());
        }
        let _ = t0;
        tempos.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let dt = tempos[tempos.len() / 2];

        // Bytes lidos da memória GLOBAL por thread: 16 por passo de K (4 f16
        // de A + 4 f16 de B), vezes threads, vezes passos.
        let threads = ((mm_ / 64) * (n / 64) * 256) as f64;
        let passos = (k / 16) as f64;
        let bytes = threads * passos * 16.0;
        let gbs = bytes / dt / 1e9;
        // Equivalente em GFLOP/s do GEMM completo, para comparar direto com bench_coop.
        let flop = 2.0 * mm_ as f64 * n as f64 * k as f64;
        let gflops_eq = flop / dt / 1e9;

        println!("{:>7} {:>10.2} {:>14.1} {:>10.1}", n, dt * 1e3, gbs, gflops_eq);
    }
    println!("\npara comparar: o GEMM cooperativo completo (bench_coop.rs) mede 19,1–19,4 ms e 7.087–7.204 GFLOP/s em 4096³");
}
