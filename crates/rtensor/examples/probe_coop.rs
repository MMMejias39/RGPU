//! Sonda mínima da matriz cooperativa: um produto 8×8 conferido contra a CPU.
//!
//! # Estado: não funcional nesta combinação
//!
//! wgpu 30.0.1, naga 30.0.1, driver NVIDIA 595.91.07, RTX 4070 Laptop.
//!
//! O que **funciona**:
//!
//! - o adaptador anuncia `EXPERIMENTAL_COOPERATIVE_MATRIX`
//! - o dispositivo aceita a feature, mediante o token `ExperimentalFeatures`
//! - o WGSL compila com `enable wgpu_cooperative_matrix;`, os tipos
//!   `coop_mat8x8<f32, A|B|C>` e as funções `coopLoad`, `coopMultiplyAdd` e
//!   `coopStore`
//! - o kernel executa, e o `coopStore` **escreve de fato** — verificado com uma
//!   marca de vida em `c[63]`, que é sobrescrita
//!
//! O que **não funciona**: o resultado é zero. Testado com ponteiros para
//! buffer de armazenamento e para memória de workgroup, com passo explícito.
//! Não encontrei documentação da semântica esperada de ponteiro e passo, e a
//! API não tem especificação publicada.
//!
//! # Dois preços, não um
//!
//! Além de não funcionar, habilitar a feature exige
//! `ExperimentalFeatures::enabled()`, que é **`unsafe fn`**. O wgpu declara que
//! estas APIs podem conter bugs que levam a comportamento indefinido a partir
//! de código aparentemente seguro. Adotá-la custaria a propriedade "zero
//! `unsafe`" do RGPU — que é um dos argumentos do projeto.
//!
//! Esta sonda fica como caso de reprodução: quem quiser retomar quando a API
//! amadurecer começa daqui, e não do zero.

use wgpu::util::DeviceExt;

const FONTE: &str = r#"
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

// Um GEMM real carrega os ladrilhos em memória de workgroup antes de
// multiplicar; é essa a forma que interessa testar.
var<workgroup> sa: array<f32, 64>;
var<workgroup> sb: array<f32, 64>;
var<workgroup> sc: array<f32, 64>;

@compute @workgroup_size(32)
fn mm(@builtin(local_invocation_id) l: vec3<u32>) {
    for (var i = l.x; i < 64u; i = i + 32u) {
        sa[i] = a[i];
        sb[i] = b[i];
        sc[i] = 0.0;
    }
    workgroupBarrier();

    let ma = coopLoad<coop_mat8x8<f32, A>>(&sa[0], 8u);
    let mb = coopLoad<coop_mat8x8<f32, B>>(&sb[0], 8u);
    var mc = coopLoad<coop_mat8x8<f32, C>>(&sc[0], 8u);
    mc = coopMultiplyAdd(ma, mb, mc);
    coopStore(mc, &sc[0], 8u);
    workgroupBarrier();

    for (var i = l.x; i < 64u; i = i + 32u) {
        c[i] = sc[i];
    }
}
"#;

fn main() {
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ad = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .find(|a| {
            let i = a.get_info();
            i.device_type == wgpu::DeviceType::DiscreteGpu && i.backend == wgpu::Backend::Vulkan
        })
        .expect("sem GPU discreta");

    let f = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    if !ad.features().contains(f) {
        eprintln!("adaptador sem matriz cooperativa");
        return;
    }
    println!("{}", ad.get_info().name);

    let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: f,
        // A matriz cooperativa é experimental e exige consentimento explícito.
        //
        // `enabled()` é `unsafe fn`: o wgpu declara que estas APIs podem conter
        // bugs que levam a comportamento indefinido a partir de código
        // aparentemente seguro. Esta sonda existe justamente para medir o preço
        // dessa concessão antes de decidir se o projeto a aceita.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        required_limits: ad.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        ..Default::default()
    }))
    .expect("dispositivo");

    // A[8×8] e B[8×8] com valores simples de conferir.
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32 + 1.0).collect();
    let b: Vec<f32> = (0..64).map(|i| (i / 8) as f32 * 0.5).collect();

    let buf = |dados: &[f32], escrita: bool| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(dados),
            usage: wgpu::BufferUsages::STORAGE
                | if escrita { wgpu::BufferUsages::COPY_SRC } else { wgpu::BufferUsages::empty() },
        })
    };
    let ba = buf(&a, false);
    let bb = buf(&b, false);
    let bc = buf(&vec![0.0f32; 64], true);

    let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(FONTE.into()),
    });
    let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
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

    let leitura = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
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
    enc.copy_buffer_to_buffer(&bc, 0, &leitura, 0, 256);
    q.submit(Some(enc.finish()));

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

    // Referência na CPU.
    let mut esperado = vec![0.0f32; 64];
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                esperado[i * 8 + j] += a[i * 8 + k] * b[k * 8 + j];
            }
        }
    }
    let erro = esperado
        .iter()
        .zip(&obtido)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    println!("erro máximo absoluto: {erro:.3e}");
    println!("esperado[0..4]: {:?}", &esperado[..4]);
    println!("obtido  [0..4]: {:?}", &obtido[..4]);
    println!("c[63] = {} (1234 = coopStore não escreveu; outro = escreveu)", obtido[63]);
    println!("{}", if erro < 1e-4 { "MATRIZ COOPERATIVA FUNCIONA" } else { "RESULTADO ERRADO" });
}
