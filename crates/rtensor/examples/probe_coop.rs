//! Sonda da matriz cooperativa, revisada depois do primeiro diagnóstico.
//!
//! # Histórico e desfecho
//!
//! A primeira versão desta sonda (wgpu 30.0.1, naga 30.0.1, driver NVIDIA
//! 595.91.07, RTX 4070 Laptop) compilava e executava, mas devolvia **zero**.
//! Foi arquivada como não funcional e a técnica adiada — ver
//! RESULTADOS-NEGATIVOS.md.
//!
//! **Desfecho:** a hipótese 1 abaixo era a causa — a placa não anuncia `8×8
//! f32`, e a configuração fora da lista é comportamento indefinido. O mesmo
//! produto nas configurações anunciadas funciona com erro zero: ver
//! `probe_coop_f16.rs`. Esta sonda fica como registro do diagnóstico e como
//! reprodução do sintoma (os zeros) para quem quiser conferir.
//!
//! Revisão posterior encontrou três problemas possíveis nela, cada um testável
//! isoladamente — o que esta versão faz:
//!
//! 1. **Configuração não suportada.** A sonda fixou `8×8 f32` sem consultar
//!    `Adapter::cooperative_matrix_properties()`. O teste upstream do wgpu usa
//!    `16×16 f16` como primeira escolha e só cai para `8×8 f32` se a primeira
//!    não existir — sinal de que `8×8 f32` não é universal. Usar uma
//!    configuração fora da lista do adaptador é comportamento indefinido, e um
//!    driver pode responder com zeros.
//! 2. **Layout.** A sonda usava `coopLoad`, que lê **column-major**, sobre
//!    dados em row-major. O wgpu agora publica a especificação da extensão
//!    (`docs/api-specs/cooperative_matrix.md` no repositório deles), que define
//!    `coopLoad`/`coopStore` como column-major e `coopLoadT`/`coopStoreT` como
//!    row-major — a forma natural para os dados do rtensor.
//! 3. **Forma do workgroup.** A sonda usava `@workgroup_size(32)`; o exemplo
//!    oficial usa `@workgroup_size(8, 8, 1)`. O naga emite as matrizes com
//!    **escopo `Subgroup`** (`back/spv/writer.rs`), então o grupo cooperante é
//!    o subgrupo — mas a forma do workgroup pode ainda assim importar.
//!
//! Além disso, o wgpu hoje publica um **exemplo oficial com teste de CI**
//! (`examples/features/src/cooperative_matrix`), e o `coopStore` da primeira
//! sonda não tinha como ser distinguido de "não escreveu": o código comitado
//! inicializa tudo com zero, então `c[63] = 0` não provava nada. Esta versão
//! escreve uma marca de vida (1234) em `C` antes do dispatch.
//!
//! # O que esta sonda mede
//!
//! As configurações anunciadas pelo adaptador e, se `8×8 f32` estiver na
//! lista, cinco variantes do mesmo produto 8×8 conferido contra a CPU:
//!
//! | Variante | Ponteiros | Leitura | Workgroup |
//! |---|---|---|---|
//! | `exemplo` | armazenamento | `coopLoadT` | 8×8 (o do exemplo oficial) |
//! | `col` | armazenamento | `coopLoad` | 8×8 (column-major sobre row-major: transposto, **não** zero) |
//! | `wg` | memória de workgroup | `coopLoadT` | 8×8 (a forma que um GEMM real usaria) |
//! | `wg32` | memória de workgroup | `coopLoadT` | 32 (1-D) |
//! | `original` | memória de workgroup | `coopLoad` | 32 (o kernel da primeira sonda) |
//!
//! Se nenhuma variante acertar, a conclusão continua sendo adiar — mas agora
//! com a causa localizada em vez de presumida.

use wgpu::util::DeviceExt;

const MARCA_VIDA: f32 = 1234.0;

const EXEMPLO: &str = r#"
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn mm(@builtin(local_invocation_id) l: vec3<u32>) {
    let ma = coopLoadT<coop_mat8x8<f32, A>>(&a[0], 8u);
    let mb = coopLoadT<coop_mat8x8<f32, B>>(&b[0], 8u);
    var mc = coopLoadT<coop_mat8x8<f32, C>>(&c[0], 8u);
    mc = coopMultiplyAdd(ma, mb, mc);
    coopStoreT(mc, &c[0], 8u);
}
"#;

const COL: &str = r#"
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn mm(@builtin(local_invocation_id) l: vec3<u32>) {
    let ma = coopLoad<coop_mat8x8<f32, A>>(&a[0], 8u);
    let mb = coopLoad<coop_mat8x8<f32, B>>(&b[0], 8u);
    var mc = coopLoad<coop_mat8x8<f32, C>>(&c[0], 8u);
    mc = coopMultiplyAdd(ma, mb, mc);
    coopStore(mc, &c[0], 8u);
}
"#;

const WG: &str = r#"
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

var<workgroup> sa: array<f32, 64>;
var<workgroup> sb: array<f32, 64>;
var<workgroup> sc: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn mm(@builtin(local_invocation_id) l: vec3<u32>) {
    let li = l.x + l.y * 8u;
    for (var i = li; i < 64u; i += 64u) {
        sa[i] = a[i];
        sb[i] = b[i];
        sc[i] = 0.0;
    }
    workgroupBarrier();
    let ma = coopLoadT<coop_mat8x8<f32, A>>(&sa[0], 8u);
    let mb = coopLoadT<coop_mat8x8<f32, B>>(&sb[0], 8u);
    var mc = coopLoadT<coop_mat8x8<f32, C>>(&sc[0], 8u);
    mc = coopMultiplyAdd(ma, mb, mc);
    coopStoreT(mc, &sc[0], 8u);
    workgroupBarrier();
    for (var i = li; i < 64u; i += 64u) {
        c[i] = sc[i];
    }
}
"#;

const WG32: &str = r#"
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

var<workgroup> sa: array<f32, 64>;
var<workgroup> sb: array<f32, 64>;
var<workgroup> sc: array<f32, 64>;

@compute @workgroup_size(32, 1, 1)
fn mm(@builtin(local_invocation_id) l: vec3<u32>) {
    for (var i = l.x; i < 64u; i += 32u) {
        sa[i] = a[i];
        sb[i] = b[i];
        sc[i] = 0.0;
    }
    workgroupBarrier();
    let ma = coopLoadT<coop_mat8x8<f32, A>>(&sa[0], 8u);
    let mb = coopLoadT<coop_mat8x8<f32, B>>(&sb[0], 8u);
    var mc = coopLoadT<coop_mat8x8<f32, C>>(&sc[0], 8u);
    mc = coopMultiplyAdd(ma, mb, mc);
    coopStoreT(mc, &sc[0], 8u);
    workgroupBarrier();
    for (var i = l.x; i < 64u; i += 32u) {
        c[i] = sc[i];
    }
}
"#;

const ORIGINAL: &str = r#"
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

var<workgroup> sa: array<f32, 64>;
var<workgroup> sb: array<f32, 64>;
var<workgroup> sc: array<f32, 64>;

@compute @workgroup_size(32, 1, 1)
fn mm(@builtin(local_invocation_id) l: vec3<u32>) {
    for (var i = l.x; i < 64u; i += 32u) {
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
    for (var i = l.x; i < 64u; i += 32u) {
        c[i] = sc[i];
    }
}
"#;

/// Dados de entrada: `A[i][j] = j + 1` e `B[i][j] = i · 0,5`, de modo que
/// `C = A·B` dá 84 em toda posição — a mesma carga da primeira sonda.
fn dados() -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..64).map(|i| (i % 8) as f32 + 1.0).collect();
    let b: Vec<f32> = (0..64).map(|i| (i / 8) as f32 * 0.5).collect();
    (a, b)
}

/// Referência em f64 para as variantes com leitura row-major.
fn esperado_rowmajor(a: &[f32], b: &[f32]) -> Vec<f64> {
    let mut e = vec![0.0f64; 64];
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                e[i * 8 + j] += a[i * 8 + k] as f64 * b[k * 8 + j] as f64;
            }
        }
    }
    e
}

/// Referência em f64 para a leitura column-major sobre dados row-major: a
/// carga interpreta `ptr[r + c·8]` e por isso lê a transposta; e a escrita
/// column-major devolve `out[r + c·8] = P[r][c]`. É o que a variante `col` e
/// o kernel `original` fazem — resultado transposto, **não** zero.
fn esperado_colmajor(a: &[f32], b: &[f32]) -> Vec<f64> {
    let mut e = vec![0.0f64; 64];
    for r in 0..8 {
        for c in 0..8 {
            for k in 0..8 {
                e[r + c * 8] += a[k * 8 + r] as f64 * b[c * 8 + k] as f64;
            }
        }
    }
    e
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

    println!("adaptador: {} — driver {:?}", ad.get_info().name, ad.get_info().driver);

    // Passo 1: o que o adaptador anuncia. É isto que a primeira sonda não
    // consultou — e cuja resposta decide se `8×8 f32` sequer existe aqui.
    let props = ad.cooperative_matrix_properties();
    if props.is_empty() {
        println!("o adaptador anuncia nenhuma configuração de matriz cooperativa");
        return;
    }
    println!("configurações anunciadas:");
    for p in &props {
        println!(
            "  {}×{}×{}  AB: {:?}  CR: {:?}  saturando: {}",
            p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type, p.saturating_accumulation
        );
    }
    let tem_8x8_f32 = props
        .iter()
        .any(|p| p.m_size == 8 && p.n_size == 8 && p.k_size == 8);
    println!(
        "8×8 f32 anunciado: {}\n",
        if tem_8x8_f32 {
            "sim"
        } else {
            "NÃO — configuração fora da lista é comportamento indefinido"
        }
    );

    let f = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    if !ad.features().contains(f) {
        eprintln!("adaptador sem a feature de matriz cooperativa");
        return;
    }

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

    let (a, b) = dados();
    let esperado_row = esperado_rowmajor(&a, &b);
    let esperado_col = esperado_colmajor(&a, &b);

    let buf = |dados: &[f32], escrita: bool| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(dados),
            usage: wgpu::BufferUsages::STORAGE
                | if escrita { wgpu::BufferUsages::COPY_SRC } else { wgpu::BufferUsages::empty() },
        })
    };

    let variantes: [(&str, &str, &Vec<f64>); 5] = [
        ("exemplo", EXEMPLO, &esperado_row),
        ("col", COL, &esperado_col),
        ("wg", WG, &esperado_row),
        ("wg32", WG32, &esperado_row),
        ("original", ORIGINAL, &esperado_col),
    ];

    for (nome, fonte, esperado) in variantes {
        let ba = buf(&a, false);
        let bb = buf(&b, false);
        // Marca de vida: se `c` voltar com 1234, o `coopStore` não escreveu
        // nada — distinção que a primeira sonda não era capaz de fazer.
        let bc = buf(&vec![MARCA_VIDA; 64], true);

        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(nome),
            source: wgpu::ShaderSource::Wgsl((*fonte).into()),
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
        let obtido: Vec<f64> = bytemuck::cast_slice(&vista[..])
            .iter()
            .map(|x: &f32| *x as f64)
            .collect();
        drop(vista);

        let erro = esperado
            .iter()
            .zip(&obtido)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        let escreveu = obtido.iter().any(|x| (*x - MARCA_VIDA as f64).abs() > 1.0);
        let veredito = if erro < 1e-4 {
            "CORRETO"
        } else if !escreveu {
            "não escreveu (só a marca de vida)"
        } else {
            "escreveu, com valor errado"
        };
        println!(
            "{:>9}: erro máx. {erro:.3e} — c[0..4] = {:?} — {veredito}",
            nome,
            &obtido[..4]
        );
    }
}
