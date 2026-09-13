//! GEMM com ladrilhamento em dois níveis — o desenho que o cuBLAS usa.
//!
//! # Por que o kernel ingênuo satura cedo
//!
//! Com um elemento de saída por thread, cada produto `a·b` exige duas leituras
//! da memória compartilhada. A intensidade aritmética é 1 FMA por 2 leituras, e
//! o kernel fica limitado pela banda da memória compartilhada, não pelas ULAs.
//!
//! # Ladrilhamento em dois níveis
//!
//! - **Nível 1 (workgroup → memória compartilhada):** um ladrilho de saída
//!   `64×64` e um passo `K` de 16. Carrega `Aₜ ∈ ℝ^{64×16}` e `Bₜ ∈ ℝ^{16×64}`,
//!   8 KB no total, bem abaixo dos 48 KB disponíveis.
//! - **Nível 2 (thread → registradores):** cada uma das 256 threads calcula um
//!   bloco `4×4` da saída. No laço interno ela lê 4 valores de `A` e 4 de `B`
//!   e faz 16 FMAs — intensidade de **2 FMAs por leitura**, 4× a do ingênuo.
//!
//! O acumulador é `array<vec4<f32>, 4>` com índices constantes, para que o
//! compilador o mantenha em registradores em vez de derramar para memória local.
//!
//! As três variantes diferem apenas em como preenchem os ladrilhos:
//!
//! - `mm`      — `A[M,K] · B[K,N]`
//! - `mm_atb`  — `Aᵀ` com `A[K,M]`, `· B[K,N]`   (`∂L/∂W = Xᵀδ`)
//! - `mm_abt`  — `A[M,K] · Bᵀ` com `B[N,K]`      (`∂L/∂X = δWᵀ`)
//!
//! Cada mapeamento de carga é escolhido para que threads vizinhas leiam
//! endereços vizinhos — sem isso a coalescência se perde e o ganho evapora.
pub const MM_FAST: &str = r#"
struct Dims { m: u32, n: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TM: u32 = 64u;   // ladrilho de saída em linhas
const TN: u32 = 64u;   // ladrilho de saída em colunas
const TK: u32 = 16u;   // passo na dimensão interna
const THREADS: u32 = 256u;

var<workgroup> sa: array<f32, 1024>;   // [TK][TM]
var<workgroup> sb: array<f32, 1024>;   // [TK][TN]

fn acumula(ty: u32, tx: u32, acc: ptr<function, array<vec4<f32>, 4>>) {
    for (var kk = 0u; kk < TK; kk = kk + 1u) {
        let ab = kk * TM + ty * 4u;
        let bb = kk * TN + tx * 4u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        (*acc)[0] = (*acc)[0] + av.x * bv;
        (*acc)[1] = (*acc)[1] + av.y * bv;
        (*acc)[2] = (*acc)[2] + av.z * bv;
        (*acc)[3] = (*acc)[3] + av.w * bv;
    }
}

fn guarda(linha: u32, col: u32, v: f32) {
    if (linha < d.m && col < d.n) { c[linha * d.n + col] = v; }
}

fn escreve(linha0: u32, col0: u32, acc: array<vec4<f32>, 4>) {
    guarda(linha0, col0, acc[0].x);
    guarda(linha0, col0 + 1u, acc[0].y);
    guarda(linha0, col0 + 2u, acc[0].z);
    guarda(linha0, col0 + 3u, acc[0].w);
    guarda(linha0 + 1u, col0, acc[1].x);
    guarda(linha0 + 1u, col0 + 1u, acc[1].y);
    guarda(linha0 + 1u, col0 + 2u, acc[1].z);
    guarda(linha0 + 1u, col0 + 3u, acc[1].w);
    guarda(linha0 + 2u, col0, acc[2].x);
    guarda(linha0 + 2u, col0 + 1u, acc[2].y);
    guarda(linha0 + 2u, col0 + 2u, acc[2].z);
    guarda(linha0 + 2u, col0 + 3u, acc[2].w);
    guarda(linha0 + 3u, col0, acc[3].x);
    guarda(linha0 + 3u, col0 + 1u, acc[3].y);
    guarda(linha0 + 3u, col0 + 2u, acc[3].z);
    guarda(linha0 + 3u, col0 + 3u, acc[3].w);
}

fn passos() -> u32 { return (d.k + TK - 1u) / TK; }

@compute @workgroup_size(16, 16)
fn mm(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let lin0 = w.y * TM;
    let col0 = w.x * TN;
    var acc = array<vec4<f32>, 4>();

    for (var t = 0u; t < passos(); t = t + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let mm_ = idx / TK;            // 0..63, linha dentro do ladrilho
            let kx = idx % TK;             // 0..15, contígua entre threads vizinhas
            let gr = lin0 + mm_;
            let gk = t * TK + kx;
            sa[kx * TM + mm_] = select(0.0, a[gr * d.k + gk], gr < d.m && gk < d.k);
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let kx = idx / TN;             // 0..15
            let nn = idx % TN;             // 0..63, contígua
            let gk = t * TK + kx;
            let gc = col0 + nn;
            sb[kx * TN + nn] = select(0.0, b[gk * d.n + gc], gk < d.k && gc < d.n);
        }
        workgroupBarrier();
        acumula(l.y, l.x, &acc);
        workgroupBarrier();
    }
    escreve(lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}

@compute @workgroup_size(16, 16)
fn mm_atb(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let lin0 = w.y * TM;
    let col0 = w.x * TN;
    var acc = array<vec4<f32>, 4>();

    for (var t = 0u; t < passos(); t = t + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let kx = idx / TM;             // 0..15
            let mm_ = idx % TM;            // 0..63, contígua — A é [K, M]
            let gr = lin0 + mm_;
            let gk = t * TK + kx;
            sa[kx * TM + mm_] = select(0.0, a[gk * d.m + gr], gr < d.m && gk < d.k);
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let kx = idx / TN;
            let nn = idx % TN;
            let gk = t * TK + kx;
            let gc = col0 + nn;
            sb[kx * TN + nn] = select(0.0, b[gk * d.n + gc], gk < d.k && gc < d.n);
        }
        workgroupBarrier();
        acumula(l.y, l.x, &acc);
        workgroupBarrier();
    }
    escreve(lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}

@compute @workgroup_size(16, 16)
fn mm_abt(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let lin0 = w.y * TM;
    let col0 = w.x * TN;
    var acc = array<vec4<f32>, 4>();

    for (var t = 0u; t < passos(); t = t + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let mm_ = idx / TK;
            let kx = idx % TK;
            let gr = lin0 + mm_;
            let gk = t * TK + kx;
            sa[kx * TM + mm_] = select(0.0, a[gr * d.k + gk], gr < d.m && gk < d.k);
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let nn = idx / TK;             // 0..63
            let kx = idx % TK;             // 0..15, contígua — B é [N, K]
            let gk = t * TK + kx;
            let gc = col0 + nn;
            sb[kx * TN + nn] = select(0.0, b[gc * d.k + gk], gk < d.k && gc < d.n);
        }
        workgroupBarrier();
        acumula(l.y, l.x, &acc);
        workgroupBarrier();
    }
    escreve(lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}
"#;

/// GEMM com epílogo fundido: `A = relu(X W + 1ₙb)` num único kernel.
///
/// # O que a fusão elimina
///
/// Sem fusão, cada camada oculta custa três dispatches e cinco travessias de
/// `Z ∈ ℝ^{n×m}` pela memória global:
///
/// ```text
/// mm        escreve Z            1 escrita
/// bias_add  lê Z, escreve Z      1 leitura + 1 escrita
/// relu      lê Z, escreve A      1 leitura + 1 escrita
/// ```
///
/// Com o epílogo dentro do GEMM, o viés e a ReLU são aplicados ao acumulador
/// **ainda em registradores**, e sai uma escrita só. Some-se a isso dois
/// dispatches a menos, e portanto duas barreiras de memória a menos por camada.
///
/// # Por que não é preciso guardar a pré-ativação
///
/// O adjunto da ReLU precisa saber onde `Z > 0`. Como `A = max(Z, 0)`, vale
/// `A > 0 ⟺ Z > 0` — a saída carrega a mesma máscara que a entrada. É a mesma
/// observação registrada no kernel do TensorFlow: *"features: either the inputs
/// that were passed to the Relu, or its outputs (using either one yields the
/// same result here)"*. Guardar `Z` seria redundante, e a camada oculta passa a
/// precisar de um buffer a menos.
///
/// `flag_relu` no uniforme distingue camada oculta (com ReLU) de camada de
/// saída (só o viés), para que um kernel sirva às duas.
pub const MM_EPILOGO: &str = r#"
struct Dims { m: u32, n: u32, k: u32, flag_relu: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<storage, read> vies: array<f32>;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

fn epilogo(v: f32, col: u32) -> f32 {
    let z = v + vies[col];
    if (d.flag_relu == 1u) { return max(z, 0.0); }
    return z;
}

fn guarda(linha: u32, col: u32, v: f32) {
    if (linha < d.m && col < d.n) { c[linha * d.n + col] = epilogo(v, col); }
}

fn escreve(linha0: u32, col0: u32, acc: array<vec4<f32>, 4>) {
    guarda(linha0, col0, acc[0].x);
    guarda(linha0, col0 + 1u, acc[0].y);
    guarda(linha0, col0 + 2u, acc[0].z);
    guarda(linha0, col0 + 3u, acc[0].w);
    guarda(linha0 + 1u, col0, acc[1].x);
    guarda(linha0 + 1u, col0 + 1u, acc[1].y);
    guarda(linha0 + 1u, col0 + 2u, acc[1].z);
    guarda(linha0 + 1u, col0 + 3u, acc[1].w);
    guarda(linha0 + 2u, col0, acc[2].x);
    guarda(linha0 + 2u, col0 + 1u, acc[2].y);
    guarda(linha0 + 2u, col0 + 2u, acc[2].z);
    guarda(linha0 + 2u, col0 + 3u, acc[2].w);
    guarda(linha0 + 3u, col0, acc[3].x);
    guarda(linha0 + 3u, col0 + 1u, acc[3].y);
    guarda(linha0 + 3u, col0 + 2u, acc[3].z);
    guarda(linha0 + 3u, col0 + 3u, acc[3].w);
}

@compute @workgroup_size(16, 16)
fn mm_bias(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let lin0 = w.y * TM;
    let col0 = w.x * TN;
    var acc = array<vec4<f32>, 4>();

    let passos = (d.k + TK - 1u) / TK;
    for (var t = 0u; t < passos; t = t + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let mm_ = idx / TK;
            let kx = idx % TK;
            let gr = lin0 + mm_;
            let gk = t * TK + kx;
            sa[kx * TM + mm_] = select(0.0, a[gr * d.k + gk], gr < d.m && gk < d.k);
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let kx = idx / TN;
            let nn = idx % TN;
            let gk = t * TK + kx;
            let gc = col0 + nn;
            sb[kx * TN + nn] = select(0.0, b[gk * d.n + gc], gk < d.k && gc < d.n);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TK; kk = kk + 1u) {
            let ab = kk * TM + l.y * 4u;
            let bb = kk * TN + l.x * 4u;
            let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
            let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
            acc[0] = acc[0] + av.x * bv;
            acc[1] = acc[1] + av.y * bv;
            acc[2] = acc[2] + av.z * bv;
            acc[3] = acc[3] + av.w * bv;
        }
        workgroupBarrier();
    }
    escreve(lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}
"#;
