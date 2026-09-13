//! Kernels WGSL. Cada família tem seu próprio módulo para que o layout de
//! bindings derivado automaticamente case exatamente com o que é vinculado.

/// Produto matricial com ladrilhos 16×16 em memória de workgroup.
///
/// As três variantes compartilham a semântica `C[M,N] = A ·  B` com dimensão
/// interna `K`; mudam apenas os índices de leitura:
///
/// - `mm`      — `A[M,K] · B[K,N]`            (avanço: `Z = X W`)
/// - `mm_atb`  — `Aᵀ` com `A[K,M]`, `· B[K,N]` (`∂L/∂W = Xᵀ δ`)
/// - `mm_abt`  — `A[M,K] · Bᵀ` com `B[N,K]`    (`∂L/∂X = δ Wᵀ`)
pub const MM: &str = r#"
struct Dims { m: u32, n: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TS: u32 = 16u;
var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

fn tiles(k: u32) -> u32 { return (k + TS - 1u) / TS; }

@compute @workgroup_size(16, 16)
fn mm(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let row = w.y * TS + l.y;
    let col = w.x * TS + l.x;
    var acc = 0.0;
    for (var t = 0u; t < tiles(d.k); t = t + 1u) {
        let ac = t * TS + l.x;
        let br = t * TS + l.y;
        ta[l.y * TS + l.x] = select(0.0, a[row * d.k + ac], row < d.m && ac < d.k);
        tb[l.y * TS + l.x] = select(0.0, b[br * d.n + col], br < d.k && col < d.n);
        workgroupBarrier();
        for (var i = 0u; i < TS; i = i + 1u) {
            acc = acc + ta[l.y * TS + i] * tb[i * TS + l.x];
        }
        workgroupBarrier();
    }
    if (row < d.m && col < d.n) { c[row * d.n + col] = acc; }
}

@compute @workgroup_size(16, 16)
fn mm_atb(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let row = w.y * TS + l.y;
    let col = w.x * TS + l.x;
    var acc = 0.0;
    for (var t = 0u; t < tiles(d.k); t = t + 1u) {
        let ar = t * TS + l.x;
        let br = t * TS + l.y;
        // A guardada como [K, M]; lemos a transposta.
        ta[l.y * TS + l.x] = select(0.0, a[ar * d.m + row], row < d.m && ar < d.k);
        tb[l.y * TS + l.x] = select(0.0, b[br * d.n + col], br < d.k && col < d.n);
        workgroupBarrier();
        for (var i = 0u; i < TS; i = i + 1u) {
            acc = acc + ta[l.y * TS + i] * tb[i * TS + l.x];
        }
        workgroupBarrier();
    }
    if (row < d.m && col < d.n) { c[row * d.n + col] = acc; }
}

@compute @workgroup_size(16, 16)
fn mm_abt(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let row = w.y * TS + l.y;
    let col = w.x * TS + l.x;
    var acc = 0.0;
    for (var t = 0u; t < tiles(d.k); t = t + 1u) {
        let ac = t * TS + l.x;
        let bc = t * TS + l.y;
        ta[l.y * TS + l.x] = select(0.0, a[row * d.k + ac], row < d.m && ac < d.k);
        // B guardada como [N, K]; lemos a transposta.
        tb[l.y * TS + l.x] = select(0.0, b[col * d.k + bc], bc < d.k && col < d.n);
        workgroupBarrier();
        for (var i = 0u; i < TS; i = i + 1u) {
            acc = acc + ta[l.y * TS + i] * tb[i * TS + l.x];
        }
        workgroupBarrier();
    }
    if (row < d.m && col < d.n) { c[row * d.n + col] = acc; }
}
"#;

/// `Z ← Z + 1ₙ b` (viés difundido) e `Y ← max(X, 0)`.
pub const VEC: &str = r#"
// `gx` é a largura da grade de workgroups: um dispatch 1-D estoura o teto de
// 65535 grupos por dimensão já em 16,7M elementos (uma camada 4096×4096), então
// a grade é 2-D e o índice linear é reconstruído aqui.
struct P { n: u32, cols: u32, gx: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

fn linear(w: vec3<u32>, l: vec3<u32>, gx: u32) -> u32 {
    return (w.y * gx + w.x) * 256u + l.x;
}

@compute @workgroup_size(256)
fn bias_add(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i = linear(w, l, p.gx);
    if (i >= p.n) { return; }
    dst[i] = dst[i] + src[i % p.cols];
}

@compute @workgroup_size(256)
fn relu(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i = linear(w, l, p.gx);
    if (i >= p.n) { return; }
    dst[i] = max(src[i], 0.0);
}
"#;

/// `δ ← δ ⊙ 1[Z > 0]` — o adjunto da ReLU, avaliado na pré-ativação.
///
/// Opera **in-place** sobre `δ`, que já contém `Ā = δ⁺(W⁺)ᵀ`. Isso evita ligar
/// o mesmo buffer como leitura e escrita no mesmo dispatch, o que o WebGPU
/// proíbe: `STORAGE_READ_WRITE` é um uso exclusivo dentro de um passe.
pub const RELU_BWD: &str = r#"
struct P { n: u32, cols: u32, gx: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> dz: array<f32>;

@compute @workgroup_size(256)
fn relu_bwd(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i = (w.y * p.gx + w.x) * 256u + l.x;
    if (i >= p.n) { return; }
    if (z[i] <= 0.0) { dz[i] = 0.0; }
}
"#;

/// Softmax + entropia cruzada fundidos, uma linha por thread.
///
/// Calcula `p = softmax(z)` estabilizado pelo máximo, grava a perda da linha e
/// já devolve o adjunto `∂L/∂Z = (P − Y)/n` — o mesmo cancelamento que o
/// `xent_op.h` do TensorFlow faz.
pub const XENT: &str = r#"
struct P { rows: u32, cols: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> labels: array<u32>;
@group(0) @binding(3) var<storage, read_write> delta: array<f32>;
@group(0) @binding(4) var<storage, read_write> loss: array<f32>;

@compute @workgroup_size(64)
fn xent(@builtin(global_invocation_id) g: vec3<u32>) {
    let r = g.x;
    if (r >= p.rows) { return; }
    let base = r * p.cols;

    var mx = -3.4e38;
    for (var j = 0u; j < p.cols; j = j + 1u) { mx = max(mx, logits[base + j]); }

    var sum = 0.0;
    for (var j = 0u; j < p.cols; j = j + 1u) { sum = sum + exp(logits[base + j] - mx); }

    let y = labels[r];
    loss[r] = -(logits[base + y] - mx - log(sum));

    let inv = 1.0 / f32(p.rows);
    for (var j = 0u; j < p.cols; j = j + 1u) {
        let prob = exp(logits[base + j] - mx) / sum;
        delta[base + j] = (prob - select(0.0, 1.0, j == y)) * inv;
    }
}
"#;

/// `∂L/∂b = 1ₙᵀ δ` — redução por eixo em dois estágios.
///
/// Uma thread por coluna limita o paralelismo a `cols` threads: com 1024
/// colunas são 1024 threads numa GPU que quer centenas de milhares, e o passo
/// de treino fica preso nessa redução assim que o lote cresce.
///
/// O mesmo kernel roda duas vezes:
///
/// 1. cada thread soma um bloco de `chunk` linhas → `parcial ∈ ℝ^{⌈n/chunk⌉×m}`
/// 2. o mesmo kernel soma as parciais → `∂L/∂b ∈ ℝ^{1×m}`
///
/// Threads vizinhas leem colunas vizinhas da mesma linha, então os acessos
/// permanecem coalescidos nos dois estágios.
pub const COLSUM: &str = r#"
struct P { rows: u32, cols: u32, chunk: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn colsum(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let j = w.x * 64u + l.x;
    if (j >= p.cols) { return; }
    let ini = w.y * p.chunk;
    if (ini >= p.rows) { return; }
    let fim = min(ini + p.chunk, p.rows);
    var acc = 0.0;
    for (var i = ini; i < fim; i = i + 1u) { acc = acc + src[i * p.cols + j]; }
    dst[w.y * p.cols + j] = acc;
}
"#;

/// Adam fundido: `m`, `v` e `θ` atualizados numa única passada.
///
/// `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g²`,
/// `θ ← θ − α (m/c₁) ⊘ (√(v/c₂) + ε)`, com `c₁ = 1−β₁ᵗ`, `c₂ = 1−β₂ᵗ`.
pub const ADAM: &str = r#"
struct P { n: u32, lr: f32, b1: f32, b2: f32, eps: f32, c1: f32, c2: f32, gx: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> theta: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn adam(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i = (w.y * p.gx + w.x) * 256u + l.x;
    if (i >= p.n) { return; }
    let grad = g[i];
    let mi = p.b1 * m[i] + (1.0 - p.b1) * grad;
    let vi = p.b2 * v[i] + (1.0 - p.b2) * grad * grad;
    m[i] = mi;
    v[i] = vi;
    theta[i] = theta[i] - p.lr * (mi / p.c1) / (sqrt(vi / p.c2) + p.eps);
}
"#;
