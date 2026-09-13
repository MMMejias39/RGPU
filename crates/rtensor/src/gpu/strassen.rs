//! Strassen de um nível: sete produtos no lugar de oito.
//!
//! Partindo `A`, `B` e `C` em blocos `2×2`, o algoritmo clássico faz oito
//! multiplicações de bloco. Strassen (1969) faz **sete**, à custa de somas:
//!
//! ```text
//! M₁ = (A₁₁ + A₂₂)(B₁₁ + B₂₂)      C₁₁ = M₁ + M₄ − M₅ + M₇
//! M₂ = (A₂₁ + A₂₂) B₁₁             C₁₂ = M₃ + M₅
//! M₃ = A₁₁ (B₁₂ − B₂₂)             C₂₁ = M₂ + M₄
//! M₄ = A₂₂ (B₂₁ − B₁₁)             C₂₂ = M₁ − M₂ + M₃ + M₆
//! M₅ = (A₁₁ + A₁₂) B₂₂
//! M₆ = (A₂₁ − A₁₁)(B₁₁ + B₁₂)
//! M₇ = (A₁₂ − A₂₂)(B₂₁ + B₂₂)
//! ```
//!
//! Cada produto tem metade das dimensões, então custa 1/8 do original. Sete
//! deles somam **7/8**: 12,5% de multiplicações a menos. As somas são `O(n²)`
//! contra `O(n³)` das multiplicações, e se diluem quando `n` cresce.
//!
//! # O custo numérico
//!
//! Strassen satisfaz apenas um limite de erro **por norma**,
//! `‖C − Ĉ‖ ≤ f(n)ε‖A‖‖B‖`, e não o limite por elemento do algoritmo clássico.
//! As subtrações de blocos podem cancelar dígitos significativos. A literatura
//! mede cerca de duas ordens de grandeza de erro a mais em `n = 16384`, e a
//! degradação acelera acima de dois níveis de recursão — por isso aqui há
//! **um** nível apenas, e a comparação contra a CPU em `tests/gpu.rs` mede
//! quanto custou.
//!
//! # Por que empacotar
//!
//! Os blocos são sub-matrizes com passo de linha igual ao da matriz inteira. Em
//! vez de ensinar o GEMM a lidar com passo arbitrário — o que tocaria quatro
//! kernels —, as combinações de blocos são escritas contíguas num buffer
//! temporário. O empacotamento move `O(n²)` dados para um produto que faz
//! `O(n³)` operações, então se paga.

/// `dst = s₁ · src[bloco₁] + s₂ · src[bloco₂]`, com `dst` contíguo.
///
/// `s₂ = 0` copia um bloco só. Os índices `(r, c)` são o canto superior
/// esquerdo de cada bloco dentro da matriz de passo `ld`.
pub const COMBINA: &str = r#"
struct P {
    linhas: u32, colunas: u32, ld: u32, gx: u32,
    r1: u32, c1: u32, r2: u32, c2: u32,
    s1: f32, s2: f32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn combina(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let idx = (w.y * p.gx + w.x) * 256u + l.x;
    if (idx >= p.linhas * p.colunas) { return; }
    let i = idx / p.colunas;
    let j = idx % p.colunas;
    let a = src[(p.r1 + i) * p.ld + p.c1 + j];
    let b = src[(p.r2 + i) * p.ld + p.c2 + j];
    dst[idx] = p.s1 * a + p.s2 * b;
}
"#;

/// `C[bloco] = s₁M₁ + s₂M₂ + s₃M₃ + s₄M₄`, escrevendo num bloco de `C`.
///
/// Sinal zero descarta o termo; `C₁₂` e `C₂₁` usam só dois dos quatro.
pub const ESPALHA: &str = r#"
struct P {
    linhas: u32, colunas: u32, ld: u32, gx: u32,
    r0: u32, c0: u32, p0: u32, p1: u32,
    s1: f32, s2: f32, s3: f32, s4: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> m1: array<f32>;
@group(0) @binding(2) var<storage, read> m2: array<f32>;
@group(0) @binding(3) var<storage, read> m3: array<f32>;
@group(0) @binding(4) var<storage, read> m4: array<f32>;
@group(0) @binding(5) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn espalha(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let idx = (w.y * p.gx + w.x) * 256u + l.x;
    if (idx >= p.linhas * p.colunas) { return; }
    let i = idx / p.colunas;
    let j = idx % p.colunas;
    let v = p.s1 * m1[idx] + p.s2 * m2[idx] + p.s3 * m3[idx] + p.s4 * m4[idx];
    c[(p.r0 + i) * p.ld + p.c0 + j] = v;
}
"#;
