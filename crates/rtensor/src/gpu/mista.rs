//! GEMM em precisão mista: operandos em meia precisão, acumulação em `f32`.
//!
//! # O que isto NÃO entrega
//!
//! Esta implementação nasceu de uma hipótese que a medição derrubou.
//!
//! A hipótese era que o GEMM estava limitado por banda de memória: 2048³ lê
//! ~1 GB em 4,28 ms, o que daria 234 GB/s contra os ~256 GB/s da placa. Meia
//! precisão cortaria os bytes pela metade e quase dobraria a vazão.
//!
//! **A conta estava errada.** Aquele 1 GB supõe reuso zero de L2 — é um limite
//! superior de tráfego, não o tráfego real. Com a rasterização de L2 em
//! operação, a ida efetiva à DRAM é bem menor, e a placa não está perto de
//! saturada.
//!
//! O que se mediu:
//!
//! | Tamanho | Velocidade | Eficiência energética | Erro |
//! |---|---|---|---|
//! | 2048³ | +6,6% | −10,9% | 6.827× |
//! | 4096³ | +1,2% | −5,4% | 5.983× |
//! | 6144³ | +2,3% | −10,0% | 3.864× |
//!
//! Ganho de velocidade marginal, **energia consistentemente pior**, e quatro
//! ordens de grandeza de precisão a menos. O desempacotamento custa ULA por
//! elemento lido, e numa carga que não é limitada por banda essa conta extra
//! gasta potência sem comprar tempo.
//!
//! # Para que serve, então
//!
//! **Capacidade de memória, não velocidade.** Os operandos ocupam metade do
//! espaço, o que permite modelos maiores ou lotes maiores quando a VRAM é o
//! limite. É uma troca de precisão por tamanho, e nessa troca ela é honesta.
//!
//! # Por que a acumulação fica em f32
//!
//! Somar milhares de termos em `f16`
//! perderia dígitos rápido: são 11 bits de mantissa contra 24. Os ladrilhos em
//! memória compartilhada também ficam em `f32` — eles já estão dentro do chip, e
//! o que se quer reduzir é a ida à DRAM. É o mesmo arranjo que os tensor cores
//! fazem em hardware.
//!
//! # O empacotamento
//!
//! Dois `f16` por palavra de 32 bits, via `pack2x16float`. A leitura endereça o
//! **elemento**, não a palavra, e escolhe a metade — de modo que não há
//! restrição de alinhamento nas dimensões, ao contrário do que aconteceu com as
//! leituras `vec4`. Threads vizinhas leem elementos vizinhos, caem na mesma
//! palavra e usam metades diferentes: a leitura continua coalescida e a cache
//! absorve a duplicata.
//!
//! # O custo
//!
//! `f16` tem ~3 dígitos decimais significativos e alcance até 65504. Valores
//! fora desse alcance saturam para infinito, e a conversão perde precisão em
//! todos. `tests/gpu.rs` mede quanto, em vez de supor.

pub const MM_F16: &str = r#"
struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;
const LD: u32 = 65u;
const BUF: u32 = 1040u;

// A memória compartilhada e os acumuladores permanecem em f32: só o tráfego
// global é reduzido. Somar 4096 termos em meia precisão perderia dígitos rápido
// — f16 tem 11 bits de mantissa contra 24 do f32.
var<workgroup> sa: array<f32, 2080>;
var<workgroup> sb: array<f32, 2080>;

// Dois f16 por u32. O índice é o do elemento, não o da palavra: threads
// vizinhas leem elementos vizinhos, caem na mesma palavra e usam metades
// diferentes — a leitura continua coalescida e a cache absorve a duplicata.
fn a16(idx: u32) -> f32 {
    let par = unpack2x16float(a[idx >> 1u]);
    if ((idx & 1u) == 0u) { return par.x; }
    return par.y;
}

fn b16(idx: u32) -> f32 {
    let par = unpack2x16float(b[idx >> 1u]);
    if ((idx & 1u) == 0u) { return par.x; }
    return par.y;
}

fn le_a_mk(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gr = lin0 + idx / TK;
        let gk = t * TK + idx % TK;
        r[i] = select(0.0, a16(gr * d.k + gk), gr < d.m && gk < d.k);
    }
    return r;
}

fn le_a_km(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gk = t * TK + idx / TM;
        let gr = lin0 + idx % TM;
        r[i] = select(0.0, a16(gk * d.m + gr), gr < d.m && gk < d.k);
    }
    return r;
}

fn le_b_kn(tid: u32, col0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gk = t * TK + idx / TN;
        let gc = col0 + idx % TN;
        r[i] = select(0.0, b16(gk * d.n + gc), gk < d.k && gc < d.n);
    }
    return r;
}

fn le_b_nk(tid: u32, col0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gc = col0 + idx / TK;
        let gk = t * TK + idx % TK;
        r[i] = select(0.0, b16(gc * d.k + gk), gk < d.k && gc < d.n);
    }
    return r;
}

fn guarda_a_mk(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sa[buf * BUF + (idx % TK) * LD + idx / TK] = r[i];
    }
}

fn guarda_a_km(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sa[buf * BUF + (idx / TM) * LD + idx % TM] = r[i];
    }
}

fn guarda_b_kn(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sb[buf * BUF + (idx / TN) * LD + idx % TN] = r[i];
    }
}

fn guarda_b_nk(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sb[buf * BUF + (idx % TK) * LD + idx / TK] = r[i];
    }
}

fn acumula(buf: u32, ty: u32, tx: u32, acc: ptr<function, array<vec4<f32>, 4>>) {
    for (var kk = 0u; kk < TK; kk = kk + 1u) {
        let ab = buf * BUF + kk * LD + ty * 4u;
        let bb = buf * BUF + kk * LD + tx * 4u;
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

fn bloco(w: vec3<u32>) -> vec2<u32> {
    let pid = w.y * d.grid_x + w.x;
    let nm = (d.m + TM - 1u) / TM;
    let nn = (d.n + TN - 1u) / TN;
    let grupo = max(d.grupo, 1u);
    let por_grupo = grupo * nn;
    let gid = pid / por_grupo;
    let m0 = gid * grupo;
    let tam = max(min(nm - m0, grupo), 1u);
    return vec2<u32>(m0 + (pid % tam), (pid % por_grupo) / tam);
}

fn fora(w: vec3<u32>) -> bool {
    let pid = w.y * d.grid_x + w.x;
    let nm = (d.m + TM - 1u) / TM;
    let nn = (d.n + TN - 1u) / TN;
    return pid >= nm * nn;
}

@compute @workgroup_size(16, 16)
fn mm(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * TM;
    let col0 = b_id.y * TN;
    var acc = array<vec4<f32>, 4>();
    let n = passos();

    guarda_a_mk(tid, 0u, le_a_mk(tid, lin0, 0u));
    guarda_b_kn(tid, 0u, le_b_kn(tid, col0, 0u));
    workgroupBarrier();

    var cur = 0u;
    for (var t = 0u; t < n; t = t + 1u) {
        var ra: vec4<f32>;
        var rb: vec4<f32>;
        let tem_proximo = t + 1u < n;
        if (tem_proximo) {
            ra = le_a_mk(tid, lin0, t + 1u);
            rb = le_b_kn(tid, col0, t + 1u);
        }
        acumula(cur, l.y, l.x, &acc);
        if (tem_proximo) {
            guarda_a_mk(tid, 1u - cur, ra);
            guarda_b_kn(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }
    escreve(lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}

@compute @workgroup_size(16, 16)
fn mm_atb(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * TM;
    let col0 = b_id.y * TN;
    var acc = array<vec4<f32>, 4>();
    let n = passos();

    guarda_a_km(tid, 0u, le_a_km(tid, lin0, 0u));
    guarda_b_kn(tid, 0u, le_b_kn(tid, col0, 0u));
    workgroupBarrier();

    var cur = 0u;
    for (var t = 0u; t < n; t = t + 1u) {
        var ra: vec4<f32>;
        var rb: vec4<f32>;
        let tem_proximo = t + 1u < n;
        if (tem_proximo) {
            ra = le_a_km(tid, lin0, t + 1u);
            rb = le_b_kn(tid, col0, t + 1u);
        }
        acumula(cur, l.y, l.x, &acc);
        if (tem_proximo) {
            guarda_a_km(tid, 1u - cur, ra);
            guarda_b_kn(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }
    escreve(lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}

@compute @workgroup_size(16, 16)
fn mm_abt(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * TM;
    let col0 = b_id.y * TN;
    var acc = array<vec4<f32>, 4>();
    let n = passos();

    guarda_a_mk(tid, 0u, le_a_mk(tid, lin0, 0u));
    guarda_b_nk(tid, 0u, le_b_nk(tid, col0, 0u));
    workgroupBarrier();

    var cur = 0u;
    for (var t = 0u; t < n; t = t + 1u) {
        var ra: vec4<f32>;
        var rb: vec4<f32>;
        let tem_proximo = t + 1u < n;
        if (tem_proximo) {
            ra = le_a_mk(tid, lin0, t + 1u);
            rb = le_b_nk(tid, col0, t + 1u);
        }
        acumula(cur, l.y, l.x, &acc);
        if (tem_proximo) {
            guarda_a_mk(tid, 1u - cur, ra);
            guarda_b_nk(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }
    escreve(lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}
"#;

/// Converte um buffer `f32` em pares de meia precisão empacotados.
pub const EMPACOTA: &str = r#"
struct P { n: u32, gx: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn empacota(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i = (w.y * p.gx + w.x) * 256u + l.x;
    let palavras = (p.n + 1u) / 2u;
    if (i >= palavras) { return; }
    let x = src[2u * i];
    let y = select(0.0, src[2u * i + 1u], 2u * i + 1u < p.n);
    dst[i] = pack2x16float(vec2<f32>(x, y));
}
"#;
