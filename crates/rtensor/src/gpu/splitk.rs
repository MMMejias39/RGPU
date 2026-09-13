//! GEMM com partição da dimensão interna, e a redução que o acompanha.

/// GEMM particionado na dimensão interna.
///
/// # O problema
///
/// Com `M` e `N` pequenos, a saída inteira cabe em poucos blocos: um GEMM
/// `64×64×8192` ocupa **um** workgroup, 256 threads, numa placa de 36 SMs. O
/// trabalho existe — são 8192 passos de acumulação —, mas não há paralelismo
/// para distribuí-lo. Medido: **55 GFLOP/s**, 1,3% da capacidade.
///
/// Não é um caso raro. `∂L/∂W = Xᵀδ` tem `K` igual ao tamanho do lote, e `M`,
/// `N` iguais às dimensões da camada: lote grande com camada estreita cai
/// exatamente aqui.
///
/// # A partição
///
/// A faixa de `K` é dividida em `fatias`, cada uma somando a sua parte num
/// plano próprio de `parciais`. Um segundo passe soma os planos. O paralelismo
/// passa de `blocos` para `blocos × fatias`.
///
/// Cada fatia escreve num plano separado — não há escrita concorrente no mesmo
/// endereço, e portanto nenhuma necessidade de atômicos, que em WGSL não
/// existem para `f32`.
///
/// # O custo
///
/// `fatias × M × N` floats de memória temporária, mais um passe de redução. Com
/// `M` e `N` pequenos isso é irrisório; é justamente por isso que a técnica se
/// aplica onde se aplica.
pub const MM_SPLITK: &str = r#"
struct Dims {
    m: u32, n: u32, k: u32, grid_x: u32,
    fatias: u32, tiles_fatia: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> parciais: array<f32>;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;
const LD: u32 = 65u;
const BUF: u32 = 1040u;

var<workgroup> sa: array<f32, 2080>;
var<workgroup> sb: array<f32, 2080>;

fn le_a_mk(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gr = lin0 + idx / TK;
        let gk = t * TK + idx % TK;
        r[i] = select(0.0, a[gr * d.k + gk], gr < d.m && gk < d.k);
    }
    return r;
}

fn le_a_km(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gk = t * TK + idx / TM;
        let gr = lin0 + idx % TM;
        r[i] = select(0.0, a[gk * d.m + gr], gr < d.m && gk < d.k);
    }
    return r;
}

fn le_b_kn(tid: u32, col0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gk = t * TK + idx / TN;
        let gc = col0 + idx % TN;
        r[i] = select(0.0, b[gk * d.n + gc], gk < d.k && gc < d.n);
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

// Cada fatia escreve num plano próprio de `parciais`, sem disputa entre elas.
fn guarda(fatia: u32, linha: u32, col: u32, v: f32) {
    if (linha < d.m && col < d.n) {
        parciais[fatia * d.m * d.n + linha * d.n + col] = v;
    }
}

fn escreve(fatia: u32, linha0: u32, col0: u32, acc: array<vec4<f32>, 4>) {
    guarda(fatia, linha0, col0, acc[0].x);
    guarda(fatia, linha0, col0 + 1u, acc[0].y);
    guarda(fatia, linha0, col0 + 2u, acc[0].z);
    guarda(fatia, linha0, col0 + 3u, acc[0].w);
    guarda(fatia, linha0 + 1u, col0, acc[1].x);
    guarda(fatia, linha0 + 1u, col0 + 1u, acc[1].y);
    guarda(fatia, linha0 + 1u, col0 + 2u, acc[1].z);
    guarda(fatia, linha0 + 1u, col0 + 3u, acc[1].w);
    guarda(fatia, linha0 + 2u, col0, acc[2].x);
    guarda(fatia, linha0 + 2u, col0 + 1u, acc[2].y);
    guarda(fatia, linha0 + 2u, col0 + 2u, acc[2].z);
    guarda(fatia, linha0 + 2u, col0 + 3u, acc[2].w);
    guarda(fatia, linha0 + 3u, col0, acc[3].x);
    guarda(fatia, linha0 + 3u, col0 + 1u, acc[3].y);
    guarda(fatia, linha0 + 3u, col0 + 2u, acc[3].z);
    guarda(fatia, linha0 + 3u, col0 + 3u, acc[3].w);
}

@compute @workgroup_size(16, 16)
fn mm(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let pid = w.y * d.grid_x + w.x;
    let nm = (d.m + TM - 1u) / TM;
    let nn = (d.n + TN - 1u) / TN;
    let blocos = nm * nn;
    if (pid >= blocos * d.fatias) { return; }

    let fatia = pid / blocos;
    let local = pid % blocos;
    let lin0 = (local / nn) * TM;
    let col0 = (local % nn) * TN;

    // Esta fatia percorre só a sua faixa de K.
    let t0 = fatia * d.tiles_fatia;
    let passos_k = (d.k + TK - 1u) / TK;
    let t1 = min(t0 + d.tiles_fatia, passos_k);
    if (t0 >= t1) { return; }

    let tid = l.y * 16u + l.x;
    var acc = array<vec4<f32>, 4>();

    guarda_a_mk(tid, 0u, le_a_mk(tid, lin0, t0));
    guarda_b_kn(tid, 0u, le_b_kn(tid, col0, t0));
    workgroupBarrier();

    var cur = 0u;
    for (var t = t0; t < t1; t = t + 1u) {
        var ra: vec4<f32>;
        var rb: vec4<f32>;
        let tem_proximo = t + 1u < t1;
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
    escreve(fatia, lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}

@compute @workgroup_size(16, 16)
fn mm_atb(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let pid = w.y * d.grid_x + w.x;
    let nm = (d.m + TM - 1u) / TM;
    let nn = (d.n + TN - 1u) / TN;
    let blocos = nm * nn;
    if (pid >= blocos * d.fatias) { return; }

    let fatia = pid / blocos;
    let local = pid % blocos;
    let lin0 = (local / nn) * TM;
    let col0 = (local % nn) * TN;

    // Esta fatia percorre só a sua faixa de K.
    let t0 = fatia * d.tiles_fatia;
    let passos_k = (d.k + TK - 1u) / TK;
    let t1 = min(t0 + d.tiles_fatia, passos_k);
    if (t0 >= t1) { return; }

    let tid = l.y * 16u + l.x;
    var acc = array<vec4<f32>, 4>();

    guarda_a_km(tid, 0u, le_a_km(tid, lin0, t0));
    guarda_b_kn(tid, 0u, le_b_kn(tid, col0, t0));
    workgroupBarrier();

    var cur = 0u;
    for (var t = t0; t < t1; t = t + 1u) {
        var ra: vec4<f32>;
        var rb: vec4<f32>;
        let tem_proximo = t + 1u < t1;
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
    escreve(fatia, lin0 + l.y * 4u, col0 + l.x * 4u, acc);
}
"#;

/// Soma os planos de `parciais` em `C`.
pub const REDUZ: &str = r#"
struct P { n: u32, fatias: u32, gx: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> parciais: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn reduz(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i = (w.y * p.gx + w.x) * 256u + l.x;
    if (i >= p.n) { return; }
    var s = 0.0;
    for (var f = 0u; f < p.fatias; f = f + 1u) {
        s = s + parciais[f * p.n + i];
    }
    c[i] = s;
}
"#;
