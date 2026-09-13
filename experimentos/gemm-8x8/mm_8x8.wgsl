// GEMM com bloco 8x8 por thread — IMPLEMENTADO, MEDIDO E REJEITADO.
//
// Ladrilho 128x128 por workgroup, passo K de 8, 16 acumuladores vec4 por
// thread, todos com índice constante. Quatro FMAs por leitura de memória
// compartilhada — a mesma intensidade aritmética que o cuBLAS usa.
//
// Passou nos 42 casos de conferência contra a CPU (7 formatos x 3 variantes
// x 2 kernels). E mediu PIOR que o bloco 4x4 que está em produção:
//
//     4096³    4x4: 4.030 GFLOP/s    8x8: 3.846 GFLOP/s    -4,6%
//     2048³    4x4: 4.128 GFLOP/s    8x8: 3.976 GFLOP/s    -3,7%
//
// A razão está em examples/ocupacao.rs: aritmética pura sustenta 15 a 18
// TFLOP/s nesta placa, e o GEMM anda a 4. Estando 4x longe do limite das
// ULAs, o gargalo não é intensidade aritmética — dobrá-la só custa ocupação.
//
// Guardado aqui para que a tentativa não precise ser refeita, e para que
// quem discordar da conclusão possa medir por conta própria.

struct Dims { m: u32, n: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TM: u32 = 128u;
const TN: u32 = 128u;
const TK: u32 = 8u;
const THREADS: u32 = 256u;
const LDA: u32 = 129u;
const LDB: u32 = 129u;
const BUF: u32 = 1032u;

var<workgroup> sa: array<f32, 2064>;
var<workgroup> sb: array<f32, 2064>;

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
        sa[buf * BUF + (idx % TK) * LDA + idx / TK] = r[i];
    }
}

fn guarda_b_kn(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sb[buf * BUF + (idx / TN) * LDB + idx % TN] = r[i];
    }
}

fn guarda(linha: u32, col: u32, v: f32) {
    if (linha < d.m && col < d.n) { c[linha * d.n + col] = v; }
}

fn passos() -> u32 { return (d.k + TK - 1u) / TK; }

fn acumula(buf: u32, ty: u32, tx: u32, acc: ptr<function, array<vec4<f32>, 16>>) {
    for (var kk = 0u; kk < TK; kk = kk + 1u) {
        let ab = buf * BUF + kk * LDA + ty * 8u;
        let bb = buf * BUF + kk * LDB + tx * 8u;
        let a0 = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let a1 = vec4<f32>(sa[ab + 4u], sa[ab + 5u], sa[ab + 6u], sa[ab + 7u]);
        let b0 = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        let b1 = vec4<f32>(sb[bb + 4u], sb[bb + 5u], sb[bb + 6u], sb[bb + 7u]);
        acc[0] = acc[0] + a0.x * b0;
        acc[1] = acc[1] + a0.x * b1;
        acc[2] = acc[2] + a0.y * b0;
        acc[3] = acc[3] + a0.y * b1;
        acc[4] = acc[4] + a0.z * b0;
        acc[5] = acc[5] + a0.z * b1;
        acc[6] = acc[6] + a0.w * b0;
        acc[7] = acc[7] + a0.w * b1;
        acc[8] = acc[8] + a1.x * b0;
        acc[9] = acc[9] + a1.x * b1;
        acc[10] = acc[10] + a1.y * b0;
        acc[11] = acc[11] + a1.y * b1;
        acc[12] = acc[12] + a1.z * b0;
        acc[13] = acc[13] + a1.z * b1;
        acc[14] = acc[14] + a1.w * b0;
        acc[15] = acc[15] + a1.w * b1;
    }
}

fn escreve(lin0: u32, col0: u32, acc: array<vec4<f32>, 16>) {
    guarda(lin0 + 0u, col0 + 0u, acc[0].x);
    guarda(lin0 + 0u, col0 + 1u, acc[0].y);
    guarda(lin0 + 0u, col0 + 2u, acc[0].z);
    guarda(lin0 + 0u, col0 + 3u, acc[0].w);
    guarda(lin0 + 0u, col0 + 4u, acc[1].x);
    guarda(lin0 + 0u, col0 + 5u, acc[1].y);
    guarda(lin0 + 0u, col0 + 6u, acc[1].z);
    guarda(lin0 + 0u, col0 + 7u, acc[1].w);
    guarda(lin0 + 1u, col0 + 0u, acc[2].x);
    guarda(lin0 + 1u, col0 + 1u, acc[2].y);
    guarda(lin0 + 1u, col0 + 2u, acc[2].z);
    guarda(lin0 + 1u, col0 + 3u, acc[2].w);
    guarda(lin0 + 1u, col0 + 4u, acc[3].x);
    guarda(lin0 + 1u, col0 + 5u, acc[3].y);
    guarda(lin0 + 1u, col0 + 6u, acc[3].z);
    guarda(lin0 + 1u, col0 + 7u, acc[3].w);
    guarda(lin0 + 2u, col0 + 0u, acc[4].x);
    guarda(lin0 + 2u, col0 + 1u, acc[4].y);
    guarda(lin0 + 2u, col0 + 2u, acc[4].z);
    guarda(lin0 + 2u, col0 + 3u, acc[4].w);
    guarda(lin0 + 2u, col0 + 4u, acc[5].x);
    guarda(lin0 + 2u, col0 + 5u, acc[5].y);
    guarda(lin0 + 2u, col0 + 6u, acc[5].z);
    guarda(lin0 + 2u, col0 + 7u, acc[5].w);
    guarda(lin0 + 3u, col0 + 0u, acc[6].x);
    guarda(lin0 + 3u, col0 + 1u, acc[6].y);
    guarda(lin0 + 3u, col0 + 2u, acc[6].z);
    guarda(lin0 + 3u, col0 + 3u, acc[6].w);
    guarda(lin0 + 3u, col0 + 4u, acc[7].x);
    guarda(lin0 + 3u, col0 + 5u, acc[7].y);
    guarda(lin0 + 3u, col0 + 6u, acc[7].z);
    guarda(lin0 + 3u, col0 + 7u, acc[7].w);
    guarda(lin0 + 4u, col0 + 0u, acc[8].x);
    guarda(lin0 + 4u, col0 + 1u, acc[8].y);
    guarda(lin0 + 4u, col0 + 2u, acc[8].z);
    guarda(lin0 + 4u, col0 + 3u, acc[8].w);
    guarda(lin0 + 4u, col0 + 4u, acc[9].x);
    guarda(lin0 + 4u, col0 + 5u, acc[9].y);
    guarda(lin0 + 4u, col0 + 6u, acc[9].z);
    guarda(lin0 + 4u, col0 + 7u, acc[9].w);
    guarda(lin0 + 5u, col0 + 0u, acc[10].x);
    guarda(lin0 + 5u, col0 + 1u, acc[10].y);
    guarda(lin0 + 5u, col0 + 2u, acc[10].z);
    guarda(lin0 + 5u, col0 + 3u, acc[10].w);
    guarda(lin0 + 5u, col0 + 4u, acc[11].x);
    guarda(lin0 + 5u, col0 + 5u, acc[11].y);
    guarda(lin0 + 5u, col0 + 6u, acc[11].z);
    guarda(lin0 + 5u, col0 + 7u, acc[11].w);
    guarda(lin0 + 6u, col0 + 0u, acc[12].x);
    guarda(lin0 + 6u, col0 + 1u, acc[12].y);
    guarda(lin0 + 6u, col0 + 2u, acc[12].z);
    guarda(lin0 + 6u, col0 + 3u, acc[12].w);
    guarda(lin0 + 6u, col0 + 4u, acc[13].x);
    guarda(lin0 + 6u, col0 + 5u, acc[13].y);
    guarda(lin0 + 6u, col0 + 6u, acc[13].z);
    guarda(lin0 + 6u, col0 + 7u, acc[13].w);
    guarda(lin0 + 7u, col0 + 0u, acc[14].x);
    guarda(lin0 + 7u, col0 + 1u, acc[14].y);
    guarda(lin0 + 7u, col0 + 2u, acc[14].z);
    guarda(lin0 + 7u, col0 + 3u, acc[14].w);
    guarda(lin0 + 7u, col0 + 4u, acc[15].x);
    guarda(lin0 + 7u, col0 + 5u, acc[15].y);
    guarda(lin0 + 7u, col0 + 6u, acc[15].z);
    guarda(lin0 + 7u, col0 + 7u, acc[15].w);
}

@compute @workgroup_size(16, 16)
fn mm(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let lin0 = w.y * TM;
    let col0 = w.x * TN;
    var acc = array<vec4<f32>, 16>();
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
    escreve(lin0 + l.y * 8u, col0 + l.x * 8u, acc);
}
