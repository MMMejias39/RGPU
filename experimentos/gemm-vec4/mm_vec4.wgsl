// GEMM com leituras globais de 128 bits — IMPLEMENTADO, MEDIDO E REJEITADO.
//
// Bindings de A e B como array<vec4<f32>>: cada thread faz uma leitura de 128
// bits em vez de quatro de 32. Exige que a dimensão percorrida seja múltipla
// de 4, e por isso convivia com o kernel escalar, escolhido pelo despacho
// conforme a forma.
//
// Passou nos 42 casos de conferência contra a CPU. E não rendeu nada:
//
//     2048³         escalar: 4.230 GFLOP/s    vec4: 4.172 GFLOP/s
//     4096³         escalar: 4.030            vec4: 4.044
//     4096x4096x64  escalar: 3.541            vec4: 3.306
//
// O último formato é o teste decisivo: com K=64 e M=N=4096 há pouquíssimo
// cálculo por byte lido, que é justamente onde uma leitura mais larga deveria
// aparecer. Não apareceu.
//
// Revertido por não pagar a complexidade: três pipelines a mais, um módulo
// WGSL a mais e lógica de alinhamento no despacho, em troca de zero.


struct Dims { m: u32, n: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;
const LD: u32 = 65u;
const BUF: u32 = 1040u;

var<workgroup> sa: array<f32, 2080>;
var<workgroup> sb: array<f32, 2080>;

// --- leituras globais de 128 bits ---------------------------------------
//
// Cada thread traz um vec4 — quatro floats contíguos numa só instrução, em vez
// de quatro instruções de 32 bits. Os 1024 elementos do ladrilho saem em 256
// leituras, uma por thread.
//
// O vec4 está inteiro dentro ou inteiro fora dos limites, nunca partido, porque
// a dimensão percorrida é múltipla de 4 — é o que o despacho verifica antes de
// escolher este kernel.

// A guardada como [M, K]: 4 valores de K consecutivos.
fn le_a_mk(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    let linha = lin0 + tid / 4u;
    let k0 = t * TK + (tid % 4u) * 4u;
    if (linha >= d.m || k0 >= d.k) { return vec4<f32>(0.0); }
    return a[(linha * d.k + k0) / 4u];
}

// A guardada como [K, M]: 4 valores de M consecutivos.
fn le_a_km(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    let gk = t * TK + tid / 16u;
    let m0 = lin0 + (tid % 16u) * 4u;
    if (gk >= d.k || m0 >= d.m) { return vec4<f32>(0.0); }
    return a[(gk * d.m + m0) / 4u];
}

// B guardada como [K, N]: 4 valores de N consecutivos.
fn le_b_kn(tid: u32, col0: u32, t: u32) -> vec4<f32> {
    let gk = t * TK + tid / 16u;
    let n0 = col0 + (tid % 16u) * 4u;
    if (gk >= d.k || n0 >= d.n) { return vec4<f32>(0.0); }
    return b[(gk * d.n + n0) / 4u];
}

// B guardada como [N, K]: 4 valores de K consecutivos.
fn le_b_nk(tid: u32, col0: u32, t: u32) -> vec4<f32> {
    let col = col0 + tid / 4u;
    let k0 = t * TK + (tid % 4u) * 4u;
    if (col >= d.n || k0 >= d.k) { return vec4<f32>(0.0); }
    return b[(col * d.k + k0) / 4u];
}

// --- escrita no buffer compartilhado, sempre em [K][M] e [K][N] ---------

fn guarda_a_mk(tid: u32, buf: u32, r: vec4<f32>) {
    let m = tid / 4u;
    let k0 = (tid % 4u) * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        sa[buf * BUF + (k0 + j) * LD + m] = r[j];
    }
}

fn guarda_a_km(tid: u32, buf: u32, r: vec4<f32>) {
    let k = tid / 16u;
    let m0 = (tid % 16u) * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        sa[buf * BUF + k * LD + m0 + j] = r[j];
    }
}

fn guarda_b_kn(tid: u32, buf: u32, r: vec4<f32>) {
    let k = tid / 16u;
    let n0 = (tid % 16u) * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        sb[buf * BUF + k * LD + n0 + j] = r[j];
    }
}

fn guarda_b_nk(tid: u32, buf: u32, r: vec4<f32>) {
    let n = tid / 4u;
    let k0 = (tid % 4u) * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        sb[buf * BUF + (k0 + j) * LD + n] = r[j];
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

@compute @workgroup_size(16, 16)
fn mm(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let tid = l.y * 16u + l.x;
    let lin0 = w.y * TM;
    let col0 = w.x * TN;
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
    let tid = l.y * 16u + l.x;
    let lin0 = w.y * TM;
    let col0 = w.x * TN;
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
    let tid = l.y * 16u + l.x;
    let lin0 = w.y * TM;
    let col0 = w.x * TN;
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
