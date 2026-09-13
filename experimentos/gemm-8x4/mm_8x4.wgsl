// GEMM com bloco 8x4 — IMPLEMENTADO, MEDIDO E REJEITADO.
//
// Ladrilho retangular 128x64, 8 acumuladores vec4 por thread com índices
// constantes. Lê 8 valores de A e 4 de B — 48 bytes — para 32 FMAs: 1,33 flop
// por byte de memória compartilhada, contra 1,0 do bloco 4x4 em produção.
//
// Passou nas 63 combinações de conferência contra a CPU (7 formatos x 3
// variantes x 3 kernels). E mediu neutro:
//
//   2048³   4x4: 4242 / 4277 / 4202     8x4: 4200 / 4271 / 4241
//   4096³   4x4: 4195 / 4243            8x4: 4202 / 4253
//
// Isso contraria a previsão. A sonda banda_compartilhada.rs tinha indicado que
// o gargalo era o canal de memória compartilhada, o que daria ao 8x4 um teto de
// ~6,7 TFLOP/s contra os ~5 do 4x4. O ganho não apareceu, e não é ocupação: o
// 8x4 usa MENOS memória compartilhada (12 KB contra 16 KB por workgroup).
//
// A leitura mais consistente com todos os dados é que o laço interno satura a
// vazão de instruções do conjunto leitura-compartilhada + FMA, e mudar a
// proporção entre as duas não ajuda porque ambas estão perto do limite. Na
// mesma sonda, remover 7/8 das multiplicações rendeu só 17%, e eliminar os
// conflitos de banco, 28%: nenhum dos dois isoladamente domina.
//
// Se isso estiver certo, o caminho que resta é reduzir as DUAS de uma vez — que
// é o que a instrução de matriz cooperativa faz, cobrindo um ladrilho inteiro
// por instrução. É a única técnica da lista ainda não tentada, e a placa
// oferece: EXPERIMENTAL_COOPERATIVE_MATRIX.


struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// Ladrilho retangular: 16 threads em `y` × 8 linhas = 128; 16 em `x` × 4
// colunas = 64.
const TM: u32 = 128u;
const TN: u32 = 64u;
const TK: u32 = 8u;
const THREADS: u32 = 256u;
const LDA: u32 = 129u;
const LDB: u32 = 65u;
const BUFA: u32 = 1032u;   // TK * LDA
const BUFB: u32 = 520u;    // TK * LDB

var<workgroup> sa: array<f32, 2064>;
var<workgroup> sb: array<f32, 1040>;

// A traz 128×8 = 1024 elementos: 4 por thread.
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

// B traz 8×64 = 512 elementos: 2 por thread.
fn le_b_kn(tid: u32, col0: u32, t: u32) -> vec2<f32> {
    var r: vec2<f32>;
    for (var i = 0u; i < 2u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gk = t * TK + idx / TN;
        let gc = col0 + idx % TN;
        r[i] = select(0.0, b[gk * d.n + gc], gk < d.k && gc < d.n);
    }
    return r;
}

fn le_b_nk(tid: u32, col0: u32, t: u32) -> vec2<f32> {
    var r: vec2<f32>;
    for (var i = 0u; i < 2u; i = i + 1u) {
        let idx = tid + i * THREADS;
        let gc = col0 + idx / TK;
        let gk = t * TK + idx % TK;
        r[i] = select(0.0, b[gc * d.k + gk], gk < d.k && gc < d.n);
    }
    return r;
}

fn guarda_a_mk(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sa[buf * BUFA + (idx % TK) * LDA + idx / TK] = r[i];
    }
}

fn guarda_a_km(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sa[buf * BUFA + (idx / TM) * LDA + idx % TM] = r[i];
    }
}

fn guarda_b_kn(tid: u32, buf: u32, r: vec2<f32>) {
    for (var i = 0u; i < 2u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sb[buf * BUFB + (idx / TN) * LDB + idx % TN] = r[i];
    }
}

fn guarda_b_nk(tid: u32, buf: u32, r: vec2<f32>) {
    for (var i = 0u; i < 2u; i = i + 1u) {
        let idx = tid + i * THREADS;
        sb[buf * BUFB + (idx % TK) * LDB + idx / TK] = r[i];
    }
}

// 8 valores de A e 4 de B — 48 bytes — para 32 FMAs, ou 64 flops.
// Intensidade de 1,33 flop por byte, contra 1,0 do bloco 4×4.
fn acumula(buf: u32, ty: u32, tx: u32, acc: ptr<function, array<vec4<f32>, 8>>) {
    for (var kk = 0u; kk < TK; kk = kk + 1u) {
        let ab = buf * BUFA + kk * LDA + ty * 8u;
        let bb = buf * BUFB + kk * LDB + tx * 4u;
        let a0 = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let a1 = vec4<f32>(sa[ab + 4u], sa[ab + 5u], sa[ab + 6u], sa[ab + 7u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc[0] = acc[0] + a0.x * bv;
        acc[1] = acc[1] + a0.y * bv;
        acc[2] = acc[2] + a0.z * bv;
        acc[3] = acc[3] + a0.w * bv;
        acc[4] = acc[4] + a1.x * bv;
        acc[5] = acc[5] + a1.y * bv;
        acc[6] = acc[6] + a1.z * bv;
        acc[7] = acc[7] + a1.w * bv;
    }
}

fn guarda(linha: u32, col: u32, v: f32) {
    if (linha < d.m && col < d.n) { c[linha * d.n + col] = v; }
}

fn escreve(lin0: u32, col0: u32, acc: array<vec4<f32>, 8>) {
    guarda(lin0 + 0u, col0 + 0u, acc[0].x);
    guarda(lin0 + 0u, col0 + 1u, acc[0].y);
    guarda(lin0 + 0u, col0 + 2u, acc[0].z);
    guarda(lin0 + 0u, col0 + 3u, acc[0].w);
    guarda(lin0 + 1u, col0 + 0u, acc[1].x);
    guarda(lin0 + 1u, col0 + 1u, acc[1].y);
    guarda(lin0 + 1u, col0 + 2u, acc[1].z);
    guarda(lin0 + 1u, col0 + 3u, acc[1].w);
    guarda(lin0 + 2u, col0 + 0u, acc[2].x);
    guarda(lin0 + 2u, col0 + 1u, acc[2].y);
    guarda(lin0 + 2u, col0 + 2u, acc[2].z);
    guarda(lin0 + 2u, col0 + 3u, acc[2].w);
    guarda(lin0 + 3u, col0 + 0u, acc[3].x);
    guarda(lin0 + 3u, col0 + 1u, acc[3].y);
    guarda(lin0 + 3u, col0 + 2u, acc[3].z);
    guarda(lin0 + 3u, col0 + 3u, acc[3].w);
    guarda(lin0 + 4u, col0 + 0u, acc[4].x);
    guarda(lin0 + 4u, col0 + 1u, acc[4].y);
    guarda(lin0 + 4u, col0 + 2u, acc[4].z);
    guarda(lin0 + 4u, col0 + 3u, acc[4].w);
    guarda(lin0 + 5u, col0 + 0u, acc[5].x);
    guarda(lin0 + 5u, col0 + 1u, acc[5].y);
    guarda(lin0 + 5u, col0 + 2u, acc[5].z);
    guarda(lin0 + 5u, col0 + 3u, acc[5].w);
    guarda(lin0 + 6u, col0 + 0u, acc[6].x);
    guarda(lin0 + 6u, col0 + 1u, acc[6].y);
    guarda(lin0 + 6u, col0 + 2u, acc[6].z);
    guarda(lin0 + 6u, col0 + 3u, acc[6].w);
    guarda(lin0 + 7u, col0 + 0u, acc[7].x);
    guarda(lin0 + 7u, col0 + 1u, acc[7].y);
    guarda(lin0 + 7u, col0 + 2u, acc[7].z);
    guarda(lin0 + 7u, col0 + 3u, acc[7].w);
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
    var acc = array<vec4<f32>, 8>();
    let n = passos();

    guarda_a_mk(tid, 0u, le_a_mk(tid, lin0, 0u));
    guarda_b_kn(tid, 0u, le_b_kn(tid, col0, 0u));
    workgroupBarrier();

    var cur = 0u;
    for (var t = 0u; t < n; t = t + 1u) {
        var ra: vec4<f32>;
        var rb: vec2<f32>;
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
    escreve(lin0 + l.y * 8u, col0 + l.x * 4u, acc);
}

@compute @workgroup_size(16, 16)
fn mm_atb(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * TM;
    let col0 = b_id.y * TN;
    var acc = array<vec4<f32>, 8>();
    let n = passos();

    guarda_a_km(tid, 0u, le_a_km(tid, lin0, 0u));
    guarda_b_kn(tid, 0u, le_b_kn(tid, col0, 0u));
    workgroupBarrier();

    var cur = 0u;
    for (var t = 0u; t < n; t = t + 1u) {
        var ra: vec4<f32>;
        var rb: vec2<f32>;
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
    escreve(lin0 + l.y * 8u, col0 + l.x * 4u, acc);
}

@compute @workgroup_size(16, 16)
fn mm_abt(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * TM;
    let col0 = b_id.y * TN;
    var acc = array<vec4<f32>, 8>();
    let n = passos();

    guarda_a_mk(tid, 0u, le_a_mk(tid, lin0, 0u));
    guarda_b_nk(tid, 0u, le_b_nk(tid, col0, 0u));
    workgroupBarrier();

    var cur = 0u;
    for (var t = 0u; t < n; t = t + 1u) {
        var ra: vec4<f32>;
        var rb: vec2<f32>;
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
    escreve(lin0 + l.y * 8u, col0 + l.x * 4u, acc);
}
