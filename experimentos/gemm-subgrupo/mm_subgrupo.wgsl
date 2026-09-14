// GEMM com `A` via subgroupShuffle — IMPLEMENTADO, MEDIDO E REJEITADO.
//
// A ideia: dentro de um subgrupo de 32 lanes, o endereço de `av` no laço
// interno do GEMM depende só de `ty` (2 valores distintos por subgrupo);
// cruzado com os 16 passos de K do ladrilho, dá exatamente 32 combinações —
// o número de lanes do subgrupo. Cada lane passa a "possuir" um par
// (b = lane/16, kk_própria = lane%16), lê `av` da memória compartilhada uma
// única vez (contra 16 antes) e usa `subgroupShuffle` para buscar o valor das
// outras 15 iterações de K a partir de outra lane do seu meio-subgrupo, sem
// tocar a memória compartilhada de novo. `B` não tem esse encaixe (16
// endereços distintos × 16 passos = 256 combinações para 32 lanes, 8
// registradores por lane) e continua como estava.
//
// `SUBGROUP` é feature estável do wgpu — não é `EXPERIMENTAL_*`, não exige
// `unsafe` e não exige operandos `f16` (ao contrário da matriz cooperativa,
// ver a seção "Tensor cores" em RESULTADOS-NEGATIVOS.md).
//
// Requer workgroup 1-D: `@builtin(subgroup_invocation_id)` não é aceito pelo
// naga 30.0.1 com workgroup multidimensional — erro medido: "Workgroup size
// is multi dimensional, `@builtin(subgroup_id)` and
// `@builtin(subgroup_invocation_id)` are not supported." Daí
// `local_invocation_index` no lugar de `local_invocation_id`.
//
// # O que foi medido
//
// A sonda isolada (`crates/rtensor/examples/sonda_subgrupo.rs`, mantida no
// repositório) reproduz só o padrão de leitura de `av` — mesma aritmética,
// mesma contagem de acessos — comparando a via da memória compartilhada com a
// via `subgroupShuffle`: **+3% a +9% (média ~6%, 8 execuções, sempre
// positivo)**. É um resultado real e reprodutível.
//
// Mas plugado neste kernel completo (ladrilhamento em dois níveis, buffer
// duplo, rasterização L2, tudo em produção) e medido no mesmo processo contra
// `MM_FAST` em 4096³, 8 execuções, grupo_l2=8:
//
//   +2,5%  +2,0%  +1,3%  -1,6%  -1,7%  -2,3%  -2,7%  -3,4%   (média: -0,7%)
//
// Sinal alterna, cruza o zero, sem tendência: **dentro do ruído**. A mesma
// leitura já observada no laço interno do GEMM desde `banda_compartilhada.rs`
// (a leitura de `av` já era quase de graça, endereço em broadcast) explica por
// que o ganho isolado não sobrevive: no kernel completo, a economia de
// instruções de carga é compensada por outra pressão — registradores extras
// por lane (`av_reg`, `base`, `kk_propria`) competindo com o acumulador
// 4×4 e o buffer duplo pela mesma cota de registradores/ocupação, o mesmo
// padrão de "diagnóstico certo, peso superestimado" que already apareceu no
// cache de descritores, na redução de coluna, no padding de bancos, no bloco
// 8×8 e nas leituras vec4.
//
// Kernel preservado aqui — correto (passou pelos 12 testes de `tests/gpu.rs`
// antes de ser revertido) e mensurável — para quem quiser reproduzir ou
// discordar da rejeição.

struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

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

fn le_b_nk(tid: u32, col0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    for (var i = 0u; i < 4u; i = i + 1u) {
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

// --- núcleo com `av` via subgroupShuffle --------------------------------
//
// `kk_propria` é o passo de K que esta lane carrega da memória compartilhada;
// `base` é o início do seu meio-subgrupo (0 ou 16). Uma leitura, não 16.
fn acumula(buf: u32, ty: u32, tx: u32, lane: u32, acc: ptr<function, array<vec4<f32>, 4>>) {
    let kk_propria = lane % 16u;
    let base = lane - kk_propria;
    let av0 = buf * BUF + kk_propria * LD + ty * 4u;
    let av_reg = vec4<f32>(sa[av0], sa[av0 + 1u], sa[av0 + 2u], sa[av0 + 3u]);

    for (var kk = 0u; kk < TK; kk = kk + 1u) {
        let av = subgroupShuffle(av_reg, base + kk);
        let bb = buf * BUF + kk * LD + tx * 4u;
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

@compute @workgroup_size(256)
fn mm(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) w: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    if (fora(w)) { return; }
    let ty = tid / 16u;
    let tx = tid % 16u;
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
        acumula(cur, ty, tx, lane, &acc);
        if (tem_proximo) {
            guarda_a_mk(tid, 1u - cur, ra);
            guarda_b_kn(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }
    escreve(lin0 + ty * 4u, col0 + tx * 4u, acc);
}

@compute @workgroup_size(256)
fn mm_atb(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) w: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    if (fora(w)) { return; }
    let ty = tid / 16u;
    let tx = tid % 16u;
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
        acumula(cur, ty, tx, lane, &acc);
        if (tem_proximo) {
            guarda_a_km(tid, 1u - cur, ra);
            guarda_b_kn(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }
    escreve(lin0 + ty * 4u, col0 + tx * 4u, acc);
}

@compute @workgroup_size(256)
fn mm_abt(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) w: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    if (fora(w)) { return; }
    let ty = tid / 16u;
    let tx = tid % 16u;
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
        acumula(cur, ty, tx, lane, &acc);
        if (tem_proximo) {
            guarda_a_mk(tid, 1u - cur, ra);
            guarda_b_nk(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }
    escreve(lin0 + ty * 4u, col0 + tx * 4u, acc);
}
