// GEMM especializado para formas alinhadas — IMPLEMENTADO, MEDIDO, SEM EFEITO.
//
// Duas especializações, geradas para M e N múltiplos de 64 e K de 16:
//
//   1. laço de K desenrolado, com os 16 deslocamentos literais
//   2. leituras globais sem guarda de limite (o `select` e as duas comparações
//      por elemento, 8 por thread por ladrilho)
//
// A hipótese vinha do diagnóstico: a sonda banda_compartilhada.rs localizou o
// gargalo na vazão de instruções, e estas são instruções de puro overhead.
// Também vinha do que funcionara no rqubit, onde especializar com índices
// literais rendeu 5,32x e 7,65x sobre o kernel genérico.
//
// Não rendeu. Quatro execuções em 4096³, três variantes cada:
//
//   execução   geral    sem guardas+laço   sem guardas desenrolado
//          1  4274,8              4280,6                    4285,7
//          2  4273,8              4293,6                    4275,8
//          3  4220,8              4212,2                    4229,0
//          4  4185,9              4200,9                    4145,9
//
// Dentro de cada execução as três diferem menos de 1%; entre execuções a
// deriva térmica é de 3%. São indistinguíveis.
//
// A explicação provável: `TK` é uma constante, então o compilador já
// desenrolava o laço — a versão manual só aumentou o código-fonte. E as guardas
// são 8 instruções contra 128 leituras compartilhadas e 64 FMAs por ladrilho:
// 4% do trabalho, abaixo do que a medição resolve.
//
// Uma nota de método: a primeira medição deu -3,3% e -4,4%, e eu quase publiquei
// "especialização piora". Era deriva térmica entre as duas metades da mesma
// execução. Só a repetição com as três variantes lado a lado mostrou que não há
// diferença nenhuma.

/// GEMM especializado para formas alinhadas: `M` e `N` múltiplos de 64, `K` de 16.
///
/// # A hipótese
///
/// A sonda `examples/banda_compartilhada.rs` localizou o gargalo na **vazão de
/// instruções** do conjunto leitura-compartilhada + FMA: remover 7/8 das
/// multiplicações rende só 17%, e eliminar os conflitos de banco, 28% — nenhum
/// dos dois domina. Se o limite é emitir instruções, o ataque é **emitir
/// menos**, e há duas fontes de desperdício puro:
///
/// 1. **O laço de `K`.** São 16 iterações com contador, comparação e uma
///    multiplicação por `LD` em cada. Aqui os 16 passos são gerados com
///    deslocamento **literal**, e não resta laço.
/// 2. **As guardas de limite.** Cada leitura global faz duas comparações e um
///    `select` — 8 por thread por ladrilho — que em forma alinhada são sempre
///    verdadeiras. Este kernel só é despachado quando a forma garante isso, e
///    então não precisa delas.
///
/// O mesmo raciocínio que fez os kernels quânticos especializados renderem
/// 5,32× e 7,65× sobre o genérico: trocar generalidade por instruções a menos.
pub const MM_ALINHADO: &str = r#"
struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

var<workgroup> sa: array<f32, 2080>;
var<workgroup> sb: array<f32, 2080>;

fn le_a_mk(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    {
        let idx0 = tid + 0u;
        r[0] = a[(lin0 + idx0 / 16u) * d.k + t * 16u + idx0 % 16u];
        let idx1 = tid + 256u;
        r[1] = a[(lin0 + idx1 / 16u) * d.k + t * 16u + idx1 % 16u];
        let idx2 = tid + 512u;
        r[2] = a[(lin0 + idx2 / 16u) * d.k + t * 16u + idx2 % 16u];
        let idx3 = tid + 768u;
        r[3] = a[(lin0 + idx3 / 16u) * d.k + t * 16u + idx3 % 16u];
    }
    return r;
}
fn le_a_km(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    {
        let idx0 = tid + 0u;
        r[0] = a[(t * 16u + idx0 / 64u) * d.m + lin0 + idx0 % 64u];
        let idx1 = tid + 256u;
        r[1] = a[(t * 16u + idx1 / 64u) * d.m + lin0 + idx1 % 64u];
        let idx2 = tid + 512u;
        r[2] = a[(t * 16u + idx2 / 64u) * d.m + lin0 + idx2 % 64u];
        let idx3 = tid + 768u;
        r[3] = a[(t * 16u + idx3 / 64u) * d.m + lin0 + idx3 % 64u];
    }
    return r;
}
fn le_b_kn(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    {
        let idx0 = tid + 0u;
        r[0] = b[(t * 16u + idx0 / 64u) * d.n + lin0 + idx0 % 64u];
        let idx1 = tid + 256u;
        r[1] = b[(t * 16u + idx1 / 64u) * d.n + lin0 + idx1 % 64u];
        let idx2 = tid + 512u;
        r[2] = b[(t * 16u + idx2 / 64u) * d.n + lin0 + idx2 % 64u];
        let idx3 = tid + 768u;
        r[3] = b[(t * 16u + idx3 / 64u) * d.n + lin0 + idx3 % 64u];
    }
    return r;
}
fn le_b_nk(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    {
        let idx0 = tid + 0u;
        r[0] = b[(lin0 + idx0 / 16u) * d.k + t * 16u + idx0 % 16u];
        let idx1 = tid + 256u;
        r[1] = b[(lin0 + idx1 / 16u) * d.k + t * 16u + idx1 % 16u];
        let idx2 = tid + 512u;
        r[2] = b[(lin0 + idx2 / 16u) * d.k + t * 16u + idx2 % 16u];
        let idx3 = tid + 768u;
        r[3] = b[(lin0 + idx3 / 16u) * d.k + t * 16u + idx3 % 16u];
    }
    return r;
}

fn guarda_a_mk(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * 256u;
        sa[buf * 1040u + (idx % 16u) * 65u + idx / 16u] = r[i];
    }
}
fn guarda_a_km(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * 256u;
        sa[buf * 1040u + (idx / 64u) * 65u + idx % 64u] = r[i];
    }
}
fn guarda_b_kn(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * 256u;
        sb[buf * 1040u + (idx / 64u) * 65u + idx % 64u] = r[i];
    }
}
fn guarda_b_nk(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * 256u;
        sb[buf * 1040u + (idx % 16u) * 65u + idx / 16u] = r[i];
    }
}


fn escreve(lr: u32, lc: u32, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>) {
    let n = d.n;
    c[lr * n + lc] = a0.x;       c[lr * n + lc + 1u] = a0.y;
    c[lr * n + lc + 2u] = a0.z;  c[lr * n + lc + 3u] = a0.w;
    c[(lr + 1u) * n + lc] = a1.x;      c[(lr + 1u) * n + lc + 1u] = a1.y;
    c[(lr + 1u) * n + lc + 2u] = a1.z; c[(lr + 1u) * n + lc + 3u] = a1.w;
    c[(lr + 2u) * n + lc] = a2.x;      c[(lr + 2u) * n + lc + 1u] = a2.y;
    c[(lr + 2u) * n + lc + 2u] = a2.z; c[(lr + 2u) * n + lc + 3u] = a2.w;
    c[(lr + 3u) * n + lc] = a3.x;      c[(lr + 3u) * n + lc + 1u] = a3.y;
    c[(lr + 3u) * n + lc + 2u] = a3.z; c[(lr + 3u) * n + lc + 3u] = a3.w;
}

fn bloco(w: vec3<u32>) -> vec2<u32> {
    let pid = w.y * d.grid_x + w.x;
    let nm = d.m / 64u;
    let nn = d.n / 64u;
    let grupo = max(d.grupo, 1u);
    let por_grupo = grupo * nn;
    let gid = pid / por_grupo;
    let m0 = gid * grupo;
    let tam = max(min(nm - m0, grupo), 1u);
    return vec2<u32>(m0 + (pid % tam), (pid % por_grupo) / tam);
}

fn fora(w: vec3<u32>) -> bool {
    return w.y * d.grid_x + w.x >= (d.m / 64u) * (d.n / 64u);
}

@compute @workgroup_size(16, 16)
fn mm(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * 64u;
    let col0 = b_id.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let n = d.k / 16u;

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
        let base_a = cur * 1040u + l.y * 4u;
        let base_b = cur * 1040u + l.x * 4u;
    {
        let ab = base_a + 0u;
        let bb = base_b + 0u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 65u;
        let bb = base_b + 65u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 130u;
        let bb = base_b + 130u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 195u;
        let bb = base_b + 195u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 260u;
        let bb = base_b + 260u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 325u;
        let bb = base_b + 325u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 390u;
        let bb = base_b + 390u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 455u;
        let bb = base_b + 455u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 520u;
        let bb = base_b + 520u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 585u;
        let bb = base_b + 585u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 650u;
        let bb = base_b + 650u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 715u;
        let bb = base_b + 715u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 780u;
        let bb = base_b + 780u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 845u;
        let bb = base_b + 845u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 910u;
        let bb = base_b + 910u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 975u;
        let bb = base_b + 975u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
        if (tem_proximo) {
            guarda_a_mk(tid, 1u - cur, ra);
            guarda_b_kn(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }

    let lr = lin0 + l.y * 4u;
    let lc = col0 + l.x * 4u;
    escreve(lr, lc, acc0, acc1, acc2, acc3);
}

@compute @workgroup_size(16, 16)
fn mm_atb(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * 64u;
    let col0 = b_id.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let n = d.k / 16u;

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
        let base_a = cur * 1040u + l.y * 4u;
        let base_b = cur * 1040u + l.x * 4u;
    {
        let ab = base_a + 0u;
        let bb = base_b + 0u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 65u;
        let bb = base_b + 65u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 130u;
        let bb = base_b + 130u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 195u;
        let bb = base_b + 195u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 260u;
        let bb = base_b + 260u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 325u;
        let bb = base_b + 325u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 390u;
        let bb = base_b + 390u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 455u;
        let bb = base_b + 455u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 520u;
        let bb = base_b + 520u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 585u;
        let bb = base_b + 585u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 650u;
        let bb = base_b + 650u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 715u;
        let bb = base_b + 715u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 780u;
        let bb = base_b + 780u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 845u;
        let bb = base_b + 845u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 910u;
        let bb = base_b + 910u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 975u;
        let bb = base_b + 975u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
        if (tem_proximo) {
            guarda_a_km(tid, 1u - cur, ra);
            guarda_b_kn(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }

    let lr = lin0 + l.y * 4u;
    let lc = col0 + l.x * 4u;
    escreve(lr, lc, acc0, acc1, acc2, acc3);
}

@compute @workgroup_size(16, 16)
fn mm_abt(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * 64u;
    let col0 = b_id.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let n = d.k / 16u;

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
        let base_a = cur * 1040u + l.y * 4u;
        let base_b = cur * 1040u + l.x * 4u;
    {
        let ab = base_a + 0u;
        let bb = base_b + 0u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 65u;
        let bb = base_b + 65u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 130u;
        let bb = base_b + 130u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 195u;
        let bb = base_b + 195u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 260u;
        let bb = base_b + 260u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 325u;
        let bb = base_b + 325u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 390u;
        let bb = base_b + 390u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 455u;
        let bb = base_b + 455u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 520u;
        let bb = base_b + 520u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 585u;
        let bb = base_b + 585u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 650u;
        let bb = base_b + 650u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 715u;
        let bb = base_b + 715u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 780u;
        let bb = base_b + 780u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 845u;
        let bb = base_b + 845u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 910u;
        let bb = base_b + 910u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
    {
        let ab = base_a + 975u;
        let bb = base_b + 975u;
        let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
        let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
        acc0 = acc0 + av.x * bv;
        acc1 = acc1 + av.y * bv;
        acc2 = acc2 + av.z * bv;
        acc3 = acc3 + av.w * bv;
    }
        if (tem_proximo) {
            guarda_a_mk(tid, 1u - cur, ra);
            guarda_b_nk(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }

    let lr = lin0 + l.y * 4u;
    let lc = col0 + l.x * 4u;
    escreve(lr, lc, acc0, acc1, acc2, acc3);
}
"#;

/// Igual ao [`MM_ALINHADO`], mas com o laço de `K` **preservado**.
///
/// Existe para separar os dois efeitos que o alinhado juntava: remover as
/// guardas de limite e desenrolar o laço. Sem esta variante, um resultado
/// negativo não diria qual das duas mudanças custou.
pub const MM_ALINHADO_LACO: &str = r#"
struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

var<workgroup> sa: array<f32, 2080>;
var<workgroup> sb: array<f32, 2080>;

fn le_a_mk(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    {
        let idx0 = tid + 0u;
        r[0] = a[(lin0 + idx0 / 16u) * d.k + t * 16u + idx0 % 16u];
        let idx1 = tid + 256u;
        r[1] = a[(lin0 + idx1 / 16u) * d.k + t * 16u + idx1 % 16u];
        let idx2 = tid + 512u;
        r[2] = a[(lin0 + idx2 / 16u) * d.k + t * 16u + idx2 % 16u];
        let idx3 = tid + 768u;
        r[3] = a[(lin0 + idx3 / 16u) * d.k + t * 16u + idx3 % 16u];
    }
    return r;
}
fn le_a_km(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    {
        let idx0 = tid + 0u;
        r[0] = a[(t * 16u + idx0 / 64u) * d.m + lin0 + idx0 % 64u];
        let idx1 = tid + 256u;
        r[1] = a[(t * 16u + idx1 / 64u) * d.m + lin0 + idx1 % 64u];
        let idx2 = tid + 512u;
        r[2] = a[(t * 16u + idx2 / 64u) * d.m + lin0 + idx2 % 64u];
        let idx3 = tid + 768u;
        r[3] = a[(t * 16u + idx3 / 64u) * d.m + lin0 + idx3 % 64u];
    }
    return r;
}
fn le_b_kn(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    {
        let idx0 = tid + 0u;
        r[0] = b[(t * 16u + idx0 / 64u) * d.n + lin0 + idx0 % 64u];
        let idx1 = tid + 256u;
        r[1] = b[(t * 16u + idx1 / 64u) * d.n + lin0 + idx1 % 64u];
        let idx2 = tid + 512u;
        r[2] = b[(t * 16u + idx2 / 64u) * d.n + lin0 + idx2 % 64u];
        let idx3 = tid + 768u;
        r[3] = b[(t * 16u + idx3 / 64u) * d.n + lin0 + idx3 % 64u];
    }
    return r;
}
fn le_b_nk(tid: u32, lin0: u32, t: u32) -> vec4<f32> {
    var r: vec4<f32>;
    {
        let idx0 = tid + 0u;
        r[0] = b[(lin0 + idx0 / 16u) * d.k + t * 16u + idx0 % 16u];
        let idx1 = tid + 256u;
        r[1] = b[(lin0 + idx1 / 16u) * d.k + t * 16u + idx1 % 16u];
        let idx2 = tid + 512u;
        r[2] = b[(lin0 + idx2 / 16u) * d.k + t * 16u + idx2 % 16u];
        let idx3 = tid + 768u;
        r[3] = b[(lin0 + idx3 / 16u) * d.k + t * 16u + idx3 % 16u];
    }
    return r;
}

fn guarda_a_mk(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * 256u;
        sa[buf * 1040u + (idx % 16u) * 65u + idx / 16u] = r[i];
    }
}
fn guarda_a_km(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * 256u;
        sa[buf * 1040u + (idx / 64u) * 65u + idx % 64u] = r[i];
    }
}
fn guarda_b_kn(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * 256u;
        sb[buf * 1040u + (idx / 64u) * 65u + idx % 64u] = r[i];
    }
}
fn guarda_b_nk(tid: u32, buf: u32, r: vec4<f32>) {
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = tid + i * 256u;
        sb[buf * 1040u + (idx % 16u) * 65u + idx / 16u] = r[i];
    }
}


fn escreve(lr: u32, lc: u32, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>) {
    let n = d.n;
    c[lr * n + lc] = a0.x;       c[lr * n + lc + 1u] = a0.y;
    c[lr * n + lc + 2u] = a0.z;  c[lr * n + lc + 3u] = a0.w;
    c[(lr + 1u) * n + lc] = a1.x;      c[(lr + 1u) * n + lc + 1u] = a1.y;
    c[(lr + 1u) * n + lc + 2u] = a1.z; c[(lr + 1u) * n + lc + 3u] = a1.w;
    c[(lr + 2u) * n + lc] = a2.x;      c[(lr + 2u) * n + lc + 1u] = a2.y;
    c[(lr + 2u) * n + lc + 2u] = a2.z; c[(lr + 2u) * n + lc + 3u] = a2.w;
    c[(lr + 3u) * n + lc] = a3.x;      c[(lr + 3u) * n + lc + 1u] = a3.y;
    c[(lr + 3u) * n + lc + 2u] = a3.z; c[(lr + 3u) * n + lc + 3u] = a3.w;
}

fn bloco(w: vec3<u32>) -> vec2<u32> {
    let pid = w.y * d.grid_x + w.x;
    let nm = d.m / 64u;
    let nn = d.n / 64u;
    let grupo = max(d.grupo, 1u);
    let por_grupo = grupo * nn;
    let gid = pid / por_grupo;
    let m0 = gid * grupo;
    let tam = max(min(nm - m0, grupo), 1u);
    return vec2<u32>(m0 + (pid % tam), (pid % por_grupo) / tam);
}

fn fora(w: vec3<u32>) -> bool {
    return w.y * d.grid_x + w.x >= (d.m / 64u) * (d.n / 64u);
}

@compute @workgroup_size(16, 16)
fn mm(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * 64u;
    let col0 = b_id.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let n = d.k / 16u;

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
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let ab = cur * 1040u + kk * 65u + l.y * 4u;
            let bb = cur * 1040u + kk * 65u + l.x * 4u;
            let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
            let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
            acc0 = acc0 + av.x * bv;
            acc1 = acc1 + av.y * bv;
            acc2 = acc2 + av.z * bv;
            acc3 = acc3 + av.w * bv;
        }
        if (tem_proximo) {
            guarda_a_mk(tid, 1u - cur, ra);
            guarda_b_kn(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }

    let lr = lin0 + l.y * 4u;
    let lc = col0 + l.x * 4u;
    escreve(lr, lc, acc0, acc1, acc2, acc3);
}

@compute @workgroup_size(16, 16)
fn mm_atb(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * 64u;
    let col0 = b_id.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let n = d.k / 16u;

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
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let ab = cur * 1040u + kk * 65u + l.y * 4u;
            let bb = cur * 1040u + kk * 65u + l.x * 4u;
            let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
            let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
            acc0 = acc0 + av.x * bv;
            acc1 = acc1 + av.y * bv;
            acc2 = acc2 + av.z * bv;
            acc3 = acc3 + av.w * bv;
        }
        if (tem_proximo) {
            guarda_a_km(tid, 1u - cur, ra);
            guarda_b_kn(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }

    let lr = lin0 + l.y * 4u;
    let lc = col0 + l.x * 4u;
    escreve(lr, lc, acc0, acc1, acc2, acc3);
}

@compute @workgroup_size(16, 16)
fn mm_abt(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * 64u;
    let col0 = b_id.y * 64u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let n = d.k / 16u;

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
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let ab = cur * 1040u + kk * 65u + l.y * 4u;
            let bb = cur * 1040u + kk * 65u + l.x * 4u;
            let av = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
            let bv = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
            acc0 = acc0 + av.x * bv;
            acc1 = acc1 + av.y * bv;
            acc2 = acc2 + av.z * bv;
            acc3 = acc3 + av.w * bv;
        }
        if (tem_proximo) {
            guarda_a_mk(tid, 1u - cur, ra);
            guarda_b_nk(tid, 1u - cur, rb);
        }
        workgroupBarrier();
        cur = 1u - cur;
    }

    let lr = lin0 + l.y * 4u;
    let lc = col0 + l.x * 4u;
    escreve(lr, lc, acc0, acc1, acc2, acc3);
}
"#;
