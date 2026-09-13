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
//! O bloco `8×8`, que o cuBLAS usa e que dobraria a intensidade para 4 FMAs por
//! leitura, foi implementado e **medido como 5% mais lento**: 3.846 contra
//! 4.030 GFLOP/s em 4096³. A razão está em `examples/ocupacao.rs`: a sonda
//! mostra que aritmética pura sustenta 15 a 18 TFLOP/s nesta placa, enquanto o
//! GEMM anda a 4. Estando 4× longe do limite das ULAs, o gargalo não é
//! intensidade aritmética, e dobrá-la só custa ocupação.
//!
//! Os acumuladores são acessados **sempre por índice constante**. Isso não é
//! estilo: a mesma sonda mediu que um array percorrido por índice de laço desaba
//! para 25% da vazão, porque o compilador deixa de desenrolar e derrama os
//! acumuladores para memória local.
//!
//! O acumulador é `array<vec4<f32>, 4>` com índices constantes, para que o
//! compilador o mantenha em registradores em vez de derramar para memória local.
//!
//! # Buffer duplo
//!
//! Sem sobreposição, cada iteração do laço de `K` para esperando a memória
//! global antes de poder calcular. O WebGPU não tem cópia assíncrona (`cp.async`
//! do CUDA), mas o efeito se obtém à mão: as leituras do ladrilho `t+1` são
//! **emitidas antes** do cálculo sobre o ladrilho `t` e ficam em registradores,
//! de modo que a latência corre por baixo da aritmética.
//!
//! São dois ladrilhos em memória compartilhada, alternados: 16 KB dos 48 KB
//! disponíveis. E como escrever no buffer ocioso não conflita com ler o ativo,
//! o padrão também reduz de duas para **uma barreira por iteração**.
//!
//! # Padding contra conflito de bancos
//!
//! A memória compartilhada tem 32 bancos de 4 bytes, e o banco de um endereço é
//! `(endereço/4) % 32`. Com o ladrilho de `A` guardado com passo `TM = 64`, a
//! escrita `sa[(idx%16)*64 + idx/16]` faz as 16 primeiras threads de um warp
//! endereçarem 0, 64, 128, … 960 — todos múltiplos de 32, **todos no banco 0**.
//! São 16 vias de conflito, e o hardware serializa os 16 acessos.
//!
//! Basta um elemento de padding: com passo `65`, os mesmos endereços viram
//! 0, 65, 130, … e caem nos bancos 0, 1, 2, … — um por banco, sem conflito. O
//! custo é 16 floats por ladrilho, e nada muda na aritmética.
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
struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;
// Passo de linha com um elemento de padding: quebra o conflito de bancos.
const LD: u32 = 65u;
const BUF: u32 = 1040u;   // TK * LD; há dois ladrilhos, alternados

var<workgroup> sa: array<f32, 2080>;
var<workgroup> sb: array<f32, 2080>;

// --- leituras globais, uma por variante ---------------------------------

// A guardada como [M, K]: lê a linha, `kx` contíguo entre threads vizinhas.
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

// A guardada como [K, M]: a transposta já está no layout desejado.
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

// B guardada como [K, N].
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

// B guardada como [N, K]: lê a linha de B, `kx` contíguo.
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

// --- escrita no buffer compartilhado ------------------------------------
//
// O destino em memória compartilhada é sempre [K][M] e [K][N], qualquer que
// seja o layout de origem: é o que o laço interno quer ler.

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

// --- núcleo compartilhado -----------------------------------------------

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

// Ordem de percurso dos blocos, com consciência de L2.
//
// A grade é despachada linearmente e o índice do bloco é reconstruído aqui, em
// vez de vir direto de `workgroup_id`. Percorrer em linha — o padrão — faz
// blocos vizinhos compartilharem o painel de `A` mas nunca o de `B`: ao chegar
// na linha seguinte, os painéis de `B` já saíram da L2 e são relidos.
//
// Agrupando `grupo` linhas e descendo dentro do grupo antes de avançar coluna,
// blocos consecutivos passam a compartilhar o painel de `B`, e os `grupo`
// painéis de `A` ficam residentes enquanto o grupo é varrido. É a blocagem de
// L2 do Goto, na forma que uma GPU permite: não se controla a cache, controla-se
// a ordem em que os blocos a visitam.
//
// `grupo = 1` reproduz exatamente o percurso em linha, o que torna a comparação
// entre os dois uma troca de uniforme, sem recompilar.
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
struct Dims { m: u32, n: u32, k: u32, grid_x: u32, grupo: u32, flag_relu: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<storage, read> vies: array<f32>;

const TM: u32 = 64u;
const TN: u32 = 64u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;
const LD: u32 = 65u;   // padding contra conflito de bancos

var<workgroup> sa: array<f32, 1040>;
var<workgroup> sb: array<f32, 1040>;

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

// Ordem de percurso dos blocos, com consciência de L2.
//
// A grade é despachada linearmente e o índice do bloco é reconstruído aqui, em
// vez de vir direto de `workgroup_id`. Percorrer em linha — o padrão — faz
// blocos vizinhos compartilharem o painel de `A` mas nunca o de `B`: ao chegar
// na linha seguinte, os painéis de `B` já saíram da L2 e são relidos.
//
// Agrupando `grupo` linhas e descendo dentro do grupo antes de avançar coluna,
// blocos consecutivos passam a compartilhar o painel de `B`, e os `grupo`
// painéis de `A` ficam residentes enquanto o grupo é varrido. É a blocagem de
// L2 do Goto, na forma que uma GPU permite: não se controla a cache, controla-se
// a ordem em que os blocos a visitam.
//
// `grupo = 1` reproduz exatamente o percurso em linha, o que torna a comparação
// entre os dois uma troca de uniforme, sem recompilar.
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
fn mm_bias(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    if (fora(w)) { return; }
    let tid = l.y * 16u + l.x;
    let b_id = bloco(w);
    let lin0 = b_id.x * TM;
    let col0 = b_id.y * TN;
    var acc = array<vec4<f32>, 4>();

    let passos = (d.k + TK - 1u) / TK;
    for (var t = 0u; t < passos; t = t + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let mm_ = idx / TK;
            let kx = idx % TK;
            let gr = lin0 + mm_;
            let gk = t * TK + kx;
            sa[kx * LD + mm_] = select(0.0, a[gr * d.k + gk], gr < d.m && gk < d.k);
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let kx = idx / TN;
            let nn = idx % TN;
            let gk = t * TK + kx;
            let gc = col0 + nn;
            sb[kx * LD + nn] = select(0.0, b[gk * d.n + gc], gk < d.k && gc < d.n);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TK; kk = kk + 1u) {
            let ab = kk * LD + l.y * 4u;
            let bb = kk * LD + l.x * 4u;
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
