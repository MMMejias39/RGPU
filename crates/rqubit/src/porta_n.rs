//! Porta de `N` qubits e a álgebra para compor portas em conjuntos diferentes.
//!
//! É o que permite fundir além de dois qubits: duas portas que compartilham
//! algum qubit viram uma só, sobre a união dos seus conjuntos.

use crate::{mul, soma, Complexo, Porta1, Porta2};

/// Porta sobre um conjunto arbitrário de qubits.
///
/// A base é ordenada por `Σ bᵢ·2ⁱ`, onde `bᵢ` é o bit do qubit `alvos[i]` — a
/// mesma convenção de [`Porta2`], estendida.
#[derive(Clone, Debug)]
pub struct PortaN {
    pub alvos: Vec<usize>,
    /// Matriz `2ᵏ × 2ᵏ` em ordem de linha, com `k = alvos.len()`.
    pub m: Vec<Complexo>,
}

impl PortaN {
    pub fn dim(&self) -> usize {
        1 << self.alvos.len()
    }

    pub fn de_uma(p: &Porta1, q: usize) -> PortaN {
        PortaN { alvos: vec![q], m: vec![p.u00, p.u01, p.u10, p.u11] }
    }

    pub fn de_duas(g: &Porta2, q0: usize, q1: usize) -> PortaN {
        let mut m = Vec::with_capacity(16);
        for linha in 0..4 {
            for col in 0..4 {
                m.push(g.u[linha][col]);
            }
        }
        PortaN { alvos: vec![q0, q1], m }
    }

    /// Reescreve a porta sobre um conjunto maior de qubits.
    ///
    /// Para um índice `i` da base do conjunto novo, extrai os bits que
    /// correspondem aos alvos originais; os demais precisam coincidir entre
    /// linha e coluna, senão o elemento é zero. É o produto de Kronecker com a
    /// identidade, feito por índices em vez de por blocos.
    pub fn expandir(&self, novos: &[usize]) -> PortaN {
        debug_assert!(self.alvos.iter().all(|q| novos.contains(q)));
        let k = novos.len();
        let dim = 1 << k;
        // Posição de cada alvo original dentro do conjunto novo.
        let pos: Vec<usize> = self
            .alvos
            .iter()
            .map(|q| novos.iter().position(|n| n == q).expect("alvo ausente"))
            .collect();
        // Bits do conjunto novo que não pertencem à porta original.
        let resto: Vec<usize> = (0..k).filter(|b| !pos.contains(b)).collect();

        let mut m = vec![(0.0, 0.0); dim * dim];
        for linha in 0..dim {
            for col in 0..dim {
                // Os bits fora da porta têm de ser iguais nos dois índices.
                if resto.iter().any(|&b| (linha >> b) & 1 != (col >> b) & 1) {
                    continue;
                }
                let mut li = 0usize;
                let mut ci = 0usize;
                for (i, &b) in pos.iter().enumerate() {
                    li |= ((linha >> b) & 1) << i;
                    ci |= ((col >> b) & 1) << i;
                }
                m[linha * dim + col] = self.m[li * self.dim() + ci];
            }
        }
        PortaN { alvos: novos.to_vec(), m }
    }

    /// `self ∘ antes`: aplica `antes` e depois `self`, numa porta só.
    ///
    /// As duas são levadas à união dos seus conjuntos de qubits antes de
    /// multiplicar.
    pub fn compor(&self, antes: &PortaN) -> PortaN {
        let mut uniao = self.alvos.clone();
        for q in &antes.alvos {
            if !uniao.contains(q) {
                uniao.push(*q);
            }
        }
        let a = self.expandir(&uniao);
        let b = antes.expandir(&uniao);
        let dim = a.dim();

        let mut m = vec![(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = (0.0, 0.0);
                for t in 0..dim {
                    acc = soma(acc, mul(a.m[i * dim + t], b.m[t * dim + j]));
                }
                m[i * dim + j] = acc;
            }
        }
        PortaN { alvos: uniao, m }
    }

    /// Converte de volta para [`Porta1`], quando tem um alvo só.
    pub fn como_uma(&self) -> Porta1 {
        assert_eq!(self.alvos.len(), 1);
        Porta1 { u00: self.m[0], u01: self.m[1], u10: self.m[2], u11: self.m[3] }
    }

    /// Converte de volta para [`Porta2`], quando tem dois alvos.
    pub fn como_duas(&self) -> Porta2 {
        assert_eq!(self.alvos.len(), 2);
        let mut u = [[(0.0, 0.0); 4]; 4];
        for linha in 0..4 {
            for col in 0..4 {
                u[linha][col] = self.m[linha * 4 + col];
            }
        }
        Porta2 { u }
    }

    /// Aplica na CPU — referência para conferir o kernel.
    pub fn aplicar_cpu(&self, psi: &mut [Complexo]) {
        let k = self.alvos.len();
        let dim = 1 << k;
        let mut ordenados = self.alvos.clone();
        ordenados.sort_unstable();

        for g in 0..psi.len() / dim {
            // Insere `k` bits zero nas posições dos alvos, em ordem crescente.
            let mut base = g;
            for &q in &ordenados {
                let baixo = base & ((1 << q) - 1);
                base = ((base >> q) << (q + 1)) | baixo;
            }
            let idx: Vec<usize> = (0..dim)
                .map(|mascara| {
                    let mut i = base;
                    for (b, &q) in self.alvos.iter().enumerate() {
                        if (mascara >> b) & 1 == 1 {
                            i |= 1 << q;
                        }
                    }
                    i
                })
                .collect();
            let entrada: Vec<Complexo> = idx.iter().map(|&i| psi[i]).collect();
            for linha in 0..dim {
                let mut acc = (0.0, 0.0);
                for col in 0..dim {
                    acc = soma(acc, mul(self.m[linha * dim + col], entrada[col]));
                }
                psi[idx[linha]] = acc;
            }
        }
    }
}

/// Kernel genérico de `N` qubits, com `N` até 4.
///
/// # Medido: não compensa hoje
///
/// Fundir além de dois qubits reduz muito a contagem de portas e **aumenta** o
/// tempo. Circuito de 4 camadas em 26 qubits:
///
/// Circuito de 4 camadas em 26 qubits, com **este** kernel servindo 3 e 4
/// qubits:
///
/// | Máx. qubits | Portas | ms | Ganho |
/// |---|---|---|---|
/// | sem fusão | 308 | 1.493 | — |
/// | 2 | 112 | 545 | 2,74× |
/// | 3 | 58 | 595 | 2,51× |
/// | 4 | 40 | 667 | 2,24× |
///
/// Com 40 portas em vez de 112 o circuito ficava 22% mais lento. A causa está
/// no desenho abaixo: 64 threads por workgroup em vez de 256, estagiagem em
/// memória compartilhada, e laços com limite variável que o compilador não
/// desenrola.
///
/// [`PORTA3`] e [`PORTA4`] confirmaram o diagnóstico: com índices constantes, o
/// mesmo circuito caiu de 595 para **294 ms** em três qubits, e de 690 para
/// **204 ms** em quatro. A fusão até 4 passou de pior opção a melhor, com
/// **7,65×**.
///
/// Era o desenho, não a ideia. Este kernel fica como referência e caso de
/// comparação; o caminho padrão não o usa mais.
///
/// Cada thread cuida de um grupo de `2ᴺ` amplitudes e as estagia em memória de
/// workgroup — **na sua própria fatia**, sem compartilhar com as vizinhas, o que
/// dispensa barreiras e torna o retorno antecipado seguro.
///
/// A estagiagem existe porque o produto matriz-vetor precisa indexar as
/// amplitudes dinamicamente, e um array local com índice variável derrama para
/// memória local — medido em `rtensor/examples/ocupacao.rs`, onde a vazão cai a
/// 25%. Memória de workgroup não tem esse problema.
///
/// Com 64 threads e `N = 4` são 16 KB de memória compartilhada.
pub const PORTA_N: &str = r#"
struct P {
    grupos: u32, n: u32, gx: u32, pad: u32,
    alvos: vec4<u32>,
    ordenados: vec4<u32>,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> matriz: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> psi: array<vec2<f32>>;

// 64 threads × 2 fatias × 16 amplitudes.
var<workgroup> buf: array<vec2<f32>, 2048>;

fn alvo(i: u32) -> u32 {
    if (i == 0u) { return p.alvos.x; }
    if (i == 1u) { return p.alvos.y; }
    if (i == 2u) { return p.alvos.z; }
    return p.alvos.w;
}

fn ordenado(i: u32) -> u32 {
    if (i == 0u) { return p.ordenados.x; }
    if (i == 1u) { return p.ordenados.y; }
    if (i == 2u) { return p.ordenados.z; }
    return p.ordenados.w;
}

@compute @workgroup_size(64)
fn porta_n(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let g = (w.y * p.gx + w.x) * 64u + l.x;
    if (g >= p.grupos) { return; }

    let dim = 1u << p.n;
    let entrada = l.x * 32u;
    let saida = entrada + 16u;

    // Insere `n` bits zero nas posições dos alvos, em ordem crescente.
    var base = g;
    for (var i = 0u; i < p.n; i = i + 1u) {
        let q = ordenado(i);
        let baixo = base & ((1u << q) - 1u);
        base = ((base >> q) << (q + 1u)) | baixo;
    }

    for (var m = 0u; m < dim; m = m + 1u) {
        var idx = base;
        for (var b = 0u; b < p.n; b = b + 1u) {
            if (((m >> b) & 1u) == 1u) { idx = idx | (1u << alvo(b)); }
        }
        buf[entrada + m] = psi[idx];
    }

    for (var linha = 0u; linha < dim; linha = linha + 1u) {
        var acc = vec2<f32>(0.0, 0.0);
        for (var col = 0u; col < dim; col = col + 1u) {
            let c = matriz[linha * dim + col];
            let a = buf[entrada + col];
            acc = acc + vec2<f32>(c.x * a.x - c.y * a.y, c.x * a.y + c.y * a.x);
        }
        buf[saida + linha] = acc;
    }

    for (var m = 0u; m < dim; m = m + 1u) {
        var idx = base;
        for (var b = 0u; b < p.n; b = b + 1u) {
            if (((m >> b) & 1u) == 1u) { idx = idx | (1u << alvo(b)); }
        }
        psi[idx] = buf[saida + m];
    }
}
"#;

/// Kernel especializado de três qubits.
///
/// # Por que existe
///
/// O kernel genérico [`PORTA_N`] estagia as amplitudes em memória de workgroup
/// e percorre a matriz com laços de limite variável — necessário para servir a
/// qualquer `N`, e caro. Medido, fundir em unitárias de 3 qubits com ele
/// deixava o circuito **mais lento** que fundir só até 2.
///
/// Aqui `N` é fixo: as 8 amplitudes ficam em registradores, os 64 elementos da
/// matriz são endereçados por **índices literais**, e o workgroup volta a ter
/// 256 threads. Não há memória compartilhada nem um único laço.
///
/// A escolha de índices constantes não é estilo: `rtensor/examples/ocupacao.rs`
/// mediu que um array percorrido por índice de laço desaba para 25% da vazão,
/// porque o compilador deixa de desenrolar e derrama para memória local.
pub const PORTA3: &str = r#"
// Mesmo layout do kernel genérico, para que os dois compartilhem o uniforme.
// `n` é sempre 3 aqui e fica sem uso.
struct P {
    grupos: u32, n: u32, gx: u32, pad: u32,
    alvos: vec4<u32>,
    ordenados: vec4<u32>,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> matriz: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> psi: array<vec2<f32>>;

fn cmul(c: vec2<f32>, a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(c.x * a.x - c.y * a.y, c.x * a.y + c.y * a.x);
}

@compute @workgroup_size(256)
fn porta3(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let g = (w.y * p.gx + w.x) * 256u + l.x;
    if (g >= p.grupos) { return; }

    // Insere três bits zero nas posições dos alvos, em ordem crescente. Cada
    // inserção usa a posição final, e por isso a ordem importa: aplicar do
    // menor para o maior mantém as posições seguintes válidas.
    let o0 = p.ordenados.x;
    let o1 = p.ordenados.y;
    let o2 = p.ordenados.z;
    var base = ((g >> o0) << (o0 + 1u)) | (g & ((1u << o0) - 1u));
    base = ((base >> o1) << (o1 + 1u)) | (base & ((1u << o1) - 1u));
    base = ((base >> o2) << (o2 + 1u)) | (base & ((1u << o2) - 1u));

    // Os bits na ordem em que a base da matriz os espera — `alvos`, não
    // `ordenados`: a base é Σ bᵢ·2ⁱ sobre os alvos como o chamador os deu.
    let b0 = 1u << p.alvos.x;
    let b1 = 1u << p.alvos.y;
    let b2 = 1u << p.alvos.z;

    let i0 = base;
    let i1 = base | b0;
    let i2 = base | b1;
    let i3 = base | b0 | b1;
    let i4 = base | b2;
    let i5 = base | b0 | b2;
    let i6 = base | b1 | b2;
    let i7 = base | b0 | b1 | b2;

    let a0 = psi[i0];
    let a1 = psi[i1];
    let a2 = psi[i2];
    let a3 = psi[i3];
    let a4 = psi[i4];
    let a5 = psi[i5];
    let a6 = psi[i6];
    let a7 = psi[i7];

    let r0 = cmul(matriz[0], a0)
        + cmul(matriz[1], a1)
        + cmul(matriz[2], a2)
        + cmul(matriz[3], a3)
        + cmul(matriz[4], a4)
        + cmul(matriz[5], a5)
        + cmul(matriz[6], a6)
        + cmul(matriz[7], a7);
    let r1 = cmul(matriz[8], a0)
        + cmul(matriz[9], a1)
        + cmul(matriz[10], a2)
        + cmul(matriz[11], a3)
        + cmul(matriz[12], a4)
        + cmul(matriz[13], a5)
        + cmul(matriz[14], a6)
        + cmul(matriz[15], a7);
    let r2 = cmul(matriz[16], a0)
        + cmul(matriz[17], a1)
        + cmul(matriz[18], a2)
        + cmul(matriz[19], a3)
        + cmul(matriz[20], a4)
        + cmul(matriz[21], a5)
        + cmul(matriz[22], a6)
        + cmul(matriz[23], a7);
    let r3 = cmul(matriz[24], a0)
        + cmul(matriz[25], a1)
        + cmul(matriz[26], a2)
        + cmul(matriz[27], a3)
        + cmul(matriz[28], a4)
        + cmul(matriz[29], a5)
        + cmul(matriz[30], a6)
        + cmul(matriz[31], a7);
    let r4 = cmul(matriz[32], a0)
        + cmul(matriz[33], a1)
        + cmul(matriz[34], a2)
        + cmul(matriz[35], a3)
        + cmul(matriz[36], a4)
        + cmul(matriz[37], a5)
        + cmul(matriz[38], a6)
        + cmul(matriz[39], a7);
    let r5 = cmul(matriz[40], a0)
        + cmul(matriz[41], a1)
        + cmul(matriz[42], a2)
        + cmul(matriz[43], a3)
        + cmul(matriz[44], a4)
        + cmul(matriz[45], a5)
        + cmul(matriz[46], a6)
        + cmul(matriz[47], a7);
    let r6 = cmul(matriz[48], a0)
        + cmul(matriz[49], a1)
        + cmul(matriz[50], a2)
        + cmul(matriz[51], a3)
        + cmul(matriz[52], a4)
        + cmul(matriz[53], a5)
        + cmul(matriz[54], a6)
        + cmul(matriz[55], a7);
    let r7 = cmul(matriz[56], a0)
        + cmul(matriz[57], a1)
        + cmul(matriz[58], a2)
        + cmul(matriz[59], a3)
        + cmul(matriz[60], a4)
        + cmul(matriz[61], a5)
        + cmul(matriz[62], a6)
        + cmul(matriz[63], a7);

    psi[i0] = r0;
    psi[i1] = r1;
    psi[i2] = r2;
    psi[i3] = r3;
    psi[i4] = r4;
    psi[i5] = r5;
    psi[i6] = r6;
    psi[i7] = r7;
}
"#;

/// Kernel especializado de quatro qubits.
///
/// As 16 amplitudes ficam em registradores e os 256 elementos da matriz são
/// endereçados por índices literais — 16 linhas de 16 termos, geradas.
///
/// # A economia que torna isto viável
///
/// Calcular as 16 saídas antes de escrever qualquer uma manteria 32 valores
/// complexos vivos: 64 floats. Mas as **entradas já estão em registradores**,
/// então sobrescrever `psi[i]` não afeta as linhas seguintes — cada saída é
/// escrita assim que calculada, e sobra um acumulador vivo por vez em cima das
/// 16 entradas.
///
/// Sem isso a pressão de registradores derrubaria a ocupação, que é exatamente
/// o que `rtensor/examples/ocupacao.rs` mediu no outro domínio.
pub const PORTA4: &str = r#"
// Mesmo layout de uniforme dos outros kernels de N qubits.
struct P {
    grupos: u32, n: u32, gx: u32, pad: u32,
    alvos: vec4<u32>,
    ordenados: vec4<u32>,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> matriz: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> psi: array<vec2<f32>>;

fn cmul(c: vec2<f32>, a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(c.x * a.x - c.y * a.y, c.x * a.y + c.y * a.x);
}

@compute @workgroup_size(256)
fn porta4(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let g = (w.y * p.gx + w.x) * 256u + l.x;
    if (g >= p.grupos) { return; }

    // Quatro bits zero inseridos nas posições dos alvos, em ordem crescente.
    let o0 = p.ordenados.x;
    let o1 = p.ordenados.y;
    let o2 = p.ordenados.z;
    let o3 = p.ordenados.w;
    var base = ((g >> o0) << (o0 + 1u)) | (g & ((1u << o0) - 1u));
    base = ((base >> o1) << (o1 + 1u)) | (base & ((1u << o1) - 1u));
    base = ((base >> o2) << (o2 + 1u)) | (base & ((1u << o2) - 1u));
    base = ((base >> o3) << (o3 + 1u)) | (base & ((1u << o3) - 1u));

    let b0 = 1u << p.alvos.x;
    let b1 = 1u << p.alvos.y;
    let b2 = 1u << p.alvos.z;
    let b3 = 1u << p.alvos.w;

    let i0 = base;
    let i1 = base | b0;
    let i2 = base | b1;
    let i3 = base | b0 | b1;
    let i4 = base | b2;
    let i5 = base | b0 | b2;
    let i6 = base | b1 | b2;
    let i7 = base | b0 | b1 | b2;
    let i8 = base | b3;
    let i9 = base | b0 | b3;
    let i10 = base | b1 | b3;
    let i11 = base | b0 | b1 | b3;
    let i12 = base | b2 | b3;
    let i13 = base | b0 | b2 | b3;
    let i14 = base | b1 | b2 | b3;
    let i15 = base | b0 | b1 | b2 | b3;

    let a0 = psi[i0];
    let a1 = psi[i1];
    let a2 = psi[i2];
    let a3 = psi[i3];
    let a4 = psi[i4];
    let a5 = psi[i5];
    let a6 = psi[i6];
    let a7 = psi[i7];
    let a8 = psi[i8];
    let a9 = psi[i9];
    let a10 = psi[i10];
    let a11 = psi[i11];
    let a12 = psi[i12];
    let a13 = psi[i13];
    let a14 = psi[i14];
    let a15 = psi[i15];

    psi[i0] = cmul(matriz[0], a0)
        + cmul(matriz[1], a1)
        + cmul(matriz[2], a2)
        + cmul(matriz[3], a3)
        + cmul(matriz[4], a4)
        + cmul(matriz[5], a5)
        + cmul(matriz[6], a6)
        + cmul(matriz[7], a7)
        + cmul(matriz[8], a8)
        + cmul(matriz[9], a9)
        + cmul(matriz[10], a10)
        + cmul(matriz[11], a11)
        + cmul(matriz[12], a12)
        + cmul(matriz[13], a13)
        + cmul(matriz[14], a14)
        + cmul(matriz[15], a15);
    psi[i1] = cmul(matriz[16], a0)
        + cmul(matriz[17], a1)
        + cmul(matriz[18], a2)
        + cmul(matriz[19], a3)
        + cmul(matriz[20], a4)
        + cmul(matriz[21], a5)
        + cmul(matriz[22], a6)
        + cmul(matriz[23], a7)
        + cmul(matriz[24], a8)
        + cmul(matriz[25], a9)
        + cmul(matriz[26], a10)
        + cmul(matriz[27], a11)
        + cmul(matriz[28], a12)
        + cmul(matriz[29], a13)
        + cmul(matriz[30], a14)
        + cmul(matriz[31], a15);
    psi[i2] = cmul(matriz[32], a0)
        + cmul(matriz[33], a1)
        + cmul(matriz[34], a2)
        + cmul(matriz[35], a3)
        + cmul(matriz[36], a4)
        + cmul(matriz[37], a5)
        + cmul(matriz[38], a6)
        + cmul(matriz[39], a7)
        + cmul(matriz[40], a8)
        + cmul(matriz[41], a9)
        + cmul(matriz[42], a10)
        + cmul(matriz[43], a11)
        + cmul(matriz[44], a12)
        + cmul(matriz[45], a13)
        + cmul(matriz[46], a14)
        + cmul(matriz[47], a15);
    psi[i3] = cmul(matriz[48], a0)
        + cmul(matriz[49], a1)
        + cmul(matriz[50], a2)
        + cmul(matriz[51], a3)
        + cmul(matriz[52], a4)
        + cmul(matriz[53], a5)
        + cmul(matriz[54], a6)
        + cmul(matriz[55], a7)
        + cmul(matriz[56], a8)
        + cmul(matriz[57], a9)
        + cmul(matriz[58], a10)
        + cmul(matriz[59], a11)
        + cmul(matriz[60], a12)
        + cmul(matriz[61], a13)
        + cmul(matriz[62], a14)
        + cmul(matriz[63], a15);
    psi[i4] = cmul(matriz[64], a0)
        + cmul(matriz[65], a1)
        + cmul(matriz[66], a2)
        + cmul(matriz[67], a3)
        + cmul(matriz[68], a4)
        + cmul(matriz[69], a5)
        + cmul(matriz[70], a6)
        + cmul(matriz[71], a7)
        + cmul(matriz[72], a8)
        + cmul(matriz[73], a9)
        + cmul(matriz[74], a10)
        + cmul(matriz[75], a11)
        + cmul(matriz[76], a12)
        + cmul(matriz[77], a13)
        + cmul(matriz[78], a14)
        + cmul(matriz[79], a15);
    psi[i5] = cmul(matriz[80], a0)
        + cmul(matriz[81], a1)
        + cmul(matriz[82], a2)
        + cmul(matriz[83], a3)
        + cmul(matriz[84], a4)
        + cmul(matriz[85], a5)
        + cmul(matriz[86], a6)
        + cmul(matriz[87], a7)
        + cmul(matriz[88], a8)
        + cmul(matriz[89], a9)
        + cmul(matriz[90], a10)
        + cmul(matriz[91], a11)
        + cmul(matriz[92], a12)
        + cmul(matriz[93], a13)
        + cmul(matriz[94], a14)
        + cmul(matriz[95], a15);
    psi[i6] = cmul(matriz[96], a0)
        + cmul(matriz[97], a1)
        + cmul(matriz[98], a2)
        + cmul(matriz[99], a3)
        + cmul(matriz[100], a4)
        + cmul(matriz[101], a5)
        + cmul(matriz[102], a6)
        + cmul(matriz[103], a7)
        + cmul(matriz[104], a8)
        + cmul(matriz[105], a9)
        + cmul(matriz[106], a10)
        + cmul(matriz[107], a11)
        + cmul(matriz[108], a12)
        + cmul(matriz[109], a13)
        + cmul(matriz[110], a14)
        + cmul(matriz[111], a15);
    psi[i7] = cmul(matriz[112], a0)
        + cmul(matriz[113], a1)
        + cmul(matriz[114], a2)
        + cmul(matriz[115], a3)
        + cmul(matriz[116], a4)
        + cmul(matriz[117], a5)
        + cmul(matriz[118], a6)
        + cmul(matriz[119], a7)
        + cmul(matriz[120], a8)
        + cmul(matriz[121], a9)
        + cmul(matriz[122], a10)
        + cmul(matriz[123], a11)
        + cmul(matriz[124], a12)
        + cmul(matriz[125], a13)
        + cmul(matriz[126], a14)
        + cmul(matriz[127], a15);
    psi[i8] = cmul(matriz[128], a0)
        + cmul(matriz[129], a1)
        + cmul(matriz[130], a2)
        + cmul(matriz[131], a3)
        + cmul(matriz[132], a4)
        + cmul(matriz[133], a5)
        + cmul(matriz[134], a6)
        + cmul(matriz[135], a7)
        + cmul(matriz[136], a8)
        + cmul(matriz[137], a9)
        + cmul(matriz[138], a10)
        + cmul(matriz[139], a11)
        + cmul(matriz[140], a12)
        + cmul(matriz[141], a13)
        + cmul(matriz[142], a14)
        + cmul(matriz[143], a15);
    psi[i9] = cmul(matriz[144], a0)
        + cmul(matriz[145], a1)
        + cmul(matriz[146], a2)
        + cmul(matriz[147], a3)
        + cmul(matriz[148], a4)
        + cmul(matriz[149], a5)
        + cmul(matriz[150], a6)
        + cmul(matriz[151], a7)
        + cmul(matriz[152], a8)
        + cmul(matriz[153], a9)
        + cmul(matriz[154], a10)
        + cmul(matriz[155], a11)
        + cmul(matriz[156], a12)
        + cmul(matriz[157], a13)
        + cmul(matriz[158], a14)
        + cmul(matriz[159], a15);
    psi[i10] = cmul(matriz[160], a0)
        + cmul(matriz[161], a1)
        + cmul(matriz[162], a2)
        + cmul(matriz[163], a3)
        + cmul(matriz[164], a4)
        + cmul(matriz[165], a5)
        + cmul(matriz[166], a6)
        + cmul(matriz[167], a7)
        + cmul(matriz[168], a8)
        + cmul(matriz[169], a9)
        + cmul(matriz[170], a10)
        + cmul(matriz[171], a11)
        + cmul(matriz[172], a12)
        + cmul(matriz[173], a13)
        + cmul(matriz[174], a14)
        + cmul(matriz[175], a15);
    psi[i11] = cmul(matriz[176], a0)
        + cmul(matriz[177], a1)
        + cmul(matriz[178], a2)
        + cmul(matriz[179], a3)
        + cmul(matriz[180], a4)
        + cmul(matriz[181], a5)
        + cmul(matriz[182], a6)
        + cmul(matriz[183], a7)
        + cmul(matriz[184], a8)
        + cmul(matriz[185], a9)
        + cmul(matriz[186], a10)
        + cmul(matriz[187], a11)
        + cmul(matriz[188], a12)
        + cmul(matriz[189], a13)
        + cmul(matriz[190], a14)
        + cmul(matriz[191], a15);
    psi[i12] = cmul(matriz[192], a0)
        + cmul(matriz[193], a1)
        + cmul(matriz[194], a2)
        + cmul(matriz[195], a3)
        + cmul(matriz[196], a4)
        + cmul(matriz[197], a5)
        + cmul(matriz[198], a6)
        + cmul(matriz[199], a7)
        + cmul(matriz[200], a8)
        + cmul(matriz[201], a9)
        + cmul(matriz[202], a10)
        + cmul(matriz[203], a11)
        + cmul(matriz[204], a12)
        + cmul(matriz[205], a13)
        + cmul(matriz[206], a14)
        + cmul(matriz[207], a15);
    psi[i13] = cmul(matriz[208], a0)
        + cmul(matriz[209], a1)
        + cmul(matriz[210], a2)
        + cmul(matriz[211], a3)
        + cmul(matriz[212], a4)
        + cmul(matriz[213], a5)
        + cmul(matriz[214], a6)
        + cmul(matriz[215], a7)
        + cmul(matriz[216], a8)
        + cmul(matriz[217], a9)
        + cmul(matriz[218], a10)
        + cmul(matriz[219], a11)
        + cmul(matriz[220], a12)
        + cmul(matriz[221], a13)
        + cmul(matriz[222], a14)
        + cmul(matriz[223], a15);
    psi[i14] = cmul(matriz[224], a0)
        + cmul(matriz[225], a1)
        + cmul(matriz[226], a2)
        + cmul(matriz[227], a3)
        + cmul(matriz[228], a4)
        + cmul(matriz[229], a5)
        + cmul(matriz[230], a6)
        + cmul(matriz[231], a7)
        + cmul(matriz[232], a8)
        + cmul(matriz[233], a9)
        + cmul(matriz[234], a10)
        + cmul(matriz[235], a11)
        + cmul(matriz[236], a12)
        + cmul(matriz[237], a13)
        + cmul(matriz[238], a14)
        + cmul(matriz[239], a15);
    psi[i15] = cmul(matriz[240], a0)
        + cmul(matriz[241], a1)
        + cmul(matriz[242], a2)
        + cmul(matriz[243], a3)
        + cmul(matriz[244], a4)
        + cmul(matriz[245], a5)
        + cmul(matriz[246], a6)
        + cmul(matriz[247], a7)
        + cmul(matriz[248], a8)
        + cmul(matriz[249], a9)
        + cmul(matriz[250], a10)
        + cmul(matriz[251], a11)
        + cmul(matriz[252], a12)
        + cmul(matriz[253], a13)
        + cmul(matriz[254], a14)
        + cmul(matriz[255], a15);
}
"#;
