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
/// [`PORTA3`] confirmou o diagnóstico: especializando três qubits com índices
/// constantes, o mesmo circuito caiu de 595 para **293 ms**, e a fusão até 3
/// passou a ser a melhor opção. Quatro qubits ainda usa este kernel, e ainda
/// perde — especializá-lo é o passo seguinte óbvio.
///
/// Por isso [`Estado::aplicar_circuito_fundido`] despacha portas de 1, 2 e 3
/// qubits para os kernels especializados, e só usa este em 4.
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
