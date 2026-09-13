//! Otimizadores.
//!
//! # Convenções
//!
//! | Símbolo | Significado |
//! |---|---|
//! | `θ ∈ ℝ^{d×m}` | um parâmetro qualquer (a mesma álgebra vale para `W` e `b`) |
//! | `g_t = ∂L/∂θ` | gradiente do passo `t`, mesmo shape de `θ` (convenção de denominador) |
//! | `α` | taxa de aprendizado (`lr`) |
//! | `μ` | coeficiente de momentum |
//! | `β₁, β₂` | decaimentos exponenciais dos momentos do Adam |
//! | `⊙`, `⊘`, `√` | Hadamard, divisão e raiz, todos elemento a elemento |
//! | `1` | tensor de uns com o shape de `θ` |
//!
//! Todas as recorrências abaixo são **pontuais**: cada coordenada de `θ` evolui
//! independentemente das outras. Não há produto matricial nenhum aqui, só
//! combinações lineares e operações elemento a elemento sobre buffers do mesmo
//! shape — ou seja, `axpy` e produtos de Hadamard.
//!
//! # Fusão das passadas
//!
//! Escrever a recorrência como cadeia de `scale`/`add`/`mul` de `Tensor` é
//! transparente mas caro: cada operação aloca um tensor intermediário e faz uma
//! varredura própria. O Adam encadeava sete delas por parâmetro e por passo
//! (`m·β₁`, `g·(1−β₁)`, soma, `m·1/c₁`, `v·β₂`, `g⊙g`, `·(1−β₂)`, soma,
//! `v·1/c₂`, o `zip` final e o `sub`) — sete alocações e dez varreduras onde o
//! trabalho útil cabe em uma.
//!
//! Aqui cada `step` é um único laço fundido sobre os slices `θ`, `g`, `m`, `v`,
//! atualizando os três buffers de estado in-place. Zero alocação por passo
//! (depois do primeiro, que cria `m` e `v` zerados) e uma só passada de memória.
//! A aritmética é executada exatamente na mesma ordem da versão original, então
//! o resultado é bit a bit o mesmo.

use std::collections::HashMap;

use crate::tape::{Grads, Param};
use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, params: &[Param], grads: &Grads);
}

/// Descida de gradiente estocástica, com momentum opcional.
///
/// # Recorrência vetorial
///
/// Sem momentum (`μ = 0`) é um único `axpy`:
///
/// ```text
/// θ_t = θ_{t−1} − α g_t
/// ```
///
/// Com momentum, a velocidade é uma média móvel não normalizada dos gradientes
/// (aqui na variante "clássica", com `α` já embutido em `v`):
///
/// ```text
/// v_t = μ v_{t−1} + α g_t
/// θ_t = θ_{t−1} − v_t
/// ```
///
/// Desenrolando, `v_t = α Σ_{i≤t} μ^{t−i} g_i`: o passo efetivo num gradiente
/// constante converge para `α/(1−μ)`, o fator `1/(1−μ)` de amplificação.
pub struct Sgd {
    lr: f32,
    momentum: f32,
    velocity: HashMap<usize, Tensor>,
}

impl Sgd {
    pub fn new(lr: f32) -> Sgd {
        Sgd { lr, momentum: 0.0, velocity: HashMap::new() }
    }

    pub fn with_momentum(lr: f32, momentum: f32) -> Sgd {
        Sgd { lr, momentum, velocity: HashMap::new() }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &[Param], grads: &Grads) {
        let (lr, mu) = (self.lr, self.momentum);
        for p in params {
            let g = match grads.of(p) {
                Some(g) => g,
                None => continue,
            };
            if mu == 0.0 {
                // θ ← θ − α g, um axpy in-place.
                p.update(|t| {
                    debug_assert_eq!(t.len(), g.len());
                    for (ti, &gi) in t.data_mut().iter_mut().zip(g.data()) {
                        *ti -= gi * lr;
                    }
                });
            } else {
                // v ← μ v + α g ; θ ← θ − v, numa passada só.
                let v = self
                    .velocity
                    .entry(p.id())
                    .or_insert_with(|| Tensor::zeros(&p.shape()));
                for (vi, &gi) in v.data_mut().iter_mut().zip(g.data()) {
                    *vi = *vi * mu + gi * lr;
                }
                let v = &*v;
                p.update(|t| {
                    debug_assert_eq!(t.len(), v.len());
                    for (ti, &vi) in t.data_mut().iter_mut().zip(v.data()) {
                        *ti -= vi;
                    }
                });
            }
        }
    }
}

/// Adam (Kingma & Ba, 2014), com correção de viés.
///
/// # Recorrência vetorial
///
/// Dois momentos exponenciais, o primeiro da média e o segundo da energia:
///
/// ```text
/// m_t = β₁ m_{t−1} + (1 − β₁) g_t                 (1º momento, ℝ^{shape de θ})
/// v_t = β₂ v_{t−1} + (1 − β₂) (g_t ⊙ g_t)         (2º momento, não centrado)
/// ```
///
/// Como `m₀ = v₀ = 0`, o desenrolamento dá `E[m_t] ≈ (1 − β₁ᵗ) E[g]`, isto é,
/// os momentos começam enviesados para zero. A correção divide pelo peso total
/// acumulado da média geométrica:
///
/// ```text
/// m̂_t = m_t / (1 − β₁ᵗ)
/// v̂_t = v_t / (1 − β₂ᵗ)
/// θ_t = θ_{t−1} − α · m̂_t ⊘ (√v̂_t + ε1)
/// ```
///
/// A divisão por `√v̂` é um pré-condicionamento diagonal: equivale a multiplicar
/// o gradiente por `diag(√v̂ + ε)⁻¹`, uma aproximação barata do inverso da raiz
/// da diagonal da Hessiana empírica. Por isso o passo é aproximadamente
/// invariante a reescalamentos `g → cg` — o `c` cancela entre `m̂` e `√v̂`.
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: i32,
    m: HashMap<usize, Tensor>,
    v: HashMap<usize, Tensor>,
}

impl Adam {
    pub fn new(lr: f32) -> Adam {
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: HashMap::new(), v: HashMap::new() }
    }

    pub fn with_betas(lr: f32, beta1: f32, beta2: f32) -> Adam {
        Adam { lr, beta1, beta2, eps: 1e-8, t: 0, m: HashMap::new(), v: HashMap::new() }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &[Param], grads: &Grads) {
        self.t += 1;
        let c1 = 1.0 - self.beta1.powi(self.t);
        let c2 = 1.0 - self.beta2.powi(self.t);
        // Fatores de correção pré-invertidos: a versão em Tensor usava
        // `scale(1.0 / c)`, então multiplicar pelo recíproco mantém o bit exato.
        let (inv_c1, inv_c2) = (1.0 / c1, 1.0 / c2);
        let (b1, b2) = (self.beta1, self.beta2);
        let (omb1, omb2) = (1.0 - b1, 1.0 - b2);
        let (lr, eps) = (self.lr, self.eps);

        for p in params {
            let g = match grads.of(p) {
                Some(g) => g,
                None => continue,
            };
            let m = self.m.entry(p.id()).or_insert_with(|| Tensor::zeros(&p.shape()));
            let v = self.v.entry(p.id()).or_insert_with(|| Tensor::zeros(&p.shape()));

            // Passada fundida: atualiza m, v e θ lendo g uma única vez.
            // Os quatro slices têm o mesmo comprimento, então o `zip` triplo
            // dispensa índice e checagem de limites.
            let (md, vd) = (m.data_mut(), v.data_mut());
            p.update(|t| {
                debug_assert_eq!(t.len(), g.len());
                let it = t.data_mut().iter_mut().zip(md.iter_mut()).zip(vd.iter_mut());
                for (((ti, mi), vi), &gi) in it.zip(g.data()) {
                    *mi = *mi * b1 + gi * omb1;
                    *vi = *vi * b2 + (gi * gi) * omb2;
                    let m_hat = *mi * inv_c1;
                    let v_hat = *vi * inv_c2;
                    *ti -= (m_hat / (v_hat.sqrt() + eps)) * lr;
                }
            });
        }
    }
}
