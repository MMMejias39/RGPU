//! Operações diferenciáveis sobre `Var`.
//!
//! Cada função calcula o valor à frente e registra na fita a regra da cadeia
//! correspondente. Gradientes de operandos difundidos são somados de volta ao
//! shape original por [`Tensor::reduce_to`].
//!
//! # Convenções
//!
//! | Símbolo | Significado |
//! |---|---|
//! | `A, B, X, Z` | tensores de entrada de uma operação |
//! | `C`, `Y`, `S` | tensor de saída |
//! | `Ā = ∂L/∂A` | adjunto de `A`; **mesmo shape de `A`** (convenção de denominador) |
//! | `g = C̄` | adjunto que chega pela fita, com o shape da saída |
//! | `⊙`, `⊘` | produto e divisão de Hadamard (elemento a elemento) |
//! | `Jᵀg` | produto vetor-jacobiana; nunca se materializa `J` |
//! | `1ₙ` | vetor coluna de uns; `ρ_S(·)` é a redução ao shape `S` (`1ᵀ` nos eixos difundidos) |
//!
//! # As três famílias
//!
//! 1. **Pontuais** (`square`, `exp`, `relu`, `tanh`, …). A jacobiana é diagonal,
//!    `J = diag(f'(x))`, logo `Jᵀg = g ⊙ f'.(X)` — nunca se forma a matriz
//!    `n×n`, só um Hadamard. Onde `f'` se escreve em função da saída
//!    (`sigmoid`, `tanh`, `exp`, `sqrt`), guarda-se a saída em vez da entrada.
//!
//! 2. **Bilineares com broadcasting** (`add`, `sub`, `mul`, `div`). A regra é a
//!    pontual composta com a redução: difundir é `A ↦ 1ₙA`, cujo adjunto é
//!    `1ₙᵀ(·)`. Por isso todo adjunto aqui termina em `ρ_{shape(A)}(·)`.
//!
//! 3. **Contrações** (`matmul`, `sum`, `mean`, `softmax`). São as únicas em que
//!    a jacobiana não é diagonal; ver a derivação em cada uma.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::tape::Var;
use crate::tensor::Tensor;

impl Var {
    /// `C = A ⊕ B` (soma com broadcasting).
    ///
    /// ```text
    /// Ā = ρ_{shape(A)}(g)
    /// B̄ = ρ_{shape(B)}(g)
    /// ```
    ///
    /// É este nó que implementa o `+ 1ₙb` de uma camada densa: com
    /// `b ∈ ℝ^{1×m}` e `g = δ ∈ ℝ^{n×m}`, a redução é a contração
    /// `∂L/∂b = 1ₙᵀδ`, a soma das linhas de `δ`.
    pub fn add(&self, other: &Var) -> Var {
        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        let value = self.value().add(other.value());
        self.binary(other, value, move |g| (g.reduce_to(&sa), g.reduce_to(&sb)))
    }

    /// `C = A ⊖ B`.
    ///
    /// ```text
    /// Ā = ρ_{shape(A)}(g)
    /// B̄ = ρ_{shape(B)}(−g)
    /// ```
    pub fn sub(&self, other: &Var) -> Var {
        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        let value = self.value().sub(other.value());
        self.binary(other, value, move |g| (g.reduce_to(&sa), g.neg().reduce_to(&sb)))
    }

    /// `C = A ⊙ B` (Hadamard com broadcasting).
    ///
    /// ```text
    /// Ā = ρ_{shape(A)}(g ⊙ B)
    /// B̄ = ρ_{shape(B)}(g ⊙ A)
    /// ```
    ///
    /// A jacobiana em relação a `A` é `diag(vec(B))`; simétrica, então
    /// `Jᵀg = Jg = g ⊙ B`.
    pub fn mul(&self, other: &Var) -> Var {
        let (a, b) = (self.value().clone(), other.value().clone());
        let value = a.mul(&b);
        let (sa, sb) = (a.shape().to_vec(), b.shape().to_vec());
        self.binary(other, value, move |g| {
            (g.mul(&b).reduce_to(&sa), g.mul(&a).reduce_to(&sb))
        })
    }

    /// `C = A ⊘ B`.
    ///
    /// De `dC = dA ⊘ B − (A ⊘ B⊙B) ⊙ dB`:
    ///
    /// ```text
    /// Ā = ρ_{shape(A)}(g ⊘ B)
    /// B̄ = ρ_{shape(B)}(−g ⊙ A ⊘ (B ⊙ B))
    /// ```
    pub fn div(&self, other: &Var) -> Var {
        let (a, b) = (self.value().clone(), other.value().clone());
        let value = a.div(&b);
        let (sa, sb) = (a.shape().to_vec(), b.shape().to_vec());
        self.binary(other, value, move |g| {
            let da = g.div(&b).reduce_to(&sa);
            let db = g.mul(&a).div(&b.mul(&b)).neg().reduce_to(&sb);
            (da, db)
        })
    }

    /// Produto matricial `C = AB`, `A ∈ ℝ^{m×k}`, `B ∈ ℝ^{k×n}`.
    ///
    /// # Derivação
    ///
    /// A diferencial é `dC = (dA)B + A(dB)`. Com o produto interno de Frobenius
    /// `⟨U, V⟩ = tr(UᵀV)` e `dL = ⟨C̄, dC⟩`:
    ///
    /// ```text
    /// ⟨C̄, (dA)B⟩ = tr(C̄ᵀ dA B) = tr(B C̄ᵀ dA) = ⟨C̄Bᵀ, dA⟩   ⇒  Ā = C̄ Bᵀ
    /// ⟨C̄, A(dB)⟩ = tr(C̄ᵀ A dB)               = ⟨AᵀC̄, dB⟩   ⇒  B̄ = Aᵀ C̄
    /// ```
    ///
    /// ```text
    /// Ā = C̄ Bᵀ ∈ ℝ^{m×k}
    /// B̄ = Aᵀ C̄ ∈ ℝ^{k×n}
    /// ```
    ///
    /// Numa camada densa (`A = X ∈ ℝ^{n×d}`, `B = W ∈ ℝ^{d×m}`, `C̄ = δ`) isso é
    /// exatamente `∂L/∂X = δWᵀ` e `∂L/∂W = Xᵀδ`. Note que `Xᵀδ` é uma soma de
    /// `n` produtos externos `xᵢδᵢᵀ` — a versão em lote do produto externo
    /// `entrada × erro` que aparece no backprop escrito amostra a amostra.
    pub fn matmul(&self, other: &Var) -> Var {
        let (a, b) = (self.value().clone(), other.value().clone());
        let value = a.matmul(&b);
        self.binary(other, value, move |g| (g.matmul(&b.t()), a.t().matmul(g)))
    }

    /// `Y = kX`, `k` constante. Jacobiana `kI`, logo `X̄ = k g`.
    pub fn scale(&self, k: f32) -> Var {
        let value = self.value().scale(k);
        self.unary(value, move |g| g.scale(k))
    }

    /// `Y = X + k1`, `k` constante. Jacobiana `I`, logo `X̄ = g`
    /// (deslocamento por constante não altera derivada).
    pub fn offset(&self, k: f32) -> Var {
        let value = self.value().map(|x| x + k);
        self.unary(value, |g| g.clone())
    }

    /// `Y = −X`; `X̄ = −g`.
    pub fn neg(&self) -> Var {
        self.scale(-1.0)
    }

    /// `Y = X ⊙ X`. Jacobiana `diag(2x)`, logo `X̄ = 2 (g ⊙ X)`.
    pub fn square(&self) -> Var {
        let x = self.value().clone();
        let value = x.map(|v| v * v);
        self.unary(value, move |g| g.mul(&x).scale(2.0))
    }

    /// `Y = √X` (elemento a elemento). Como `y' = 1/(2√x) = 1/(2y)`, guarda-se a
    /// saída: `X̄ = g ⊘ (2Y)`.
    pub fn sqrt(&self) -> Var {
        let out = self.value().map(f32::sqrt);
        let cached = out.clone();
        self.unary(out, move |g| g.div(&cached.scale(2.0)))
    }

    /// `Y = exp.(X)`. A exponencial é sua própria derivada, então basta a saída:
    /// `X̄ = g ⊙ Y`.
    pub fn exp(&self) -> Var {
        let out = self.value().map(f32::exp);
        let cached = out.clone();
        self.unary(out, move |g| g.mul(&cached))
    }

    /// `Y = ln.(max(X, ε))`, com `ε = 1e-12` de piso. `X̄ = g ⊘ X`.
    ///
    /// O piso age só no valor à frente (evita `ln 0 = −∞`); o adjunto usa o `X`
    /// original, como na derivada da composição fora da região saturada.
    pub fn ln(&self) -> Var {
        let x = self.value().clone();
        let value = x.map(|v| v.max(1e-12).ln());
        self.unary(value, move |g| g.div(&x))
    }

    /// `Y = max(X, 0)`. Jacobiana `diag(1[X > 0])` (subgradiente 0 em `x = 0`):
    ///
    /// ```text
    /// X̄ = g ⊙ 1[X > 0]
    /// ```
    ///
    /// Uma máscara binária — é o que torna o `matmul` seguinte esparso em
    /// linhas, e por isso o `matmul` pula fatores nulos.
    pub fn relu(&self) -> Var {
        let x = self.value().clone();
        let value = x.map(|v| v.max(0.0));
        self.unary(value, move |g| g.mul(&x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })))
    }

    /// `Y = max(X, 0) + s·min(X, 0)`. `X̄ = g ⊙ (1[X > 0] + s·1[X ≤ 0])`.
    pub fn leaky_relu(&self, slope: f32) -> Var {
        let x = self.value().clone();
        let value = x.map(|v| if v > 0.0 { v } else { slope * v });
        self.unary(value, move |g| {
            g.mul(&x.map(|v| if v > 0.0 { 1.0 } else { slope }))
        })
    }

    /// `S = σ.(X)`, `σ(x) = 1/(1 + e^{−x})`.
    ///
    /// Como `σ' = σ(1 − σ)`, a derivada se escreve só com a saída:
    ///
    /// ```text
    /// X̄ = g ⊙ S ⊙ (1 − S)
    /// ```
    pub fn sigmoid(&self) -> Var {
        let out = self.value().map(|v| 1.0 / (1.0 + (-v).exp()));
        let cached = out.clone();
        self.unary(out, move |g| g.mul(&cached.map(|s| s * (1.0 - s))))
    }

    /// `T = tanh.(X)`. Como `tanh' = 1 − tanh²`:
    ///
    /// ```text
    /// X̄ = g ⊙ (1 − T ⊙ T)
    /// ```
    pub fn tanh(&self) -> Var {
        let out = self.value().map(f32::tanh);
        let cached = out.clone();
        self.unary(out, move |g| g.mul(&cached.map(|t| 1.0 - t * t)))
    }

    /// Soma todos os elementos, devolvendo um escalar: `s = 1ᵀ vec(X)`.
    ///
    /// A jacobiana é o vetor-linha `1ᵀ`, logo o adjunto é a difusão do escalar:
    ///
    /// ```text
    /// X̄ = s̄ · 1    (o mesmo escalar em todas as posições, no shape de X)
    /// ```
    ///
    /// Soma e difusão são adjuntas uma da outra — a mesma dualidade de
    /// [`Tensor::reduce_to`].
    pub fn sum(&self) -> Var {
        let shape = self.shape().to_vec();
        let value = Tensor::scalar(self.value().sum_all());
        self.unary(value, move |g| Tensor::full(&shape, g.item()))
    }

    /// Média de todos os elementos: `s = (1/N) 1ᵀ vec(X)`, `N = |X|`.
    ///
    /// ```text
    /// X̄ = (s̄ / N) · 1
    /// ```
    pub fn mean(&self) -> Var {
        let shape = self.shape().to_vec();
        let n = self.value().len() as f32;
        let value = Tensor::scalar(self.value().mean_all());
        self.unary(value, move |g| Tensor::full(&shape, g.item() / n))
    }

    /// Softmax por linha. Para treinar classificadores prefira
    /// [`crate::losses::softmax_cross_entropy`], numericamente mais estável.
    ///
    /// # Jacobiana e adjunto
    ///
    /// Para uma linha `s = softmax(z) ∈ ℝ^c`, de `∂sᵢ/∂zⱼ = sᵢ(δᵢⱼ − sⱼ)` vem
    ///
    /// ```text
    /// J = diag(s) − s sᵀ                  (simétrica, J = Jᵀ, posto c−1)
    /// ```
    ///
    /// Aplicá-la custaria `O(c²)` e `c²` bytes de memória, mas a estrutura
    /// "diagonal menos posto-1" permite `O(c)`:
    ///
    /// ```text
    /// Jᵀg = Jg = diag(s)g − s(sᵀg) = s ⊙ g − (gᵀs) s = s ⊙ (g − (gᵀs)1)
    /// ```
    ///
    /// isto é: um produto interno por linha (`gᵀs`, um `dot`), a subtração desse
    /// escalar difundido e um Hadamard. O termo `(gᵀs)1` é a projeção que faz
    /// `1ᵀ(Jᵀg) = 0` — o adjunto sempre tem soma nula por linha, reflexo de
    /// `softmax(z + c1) = softmax(z)`, que torna `1` um vetor nulo de `J`.
    ///
    /// Em forma matricial, para `S, G ∈ ℝ^{n×c}` e `d = (G ⊙ S)1_c ∈ ℝ^{n×1}`:
    ///
    /// ```text
    /// Z̄ = S ⊙ (G − d 1_cᵀ)
    /// ```
    ///
    /// Cada linha é um slice contíguo: um `dot` e um `axpy`-Hadamard, sem o
    /// tensor intermediário `G ⊙ S`.
    pub fn softmax(&self) -> Var {
        let out = self.value().softmax_rows();
        let cached = out.clone();
        self.unary(out, move |g| {
            let (r, c) = (cached.shape()[0], cached.shape()[1]);
            let mut data = vec![0.0f32; r * c];
            let rows = data
                .chunks_exact_mut(c.max(1))
                .zip(cached.data().chunks_exact(c.max(1)))
                .zip(g.data().chunks_exact(c.max(1)));
            for ((dst, s), gr) in rows {
                // d = gᵀs (produto interno da linha)
                let dot: f32 = gr.iter().zip(s).map(|(&gi, &si)| gi * si).sum();
                // linha do adjunto: s ⊙ (g − d·1)
                for ((o, &si), &gi) in dst.iter_mut().zip(s).zip(gr) {
                    *o = si * (gi - dot);
                }
            }
            Tensor::new(&[r, c], data)
        })
    }
}

impl Add for &Var {
    type Output = Var;
    fn add(self, rhs: &Var) -> Var {
        Var::add(self, rhs)
    }
}

impl Sub for &Var {
    type Output = Var;
    fn sub(self, rhs: &Var) -> Var {
        Var::sub(self, rhs)
    }
}

impl Mul for &Var {
    type Output = Var;
    fn mul(self, rhs: &Var) -> Var {
        Var::mul(self, rhs)
    }
}

impl Div for &Var {
    type Output = Var;
    fn div(self, rhs: &Var) -> Var {
        Var::div(self, rhs)
    }
}

impl Neg for &Var {
    type Output = Var;
    fn neg(self) -> Var {
        Var::neg(self)
    }
}
