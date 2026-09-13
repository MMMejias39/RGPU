//! Camadas e modelos — o equivalente enxuto de `tf.keras.layers` / `Sequential`.
//!
//! # Convenções
//!
//! | Símbolo | Significado |
//! |---|---|
//! | `X ∈ ℝ^{n×d}` | entrada do lote: `n` amostras em linha, `d` atributos |
//! | `W ∈ ℝ^{d×m}` | pesos da camada |
//! | `b ∈ ℝ^{1×m}` | viés (vetor-linha, difundido no lote) |
//! | `Z = XW + 1ₙb ∈ ℝ^{n×m}` | pré-ativação |
//! | `A = φ.(Z)` | ativação (pontual) |
//! | `δ = ∂L/∂Z ∈ ℝ^{n×m}` | adjunto da pré-ativação |
//! | `1ₙ ∈ ℝ^{n×1}` | vetor coluna de uns |
//!
//! Convenção de denominador: `∂L/∂W ∈ ℝ^{d×m}`, `∂L/∂b ∈ ℝ^{1×m}`,
//! `∂L/∂X ∈ ℝ^{n×d}`.
//!
//! # Backprop de uma pilha
//!
//! Para camadas `1..K` com `A⁰ = X` e `A^k = φ_k.(A^{k−1}W^k + 1ₙb^k)`, o
//! backward é a recorrência
//!
//! ```text
//! δ^K = Ā^K ⊙ φ_K'.(Z^K)
//! ∂L/∂W^k = (A^{k−1})ᵀ δ^k
//! ∂L/∂b^k = 1ₙᵀ δ^k
//! δ^{k−1} = (δ^k (W^k)ᵀ) ⊙ φ_{k−1}'.(Z^{k−1})
//! ```
//!
//! Ou seja: o erro volta pela **transposta** dos mesmos pesos que o propagaram
//! para frente, modulado pela derivada da ativação. Tudo em nível BLAS-3
//! (`matmul`) e BLAS-1 (`Hadamard`) — a fita monta essa recorrência sozinha a
//! partir dos adjuntos de [`crate::ops`].

use crate::rng::Rng;
use crate::tape::{Param, Var};
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    Linear,
    Relu,
    LeakyRelu(f32),
    Sigmoid,
    Tanh,
}

impl Activation {
    /// Aplica `A = φ.(Z)`. Todas as opções são pontuais, então a jacobiana é
    /// `diag(φ'(z))` e o adjunto é `δ = Ā ⊙ φ'.(Z)` — as derivadas explícitas
    /// estão documentadas em cada método de [`crate::ops`].
    pub fn apply(&self, x: &Var) -> Var {
        match *self {
            Activation::Linear => x.clone(),
            Activation::Relu => x.relu(),
            Activation::LeakyRelu(s) => x.leaky_relu(s),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Tanh => x.tanh(),
        }
    }

    fn name(&self) -> String {
        match *self {
            Activation::Linear => "linear".into(),
            Activation::Relu => "relu".into(),
            Activation::LeakyRelu(s) => format!("leaky_relu({s})"),
            Activation::Sigmoid => "sigmoid".into(),
            Activation::Tanh => "tanh".into(),
        }
    }
}

pub trait Layer {
    fn forward(&self, x: &Var) -> Var;
    fn params(&self) -> Vec<Param>;
    fn describe(&self) -> String;
}

/// Camada totalmente conectada.
///
/// # Forward
///
/// ```text
/// Z = X W + 1ₙ b        X ∈ ℝ^{n×d}, W ∈ ℝ^{d×m}, b ∈ ℝ^{1×m}, Z ∈ ℝ^{n×m}
/// A = φ.(Z)
/// ```
///
/// O `1ₙ b` é o broadcasting do viés sobre as `n` linhas do lote — um produto
/// externo entre o vetor de uns e o viés, que [`Tensor::zip`] resolve como
/// repetição de um bloco contíguo em vez de materializar `1ₙ b`.
///
/// # Adjuntos
///
/// Com `δ = ∂L/∂Z = Ā ⊙ φ'.(Z)`:
///
/// ```text
/// ∂L/∂W = Xᵀ δ    ∈ ℝ^{d×m}      (matriz de covariância cruzada entrada × erro)
/// ∂L/∂b = 1ₙᵀ δ   ∈ ℝ^{1×m}      (soma do erro sobre o lote)
/// ∂L/∂X = δ Wᵀ    ∈ ℝ^{n×d}      (erro retropropagado)
/// ```
///
/// As três seguem da regra do [`Var::matmul`] (`Ā = C̄Bᵀ`, `B̄ = AᵀC̄`) mais a
/// redução do eixo difundido do viés (`ρ_{[1,m]}(δ) = 1ₙᵀδ`), e nenhuma delas é
/// codificada à mão: a fita as reconstrói a partir de `matmul` e `add`.
///
/// Note que `Xᵀδ = Σᵢ xᵢ δᵢᵀ` é a soma dos `n` produtos externos amostra a
/// amostra — um único GEMM `d×n × n×m` no lugar de `n` atualizações de posto 1.
pub struct Dense {
    w: Param,
    b: Param,
    activation: Activation,
}

impl Dense {
    /// Inicialização Glorot (Xavier) uniforme, a mesma usada por padrão no Keras:
    /// `Wᵢⱼ ~ U(−λ, λ)` com `λ = √(6/(d + m))`, logo `Var[Wᵢⱼ] = λ²/3 = 2/(d+m)`.
    ///
    /// Esse é o compromisso entre preservar a variância do sinal para frente
    /// (`Var[Z] ≈ d·Var[W]·Var[X]`, que pede `Var[W] = 1/d`) e para trás
    /// (`Var[∂L/∂X] ≈ m·Var[W]·Var[δ]`, que pede `Var[W] = 1/m`): a média
    /// harmônica dos dois. O viés começa em `b = 0`.
    pub fn new(inputs: usize, units: usize, activation: Activation, rng: &mut Rng) -> Dense {
        let limit = (6.0 / (inputs + units) as f32).sqrt();
        let w: Vec<f32> = (0..inputs * units).map(|_| rng.range(-limit, limit)).collect();
        Dense {
            w: Param::new("W", Tensor::new(&[inputs, units], w)),
            b: Param::new("b", Tensor::zeros(&[1, units])),
            activation,
        }
    }

    pub fn weights(&self) -> &Param {
        &self.w
    }

    pub fn bias(&self) -> &Param {
        &self.b
    }
}

impl Layer for Dense {
    /// `A = φ.(XW + 1ₙb)`. Um GEMM, um broadcasting e uma aplicação pontual.
    fn forward(&self, x: &Var) -> Var {
        let tape = x.tape();
        let w = self.w.watch(tape);
        let b = self.b.watch(tape);
        let z = x.matmul(&w).add(&b);
        self.activation.apply(&z)
    }

    fn params(&self) -> Vec<Param> {
        vec![self.w.clone(), self.b.clone()]
    }

    fn describe(&self) -> String {
        let s = self.w.shape();
        format!("Dense({} -> {}, {})", s[0], s[1], self.activation.name())
    }
}

/// Pilha sequencial de camadas.
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential { layers: Vec::new() }
    }

    pub fn add(mut self, layer: impl Layer + 'static) -> Sequential {
        self.layers.push(Box::new(layer));
        self
    }

    /// Composição `A^K = f_K ∘ … ∘ f_1 (X)`. O backward correspondente é o
    /// produto das transpostas das jacobianas na ordem inversa,
    /// `J_1ᵀ … J_Kᵀ ḡ`, montado automaticamente pela fita.
    pub fn forward(&self, x: &Var) -> Var {
        self.layers.iter().fold(x.clone(), |acc, l| l.forward(&acc))
    }

    /// Passada à frente sem gravar fita — usada para avaliar.
    pub fn predict(&self, x: &Tensor) -> Tensor {
        let tape = crate::tape::Tape::new();
        let input = crate::tape::constant(&tape, x.clone());
        self.forward(&input).value().clone()
    }

    pub fn params(&self) -> Vec<Param> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    pub fn num_params(&self) -> usize {
        self.params().iter().map(|p| p.value().len()).sum()
    }

    /// Resumo do modelo, no espírito de `model.summary()`.
    pub fn summary(&self) -> String {
        let mut out = String::from("Sequential\n");
        for (i, l) in self.layers.iter().enumerate() {
            out.push_str(&format!("  [{i}] {}\n", l.describe()));
        }
        out.push_str(&format!("  total de parâmetros: {}\n", self.num_params()));
        out
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Sequential::new()
    }
}
