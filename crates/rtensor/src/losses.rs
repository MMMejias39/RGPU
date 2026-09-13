//! Funções de perda.
//!
//! # Convenções
//!
//! | Símbolo | Significado |
//! |---|---|
//! | `Z ∈ ℝ^{n×c}` | logits (saída linear da última camada) |
//! | `P = softmax_linhas(Z) ∈ ℝ^{n×c}` | probabilidades previstas |
//! | `Y ∈ {0,1}^{n×c}` | rótulos one-hot, `Y1_c = 1ₙ` |
//! | `Ŷ, T` | previsão e alvo em perdas de regressão |
//! | `n` | tamanho do lote; `N = |·|` o número total de elementos |
//! | `⊙`, `⊘` | Hadamard e divisão elemento a elemento |
//! | `1ₙ, 1_c` | vetores coluna de uns |
//!
//! Toda perda devolve um **escalar**, e o adjunto que a fita entrega a ela é o
//! escalar `ḡ = ∂L/∂perda` (igual a `1` quando a perda é a raiz do grafo, mas
//! `scale`/`offset` a montante podem mudá-lo). Por isso cada regra abaixo
//! aparece multiplicada por `ḡ`.

use crate::tape::{constant, Var};
use crate::tensor::Tensor;

/// Erro quadrático médio: `L = (1/N) ‖Ŷ − T‖_F²`, com `N` o número de elementos.
///
/// Composto a partir das operações básicas — o autodiff deriva sozinho, e a
/// composição `sub → square → mean` reproduz
///
/// ```text
/// ∂L/∂Ŷ = (2/N) (Ŷ − T)
/// ```
pub fn mse(pred: &Var, target: &Tensor) -> Var {
    let t = constant(pred.tape(), target.clone());
    pred.sub(&t).square().mean()
}

/// Erro absoluto médio: `L = (1/N) ‖Ŷ − T‖₁ = (1/N) 1ᵀ|Ŷ − T|`.
///
/// Não é diferenciável em `Ŷ = T`; usa-se o subgradiente `sign(0) = +1`:
///
/// ```text
/// ∂L/∂Ŷ = (ḡ/N) · sign.(Ŷ − T)
/// ```
///
/// Registrado como nó único porque o sinal é constante em relação ao adjunto —
/// não há nada a recalcular no backward além de um escalamento.
pub fn mae(pred: &Var, target: &Tensor) -> Var {
    let t = constant(pred.tape(), target.clone());
    let d = pred.sub(&t);
    let sign = d.value().map(|v| if v >= 0.0 { 1.0 } else { -1.0 });
    let value = Tensor::scalar(d.value().map(f32::abs).mean_all());
    let n = d.value().len() as f32;
    d.unary(value, move |g| sign.scale(g.item() / n))
}

/// Entropia cruzada categórica calculada direto dos logits.
///
/// `logits` tem shape `[batch, classes]` e `targets` é one-hot do mesmo shape.
///
/// # Forward
///
/// ```text
/// P = softmax_linhas(Z),   pᵢⱼ = exp(zᵢⱼ − mᵢ) / Σ_l exp(zᵢₗ − mᵢ),  mᵢ = max_l zᵢₗ
/// L = −(1/n) 1ₙᵀ (Y ⊙ ln P) 1_c = −(1/n) Σᵢ ln p_{i, y(i)}
/// ```
///
/// # Adjunto: por que a composição direta cancela o termo mal condicionado
///
/// Fosse montada peça por peça (`softmax` → `ln` → `mul` → `sum`), a cadeia
/// passaria por dois fatores que quase se cancelam:
///
/// ```text
/// ∂L/∂P = −(1/n) Y ⊘ P          (explode quando pᵢⱼ → 0)
/// ∂L/∂Z = Jᵀ(∂L/∂P),  J = diag(p) − ppᵀ
/// ```
///
/// Aplicando `Jᵀg = p ⊙ (g − (gᵀp)1)` a `g = −(1/n) y ⊘ p`:
///
/// ```text
/// p ⊙ g      = −(1/n) p ⊙ (y ⊘ p) = −(1/n) y          (o p cancela exatamente)
/// gᵀp        = −(1/n) (y ⊘ p)ᵀ p  = −(1/n) 1ᵀy = −1/n (pois Y é one-hot)
/// ⇒ Jᵀg      = −(1/n) y + (1/n) p = (1/n)(p − y)
/// ```
///
/// ```text
/// ∂L/∂Z = (P − Y) / n
/// ```
///
/// O `1/p` do `ln` e o `p` da jacobiana do softmax se anulam **algebricamente**.
/// Feito em ponto flutuante, porém, o produto `p · (1/p)` para `p ~ 1e-30` passa
/// por um overflow intermediário (ou por um piso artificial como o `max(1e-12)`
/// do [`Var::ln`]) e o cancelamento entre dois números enormes perde todos os
/// dígitos significativos. Fundindo as duas etapas num nó só, o cancelamento
/// acontece na derivação, não na aritmética: o resultado `(P − Y)/n` é limitado
/// por `1/n` em módulo, sempre bem condicionado, e `∂L/∂Z` fica correto mesmo
/// para logits saturados.
///
/// O adjunto também é vetorizado de graça: `(P − Y)/n` é uma subtração de
/// tensores de shape idêntico seguida de um escalamento — um único `axpy`.
pub fn softmax_cross_entropy(logits: &Var, targets: &Tensor) -> Var {
    assert_eq!(logits.shape(), targets.shape(), "logits e rótulos com shapes diferentes");
    let probs = logits.value().softmax_rows();
    let batch = logits.shape()[0] as f32;

    // L = −(1/n) Σ yᵢⱼ ln pᵢⱼ, varrendo os dois buffers contíguos em paralelo.
    let mut total = 0.0f32;
    for (p, y) in probs.data().iter().zip(targets.data()) {
        if *y != 0.0 {
            total -= y * p.max(1e-12).ln();
        }
    }
    let value = Tensor::scalar(total / batch);

    let y = targets.clone();
    logits.unary(value, move |g| probs.sub(&y).scale(g.item() / batch))
}

/// Entropia cruzada binária a partir de probabilidades em (0, 1).
///
/// # Forward
///
/// ```text
/// L = −(1/N) 1ᵀ [ Y ⊙ ln P̃ + (1 − Y) ⊙ ln(1 − P̃) ],   P̃ = clamp(P, ε, 1−ε)
/// ```
///
/// # Adjunto
///
/// Derivando termo a termo, `∂/∂p [−y ln p − (1−y) ln(1−p)] = (p − y)/(p(1−p))`:
///
/// ```text
/// ∂L/∂P = (ḡ/N) · (P̃ − Y) ⊘ (P̃ ⊙ (1 − P̃))
/// ```
///
/// Ao contrário do caso softmax, aqui a entrada **já é** uma probabilidade:
/// não há sigmoide para fundir, então o fator `1/(p(1−p))` permanece explícito.
/// É ele quem exige o `clamp` em `[ε, 1−ε]` — sem o piso, `p = 0` ou `p = 1`
/// daria divisão por zero. (Compondo `sigmoid → binary_cross_entropy`, o
/// `σ(1−σ)` do backward da sigmoide cancela esse denominador e sobra
/// `∂L/∂Z = (P − Y)/N`, o análogo binário de `(P − Y)/n`; o cancelamento é feito
/// pela fita, com o `clamp` garantindo que nenhum dos dois fatores estoure.)
///
/// O `zip` do backward opera sobre dois tensores de shape idêntico, ou seja,
/// cai no caminho contíguo de [`Tensor::zip`].
pub fn binary_cross_entropy(probs: &Var, targets: &Tensor) -> Var {
    let p = probs.value().clone();
    let y = targets.clone();
    let batch = p.len() as f32;

    let mut total = 0.0f32;
    for (pi, yi) in p.data().iter().zip(y.data()) {
        let c = pi.clamp(1e-7, 1.0 - 1e-7);
        total -= yi * c.ln() + (1.0 - yi) * (1.0 - c).ln();
    }
    let value = Tensor::scalar(total / batch);

    probs.unary(value, move |g| {
        let d = p.zip(&y, |pi, yi| {
            let c = pi.clamp(1e-7, 1.0 - 1e-7);
            (c - yi) / (c * (1.0 - c))
        });
        d.scale(g.item() / batch)
    })
}

/// Converte rótulos inteiros em uma matriz one-hot `Y ∈ {0,1}^{n×c}`,
/// `Y[i, ℓᵢ] = 1`. É a matriz de seleção que satisfaz `Y1_c = 1ₙ`.
///
/// O buffer sai zerado de uma vez e só as `n` posições marcadas são escritas —
/// `O(n)` escritas num buffer de `nc` posições, em vez de um laço duplo.
pub fn one_hot(labels: &[usize], classes: usize) -> Tensor {
    let mut data = vec![0.0f32; labels.len() * classes];
    for (i, &l) in labels.iter().enumerate() {
        data[i * classes + l] = 1.0;
    }
    Tensor::new(&[labels.len(), classes], data)
}

/// Fração de acertos: `(1/n) Σᵢ 1[argmaxⱼ Z[i,j] = ℓᵢ]`.
///
/// O `argmax` por linha é invariante à monotonia do softmax, então pode ser
/// tomado direto dos logits — não é preciso normalizar.
pub fn accuracy(logits: &Tensor, labels: &[usize]) -> f32 {
    let pred = logits.argmax_rows();
    let hits = pred.iter().zip(labels).filter(|(a, b)| a == b).count();
    hits as f32 / labels.len() as f32
}
