//! Tensor denso N-dimensional em `f32`, com broadcasting no estilo NumPy/TensorFlow.
//!
//! # Convenções de notação
//!
//! Estas convenções valem para todos os módulos da crate:
//!
//! | Símbolo | Significado |
//! |---|---|
//! | `X ∈ ℝ^{n×d}` | lote de entrada: `n` amostras em linha, `d` atributos |
//! | `W ∈ ℝ^{d×m}` | matriz de pesos de uma camada densa |
//! | `b ∈ ℝ^{1×m}` | vetor-linha de viés |
//! | `Z = XW + 1ₙb ∈ ℝ^{n×m}` | pré-ativação |
//! | `1ₙ ∈ ℝ^{n×1}` | vetor coluna de uns |
//! | `δ = ∂L/∂Z` | adjunto (gradiente da perda em relação a `Z`) |
//! | `Ā` | notação curta para `∂L/∂A` |
//! | `⊙` | produto de Hadamard (elemento a elemento) |
//! | `Jᵀ` | transposta da jacobiana da operação |
//! | `L` | perda escalar final |
//!
//! **Convenção de denominador (layout de denominador).** Todo gradiente tem
//! exatamente o mesmo shape do objeto que o gerou: `∂L/∂W ∈ ℝ^{d×m}`,
//! `∂L/∂b ∈ ℝ^{1×m}`, `∂L/∂X ∈ ℝ^{n×d}`. Nunca se materializa uma jacobiana
//! completa; o backward é sempre o produto vetor-jacobiana `g ↦ Jᵀg`.
//!
//! # Álgebra do broadcasting
//!
//! Difundir é multiplicar por uma matriz de uns. Para `A ∈ ℝ^{1×m}` difundido a
//! `ℝ^{n×m}`, o forward é `B = 1ₙA` e portanto o adjunto é a contração
//!
//! ```text
//! Ā = 1ₙᵀ B̄ = Σᵢ B̄[i, :]        (soma sobre o eixo difundido)
//! ```
//!
//! Isso é exatamente o que [`Tensor::reduce_to`] faz: para cada eixo em que o
//! operando tinha extensão 1 e a saída tem extensão > 1, contrai com `1ᵀ(·)`.
//! Difundir e reduzir são adjuntos um do outro.
//!
//! # Layout de memória
//!
//! Os dados são `row-major` (o último eixo é o contíguo). As rotinas abaixo
//! detectam os casos em que o padrão de acesso já é contíguo — shapes iguais,
//! escalar, ou difusão só nos eixos à esquerda — e caem em caminhos rápidos
//! sobre slices, que o compilador vetoriza e onde não há checagem de limites.
//! O laço genérico com aritmética de strides sobrevive apenas como fallback.

use std::fmt;

#[derive(Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: Vec<f32>) -> Tensor {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "shape {:?} exige {} elementos, recebidos {}",
            shape,
            n,
            data.len()
        );
        Tensor { shape: shape.to_vec(), data }
    }

    pub fn scalar(v: f32) -> Tensor {
        Tensor { shape: vec![], data: vec![v] }
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    pub fn full(shape: &[usize], v: f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: vec![v; n] }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.data.len() == 1
    }

    /// Valor de um tensor de um único elemento.
    pub fn item(&self) -> f32 {
        assert!(self.is_scalar(), "item() exige tensor unitário, shape {:?}", self.shape);
        self.data[0]
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        Tensor::new(shape, self.data.clone())
    }

    /// Transposta de uma matriz 2-D: `A ∈ ℝ^{r×c} ↦ Aᵀ ∈ ℝ^{c×r}`, `Aᵀ[j,i] = A[i,j]`.
    ///
    /// Como adjunto, a transposição é sua própria inversa: se `B = Aᵀ` então
    /// `Ā = B̄ᵀ`.
    ///
    /// Implementada por blocos `T×T`: percorrer a matriz linearmente faria uma
    /// leitura contígua contra uma escrita com passo `r` (uma linha de cache
    /// jogada fora por elemento). Trabalhando em ladrilhos que cabem no L1,
    /// cada linha de cache carregada é usada `T` vezes dos dois lados.
    pub fn t(&self) -> Tensor {
        assert_eq!(self.rank(), 2, "t() exige rank 2, recebido {:?}", self.shape);
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; self.data.len()];

        const T: usize = 32;
        let mut i0 = 0;
        while i0 < r {
            let i1 = (i0 + T).min(r);
            let mut j0 = 0;
            while j0 < c {
                let j1 = (j0 + T).min(c);
                for i in i0..i1 {
                    let src = &self.data[i * c + j0..i * c + j1];
                    for (t, &v) in src.iter().enumerate() {
                        out[(j0 + t) * r + i] = v;
                    }
                }
                j0 = j1;
            }
            i0 = i1;
        }
        Tensor::new(&[c, r], out)
    }

    /// Aplicação pontual `f` a cada elemento: `Y = f.(X)`, `Y[i] = f(X[i])`.
    ///
    /// Adjunto (para `f` diferenciável): `X̄ = Ȳ ⊙ f'.(X)`.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|&x| f(x)).collect() }
    }

    /// Combina dois tensores elemento a elemento aplicando broadcasting:
    /// `C[i] = f(A[σ_A(i)], B[σ_B(i)])`, com `σ` o mapa de índices do broadcasting.
    ///
    /// # Caminhos rápidos
    ///
    /// O laço genérico monta um índice multidimensional e recalcula
    /// `Σ_ax idx[ax]·stride[ax]` por elemento — `O(rank)` multiplicações por
    /// saída. Mas os casos que dominam o uso real são todos contíguos:
    ///
    /// - `A` e `B` com o mesmo shape (`δ ⊙ M`, `G + G'`): um `zip` de slices;
    /// - um dos lados escalar (`G·k`): um `map`;
    /// - difusão só nos eixos à esquerda (`X + 1ₙb`, com `b ∈ ℝ^{1×m}`): o
    ///   operando menor é um bloco contíguo de período `m` que se repete `n`
    ///   vezes, então basta um laço sobre `chunks_exact(m)`;
    /// - difusão no último eixo em rank 2 (`X ⊘ (s 1ᵀ)`, com `s ∈ ℝ^{n×1}`):
    ///   um escalar por linha.
    ///
    /// Esses quatro casos cobrem toda a fita de autodiff; o fallback genérico
    /// só entra em formas exóticas.
    pub fn zip(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
        let shape = broadcast_shapes(&self.shape, &other.shape);
        let n: usize = shape.iter().product();

        // --- Caminho 1: ambos os operandos são blocos contíguos repetidos. ---
        // Se os dois são "tiled", o de menor prefixo difundido cobre todos os
        // eixos e tem período exatamente `n` (ver `tiled_period`).
        if let (Some(ka), Some(kb)) = (tiled_period(&self.shape, &shape), tiled_period(&other.shape, &shape))
        {
            let mut data = Vec::with_capacity(n);
            if ka == n && kb == n {
                // Mesmo shape efetivo: C = f.(A, B) sobre slices alinhados.
                data.extend(self.data.iter().zip(&other.data).map(|(&a, &b)| f(a, b)));
            } else if ka == n {
                // B difundido nos eixos à esquerda: repete `other.data` (len kb).
                for blk in self.data.chunks_exact(kb) {
                    data.extend(blk.iter().zip(&other.data).map(|(&a, &b)| f(a, b)));
                }
            } else if kb == n {
                // A difundido nos eixos à esquerda: repete `self.data` (len ka).
                for blk in other.data.chunks_exact(ka) {
                    data.extend(self.data.iter().zip(blk).map(|(&a, &b)| f(a, b)));
                }
            } else {
                // Ambos menores que n: só ocorre se n == 1 (ka == kb == 1).
                for i in 0..n {
                    data.push(f(self.data[i % ka], other.data[i % kb]));
                }
            }
            return Tensor { shape, data };
        }

        // --- Caminho 2: difusão do último eixo em rank 2 (coluna contra matriz). ---
        if shape.len() == 2 {
            let (r, c) = (shape[0], shape[1]);
            if self.shape == shape && is_column(&other.shape, r) {
                let mut data = Vec::with_capacity(n);
                for (row, &s) in self.data.chunks_exact(c).zip(&other.data) {
                    data.extend(row.iter().map(|&a| f(a, s)));
                }
                return Tensor { shape, data };
            }
            if other.shape == shape && is_column(&self.shape, r) {
                let mut data = Vec::with_capacity(n);
                for (row, &s) in other.data.chunks_exact(c).zip(&self.data) {
                    data.extend(row.iter().map(|&b| f(s, b)));
                }
                return Tensor { shape, data };
            }
        }

        // --- Fallback genérico: aritmética de strides por elemento. ---
        let sa = broadcast_strides(&self.shape, &shape);
        let sb = broadcast_strides(&other.shape, &shape);
        let mut data = Vec::with_capacity(n);
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..n {
            let ia: usize = idx.iter().zip(&sa).map(|(i, s)| i * s).sum();
            let ib: usize = idx.iter().zip(&sb).map(|(i, s)| i * s).sum();
            data.push(f(self.data[ia], other.data[ib]));
            bump(&mut idx, &shape);
        }
        Tensor { shape, data }
    }

    /// `C = A + B` (com broadcasting). Adjuntos: `Ā = reduce(C̄)`, `B̄ = reduce(C̄)`.
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.zip(other, |a, b| a + b)
    }

    /// `C = A − B`. Adjuntos: `Ā = reduce(C̄)`, `B̄ = reduce(−C̄)`.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.zip(other, |a, b| a - b)
    }

    /// `C = A ⊙ B`. Adjuntos: `Ā = reduce(C̄ ⊙ B)`, `B̄ = reduce(C̄ ⊙ A)`.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.zip(other, |a, b| a * b)
    }

    /// `C = A ⊘ B`. Adjuntos: `Ā = reduce(C̄ ⊘ B)`, `B̄ = reduce(−C̄ ⊙ A ⊘ B⊙B)`.
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.zip(other, |a, b| a / b)
    }

    /// `Y = kX` (escalamento). Adjunto: `X̄ = kȲ`.
    pub fn scale(&self, k: f32) -> Tensor {
        self.map(|x| x * k)
    }

    /// `Y = −X`. Adjunto: `X̄ = −Ȳ`.
    pub fn neg(&self) -> Tensor {
        self.map(|x| -x)
    }

    /// `s = 1ᵀ vec(X)`. Adjunto: `X̄ = s̄ · 1` (o escalar espalhado no shape de `X`).
    pub fn sum_all(&self) -> f32 {
        self.data.iter().sum()
    }

    /// `s = (1/N) 1ᵀ vec(X)` com `N = |X|`. Adjunto: `X̄ = (s̄/N) · 1`.
    pub fn mean_all(&self) -> f32 {
        self.sum_all() / self.data.len() as f32
    }

    pub fn max_all(&self) -> f32 {
        self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Produto matricial 2-D: `C = AB`, com `A ∈ ℝ^{m×k}`, `B ∈ ℝ^{k×n}`,
    /// `C[i,j] = Σ_p A[i,p] B[p,j]`.
    ///
    /// # Adjuntos
    ///
    /// De `dL = ⟨C̄, dA·B + A·dB⟩ = ⟨C̄Bᵀ, dA⟩ + ⟨AᵀC̄, dB⟩` vem
    ///
    /// ```text
    /// Ā = C̄ Bᵀ ∈ ℝ^{m×k}
    /// B̄ = Aᵀ C̄ ∈ ℝ^{k×n}
    /// ```
    ///
    /// Ambos já saem no shape correto pela convenção de denominador — é a regra
    /// mnemônica "transpõe o outro fator e põe do lado que fecha as dimensões".
    ///
    /// # Ordem dos laços
    ///
    /// A forma usada é `ikj` (um `axpy` por elemento de `A`): fixado `A[i,p]`,
    /// acumula-se `C[i,:] ← C[i,:] + A[i,p]·B[p,:]`. Tanto a leitura de `B[p,:]`
    /// quanto a escrita em `C[i,:]` são contíguas, ao contrário do `ijk` clássico,
    /// que lê `B` com passo `n`. Os laços usam iteradores sobre `chunks` e `zip`,
    /// o que elimina a checagem de limites do miolo.
    ///
    /// Para matrizes grandes (`B` acima de ~128 KB) há blocagem em painéis de
    /// colunas de `B`/`C` e de profundidade em `p`, de modo que o painel ativo
    /// permaneça em cache ao longo de todas as `m` linhas.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rank(), 2, "matmul exige rank 2, recebido {:?}", self.shape);
        assert_eq!(other.rank(), 2, "matmul exige rank 2, recebido {:?}", other.shape);
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "dimensões incompatíveis: {:?} x {:?}", self.shape, other.shape);

        let mut out = vec![0.0f32; m * n];
        if m == 0 || n == 0 || k == 0 {
            return Tensor::new(&[m, n], out);
        }

        // Painel de B pequeno: uma única varredura ikj, sem custo de blocagem.
        if k * n <= 32_768 {
            for (dst, a_row) in out.chunks_exact_mut(n).zip(self.data.chunks_exact(k)) {
                for (&a, b_row) in a_row.iter().zip(other.data.chunks_exact(n)) {
                    if a == 0.0 {
                        continue;
                    }
                    for (d, &b) in dst.iter_mut().zip(b_row) {
                        *d += a * b;
                    }
                }
            }
            return Tensor::new(&[m, n], out);
        }

        // Blocagem: (painel de colunas jc) × (profundidade pc) fica residente.
        const JC: usize = 256;
        const PC: usize = 128;
        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + JC).min(n);
            let mut p0 = 0;
            while p0 < k {
                let p1 = (p0 + PC).min(k);
                for i in 0..m {
                    let a_row = &self.data[i * k + p0..i * k + p1];
                    let dst = &mut out[i * n + j0..i * n + j1];
                    for (t, &a) in a_row.iter().enumerate() {
                        if a == 0.0 {
                            continue;
                        }
                        let p = p0 + t;
                        let b_row = &other.data[p * n + j0..p * n + j1];
                        for (d, &b) in dst.iter_mut().zip(b_row) {
                            *d += a * b;
                        }
                    }
                }
                p0 = p1;
            }
            j0 = j1;
        }
        Tensor::new(&[m, n], out)
    }

    /// Soma o tensor de volta ao shape `target`, desfazendo um broadcasting.
    ///
    /// # Álgebra
    ///
    /// Difundir é multiplicar por uma matriz de uns; reduzir é o adjunto dessa
    /// multiplicação, ou seja, a contração `1ᵀ(·)` sobre cada eixo difundido:
    ///
    /// ```text
    /// [n, m] → [1, m]   Ḡ = 1ₙᵀ G          (soma das linhas; gradiente do viés b)
    /// [n, m] → [n, 1]   Ḡ = G 1ₘ           (soma das colunas)
    /// [n, m] → []       Ḡ = 1ₙᵀ G 1ₘ       (soma total)
    /// ```
    ///
    /// É o passo que torna o gradiente de um viés `[1, m]` correto num lote
    /// `[n, m]`: `∂L/∂b = 1ₙᵀδ`.
    ///
    /// # Caminhos rápidos
    ///
    /// - `target` já igual ao shape: identidade;
    /// - `target` unitário: `sum_all`;
    /// - eixos difundidos todos à esquerda (`[n,m] → [1,m]`, `[n,m] → [m]`): a
    ///   saída é um acumulador contíguo de comprimento `m` varrido `n` vezes,
    ///   `out += G[i, :]` — um `axpy` por linha;
    /// - último eixo difundido em rank 2 (`[n,m] → [n,1]`): soma de cada linha.
    pub fn reduce_to(&self, target: &[usize]) -> Tensor {
        if self.shape == target {
            return self.clone();
        }
        let n: usize = target.iter().product();

        // Redução total: 1ᵀ vec(G).
        if n == 1 {
            return Tensor::new(target, vec![self.sum_all()]);
        }

        // Eixos difundidos à esquerda: out[j] = Σ_blocos G[bloco·k + j].
        if let Some(k) = tiled_period(target, &self.shape) {
            let mut out = vec![0.0f32; k];
            for blk in self.data.chunks_exact(k) {
                for (o, &g) in out.iter_mut().zip(blk) {
                    *o += g;
                }
            }
            return Tensor::new(target, out);
        }

        // Último eixo difundido em rank 2: out[i] = Σ_j G[i, j].
        if self.shape.len() == 2 && is_column(target, self.shape[0]) {
            let c = self.shape[1];
            let out: Vec<f32> = self.data.chunks_exact(c).map(|row| row.iter().sum()).collect();
            return Tensor::new(target, out);
        }

        // Fallback genérico.
        let mut out = vec![0.0f32; n];
        let st = broadcast_strides(target, &self.shape);
        let mut idx = vec![0usize; self.shape.len()];
        for lin in 0..self.data.len() {
            let it: usize = idx.iter().zip(&st).map(|(i, s)| i * s).sum();
            out[it] += self.data[lin];
            bump(&mut idx, &self.shape);
        }
        Tensor::new(target, out)
    }

    /// Softmax por linha de uma matriz `Z ∈ ℝ^{n×c}`, estabilizado pelo máximo.
    ///
    /// ```text
    /// sᵢ = softmax(zᵢ),  sᵢⱼ = exp(zᵢⱼ − max_l zᵢₗ) / Σ_l exp(zᵢₗ − max_l zᵢₗ)
    /// ```
    ///
    /// Subtrair o máximo é a invariância `softmax(z + c1) = softmax(z)`: não
    /// muda o resultado exato e evita `exp` de argumento grande (overflow) —
    /// com o deslocamento, o maior expoente é sempre `exp(0) = 1`.
    ///
    /// A jacobiana de uma linha é `J = diag(s) − ssᵀ` (simétrica), logo
    /// `Jᵀg = Jg = s ⊙ (g − (gᵀs)1)` — ver [`crate::ops`].
    ///
    /// Cada linha é um slice contíguo de `c` elementos, percorrido três vezes
    /// (máximo, `exp` + soma, normalização) sem índice explícito.
    pub fn softmax_rows(&self) -> Tensor {
        assert_eq!(self.rank(), 2, "softmax_rows exige rank 2");
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; r * c];
        for (dst, row) in out.chunks_exact_mut(c.max(1)).zip(self.data.chunks_exact(c.max(1))) {
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (d, &z) in dst.iter_mut().zip(row) {
                let e = (z - mx).exp();
                *d = e;
                sum += e;
            }
            for d in dst.iter_mut() {
                *d /= sum;
            }
        }
        Tensor::new(&[r, c], out)
    }

    /// Índice do maior valor de cada linha: `argmax_j Z[i, j]` (primeiro em caso
    /// de empate). Varredura contígua por linha via `chunks_exact`.
    pub fn argmax_rows(&self) -> Vec<usize> {
        assert_eq!(self.rank(), 2, "argmax_rows exige rank 2");
        let c = self.shape[1];
        self.data
            .chunks_exact(c.max(1))
            .map(|row| {
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (j, &v) in row.iter().enumerate() {
                    if j == 0 || v > bv {
                        bv = v;
                        best = j;
                    }
                }
                best
            })
            .collect()
    }

    /// Recorta as linhas `[start, end)` — usado para montar mini-lotes.
    /// É `X[start:end, :]`, uma fatia já contígua em row-major.
    pub fn rows(&self, start: usize, end: usize) -> Tensor {
        assert_eq!(self.rank(), 2, "rows() exige rank 2");
        let c = self.shape[1];
        Tensor::new(&[end - start, c], self.data[start * c..end * c].to_vec())
    }

    /// Reordena as linhas segundo `order` — usado para embaralhar o dataset.
    ///
    /// Algebricamente é `PX`, com `P` a matriz de permutação/seleção de linhas
    /// `P[t, order[t]] = 1`. Em vez de um produto `O(n²c)`, cada linha é uma
    /// cópia contígua de `c` elementos: o buffer é alocado de uma vez e
    /// preenchido com `copy_from_slice` (um `memcpy` por linha), sem o teste de
    /// capacidade que um `extend` faria a cada linha.
    pub fn take_rows(&self, order: &[usize]) -> Tensor {
        assert_eq!(self.rank(), 2, "take_rows() exige rank 2");
        let c = self.shape[1];
        let mut out = vec![0.0f32; order.len() * c];
        for (dst, &i) in out.chunks_exact_mut(c.max(1)).zip(order) {
            dst.copy_from_slice(&self.data[i * c..i * c + c]);
        }
        Tensor::new(&[order.len(), c], out)
    }
}

/// Avança um índice multidimensional em uma posição (ordem row-major).
fn bump(idx: &mut [usize], shape: &[usize]) {
    for ax in (0..shape.len()).rev() {
        idx[ax] += 1;
        if idx[ax] < shape[ax] {
            return;
        }
        idx[ax] = 0;
    }
}

/// `true` se `from`, alinhado à direita com um eixo 0 de extensão `r`, é uma
/// coluna `ℝ^{r×1}` (ou `ℝ^{r}` degenerado não conta: precisa do eixo unitário).
fn is_column(from: &[usize], r: usize) -> bool {
    from.len() == 2 && from[0] == r && from[1] == 1
}

/// Período do bloco contíguo, se difundir `from` para `to` equivale a repetir o
/// buffer de `from` inteiro e sem saltos.
///
/// Isso acontece exatamente quando todos os eixos difundidos (extensão 1 em
/// `from`, > 1 em `to`) estão *à esquerda* dos eixos preservados — o caso de
/// `b ∈ ℝ^{1×m}` somado a `X ∈ ℝ^{n×m}`, em que `1ₙb` é o buffer de `b`
/// repetido `n` vezes. Devolve o período (`= |from|`) ou `None`.
fn tiled_period(from: &[usize], to: &[usize]) -> Option<usize> {
    if from.len() > to.len() {
        return None;
    }
    let k: usize = from.iter().product();
    let st = broadcast_strides(from, to);
    let mut cum = 1usize;
    for i in (0..to.len()).rev() {
        if to[i] == 1 {
            continue;
        }
        if cum < k {
            // Eixo ainda dentro do bloco contíguo: stride tem de casar.
            if st[i] != cum {
                return None;
            }
            cum *= to[i];
        } else if st[i] != 0 {
            // Fora do bloco só se admite eixo difundido (stride 0).
            return None;
        }
    }
    if cum == k {
        Some(k)
    } else {
        None
    }
}

/// Shape resultante do broadcasting de `a` com `b`, alinhando pela direita.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            panic!("shapes incompatíveis para broadcasting: {:?} e {:?}", a, b)
        };
    }
    out
}

/// Strides de `from` vistas no espaço de índices de `to` (0 nos eixos difundidos).
fn broadcast_strides(from: &[usize], to: &[usize]) -> Vec<usize> {
    let mut base = vec![0usize; from.len()];
    let mut acc = 1;
    for i in (0..from.len()).rev() {
        base[i] = acc;
        acc *= from[i];
    }
    let off = to.len() - from.len();
    let mut out = vec![0usize; to.len()];
    for i in 0..to.len() {
        if i < off {
            out[i] = 0;
        } else {
            let d = from[i - off];
            out[i] = if d == 1 && to[i] != 1 { 0 } else { base[i - off] };
        }
    }
    out
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data=", self.shape)?;
        if self.data.len() <= 12 {
            write!(f, "{:?})", self.data)
        } else {
            write!(f, "[{:?}, ... {} valores])", &self.data[..6], self.data.len())
        }
    }
}
