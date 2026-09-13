//! Diferenciação automática em modo reverso — o equivalente do `tf.GradientTape`.
//!
//! Cada operação registra um nó na fita (`Tape`) contendo seus pais e uma closure
//! que converte o gradiente da saída nos gradientes das entradas. Como os nós são
//! criados em ordem topológica, o backward é uma única varredura de trás para frente.
//!
//! # Formulação
//!
//! Seja o grafo `v_1, …, v_N` em ordem topológica, `v_N = L` escalar, e
//! `v_k = f_k(v_{pais(k)})`. O modo reverso propaga os adjuntos
//! `v̄_k = ∂L/∂v_k` (mesmo shape de `v_k`, convenção de denominador) por
//!
//! ```text
//! v̄_N = 1                                    (semente escalar)
//! v̄_j = Σ_{k : j ∈ pais(k)}  J_{k→j}ᵀ v̄_k     (regra da cadeia + acumulação)
//! ```
//!
//! A closure de cada nó é exatamente a aplicação `v̄_k ↦ J_{k→j}ᵀ v̄_k` — nunca
//! se constrói a jacobiana `J`, só o produto vetor-jacobiana. A soma sobre `k`
//! é a acumulação de gradiente quando uma variável é reutilizada (um "fan-out"
//! no grafo vira uma soma no adjunto: difusão e soma são adjuntas).
//!
//! Como `pais(k) ⊂ {1, …, k−1}` por construção, varrer `k = N, …, 1` garante que
//! `v̄_k` já esteja completo quando o nó `k` é processado. Essa mesma invariante
//! permite fatiar o vetor de adjuntos em `[0, k)` (mutável, os pais) e `[k, …)`
//! (só leitura, o adjunto corrente) — é o que evita clonar `v̄_k` a cada nó.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::tensor::Tensor;

type Backward = Box<dyn Fn(&Tensor) -> Vec<Tensor>>;

struct Node {
    parents: Vec<usize>,
    backward: Option<Backward>,
}

/// Fita de gravação. Criada a cada passo de treino e descartada depois.
pub struct Tape {
    nodes: RefCell<Vec<Node>>,
    /// Nó-folha já criado para cada parâmetro, indexado pelo id do parâmetro.
    param_nodes: RefCell<HashMap<usize, usize>>,
}

impl Tape {
    pub fn new() -> Rc<Tape> {
        Rc::new(Tape { nodes: RefCell::new(Vec::new()), param_nodes: RefCell::new(HashMap::new()) })
    }

    fn push(&self, parents: Vec<usize>, backward: Option<Backward>) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(Node { parents, backward });
        nodes.len() - 1
    }

    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Uma constante (entrada, rótulo): participa do grafo mas não recebe gradiente útil.
pub fn constant(tape: &Rc<Tape>, value: Tensor) -> Var {
    let idx = tape.push(Vec::new(), None);
    Var { tape: Rc::clone(tape), idx, value }
}

/// Peso treinável. Clonar um `Param` compartilha o mesmo buffer (como `tf.Variable`).
#[derive(Clone)]
pub struct Param {
    id: usize,
    name: String,
    value: Rc<RefCell<Tensor>>,
}

static NEXT_PARAM_ID: AtomicUsize = AtomicUsize::new(0);

impl Param {
    pub fn new(name: &str, value: Tensor) -> Param {
        Param {
            id: NEXT_PARAM_ID.fetch_add(1, Ordering::Relaxed),
            name: name.to_string(),
            value: Rc::new(RefCell::new(value)),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> std::cell::Ref<'_, Tensor> {
        self.value.borrow()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.value.borrow().shape().to_vec()
    }

    /// Aplica uma atualização in-place (usado pelos otimizadores).
    ///
    /// O limite é `FnMut` (e não `Fn`) para que um otimizador possa atualizar,
    /// na mesma passada e sem alocar nada, tanto `θ` quanto os buffers de estado
    /// que ele carrega (`m`, `v`, velocidade) — ver [`crate::optim`].
    pub fn update(&self, mut f: impl FnMut(&mut Tensor)) {
        f(&mut self.value.borrow_mut());
    }

    pub fn set(&self, t: Tensor) {
        *self.value.borrow_mut() = t;
    }

    /// Liga este parâmetro a uma fita, criando (ou reaproveitando) seu nó-folha.
    pub fn watch(&self, tape: &Rc<Tape>) -> Var {
        if let Some(&idx) = tape.param_nodes.borrow().get(&self.id) {
            return Var { tape: Rc::clone(tape), idx, value: self.value.borrow().clone() };
        }
        let idx = tape.push(Vec::new(), None);
        tape.param_nodes.borrow_mut().insert(self.id, idx);
        Var { tape: Rc::clone(tape), idx, value: self.value.borrow().clone() }
    }
}

/// Um nó do grafo: carrega o valor calculado e a posição na fita.
#[derive(Clone)]
pub struct Var {
    pub(crate) tape: Rc<Tape>,
    pub(crate) idx: usize,
    pub(crate) value: Tensor,
}

impl Var {
    pub fn value(&self) -> &Tensor {
        &self.value
    }

    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }

    pub fn tape(&self) -> &Rc<Tape> {
        &self.tape
    }

    pub(crate) fn unary(
        &self,
        value: Tensor,
        backward: impl Fn(&Tensor) -> Tensor + 'static,
    ) -> Var {
        let idx = self
            .tape
            .push(vec![self.idx], Some(Box::new(move |g| vec![backward(g)])));
        Var { tape: Rc::clone(&self.tape), idx, value }
    }

    pub(crate) fn binary(
        &self,
        other: &Var,
        value: Tensor,
        backward: impl Fn(&Tensor) -> (Tensor, Tensor) + 'static,
    ) -> Var {
        assert!(
            Rc::ptr_eq(&self.tape, &other.tape),
            "operandos pertencem a fitas diferentes"
        );
        let idx = self.tape.push(
            vec![self.idx, other.idx],
            Some(Box::new(move |g| {
                let (a, b) = backward(g);
                vec![a, b]
            })),
        );
        Var { tape: Rc::clone(&self.tape), idx, value }
    }

    /// Propaga os gradientes a partir deste nó (que precisa ser escalar).
    pub fn backward(&self) -> Grads {
        assert!(
            self.value.is_scalar(),
            "backward() precisa partir de um escalar, shape {:?}",
            self.value.shape()
        );
        let nodes = self.tape.nodes.borrow();
        let mut grads: Vec<Option<Tensor>> = vec![None; nodes.len()];
        grads[self.idx] = Some(Tensor::ones(self.value.shape()));

        for i in (0..nodes.len()).rev() {
            let node = &nodes[i];
            let backward = match &node.backward {
                Some(b) => b,
                None => continue,
            };
            // Todo pai foi criado antes do filho, logo `parent < i`: o corte em
            // `i` separa os adjuntos a escrever (pais) do adjunto a ler (`v̄_i`),
            // e a closure recebe `&v̄_i` sem cópia.
            let (ante, resto) = grads.split_at_mut(i);
            let g = match &resto[0] {
                Some(g) => g,
                None => continue,
            };
            for (parent, gp) in node.parents.iter().zip(backward(g)) {
                debug_assert!(*parent < i, "fita fora de ordem topológica");
                match &mut ante[*parent] {
                    // Acumulação `v̄_j += J^T v̄_i` in-place: os dois adjuntos têm
                    // o shape de `v_j`, então é um axpy sobre slices contíguos,
                    // sem alocar o tensor-soma.
                    Some(acc) if acc.shape() == gp.shape() => {
                        for (a, &b) in acc.data_mut().iter_mut().zip(gp.data()) {
                            *a += b;
                        }
                    }
                    Some(acc) => *acc = acc.add(&gp),
                    None => ante[*parent] = Some(gp),
                }
            }
        }

        Grads { grads, param_nodes: self.tape.param_nodes.borrow().clone() }
    }
}

/// Resultado de um backward: gradiente acumulado por nó, consultável por parâmetro.
pub struct Grads {
    grads: Vec<Option<Tensor>>,
    param_nodes: HashMap<usize, usize>,
}

impl Grads {
    /// Gradiente de um parâmetro. `None` se ele não participou deste grafo.
    pub fn of(&self, p: &Param) -> Option<&Tensor> {
        let idx = *self.param_nodes.get(&p.id())?;
        self.grads[idx].as_ref()
    }

    /// Gradiente de um nó qualquer do grafo.
    pub fn of_var(&self, v: &Var) -> Option<&Tensor> {
        self.grads[v.idx].as_ref()
    }
}
