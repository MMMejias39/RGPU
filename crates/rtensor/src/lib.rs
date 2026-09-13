//! # rtensor
//!
//! Um núcleo mínimo no estilo TensorFlow, escrito em Rust puro e sem dependências.
//!
//! As quatro peças que sustentam qualquer framework de deep learning estão aqui:
//!
//! 1. [`tensor::Tensor`] — buffer denso N-dimensional com broadcasting NumPy.
//! 2. [`tape::Tape`] — diferenciação automática reversa, análoga a `tf.GradientTape`.
//! 3. [`nn`] — camadas (`Dense`), ativações e `Sequential`, no espírito do Keras.
//! 4. [`optim`] — SGD com momentum e Adam.
//!
//! ```
//! use rtensor::prelude::*;
//!
//! let mut rng = Rng::new(7);
//! let model = Sequential::new()
//!     .add(Dense::new(2, 8, Activation::Tanh, &mut rng))
//!     .add(Dense::new(8, 1, Activation::Sigmoid, &mut rng));
//!
//! let x = Tensor::new(&[4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]);
//! let y = Tensor::new(&[4, 1], vec![0., 1., 1., 0.]);
//! let mut opt = Adam::new(0.1);
//!
//! for _ in 0..400 {
//!     let tape = Tape::new();
//!     let input = constant(&tape, x.clone());
//!     let loss = losses::mse(&model.forward(&input), &y);
//!     let grads = loss.backward();
//!     opt.step(&model.params(), &grads);
//! }
//!
//! let pred = model.predict(&x);
//! assert!(pred.data()[0] < 0.5 && pred.data()[1] > 0.5);
//! ```

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod losses;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod rng;
pub mod tape;
pub mod tensor;

pub mod prelude {
    pub use crate::losses;
    pub use crate::nn::{Activation, Dense, Layer, Sequential};
    pub use crate::optim::{Adam, Optimizer, Sgd};
    pub use crate::rng::Rng;
    pub use crate::tape::{constant, Grads, Param, Tape, Var};
    pub use crate::tensor::Tensor;
}
