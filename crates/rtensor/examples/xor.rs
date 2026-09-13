//! O "olá mundo" das redes neurais: XOR, que nenhum modelo linear resolve.
//!
//! Execute com: cargo run --release --example xor

use rtensor::prelude::*;

fn main() {
    let mut rng = Rng::new(2024);
    let model = Sequential::new()
        .add(Dense::new(2, 8, Activation::Tanh, &mut rng))
        .add(Dense::new(8, 1, Activation::Sigmoid, &mut rng));

    print!("{}", model.summary());

    let x = Tensor::new(&[4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]);
    let y = Tensor::new(&[4, 1], vec![0., 1., 1., 0.]);

    let params = model.params();
    let mut opt = Adam::new(0.08);

    println!("\népoca      perda");
    for epoca in 1..=1500 {
        // Uma fita nova por passo, como um `with tf.GradientTape()`.
        let tape = Tape::new();
        let entrada = constant(&tape, x.clone());
        let saida = model.forward(&entrada);
        let perda = losses::binary_cross_entropy(&saida, &y);

        let grads = perda.backward();
        opt.step(&params, &grads);

        if epoca % 150 == 0 || epoca == 1 {
            println!("{epoca:>5}   {:.6}", perda.value().item());
        }
    }

    println!("\n  a    b   esperado   previsto");
    let p = model.predict(&x);
    for i in 0..4 {
        println!(
            "  {}    {}      {}       {:.4}",
            x.data()[i * 2],
            x.data()[i * 2 + 1],
            y.data()[i],
            p.data()[i]
        );
    }
}
