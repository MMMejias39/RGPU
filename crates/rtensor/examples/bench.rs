//! Benchmark comparável com o TensorFlow: gera as espirais, exporta os dados em
//! CSV para que o outro lado treine sobre exatamente as mesmas amostras, e mede
//! apenas o laço de treino.
//!
//! Uso: cargo run --release --example bench -- <diretório_de_saída>

use std::fs;
use std::time::Instant;

use rtensor::prelude::*;

const CLASSES: usize = 3;
const POR_CLASSE: usize = 220;
const LOTE: usize = 32;
const EPOCAS: usize = 60;

fn espirais(rng: &mut Rng) -> (Tensor, Vec<usize>) {
    let n = CLASSES * POR_CLASSE;
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for c in 0..CLASSES {
        for i in 0..POR_CLASSE {
            let r = i as f32 / POR_CLASSE as f32;
            let t = c as f32 * 4.0 + r * 4.0 + rng.normal() * 0.12;
            x.push(r * (t * 1.25).sin());
            x.push(r * (t * 1.25).cos());
            y.push(c);
        }
    }
    (Tensor::new(&[n, 2], x), y)
}

fn csv(x: &Tensor, y: &[usize]) -> String {
    let c = x.shape()[1];
    let mut s = String::new();
    for (i, rotulo) in y.iter().enumerate() {
        for j in 0..c {
            s.push_str(&format!("{:.9},", x.data()[i * c + j]));
        }
        s.push_str(&format!("{rotulo}\n"));
    }
    s
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

    let mut rng = Rng::new(7);
    let (x, rotulos) = espirais(&mut rng);
    let n = rotulos.len();

    let mut ordem: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut ordem);
    let corte = n * 8 / 10;
    let (idx_treino, idx_teste) = ordem.split_at(corte);

    let x_treino = x.take_rows(idx_treino);
    let y_treino: Vec<usize> = idx_treino.iter().map(|&i| rotulos[i]).collect();
    let x_teste = x.take_rows(idx_teste);
    let y_teste: Vec<usize> = idx_teste.iter().map(|&i| rotulos[i]).collect();

    fs::write(format!("{dir}/treino.csv"), csv(&x_treino, &y_treino)).unwrap();
    fs::write(format!("{dir}/teste.csv"), csv(&x_teste, &y_teste)).unwrap();

    let model = Sequential::new()
        .add(Dense::new(2, 64, Activation::Relu, &mut rng))
        .add(Dense::new(64, 32, Activation::Relu, &mut rng))
        .add(Dense::new(32, CLASSES, Activation::Linear, &mut rng));

    let params = model.params();
    let mut opt = Adam::new(0.01);
    let mut posicoes: Vec<usize> = (0..corte).collect();

    // Só o laço de treino entra na medição.
    let inicio = Instant::now();
    for _ in 0..EPOCAS {
        rng.shuffle(&mut posicoes);
        for ini in (0..corte).step_by(LOTE) {
            let fim = (ini + LOTE).min(corte);
            let bx = x_treino.take_rows(&posicoes[ini..fim]);
            let rot: Vec<usize> = posicoes[ini..fim].iter().map(|&i| y_treino[i]).collect();
            let by = losses::one_hot(&rot, CLASSES);

            let tape = Tape::new();
            let entrada = constant(&tape, bx);
            let perda = losses::softmax_cross_entropy(&model.forward(&entrada), &by);
            opt.step(&params, &perda.backward());
        }
    }
    let decorrido = inicio.elapsed().as_secs_f64();

    let acc_treino = losses::accuracy(&model.predict(&x_treino), &y_treino);
    let acc_teste = losses::accuracy(&model.predict(&x_teste), &y_teste);

    println!("motor=rtensor");
    println!("parametros={}", model.num_params());
    println!("treino_s={decorrido:.4}");
    println!("acc_treino={:.4}", acc_treino);
    println!("acc_teste={:.4}", acc_teste);
}
