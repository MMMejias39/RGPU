//! Benchmark de carga pesada: aqui o tempo é dominado pelos GEMM, não pelo
//! overhead por passo. Os dados são sintéticos e gerados por uma fórmula
//! determinística, idêntica à do script Python equivalente.
//!
//! Uso: cargo run --release --example bench_grande

use std::time::Instant;

use rtensor::prelude::*;

const CLASSES: usize = 10;
const EPOCAS: usize = 3;

/// Dimensões vindas da linha de comando: N D H LOTE (com padrões).
fn dims() -> (usize, usize, usize, usize) {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    (
        *a.first().unwrap_or(&4096),
        *a.get(1).unwrap_or(&128),
        *a.get(2).unwrap_or(&512),
        *a.get(3).unwrap_or(&256),
    )
}

fn main() {
    let (n_amostras, d, h, lote) = dims();
    // x[i][j] = sin(i*0.01 + j*0.03); rótulo = i % CLASSES
    let mut x = Vec::with_capacity(n_amostras * d);
    for i in 0..n_amostras {
        for j in 0..d {
            x.push((i as f32 * 0.01 + j as f32 * 0.03).sin());
        }
    }
    let x = Tensor::new(&[n_amostras, d], x);
    let rotulos: Vec<usize> = (0..n_amostras).map(|i| i % CLASSES).collect();

    let mut rng = Rng::new(7);
    let model = Sequential::new()
        .add(Dense::new(d, h, Activation::Relu, &mut rng))
        .add(Dense::new(h, h, Activation::Relu, &mut rng))
        .add(Dense::new(h, CLASSES, Activation::Linear, &mut rng));

    let params = model.params();
    let mut opt = Adam::new(0.001);

    let inicio = Instant::now();
    let mut perda_final = 0.0;
    for _ in 0..EPOCAS {
        for ini in (0..n_amostras).step_by(lote) {
            let fim = (ini + lote).min(n_amostras);
            let bx = x.rows(ini, fim);
            let by = losses::one_hot(&rotulos[ini..fim], CLASSES);

            let tape = Tape::new();
            let entrada = constant(&tape, bx);
            let perda = losses::softmax_cross_entropy(&model.forward(&entrada), &by);
            perda_final = perda.value().item();
            opt.step(&params, &perda.backward());
        }
    }
    let decorrido = inicio.elapsed().as_secs_f64();

    // MACs: (D*H + H*H + H*C) por amostra, x3 para frente+trás, x2 flops por MAC.
    let macs = (d * h + h * h + h * CLASSES) as f64;
    let gflop = macs * 3.0 * 2.0 * n_amostras as f64 * EPOCAS as f64 / 1e9;

    println!("motor=rtensor  N={n_amostras} D={d} H={h} lote={lote}");
    println!("parametros={}", model.num_params());
    println!("treino_s={decorrido:.3}");
    println!("gflops={:.2}", gflop / decorrido);
    println!("perda_final={perda_final:.4}");
}
