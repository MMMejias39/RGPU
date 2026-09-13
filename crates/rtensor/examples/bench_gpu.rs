//! Mesma carga do `bench_grande`, agora na GPU.
//!
//! Uso: cargo run --release --features gpu --example bench_gpu -- [N D H LOTE]
//!
//! Os lotes são enviados ao dispositivo **antes** da medição, para que o laço
//! cronometrado seja só computação — do mesmo jeito que o TensorFlow mantém o
//! dataset residente. Um `sync()` no fim garante que a GPU terminou de verdade,
//! já que as submissões são assíncronas.

use std::time::Instant;

use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
use rtensor::prelude::*;

const CLASSES: usize = 10;
const EPOCAS: usize = 3;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let n = *a.first().unwrap_or(&4096);
    let d = *a.get(1).unwrap_or(&128);
    let h = *a.get(2).unwrap_or(&512);
    let lote = *a.get(3).unwrap_or(&256);

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("sem GPU: {e}");
            std::process::exit(1);
        }
    };

    // Mesma fórmula determinística dos outros benchmarks.
    let mut x = Vec::with_capacity(n * d);
    for i in 0..n {
        for j in 0..d {
            x.push((i as f32 * 0.01 + j as f32 * 0.03).sin());
        }
    }
    let x = Tensor::new(&[n, d], x);
    let rotulos: Vec<usize> = (0..n).map(|i| i % CLASSES).collect();

    let mut rng = Rng::new(7);
    let model = Sequential::new()
        .add(Dense::new(d, h, Activation::Relu, &mut rng))
        .add(Dense::new(h, h, Activation::Relu, &mut rng))
        .add(Dense::new(h, CLASSES, Activation::Linear, &mut rng));

    let mut mlp = GpuMlp::from_params(&gpu, &model.params(), lote, 0.001);

    // Dataset residente no dispositivo, fora da medição.
    let lotes: Vec<_> = (0..n)
        .step_by(lote)
        .filter(|&ini| ini + lote <= n)
        .map(|ini| {
            (
                gpu.upload(&x.rows(ini, ini + lote)),
                upload_labels(&gpu, &rotulos[ini..ini + lote]),
            )
        })
        .collect();
    gpu.sync();

    let inicio = Instant::now();
    for _ in 0..EPOCAS {
        for (bx, by) in &lotes {
            mlp.step(&gpu, bx, by);
        }
    }
    gpu.sync();
    let decorrido = inicio.elapsed().as_secs_f64();

    let macs = (d * h + h * h + h * CLASSES) as f64;
    let passos = (lotes.len() * EPOCAS) as f64;
    let gflop = macs * 3.0 * 2.0 * lote as f64 * passos / 1e9;

    println!("motor=rtensor-gpu  N={n} D={d} H={h} lote={lote}");
    println!("dispositivo={}", gpu.info());
    println!("parametros={}", model.num_params());
    println!("treino_s={decorrido:.3}");
    println!("gflops={:.2}", gflop / decorrido);
    println!("perda_final={:.4}", mlp.last_loss(&gpu));
}
