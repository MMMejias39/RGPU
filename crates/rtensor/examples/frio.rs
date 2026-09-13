//! Partida a frio: do início do processo até o primeiro passo de treino
//! concluído. Meça por fora com `time`.

use std::time::Instant;

use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
use rtensor::prelude::*;

fn main() {
    let t0 = Instant::now();
    let gpu = Gpu::new().expect("sem GPU");
    let t_dispositivo = t0.elapsed().as_secs_f64();

    let (d, h, lote) = (512usize, 1024usize, 128usize);
    let mut rng = Rng::new(7);
    let model = Sequential::new()
        .add(Dense::new(d, h, Activation::Relu, &mut rng))
        .add(Dense::new(h, h, Activation::Relu, &mut rng))
        .add(Dense::new(h, 10, Activation::Linear, &mut rng));

    let x = Tensor::new(&[lote, d], (0..lote * d).map(|i| (i as f32 * 0.01).sin()).collect());
    let rotulos: Vec<usize> = (0..lote).map(|i| i % 10).collect();

    let mut mlp = GpuMlp::from_params(&gpu, &model.params(), lote, 0.001);
    let gx = gpu.upload(&x);
    let gy = upload_labels(&gpu, &rotulos);

    let t1 = Instant::now();
    mlp.step(&gpu, &gx, &gy);
    let perda = mlp.last_loss(&gpu);
    let t_passo = t1.elapsed().as_secs_f64();

    println!(
        "teste=frio|abrir_dispositivo_ms={:.1}|primeiro_passo_ms={:.1}|total_ms={:.1}|perda={perda:.4}",
        t_dispositivo * 1e3,
        t_passo * 1e3,
        t0.elapsed().as_secs_f64() * 1e3
    );
}
