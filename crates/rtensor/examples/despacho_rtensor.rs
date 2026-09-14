//! Equivalente a `benchmarks/python/despacho_tf.py` e `despacho_torch.py`:
//! isola o custo de submissão (CPU) do custo de computação real (GPU), no
//! mesmo padrão de medição da suíte — `reps` chamadas, sincroniza uma vez no
//! fim. O "passo trivial" é um `relu` num tensor de 1 elemento: a submissão
//! do comando pesa o mesmo de um passo real, o trabalho da GPU é desprezível.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example despacho_rtensor

use std::time::Instant;

use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
use rtensor::prelude::*;

fn main() {
    let gpu = Gpu::new().expect("sem GPU");
    println!("{}\n", gpu.info());

    let trivial_x = gpu.zeros(&[1]);
    let trivial_y = gpu.zeros(&[1]);
    let reps_trivial = 200;

    println!("{:>8} {:>12} {:>12} {:>10}", "lote", "real_ms", "trivial_ms", "despacho%");
    for lote in [1usize, 8, 32, 128, 512, 2048] {
        let d = 512;
        let h = 1024;
        let mut rng = Rng::new(7);
        let model = Sequential::new()
            .add(Dense::new(d, h, Activation::Relu, &mut rng))
            .add(Dense::new(h, h, Activation::Relu, &mut rng))
            .add(Dense::new(h, 10, Activation::Linear, &mut rng));
        let params = model.params();
        let x = Tensor::new(&[lote, d], (0..lote * d).map(|i| (i as f32 * 0.01).sin()).collect());
        let rotulos: Vec<usize> = (0..lote).map(|i| i % 10).collect();

        let mut mlp = GpuMlp::from_params(&gpu, &params, lote, 0.001);
        let gx = gpu.upload(&x);
        let gy = upload_labels(&gpu, &rotulos);

        let reps = if lote >= 512 { 5 } else { 20 };
        mlp.step(&gpu, &gx, &gy);
        gpu.sync();
        let t_real = {
            let t0 = Instant::now();
            for _ in 0..reps {
                mlp.step(&gpu, &gx, &gy);
            }
            gpu.sync();
            t0.elapsed().as_secs_f64() / reps as f64
        };

        // Passo trivial: mesmo padrão de submissão (um `CommandEncoder`, um
        // dispatch, `submit`), sem o resto do plano.
        let op = gpu.op_relu(&trivial_x, &trivial_y);
        let rodar_trivial = || {
            let mut enc = gpu.encoder();
            gpu.record_many(&mut enc, &[&op]);
            gpu.submit(enc);
        };
        rodar_trivial();
        gpu.sync();
        let t_trivial = {
            let t0 = Instant::now();
            for _ in 0..reps_trivial {
                rodar_trivial();
            }
            gpu.sync();
            t0.elapsed().as_secs_f64() / reps_trivial as f64
        };

        println!(
            "{:>8} {:>12.4} {:>12.4} {:>10.1}",
            lote,
            t_real * 1e3,
            t_trivial * 1e3,
            100.0 * t_trivial / t_real
        );
    }
}
