//! As espirais na GPU — mesma tarefa do `bench.rs`, mesmo modelo, mesmos dados.
//!
//! Uso: cargo run --release --features gpu --example bench_espiral_gpu
//!
//! Embaralha a cada época como os outros benchmarks, o que obriga a reenviar os
//! lotes; a 512 amostras de 2 características isso é ruído no cronômetro.
//! Os lotes parciais são descartados, porque o modelo de GPU fixa o tamanho do
//! lote na construção — 16 lotes de 32 por época.

use std::time::Instant;

use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
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

fn main() {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("sem GPU: {e}");
            std::process::exit(1);
        }
    };

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

    let model = Sequential::new()
        .add(Dense::new(2, 64, Activation::Relu, &mut rng))
        .add(Dense::new(64, 32, Activation::Relu, &mut rng))
        .add(Dense::new(32, CLASSES, Activation::Linear, &mut rng));

    let params = model.params();
    let mut mlp = GpuMlp::from_params(&gpu, &params, LOTE, 0.01);
    let mut posicoes: Vec<usize> = (0..corte).collect();
    let lotes = corte / LOTE;

    let inicio = Instant::now();
    for _ in 0..EPOCAS {
        rng.shuffle(&mut posicoes);
        for l in 0..lotes {
            let fatia = &posicoes[l * LOTE..(l + 1) * LOTE];
            let bx = gpu.upload(&x_treino.take_rows(fatia));
            let rot: Vec<usize> = fatia.iter().map(|&i| y_treino[i]).collect();
            let by = upload_labels(&gpu, &rot);
            mlp.step(&gpu, &bx, &by);
        }
    }
    gpu.sync();
    let decorrido = inicio.elapsed().as_secs_f64();

    // Traz os pesos de volta e avalia com o caminho de CPU — prova que os pesos
    // treinados na GPU são os mesmos objetos que o resto da crate usa.
    for (p, w) in params.iter().zip(mlp.weights(&gpu)) {
        p.set(w);
    }

    println!("motor=rtensor-gpu");
    println!("dispositivo={}", gpu.info());
    println!("parametros={}", model.num_params());
    println!("treino_s={decorrido:.4}");
    println!("acc_treino={:.4}", losses::accuracy(&model.predict(&x_treino), &y_treino));
    println!("acc_teste={:.4}", losses::accuracy(&model.predict(&x_teste), &y_teste));
}
