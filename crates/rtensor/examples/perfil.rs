//! Perfilamento por kernel com marcas de tempo do dispositivo.
//!
//! Uso: cargo run --release --features gpu --example perfil -- [lote] [largura] [camadas]
//!
//! Compara o somatório do tempo dos kernels (medido na GPU) com o tempo de
//! parede do passo (medido no host). A diferença entre os dois é o custo que
//! **não** está nos kernels: gravação de comandos, submissão e sincronização.

use std::time::Instant;

use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
use rtensor::prelude::*;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let lote = *a.first().unwrap_or(&256);
    let largura = *a.get(1).unwrap_or(&576);
    let camadas = *a.get(2).unwrap_or(&12);

    let gpu = Gpu::new().expect("sem GPU");
    println!("{}", gpu.info());
    if !gpu.tem_perfilamento() {
        eprintln!("adaptador sem TIMESTAMP_QUERY — sem perfilamento");
        return;
    }

    let d = 512;
    let mut dims = vec![d];
    dims.extend(std::iter::repeat(largura).take(camadas));
    dims.push(10);

    let mut rng = Rng::new(7);
    let mut model = Sequential::new();
    for i in 0..dims.len() - 1 {
        let ativa = if i == dims.len() - 2 { Activation::Linear } else { Activation::Relu };
        model = model.add(Dense::new(dims[i], dims[i + 1], ativa, &mut rng));
    }

    let x = Tensor::new(
        &[lote, d],
        (0..lote * d).map(|i| (i as f32 * 0.01).sin()).collect(),
    );
    let rotulos: Vec<usize> = (0..lote).map(|i| i % 10).collect();

    let mut mlp = GpuMlp::from_params(&gpu, &model.params(), lote, 0.001);
    let gx = gpu.upload(&x);
    let gy = upload_labels(&gpu, &rotulos);

    println!(
        "lote={lote} largura={largura} camadas={} parametros={} dispatches/passo={}\n",
        dims.len() - 1,
        model.num_params(),
        mlp.dispatches_por_passo()
    );

    // Aquece.
    for _ in 0..5 {
        mlp.step(&gpu, &gx, &gy);
    }
    gpu.sync();

    // Quantos passos cabem nas marcas disponíveis.
    let passos = (gpu.capacidade_perfil() / mlp.dispatches_por_passo()).clamp(1, 20);

    let linhas = gpu.perfilar(|| {
        for _ in 0..passos {
            mlp.step(&gpu, &gx, &gy);
        }
        gpu.sync();
    });

    // O tempo de parede é medido à parte, com o perfilamento desligado: ligado,
    // cada operação vira um passe separado e o número deixa de ser comparável.
    let inicio = Instant::now();
    for _ in 0..passos {
        mlp.step(&gpu, &gx, &gy);
    }
    gpu.sync();
    let parede = inicio.elapsed().as_secs_f64() * 1e3 / passos as f64;

    println!("perfilando {passos} passos\n");

    println!("{:<16} {:>9} {:>12} {:>10}", "kernel", "chamadas", "ms/passo", "% do total");
    let total: f64 = linhas.iter().map(|(_, _, ms)| ms).sum::<f64>() / passos as f64;
    for (rotulo, chamadas, ms) in &linhas {
        let por_passo = ms / passos as f64;
        println!(
            "{:<16} {:>9} {:>12.4} {:>9.1}%",
            rotulo,
            *chamadas as f64 / passos as f64,
            por_passo,
            100.0 * por_passo / total
        );
    }

    println!("\n{:<16} {:>12.4} ms", "soma kernels", total);
    println!("{:<16} {:>12.4} ms", "tempo de parede", parede);
    println!(
        "{:<16} {:>12.4} ms  ({:.0}% do passo)",
        "fora dos kernels",
        parede - total,
        100.0 * (parede - total) / parede
    );
}
