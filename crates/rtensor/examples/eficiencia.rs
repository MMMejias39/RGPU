//! Quanta energia se desperdiça quando a carga não preenche a GPU.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example eficiencia
//!
//! Uma GPU ociosa e uma GPU ocupada consomem quase o mesmo: o custo fixo de
//! manter o chip ligado domina quando não há trabalho suficiente. A métrica que
//! revela isso é **joules por amostra treinada** — não joules por segundo, nem
//! FLOP por segundo.
//!
//! A varredura de lote mede exatamente isso: mesmo modelo, mesma placa, mesmo
//! código; só muda quanto trabalho chega por vez. Se a energia por amostra cair
//! ordens de grandeza com o lote, o desperdício não está no silício — está em
//! quem o alimenta.

use rgpu_power::Medidor;
use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
use rtensor::prelude::*;

const CLASSES: usize = 10;
const D: usize = 512;
const H: usize = 1024;
const JANELA_S: f64 = 5.0;

fn flop_por_amostra() -> f64 {
    (D * H + H * H + H * CLASSES) as f64 * 3.0 * 2.0
}

fn main() {
    let mut medidor = Medidor::novo().com_intervalo(50);
    let gpu = match Gpu::com_adaptador("nvidia") {
        Ok(g) => g,
        Err(e) => {
            eprintln!("este experimento precisa da GPU NVIDIA: {e}");
            std::process::exit(1);
        }
    };
    println!("{}", gpu.info());
    println!("modelo {D}→{H}→{H}→{CLASSES}\n");

    print!("calibrando a potência ociosa (4 s)... ");
    medidor.calibrar_ociosidade(4.0);
    let ocioso = medidor.ociosidade_w().unwrap_or(0.0);
    println!("{ocioso:.2} W\n");

    println!(
        "{:>6} {:>10} {:>10} {:>11} {:>12} {:>13} {:>11}",
        "lote", "ms/passo", "média W", "J/passo", "µJ/amostra", "amostras/s", "GFLOP/J"
    );

    let mut base: Option<f64> = None;
    for lote in [1usize, 8, 32, 128, 512, 2048] {
        let x = Tensor::new(
            &[lote, D],
            (0..lote * D)
                .map(|i| ((i / D) as f32 * 0.01 + (i % D) as f32 * 0.03).sin())
                .collect(),
        );
        let rotulos: Vec<usize> = (0..lote).map(|i| i % CLASSES).collect();

        let mut rng = Rng::new(7);
        let model = Sequential::new()
            .add(Dense::new(D, H, Activation::Relu, &mut rng))
            .add(Dense::new(H, H, Activation::Relu, &mut rng))
            .add(Dense::new(H, CLASSES, Activation::Linear, &mut rng));
        let mut mlp = GpuMlp::from_params(&gpu, &model.params(), lote, 0.001);
        let gx = gpu.upload(&x);
        let gy = upload_labels(&gpu, &rotulos);

        for _ in 0..10 {
            mlp.step(&gpu, &gx, &gy);
        }
        gpu.sync();

        // Calibra o número de passos para encher a janela de medição.
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            mlp.step(&gpu, &gx, &gy);
        }
        gpu.sync();
        let por_passo = t0.elapsed().as_secs_f64() / 20.0;
        let passos = ((JANELA_S / por_passo) as usize).clamp(20, 500_000);

        let (_, m) = medidor.medir(|| {
            for _ in 0..passos {
                mlp.step(&gpu, &gx, &gy);
            }
            gpu.sync();
        });

        let s_por_passo = m.duracao_s / passos as f64;
        let amostras = (passos * lote) as f64;
        let j_passo = m.energia_gpu_j().map(|j| j / passos as f64);
        let uj_amostra = m.energia_gpu_j().map(|j| j / amostras * 1e6);
        let media_w = m.gpu.as_ref().map_or(0.0, |g| g.media_w);
        let gflop_j = m
            .energia_gpu_j()
            .filter(|j| *j > 1e-6)
            .map(|j| flop_por_amostra() * amostras / j / 1e9);

        if base.is_none() {
            base = uj_amostra;
        }

        println!(
            "{:>6} {:>10.4} {:>10.1} {:>11} {:>12} {:>13.0} {:>11}",
            lote,
            s_por_passo * 1e3,
            media_w,
            j_passo.map_or("—".into(), |v| format!("{v:.4}")),
            uj_amostra.map_or("—".into(), |v| format!("{v:.1}")),
            lote as f64 / s_por_passo,
            gflop_j.map_or("—".into(), |v| format!("{v:.1}")),
        );
    }

    if let (Some(b), Some(_)) = (base, medidor.ociosidade_w()) {
        println!(
            "\nEnergia por amostra com lote 1: {b:.1} µJ.\n\
             A potência média mal se move entre os lotes — o chip custa quase o\n\
             mesmo ligado, fazendo muito ou pouco. O que muda é quanto trabalho\n\
             útil sai por joule."
        );
    }
}
