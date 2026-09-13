//! Precisão mista contra `f32`: tempo, energia e precisão.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example bench_mista -- [N]
//!
//! Menos bits movidos é menos energia por operação, não só menos tempo. Se o
//! ganho energético superar o de velocidade, a meia precisão vale mesmo em
//! cargas que não são limitadas por tempo — e esse é o argumento que o RGPU
//! existe para medir.

use std::time::Instant;

use rgpu_power::Medidor;
use rtensor::gpu::Gpu;
use rtensor::prelude::*;

struct Linha {
    nome: &'static str,
    ms: f64,
    gflops: f64,
    j: Option<f64>,
    gflop_j: Option<f64>,
    erro: f32,
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let gpu = Gpu::com_adaptador("nvidia").expect("sem GPU NVIDIA");
    println!("{}", gpu.info());
    println!("C[{n}×{n}] = A[{n}×{n}] · B[{n}×{n}]\n");

    let mut medidor = Medidor::novo().com_intervalo(50);
    medidor.calibrar_ociosidade(4.0);

    let a = Tensor::new(&[n, n], (0..n * n).map(|i| (i as f32 * 0.0013).sin()).collect());
    let b = Tensor::new(&[n, n], (0..n * n).map(|i| (i as f32 * 0.0009).cos()).collect());
    let esperado = a.matmul(&b);
    let gflop = 2.0 * (n as f64).powi(3) / 1e9;

    let ga = gpu.upload(&a);
    let gb = gpu.upload(&b);
    let (a16, _, op_a) = gpu.upload_f16(&a);
    let (b16, _, op_b) = gpu.upload_f16(&b);

    // Conversão uma vez, fora da medição: num treino os pesos ficariam
    // residentes em meia precisão, convertidos só quando atualizados.
    let mut enc = gpu.encoder();
    gpu.record(&mut enc, &op_a);
    gpu.record(&mut enc, &op_b);
    gpu.submit(enc);
    gpu.sync();

    let c1 = gpu.zeros(&[n, n]);
    let c2 = gpu.zeros(&[n, n]);
    let op16 = gpu.op_matmul_f16(&a16, &b16, &c2, false, false);

    let erro = |t: &Tensor| -> f32 {
        esperado
            .data()
            .iter()
            .zip(t.data())
            .map(|(x, y)| (x - y).abs() / (1.0 + x.abs()))
            .fold(0.0, f32::max)
    };

    let mut linhas = Vec::new();
    for (nome, usa16) in [("f32", false), ("mista (f16/f32)", true)] {
        // Janela longa o bastante para o sensor de potência acompanhar.
        let reps = (6.0 / (gflop / 4000.0)).max(20.0) as usize;

        let (_, m) = medidor.medir(|| {
            let mut enc = gpu.encoder();
            for _ in 0..reps {
                if usa16 {
                    gpu.record(&mut enc, &op16);
                } else {
                    gpu.matmul(&mut enc, &ga, &gb, &c1);
                }
            }
            gpu.submit(enc);
            gpu.sync();
        });

        let ms = m.duracao_s / reps as f64 * 1e3;
        let j = m.energia_gpu_total_j().map(|t| t / reps as f64);
        linhas.push(Linha {
            nome,
            ms,
            gflops: gflop / (ms / 1e3),
            j,
            gflop_j: j.filter(|v| *v > 1e-9).map(|v| gflop / v),
            erro: erro(&gpu.download(if usa16 { &c2 } else { &c1 })),
        });
    }

    println!(
        "{:<18} {:>9} {:>11} {:>10} {:>11} {:>11}",
        "precisão", "ms", "GFLOP/s", "J/produto", "GFLOP/J", "erro rel."
    );
    for l in &linhas {
        println!(
            "{:<18} {:>9.3} {:>11.1} {:>10} {:>11} {:>11.2e}",
            l.nome,
            l.ms,
            l.gflops,
            l.j.map_or("—".into(), |v| format!("{v:.4}")),
            l.gflop_j.map_or("—".into(), |v| format!("{v:.1}")),
            l.erro
        );
    }

    let (f32l, f16l) = (&linhas[0], &linhas[1]);
    println!("\nvelocidade: {:+.1}%", 100.0 * (f32l.ms / f16l.ms - 1.0));
    if let (Some(a), Some(b)) = (f32l.gflop_j, f16l.gflop_j) {
        println!("eficiência: {:+.1}%", 100.0 * (b / a - 1.0));
    }
    println!("erro:       {:.0}× maior", f16l.erro / f32l.erro.max(1e-12));
    let _ = Instant::now();
}
