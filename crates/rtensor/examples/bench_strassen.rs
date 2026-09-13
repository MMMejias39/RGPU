//! Strassen de um nível contra o GEMM clássico: velocidade e precisão.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example bench_strassen -- [M N K]
//!
//! Strassen faz 7/8 das multiplicações, mas paga empacotamento `O(n²)` e 25
//! dispatches em vez de 1. O ponto de equilíbrio é empírico: as somas se diluem
//! conforme `n` cresce, então há um tamanho abaixo do qual não compensa.

use std::time::Instant;

use rtensor::gpu::{Gpu, Strassen};
use rtensor::prelude::*;

fn medir(gpu: &Gpu, reps: usize, mut grava: impl FnMut(&mut wgpu::CommandEncoder)) -> f64 {
    let mut enc = gpu.encoder();
    grava(&mut enc);
    gpu.submit(enc);
    gpu.sync();

    let inicio = Instant::now();
    let mut enc = gpu.encoder();
    for _ in 0..reps {
        grava(&mut enc);
    }
    gpu.submit(enc);
    gpu.sync();
    inicio.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    let arg: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let m = *arg.first().unwrap_or(&2048);
    let n = *arg.get(1).unwrap_or(&2048);
    let k = *arg.get(2).unwrap_or(&2048);

    let gpu = Gpu::com_adaptador("nvidia").expect("sem GPU NVIDIA");
    println!("{}", gpu.info());
    println!("C[{m}×{n}] = A[{m}×{k}] · B[{k}×{n}]\n");

    let a = Tensor::new(&[m, k], (0..m * k).map(|i| (i as f32 * 0.0017).sin()).collect());
    let b = Tensor::new(&[k, n], (0..k * n).map(|i| (i as f32 * 0.0011).cos()).collect());

    let ga = gpu.upload(&a);
    let gb = gpu.upload(&b);
    let c1 = gpu.zeros(&[m, n]);
    let c2 = gpu.zeros(&[m, n]);

    let plano = match Strassen::novo(&gpu, &ga, &gb, &c2) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let reps = if m * n * k > 1 << 30 { 5 } else { 20 };
    let t_classico = medir(&gpu, reps, |enc| gpu.matmul(enc, &ga, &gb, &c1));
    let t_strassen = medir(&gpu, reps, |enc| plano.executar(&gpu, enc));

    // GFLOP contados pelo algoritmo clássico: é a mesma conta útil entregue,
    // então a comparação mede trabalho por segundo, não operações por segundo.
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;

    println!("{:<14} {:>10} {:>12} {:>12}", "algoritmo", "ms", "GFLOP/s", "dispatches");
    println!("{:<14} {:>10.3} {:>12.1} {:>12}", "clássico", t_classico * 1e3, gflop / t_classico, 1);
    println!(
        "{:<14} {:>10.3} {:>12.1} {:>12}",
        "Strassen",
        t_strassen * 1e3,
        gflop / t_strassen,
        plano.dispatches()
    );
    println!("\nganho: {:+.1}%", 100.0 * (t_classico / t_strassen - 1.0));

    // Precisão: as duas contra a CPU.
    let esperado = a.matmul(&b);
    let erro = |t: &Tensor| -> f32 {
        esperado
            .data()
            .iter()
            .zip(t.data())
            .map(|(x, y)| (x - y).abs() / (1.0 + x.abs()))
            .fold(0.0, f32::max)
    };
    println!(
        "erro relativo máximo: clássico {:.2e}, Strassen {:.2e}",
        erro(&gpu.download(&c1)),
        erro(&gpu.download(&c2))
    );
}
