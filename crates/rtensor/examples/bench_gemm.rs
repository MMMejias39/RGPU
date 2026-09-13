//! Compara o GEMM ingênuo (16×16, um elemento por thread) com o ladrilhado em
//! dois níveis (64×64 por workgroup, bloco 4×4 por thread em registradores).
//!
//! Uso: cargo run --release --features gpu --example bench_gemm -- [M N K]

use std::time::Instant;

use rtensor::gpu::Gpu;
use rtensor::prelude::*;

fn medir(gpu: &Gpu, a: &Tensor, b: &Tensor, m: usize, n: usize, repeticoes: usize) -> f64 {
    let ga = gpu.upload(a);
    let gb = gpu.upload(b);
    let gc = gpu.zeros(&[m, n]);
    gpu.sync();

    // Aquece.
    let mut enc = gpu.encoder();
    gpu.matmul(&mut enc, &ga, &gb, &gc);
    gpu.submit(enc);
    gpu.sync();

    let inicio = Instant::now();
    let mut enc = gpu.encoder();
    for _ in 0..repeticoes {
        gpu.matmul(&mut enc, &ga, &gb, &gc);
    }
    gpu.submit(enc);
    gpu.sync();
    inicio.elapsed().as_secs_f64()
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let m = *a.first().unwrap_or(&1024);
    let n = *a.get(1).unwrap_or(&1024);
    let k = *a.get(2).unwrap_or(&1024);
    let reps = 50;

    let mut gpu = Gpu::new().expect("sem GPU");
    println!("{}", gpu.info());
    println!("C[{m}×{n}] = A[{m}×{k}] · B[{k}×{n}], {reps} repetições\n");

    let ta = Tensor::new(&[m, k], (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect());
    let tb = Tensor::new(&[k, n], (0..k * n).map(|i| (i as f32 * 0.002).cos()).collect());
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 * reps as f64 / 1e9;

    gpu.set_fast_gemm(false);
    let lento = medir(&gpu, &ta, &tb, m, n, reps);
    gpu.set_fast_gemm(true);

    println!("ingênuo (16×16, 1 elem/thread) : {:>8.3} s   {:>8.1} GFLOP/s\n", lento, gflop / lento);

    // `grupo = 1` é o percurso em linha; os demais agrupam linhas de blocos para
    // que os painéis compartilhados ainda estejam na L2 quando forem reusados.
    println!("{:>8} {:>12} {:>12} {:>10}", "grupo", "s", "GFLOP/s", "vs linha");
    let mut base = 0.0;
    for grupo in [1u32, 2, 4, 8, 16, 32] {
        gpu.set_grupo_l2(grupo);
        let t = medir(&gpu, &ta, &tb, m, n, reps);
        let g = gflop / t;
        if grupo == 1 {
            base = g;
        }
        let rotulo = if grupo == 1 { "  (linha)".to_string() } else { format!("{:+8.1}%", 100.0 * (g / base - 1.0)) };
        println!("{grupo:>8} {t:>12.4} {g:>12.1} {rotulo:>10}");
    }
    gpu.set_grupo_l2(8);
}
