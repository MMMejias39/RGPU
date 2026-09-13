//! Suíte de comparação do rtensor: vários tipos de teste, não só treino de MLP.
//!
//! Uso: cargo run --release --features gpu --example suite -- <teste>
//!   gemm    formatos de produto matricial (quadrado, alto-fino, largo, K-dominante)
//!   batch   varredura de tamanho de lote — latência por passo e amostras/s
//!   arch    profundidade contra largura, a parâmetros aproximadamente iguais
//!   infer   só avanço (inferência)
//!
//! Imprime linhas `chave=valor` separadas por `|`, para tabular depois.

use std::time::Instant;

use rtensor::prelude::*;
#[cfg(feature = "gpu")]
use rtensor::gpu::{upload_labels, Gpu, GpuMlp};

const CLASSES: usize = 10;

fn agora<F: FnMut()>(mut f: F, reps: usize) -> f64 {
    let inicio = Instant::now();
    for _ in 0..reps {
        f();
    }
    inicio.elapsed().as_secs_f64() / reps as f64
}

fn dados(n: usize, d: usize) -> Tensor {
    Tensor::new(
        &[n, d],
        (0..n * d).map(|i| ((i / d) as f32 * 0.01 + (i % d) as f32 * 0.03).sin()).collect(),
    )
}

// ------------------------------------------------------------------- gemm

#[cfg(feature = "gpu")]
fn gemm(gpu: &Gpu) {
    // (rótulo, M, N, K) — formatos com perfis de acesso bem diferentes.
    let casos: [(&str, usize, usize, usize); 6] = [
        ("quadrado-2048", 2048, 2048, 2048),
        ("quadrado-512", 512, 512, 512),
        ("alto-fino", 8192, 64, 64),
        ("largo", 64, 8192, 64),
        ("K-dominante", 64, 64, 8192),
        ("lote-mlp", 1024, 2048, 1024),
    ];

    for (nome, m, n, k) in casos {
        let a = dados(m, k);
        let b = dados(k, n);
        let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;

        // CPU: poucas repetições nos casos caros.
        let reps_cpu = if gflop > 2.0 { 1 } else { 5 };
        let t_cpu = agora(|| { a.matmul(&b); }, reps_cpu);

        let ga = gpu.upload(&a);
        let gb = gpu.upload(&b);
        let gc = gpu.zeros(&[m, n]);
        let mut enc = gpu.encoder();
        gpu.matmul(&mut enc, &ga, &gb, &gc);
        gpu.submit(enc);
        gpu.sync();

        let reps_gpu = 20;
        let t_gpu = {
            let inicio = Instant::now();
            let mut enc = gpu.encoder();
            for _ in 0..reps_gpu {
                gpu.matmul(&mut enc, &ga, &gb, &gc);
            }
            gpu.submit(enc);
            gpu.sync();
            inicio.elapsed().as_secs_f64() / reps_gpu as f64
        };

        println!(
            "teste=gemm|caso={nome}|m={m}|n={n}|k={k}|cpu_ms={:.4}|cpu_gflops={:.1}|gpu_ms={:.4}|gpu_gflops={:.1}",
            t_cpu * 1e3,
            gflop / t_cpu,
            t_gpu * 1e3,
            gflop / t_gpu
        );
    }
}

// ------------------------------------------------------------------ modelo

fn construir(dims: &[usize], rng: &mut Rng) -> Sequential {
    let mut m = Sequential::new();
    for i in 0..dims.len() - 1 {
        let ativa = if i == dims.len() - 2 { Activation::Linear } else { Activation::Relu };
        m = m.add(Dense::new(dims[i], dims[i + 1], ativa, rng));
    }
    m
}

fn passo_cpu(model: &Sequential, opt: &mut Adam, params: &[Param], x: &Tensor, y: &Tensor) {
    let tape = Tape::new();
    let entrada = constant(&tape, x.clone());
    let perda = losses::softmax_cross_entropy(&model.forward(&entrada), y);
    opt.step(params, &perda.backward());
}

// ------------------------------------------------------------------- batch

#[cfg(feature = "gpu")]
fn batch(gpu: &Gpu) {
    let d = 512;
    let h = 1024;
    let dims = [d, h, h, CLASSES];

    for lote in [1usize, 8, 32, 128, 512, 2048] {
        let mut rng = Rng::new(7);
        let model = construir(&dims, &mut rng);
        let params = model.params();
        let x = dados(lote, d);
        let rotulos: Vec<usize> = (0..lote).map(|i| i % CLASSES).collect();
        let y = losses::one_hot(&rotulos, CLASSES);

        let reps = if lote >= 512 { 5 } else { 20 };

        // A GPU é medida primeiro: neste laptop o Max-Q divide orçamento de
        // energia com a CPU, e medir depois de uma carga pesada de CPU reduz o
        // clock da GPU e distorce o resultado em até 3×.
        let mut mlp = GpuMlp::from_params(gpu, &params, lote, 0.001);
        let gx = gpu.upload(&x);
        let gy = upload_labels(gpu, &rotulos);
        mlp.step(gpu, &gx, &gy);
        gpu.sync();
        let t_gpu = {
            let inicio = Instant::now();
            for _ in 0..reps {
                mlp.step(gpu, &gx, &gy);
            }
            gpu.sync();
            inicio.elapsed().as_secs_f64() / reps as f64
        };

        let mut opt = Adam::new(0.001);
        passo_cpu(&model, &mut opt, &params, &x, &y);
        let t_cpu = agora(|| passo_cpu(&model, &mut opt, &params, &x, &y), reps);

        println!(
            "teste=batch|lote={lote}|cpu_ms={:.4}|cpu_amostras_s={:.0}|gpu_ms={:.4}|gpu_amostras_s={:.0}",
            t_cpu * 1e3,
            lote as f64 / t_cpu,
            t_gpu * 1e3,
            lote as f64 / t_gpu
        );
    }
}

// -------------------------------------------------------------------- arch

#[cfg(feature = "gpu")]
fn arch(gpu: &Gpu) {
    let lote = 256;
    // Três formatos com contagem de parâmetros parecida (~4,2M).
    let casos: [(&str, Vec<usize>); 3] = [
        ("raso-largo", vec![512, 2048, 1024, CLASSES]),
        ("medio", vec![512, 1024, 1024, 1024, 1024, CLASSES]),
        ("profundo-estreito", vec![512, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, 576, CLASSES]),
    ];

    for (nome, dims) in casos {
        let mut rng = Rng::new(7);
        let model = construir(&dims, &mut rng);
        let params = model.params();
        let x = dados(lote, dims[0]);
        let rotulos: Vec<usize> = (0..lote).map(|i| i % CLASSES).collect();
        let y = losses::one_hot(&rotulos, CLASSES);

        let mut mlp = GpuMlp::from_params(gpu, &params, lote, 0.001);
        let gx = gpu.upload(&x);
        let gy = upload_labels(gpu, &rotulos);
        mlp.step(gpu, &gx, &gy);
        gpu.sync();
        let t_gpu = {
            let inicio = Instant::now();
            for _ in 0..20 {
                mlp.step(gpu, &gx, &gy);
            }
            gpu.sync();
            inicio.elapsed().as_secs_f64() / 20.0
        };

        let mut opt = Adam::new(0.001);
        passo_cpu(&model, &mut opt, &params, &x, &y);
        let t_cpu = agora(|| passo_cpu(&model, &mut opt, &params, &x, &y), 10);

        println!(
            "teste=arch|caso={nome}|camadas={}|parametros={}|cpu_ms={:.4}|gpu_ms={:.4}",
            dims.len() - 1,
            model.num_params(),
            t_cpu * 1e3,
            t_gpu * 1e3
        );
    }
}

// ------------------------------------------------------------------- infer

#[cfg(feature = "gpu")]
fn infer(gpu: &Gpu) {
    let d = 512;
    let h = 1024;
    let dims = [d, h, h, CLASSES];

    for lote in [1usize, 32, 512, 4096] {
        let mut rng = Rng::new(7);
        let model = construir(&dims, &mut rng);
        let x = dados(lote, d);

        let mlp = GpuMlp::from_params(gpu, &model.params(), lote, 0.0);
        let gx = gpu.upload(&x);
        mlp.forward_only(gpu, &gx);
        gpu.sync();
        let reps = 20;
        let t_gpu = {
            let inicio = Instant::now();
            for _ in 0..reps {
                mlp.forward_only(gpu, &gx);
            }
            gpu.sync();
            inicio.elapsed().as_secs_f64() / reps as f64
        };

        model.predict(&x);
        let t_cpu = agora(|| { model.predict(&x); }, if lote >= 512 { 5 } else { 20 });

        println!(
            "teste=infer|lote={lote}|cpu_ms={:.4}|cpu_amostras_s={:.0}|gpu_ms={:.4}|gpu_amostras_s={:.0}",
            t_cpu * 1e3,
            lote as f64 / t_cpu,
            t_gpu * 1e3,
            lote as f64 / t_gpu
        );
    }
}

// -------------------------------------------------------------------- elem

#[cfg(feature = "gpu")]
fn elem(gpu: &Gpu) {
    for (linhas, cols) in [(512usize, 1024usize), (4096, 1024), (8192, 2048)] {
        let n = linhas * cols;
        let x = gpu.zeros(&[linhas, cols]);
        let y = gpu.zeros(&[linhas, cols]);
        let b = gpu.zeros(&[1, cols]);
        gpu.sync();

        let reps = 50;
        let medir = |f: &dyn Fn(&mut wgpu::CommandEncoder)| -> f64 {
            let mut enc = gpu.encoder();
            f(&mut enc);
            gpu.submit(enc);
            gpu.sync();
            let inicio = Instant::now();
            let mut enc = gpu.encoder();
            for _ in 0..reps {
                f(&mut enc);
            }
            gpu.submit(enc);
            gpu.sync();
            inicio.elapsed().as_secs_f64() / reps as f64
        };

        let t_bias = medir(&|e| gpu.bias_add(e, &x, &b));
        let t_relu = medir(&|e| gpu.relu(e, &x, &y));

        // Cada kernel lê e escreve n floats (o bias lê n+cols, escreve n).
        let gb = 2.0 * n as f64 * 4.0 / 1e9;
        println!(
            "teste=elem|linhas={linhas}|cols={cols}|bias_ms={:.4}|bias_GBs={:.1}|relu_ms={:.4}|relu_GBs={:.1}",
            t_bias * 1e3,
            gb / t_bias,
            t_relu * 1e3,
            gb / t_relu
        );
    }
}

fn main() {
    let teste = std::env::args().nth(1).unwrap_or_else(|| "gemm".into());

    #[cfg(feature = "gpu")]
    {
        let gpu = Gpu::new().expect("sem GPU");
        eprintln!("dispositivo: {}", gpu.info());
        match teste.as_str() {
            "gemm" => gemm(&gpu),
            "batch" => batch(&gpu),
            "arch" => arch(&gpu),
            "infer" => infer(&gpu),
            "elem" => elem(&gpu),
            outro => eprintln!("teste desconhecido: {outro}"),
        }
    }
    #[cfg(not(feature = "gpu"))]
    eprintln!("compile com --features gpu ({teste})");
}
