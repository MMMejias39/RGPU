//! Mesma carga do rtensor, no candle.
//!
//! Uso: candle_bench [lote] [cpu|cuda]

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, ops, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

const D: usize = 512;
const H: usize = 1024;
const C: usize = 10;

struct Mlp {
    l1: Linear,
    l2: Linear,
    l3: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        self.l3.forward(&x)
    }
}

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let lote: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let quer_cuda = args.get(2).map(|s| s == "cuda").unwrap_or(false);

    let dev = if quer_cuda {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    let nome = if dev.is_cuda() { "candle-cuda" } else { "candle-cpu" };

    let dados: Vec<f32> = (0..lote * D)
        .map(|i| ((i / D) as f32 * 0.01 + (i % D) as f32 * 0.03).sin())
        .collect();
    let x = Tensor::from_vec(dados, (lote, D), &dev)?;
    let rotulos: Vec<u32> = (0..lote).map(|i| (i % C) as u32).collect();
    let y = Tensor::from_vec(rotulos, lote, &dev)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Mlp {
        l1: candle_nn::linear(D, H, vb.pp("l1"))?,
        l2: candle_nn::linear(H, H, vb.pp("l2"))?,
        l3: candle_nn::linear(H, C, vb.pp("l3"))?,
    };
    let mut otim = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;

    let mut passo = || -> candle_core::Result<()> {
        let logits = model.forward(&x)?;
        let l = loss::cross_entropy(&logits, &y)?;
        otim.backward_step(&l)
    };

    for _ in 0..5 {
        passo()?;
    }

    let passos = if dev.is_cuda() { 200 } else { 20 };
    let inicio = Instant::now();
    for _ in 0..passos {
        passo()?;
    }
    // Força a sincronização lendo um escalar.
    let _ = ops::softmax(&model.forward(&x)?, 1)?.sum_all()?.to_scalar::<f32>()?;
    let dt = inicio.elapsed().as_secs_f64() / passos as f64;

    let flop = (D * H + H * H + H * C) as f64 * 3.0 * 2.0 * lote as f64;
    println!("motor={nome}|lote={lote}|ms={:.4}|gflops={:.1}", dt * 1e3, flop / dt / 1e9);
    Ok(())
}
