//! Mesma carga do rtensor, no burn com backend wgpu.
//!
//! O burn e o rtensor passam pelo mesmo caminho de hardware — wgpu sobre
//! Vulkan —, então a diferença entre eles é implementação, não API nem driver.

use std::time::Instant;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};

const D: usize = 512;
const H: usize = 1024;
const C: usize = 10;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    act: Relu,
}

impl<B: Backend> Mlp<B> {
    fn novo(dev: &B::Device) -> Self {
        Mlp {
            l1: LinearConfig::new(D, H).init(dev),
            l2: LinearConfig::new(H, H).init(dev),
            l3: LinearConfig::new(H, C).init(dev),
            act: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.act.forward(self.l1.forward(x));
        let x = self.act.forward(self.l2.forward(x));
        self.l3.forward(x)
    }
}

fn main() {
    let lote: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    type B = Autodiff<Wgpu>;
    let dev = WgpuDevice::default();

    let dados: Vec<f32> = (0..lote * D)
        .map(|i| ((i / D) as f32 * 0.01 + (i % D) as f32 * 0.03).sin())
        .collect();
    let x: Tensor<B, 2> = Tensor::from_data(TensorData::new(dados, [lote, D]), &dev);
    let rotulos: Vec<i32> = (0..lote).map(|i| (i % C) as i32).collect();
    let y: Tensor<B, 1, Int> = Tensor::from_data(TensorData::new(rotulos, [lote]), &dev);

    let mut model = Mlp::<B>::novo(&dev);
    let mut otim = AdamConfig::new().init();
    let perda = CrossEntropyLossConfig::new().init(&dev);

    let mut passo = |model: Mlp<B>, otim: &mut _| -> Mlp<B> {
        let saida = model.forward(x.clone());
        let l = perda.forward(saida, y.clone());
        let grads = l.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        Optimizer::step(otim, 1e-3, model, grads)
    };

    for _ in 0..10 {
        model = passo(model, &mut otim);
    }
    <B as Backend>::sync(&dev);

    const PASSOS: usize = 200;
    let inicio = Instant::now();
    for _ in 0..PASSOS {
        model = passo(model, &mut otim);
    }
    <B as Backend>::sync(&dev);
    let dt = inicio.elapsed().as_secs_f64() / PASSOS as f64;

    let flop = (D * H + H * H + H * C) as f64 * 3.0 * 2.0 * lote as f64;
    println!("motor=burn-wgpu|lote={lote}|ms={:.4}|gflops={:.1}", dt * 1e3, flop / dt / 1e9);
}
