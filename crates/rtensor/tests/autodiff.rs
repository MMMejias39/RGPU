//! Verificação do motor: álgebra do tensor e conferência numérica dos gradientes.

use rtensor::prelude::*;

/// Gradiente numérico por diferenças centrais: (f(x+h) - f(x-h)) / 2h.
fn numeric_grad(p: &Param, i: usize, mut loss_fn: impl FnMut() -> f32) -> f32 {
    let h = 1e-3f32;
    let orig = p.value().data()[i];

    p.update(|t| t.data_mut()[i] = orig + h);
    let up = loss_fn();
    p.update(|t| t.data_mut()[i] = orig - h);
    let down = loss_fn();
    p.update(|t| t.data_mut()[i] = orig);

    (up - down) / (2.0 * h)
}

fn check_gradients(params: &[Param], analytic: &Grads, mut loss_fn: impl FnMut() -> f32) {
    for p in params {
        let n = p.value().len();
        let g = analytic.of(p).expect("parâmetro sem gradiente").clone();
        for i in 0..n {
            let num = numeric_grad(p, i, &mut loss_fn);
            let ana = g.data()[i];
            let tol = 2e-2 * (1.0 + ana.abs().max(num.abs()));
            assert!(
                (num - ana).abs() < tol,
                "gradiente divergente em {}[{}]: numérico {:.6} vs autodiff {:.6}",
                p.name(),
                i,
                num,
                ana
            );
        }
    }
}

#[test]
fn matmul_confere_com_calculo_manual() {
    let a = Tensor::new(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
    let b = Tensor::new(&[3, 2], vec![7., 8., 9., 10., 11., 12.]);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[58., 64., 139., 154.]);
}

#[test]
fn broadcasting_soma_linha_em_todo_o_lote() {
    let x = Tensor::new(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
    let b = Tensor::new(&[1, 3], vec![10., 20., 30.]);
    assert_eq!(x.add(&b).data(), &[11., 22., 33., 14., 25., 36.]);
}

#[test]
fn reduce_to_desfaz_o_broadcasting() {
    let g = Tensor::new(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
    assert_eq!(g.reduce_to(&[1, 3]).data(), &[5., 7., 9.]);
    assert_eq!(g.reduce_to(&[2, 1]).data(), &[6., 15.]);
}

#[test]
fn softmax_normaliza_cada_linha() {
    let z = Tensor::new(&[2, 3], vec![1., 2., 3., 100., 100., 100.]);
    let s = z.softmax_rows();
    for i in 0..2 {
        let soma: f32 = (0..3).map(|j| s.data()[i * 3 + j]).sum();
        assert!((soma - 1.0).abs() < 1e-5, "linha {i} soma {soma}");
    }
    // Sem a estabilização pelo máximo, a segunda linha estouraria para NaN.
    assert!(s.data().iter().all(|v| v.is_finite()));
}

#[test]
fn transposta_e_involutiva() {
    let a = Tensor::new(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
    assert_eq!(a.t().t(), a);
    assert_eq!(a.t().shape(), &[3, 2]);
}

#[test]
fn gradiente_de_mlp_com_mse_bate_com_o_numerico() {
    let mut rng = Rng::new(11);
    let model = Sequential::new()
        .add(Dense::new(3, 4, Activation::Tanh, &mut rng))
        .add(Dense::new(4, 2, Activation::Linear, &mut rng));

    let x = Tensor::new(&[5, 3], (0..15).map(|i| (i as f32 * 0.37).sin()).collect());
    let y = Tensor::new(&[5, 2], (0..10).map(|i| (i as f32 * 0.21).cos()).collect());

    let forward = |m: &Sequential| -> f32 {
        let tape = Tape::new();
        let input = constant(&tape, x.clone());
        losses::mse(&m.forward(&input), &y).value().item()
    };

    let tape = Tape::new();
    let input = constant(&tape, x.clone());
    let grads = losses::mse(&model.forward(&input), &y).backward();

    check_gradients(&model.params(), &grads, || forward(&model));
}

#[test]
fn gradiente_de_entropia_cruzada_bate_com_o_numerico() {
    let mut rng = Rng::new(23);
    let model = Sequential::new()
        .add(Dense::new(4, 6, Activation::Relu, &mut rng))
        .add(Dense::new(6, 3, Activation::Linear, &mut rng));

    let x = Tensor::new(&[8, 4], (0..32).map(|i| (i as f32 * 0.13).cos() * 1.5).collect());
    let alvos = losses::one_hot(&[0, 1, 2, 1, 0, 2, 2, 0], 3);

    let forward = |m: &Sequential| -> f32 {
        let tape = Tape::new();
        let input = constant(&tape, x.clone());
        losses::softmax_cross_entropy(&m.forward(&input), &alvos).value().item()
    };

    let tape = Tape::new();
    let input = constant(&tape, x.clone());
    let grads = losses::softmax_cross_entropy(&model.forward(&input), &alvos).backward();

    check_gradients(&model.params(), &grads, || forward(&model));
}

#[test]
fn gradiente_de_sigmoid_com_bce_bate_com_o_numerico() {
    let mut rng = Rng::new(5);
    let model = Sequential::new()
        .add(Dense::new(2, 5, Activation::Sigmoid, &mut rng))
        .add(Dense::new(5, 1, Activation::Sigmoid, &mut rng));

    let x = Tensor::new(&[4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]);
    let y = Tensor::new(&[4, 1], vec![0., 1., 1., 0.]);

    let forward = |m: &Sequential| -> f32 {
        let tape = Tape::new();
        let input = constant(&tape, x.clone());
        losses::binary_cross_entropy(&m.forward(&input), &y).value().item()
    };

    let tape = Tape::new();
    let input = constant(&tape, x.clone());
    let grads = losses::binary_cross_entropy(&model.forward(&input), &y).backward();

    check_gradients(&model.params(), &grads, || forward(&model));
}

#[test]
fn gradiente_de_softmax_explicito_bate_com_o_numerico() {
    let mut rng = Rng::new(31);
    let model = Sequential::new().add(Dense::new(3, 3, Activation::Linear, &mut rng));
    let x = Tensor::new(&[4, 3], (0..12).map(|i| (i as f32 * 0.29).sin()).collect());
    let y = losses::one_hot(&[2, 0, 1, 2], 3);

    // -média(y * log softmax(z)) montado peça por peça, para exercitar softmax().
    let forward = |m: &Sequential| -> f32 {
        let tape = Tape::new();
        let input = constant(&tape, x.clone());
        let p = m.forward(&input).softmax();
        let alvo = constant(&tape, y.clone());
        p.ln().mul(&alvo).sum().scale(-0.25).value().item()
    };

    let tape = Tape::new();
    let input = constant(&tape, x.clone());
    let p = model.forward(&input).softmax();
    let alvo = constant(&tape, y.clone());
    let grads = p.ln().mul(&alvo).sum().scale(-0.25).backward();

    check_gradients(&model.params(), &grads, || forward(&model));
}

#[test]
fn gradiente_acumula_quando_a_variavel_e_reutilizada() {
    // f(w) = sum(w * w + w) => df/dw = 2w + 1, exige acumulação no nó-folha.
    let w = Param::new("w", Tensor::new(&[1, 3], vec![1.0, 2.0, 3.0]));
    let tape = Tape::new();
    let v = w.watch(&tape);
    let f = v.mul(&v).add(&v).sum();
    let g = f.backward();
    assert_eq!(g.of(&w).unwrap().data(), &[3.0, 5.0, 7.0]);
}

#[test]
fn xor_converge() {
    let mut rng = Rng::new(2024);
    let model = Sequential::new()
        .add(Dense::new(2, 8, Activation::Tanh, &mut rng))
        .add(Dense::new(8, 1, Activation::Sigmoid, &mut rng));

    let x = Tensor::new(&[4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]);
    let y = Tensor::new(&[4, 1], vec![0., 1., 1., 0.]);
    let mut opt = Adam::new(0.08);
    let params = model.params();

    let mut perda = f32::INFINITY;
    for _ in 0..1500 {
        let tape = Tape::new();
        let input = constant(&tape, x.clone());
        let loss = losses::binary_cross_entropy(&model.forward(&input), &y);
        perda = loss.value().item();
        opt.step(&params, &loss.backward());
    }

    assert!(perda < 0.05, "XOR não convergiu: perda final {perda}");
    let p = model.predict(&x);
    for (saida, alvo) in p.data().iter().zip(y.data()) {
        assert_eq!(*saida > 0.5, *alvo > 0.5, "previsão {saida} para alvo {alvo}");
    }
}

#[test]
fn sgd_com_momentum_reduz_a_perda() {
    let w = Param::new("w", Tensor::new(&[1, 2], vec![4.0, -3.0]));
    let mut opt = Sgd::with_momentum(0.05, 0.9);
    let quad = || {
        let tape = Tape::new();
        let v = w.watch(&tape);
        v.square().sum()
    };

    let inicial = quad().value().item();
    for _ in 0..60 {
        let loss = quad();
        opt.step(std::slice::from_ref(&w), &loss.backward());
    }
    let final_ = quad().value().item();
    assert!(final_ < inicial * 1e-2, "de {inicial} para {final_}");
}
