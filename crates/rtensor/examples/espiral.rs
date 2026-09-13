//! Classificação em 3 classes sobre o dataset das espirais entrelaçadas —
//! não separável linearmente, o teste clássico para uma MLP com ReLU.
//!
//! Execute com: cargo run --release --example espiral

use rtensor::prelude::*;

const CLASSES: usize = 3;
const POR_CLASSE: usize = 220;
const LOTE: usize = 32;

/// Gera espirais: cada classe é um braço girando com raio crescente.
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
    let mut rng = Rng::new(7);
    let (x, rotulos) = espirais(&mut rng);
    let n = rotulos.len();

    // Separação treino/teste depois de embaralhar.
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

    print!("{}", model.summary());
    println!("  treino: {} amostras | teste: {} amostras\n", corte, n - corte);

    let params = model.params();
    let mut opt = Adam::new(0.01);
    let mut posicoes: Vec<usize> = (0..corte).collect();

    println!("época    perda    acc treino   acc teste");
    for epoca in 1..=60 {
        rng.shuffle(&mut posicoes);
        let mut soma = 0.0;
        let mut lotes = 0;

        for inicio in (0..corte).step_by(LOTE) {
            let fim = (inicio + LOTE).min(corte);
            let bx = x_treino.take_rows(&posicoes[inicio..fim]);
            let rot: Vec<usize> = posicoes[inicio..fim].iter().map(|&i| y_treino[i]).collect();
            let by = losses::one_hot(&rot, CLASSES);

            let tape = Tape::new();
            let entrada = constant(&tape, bx);
            let logits = model.forward(&entrada);
            let perda = losses::softmax_cross_entropy(&logits, &by);

            soma += perda.value().item();
            lotes += 1;
            opt.step(&params, &perda.backward());
        }

        if epoca % 5 == 0 || epoca == 1 {
            let acc_treino = losses::accuracy(&model.predict(&x_treino), &y_treino);
            let acc_teste = losses::accuracy(&model.predict(&x_teste), &y_teste);
            println!(
                "{epoca:>5}   {:.4}      {:>6.2}%      {:>6.2}%",
                soma / lotes as f32,
                acc_treino * 100.0,
                acc_teste * 100.0
            );
        }
    }

    println!("\nFronteira de decisão aprendida (região 2-D, uma letra por classe):");
    desenhar(&model);
}

/// Desenha a fronteira de decisão em ASCII avaliando uma grade de pontos.
fn desenhar(model: &Sequential) {
    const L: usize = 29;
    let marcas = [b'.', b'o', b'#'];
    let mut pontos = Vec::with_capacity(L * L * 2);
    for linha in 0..L {
        for col in 0..L {
            pontos.push(col as f32 / (L - 1) as f32 * 2.4 - 1.2);
            pontos.push(1.2 - linha as f32 / (L - 1) as f32 * 2.4);
        }
    }
    let grade = Tensor::new(&[L * L, 2], pontos);
    let classe = model.predict(&grade).argmax_rows();

    for linha in 0..L {
        let s: String = (0..L).map(|c| marcas[classe[linha * L + c]] as char).collect();
        println!("  {s}");
    }
}
