//! Confere o backend de GPU contra o motor de CPU.
//!
//! O caminho de CPU deriva tudo pela fita; o de GPU usa retropropagação escrita
//! à mão em WGSL. Partindo dos mesmos pesos e dos mesmos dados, os dois têm de
//! produzir os mesmos gradientes e a mesma trajetória de treino.
//!
//! Sem GPU disponível, os testes passam com um aviso em vez de falhar.
#![cfg(feature = "gpu")]

use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
use rtensor::prelude::*;

const N: usize = 64;
const D: usize = 8;
const H: usize = 16;
const C: usize = 4;

/// Dados com rótulo aprendível: a classe é o quadrante definido pelos sinais
/// das duas primeiras características — separável por uma MLP, mas não por um
/// modelo linear.
fn dados() -> (Tensor, Vec<usize>) {
    let x: Vec<f32> = (0..N * D).map(|i| ((i as f32) * 0.37).sin()).collect();
    let rotulos: Vec<usize> = (0..N)
        .map(|i| {
            let (a, b) = (x[i * D], x[i * D + 1]);
            usize::from(a > 0.0) + 2 * usize::from(b > 0.0)
        })
        .collect();
    (Tensor::new(&[N, D], x), rotulos)
}

fn modelo_cpu(rng: &mut Rng) -> Sequential {
    Sequential::new()
        .add(Dense::new(D, H, Activation::Relu, rng))
        .add(Dense::new(H, H, Activation::Relu, rng))
        .add(Dense::new(H, C, Activation::Linear, rng))
}

fn abrir() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => {
            eprintln!("GPU: {}", g.info());
            Some(g)
        }
        Err(e) => {
            eprintln!("sem GPU disponível ({e}) — teste ignorado");
            None
        }
    }
}

/// Distância relativa máxima entre dois tensores.
fn erro_rel(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.shape(), b.shape(), "shapes diferentes");
    a.data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| (x - y).abs() / (1.0 + x.abs().max(y.abs())))
        .fold(0.0, f32::max)
}

#[test]
fn gradientes_da_gpu_batem_com_os_da_cpu() {
    let Some(gpu) = abrir() else { return };
    let (x, rotulos) = dados();
    let alvos = losses::one_hot(&rotulos, C);

    let mut rng = Rng::new(42);
    let model = modelo_cpu(&mut rng);
    let params = model.params();

    // Um passo de avanço + retropropagação na CPU.
    let tape = Tape::new();
    let entrada = constant(&tape, x.clone());
    let perda_cpu = losses::softmax_cross_entropy(&model.forward(&entrada), &alvos);
    let grads_cpu = perda_cpu.backward();

    // O mesmo passo na GPU, a partir dos mesmos pesos.
    let mut mlp = GpuMlp::from_params(&gpu, &params, N, 0.0);
    let gx = gpu.upload(&x);
    let gy = upload_labels(&gpu, &rotulos);
    mlp.step(&gpu, &gx, &gy);

    let perda_gpu = mlp.last_loss(&gpu);
    assert!(
        (perda_cpu.value().item() - perda_gpu).abs() < 1e-4,
        "perda: CPU {} vs GPU {}",
        perda_cpu.value().item(),
        perda_gpu
    );

    for (p, g_gpu) in params.iter().zip(mlp.grads(&gpu)) {
        let g_cpu = grads_cpu.of(p).expect("parâmetro sem gradiente");
        let e = erro_rel(g_cpu, &g_gpu);
        assert!(e < 2e-4, "gradiente de {} diverge: erro relativo {e:.2e}", p.name());
    }
}

#[test]
fn treino_na_gpu_segue_a_mesma_trajetoria_da_cpu() {
    let Some(gpu) = abrir() else { return };
    let (x, rotulos) = dados();
    let alvos = losses::one_hot(&rotulos, C);

    let mut rng = Rng::new(7);
    let model = modelo_cpu(&mut rng);
    let params = model.params();

    let mut mlp = GpuMlp::from_params(&gpu, &params, N, 0.05);
    let gx = gpu.upload(&x);
    let gy = upload_labels(&gpu, &rotulos);

    let mut opt = Adam::new(0.05);
    let mut perda_cpu = 0.0;

    for passo in 1..=120 {
        let tape = Tape::new();
        let entrada = constant(&tape, x.clone());
        let perda = losses::softmax_cross_entropy(&model.forward(&entrada), &alvos);
        perda_cpu = perda.value().item();
        opt.step(&params, &perda.backward());

        mlp.step(&gpu, &gx, &gy);
        let perda_gpu = mlp.last_loss(&gpu);

        assert!(
            (perda_cpu - perda_gpu).abs() < 2e-3,
            "passo {passo}: CPU {perda_cpu:.6} vs GPU {perda_gpu:.6}"
        );
    }

    // Depois de 120 passos os pesos ainda têm de coincidir.
    for (p, w_gpu) in params.iter().zip(mlp.weights(&gpu)) {
        let e = erro_rel(&p.value(), &w_gpu);
        assert!(e < 2e-3, "peso {} divergiu: erro relativo {e:.2e}", p.name());
    }

    assert!(perda_cpu < 0.15, "o treino não convergiu: perda {perda_cpu}");
}

/// O mesmo conferência da trajetória, na forma que aciona o Split-K no plano
/// de treino: lote grande com camada de saída estreita — `dW = entradaᵀ·δ` tem
/// `K` igual ao lote e sai com pouquíssimos blocos, exatamente o caso
/// patológico que a partição corrige.
#[test]
fn treino_com_camada_estreita_usa_split_k_e_segue_a_cpu() {
    const LOTE: usize = 512;
    const DD: usize = 8;
    const HH: usize = 128;
    const CC: usize = 4;

    let Some(gpu) = abrir() else { return };

    // Mesma construção de dados de `dados()`, com mais amostras — o lote grande
    // é o que liga o particionamento: no `dW` de saída (128×4, K=512) são 8
    // fatias, e no forward da saída (512×4, K=128), 2.
    let xs: Vec<f32> = (0..LOTE * DD).map(|i| ((i as f32) * 0.37).sin()).collect();
    let rotulos: Vec<usize> = (0..LOTE)
        .map(|i| {
            let (a, b) = (xs[i * DD], xs[i * DD + 1]);
            usize::from(a > 0.0) + 2 * usize::from(b > 0.0)
        })
        .collect();
    let x = Tensor::new(&[LOTE, DD], xs);
    let alvos = losses::one_hot(&rotulos, CC);

    let mut rng = Rng::new(7);
    let model = Sequential::new()
        .add(Dense::new(DD, HH, Activation::Relu, &mut rng))
        .add(Dense::new(HH, CC, Activation::Linear, &mut rng));
    let params = model.params();

    let mut mlp = GpuMlp::from_params(&gpu, &params, LOTE, 0.05);

    // O plano tem de ter acionado a partição, ou este teste não testa o que
    // diz: 4 no avanço (mm_bias + produto/redução/bias_add da saída), 11 no
    // backward (softmax + por camada: produto/redução do dW + colsum, mais
    // dA/relu na oculta) e 4 no Adam.
    assert_eq!(
        mlp.dispatches_por_passo(),
        19,
        "o plano não acionou o Split-K como esperado"
    );

    let gx = gpu.upload(&x);
    let gy = upload_labels(&gpu, &rotulos);

    let mut opt = Adam::new(0.05);
    let mut perda_cpu = 0.0;

    for passo in 1..=60 {
        let tape = Tape::new();
        let entrada = constant(&tape, x.clone());
        let perda = losses::softmax_cross_entropy(&model.forward(&entrada), &alvos);
        perda_cpu = perda.value().item();
        opt.step(&params, &perda.backward());

        mlp.step(&gpu, &gx, &gy);
        let perda_gpu = mlp.last_loss(&gpu);

        assert!(
            (perda_cpu - perda_gpu).abs() < 2e-3,
            "passo {passo}: CPU {perda_cpu:.6} vs GPU {perda_gpu:.6}"
        );
    }

    for (p, w_gpu) in params.iter().zip(mlp.weights(&gpu)) {
        let e = erro_rel(&p.value(), &w_gpu);
        assert!(e < 2e-3, "peso {} divergiu: erro relativo {e:.2e}", p.name());
    }

    assert!(perda_cpu < 0.2, "o treino não convergiu: perda {perda_cpu}");
}

#[test]
fn ida_e_volta_preserva_o_tensor() {
    let Some(gpu) = abrir() else { return };
    let t = Tensor::new(&[3, 5], (0..15).map(|i| i as f32 * 0.5 - 3.0).collect());
    let volta = gpu.download(&gpu.upload(&t));
    assert_eq!(t, volta);
}

#[test]
fn matmul_da_gpu_bate_com_o_da_cpu() {
    let Some(gpu) = abrir() else { return };
    let a = Tensor::new(&[37, 23], (0..37 * 23).map(|i| (i as f32 * 0.11).sin()).collect());
    let b = Tensor::new(&[23, 19], (0..23 * 19).map(|i| (i as f32 * 0.07).cos()).collect());

    let ga = gpu.upload(&a);
    let gb = gpu.upload(&b);
    let gc = gpu.zeros(&[37, 19]);

    let mut enc = gpu.encoder();
    gpu.matmul(&mut enc, &ga, &gb, &gc);
    gpu.submit(enc);

    // Dimensões não múltiplas de 16 exercitam o recorte dos ladrilhos.
    let e = erro_rel(&a.matmul(&b), &gpu.download(&gc));
    assert!(e < 1e-5, "matmul diverge: erro relativo {e:.2e}");
}

/// Exercita as três variantes do GEMM em dimensões que não são múltiplas do
/// ladrilho, comparando os dois kernels entre si e ambos com a CPU.
#[test]
fn gemm_confere_em_todas_as_variantes_e_dimensoes() {
    let Some(mut gpu) = abrir() else { return };

    // Inclui casos degenerados (1), abaixo do ladrilho (17), exatamente no
    // ladrilho (64), logo acima (65) e sem alinhamento nenhum (130, 199).
    let casos = [
        (1, 1, 1),
        (17, 33, 65),
        (64, 64, 64),
        (65, 64, 63),
        (130, 96, 80),
        (199, 130, 71),
        (256, 512, 128),
    ];

    for (m, n, k) in casos {
        let a = Tensor::new(&[m, k], (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect());
        let b = Tensor::new(&[k, n], (0..k * n).map(|i| (i as f32 * 0.007).cos()).collect());
        let bt = b.t(); // [n, k], para a variante A·Bᵀ
        let at = a.t(); // [k, m], para a variante Aᵀ·B

        let esperado = a.matmul(&b);

        for rapido in [false, true] {
            gpu.set_fast_gemm(rapido);
            let rotulo = if rapido { "ladrilhado" } else { "ingênuo" };

            let ga = gpu.upload(&a);
            let gb = gpu.upload(&b);
            let gat = gpu.upload(&at);
            let gbt = gpu.upload(&bt);

            let c1 = gpu.zeros(&[m, n]);
            let c2 = gpu.zeros(&[m, n]);
            let c3 = gpu.zeros(&[m, n]);
            let mut enc = gpu.encoder();
            gpu.matmul(&mut enc, &ga, &gb, &c1);
            gpu.matmul_at_b(&mut enc, &gat, &gb, &c2);
            gpu.matmul_a_bt(&mut enc, &ga, &gbt, &c3);
            gpu.submit(enc);

            for (nome, saida) in [("mm", &c1), ("mm_atb", &c2), ("mm_abt", &c3)] {
                let e = erro_rel(&esperado, &gpu.download(saida));
                assert!(
                    e < 1e-5,
                    "{rotulo}/{nome} em {m}×{n}×{k}: erro relativo {e:.2e}"
                );
            }
        }
    }
}

/// O treino tem de dar o mesmo resultado com qualquer um dos dois kernels.
#[test]
fn os_dois_gemms_treinam_igual() {
    let Some(mut gpu) = abrir() else { return };
    let (x, rotulos) = dados();

    let mut pesos_finais = Vec::new();
    for rapido in [false, true] {
        gpu.set_fast_gemm(rapido);
        let mut rng = Rng::new(7);
        let model = modelo_cpu(&mut rng);
        let mut mlp = GpuMlp::from_params(&gpu, &model.params(), N, 0.05);
        let gx = gpu.upload(&x);
        let gy = upload_labels(&gpu, &rotulos);
        for _ in 0..60 {
            mlp.step(&gpu, &gx, &gy);
        }
        pesos_finais.push(mlp.weights(&gpu));
    }

    for (i, (lento, rapido)) in pesos_finais[0].iter().zip(&pesos_finais[1]).enumerate() {
        let e = erro_rel(lento, rapido);
        assert!(e < 1e-3, "tensor {i}: os dois GEMMs divergiram, erro {e:.2e}");
    }
}

/// Strassen precisa acertar a conta e, sendo menos estável que o algoritmo
/// clássico, o quanto ele perde em precisão é medido aqui em vez de suposto.
#[test]
fn strassen_confere_e_o_erro_extra_e_medido() {
    use rtensor::gpu::Strassen;
    let Some(gpu) = abrir() else { return };

    for (m, n, k) in [(64usize, 64usize, 64usize), (128, 256, 128), (512, 512, 512)] {
        let a = Tensor::new(&[m, k], (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect());
        let b = Tensor::new(&[k, n], (0..k * n).map(|i| (i as f32 * 0.011).cos()).collect());
        let esperado = a.matmul(&b);

        let ga = gpu.upload(&a);
        let gb = gpu.upload(&b);
        let c_classico = gpu.zeros(&[m, n]);
        let c_strassen = gpu.zeros(&[m, n]);

        let mut enc = gpu.encoder();
        gpu.matmul(&mut enc, &ga, &gb, &c_classico);
        gpu.submit(enc);

        let plano = Strassen::novo(&gpu, &ga, &gb, &c_strassen).expect("plano");
        let mut enc = gpu.encoder();
        plano.executar(&gpu, &mut enc);
        gpu.submit(enc);

        let e_classico = erro_rel(&esperado, &gpu.download(&c_classico));
        let e_strassen = erro_rel(&esperado, &gpu.download(&c_strassen));

        eprintln!(
            "{m}×{n}×{k}: clássico {e_classico:.2e}, Strassen {e_strassen:.2e} \
             ({:.1}× o erro), {} dispatches",
            e_strassen / e_classico.max(1e-12),
            plano.dispatches()
        );

        // Strassen é menos estável, mas não a ponto de estar errado.
        assert!(
            e_strassen < 1e-4,
            "{m}×{n}×{k}: Strassen com erro {e_strassen:.2e}"
        );
    }
}

#[test]
fn strassen_recusa_dimensoes_impares() {
    use rtensor::gpu::Strassen;
    let Some(gpu) = abrir() else { return };
    let a = gpu.zeros(&[15, 8]);
    let b = gpu.zeros(&[8, 8]);
    let c = gpu.zeros(&[15, 8]);
    let erro = match Strassen::novo(&gpu, &a, &b, &c) {
        Err(e) => e,
        Ok(_) => panic!("deveria ter recusado dimensão ímpar"),
    };
    assert!(erro.contains("pares"), "mensagem inesperada: {erro}");
}

/// O GEMM particionado precisa dar o mesmo resultado do inteiro, em formas
/// K-dominantes e com números de fatias diferentes.
#[test]
fn split_k_confere_com_a_cpu() {
    let Some(gpu) = abrir() else { return };

    // (M, N, K) — formas onde a saída é pequena e a dimensão interna é grande.
    for (m, n, k) in [(64usize, 64usize, 4096usize), (32, 96, 2048), (128, 64, 1024)] {
        let a = Tensor::new(&[m, k], (0..m * k).map(|i| (i as f32 * 0.013).sin()).collect());
        let b = Tensor::new(&[k, n], (0..k * n).map(|i| (i as f32 * 0.007).cos()).collect());
        let esperado = a.matmul(&b);
        let at = a.t(); // [k, m], para a variante transposta

        let ga = gpu.upload(&a);
        let gat = gpu.upload(&at);
        let gb = gpu.upload(&b);

        // Referência: o mesmo GEMM sem particionar, na mesma GPU. É contra ela
        // que o particionado é julgado — a diferença para a CPU inclui o erro
        // de ordem de soma, que existe nos dois.
        let c_inteiro = gpu.zeros(&[m, n]);
        let mut enc = gpu.encoder();
        gpu.matmul(&mut enc, &ga, &gb, &c_inteiro);
        gpu.submit(enc);
        let e_inteiro = erro_rel(&esperado, &gpu.download(&c_inteiro));

        for fatias in [1usize, 2, 4, 8, 16] {
            for (transposta, entrada) in [(false, &ga), (true, &gat)] {
                let c = gpu.zeros(&[m, n]);
                let parciais = gpu.zeros(&[fatias * m, n]);
                let mut enc = gpu.encoder();
                for op in gpu.ops_split_k(entrada, &gb, &c, &parciais, fatias, transposta) {
                    gpu.record(&mut enc, &op);
                }
                gpu.submit(enc);

                let e = erro_rel(&esperado, &gpu.download(&c));
                // Particionar muda a ordem das somas, então o erro muda — mas
                // não deve degradar. O fator 3 cobre a variação de ordenação
                // sem deixar passar um erro de indexação, que daria ordens de
                // grandeza.
                assert!(
                    e < (e_inteiro * 3.0).max(2e-5),
                    "{m}×{n}×{k}, {fatias} fatias, transposta={transposta}: \
                     erro {e:.2e} contra {e_inteiro:.2e} do GEMM inteiro"
                );
            }
        }
    }
}

#[test]
fn fatias_sugeridas_respeita_a_forma() {
    use rtensor::gpu::Gpu;
    // Saída pequena e K grande: vale particionar.
    assert!(Gpu::fatias_sugeridas(64, 64, 8192) > 1);
    // Saída grande: já há paralelismo, não particiona.
    assert_eq!(Gpu::fatias_sugeridas(4096, 4096, 4096), 1);
    // K curto: não há o que fatiar.
    assert_eq!(Gpu::fatias_sugeridas(64, 64, 64), 1);
}

/// Precisão mista tem de acertar a conta, e o quanto ela perde em precisão é
/// medido — `f16` guarda ~3 dígitos decimais contra ~7 do `f32`.
#[test]
fn precisao_mista_confere_e_o_erro_e_medido() {
    let Some(gpu) = abrir() else { return };

    for (m, n, k) in [(64usize, 64usize, 64usize), (256, 128, 512), (512, 512, 256)] {
        let a = Tensor::new(&[m, k], (0..m * k).map(|i| (i as f32 * 0.019).sin()).collect());
        let b = Tensor::new(&[k, n], (0..k * n).map(|i| (i as f32 * 0.013).cos()).collect());
        let esperado = a.matmul(&b);
        let bt = b.t();

        let (a16, _a32, op_a) = gpu.upload_f16(&a);
        let (b16, _b32, op_b) = gpu.upload_f16(&b);
        let (bt16, _bt32, op_bt) = gpu.upload_f16(&bt);
        let at = a.t();
        let (at16, _at32, op_at) = gpu.upload_f16(&at);

        let c32 = gpu.zeros(&[m, n]);
        let ga = gpu.upload(&a);
        let gb = gpu.upload(&b);
        let mut enc = gpu.encoder();
        gpu.matmul(&mut enc, &ga, &gb, &c32);
        gpu.submit(enc);
        let e32 = erro_rel(&esperado, &gpu.download(&c32));

        // As três combinações que a retropropagação usa.
        for (nome, a_op, b_op, tr, btr) in [
            ("A·B", &a16, &b16, false, false),
            ("Aᵀ·B", &at16, &b16, true, false),
            ("A·Bᵀ", &a16, &bt16, false, true),
        ] {
            let c = gpu.zeros(&[m, n]);
            let mut enc = gpu.encoder();
            for op in [&op_a, &op_b, &op_bt, &op_at] {
                gpu.record(&mut enc, op);
            }
            let mm = gpu.op_matmul_f16(a_op, b_op, &c, tr, btr);
            gpu.record(&mut enc, &mm);
            gpu.submit(enc);

            let e16 = erro_rel(&esperado, &gpu.download(&c));

            // `f16` tem 11 bits de mantissa, então cada elemento já entra com
            // erro relativo ~2⁻¹¹. Somando `K` termos, o erro esperado cresce
            // com `√K`. O limite abaixo é dez vezes esse valor: largo o
            // bastante para o ruído da meia precisão, apertado o bastante para
            // pegar erro de indexação, que daria ordens de grandeza.
            let esperado_f16 = 10.0 * (k as f32).sqrt() * 2.0f32.powi(-11);
            eprintln!(
                "{m}×{n}×{k} {nome}: f32 {e32:.2e}, mista {e16:.2e} \
                 ({:.0}× o erro, limite teórico {esperado_f16:.2e})",
                e16 / e32.max(1e-12)
            );
            assert!(
                e16 < esperado_f16,
                "{m}×{n}×{k} {nome}: erro {e16:.2e} acima do esperado para f16"
            );
        }
    }
}
