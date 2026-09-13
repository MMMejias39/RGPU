//! Quanto custa, em joules, treinar o mesmo modelo em cada motor e em cada
//! fabricante de GPU.
//!
//! Uso: cargo run -p rtensor --release --features gpu --example energia
//!
//! Tempo diz quão rápido; energia diz quanto custou. Um passo duas vezes mais
//! rápido consumindo três vezes mais potência é um retrocesso em eficiência, e
//! só a medição separa os dois casos.
//!
//! O mesmo código WGSL roda nas duas GPUs da máquina — é exatamente o ponto de
//! o backend não ser CUDA.

use rgpu_power::Medidor;
use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
use rtensor::prelude::*;

const CLASSES: usize = 10;
const D: usize = 512;
const H: usize = 1024;

/// Multiplicações-acumulações por amostra, ida e volta, em FLOP.
fn flop_por_passo(lote: usize) -> f64 {
    let macs = (D * H + H * H + H * CLASSES) as f64;
    macs * 3.0 * 2.0 * lote as f64
}

fn dados(lote: usize) -> (Tensor, Vec<usize>) {
    let x = Tensor::new(
        &[lote, D],
        (0..lote * D)
            .map(|i| ((i / D) as f32 * 0.01 + (i % D) as f32 * 0.03).sin())
            .collect(),
    );
    (x, (0..lote).map(|i| i % CLASSES).collect())
}

fn modelo(rng: &mut Rng) -> Sequential {
    Sequential::new()
        .add(Dense::new(D, H, Activation::Relu, rng))
        .add(Dense::new(H, H, Activation::Relu, rng))
        .add(Dense::new(H, CLASSES, Activation::Linear, rng))
}

struct Linha {
    motor: String,
    passos: usize,
    s_por_passo: f64,
    j_por_passo: Option<f64>,
    gflops: f64,
    gflop_por_j: Option<f64>,
}

/// Segundos que cada medição deve durar.
///
/// O `nvidia-smi` atualiza a leitura de potência a cada ~100 ms e ainda demora
/// a refletir uma mudança de carga. Janelas curtas medem sobretudo a inércia do
/// sensor: a primeira versão deste exemplo rodou 0,42 s e leu energia zero
/// acima da ociosidade, porque a placa mal tinha começado a subir de potência.
const JANELA_S: f64 = 6.0;

/// Converte energia por passo em eficiência, sem dividir por zero.
///
/// Energia nula não significa eficiência infinita — significa que a medição
/// não capturou nada e deve ser descartada.
fn eficiencia(flop: f64, j_por_passo: Option<f64>) -> Option<f64> {
    match j_por_passo {
        Some(j) if j > 1e-6 => Some(flop / j / 1e9),
        _ => None,
    }
}

fn treinar_gpu(medidor: &Medidor, gpu: &Gpu, lote: usize, nome: &str) -> Linha {
    let (x, rotulos) = dados(lote);
    let mut rng = Rng::new(7);
    let model = modelo(&mut rng);
    let mut mlp = GpuMlp::from_params(gpu, &model.params(), lote, 0.001);
    let gx = gpu.upload(&x);
    let gy = upload_labels(gpu, &rotulos);

    // Aquece: a primeira execução paga compilação de shader e alocação.
    for _ in 0..5 {
        mlp.step(gpu, &gx, &gy);
    }
    gpu.sync();

    // Calibra quantos passos cabem na janela alvo.
    let inicio = std::time::Instant::now();
    for _ in 0..20 {
        mlp.step(gpu, &gx, &gy);
    }
    gpu.sync();
    let por_passo = inicio.elapsed().as_secs_f64() / 20.0;
    let passos = ((JANELA_S / por_passo) as usize).clamp(20, 200_000);

    let (_, m) = medidor.medir(|| {
        for _ in 0..passos {
            mlp.step(gpu, &gx, &gy);
        }
        gpu.sync();
    });

    // O nvidia-smi só enxerga a placa da NVIDIA. Atribuir essa leitura a outro
    // adaptador contaria a potência ociosa da NVIDIA como consumo da Intel.
    let e_nvidia = gpu.info().to_lowercase().contains("nvidia");
    let j_por_passo = if e_nvidia {
        m.energia_gpu_j().map(|j| j / passos as f64)
    } else {
        // A GPU integrada vive no domínio `uncore` do RAPL.
        m.rapl_acima("uncore").map(|j| j / passos as f64)
    };

    let s_por_passo = m.duracao_s / passos as f64;
    let flop = flop_por_passo(lote);
    Linha {
        motor: nome.to_string(),
        passos,
        s_por_passo,
        j_por_passo,
        gflops: flop / s_por_passo / 1e9,
        gflop_por_j: eficiencia(flop, j_por_passo),
    }
}

fn treinar_cpu(medidor: &Medidor, lote: usize) -> Linha {
    let (x, rotulos) = dados(lote);
    let y = losses::one_hot(&rotulos, CLASSES);
    let mut rng = Rng::new(7);
    let model = modelo(&mut rng);
    let params = model.params();
    let mut opt = Adam::new(0.001);

    // Um passo para calibrar a janela.
    let inicio = std::time::Instant::now();
    {
        let tape = Tape::new();
        let entrada = constant(&tape, x.clone());
        let perda = losses::softmax_cross_entropy(&model.forward(&entrada), &y);
        opt.step(&params, &perda.backward());
    }
    let passos = ((JANELA_S / inicio.elapsed().as_secs_f64()) as usize).clamp(3, 100_000);

    let (_, m) = medidor.medir(|| {
        for _ in 0..passos {
            let tape = Tape::new();
            let entrada = constant(&tape, x.clone());
            let perda = losses::softmax_cross_entropy(&model.forward(&entrada), &y);
            opt.step(&params, &perda.backward());
        }
    });

    let s_por_passo = m.duracao_s / passos as f64;
    let flop = flop_por_passo(lote);
    // A energia da CPU vem do RAPL; sem root, não há leitura.
    let j_por_passo = m.rapl_acima("package-0").map(|j| j / passos as f64);
    Linha {
        motor: "rtensor CPU (1 núcleo)".into(),
        passos,
        s_por_passo,
        j_por_passo,
        gflops: flop / s_por_passo / 1e9,
        gflop_por_j: eficiencia(flop, j_por_passo),
    }
}

fn main() {
    let lote: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let mut medidor = Medidor::novo().com_intervalo(50);
    println!("fontes de medição: {}\n", medidor.fontes());

    println!("adaptadores visíveis:");
    for (nome, tipo, backend) in Gpu::listar() {
        println!("  {nome}  [{tipo}, {backend}]");
    }

    print!("\ncalibrando a potência ociosa (5 s)... ");
    medidor.calibrar_ociosidade(5.0);
    match medidor.ociosidade_w() {
        Some(w) => print!("NVIDIA {w:.2} W"),
        None => print!("NVIDIA indisponível"),
    }
    for (nome, w) in medidor.ociosidade_rapl_w() {
        print!("  |  {nome} {w:.2} W");
    }
    println!();

    let mut linhas = Vec::new();

    // As duas GPUs, com o mesmo código WGSL.
    for (filtro, nome) in [("nvidia", "rtensor GPU NVIDIA"), ("arc", "rtensor GPU Intel Arc")] {
        match Gpu::com_adaptador(filtro) {
            Ok(gpu) => {
                println!("\nmedindo em {} ...", gpu.info());
                linhas.push(treinar_gpu(&medidor, &gpu, lote, nome));
            }
            Err(e) => println!("\n{nome}: {e}"),
        }
    }

    println!("\nmedindo na CPU ...");
    linhas.push(treinar_cpu(&medidor, lote));

    println!("\nlote {lote}, MLP {D}→{H}→{H}→{CLASSES}\n");
    println!(
        "{:<24} {:>8} {:>11} {:>11} {:>11} {:>12}",
        "motor", "passos", "ms/passo", "J/passo", "GFLOP/s", "GFLOP/J"
    );
    for l in &linhas {
        let j = l
            .j_por_passo
            .map_or_else(|| format!("{:>11}", "—"), |v| format!("{v:>11.4}"));
        let e = l
            .gflop_por_j
            .map_or_else(|| format!("{:>12}", "—"), |v| format!("{v:>12.2}"));
        println!(
            "{:<24} {:>8} {:>11.4} {j} {:>11.1} {e}",
            l.motor,
            l.passos,
            l.s_por_passo * 1e3,
            l.gflops
        );
    }

    println!(
        "\nCada número é energia **acima da ociosidade**: NVIDIA pelo nvidia-smi,\n\
         CPU pelo domínio RAPL `package-0`, Intel Arc pelo `uncore`."
    );

    if !medidor.tem_rapl() {
        println!(
            "\nA energia da CPU e da GPU integrada vive no RAPL, que exige root desde\n\
             a mitigação do PLATYPUS (2020). Para incluí-las, rode como root ou\n\
             libere a leitura com:\n\
             \n    sudo chmod -R a+r /sys/class/powercap/intel-rapl*/energy_uj\n\
             \nIsso reabre um canal lateral de baixa largura de banda; numa máquina\n\
             pessoal o risco é pequeno, mas a escolha é sua."
        );
    }
}
