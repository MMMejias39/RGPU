//! Empurrar o clock para cima compra o quê, por joule?
//!
//! Uso: sudo -E cargo run -p rtensor --release --features gpu --example frequencia
//!      (ou rode o binário já compilado com sudo)
//!
//! Trava a GPU em cada uma de várias frequências e mede a mesma carga em todas.
//! Se a eficiência tiver pico bem abaixo do topo da faixa, então a escalada de
//! frequência — e a de potência que vem junto — compra retorno decrescente.
//!
//! A trava se desfaz sozinha ao fim de cada ponto e no encerramento. Se o
//! processo for morto por sinal, o conserto é `sudo nvidia-smi -rgc`.

use std::time::Duration;

use rgpu_power::{clock, Medidor};
use rtensor::gpu::{upload_labels, Gpu, GpuMlp};
use rtensor::prelude::*;

const CLASSES: usize = 10;
const D: usize = 512;
const H: usize = 1024;
const LOTE: usize = 512;
const JANELA_S: f64 = 4.0;

fn flop_por_passo() -> f64 {
    (D * H + H * H + H * CLASSES) as f64 * 3.0 * 2.0 * LOTE as f64
}

struct Ponto {
    pedido_mhz: u32,
    efetivo_mhz: Option<u32>,
    ms_passo: f64,
    media_w: f64,
    j_passo: f64,
    gflops: f64,
    gflop_j: f64,
}

// A varredura usa **energia total**, não "acima da ociosidade".
//
// A linha de base é função do ponto de operação: uma placa travada em 210 MHz
// consome menos trabalhando do que a mesma placa ociosa em 1980 MHz, e a
// subtração vira negativa. Medido nesta máquina: 12,3 W em carga a 210 MHz
// contra 14,15 W de ociosidade calibrada no clock padrão. Como a base muda
// junto com o que se quer comparar, ela deixa de ser subtraível.

fn main() {
    if !clock::e_root() {
        eprintln!(
            "Travar a frequência exige root — o nvidia-smi recusa por permissão.\n\
             Compile primeiro e rode o binário com sudo:\n\n    \
             cargo build -p rtensor --release --features gpu --example frequencia\n    \
             sudo ./target/release/examples/frequencia\n"
        );
        std::process::exit(1);
    }

    let frequencias = clock::amostrar(9);
    if frequencias.is_empty() {
        eprintln!("nenhuma frequência suportada foi reportada pelo nvidia-smi");
        std::process::exit(1);
    }

    let gpu = Gpu::com_adaptador("nvidia").expect("GPU NVIDIA não encontrada");
    println!("{}", gpu.info());
    println!("modelo {D}→{H}→{H}→{CLASSES}, lote {LOTE}");
    println!(
        "varrendo {} frequências, de {} a {} MHz\n",
        frequencias.len(),
        frequencias[0],
        frequencias[frequencias.len() - 1]
    );

    let mut medidor = Medidor::novo().com_intervalo(50);
    medidor.calibrar_ociosidade(4.0);
    println!(
        "ociosidade: {:.2} W\n",
        medidor.ociosidade_w().unwrap_or(0.0)
    );

    // Prepara o modelo uma vez; só a frequência muda entre os pontos.
    let x = Tensor::new(
        &[LOTE, D],
        (0..LOTE * D)
            .map(|i| ((i / D) as f32 * 0.01 + (i % D) as f32 * 0.03).sin())
            .collect(),
    );
    let rotulos: Vec<usize> = (0..LOTE).map(|i| i % CLASSES).collect();
    let mut rng = Rng::new(7);
    let model = Sequential::new()
        .add(Dense::new(D, H, Activation::Relu, &mut rng))
        .add(Dense::new(H, H, Activation::Relu, &mut rng))
        .add(Dense::new(H, CLASSES, Activation::Linear, &mut rng));
    let mut mlp = GpuMlp::from_params(&gpu, &model.params(), LOTE, 0.001);
    let gx = gpu.upload(&x);
    let gy = upload_labels(&gpu, &rotulos);

    let mut pontos = Vec::new();

    for &mhz in &frequencias {
        let trava = match clock::TravaClock::travar(mhz) {
            Ok(t) => t,
            Err(e) => {
                println!("{mhz:>5} MHz: não foi possível travar — {e}");
                continue;
            }
        };
        // Deixa a frequência assentar antes de medir.
        std::thread::sleep(Duration::from_millis(600));

        for _ in 0..20 {
            mlp.step(&gpu, &gx, &gy);
        }
        gpu.sync();

        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            mlp.step(&gpu, &gx, &gy);
        }
        gpu.sync();
        let por_passo = t0.elapsed().as_secs_f64() / 20.0;
        let passos = ((JANELA_S / por_passo) as usize).clamp(20, 500_000);

        let efetivo = clock::atual();
        let (_, m) = medidor.medir(|| {
            for _ in 0..passos {
                mlp.step(&gpu, &gx, &gy);
            }
            gpu.sync();
        });
        drop(trava);

        let Some(energia) = m.energia_gpu_total_j() else {
            println!("{mhz:>5} MHz: medição de energia falhou");
            continue;
        };
        let ms_passo = m.duracao_s / passos as f64 * 1e3;
        let j_passo = energia / passos as f64;
        let flop = flop_por_passo();

        let p = Ponto {
            pedido_mhz: mhz,
            efetivo_mhz: efetivo,
            ms_passo,
            media_w: m.gpu.as_ref().map_or(0.0, |g| g.media_w),
            j_passo,
            gflops: flop / (ms_passo / 1e3) / 1e9,
            gflop_j: if j_passo > 1e-6 { flop / j_passo / 1e9 } else { 0.0 },
        };
        println!(
            "  {:>5} MHz medido (efetivo {:>5}) — {:>7.2} GFLOP/J",
            p.pedido_mhz,
            p.efetivo_mhz.map_or("?".into(), |v| v.to_string()),
            p.gflop_j
        );
        pontos.push(p);
    }

    clock::liberar();

    if pontos.is_empty() {
        eprintln!("\nnenhum ponto medido");
        return;
    }

    let ignorado: Vec<&Ponto> = pontos
        .iter()
        .filter(|p| p.efetivo_mhz.is_some_and(|e| e + 20 < p.pedido_mhz))
        .collect();
    if !ignorado.is_empty() {
        println!(
            "\naviso: o firmware não entregou {} das frequências pedidas —\n\
             esses pontos são o mesmo ponto de operação, não pontos distintos.",
            ignorado.len()
        );
    }

    println!(
        "\n{:>10} {:>10} {:>10} {:>10} {:>10} {:>11} {:>9}",
        "pedido", "efetivo", "ms/passo", "média W", "J total", "GFLOP/s", "GFLOP/J"
    );
    for p in &pontos {
        println!(
            "{:>7} MHz {:>7} MHz {:>10.3} {:>10.1} {:>10.4} {:>11.1} {:>9.2}",
            p.pedido_mhz,
            p.efetivo_mhz.map_or("?".into(), |v| v.to_string()),
            p.ms_passo,
            p.media_w,
            p.j_passo,
            p.gflops,
            p.gflop_j
        );
    }

    let melhor = pontos
        .iter()
        .max_by(|a, b| a.gflop_j.total_cmp(&b.gflop_j))
        .unwrap();
    let rapido = pontos
        .iter()
        .max_by(|a, b| a.gflops.total_cmp(&b.gflops))
        .unwrap();

    println!("\nmais eficiente: {} MHz, {:.2} GFLOP/J", melhor.pedido_mhz, melhor.gflop_j);
    println!("mais rápido:    {} MHz, {:.1} GFLOP/s", rapido.pedido_mhz, rapido.gflops);

    if melhor.pedido_mhz < rapido.pedido_mhz {
        let perda_vel = 100.0 * (1.0 - melhor.gflops / rapido.gflops);
        let ganho_ef = 100.0 * (melhor.gflop_j / rapido.gflop_j - 1.0);
        let menos_potencia = 100.0 * (1.0 - melhor.media_w / rapido.media_w);
        println!(
            "\nDo ponto mais rápido para o mais eficiente: {perda_vel:.0}% menos\n\
             velocidade, {ganho_ef:.0}% mais trabalho por joule, {menos_potencia:.0}% menos\n\
             potência ({:.1} W contra {:.1} W).",
            melhor.media_w, rapido.media_w
        );
        let ganho_vel = 100.0 * (rapido.gflops / melhor.gflops - 1.0);
        let ganho_pot = 100.0 * (rapido.media_w / melhor.media_w - 1.0);
        println!(
            "Na direção oposta: subir de {} para {} MHz dá {ganho_vel:.0}% de\n\
             desempenho custando {ganho_pot:.0}% de potência.",
            melhor.pedido_mhz, rapido.pedido_mhz
        );
    } else {
        println!(
            "\nA eficiência tem pico no topo da faixa: nesta carga, subir a\n\
             frequência não custa eficiência."
        );
    }
}
