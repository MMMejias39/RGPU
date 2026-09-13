//! Quanto a fusão de portas rende, em tempo e energia.
//!
//! Uso: cargo run -p rqubit --release --example bench_fusao -- [camadas]
//!
//! Circuito em camadas, no formato típico de algoritmo variacional: rotações em
//! todos os qubits, depois emaranhamento em cadeia. É onde a fusão tem o que
//! fazer — cada camada de rotações desaparece dentro dos CNOTs seguintes.

use std::time::Instant;

use rgpu_power::Medidor;
use rqubit::{Circuito, Estado, Porta1, Porta2, Precisao};
use rtensor::gpu::Gpu;

fn circuito(qubits: usize, camadas: usize) -> Circuito {
    let mut c = Circuito::novo(qubits);
    for camada in 0..camadas {
        for q in 0..qubits {
            c.uma(Porta1::ry(0.19 + camada as f32 * 0.07 + q as f32 * 0.023), q);
            c.uma(Porta1::fase(0.11 + q as f32 * 0.017), q);
        }
        for q in 0..qubits - 1 {
            c.duas(Porta2::cnot(), q, q + 1);
        }
    }
    c
}

fn main() {
    let camadas: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let gpu = Gpu::com_adaptador("nvidia").expect("sem GPU NVIDIA");
    println!("{}", gpu.info());
    println!("circuito em {camadas} camadas: rotações em todos os qubits + CNOTs em cadeia\n");

    let mut medidor = Medidor::novo().com_intervalo(50);
    medidor.calibrar_ociosidade(4.0);

    println!(
        "{:>7} {:>8} {:>7} {:>9} {:>11} {:>9} {:>13}",
        "qubits", "portas", "máx.", "fundidas", "ms", "ganho", "energia"
    );

    for qubits in [22usize, 24, 26] {
        let c = circuito(qubits, camadas);
        let estado = Estado::novo_com(&gpu, qubits, Precisao::F32).expect("estado");

        // `max = 0` significa sem fusão; 2 a 4 são os limites de qubits por
        // unitária fundida.
        let medir = |max: usize| -> (f64, usize, Option<f64>) {
            let rodar = |repeticoes: usize| {
                let mut enc = gpu.encoder();
                let mut n = 0;
                for _ in 0..repeticoes {
                    n = if max == 0 {
                        estado.aplicar_circuito(&gpu, &mut enc, &c, false)
                    } else {
                        estado.aplicar_circuito_fundido(&gpu, &mut enc, &c, max)
                    };
                }
                gpu.submit(enc);
                gpu.sync();
                n
            };
            let n = rodar(1);
            let t0 = Instant::now();
            rodar(3);
            let por_circuito = t0.elapsed().as_secs_f64() / 3.0;

            let reps = ((4.0 / por_circuito) as usize).clamp(3, 2000);
            let (_, m) = medidor.medir(|| {
                rodar(reps);
            });
            let j = m.energia_gpu_total_j().map(|v| v / reps as f64);
            (por_circuito, n, j)
        };

        let (t_base, n_base, j_base) = medir(0);
        println!(
            "{:>7} {:>8} {:>7} {:>9} {:>11.3} {:>9} {:>13}",
            qubits, n_base, "—", n_base, t_base * 1e3, "(base)", "(base)"
        );
        for max in 2..=4usize {
            let (t, n, j) = medir(max);
            let energia = match (j_base, j) {
                (Some(a), Some(b)) if b > 1e-9 => format!("{:.2}× menos", a / b),
                _ => "—".into(),
            };
            println!(
                "{:>7} {:>8} {:>7} {:>9} {:>11.3} {:>8.2}× {:>13}",
                "", "", max, n, t * 1e3, t_base / t, energia
            );
        }
        println!();
    }

    println!(
        "\nCada porta é uma passada completa pelo vetor de estado, e a simulação\n\
         é limitada por banda: menos passadas é proporcionalmente menos tempo\n\
         e menos energia. A fusão não muda um byte do kernel."
    );
}
