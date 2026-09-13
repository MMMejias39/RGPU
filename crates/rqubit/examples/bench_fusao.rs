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
        "{:>7} {:>8} {:>9} {:>11} {:>11} {:>9} {:>11}",
        "qubits", "portas", "fundidas", "direto ms", "fundido ms", "ganho", "energia"
    );

    for qubits in [20usize, 22, 24, 26] {
        let c = circuito(qubits, camadas);
        let estado = Estado::novo_com(&gpu, qubits, Precisao::F32).expect("estado");

        let medir = |fundir: bool| -> (f64, usize, Option<f64>) {
            let rodar = |repeticoes: usize| {
                let mut enc = gpu.encoder();
                let mut n = 0;
                for _ in 0..repeticoes {
                    n = estado.aplicar_circuito(&gpu, &mut enc, &c, fundir);
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

        let (t_direto, n_direto, j_direto) = medir(false);
        let (t_fundido, n_fundido, j_fundido) = medir(true);

        let energia = match (j_direto, j_fundido) {
            (Some(a), Some(b)) if b > 1e-9 => format!("{:.2}× menos", a / b),
            _ => "—".into(),
        };

        println!(
            "{:>7} {:>8} {:>9} {:>11.3} {:>11.3} {:>8.2}× {:>11}",
            qubits,
            n_direto,
            n_fundido,
            t_direto * 1e3,
            t_fundido * 1e3,
            t_direto / t_fundido,
            energia
        );
    }

    println!(
        "\nCada porta é uma passada completa pelo vetor de estado, e a simulação\n\
         é limitada por banda: menos passadas é proporcionalmente menos tempo\n\
         e menos energia. A fusão não muda um byte do kernel."
    );
}
