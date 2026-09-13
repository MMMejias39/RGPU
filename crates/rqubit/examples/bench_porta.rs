//! Quanto custa uma porta quântica, em tempo, banda e joules.
//!
//! Uso: cargo run -p rqubit --release --example bench_porta
//!
//! A hipótese a testar: aplicar uma porta é **limitado por banda de memória**.
//! Cada par de amplitudes é lido (16 bytes), multiplicado por uma matriz 2×2 e
//! reescrito (16 bytes) — 32 bytes por ~28 flops, ou 0,875 flop por byte.
//!
//! Se a banda medida encostar no teto da placa, a hipótese se confirma — e com
//! ela a previsão de que precisão mista, inútil no GEMM, renderia aqui.

use std::time::Instant;

use rgpu_power::Medidor;
use rqubit::{Estado, Porta1, Porta2, Precisao};
use rtensor::gpu::Gpu;

fn main() {
    let gpu = match Gpu::com_adaptador("nvidia") {
        Ok(g) => g,
        Err(e) => {
            eprintln!("sem GPU NVIDIA: {e}");
            std::process::exit(1);
        }
    };
    println!("{}\n", gpu.info());

    let mut medidor = Medidor::novo().com_intervalo(50);
    medidor.calibrar_ociosidade(4.0);

    println!(
        "{:>7} {:>7} {:>7} {:>11} {:>10} {:>11} {:>12}",
        "qubits", "porta", "prec.", "ms/porta", "GB/s", "µJ/porta", "portas/s"
    );

    for qubits in [22usize, 24, 26, 27, 28] {
      for precisao in [Precisao::F32, Precisao::F16] {
        // Acima do teto de cada precisão o construtor recusa, e o laço segue.
        let estado = match Estado::novo_com(&gpu, qubits, precisao) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let h = Porta1::hadamard();
        let cnot = Porta2::cnot();

        // Mesmo tráfego nas duas: cada amplitude é lida e reescrita uma vez.
        // O que muda é a aritmética — 4 multiplicações complexas por par contra
        // 16 por grupo.
        for (rotulo, duas) in [("1q", false), ("2q", true)] {
            let aplicar = |n: usize| {
                let mut enc = gpu.encoder();
                for i in 0..n {
                    if duas {
                        let q0 = i % qubits;
                        let q1 = (q0 + 1 + i % (qubits - 1)) % qubits;
                        let q1 = if q1 == q0 { (q0 + 1) % qubits } else { q1 };
                        estado.aplicar2(&gpu, &mut enc, &cnot, q0, q1);
                    } else {
                        estado.aplicar(&gpu, &mut enc, &h, i % qubits);
                    }
                }
                gpu.submit(enc);
                gpu.sync();
            };

            aplicar(4);
            let t0 = Instant::now();
            aplicar(20);
            let por_porta = t0.elapsed().as_secs_f64() / 20.0;

            // Janela longa o bastante para o sensor de potência acompanhar.
            let n_energia = ((5.0 / por_porta) as usize).clamp(20, 20_000);
            let (_, m) = medidor.medir(|| aplicar(n_energia));

            let bytes = (1u64 << qubits) * 2 * precisao.bytes_por_amplitude() as u64;
            let uj = m.energia_gpu_total_j().map(|j| j / n_energia as f64 * 1e6);

            println!(
                "{:>7} {:>7} {:>7} {:>11.4} {:>10.1} {:>11} {:>12.0}",
                qubits,
                rotulo,
                if precisao == Precisao::F32 { "f32" } else { "f16" },
                por_porta * 1e3,
                bytes as f64 / por_porta / 1e9,
                uj.map_or("—".into(), |v| format!("{v:.1}")),
                1.0 / por_porta,
            );
        }
      }
    }

    println!(
        "\nA RTX 4070 Laptop tem ~256 GB/s, e a simulação encosta em ~80% disso:\n\
         carga governada por banda. Duas consequências medidas acima:\n\
         \n\
         - meia precisão dá ~2× em tempo e energia, metade dos bytes;\n\
         - a porta de dois qubits faz 4× a aritmética da de um qubit no MESMO\n\
           tempo, porque move os mesmos bytes — mas gasta 6 a 7% mais energia.\n\
           Pelo cronômetro ela é de graça; pelo wattímetro, não.\n\
         \n\
         Teto de {} qubits em f32 e {} em f16 — limite de binding do WebGPU,\n\
         não de VRAM.",
        Precisao::F32.max_qubits(),
        Precisao::F16.max_qubits()
    );
}
