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
use rqubit::{Estado, Porta1, Precisao};
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
        "{:>7} {:>7} {:>10} {:>11} {:>10} {:>11} {:>12}",
        "qubits", "prec.", "memória", "ms/porta", "GB/s", "µJ/porta", "portas/s"
    );

    for qubits in [20usize, 22, 24, 26, 27, 28, 29] {
      for precisao in [Precisao::F32, Precisao::F16] {
        // Acima do teto de cada precisão o construtor recusa, e o laço segue.
        let estado = match Estado::novo_com(&gpu, qubits, precisao) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Hadamard no qubit 0: o padrão de acesso mais próximo possível, com os
        // dois elementos do par adjacentes na memória.
        let porta = Porta1::hadamard();
        let aplicar = |n: usize| {
            let mut enc = gpu.encoder();
            for i in 0..n {
                estado.aplicar(&gpu, &mut enc, &porta, i % qubits);
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

        // Cada par lê 2 amplitudes e escreve 2, logo o tráfego é
        // 2ⁿ × 2 × bytes_por_amplitude.
        let bytes = (1u64 << qubits) * 2 * precisao.bytes_por_amplitude() as u64;
        let uj = m
            .energia_gpu_total_j()
            .map(|j| j / n_energia as f64 * 1e6);

        println!(
            "{:>7} {:>7} {:>10} {:>11.4} {:>10.1} {:>11} {:>12.0}",
            qubits,
            if precisao == Precisao::F32 { "f32" } else { "f16" },
            format!("{} MB", estado.bytes() / (1 << 20)),
            por_porta * 1e3,
            bytes as f64 / por_porta / 1e9,
            uj.map_or("—".into(), |v| format!("{v:.1}")),
            1.0 / por_porta,
        );
      }
    }

    println!(
        "\nA RTX 4070 Laptop tem ~256 GB/s — a simulação encosta em 83% disso,\n\
         o que confirma que é carga governada por banda. É o regime oposto ao\n\
         do GEMM, e onde a precisão mista deve render o que não rendeu lá.\n\
         \n\
         O teto de {} qubits não é de VRAM: o WebGPU limita um binding de\n\
         armazenamento a 2 GB, e o vetor de estado é um buffer só.",
        rqubit::Estado::MAX_QUBITS
    );
}
