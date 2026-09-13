//! Mede a energia de um comando externo, para comparar simuladores de outros
//! ecossistemas sob o mesmo instrumento.
//!
//! Uso: medir_externo -- <comando> [args...]
//!
//! Os simuladores existentes — Qiskit Aer, cuQuantum, qsim — reportam tempo,
//! nunca joules. Como a potência da GPU é medida por fora do processo, dá para
//! instrumentá-los sem tocar no código deles.
//!
//! A energia inclui o custo do processo inteiro, interpretador Python
//! incluído. É o que se paga de verdade para rodar aquela simulação.

use std::process::Command;

use rgpu_power::Medidor;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("uso: medir_externo <comando> [args...]");
        std::process::exit(1);
    }

    let mut medidor = Medidor::novo().com_intervalo(50);
    eprintln!("calibrando ociosidade...");
    medidor.calibrar_ociosidade(4.0);

    let (saida, m) = medidor.medir(|| {
        Command::new(&args[0])
            .args(&args[1..])
            .output()
            .expect("comando falhou")
    });

    print!("{}", String::from_utf8_lossy(&saida.stdout));
    let stderr = String::from_utf8_lossy(&saida.stderr);
    if !stderr.trim().is_empty() {
        eprintln!("{stderr}");
    }

    match m.energia_gpu_total_j() {
        Some(j) => println!(
            "energia|duracao_s={:.3}|total_J={:.2}|media_W={:.1}|acima_ocioso_J={:.2}",
            m.duracao_s,
            j,
            m.gpu.as_ref().map_or(0.0, |g| g.media_w),
            m.energia_gpu_j().unwrap_or(0.0)
        ),
        None => println!("energia|indisponível"),
    }
}
