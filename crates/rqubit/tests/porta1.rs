//! Confere o kernel de porta de um qubit contra uma referência de CPU.
//!
//! Sem GPU disponível, os testes passam com um aviso em vez de falhar.

use rqubit::{estado_inicial_cpu, Complexo, Estado, Porta1};
use rtensor::gpu::Gpu;

fn abrir() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => {
            eprintln!("GPU: {}", g.info());
            Some(g)
        }
        Err(e) => {
            eprintln!("sem GPU ({e}) — teste ignorado");
            None
        }
    }
}

fn erro_max(a: &[Complexo], b: &[Complexo]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.0 - y.0).abs().max((x.1 - y.1).abs()))
        .fold(0.0, f32::max)
}

fn norma(v: &[Complexo]) -> f32 {
    v.iter().map(|c| c.0 * c.0 + c.1 * c.1).sum()
}

#[test]
fn hadamard_no_primeiro_qubit_cria_superposicao() {
    let Some(gpu) = abrir() else { return };
    let estado = Estado::novo(&gpu, 3).expect("estado");

    let mut enc = gpu.encoder();
    estado.aplicar(&gpu, &mut enc, &Porta1::hadamard(), 0);
    gpu.submit(enc);

    let psi = estado.baixar(&gpu);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    assert!((psi[0].0 - s).abs() < 1e-6, "amplitude |000⟩ = {:?}", psi[0]);
    assert!((psi[1].0 - s).abs() < 1e-6, "amplitude |001⟩ = {:?}", psi[1]);
    for a in &psi[2..] {
        assert!(a.0.abs() < 1e-6 && a.1.abs() < 1e-6, "amplitude não nula: {a:?}");
    }
}

#[test]
fn pauli_x_troca_as_amplitudes() {
    let Some(gpu) = abrir() else { return };
    let estado = Estado::novo(&gpu, 4).expect("estado");

    // X no qubit 2 leva |0000⟩ a |0100⟩, que é o índice 4.
    let mut enc = gpu.encoder();
    estado.aplicar(&gpu, &mut enc, &Porta1::x(), 2);
    gpu.submit(enc);

    let psi = estado.baixar(&gpu);
    assert!((psi[4].0 - 1.0).abs() < 1e-6, "esperado |0100⟩, veio {:?}", psi[4]);
    assert!(psi[0].0.abs() < 1e-6);
}

#[test]
fn circuito_bate_com_a_cpu_em_todos_os_qubits() {
    let Some(gpu) = abrir() else { return };
    const N: usize = 12;

    let estado = Estado::novo(&gpu, N).expect("estado");
    let mut cpu = estado_inicial_cpu(N);

    // Um circuito que toca todos os qubits com portas de naturezas diferentes.
    let circuito: Vec<(Porta1, usize)> = (0..N)
        .map(|q| (Porta1::hadamard(), q))
        .chain((0..N).map(|q| (Porta1::ry(0.3 + q as f32 * 0.17), q)))
        .chain((0..N).map(|q| (Porta1::fase(0.7 - q as f32 * 0.05), q)))
        .chain((0..N).rev().map(|q| (Porta1::z(), q)))
        .collect();

    let mut enc = gpu.encoder();
    for (porta, q) in &circuito {
        estado.aplicar(&gpu, &mut enc, porta, *q);
        porta.aplicar_cpu(&mut cpu, *q);
    }
    gpu.submit(enc);

    let psi = estado.baixar(&gpu);
    let e = erro_max(&cpu, &psi);
    eprintln!(
        "{} portas em {N} qubits: erro máximo {e:.2e}, norma {:.6}",
        circuito.len(),
        norma(&psi)
    );
    assert!(e < 1e-5, "erro máximo {e:.2e}");
}

#[test]
fn portas_preservam_a_norma() {
    let Some(gpu) = abrir() else { return };
    let estado = Estado::novo(&gpu, 10).expect("estado");

    let mut enc = gpu.encoder();
    for q in 0..10 {
        estado.aplicar(&gpu, &mut enc, &Porta1::hadamard(), q);
        estado.aplicar(&gpu, &mut enc, &Porta1::ry(0.4), q);
    }
    gpu.submit(enc);

    // Portas unitárias preservam ⟨ψ|ψ⟩ = 1.
    let n = norma(&estado.baixar(&gpu));
    assert!((n - 1.0).abs() < 1e-4, "norma {n} após 20 portas");
}

#[test]
fn recusa_qubit_fora_da_faixa() {
    assert!(Estado::novo(&Gpu::new().unwrap_or_else(|_| std::process::exit(0)), 0).is_err());
}
