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

/// Meia precisão tem de acertar a física, e o quanto ela perde é medido —
/// `f16` guarda ~3 dígitos decimais, e o erro se acumula a cada porta.
#[test]
fn meia_precisao_confere_e_o_erro_acumulado_e_medido() {
    use rqubit::Precisao;
    let Some(gpu) = abrir() else { return };
    const N: usize = 12;

    for portas_por_qubit in [1usize, 4, 16] {
        let e32 = Estado::novo_com(&gpu, N, Precisao::F32).expect("f32");
        let e16 = Estado::novo_com(&gpu, N, Precisao::F16).expect("f16");
        let mut cpu = estado_inicial_cpu(N);

        let mut enc = gpu.encoder();
        for rodada in 0..portas_por_qubit {
            for q in 0..N {
                let porta = Porta1::ry(0.21 + rodada as f32 * 0.09 + q as f32 * 0.013);
                e32.aplicar(&gpu, &mut enc, &porta, q);
                e16.aplicar(&gpu, &mut enc, &porta, q);
                porta.aplicar_cpu(&mut cpu, q);
            }
        }
        gpu.submit(enc);

        let p32 = e32.baixar(&gpu);
        let p16 = e16.baixar(&gpu);
        let (err32, err16) = (erro_max(&cpu, &p32), erro_max(&cpu, &p16));
        let total = portas_por_qubit * N;

        eprintln!(
            "{total:>3} portas em {N} qubits: f32 {err32:.2e}, f16 {err16:.2e} \
             ({:.0}× o erro), norma f16 {:.5}",
            err16 / err32.max(1e-12),
            norma(&p16)
        );

        // `f16` tem 11 bits de mantissa: erro relativo ~2⁻¹¹ por operação, que
        // cresce no máximo com √(portas) numa caminhada de erros independentes.
        let limite = 10.0 * (total as f32).sqrt() * 2.0f32.powi(-11);
        assert!(err16 < limite, "{total} portas: erro {err16:.2e} acima de {limite:.2e}");
        // A física precisa continuar de pé: estado normalizado.
        assert!((norma(&p16) - 1.0).abs() < 0.02, "norma f16 {}", norma(&p16));
    }
}

#[test]
fn meia_precisao_alcanca_mais_qubits() {
    use rqubit::Precisao;
    // O limite de binding dá 27 qubits em f32 e 28 em f16: meia precisão compra
    // um qubit, não dois, porque o teto também é potência de dois.
    assert_eq!(Precisao::F32.max_qubits(), 27);
    assert_eq!(Precisao::F16.max_qubits(), 28);
    assert_eq!(Precisao::F16.bytes_por_amplitude(), 4);
}

/// CNOT sobre |10⟩ tem de dar |11⟩, e sobre |00⟩ não fazer nada.
#[test]
fn cnot_nega_o_alvo_quando_o_controle_e_um() {
    use rqubit::Porta2;
    let Some(gpu) = abrir() else { return };
    let estado = Estado::novo(&gpu, 3).expect("estado");

    // X no qubit 1 leva |000⟩ a |010⟩; CNOT com controle 1 e alvo 0 dá |011⟩.
    let mut enc = gpu.encoder();
    estado.aplicar(&gpu, &mut enc, &Porta1::x(), 1);
    estado.aplicar2(&gpu, &mut enc, &Porta2::cnot(), 0, 1);
    gpu.submit(enc);

    let psi = estado.baixar(&gpu);
    assert!((psi[3].0 - 1.0).abs() < 1e-6, "esperado |011⟩, veio índice 3 = {:?}", psi[3]);
    for (i, a) in psi.iter().enumerate() {
        if i != 3 {
            assert!(a.0.abs() < 1e-6 && a.1.abs() < 1e-6, "índice {i} não nulo: {a:?}");
        }
    }
}

/// Hadamard seguido de CNOT produz o estado de Bell: |00⟩ e |11⟩ com amplitude
/// 1/√2, e nada nos estados intermediários. É o teste que só passa se o
/// emaranhamento estiver certo.
#[test]
fn produz_estado_de_bell() {
    use rqubit::Porta2;
    let Some(gpu) = abrir() else { return };
    let estado = Estado::novo(&gpu, 2).expect("estado");

    let mut enc = gpu.encoder();
    estado.aplicar(&gpu, &mut enc, &Porta1::hadamard(), 1);
    estado.aplicar2(&gpu, &mut enc, &Porta2::cnot(), 0, 1);
    gpu.submit(enc);

    let psi = estado.baixar(&gpu);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    assert!((psi[0].0 - s).abs() < 1e-6, "|00⟩ = {:?}", psi[0]);
    assert!(psi[1].0.abs() < 1e-6, "|01⟩ deveria ser zero: {:?}", psi[1]);
    assert!(psi[2].0.abs() < 1e-6, "|10⟩ deveria ser zero: {:?}", psi[2]);
    assert!((psi[3].0 - s).abs() < 1e-6, "|11⟩ = {:?}", psi[3]);
}

/// Circuito com portas de um e dois qubits, em todas as combinações de posição,
/// conferido contra a CPU.
#[test]
fn circuito_de_duas_portas_bate_com_a_cpu() {
    use rqubit::Porta2;
    let Some(gpu) = abrir() else { return };
    const N: usize = 10;

    let estado = Estado::novo(&gpu, N).expect("estado");
    let mut cpu = estado_inicial_cpu(N);
    let portas2 = [Porta2::cnot(), Porta2::cz(), Porta2::swap()];

    let mut enc = gpu.encoder();
    for q in 0..N {
        let p = Porta1::ry(0.3 + q as f32 * 0.11);
        estado.aplicar(&gpu, &mut enc, &p, q);
        p.aplicar_cpu(&mut cpu, q);
    }
    // Pares vizinhos, distantes e invertidos — todas as relações de posição.
    for (i, (q0, q1)) in [(0, 1), (2, 5), (9, 3), (4, 8), (7, 6)].iter().enumerate() {
        let p = portas2[i % 3];
        estado.aplicar2(&gpu, &mut enc, &p, *q0, *q1);
        p.aplicar_cpu(&mut cpu, *q0, *q1);
    }
    gpu.submit(enc);

    let psi = estado.baixar(&gpu);
    let e = erro_max(&cpu, &psi);
    eprintln!("{N} qubits, 10 portas de 1 + 5 de 2: erro {e:.2e}, norma {:.6}", norma(&psi));
    assert!(e < 1e-5, "erro máximo {e:.2e}");
    assert!((norma(&psi) - 1.0).abs() < 1e-4);
}

#[test]
fn swap_e_a_propria_inversa() {
    use rqubit::Porta2;
    let Some(gpu) = abrir() else { return };
    let estado = Estado::novo(&gpu, 6).expect("estado");
    let mut cpu = estado_inicial_cpu(6);

    let mut enc = gpu.encoder();
    for q in 0..6 {
        let p = Porta1::ry(0.5 + q as f32 * 0.2);
        estado.aplicar(&gpu, &mut enc, &p, q);
        p.aplicar_cpu(&mut cpu, q);
    }
    // Dois SWAPs iguais devolvem o estado ao que era.
    estado.aplicar2(&gpu, &mut enc, &Porta2::swap(), 1, 4);
    estado.aplicar2(&gpu, &mut enc, &Porta2::swap(), 1, 4);
    gpu.submit(enc);

    assert!(erro_max(&cpu, &estado.baixar(&gpu)) < 1e-6);
}

/// Constrói um circuito em camadas, no formato típico de algoritmo variacional:
/// rotações em todos os qubits, depois emaranhamento em cadeia.
fn circuito_em_camadas(qubits: usize, camadas: usize) -> rqubit::Circuito {
    use rqubit::{Circuito, Porta2};
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

/// A fusão precisa preservar o estado exatamente — é reescrita algébrica, não
/// aproximação.
#[test]
fn fusao_preserva_o_estado() {
    let Some(gpu) = abrir() else { return };
    const N: usize = 12;

    let circuito = circuito_em_camadas(N, 3);
    let direto = Estado::novo(&gpu, N).expect("estado");
    let fundido = Estado::novo(&gpu, N).expect("estado");

    let mut enc = gpu.encoder();
    let n_direto = direto.aplicar_circuito(&gpu, &mut enc, &circuito, false);
    let n_fundido = fundido.aplicar_circuito(&gpu, &mut enc, &circuito, true);
    gpu.submit(enc);

    let (a, b) = (direto.baixar(&gpu), fundido.baixar(&gpu));
    let e = erro_max(&a, &b);
    eprintln!(
        "{N} qubits, 3 camadas: {n_direto} portas → {n_fundido} fundidas \
         ({:.1}× menos), erro {e:.2e}",
        n_direto as f32 / n_fundido as f32
    );

    assert!(n_fundido < n_direto, "a fusão não reduziu nada");
    assert!(e < 1e-5, "fusão mudou o estado: erro {e:.2e}");
    assert!((norma(&b) - 1.0).abs() < 1e-4);
}

/// A fusão também tem de bater com a referência de CPU, não só consigo mesma.
#[test]
fn fusao_bate_com_a_cpu() {
    let Some(gpu) = abrir() else { return };
    const N: usize = 10;

    let circuito = circuito_em_camadas(N, 2);
    let estado = Estado::novo(&gpu, N).expect("estado");
    let mut cpu = estado_inicial_cpu(N);

    let mut enc = gpu.encoder();
    estado.aplicar_circuito(&gpu, &mut enc, &circuito, true);
    gpu.submit(enc);
    circuito.aplicar_cpu(&mut cpu, false); // referência sem fusão

    assert!(erro_max(&cpu, &estado.baixar(&gpu)) < 1e-5);
}

/// Portas de um qubit em sequência no mesmo qubit viram uma só.
#[test]
fn fusao_colapsa_cadeias_de_um_qubit() {
    use rqubit::{Circuito, Op};
    let mut c = Circuito::novo(3);
    for _ in 0..5 {
        c.uma(Porta1::hadamard(), 1);
    }
    let fundido = c.fundir();
    assert_eq!(c.portas(), 5);
    assert_eq!(fundido.len(), 1, "cinco portas no mesmo qubit deveriam virar uma");
    // Hadamard cinco vezes é Hadamard: ímpar, então H⁵ = H.
    match fundido[0] {
        Op::Uma(p, q) => {
            assert_eq!(q, 1);
            let s = std::f32::consts::FRAC_1_SQRT_2;
            assert!((p.u00.0 - s).abs() < 1e-5, "H⁵ deveria ser H, veio {:?}", p.u00);
        }
        _ => panic!("esperada porta de um qubit"),
    }
}

/// A fusão em unitárias maiores tem de preservar o estado exatamente, em todos
/// os limites de qubits.
#[test]
fn fusao_maior_preserva_o_estado() {
    let Some(gpu) = abrir() else { return };
    const N: usize = 10;

    let circuito = circuito_em_camadas(N, 3);
    let referencia = Estado::novo(&gpu, N).expect("estado");
    let mut enc = gpu.encoder();
    let n_direto = referencia.aplicar_circuito(&gpu, &mut enc, &circuito, false);
    gpu.submit(enc);
    let esperado = referencia.baixar(&gpu);

    for max in 2..=6usize {
        let estado = Estado::novo(&gpu, N).expect("estado");
        let mut enc = gpu.encoder();
        let n = estado.aplicar_circuito_fundido(&gpu, &mut enc, &circuito, max);
        gpu.submit(enc);

        let e = erro_max(&esperado, &estado.baixar(&gpu));
        eprintln!(
            "fusão até {max} qubits: {n_direto} portas → {n} ({:.2}× menos), erro {e:.2e}",
            n_direto as f32 / n as f32
        );
        assert!(e < 1e-5, "fusão até {max}: erro {e:.2e}");
    }
}

/// O kernel de N qubits tem de bater com a aplicação na CPU da mesma porta.
#[test]
fn kernel_de_n_qubits_bate_com_a_cpu() {
    use rqubit::{Porta2, PortaN};
    let Some(gpu) = abrir() else { return };
    const N: usize = 9;

    // Uma porta de 3 qubits construída compondo CNOTs que compartilham qubit.
    let a = PortaN::de_duas(&Porta2::cnot(), 2, 5);
    let b = PortaN::de_duas(&Porta2::cz(), 5, 7);
    let fundida = b.compor(&a);
    assert_eq!(fundida.alvos.len(), 3, "alvos: {:?}", fundida.alvos);

    let estado = Estado::novo(&gpu, N).expect("estado");
    let mut cpu = estado_inicial_cpu(N);

    let mut enc = gpu.encoder();
    for q in 0..N {
        let p = Porta1::ry(0.4 + q as f32 * 0.13);
        estado.aplicar(&gpu, &mut enc, &p, q);
        p.aplicar_cpu(&mut cpu, q);
    }
    estado.aplicar_n(&gpu, &mut enc, &fundida);
    gpu.submit(enc);
    fundida.aplicar_cpu(&mut cpu);

    let e = erro_max(&cpu, &estado.baixar(&gpu));
    assert!(e < 1e-5, "kernel de 3 qubits diverge: {e:.2e}");
}

/// Compor portas disjuntas dá a unitária sobre a união, e aplicá-la equivale a
/// aplicar as duas em sequência.
#[test]
fn composicao_de_portas_disjuntas() {
    use rqubit::PortaN;
    let Some(gpu) = abrir() else { return };
    const N: usize = 8;

    let a = PortaN::de_uma(&Porta1::hadamard(), 1);
    let b = PortaN::de_uma(&Porta1::ry(0.7), 5);
    let juntas = b.compor(&a);
    assert_eq!(juntas.alvos.len(), 2);

    let estado = Estado::novo(&gpu, N).expect("estado");
    let mut cpu = estado_inicial_cpu(N);

    let mut enc = gpu.encoder();
    estado.aplicar_n(&gpu, &mut enc, &juntas);
    gpu.submit(enc);
    Porta1::hadamard().aplicar_cpu(&mut cpu, 1);
    Porta1::ry(0.7).aplicar_cpu(&mut cpu, 5);

    assert!(erro_max(&cpu, &estado.baixar(&gpu)) < 1e-6);
}
