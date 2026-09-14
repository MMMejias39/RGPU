//! Confere o produto interno `⟨a|b⟩` contra a referência de CPU e mede o
//! desvio que a meia precisão introduz na norma — inclusive no teto de
//! 28 qubits, onde o estado só existe em `f16`.
//!
//! Sem GPU disponível, os testes passam com um aviso em vez de falhar.

use rqubit::{Complexo, Estado, Porta1, Precisao};
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

/// `⟨a|b⟩` por definição, sobre amplitudes já baixadas da GPU.
fn produto_cpu(a: &[Complexo], b: &[Complexo]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut re = 0.0;
    let mut im = 0.0;
    for (x, y) in a.iter().zip(b) {
        // conj(x)·y
        re += x.0 * y.0 + x.1 * y.1;
        im += x.0 * y.1 - x.1 * y.0;
    }
    (re, im)
}

#[test]
fn produto_interno_confere_com_a_cpu() {
    let Some(gpu) = abrir() else { return };
    let a = Estado::novo(&gpu, 6).expect("estado a");
    let b = Estado::novo(&gpu, 6).expect("estado b");

    // A: Hadamard no qubit 0 → (s, s, 0, …). B: X no qubit 0 → (0, 1, 0, …).
    let mut enc = gpu.encoder();
    a.aplicar(&gpu, &mut enc, &Porta1::hadamard(), 0);
    b.aplicar(&gpu, &mut enc, &Porta1::x(), 0);
    gpu.submit(enc);

    let esperado = produto_cpu(&a.baixar(&gpu), &b.baixar(&gpu));
    let (re, im) = a.produto_interno(&gpu, &b);
    assert!(
        (re - esperado.0).abs() < 1e-5 && (im - esperado.1).abs() < 1e-5,
        "GPU ({re}, {im}) contra CPU {esperado:?}"
    );
    // E o valor analítico: ⟨A|B⟩ = conj(s)·0 + conj(s)·1 = 1/√2.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    assert!((re - s).abs() < 1e-5, "esperado 1/√2, veio {re}");
}

#[test]
fn produto_interno_nao_modifica_os_estados() {
    let Some(gpu) = abrir() else { return };
    let a = Estado::novo(&gpu, 5).expect("estado a");
    let b = Estado::novo(&gpu, 5).expect("estado b");

    let antes_a = a.baixar(&gpu);
    let antes_b = b.baixar(&gpu);
    let _ = a.produto_interno(&gpu, &b);
    assert_eq!(a.baixar(&gpu), antes_a, "o produto alterou o estado a");
    assert_eq!(b.baixar(&gpu), antes_b, "o produto alterou o estado b");
}

#[test]
fn norma_em_meia_precisao() {
    let Some(gpu) = abrir() else { return };
    let psi = Estado::novo_com(&gpu, 6, Precisao::F16).expect("estado f16");

    let mut enc = gpu.encoder();
    for q in 0..6 {
        psi.aplicar(&gpu, &mut enc, &Porta1::hadamard(), q);
    }
    gpu.submit(enc);

    // A mesma superposição uniforme em f16. Amplitudes que são potências de
    // dois são exatas em f16, então o desvio medido de ⟨ψ|ψ⟩ é zero; a
    // tolerância cobre o caso geral, de amplitudes não representáveis.
    let (re, im) = psi.produto_interno(&gpu, &psi);
    eprintln!("⟨ψ|ψ⟩ em f16: {re} ({:+e} de 1), imaginário {im}", re - 1.0);
    assert!((re - 1.0).abs() < 0.01, "norma f16 desviou: {re}");
    assert!(im.abs() < 1e-3, "parte imaginária da norma: {im}");
}

#[test]
fn produto_interno_em_28_qubits_f16() {
    let Some(gpu) = abrir() else { return };

    // O teto: 2²⁸ amplitudes × 4 B = 1 GiB por estado, 2 GiB no par. Em f32
    // seriam 2 GiB por estado — acima do limite de binding do WebGPU, que dá
    // no máximo 27 qubits. 28 qubits em f16 também é o único formato em que
    // a grade do produto precisa de duas dimensões: são 65.536 blocos de
    // 4.096 amplitudes, um a mais que o teto de 65.535 workgroups.
    let a = Estado::novo_com(&gpu, 28, Precisao::F16).expect("estado a");
    let b = Estado::novo_com(&gpu, 28, Precisao::F16).expect("estado b");

    // ⟨0…0|0…0⟩: a amplitude 1,0 é exata em f16, o produto tem de sair 1.
    let (re, im) = a.produto_interno(&gpu, &a);
    assert!(
        (re - 1.0).abs() < 1e-6 && im.abs() < 1e-6,
        "⟨0…0|0…0⟩ = ({re}, {im})"
    );

    // Superposição uniforme em A: 28 Hadamards, amplitude 2⁻¹⁴ em todo índice.
    let mut enc = gpu.encoder();
    for q in 0..28 {
        a.aplicar(&gpu, &mut enc, &Porta1::hadamard(), q);
    }
    gpu.submit(enc);

    // ⟨ψ|ψ⟩ continua 1 — a cascata de Hadamards mantém as amplitudes em
    // potências exatas de dois, então o desvio medido é zero.
    let (re, im) = a.produto_interno(&gpu, &a);
    eprintln!("⟨ψ|ψ⟩ em 28 qubits f16: {re} ({:+e} de 1)", re - 1.0);
    assert!((re - 1.0).abs() < 0.01, "norma em 28 qubits: {re}");
    assert!(im.abs() < 1e-3, "parte imaginária: {im}");

    // ⟨0…0|ψ⟩ = a amplitude no índice 0 = 2⁻¹⁴ — exatamente a borda entre
    // normais e subnormais do f16. O Hadamard em cascata a arredonda ~28
    // vezes, então vale a tolerância relativa de ~1%.
    let (re, im) = b.produto_interno(&gpu, &a);
    let esperado = 2f32.powi(-14);
    eprintln!(
        "⟨0…0|ψ⟩ = {re} contra {esperado} ({:+e} relativo)",
        (re - esperado) / esperado
    );
    assert!(
        (re - esperado).abs() / esperado < 0.01 && im.abs() < esperado,
        "overlap com |0…0⟩ = {re}"
    );

    // O mesmo circuito em B: dois estados preparados igualmente têm de dar
    // sobreposição 1 — é o que uma matriz de kernel exige de estados reusados.
    let mut enc = gpu.encoder();
    for q in 0..28 {
        b.aplicar(&gpu, &mut enc, &Porta1::hadamard(), q);
    }
    gpu.submit(enc);
    let (re, _) = a.produto_interno(&gpu, &b);
    assert!((re - 1.0).abs() < 0.01, "fidelidade entre gêmeos: {re}");
}

#[test]
fn rotacoes_medem_o_desvio_da_norma_em_meia_precisao() {
    let Some(gpu) = abrir() else { return };
    use rqubit::estado_inicial_cpu;
    const N: usize = 12;
    let psi = Estado::novo_com(&gpu, N, Precisao::F16).expect("f16");
    let mut cpu = estado_inicial_cpu(N);

    // Ângulos não representáveis: os senos e cossenos não são potências de
    // dois, e o arredondamento para f16 tem de aparecer na norma.
    let mut enc = gpu.encoder();
    for q in 0..N {
        let porta = Porta1::ry(0.21 + q as f32 * 0.013);
        psi.aplicar(&gpu, &mut enc, &porta, q);
        porta.aplicar_cpu(&mut cpu, q);
    }
    gpu.submit(enc);

    // A física em f32: o que a norma deveria ser sem o arredondamento.
    let esperado: f32 = cpu.iter().map(|c| c.0 * c.0 + c.1 * c.1).sum();
    let (re, im) = psi.produto_interno(&gpu, &psi);
    eprintln!(
        "rotações em {N} qubits f16: ⟨ψ|ψ⟩ = {re} — desvio {:+e} de 1 (CPU {esperado})",
        re - 1.0
    );
    assert!((re - 1.0).abs() < 0.01, "norma com rotações: {re}");
    assert!(im.abs() < 1e-3, "parte imaginária: {im}");
}

#[test]
fn rotacoes_em_28_qubits_contra_o_valor_analitico() {
    let Some(gpu) = abrir() else { return };
    let psi = Estado::novo_com(&gpu, 28, Precisao::F16).expect("f16");

    // RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩, então a amplitude do |0…0⟩ depois
    // de uma rotação por qubit é o produto Π cos(θ_q/2) — um número analítico,
    // sem referência de CPU de 2 GiB.
    let mut enc = gpu.encoder();
    let mut analitico = 1.0f32;
    for q in 0..28 {
        let t = 0.21 + q as f32 * 0.013;
        psi.aplicar(&gpu, &mut enc, &Porta1::ry(t), q);
        analitico *= (t / 2.0).cos();
    }
    gpu.submit(enc);

    // A norma, agora com amplitudes não representáveis.
    let (re, im) = psi.produto_interno(&gpu, &psi);
    eprintln!(
        "28 qubits, 28 rotações: ⟨ψ|ψ⟩ = {re} ({:+e} de 1), imaginário {im}",
        re - 1.0
    );
    assert!((re - 1.0).abs() < 0.01, "norma em 28 qubits com rotações: {re}");
    assert!(im.abs() < 1e-3, "parte imaginária: {im}");

    // ⟨0…0|ψ⟩ contra o produto analítico: 28 arredondamentos em f16 no mesmo
    // caminho de amplitudes — o erro esperado é de ~10⁻³ relativo.
    let zero = Estado::novo_com(&gpu, 28, Precisao::F16).expect("estado zero");
    let (re, _) = zero.produto_interno(&gpu, &psi);
    eprintln!(
        "⟨0…0|ψ⟩ = {re} contra {analitico} ({:+e} relativo)",
        (re - analitico) / analitico
    );
    assert!(
        (re - analitico).abs() / analitico.abs() < 0.01,
        "overlap analítico: {re} contra {analitico}"
    );
}
