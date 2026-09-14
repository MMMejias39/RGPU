//! A Transformada Quântica de Fourier no teto da máquina.
//!
//! Uso: cargo run -p rqubit --release --example bench_qft
//!      cargo run -p rqubit --release --example bench_qft -- tabela 27:f32 28:f16
//!
//! Algoritmo real, não um laço de portas soltas: H em cada qubit, portas
//! controladas-fase entre todos os pares — O(n²) portas, 420 em 28 qubits —
//! e a reversão final de ordem. É o circuito canônico de benchmark, o mesmo
//! formato que Qiskit e cuQuantum usam de demonstração.
//!
//! Sem argumentos, executa três verificações matemáticas antes da medição:
//! 1. em 8 qubits, as amplitudes saem conferidas contra a referência de CPU;
//! 2. o produto interno novo prova que ⟨ψ|ψ⟩ = 1 depois da QFT completa —
//!    o estado continua normalizado, o que num kernel novo não é garantido;
//! 3. a ida e volta QFT⁻¹∘QFT sobre |x₀⟩ devolve sobreposição 1 com |x₀⟩.
//! A fase `tabela` refaz só as medições, nos tamanhos pedidos.

use std::time::Instant;

use rgpu_power::Medidor;
use rqubit::{estado_inicial_cpu, Circuito, Complexo, Estado, Porta1, Porta2, Precisao};
use rtensor::gpu::Gpu;

/// CP(θ) = diag(1, 1, 1, e^{iθ}) — a fase só no |11⟩, como a CZ.
fn fase_controlada(theta: f32) -> Porta2 {
    let mut u = [[(0.0f32, 0.0f32); 4]; 4];
    for (i, linha) in u.iter_mut().enumerate() {
        linha[i] = (1.0, 0.0);
    }
    u[3][3] = (theta.cos(), theta.sin());
    Porta2 { u }
}

/// QFT padrão: qubit q pesa 2^q. H em j, depois CP(2π/2^(k−j)) entre todo
/// par (k, j) com k > j, e a reversão final da ordem dos qubits.
fn qft(n: usize) -> Circuito {
    let mut c = Circuito::novo(n);
    for j in 0..n {
        c.uma(Porta1::hadamard(), j);
        for k in (j + 1)..n {
            let ang = std::f32::consts::TAU / (1u64 << (k - j)) as f32;
            c.duas(fase_controlada(ang), k, j);
        }
    }
    for q in 0..n / 2 {
        c.duas(Porta2::swap(), q, n - 1 - q);
    }
    c
}

/// A inversa exata: ordem das portas invertida, ângulos trocados de sinal.
fn qft_inversa(n: usize) -> Circuito {
    let mut c = Circuito::novo(n);
    for q in 0..n / 2 {
        c.duas(Porta2::swap(), q, n - 1 - q);
    }
    for j in (0..n).rev() {
        for k in ((j + 1)..n).rev() {
            let ang = -std::f32::consts::TAU / (1u64 << (k - j)) as f32;
            c.duas(fase_controlada(ang), k, j);
        }
        c.uma(Porta1::hadamard(), j);
    }
    c
}

/// Prepara |x₀⟩ com X nos qubits de bits ligados, na GPU e, opcionalmente, na CPU.
fn preparar_x(gpu: &Gpu, estado: &Estado, x: u64, mut cpu: Option<&mut Vec<Complexo>>) {
    let mut enc = gpu.encoder();
    for q in 0..estado.qubits() {
        if (x >> q) & 1 == 1 {
            let porta = Porta1::x();
            estado.aplicar(gpu, &mut enc, &porta, q);
            if let Some(psi) = cpu.as_deref_mut() {
                porta.aplicar_cpu(psi, q);
            }
        }
    }
    gpu.submit(enc);
}

fn main() {
    // Fases: sem argumentos roda tudo; `tabela` pula as verificações, e os
    // pares `qubits:prec` restringem os tamanhos medidos — para retomar sem
    // refazer o que já mediu.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fase_tabela = args.iter().any(|a| a == "tabela");
    let mut tamanhos: Vec<(usize, Precisao)> = args
        .iter()
        .filter(|a| a.contains(':'))
        .filter_map(|a| {
            let (q, p) = a.split_once(':')?;
            Some((q.parse().ok()?, if p == "f16" { Precisao::F16 } else { Precisao::F32 }))
        })
        .collect();
    if tamanhos.is_empty() {
        tamanhos = vec![
            (22usize, Precisao::F32),
            (24, Precisao::F32),
            (26, Precisao::F32),
            (26, Precisao::F16),
            (27, Precisao::F32),
            (28, Precisao::F16),
        ];
    }

    let gpu = Gpu::com_adaptador("nvidia").expect("sem GPU NVIDIA");
    println!("{}", gpu.info());
    println!("QFT completa: H em cada qubit + CP(2π/2^(k−j)) entre todos os pares + reversão\n");

    let mut medidor = Medidor::novo().com_intervalo(50);
    medidor.calibrar_ociosidade(4.0);

    // ─── 1. Conferência contra a CPU em 8 qubits ────────────────────────────
    if !fase_tabela {
        const N_CONF: usize = 8;
        let circuito_conf = qft(N_CONF);
        let estado = Estado::novo(&gpu, N_CONF).expect("estado");
        let mut cpu = estado_inicial_cpu(N_CONF);
        preparar_x(&gpu, &estado, 0b1011_0001u64, Some(&mut cpu));

        let mut enc = gpu.encoder();
        estado.aplicar_circuito(&gpu, &mut enc, &circuito_conf, false);
        gpu.submit(enc);
        gpu.sync();

        let psi_gpu = estado.baixar(&gpu);
        circuito_conf.aplicar_cpu(&mut cpu, false);
        let erro = psi_gpu
            .iter()
            .zip(&cpu)
            .map(|(g, c)| (g.0 - c.0).abs().max((g.1 - c.1).abs()))
            .fold(0.0f32, f32::max);
        let magnitudes: Vec<f32> = psi_gpu
            .iter()
            .map(|c| (c.0 * c.0 + c.1 * c.1).sqrt())
            .collect();
        // QFT de um estado da base: toda amplitude tem magnitude 1/√N.
        let magnitude_ideal = (1.0f32 / (1usize << N_CONF) as f32).sqrt();
        println!(
            "conferência em {N_CONF} qubits contra a CPU: erro máximo {erro:.2e}; \
             |amplitude| ideal {magnitude_ideal:.5}, GPU {:#.5}–{:#.5}",
            magnitudes.iter().cloned().fold(f32::MAX, f32::min),
            magnitudes.iter().cloned().fold(0.0f32, f32::max),
        );
        assert!(erro < 1e-4, "QFT discorda da CPU: {erro:.2e}");

        // ─── 2. A física continua de pé no teto: norma e ida-e-volta ───────
        // Em f32 a fusão chega a 6 qubits (kernels especializados); em f16, a 2.
        for (n, precisao, com_fusao_6) in
            [(27usize, Precisao::F32, true), (28, Precisao::F16, false)]
        {
            let c_ida = qft(n);
            let c_volta = qft_inversa(n);
            let a = Estado::novo_com(&gpu, n, precisao).unwrap_or_else(|e| panic!("{e}"));
            let b = Estado::novo_com(&gpu, n, precisao).unwrap_or_else(|e| panic!("{e}"));
            let padrao = 0xAAAA_AAAAu64 & ((1u64 << n) - 1);
            preparar_x(&gpu, &a, padrao, None);
            preparar_x(&gpu, &b, padrao, None);

            let aplicar = |estado: &Estado, c: &Circuito| {
                let mut enc = gpu.encoder();
                let portas = if com_fusao_6 {
                    estado.aplicar_circuito_fundido(&gpu, &mut enc, c, 6)
                } else {
                    estado.aplicar_circuito(&gpu, &mut enc, c, true)
                };
                gpu.submit(enc);
                portas
            };
            let total = aplicar(&a, &c_ida) + aplicar(&a, &c_volta);
            gpu.sync();

            let (norma_re, _) = a.produto_interno(&gpu, &a);
            let (overlap_re, _) = a.produto_interno(&gpu, &b);
            println!(
                "ida e volta em {n} qubits, {total} portas fundidas ({}): \
                 ⟨ψ|ψ⟩ = {norma_re:.6}, ⟨x₀|QFT⁻¹QFT|x₀⟩ = {overlap_re:.6}",
                if com_fusao_6 { "f32, fusão até 6" } else { "f16, fusão até 2" },
            );
            let tolerancia = if precisao == Precisao::F16 { 0.01 } else { 1e-4 };
            assert!((norma_re - 1.0).abs() < tolerancia, "norma {norma_re}");
            assert!((overlap_re - 1.0).abs() < tolerancia, "ida e volta {overlap_re}");
        }
    }

    // ─── 3. Tempo e energia, do menor caso ao teto ──────────────────────────
    println!(
        "\n{:>6} {:>5} {:>8} {:>9} {:>12} {:>10} {:>12}",
        "qubits", "prec", "portas", "fundidas", "ms/circuito", "ms/porta", "J/circuito"
    );
    for (qubits, precisao) in tamanhos {
        let c = qft(qubits);
        let Ok(estado) = Estado::novo_com(&gpu, qubits, precisao) else { continue };

        let mut enc = gpu.encoder();
        let fundidas2 = estado.aplicar_circuito(&gpu, &mut enc, &c, true);
        gpu.submit(enc);
        gpu.sync();
        let fundidas6 = if precisao == Precisao::F32 {
            let mut enc = gpu.encoder();
            let n = estado.aplicar_circuito_fundido(&gpu, &mut enc, &c, 6);
            gpu.submit(enc);
            gpu.sync();
            n
        } else {
            fundidas2
        };

        let executar = |modo: usize, reps: usize| -> usize {
            let mut enc = gpu.encoder();
            let mut n = 0;
            for _ in 0..reps {
                n = match modo {
                    0 => estado.aplicar_circuito(&gpu, &mut enc, &c, false),
                    1 => estado.aplicar_circuito(&gpu, &mut enc, &c, true),
                    _ => estado.aplicar_circuito_fundido(&gpu, &mut enc, &c, 6),
                };
            }
            gpu.submit(enc);
            gpu.sync();
            n
        };

        let prec = if precisao == Precisao::F16 { "f16" } else { "f32" };
        for (modo, fundidas) in [(0usize, c.portas()), (1, fundidas2), (2, fundidas6)] {
            if modo == 2 && precisao != Precisao::F32 {
                continue;
            }
            if modo == 1 && fundidas2 == c.portas() {
                continue;
            }
            executar(modo, 1); // aquece
            let t0 = Instant::now();
            executar(modo, 3);
            let por_circuito = t0.elapsed().as_secs_f64() / 3.0;
            let reps = ((4.0 / por_circuito) as usize).clamp(3, 400);
            let (_, m) = medidor.medir(|| {
                executar(modo, reps);
            });
            let j = m.energia_gpu_total_j().map(|v| v / reps as f64);
            println!(
                "{qubits:>6} {prec:>5} {:>8} {fundidas:>9} {:>12.1} {:>10.3} {:>12}",
                c.portas(),
                por_circuito * 1e3,
                por_circuito * 1e3 / fundidas.max(1) as f64,
                j.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into()),
            );
        }
    }

    println!(
        "\nnotas:\n\
         - o teto de qubits é o binding do WebGPU: 27 em f32, 28 em f16;\n\
         - a fusão até 6 qubits usa os kernels especializados, só em f32;\n\
         - ângulos CP abaixo da resolução de f16 (~5·10⁻⁴) agem como identidade:\n\
           a QFT em f16 é aproximada por construção — a QFT aproximada da literatura."
    );
}
