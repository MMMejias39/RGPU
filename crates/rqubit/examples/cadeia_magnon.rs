//! Transporte de um magnon numa cadeia quântica de spins — física real.
//!
//! Uso: cargo run -p rqubit --release --example cadeia_magnon
//!
//! O modelo é a cadeia XX de 27 sítios, o sistema-padrão de transporte
//! quântico em íons presos e qubits supercondutores:
//!
//!     H = J Σ_ligacoes (X_l X_l+1 + Y_l Y_l+1)/2
//!
//! Prepara-se um único spin virado (um magnon) no centro e evolui-se com
//! Trotter de primeira ordem: 26 portas de 2 qubits por passo — exp(-iJδ(XX+
//! YY)/2) é uma rotação de Givens no par {|01⟩,|10⟩}, que preserva o número
//! de magnons. No setor de 1 magnon, H é a cadeia de hopping J, e a evolução
//! EXATA tem soma fechada sobre os 27 modos normais de cadeia aberta:
//!
//!     U(m,n,t) = (2/(N+1)) Σ_k sin(k(m+1)π/(N+1)) sin(k(n+1)π/(N+1))
//!                · e^{-i·2Jt·cos(kπ/(N+1))}
//!
//! É a resposta quântica exata para o estado completo de 2²⁷ amplitudes,
//! calculada em O(N²) — o árbitro físico da simulação.
//!
//! A metodologia separa dois erros que costumam ser confundidos:
//! 1. erro de Trotter — do circuito, quantificado em 12 qubits contra a
//!    evolução exata, independente de hardware;
//! 2. erro de precisão — do hardware, medido contra o mesmo circuito em f32
//!    e em f16.

use std::time::Instant;

use rqubit::{estado_inicial_cpu, Circuito, Complexo, Estado, Porta1, Porta2, Precisao};
use rtensor::gpu::Gpu;

const J: f64 = 1.0;

/// exp(-i·J·δ·(XX+YY)/2) — a rotação de Givens da ligação: identidade em
/// |00⟩ e |11⟩, rotação real com fase −i no bloco {|01⟩,|10⟩}.
fn porta_ligacao(delta: f32) -> Porta2 {
    let c = ((J * delta as f64) as f32).cos();
    let s = ((J * delta as f64) as f32).sin();
    let mut u = [[(0.0f32, 0.0f32); 4]; 4];
    u[0][0] = (1.0, 0.0);
    u[1][1] = (c, 0.0);
    u[2][2] = (c, 0.0);
    u[3][3] = (1.0, 0.0);
    u[1][2] = (0.0, -s);
    u[2][1] = (0.0, -s);
    Porta2 { u }
}

/// Um passo de Trotter: ligações pares e depois ímpares. 26 portas em 27 sítios.
fn circuito_trotter(n: usize, passos: usize, delta: f32) -> Circuito {
    let mut c = Circuito::novo(n);
    for _ in 0..passos {
        for m in (0..n - 1).step_by(2) {
            c.duas(porta_ligacao(delta), m, m + 1);
        }
        for m in (1..n - 1).step_by(2) {
            c.duas(porta_ligacao(delta), m, m + 1);
        }
    }
    c
}

/// Amplitude exata ⟨m|e^{-iHt}|s0⟩ no setor de 1 magnon, soma modal fechada.
fn amplitude_exata(m: usize, s0: usize, n: usize, t: f64) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for k in 1..=n {
        let th = std::f64::consts::PI * (k as f64) / (n as f64 + 1.0);
        let peso = (2.0 / (n as f64 + 1.0))
            * (th * (m as f64 + 1.0)).sin()
            * (th * (s0 as f64 + 1.0)).sin();
        let fase = -2.0 * J * t * th.cos();
        re += peso * fase.cos();
        im += peso * fase.sin();
    }
    (re, im)
}

/// Observáveis físicos extraídos das amplitudes baixadas: o estado inteiro
/// mora nos 2ⁿ sítios com um só bit ligado — o setor de 1 magnon.
fn observaveis(psi: &[Complexo], n: usize, s0: usize, t: f64) -> (Vec<f64>, f64, f64, f64, f64) {
    let p: Vec<f64> = (0..n)
        .map(|m| {
            let a = psi[1usize << m];
            (a.0 * a.0 + a.1 * a.1) as f64
        })
        .collect();
    let soma_p: f64 = p.iter().sum();
    // E = 2J Σ Re(ψ*_m ψ_{m+1}): o hamiltoniano é a cadeia de hopping J.
    let mut energia = 0.0;
    for m in 0..n - 1 {
        let a = psi[1usize << m];
        let b = psi[1usize << (m + 1)];
        energia += 2.0 * J * ((a.0 * b.0 + a.1 * b.1) as f64);
    }
    let centro: f64 = p.iter().enumerate().map(|(m, &v)| m as f64 * v).sum();
    let dispersao: f64 = (p
        .iter()
        .enumerate()
        .map(|(m, &v)| (m as f64 - centro).powi(2) * v)
        .sum::<f64>())
    .sqrt();
    let _ = (s0, t);
    (p, soma_p, energia, centro, dispersao)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let so_tempo = args.iter().any(|a| a == "tempo");

    let gpu = Gpu::com_adaptador("nvidia").expect("sem GPU NVIDIA");
    println!("{}", gpu.info());

    const N: usize = 27; // sítios
    const S0: usize = 13; // magnon no centro
    const DT: f64 = 0.1;
    const PASSOS: usize = 100; // t final = 10
    const T_FOTO: usize = 10; // passos entre fotos (t = 1, 2, …, 10)

    println!(
        "cadeia XX de {N} sítios, J = {J}, magnon inicial no sítio {S0} (centro);\n\
         Trotter de 1ª ordem, Δt = {DT}, {PASSOS} passos = {} portas de 2 qubits;\n\
         frente balística a v = 2J alcança as bordas em t ≈ {:.1}\n",
        26 * PASSOS,
        (S0.min(N - 1 - S0) as f64) / (2.0 * J)
    );

    // ─── 1. Os dois erros separados, em 12 qubits ───────────────────────────
    if !so_tempo {
        const NC: usize = 12;
        const S0C: usize = NC / 2; // o centro desta cadeia menor
        let circ = circuito_trotter(NC, PASSOS, DT as f32);

        // (a) o MESMO circuito na CPU, em precisão simples — a referência
        // do circuito, sem hardware envolvido.
        let mut cpu = estado_inicial_cpu(NC);
        Porta1::x().aplicar_cpu(&mut cpu, S0C);
        let t0 = Instant::now();
        circ.aplicar_cpu(&mut cpu, false);
        let _ = t0.elapsed();
        let (p_cpu, _, _, _, _) = observaveis(&cpu, NC, S0C, PASSOS as f64 * DT);
        let mut erro_trotter = 0.0f64;
        for m in 0..NC {
            let ex = amplitude_exata(m, S0C, NC, PASSOS as f64 * DT);
            erro_trotter = erro_trotter.max(((p_cpu[m] as f64) - ex.0 * ex.0 - ex.1 * ex.1).abs());
        }

        // (b) GPU f32 contra o mesmo circuito na CPU — erro de implementação.
        let e32 = Estado::novo(&gpu, NC).expect("estado");
        let mut psi_cpu = estado_inicial_cpu(NC);
        Porta1::x().aplicar_cpu(&mut psi_cpu, S0C);
        {
            let mut enc = gpu.encoder();
            e32.aplicar(&gpu, &mut enc, &Porta1::x(), S0C);
            gpu.submit(enc);
        }
        let mut enc = gpu.encoder();
        e32.aplicar_circuito(&gpu, &mut enc, &circ, false);
        gpu.submit(enc);
        gpu.sync();
        let gpu32 = e32.baixar(&gpu);
        let erro_impl = cpu
            .iter()
            .zip(&gpu32)
            .map(|(c, g)| (c.0 - g.0).abs().max((c.1 - g.1).abs()))
            .fold(0.0f32, f32::max);

        // (c) GPU f16 contra a CPU — erro de meia precisão na física.
        let e16 = Estado::novo_com(&gpu, NC, Precisao::F16).expect("f16");
        {
            let mut enc = gpu.encoder();
            e16.aplicar(&gpu, &mut enc, &Porta1::x(), S0C);
            gpu.submit(enc);
        }
        let mut enc = gpu.encoder();
        e16.aplicar_circuito(&gpu, &mut enc, &circ, true);
        gpu.submit(enc);
        gpu.sync();
        let gpu16 = e16.baixar(&gpu);
        let erro_f16 = cpu
            .iter()
            .zip(&gpu16)
            .map(|(c, g)| (c.0 - g.0).abs().max((c.1 - g.1).abs()))
            .fold(0.0f32, f32::max);

        println!(
            "verificação em {NC} qubits, t = {:.0}:\n  \
             erro de Trotter (CPU contra exato):      {erro_trotter:.2e}\n  \
             erro de implementação (GPU f32 vs CPU): {erro_impl:.2e}\n  \
             erro de precisão (GPU f16 vs CPU):      {erro_f16:.2e}\n",
            PASSOS as f64 * DT
        );
    }

    // ─── 2. Trajetórias no teto: 27 qubits, f32 e f16 ───────────────────────
    let mut tempos = [0.0f64; 2];
    let mut fisica = [Vec::new(), Vec::new()];
    let mut finais: [Option<Vec<Complexo>>; 2] = [None, None];

    for (prec_i, precisao) in [Precisao::F32, Precisao::F16].into_iter().enumerate() {
        let nome = if precisao == Precisao::F16 { "f16" } else { "f32" };
        let Ok(estado) = Estado::novo_com(&gpu, N, precisao) else { continue };
        let Ok(zero) = Estado::novo_com(&gpu, N, precisao) else { continue };
        {
            let mut enc = gpu.encoder();
            estado.aplicar(&gpu, &mut enc, &Porta1::x(), S0);
            zero.aplicar(&gpu, &mut enc, &Porta1::x(), S0);
            gpu.submit(enc);
        }
        gpu.sync();

        if !so_tempo {
            println!("\ntrajetória em {N} qubits ({nome}): t, eco |⟨ψ₀|ψ(t)⟩|², norma, \
                      energia, erro máx. de P contra o exato, ⟨m⟩, σ(m)");
        }
        let t0 = Instant::now();
        
        for foto in 1..=PASSOS / T_FOTO {
            let mut enc = gpu.encoder();
            let c = circuito_trotter(N, T_FOTO, DT as f32);
            estado.aplicar_circuito(&gpu, &mut enc, &c, false);
            gpu.submit(enc);
            gpu.sync();

            let (eco_re, eco_im) = estado.produto_interno(&gpu, &zero);
            let eco = (eco_re as f64) * (eco_re as f64) + (eco_im as f64) * (eco_im as f64);
            if so_tempo {
                continue;
            }
            let psi = estado.baixar(&gpu);
            let t = (foto * T_FOTO) as f64 * DT;
            let (p, soma_p, energia, centro, dispersao) = observaveis(&psi, N, S0, t);
            let mut erro_p = 0.0f64;
            for m in 0..N {
                let ex = amplitude_exata(m, S0, N, t);
                erro_p = erro_p.max(p[m] - (ex.0 * ex.0 + ex.1 * ex.1));
            }
            let erro_p = erro_p.abs();
            println!(
                "  t = {:>4.1} | eco {:>6.4} | norma {:>6.4} | E = {:>7.4} | \
                 erro P {erro_p:.2e} | ⟨m⟩ = {:>5.2} | σ = {:>4.2}",
                t, eco, soma_p, energia, centro, dispersao
            );
            if foto == PASSOS / T_FOTO {
                finais[prec_i] = Some(psi);
            }
            fisica[prec_i].push((t, soma_p, eco, erro_p));
        }
        tempos[prec_i] = t0.elapsed().as_secs_f64();
        if so_tempo {
            println!(
                "trajetória de 2.600 portas em {N} qubits ({}): {:.2} s ({:.2} ms/porta)",
                nome,
                tempos[prec_i],
                tempos[prec_i] * 1e3 / (26.0 * PASSOS as f64)
            );
        } else {
            println!("  (evolução + {} leituras: {:.1} s)", PASSOS / T_FOTO, tempos[prec_i]);
        }
    }

    // ─── 3. O perfil final contra a física exata ────────────────────────────
    if !so_tempo {
        let t_final = PASSOS as f64 * DT;
        println!("\nP(m, t = {t_final}) — a onda partiu do sítio {S0}, refletiu nas bordas:");
        println!("{:>4} {:>10} {:>10} {:>10} {:>10}", "m", "exato", "f32", "f16", "fronteira");
        for m in (0..N).step_by(1) {
            let ex = amplitude_exata(m, S0, N, t_final);
            let p_ex = ex.0 * ex.0 + ex.1 * ex.1;
            let p32 = finais[0]
                .as_ref()
                .map(|v| {
                    let a = v[1usize << m];
                    (a.0 * a.0 + a.1 * a.1) as f64
                })
                .unwrap_or(f64::NAN);
            let p16 = finais[1]
                .as_ref()
                .map(|v| {
                    let a = v[1usize << m];
                    (a.0 * a.0 + a.1 * a.1) as f64
                })
                .unwrap_or(f64::NAN);
            let borda = if m == 0 || m == N - 1 { "  ← borda" } else { "" };
            println!(
                "{m:>4} {p_ex:>10.5} {p32:>10.5} {p16:>10.5}{borda}"
            );
        }
    }

    println!(
        "\nnotas:\n\
         - a soma de P(m) é a norma no setor de 1 magnon: cai só por vazamento\n\
           aritmético (f16 vaza mais — o vazamento é física simulada, não bug);\n\
         - a energia E = 2J Σ Re(ψ*_m ψ_m+1) mede a conservação do Trotter;\n\
         - o eco |⟨ψ₀|ψ(t)⟩|² é o observável de Loschmidt, medido com o produto\n\
           interno novo, sem tocar nos estados."
    );
}
