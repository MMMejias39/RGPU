//! Medição de energia para cargas de computação.
//!
//! Tempo diz quão rápido; energia diz quanto custou. Esta crate mede a segunda
//! coisa, sem dependências externas e sem `unsafe`.
//!
//! # Fontes
//!
//! | Fonte | O que cobre | Privilégio |
//! |---|---|---|
//! | `nvidia-smi` | potência instantânea da GPU NVIDIA | nenhum |
//! | RAPL (`/sys/class/powercap`) | pacote da CPU, núcleos, `uncore` (iGPU), plataforma | **root** |
//!
//! A leitura do RAPL exige privilégio desde 2020: os contadores de energia
//! vazam informação suficiente para o ataque de canal lateral PLATYPUS, e as
//! distribuições passaram a restringi-los a `root`. Sem esse acesso o medidor
//! ainda entrega a energia da GPU NVIDIA, que é o essencial; com ele, entrega
//! também CPU e GPU integrada.
//!
//! # Como a energia é obtida
//!
//! O `nvidia-smi` reporta **potência instantânea**, não energia. O medidor
//! amostra em intervalo fixo e integra pela regra do trapézio:
//!
//! ```text
//! E = Σᵢ (tᵢ₊₁ − tᵢ) · (Pᵢ₊₁ + Pᵢ) / 2
//! ```
//!
//! O RAPL, ao contrário, já expõe um **contador de energia acumulada**; basta a
//! diferença entre início e fim, com tratamento da volta ao zero.
//!
//! # Energia acima da ociosidade
//!
//! Uma GPU ligada consome mesmo sem trabalho — 14,7 W numa RTX 4070 Laptop em
//! repouso. Atribuir esse consumo à carga superestima o custo dela. Por isso o
//! medidor calibra a ociosidade e reporta as duas leituras: a energia total da
//! janela e a parcela **acima da linha de base**.
//!
//! ```no_run
//! use rgpu_power::Medidor;
//!
//! let mut medidor = Medidor::novo();
//! medidor.calibrar_ociosidade(2.0);
//!
//! let (resultado, medicao) = medidor.medir(|| minha_carga());
//! println!("{}", medicao.relatorio());
//! # fn minha_carga() -> u32 { 0 }
//! ```

pub mod clock;

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// Uma leitura de potência: segundos desde o início da janela, e watts.
pub type Amostra = (f64, f64);

// ----------------------------------------------------------------- nvidia-smi

/// Amostrador de potência da GPU NVIDIA, alimentado por `nvidia-smi` em modo
/// contínuo (`-lms`). Um único processo é aberto para toda a janela: chamar o
/// `nvidia-smi` a cada amostra custaria dezenas de milissegundos por leitura.
pub struct MonitorGpu {
    filho: Child,
    amostras: Arc<Mutex<Vec<Amostra>>>,
    parar: Arc<AtomicBool>,
    leitor: Option<JoinHandle<()>>,
}

impl MonitorGpu {
    /// Abre o monitor. Devolve `None` se não houver `nvidia-smi` utilizável.
    pub fn iniciar(intervalo_ms: u64, inicio: Instant) -> Option<MonitorGpu> {
        let mut filho = Command::new("nvidia-smi")
            .args([
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                &format!("-lms={intervalo_ms}"),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        let saida = filho.stdout.take()?;
        let amostras = Arc::new(Mutex::new(Vec::new()));
        let parar = Arc::new(AtomicBool::new(false));

        let a = Arc::clone(&amostras);
        let p = Arc::clone(&parar);
        let leitor = std::thread::spawn(move || {
            for linha in BufReader::new(saida).lines() {
                if p.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(linha) = linha else { break };
                if let Ok(w) = linha.trim().parse::<f64>() {
                    a.lock().unwrap().push((inicio.elapsed().as_secs_f64(), w));
                }
            }
        });

        Some(MonitorGpu { filho, amostras, parar, leitor: Some(leitor) })
    }

    /// Encerra o monitor e devolve as amostras coletadas.
    pub fn parar(mut self) -> Vec<Amostra> {
        self.parar.store(true, Ordering::Relaxed);
        let _ = self.filho.kill();
        let _ = self.filho.wait();
        if let Some(h) = self.leitor.take() {
            let _ = h.join();
        }
        let v = self.amostras.lock().unwrap().clone();
        v
    }
}

/// Integra potência no tempo pela regra do trapézio, em joules.
pub fn integrar(amostras: &[Amostra]) -> f64 {
    amostras
        .windows(2)
        .map(|p| (p[1].0 - p[0].0) * (p[1].1 + p[0].1) / 2.0)
        .sum()
}

// ------------------------------------------------------------------ RAPL

/// Um domínio de energia do RAPL.
pub struct DominioRapl {
    pub nome: String,
    caminho: PathBuf,
    maximo: u64,
}

impl DominioRapl {
    fn ler(&self) -> Option<u64> {
        fs::read_to_string(&self.caminho).ok()?.trim().parse().ok()
    }
}

/// Conjunto de domínios RAPL legíveis por este processo.
pub struct Rapl {
    pub dominios: Vec<DominioRapl>,
}

impl Rapl {
    /// Varre `/sys/class/powercap` e fica só com o que dá para ler de fato.
    pub fn detectar() -> Rapl {
        let mut dominios = Vec::new();
        let Ok(entradas) = fs::read_dir("/sys/class/powercap") else {
            return Rapl { dominios };
        };

        for e in entradas.flatten() {
            let dir = e.path();
            let nome_dir = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !nome_dir.starts_with("intel-rapl:") {
                continue;
            }
            let energia = dir.join("energy_uj");
            // Só entra se a leitura funcionar agora — sem root, não funciona.
            if fs::read_to_string(&energia).is_err() {
                continue;
            }
            let nome = fs::read_to_string(dir.join("name"))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| nome_dir.to_string());
            let maximo = fs::read_to_string(dir.join("max_energy_range_uj"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(u64::MAX);
            dominios.push(DominioRapl { nome, caminho: energia, maximo });
        }
        dominios.sort_by(|a, b| a.nome.cmp(&b.nome));
        Rapl { dominios }
    }

    pub fn disponivel(&self) -> bool {
        !self.dominios.is_empty()
    }

    fn instantaneo(&self) -> Vec<Option<u64>> {
        self.dominios.iter().map(|d| d.ler()).collect()
    }

    /// Energia consumida entre dois instantâneos, em joules por domínio.
    /// O contador é de 32 bits em muitos processadores e dá a volta; quando
    /// isso acontece, soma-se o alcance máximo.
    fn delta(&self, antes: &[Option<u64>], depois: &[Option<u64>]) -> Vec<(String, f64)> {
        self.dominios
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                let (a, b) = (antes.get(i)?.as_ref()?, depois.get(i)?.as_ref()?);
                let bruto = if b >= a { b - a } else { (self.maximo(i) - a) + b };
                Some((d.nome.clone(), bruto as f64 / 1e6))
            })
            .collect()
    }

    fn maximo(&self, i: usize) -> u64 {
        self.dominios[i].maximo
    }
}

// ---------------------------------------------------------------- medidor

/// Energia atribuída à GPU NVIDIA numa janela de medição.
pub struct EnergiaGpu {
    pub total_j: f64,
    pub media_w: f64,
    pub pico_w: f64,
    /// Energia descontada a linha de base ociosa, quando calibrada.
    pub acima_ociosidade_j: Option<f64>,
    pub amostras: usize,
}

/// Resultado de uma medição.
pub struct Medicao {
    pub duracao_s: f64,
    pub gpu: Option<EnergiaGpu>,
    /// Energia bruta por domínio RAPL na janela.
    pub rapl_j: Vec<(String, f64)>,
    /// Energia por domínio RAPL descontada a linha de base ociosa, quando
    /// calibrada. Sem o desconto, o consumo de repouso da máquina — que existe
    /// com ou sem a carga — seria atribuído ao trabalho medido.
    pub rapl_acima_j: Vec<(String, f64)>,
}

impl Medicao {
    /// Energia que melhor representa o custo da carga: acima da ociosidade
    /// quando há calibração, total caso contrário.
    pub fn energia_gpu_j(&self) -> Option<f64> {
        self.gpu.as_ref().map(|g| g.acima_ociosidade_j.unwrap_or(g.total_j))
    }

    /// Energia bruta de um domínio RAPL pelo nome
    /// (`package-0`, `core`, `uncore`, `psys`).
    pub fn rapl(&self, nome: &str) -> Option<f64> {
        self.rapl_j.iter().find(|(n, _)| n == nome).map(|(_, j)| *j)
    }

    /// Energia de um domínio RAPL acima da ociosidade. Cai para a leitura
    /// bruta se não houve calibração.
    pub fn rapl_acima(&self, nome: &str) -> Option<f64> {
        self.rapl_acima_j
            .iter()
            .find(|(n, _)| n == nome)
            .map(|(_, j)| *j)
            .or_else(|| self.rapl(nome))
    }

    pub fn relatorio(&self) -> String {
        let mut s = format!("duração {:.3} s\n", self.duracao_s);
        match &self.gpu {
            Some(g) => {
                s.push_str(&format!(
                    "  GPU NVIDIA: {:.2} J totais, média {:.1} W, pico {:.1} W ({} amostras)\n",
                    g.total_j, g.media_w, g.pico_w, g.amostras
                ));
                if let Some(j) = g.acima_ociosidade_j {
                    s.push_str(&format!("  GPU acima da ociosidade: {j:.2} J\n"));
                }
            }
            None => s.push_str("  GPU NVIDIA: indisponível\n"),
        }
        if self.rapl_j.is_empty() {
            s.push_str("  RAPL: indisponível (exige root)\n");
        } else {
            for (nome, j) in &self.rapl_j {
                let acima = self
                    .rapl_acima_j
                    .iter()
                    .find(|(n, _)| n == nome)
                    .map(|(_, v)| format!(", {v:.2} J acima da ociosidade"))
                    .unwrap_or_default();
                s.push_str(&format!("  RAPL {nome}: {j:.2} J{acima}\n"));
            }
        }
        s
    }
}

/// O medidor. Detecta as fontes disponíveis na construção.
pub struct Medidor {
    intervalo_ms: u64,
    rapl: Rapl,
    ociosidade_w: Option<f64>,
    /// Potência ociosa por domínio RAPL, em watts.
    ociosidade_rapl_w: Vec<(String, f64)>,
    tem_nvidia: bool,
}

impl Default for Medidor {
    fn default() -> Self {
        Self::novo()
    }
}

impl Medidor {
    pub fn novo() -> Medidor {
        let tem_nvidia = Command::new("nvidia-smi")
            .arg("--query-gpu=power.draw")
            .arg("--format=csv,noheader,nounits")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        Medidor {
            intervalo_ms: 50,
            rapl: Rapl::detectar(),
            ociosidade_w: None,
            ociosidade_rapl_w: Vec::new(),
            tem_nvidia,
        }
    }

    pub fn com_intervalo(mut self, ms: u64) -> Medidor {
        self.intervalo_ms = ms;
        self
    }

    pub fn tem_nvidia(&self) -> bool {
        self.tem_nvidia
    }

    pub fn tem_rapl(&self) -> bool {
        self.rapl.disponivel()
    }

    /// Descreve o que dá para medir nesta máquina.
    pub fn fontes(&self) -> String {
        let mut s = String::new();
        s.push_str(if self.tem_nvidia {
            "nvidia-smi: sim"
        } else {
            "nvidia-smi: não"
        });
        if self.rapl.disponivel() {
            let nomes: Vec<&str> = self.rapl.dominios.iter().map(|d| d.nome.as_str()).collect();
            s.push_str(&format!("  |  RAPL: {}", nomes.join(", ")));
        } else {
            s.push_str("  |  RAPL: sem permissão (exige root)");
        }
        s
    }

    /// Mede a potência ociosa por alguns segundos, para descontar depois —
    /// tanto da GPU NVIDIA quanto de cada domínio RAPL. Chame com a máquina em
    /// repouso.
    pub fn calibrar_ociosidade(&mut self, segundos: f64) {
        let inicio = Instant::now();
        let monitor = if self.tem_nvidia {
            MonitorGpu::iniciar(self.intervalo_ms, inicio)
        } else {
            None
        };
        let rapl_antes = self.rapl.instantaneo();

        std::thread::sleep(std::time::Duration::from_secs_f64(segundos));

        let decorrido = inicio.elapsed().as_secs_f64().max(1e-9);
        let rapl_depois = self.rapl.instantaneo();
        self.ociosidade_rapl_w = self
            .rapl
            .delta(&rapl_antes, &rapl_depois)
            .into_iter()
            .map(|(nome, j)| (nome, j / decorrido))
            .collect();

        if let Some(monitor) = monitor {
            let amostras = monitor.parar();
            if amostras.len() >= 2 {
                let soma: f64 = amostras.iter().map(|(_, w)| w).sum();
                self.ociosidade_w = Some(soma / amostras.len() as f64);
            }
        }
    }

    /// Potência ociosa medida por domínio RAPL.
    pub fn ociosidade_rapl_w(&self) -> &[(String, f64)] {
        &self.ociosidade_rapl_w
    }

    pub fn ociosidade_w(&self) -> Option<f64> {
        self.ociosidade_w
    }

    /// Executa `f` medindo energia. Devolve o resultado de `f` e a medição.
    pub fn medir<T>(&self, f: impl FnOnce() -> T) -> (T, Medicao) {
        let inicio = Instant::now();
        let monitor = if self.tem_nvidia {
            MonitorGpu::iniciar(self.intervalo_ms, inicio)
        } else {
            None
        };
        let rapl_antes = self.rapl.instantaneo();

        let saida = f();

        let duracao_s = inicio.elapsed().as_secs_f64();
        let rapl_depois = self.rapl.instantaneo();
        let amostras = monitor.map(|m| m.parar()).unwrap_or_default();

        let gpu = (amostras.len() >= 2).then(|| {
            let total_j = integrar(&amostras);
            let media_w = total_j / (amostras.last().unwrap().0 - amostras[0].0).max(1e-9);
            let pico_w = amostras.iter().map(|(_, w)| *w).fold(f64::MIN, f64::max);
            EnergiaGpu {
                total_j,
                media_w,
                pico_w,
                acima_ociosidade_j: self.ociosidade_w.map(|o| (total_j - o * duracao_s).max(0.0)),
                amostras: amostras.len(),
            }
        });

        let rapl_j = self.rapl.delta(&rapl_antes, &rapl_depois);
        let rapl_acima_j = rapl_j
            .iter()
            .filter_map(|(nome, j)| {
                let ocioso = self.ociosidade_rapl_w.iter().find(|(n, _)| n == nome)?.1;
                Some((nome.clone(), (j - ocioso * duracao_s).max(0.0)))
            })
            .collect();

        (saida, Medicao { duracao_s, gpu, rapl_j, rapl_acima_j })
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn trapezio_integra_potencia_constante() {
        // 10 W por 2 s = 20 J.
        let amostras = vec![(0.0, 10.0), (1.0, 10.0), (2.0, 10.0)];
        assert!((integrar(&amostras) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn trapezio_integra_rampa() {
        // Rampa de 0 a 10 W em 1 s = 5 J.
        assert!((integrar(&[(0.0, 0.0), (1.0, 10.0)]) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn janela_vazia_nao_quebra() {
        assert_eq!(integrar(&[]), 0.0);
        assert_eq!(integrar(&[(0.0, 50.0)]), 0.0);
    }

    #[test]
    fn medidor_relata_as_fontes() {
        let m = Medidor::novo();
        let f = m.fontes();
        assert!(f.contains("nvidia-smi"), "relatório sem nvidia-smi: {f}");
        assert!(f.contains("RAPL"), "relatório sem RAPL: {f}");
    }

    #[test]
    fn medir_devolve_o_resultado_e_a_duracao() {
        let m = Medidor::novo();
        let (v, medicao) = m.medir(|| {
            std::thread::sleep(std::time::Duration::from_millis(120));
            42u32
        });
        assert_eq!(v, 42);
        assert!(medicao.duracao_s >= 0.12, "duração {}", medicao.duracao_s);
    }
}
