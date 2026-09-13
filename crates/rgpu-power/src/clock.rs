//! Controle de frequência da GPU NVIDIA, para medir eficiência contra clock.
//!
//! # Por que clock e não potência
//!
//! Em GPU de notebook o limite de potência é território do firmware: o
//! `nvidia-smi -pl` responde *"not supported in current scope"* e `sudo` não
//! contorna, porque quem gerencia o envelope é o Dynamic Boost do fabricante.
//! Já travar a frequência falha por **permissão**, e portanto funciona como
//! root. Como a potência acompanha o clock, a varredura de frequência responde
//! à mesma pergunta: empurrar o relógio para cima compra o quê, por joule?
//!
//! # A trava se desfaz sozinha
//!
//! [`TravaClock`] devolve o controle automático no `Drop`. Sair do experimento
//! deixando a placa presa numa frequência baixa seria um efeito colateral
//! silencioso e difícil de diagnosticar depois.
//!
//! Ainda assim, `Drop` não roda se o processo for morto por sinal. Se isso
//! acontecer, o conserto é `sudo nvidia-smi -rgc`.

use std::process::{Command, Stdio};

/// `true` se este processo roda como root — sem isso, travar clock falha.
pub fn e_root() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Uid:"))?
                .split_whitespace()
                .nth(1)?
                .parse::<u32>()
                .ok()
        })
        .map(|uid| uid == 0)
        .unwrap_or(false)
}

/// Frequências de núcleo que a placa aceita, em MHz, da maior para a menor.
pub fn suportados() -> Vec<u32> {
    let saida = Command::new("nvidia-smi")
        .args(["--query-supported-clocks=graphics", "--format=csv,noheader,nounits"])
        .output()
        .ok();

    let Some(saida) = saida else { return Vec::new() };
    let mut v: Vec<u32> = String::from_utf8_lossy(&saida.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.dedup();
    v
}

/// Frequência efetiva do núcleo neste instante, em MHz.
pub fn atual() -> Option<u32> {
    let saida = Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&saida.stdout).trim().parse().ok()
}

/// Escolhe `quantos` pontos bem espalhados entre a menor e a maior frequência
/// suportada, para uma varredura com resolução uniforme.
pub fn amostrar(quantos: usize) -> Vec<u32> {
    let todos = suportados();
    if todos.is_empty() || quantos == 0 {
        return Vec::new();
    }
    if todos.len() <= quantos {
        return todos;
    }
    let (max, min) = (todos[0] as f64, *todos.last().unwrap() as f64);
    let mut pontos: Vec<u32> = (0..quantos)
        .map(|i| {
            let alvo = min + (max - min) * i as f64 / (quantos - 1) as f64;
            // O mais próximo entre os efetivamente suportados.
            *todos
                .iter()
                .min_by(|a, b| {
                    (**a as f64 - alvo)
                        .abs()
                        .total_cmp(&(**b as f64 - alvo).abs())
                })
                .unwrap()
        })
        .collect();
    pontos.sort_unstable();
    pontos.dedup();
    pontos
}

/// Trava a frequência do núcleo enquanto viver.
#[derive(Debug)]
pub struct TravaClock {
    mhz: u32,
}

impl TravaClock {
    /// Trava em `mhz`. Exige root.
    pub fn travar(mhz: u32) -> Result<TravaClock, String> {
        if !e_root() {
            return Err("travar a frequência exige root".into());
        }
        let saida = Command::new("nvidia-smi")
            .args(["-lgc", &format!("{mhz},{mhz}")])
            .output()
            .map_err(|e| format!("nvidia-smi não executou: {e}"))?;

        if !saida.status.success() {
            return Err(String::from_utf8_lossy(&saida.stderr).trim().to_string());
        }
        let texto = String::from_utf8_lossy(&saida.stdout);
        if texto.contains("not supported") || texto.contains("permission") {
            return Err(texto.trim().to_string());
        }
        Ok(TravaClock { mhz })
    }

    pub fn mhz(&self) -> u32 {
        self.mhz
    }
}

impl Drop for TravaClock {
    fn drop(&mut self) {
        let _ = Command::new("nvidia-smi")
            .arg("-rgc")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// Devolve a frequência ao controle automático, independentemente de travas.
pub fn liberar() -> bool {
    Command::new("nvidia-smi")
        .arg("-rgc")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn amostragem_respeita_o_que_a_placa_suporta() {
        let todos = suportados();
        if todos.is_empty() {
            return; // sem GPU NVIDIA nesta máquina
        }
        let pontos = amostrar(8);
        assert!(pontos.len() <= 8, "pediu 8, veio {}", pontos.len());
        assert!(!pontos.is_empty());
        for p in &pontos {
            assert!(todos.contains(p), "{p} MHz não é suportado");
        }
        assert!(pontos.windows(2).all(|w| w[0] < w[1]), "não está ordenado");
    }

    #[test]
    fn amostrar_zero_devolve_vazio() {
        assert!(amostrar(0).is_empty());
    }

    #[test]
    fn travar_sem_root_falha_com_mensagem_clara() {
        if e_root() {
            return;
        }
        let erro = TravaClock::travar(900).unwrap_err();
        assert!(erro.contains("root"), "mensagem inesperada: {erro}");
    }
}
