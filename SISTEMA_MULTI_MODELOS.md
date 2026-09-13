# 🤖 SISTEMA MULTI-MODELOS — leia isto ao iniciar qualquer instância

**Instância de IA:** este arquivo documenta o sistema de IA multi-modelo já instalado
neste computador. Antes de chamar APIs externas, use o que já está configurado aqui.

## Como usar (terminal)
- `ai "pergunta"` → roteia automaticamente (qualidade → velocidade → disponibilidade, com fallback)
- `ai rota "pergunta"` → mostra a cadeia de modelos sem executar
- `ai status` → matriz de modelos/velocidade/quota
- `ai` → menu interativo (também: `ai 1`…`ai 11`, `ai deepseek`, `ai claude`)
- Script: `~/bin/ai` | PATH já configurado no `~/.bashrc`

## Modelos locais (GPU 8 GB — gratuitos, sem quota)
| Modelo | Especialidade | Velocidade |
|---|---|---|
| `4skl/gemma4-e4b-mtp` | multimodal (visão/áudio), 64K ctx — PADRÃO p/ simples | ~172 tok/s |
| `qwen2.5-coder:7b` | código curto | ~53 tok/s |
| `qwen2.5:7b` | raciocínio leve | ~52 tok/s |
| `llama3.1:8b` | uso geral | ~49 tok/s |

## Modelos cloud (conta Ollama do usuário)
| Modelo | Especialidade | Benchmark (questão-trem) |
|---|---|---|
| `glm-5.3-flash:cloud` | multimodal, 1M ctx, análise complexa | ✔ 156 km em 2,8 s |
| `glm-5.3:cloud` | coding de fronteira | ✔ 2,7 s |
| `kimi-k2.7-code:cloud` | agentic coding | ✔ 4,2 s |
| `deepseek-v4.1-flash:cloud` | visão nativa, raciocínio | ✔ 9,4 s |

## CLIs agentic (contas pagas)
- `claude` — Fable 5.1/Mythos (topo em coding/knowledge work) — ✔ ativa
- `gemini` — ⛔ créditos esgotados (429) — recarregar em ai.studio/projects
- `qwen` (Qwen Code) — ⚠️ config quebrada: aponta p/ modelo removido `gemma4:26b` + exige OPENAI_API_KEY

## Cadeias de roteamento (implementadas em ~/bin/ai)
- Simples → gemma4 → llama → glm-flash
- Raciocínio complexo → glm-flash → deepseek → glm-5.3 → claude
- Código complexo → claude → kimi → glm-5.3
- Código simples → qwen2.5-coder → kimi
- Visão → gemma4 (imagem anexa) → glm-flash
- Força manual: incluir "usar local" ou "na nuvem" na frase

## Regras de projeto (definidas pelo usuário)
1. Tarefa rápida/rodável local → usar LOCAL; complexa ou pesada demais → cloud.
2. Distribuir por melhor qualidade da especialidade; empate → mais rápido; depois → tokens disponíveis.
3. Ao falhar (quota/429/modelo inexistente) → pular automaticamente para o próximo da cadeia.

## Contexto da instância
Este arquivo serve a qualquer instância, em qualquer pasta: **o projeto ativo é
o diretório de trabalho em que a instância foi aberta**, não um caminho fixado
aqui. Ao iniciar, trate a pasta atual como o contexto do trabalho.

Notas, materiais e correções de um projeto específico ficam na pasta do
próprio projeto — por exemplo, o material de `~/Documentos/Compressores/` tem
suas correções priorizadas em `dicas_ollama.md` da própria pasta. Este arquivo
só carrega o que é comum a todo o computador: modelos, rotas e regras.
