# CONTEXTO.md — instruções permanentes para instâncias de IA

Regras que valem para qualquer tarefa neste repositório, em qualquer sessão.

## Como reportar o que foi feito

1. **Dar maiores informações sobre o que fez.** Toda mudança vem acompanhada de:
   - o que foi alterado e **por quê** (hipótese, diagnóstico, decisão);
   - onde está o código (arquivo, função, teste);
   - o que foi medido, com o instrumento usado (cronômetro, `perfil.rs`, `rgpu-power`);
   - o que **não** foi verificado, e o que falta medir.

2. **Apresentar resultados tabelados e comparativos.** Números soltos no texto
   não bastam. Sempre que houver medição:
   - **tabela** com as condições da medição (tamanho, precisão, iterações);
   - **comparação antes/depois** — a linha de base ao lado do resultado, com o
     ganho ou a perda em % ou ×;
   - **comparativo entre alternativas** — a mesma carga medida nos concorrentes
     (cuBLAS, cuStateVec, Aer, burn, PyTorch) quando existirem;
   - valor previsto ao lado do valor **medido**, quando houver previsão — e o
     desvio entre os dois discutido, não escondido.

3. **Seguir a política do repositório** ([RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md)):
   não se reporta ganho que não foi medido; resultado dentro do ruído é dito
   como tal; derrotas entram na tabela junto com as vitórias.

O padrão de formato é o das tabelas do [README.md](README.md): condições no
cabeçalho, **negrito** no vencedor da linha, unidades explícitas.
