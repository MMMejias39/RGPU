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

## Motivação do usuário

A pergunta de fundo por trás de todo o interesse em GPU sem CUDA e em medir
energia (não só tempo) neste repositório é **exergia**, não energia bruta:
quanto do trabalho computacional é aproveitado de verdade contra quanto vira
perda — porque o produto final é **portátil e alimentado por bateria**. Um
robô ou dispositivo de borda carrega a energia consigo; cada joule gasto em
ociosidade de GPU, em despacho de software mal ajustado ou numa métrica de
eficiência enganosa (como o caso do TensorFlow, ver README) é peso e volume
de bateria que o produto não devia precisar carregar.

Isso não é preocupação abstrata de eficiência de laboratório — é restrição
de engenharia real: o equipamento não pode precisar de mais bateria, em peso
e volume, do que ele mesmo.

## Direção futura (contexto, não roteiro — nada disto foi iniciado)

O usuário já possui hardware de borda para explorar esse eixo, ainda não
auditado neste repositório:

- **Raspberry Pi 5 + Hailo** (acelerador de inferência dedicado, ~2,5 W
  típico) — alvo primário para inferência de borda.
- **Jetson Nano** — alternativa caso o par RPi5+Hailo se mostre insuficiente.

A arquitetura de destino que o usuário descreve é em três camadas
(edge-fog-cloud):

1. **Borda** — inferência simples e rápida, decisões repetitivas sem
   depender de servidor; envia percepções para a camada seguinte só quando
   necessário.
2. **Rede local** — processamento mais pesado, menos urgente, com
   armazenamento para dezenas de aparelhos.
3. **Serviço central** — armazenamento e processamento massivo, aprende com
   a arquitetura inteira e retroalimenta preventivamente as camadas
   inferiores com o que aprendeu.

Hipótese de trabalho ainda não testada, levantada nesta sessão: cada camada
tende a rodar em hardware de fabricante diferente (Hailo na borda, GPU de
data center no topo), o que reabre a mesma pergunta central deste
repositório — uma pilha portátil (o equivalente do que se fez aqui com
WGSL/`wgpu`) pode evitar reescrever o modelo três vezes, uma por camada.
Nada disso foi medido; fica registrado para quando a investigação começar.
