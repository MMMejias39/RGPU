# RGPU

Computação em GPU em **Rust puro**, independente de fabricante — sem CUDA, sem
`unsafe`, sem FFI. Os kernels são escritos em WGSL e executam sobre Vulkan,
Metal, DX12 ou WebGPU, então o mesmo código roda em NVIDIA, AMD, Intel e Apple.

## Crates

| Crate | O que é |
|---|---|
| [`rtensor`](crates/rtensor) | Framework de deep learning: tensor com broadcasting, autodiff reverso, camadas, otimizadores e backend de GPU |
| [`rgpu-power`](crates/rgpu-power) | Medição de energia: potência da GPU por `nvidia-smi`, energia da CPU e da GPU integrada por RAPL |

## Desempenho

Medido contra TensorFlow, PyTorch, [burn](https://github.com/tracel-ai/burn) e
[candle](https://github.com/huggingface/candle) — mesma máquina, mesma carga.
Os números incluem as derrotas; são elas que dão crédito ao resto.

### Tempo por passo de treino

MLP 512→1024→1024→10, RTX 4070 Laptop, Intel Meteor Lake (22 threads).
Milissegundos por passo, menor é melhor; **negrito** marca o vencedor da linha.

| Lote | rtensor GPU | burn-wgpu | TF GPU | torch CUDA | rtensor CPU | TF CPU | torch CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,66 | 0,54 | 1,93 | **0,55** | 8,46 | 6,27 | 3,24 |
| 8 | 0,67 | 1,02 | 2,05 | **0,52** | 8,21 | 7,56 | 2,74 |
| 32 | **0,73** | 1,54 | 1,98 | 0,54 | 14,01 | 9,54 | 3,15 |
| 128 | 1,47 | 1,85 | 1,79 | **0,55** | 37,98 | 12,10 | 4,56 |
| 512 | 2,34 | 1,27 | 1,15 | **0,65** | 189,48 | 10,51 | 12,53 |
| 2048 | 6,89 | 3,44 | 2,02 | **2,08** | 709,83 | 32,01 | 42,22 |

*candle CPU, medido em lote 512: 67,48 ms.*

**O PyTorch com CUDA vence em todos os tamanhos de lote**, e é notavelmente
plano — 0,52 a 0,65 ms de lote 1 a 512. O `rtensor` é competitivo em lote
pequeno e médio, e perde em lote grande.

### O resultado mais desconfortável

O **burn** passa pelo **mesmo caminho de hardware** que o `rtensor`: wgpu sobre
Vulkan, mesmo driver, mesma placa. Sem cuBLAS, sem tensor cores, sem API
privilegiada. E mesmo assim é **1,8× mais rápido em lote 512** e **2× em lote
2048**.

Como tudo está igualado por baixo, a diferença é qualidade de implementação —
nossa. O burn usa o CubeCL com **fusão de kernels**: junta operações elementares
dentro do GEMM e evita ida e volta à memória. É a otimização que ainda não
fizemos.

A inversão também é real: em lote 32 o `rtensor` é **2,1× mais rápido que o
burn**, e o burn se comporta de forma não monotônica — 1,85 ms em lote 128,
caindo para 1,27 ms em lote 512. É o autotuner dele pagando custo fixo que só se
amortiza com trabalho grande. O plano estático do `rtensor` não paga esse custo.

### Onde o rtensor ganha de todos

Partida a frio: processo inteiro, até o primeiro passo de treino concluído.

| | Tempo | Primeiro passo |
|---|---:|---:|
| **rtensor** | **0,29–0,37 s** | **2,1 ms** |
| burn | 1,26 s | — |
| PyTorch | 2,41 s | 124 ms |
| TensorFlow | 5,64 s | 1.367 ms |

De 3,4× a 17× mais rápido que os demais. Para um script que roda e termina,
isso domina tudo o mais.

### Onde o rtensor perde feio

| Motor | CPU, GFLOP/s |
|---|---:|
| torch | 461 |
| TensorFlow | ~150 |
| candle | 72 |
| **rtensor** | **26** |

18× mais lento que o PyTorch, por uma razão conhecida: **o caminho de CPU é
monothread**, enquanto os outros usam os 22 núcleos.

### GEMM

O núcleo é ladrilhado em dois níveis, como no cuBLAS: ladrilho 64×64 por
workgroup em memória compartilhada, bloco 4×4 por thread em registradores, com
**buffer duplo** — as leituras do ladrilho `t+1` são emitidas antes do cálculo
sobre o ladrilho `t`, escondendo a latência da memória atrás da aritmética.

| `C[2048³]` | GFLOP/s |
|---|---:|
| ingênuo, 1 elemento por thread | 840 |
| ladrilhado | 3.290 |
| ladrilhado + buffer duplo | 4.230 |
| + rasterização com consciência de L2 | **4.386** |
| cuBLAS (via TensorFlow, com TF32) | 11.946 |

A rasterização segue a blocagem de L2 do Goto na forma que uma GPU permite: não
se controla a cache, controla-se a **ordem em que os blocos a visitam**. Em vez
de percorrer a grade em linha, agrupam-se 8 linhas de blocos, de modo que os
painéis compartilhados ainda estejam na L2 quando forem reusados. Medido em
4096³, em três execuções: **+5,1%, +5,9% e +5,1%** sobre o percurso em linha, com
a curva subindo até o grupo 8–16 e colapsando em 32 — o formato esperado de
quando o grupo deixa de caber na cache.

O bloco `8×8` por thread, que o cuBLAS usa, foi implementado e medido **5% mais
lento** — a sonda `examples/ocupacao.rs` explica por quê: aritmética pura
sustenta 15–18 TFLOP/s nesta placa, e o GEMM anda a 4. Estando 4× longe do
limite das ULAs, dobrar a intensidade aritmética só custa ocupação.

### Strassen de um nível

Sete produtos de metade das dimensões no lugar de oito: **12,5% menos
multiplicações**, à custa de somas `O(n²)`, empacotamento dos blocos e 25
dispatches em vez de 1. Há um tamanho abaixo do qual não compensa, e ele foi
medido:

| Tamanho | Ganho | Erro máx. clássico | Erro máx. Strassen |
|---:|---:|---:|---:|
| 1024³ | −52,0% | 3,1·10⁻⁶ | 1,2·10⁻⁵ |
| 2048³ | −23,6% | 4,5·10⁻⁶ | 1,3·10⁻⁵ |
| **4096³** | **+11,2% a +13,3%** | 7,0·10⁻⁶ | 2,0·10⁻⁵ |
| 6144³ | +8,6% | 1,9·10⁻⁵ | 3,1·10⁻⁵ |
| 8192³ | +6,6% | 8,1·10⁻⁶ | 2,6·10⁻⁵ |

O cruzamento fica entre 2048 e 4096, e o pico em 4096 captura **90% do ganho
teórico** de 12,5%.

Strassen satisfaz apenas um limite de erro por norma, e não o limite por
elemento do algoritmo clássico. A literatura reporta cerca de duas ordens de
grandeza de erro a mais, mas isso é com vários níveis de recursão em `n =
16384`. Com **um** nível, nas faixas acima, o custo medido é de **2,5× a 3,2×**
o erro do algoritmo clássico — número que a literatura não dá para estes
tamanhos.

Por perder abaixo de 4096, Strassen não é automático: `Strassen::novo` monta o
plano explicitamente. `tests/gpu.rs` mede o erro extra a cada execução, em vez
de supô-lo.

Ainda 2,7× atrás do cuBLAS. Mas com o TF32 desligado o TensorFlow cai só 12%, o
que localiza a maior parte da diferença em pipelining, tiling multinível e
swizzling — todos ao alcance do WGSL — e não nos tensor cores.

### Reproduzindo

```bash
cargo run -p rtensor --release --features gpu --example suite -- batch   # rtensor
cd benchmarks/rivais && cargo run --release --bin burn_bench -- 512      # burn
cd benchmarks/rivais && cargo run --release --bin candle_bench -- 512    # candle
```

Os scripts de TensorFlow e PyTorch, com as instruções de ambiente, estão em
[`benchmarks/python/`](benchmarks/python/).

Versões medidas: burn 0.21, candle 0.11, TensorFlow 2.21.0, PyTorch 2.14.0+cu132.

## Por que energia, e não só tempo

Tempo diz quão rápido; energia diz quanto custou. Um passo duas vezes mais
rápido consumindo três vezes mais potência é um retrocesso de eficiência, e só
a medição separa os dois casos.

Nenhum dos frameworks acima publica medição de energia. Esta é a contribuição
distinta do RGPU.

### O desperdício está em como o hardware é alimentado

Mesmo modelo, mesma placa, mesmo código — só muda quanto trabalho chega por
vez. RTX 4070 Laptop, ociosidade de 14,1 W:

| Lote | Potência média | µJ por amostra | GFLOP/J |
|---:|---:|---:|---:|
| 1 | 38,1 W | **12.793,5** | 0,7 |
| 8 | 46,0 W | 2.119,2 | 4,5 |
| 32 | 47,2 W | 599,8 | 15,8 |
| 128 | 48,5 W | 290,7 | 32,7 |
| 512 | 66,8 W | 196,6 | 48,3 |
| 2048 | 69,4 W | **164,3** | 57,8 |

A potência média sobe menos de **2×** entre o lote 1 e o lote 2048. A vazão
sobe **182×**. O resultado é **78× de diferença na energia por amostra**.

O chip custa quase o mesmo ligado, fazendo muito ou pouco trabalho. Quem treina
com lote pequeno nesta placa desperdiça cerca de 98% da energia que gasta — e
nenhuma troca de hardware corrige isso.

### Entre fabricantes, o mesmo código

Energia acima da ociosidade, lote 512:

| Motor | ms/passo | J/passo | GFLOP/s | GFLOP/J |
|---|---:|---:|---:|---:|
| RTX 4070 Laptop | 1,92 | 0,0874 | 2.535 | **55,64** |
| Intel Arc (iGPU) | 12,69 | 0,1038 | 383 | **46,85** |
| CPU, 1 núcleo | 156,48 | 2,2574 | 31 | **2,15** |

A placa dedicada é **6,6× mais rápida** que a gráfica integrada, e apenas **19%
mais eficiente por joule**. A vantagem da GPU discreta é velocidade, quase nada
é eficiência — e ela consome **11,9 W só por estar ligada**.

O salto de eficiência está em sair da CPU para *qualquer* GPU: a iGPU é 26×
mais eficiente que um núcleo.

*Ressalva:* o `nvidia-smi` mede a placa inteira (GPU, VRAM, regulação); o
domínio `uncore` do RAPL cobre só a iGPU dentro do SoC. O número da Intel está
provavelmente subestimado, o que reforça a conclusão.

### Eficiência contra frequência

Nove frequências, mesma carga, **energia total** (a ociosidade não é subtraível
aqui: ela muda junto com o ponto de operação):

| Clock efetivo | GFLOP/s | Potência | GFLOP/J |
|---:|---:|---:|---:|
| 210 MHz | 251 | 12,3 W | 20,4 |
| 570 MHz | 658 | 16,6 W | 39,6 |
| 930 MHz | 1.033 | 19,8 W | 52,2 |
| **1.290 MHz** | **1.422** | **24,4 W** | **58,3** |
| 1.665 MHz | 1.810 | 31,5 W | 57,5 |
| 2.025 MHz | 2.167 | 41,3 W | 52,5 |
| 2.385 MHz | 2.516 | 59,8 W | 42,1 |
| 2.490 MHz | 2.528 | 62,3 W | 40,6 |

A curva é um sino, com pico em **1.290 MHz**. Do ponto mais rápido para o mais
eficiente: **44% menos velocidade, 44% mais trabalho por joule, 61% menos
potência** — 24,4 W contra 62,3 W.

Na direção oposta: subir de 1.290 para 2.490 MHz dá **78% de desempenho
custando 155% de potência**. A potência cresce ao dobro da taxa do desempenho.

Pedimos 2.745 e 3.105 MHz; a placa entregou 2.490 e 2.520. O firmware recusa ir
além sob carga — e o instrumento só percebeu porque lê o clock efetivo em vez
de presumir o pedido.

### Limites desta máquina

Varrer o **limite de potência** (`nvidia-smi -pl`) não é possível em GPU de
notebook: o driver responde *"not supported in current scope"*, porque quem
controla o envelope é o firmware, com o Dynamic Boost ativo. `sudo` não
contorna. Já **travar o clock** (`nvidia-smi -lgc`) falha por permissão, e
portanto funciona com `sudo`:

```bash
cargo build -p rtensor --release --features gpu --example frequencia
sudo ./target/release/examples/frequencia
```

A trava se desfaz sozinha ao fim de cada ponto e no encerramento. Se o processo
for morto por sinal, o conserto é `sudo nvidia-smi -rgc`.

A energia da CPU e da GPU integrada vem dos contadores RAPL, restritos a `root`
desde a mitigação do PLATYPUS (2020).

## Rodando

```bash
cargo test --workspace --features rtensor/gpu       # 25 testes
cargo run -p rtensor --release --features gpu --example eficiencia   # energia por ocupação
cargo run -p rtensor --release --features gpu --example energia      # energia por motor e fabricante
cargo run -p rtensor --release --features gpu --example perfil       # perfilamento por kernel
```

## O que não funcionou

Toda otimização tentada, incluindo as revertidas, e os erros de medição que
custaram mais tempo que os erros de código, estão em
[RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md). Publicar só os acertos
falsificaria a taxa de sucesso real do trabalho.

## Licença

MIT.
