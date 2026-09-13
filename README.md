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
| ladrilhado + buffer duplo | **4.002** |
| cuBLAS (via TensorFlow, com TF32) | 11.946 |

Ainda 3× atrás do cuBLAS. Mas com o TF32 desligado o TensorFlow cai só 12%, o
que localiza a maior parte da diferença em pipelining, tiling multinível e
swizzling — todos ao alcance do WGSL — e não nos tensor cores.

### Reproduzindo

```bash
cargo run -p rtensor --release --features gpu --example suite -- batch   # rtensor
cd benchmarks/rivais && cargo run --release --bin burn_bench -- 512      # burn
cd benchmarks/rivais && cargo run --release --bin candle_bench -- 512    # candle
```

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

## Licença

MIT.
