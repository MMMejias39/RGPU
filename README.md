# RGPU

Computação em GPU em **Rust puro**, independente de fabricante — sem CUDA, sem
`unsafe`, sem FFI. Os kernels são escritos em WGSL e executam sobre Vulkan,
Metal, DX12 ou WebGPU, então o mesmo código roda em NVIDIA, AMD, Intel e Apple.

## Crates

| Crate | O que é |
|---|---|
| [`rtensor`](crates/rtensor) | Framework de deep learning: tensor com broadcasting, autodiff reverso, camadas, otimizadores e backend de GPU |
| [`rgpu-power`](crates/rgpu-power) | Medição de energia: potência da GPU por `nvidia-smi`, energia da CPU e da GPU integrada por RAPL |
| [`rqubit`](crates/rqubit) | Simulador de vetor de estado quântico: kernels especializados de 1 a 6 qubits, fusão de portas e medição de energia |

## O que este repositório mostra

Três resultados, cada um medido e reproduzível:

1. **O lock-in do CUDA é contornável.** O mesmo WGSL roda em NVIDIA e Intel, e
   na simulação quântica **empata com o `cuStateVec`** — a biblioteca da própria
   NVIDIA — por porta aplicada.
2. **O desperdício está no software, não no silício.** Mudando só o tamanho do
   lote, a energia por amostra treinada varia **78×** no mesmo chip.
3. **Tempo e energia são métricas diferentes.** Em quatro ocasiões a medição de
   energia contradisse a de tempo, e nenhum dos frameworks comparados —
   TensorFlow, PyTorch, burn, candle, Qiskit Aer, cuQuantum — publica a segunda.

E duas lições de método que atravessam tudo: **a otimização certa depende do
regime**, e o regime precisa ser medido antes — a mesma precisão mista é perda
de 10% de energia no GEMM e ganho de 2× na simulação quântica. E o diagnóstico
precisa ser conferido contra o que o hardware realmente anuncia: a matriz
cooperativa, arquivada como "não funcional", funcionava — a configuração
testada não existia na placa.

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

*Ressalva:* a coluna do TensorFlow é **notavelmente plana** — 1,15 a 2,05 ms
em toda a faixa de lote, quase sem escalar com o trabalho — e isso tem
explicação: `benchmarks/python/despacho_tf.py` mede o despacho de uma
`tf.function` **trivial** (soma quase nula, mesmo padrão de chamada) contra o
passo real, no mesmo lote:

| Motor | passo real | despacho trivial | despacho, % do passo |
|---|---:|---:|---:|
| TensorFlow | 1,68–2,09 ms | **0,43–0,70 ms** | **~29%** |
| PyTorch | 1,01–2,27 ms | 0,04–0,06 ms | ~3% |
| rtensor (`despacho_rtensor.rs`) | 0,49–5,89 ms | 0,03–0,07 ms | 1–10%, caindo com o lote |

Cerca de **30% de cada número da coluna TF GPU é só despachar a chamada** —
Python entrando na função traçada —, não computar. É por isso que a coluna
quase não escala: um custo fixo de ~0,5 ms domina o tempo em lotes pequenos
e médios, e só fica pequeno relativo ao resto em lote 2048. `torch` e
`rtensor` têm despacho perto do desprezível — a diferença entre eles nesta
tabela é mesmo computação. Isso não muda o vencedor de nenhuma linha, mas
explica por que o TensorFlow parece "travado" num único número: ele está.
Metodologia e os três scripts (`despacho_tf.py`, `despacho_torch.py`,
`despacho_rtensor.rs`) em [RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md).

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

*Ressalva:* essa vantagem **não é só arquitetura CUDA-free** — parte dela é
imposto de importar Python. Decompondo em três fases (`frio_decomposto_tf.py`,
`frio_decomposto_torch.py`, mediana de 5 execuções, TensorFlow 2.21.0, PyTorch
2.14.0+cu130):

| Motor | Import da biblioteca | Init. de GPU/modelo | Primeiro passo |
|---|---:|---:|---:|
| **rtensor** | — (binário nativo) | **0,317 s** | **0,0022 s** |
| PyTorch | 1,459 s | 1,457 s | 0,229 s |
| TensorFlow | 1,871 s | 1,260 s | 1,431 s |

Cerca de **um terço** do tempo total de TensorFlow e PyTorch é só `import` —
carregar a biblioteca Python, nada de GPU. Isso não tem nada a ver com CUDA
ser ou não contornável; é o preço de a linguagem hospedeira ser interpretada,
e um binário Rust não paga esse preço.

A comparação que sobra depois de descontar o import ainda favorece o
`rtensor`, mas por uma razão diferente da alegada: **inicializar o
dispositivo** (`wgpu`: instância, adaptador, device, compilação dos shaders)
custa 0,317 s contra 1,26–1,46 s do CUDA/cuDNN e do Keras montando o grafo —
**4 a 4,6× mais rápido**, e este número, sim, é arquitetura. No **primeiro
passo** a distância é maior ainda porque o TensorFlow paga o traçado do
`tf.function` (AutoGraph + construção do grafo de gradiente) nessa mesma
medição — 650× mais lento que o `rtensor`, mas contra o PyTorch em modo eager,
sem traçado, a distância cai para 104×, que é o número mais limpo de
"despachar um passo já compilado" que este repositório tem.

Metodologia completa, incluindo por que a primeira execução de cada processo
mede 2 a 4× mais alto que as seguintes (cache de driver e de disco ainda
frios), em [RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md).

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
| cuBLAS, com TF32 (via TensorFlow) | 11.946 |

*Ressalva:* o número do cuBLAS acima foi medido chamando `tf.matmul` num laço
— a mesma metodologia que a seção de despacho, mais abaixo, mede em **~30% de
overhead de Python** nos passos de treino. A mesma multiplicação 2048³ com
TF32, via PyTorch (despacho quase nulo, já medido): **15.357–18.305
GFLOP/s** — 28% a 53% acima do número citado. Em 8192³, onde o overhead fixo
pesa menos por chamada, os dois se aproximam: TensorFlow sobe a 14.002,
PyTorch a 17.526 — a distância cai de ~30% para ~20%, mas não fecha, o que
sugere um resto que não é só despacho (talvez heurística de algoritmo do
cuBLAS diferente entre as duas bibliotecas). **Toda comparação "distância do
cuBLAS" neste documento usa o número mais baixo (11.946) como referência — a
distância real é maior, não menor.** Metodologia completa em
[RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md).

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

### Split-K: o caso patológico

Com `M` e `N` pequenos, a saída inteira cabe em poucos blocos: um GEMM
`64×64×8192` ocupa **um** workgroup, 256 threads, numa placa de 36 SMs. O
trabalho existe — são 512 ladrilhos de acumulação — mas não há paralelismo para
distribuí-lo.

Não é caso raro: `∂L/∂W = Xᵀδ` tem `K` igual ao tamanho do lote, e `M`, `N`
iguais às dimensões da camada. Lote grande com camada estreita cai exatamente
aqui.

Particionando `K` em fatias que somam num plano próprio, seguidas de uma
redução:

| Forma | Inteiro | Split-K | Ganho |
|---|---:|---:|---:|
| 64×64×8192 | 63 GFLOP/s | **935** | **14,8×** |
| 64×64×16384 | 64 | **1.364** | **21,4×** |
| 32×32×8192 | 21 | **294** | **14,3×** |
| 128×128×4096 | 334 | **1.867** | **5,6×** |

O número de fatias vem da medição: varrendo 4, 16, 64 e 128 em quatro execuções,
**64 venceu nas quatro** (12,6× a 14,5×), com 128 oscilando entre 6,1× e 13,1×.
`Gpu::fatias_sugeridas` mira `blocos × fatias ≈ 64`, que reproduz o ótimo medido
nas quatro formas testadas. No plano de treino (`GpuMlp`) a partição é acionada
por forma: `dW = entradaᵀ·δ` com lote grande e camada estreita, e o forward da
camada de saída, cuja largura pode ser menor que um ladrilho — medido no perfil
do passo em lote 2048, os dois GEMMs patológicos caíram de ~0,66 ms para
~0,18 ms de tempo de dispositivo.

Cada fatia escreve num plano separado — não há escrita concorrente no mesmo
endereço, e portanto nenhuma necessidade de atômicos, que em WGSL não existem
para `f32`.

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

Ainda 2,7× atrás do cuBLAS — na verdade mais, já que o número de referência
subestima o teto real (ver a ressalva na tabela do GEMM, acima). Mas com o
TF32 desligado o TensorFlow cai só 12% — comparação interna ao TensorFlow,
que não sofre do mesmo viés —, o que localiza a maior parte da diferença em
pipelining, tiling multinível e swizzling — todos ao alcance do WGSL — e não
nos tensor cores.

### Tensor cores (matriz cooperativa)

A técnica que restava da lista — um ladrilho inteiro por instrução — estava
registrada como **não funcional**, e o registro estava errado. A sonda original
fixava `8×8 f32`, configuração que esta placa **não anuncia**:
`cooperative_matrix_properties()` só lista combinações com `f16` em A e B.
Configuração fora da lista é comportamento indefinido — e os zeros vinham daí,
não de um bug de driver. Nas configurações anunciadas (`16×16`, operandos
`f16`), a cadeia inteira funciona: oito variantes com erro máximo **zero**
(`examples/probe_coop_f16.rs`), de buffer de armazenamento e de memória de
workgroup, com workgroup de 32 e de 256 threads.

O kernel GEMM cooperativo (`examples/bench_coop.rs`) — ladrilho de saída 64×64
por workgroup de 256 threads (8 subgrupos), dois ladrilhos de 16×16 por
subgrupo, `K` em passos de 16 com estagiagem em memória de workgroup, `f16` em
A e B com acumulador `f32`:

| `N` | cooperativo (1º corte) | escalar | ganho |
|---|---:|---:|---:|
| 2048³ | 4.369–4.466 | 4.386 | par |
| 4096³ | 5.756–5.988 | 3.804–4.243 | **+37% a +50%** |

O ganho vai além da banda: a precisão mista sozinha rendera +1 a 7% no kernel
escalar, e aqui são ~40% — os tensor cores somam aritmética de verdade. E o
erro é o esperado da conversão `f16` (~1e-5 relativo), sem acumulação visível.

**O primeiro corte não tinha buffer duplo nem rasterização L2** — as duas
otimizações que o kernel escalar já usa. Implementadas (mesmo desenho:
ladrilhos `sa`/`sb` alternados, despacho linear com `bloco()` idêntico ao de
`gemm::MM_FAST`), medidas no mesmo processo contra o corte anterior — que
evita a deriva de clock documentada em RESULTADOS-NEGATIVOS.md — em 6
execuções:

| `N` | ganho sobre o 1º corte | GFLOP/s (novo) | distância do cuBLAS |
|---|---:|---:|---:|
| 2048³ | +8,4% a +23,0% (média ~13%) | 4.386–4.809 | — |
| 4096³ | +17,6% a +32,5% (média ~27%) | **7.087–7.204** | **1,67×** (era ~2×) |

O erro numérico não muda (mesma conversão `f16`, mesma acumulação `f32`) —
só o padrão de acesso à memória. `+63%` sobre o kernel escalar em produção
(4.386 GFLOP/s), contra os +37–50% do primeiro corte. A distância do cuBLAS
citada usa o número de referência medido via TensorFlow, que a ressalva da
seção "GEMM" acima mostra estar subestimado — 1,67× é piso, não teto.

**O que ainda sobra na mesa:** o teto puro de `coopMultiplyAdd` — a mesma
aritmética sem nenhuma leitura de memória nova — mede **36 a 45,5 TFLOP/s**
(`examples/ocupacao_coop.rs`). O GEMM real usa só **~18–20%** disso. Três
hipóteses testadas e refutadas (memória de workgroup, `coopLoadT` repetido,
banda de memória global sozinha) e uma quarta inconclusiva (antecipação mais
profunda, 3 ladrilhos em vez de 2) não fecharam a conta — a causa exata
continua em aberto, provavelmente exigindo um perfilador de ocupação que
este projeto não usa. Detalhes e as quatro sondas em
[RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md).

Dois preços, e agora são os únicos obstáculos:

1. **`unsafe`.** Habilitar a feature exige `ExperimentalFeatures::enabled()`,
   que é `unsafe fn` — custaria a propriedade "zero `unsafe`" do projeto.
2. **Operandos em `f16`.** Não há configuração com operandos `f32` anunciada
   nesta placa.

No caminho houve um bug do naga 30.0.1 — `coopLoad`/`coopStore` com ponteiro
dinâmico para buffer de armazenamento derrubam o compilador SPIR-V
(`Expression is not cached!`); para memória de workgroup o mesmo padrão
funciona. O contorno é a saída por slots de workgroup com cópias planas, custo
de ~6%. Detalhes, incluindo um artefato de medição cometido e registrado no
percurso, em [RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md).

**A alternativa sem nenhum dos dois preços foi testada, e não pagou.**
`Features::SUBGROUP` do wgpu é estável — não exige `unsafe` nem `f16` — e uma
sonda isolada (`examples/sonda_subgrupo.rs`) mediu um encaixe real no laço do
GEMM: +3% a +9% trocando releituras de `A` na memória compartilhada por
`subgroupShuffle`. Implementado no kernel completo e medido no mesmo processo
contra a versão em produção, o ganho virou ruído: −3,4% a +2,5%, média −0,7%
em 8 execuções. Revertido; números completos em
[RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md).

### Reproduzindo

```bash
cargo run -p rtensor --release --features gpu --example suite -- batch   # rtensor
cargo run -p rtensor --release --features gpu --example bench_coop       # GEMM nos tensor cores
cd benchmarks/rivais && cargo run --release --bin burn_bench -- 512      # burn
cd benchmarks/rivais && cargo run --release --bin candle_bench -- 512    # candle
```

Os scripts de TensorFlow e PyTorch, com as instruções de ambiente, estão em
[`benchmarks/python/`](benchmarks/python/).

Versões medidas: burn 0.21, candle 0.11, TensorFlow 2.21.0, PyTorch 2.14.0+cu132.

## Simulação quântica

Simular qubits **não acelera** conta clássica — custa `2ⁿ` amplitudes complexas,
e é sempre mais caro que a conta equivalente. O que vale é o inverso: o laço
interno de um simulador é contração tensorial, que é o que o `rtensor` já faz.

Aplicar uma porta lê duas amplitudes, multiplica por uma matriz 2×2 e escreve
duas de volta: **0,875 flop por byte**. É o regime **oposto** ao do GEMM.

| Qubits | Precisão | ms/porta | GB/s | µJ/porta |
|---:|---:|---:|---:|---:|
| 24 | f32 | 1,473 | 182,2 | 70.255 |
| 24 | **f16** | **0,693** | 193,7 | **35.432** |
| 26 | f32 | 5,355 | 200,5 | 261.535 |
| 26 | **f16** | **2,735** | 196,3 | **138.068** |
| 27 | f32 | 10,415 | 206,2 | 518.740 |
| 27 | **f16** | **5,341** | 201,0 | **274.384** |

~200 GB/s de **256 disponíveis** — carga governada por banda, como previsto. E
é aqui que a **precisão mista finalmente paga**: guardar as amplitudes em
`complex32` dá **~2× de velocidade e ~2× menos energia**, porque metade dos
bytes numa carga limitada por bytes é metade do trabalho.

A mesma técnica foi medida como **inútil no GEMM** — +1 a 7% de velocidade e
energia *pior*. A diferença é o regime: lá o gargalo não era banda, aqui é. É o
melhor exemplo no repositório de por que o gargalo precisa ser medido antes de
escolher a otimização.

O custo é precisão: `f16` guarda ~3 dígitos decimais, e o erro medido fica em
~1,6·10⁻⁴ a 5,8·10⁻⁴ — **sem crescer com o número de portas**, com a norma do
estado dentro de 0,2% de 1.

O teto de qubits não é de VRAM: o WebGPU limita um binding a 2 GB **menos 4
bytes**. Isso dá 27 qubits em `f32` e 28 em `f16` — meia precisão compra **um**
qubit, não dois, porque o teto também é potência de dois.

Portas de **um e dois qubits** — Hadamard, Pauli-X/Z, rotação Y, fase, CNOT, CZ
e SWAP — o suficiente para circuitos universais. Conferidas contra referência de
CPU, incluindo o estado de Bell, que só passa se o emaranhamento estiver certo.

### Aritmética de graça no tempo, não na energia

A porta de dois qubits move **os mesmos bytes** que a de um qubit — cada
amplitude é lida e reescrita uma vez — mas faz **4× a aritmética**: 16
multiplicações complexas por grupo contra 4 por par.

| Qubits | Porta | Precisão | ms | µJ |
|---:|---:|---:|---:|---:|
| 27 | 1q | f32 | 10,266 | 519.581 |
| 27 | 2q | f32 | **10,186** | **557.237** |
| 27 | 1q | f16 | 5,202 | 276.686 |
| 27 | 2q | f16 | **5,178** | **293.336** |
| 28 | 1q | f16 | 10,423 | 550.032 |
| 28 | 2q | f16 | **10,357** | **585.391** |

**Mesmo tempo, 6 a 7% mais energia.** Numa carga limitada por banda, a
aritmética extra não custa relógio — mas custa watt. Pelo cronômetro a porta de
dois qubits é gratuita; pelo wattímetro, não é.

É a terceira vez neste repositório que a medição de energia diz algo que a de
tempo não diria.

### Fusão de portas

Cada porta é **uma passada completa** pelo vetor de estado, e a simulação é
limitada por banda. Fundir `k` portas numa só corta `k` passadas — é a
otimização de maior alavanca neste regime, e não muda um byte do kernel.

Portas de um qubit no mesmo qubit se multiplicam entre si. E quando aparece uma
porta de dois qubits, as pendentes nos seus dois qubits são **absorvidas**
dentro dela por produto de Kronecker: uma camada de rotações seguida de uma
camada de CNOTs custa, depois da fusão, só os CNOTs.

Circuito em 4 camadas — rotações em todos os qubits, depois CNOTs em cadeia:

| Qubits | Portas | Fundidas | Direto | Fundido | Ganho | Energia |
|---:|---:|---:|---:|---:|---:|---:|
| 22 | 260 | 84 | 68,1 ms | 20,8 ms | **3,28×** | 2,87× menos |
| 24 | 284 | 92 | 375,3 ms | 122,3 ms | **3,07×** | 3,02× menos |
| 26 | 308 | 100 | 1.567,9 ms | 510,8 ms | **3,07×** | 2,98× menos |

O ganho de energia acompanha o de tempo quase exatamente — o esperado quando se
elimina trabalho em vez de acelerá-lo. É o contraste com a porta de dois qubits,
onde 4× a aritmética saiu de graça no relógio e custou 7% no wattímetro.

A fusão é **reescrita algébrica exata**: o teste confere que o estado fundido é
idêntico ao direto, com erro de 1,2·10⁻⁷.

### Fundir em unitárias maiores

Um kernel **genérico** de até 4 qubits reduz 308 portas a 40 e deixa o circuito
22% **mais lento** — ele estagia as amplitudes em memória de workgroup, usa 64
threads em vez de 256 e percorre a matriz com laços de limite variável.

Kernels **especializados**, com as amplitudes em registradores e os elementos
da matriz endereçados por índices literais, invertem o resultado:

Foram gerados kernels especializados para 3, 4, 5 e 6 qubits. Circuito de 4
camadas em 26 qubits:

| Máx. qubits | Portas | ms | Ganho | Energia | Amplitudes em registrador |
|---:|---:|---:|---:|---:|---:|
| — | 308 | 1.540 | (base) | (base) | — |
| 2 | 112 | 561,8 | 2,74× | 2,68× menos | 8 floats |
| 3 | 58 | 289,0 | 5,33× | 4,62× menos | 16 floats |
| 4 | 40 | 201,8 | 7,63× | 5,65× menos | 32 floats |
| **5** | **30** | **154,8** | **9,95×** | **6,46× menos** | **64 floats** |
| 6 | 24 | 303,6 | 5,07× | 4,86× menos | 128 floats |

**O pico é em 5 qubits, e o limite não é a placa — é pressão de registradores.**
Em 6 qubits são 64 amplitudes complexas, 128 floats vivos por thread, e a
ocupação desaba: 20% menos portas e o dobro do tempo. A sonda
`rtensor/examples/ocupacao.rs` tinha medido que a vazão se sustenta até ~96
floats e cai depois; o precipício está entre 64 e 128, e aqui ele aparece de
novo num domínio completamente diferente.

Até o pico, o ganho acompanha a redução de portas quase exatamente — 308 contra
30 são 10,3× menos passadas para 9,95× menos tempo. É o comportamento de uma
carga limitada por banda.

O ganho de energia fica sempre **abaixo** do de tempo, e a distância cresce com
`N`. Causa conhecida: uma porta de `N` qubits faz `2ᴺ⁻¹×` mais aritmética por
amplitude que uma de 1 qubit. Numa carga limitada por banda isso não custa
relógio — e custa watt, como já se via na porta de dois qubits, onde 4× a
aritmética saiu de graça no tempo e cobrou 7% na energia.

### Contra o Qiskit Aer e o cuQuantum

Mesma carga — rotações `RY` com ângulos distintos, em rodízio pelos qubits.
Milissegundos por porta, precisão simples nos três:

| Qubits | rqubit `f32` | cuStateVec | Aer CPU (com fusão) |
|---:|---:|---:|---:|
| 22 | 0,269 | **0,133** | 1,300 |
| 24 | 1,326 | **1,317** | 3,733 |
| 26 | 5,412 | **5,309** | 13,364 |
| 27 | 10,266 | **9,796** | — |

**Empate técnico com o cuStateVec** — 1 a 5% de diferença em 24, 26 e 27
qubits, e ambos a ~200 GB/s. Não é coincidência: os dois saturam a banda de
memória, e aí não há o que otimizar além de mover os bytes. Uma biblioteca
proprietária da NVIDIA e um kernel WGSL portátil chegam ao mesmo lugar.

Em 22 qubits o cuStateVec ganha 2×: o estado é pequeno e o nosso custo por
dispatch pesa mais.

E onde nós ganhamos:

| | ms/porta em 27 qubits | J/porta em 26 qubits |
|---|---:|---:|
| rqubit `f16` | **5,202** | **0,138** |
| rqubit `f32` | 10,266 | 0,263 |
| cuStateVec `f32` | 9,796 | 0,313 |

O cuStateVec **não oferece meia precisão** para vetor de estado. Como a carga é
limitada por banda, metade dos bytes é quase metade do tempo: `rqubit` em `f16`
é **1,9× mais rápido** e **2,3× mais eficiente em energia** que o cuStateVec.

Nenhum dos dois publica joules por porta. O número do cuStateVec acima foi
medido por fora, com `rqubit/examples/medir_externo.rs` — o instrumento mede
qualquer processo, não só o nosso.

### Circuito completo, com fusão nos dois lados

Quatro camadas de rotações e CNOTs em cadeia — o formato de algoritmo
variacional. Milissegundos por circuito, precisão simples:

| Qubits | rqubit direto | **rqubit fundido** | cuStateVec | Aer (fusão on) | Aer (fusão off) |
|---:|---:|---:|---:|---:|---:|
| 20 | 18,97 | 8,22 | **6,30** | 51,87 | 38,11 |
| 22 | 68,10 | **20,79** | 27,25 | 270,40 | 257,89 |
| 24 | 375,25 | **122,32** | 298,21 | 1.194,93 | 1.145,05 |
| 26 | 1.567,93 | **510,83** | 1.296,30 | 4.094,71 | 5.641,44 |

Três leituras:

**Contra o cuStateVec, o kernel deles é ~21% mais rápido** que o nosso neste
circuito — 1.296 contra 1.568 ms sem fusão em 26 qubits. Mas o `cuStateVec` é
API de baixo nível e **não funde portas**: isso é responsabilidade de quem
chama. Com a nossa fusão, 510,8 contra 1.296,3 — **2,5× mais rápido**, apesar do
kernel mais lento.

*Ressalva:* a NVIDIA tem APIs de nível mais alto (`cutensornet`) que fariam essa
otimização. A comparação é do que cada API entrega, não do teto de cada pilha.

**A fusão do Aer quase não ajuda aqui** — 1,0× a 1,38×, e em 20 a 24 qubits
chega a atrapalhar. A nossa dá 3,07×. A diferença é de estratégia: o Aer funde
em unitárias de até 5 qubits, o que na CPU troca passadas de memória por muita
aritmética; nós só absorvemos portas de um qubit dentro das de dois, o que numa
GPU limitada por banda ganha sempre.

**Contra o Aer, 8,0× em 26 qubits** — mas é GPU contra CPU, e isso era
esperado.

### Com a fusão até 5 qubits

Refeita a comparação com o melhor de cada um, em 26 qubits:

| Motor | ms |
|---|---:|
| **rqubit, fusão até 5 qubits** | **154,8** |
| cuStateVec (sem fusão na API) | 1.272 |
| rqubit sem fusão | 1.540 |
| Aer, melhor limite de fusão (5) | 3.806 |
| Aer sem fusão | 6.165 |

**8,2× mais rápido que o cuStateVec e 24,6× que o Aer** — com a ressalva já
registrada de que o cuStateVec é API de baixo nível e a NVIDIA tem camadas
superiores que fundiriam, e de que o Aer é CPU.

Ao Aer foi dada a mesma chance de ajuste: varrendo o limite de fusão dele de 2 a
5, o melhor foi 5, com 1,62× sobre o próprio caso sem fusão.

Duas armadilhas que invalidariam a comparação, e como foram tratadas, estão em
[`benchmarks/quantum/`](benchmarks/quantum/): portas que se cancelam no
transpilador, e a fusão de portas do Aer.

### A QFT completa: um algoritmo real no teto

Portas isoladas medem o kernel; um algoritmo mede a pilha. A Transformada
Quântica de Fourier inteira — H em cada qubit, fases controladas entre todos
os pares, reversão final — tem **O(n²) portas: 420 em 28 qubits**, cada uma
uma passada por 1 GiB de estado. É o circuito canônico de benchmark, e é
onde o produto interno novo prova a física: ⟨ψ|ψ⟩ = 1 depois da QFT, e a
ida-e-volta QFT⁻¹∘QFT devolve sobreposição 1 — 0,999998 em f32; 0,9985 em
f16, o arredondamento acumulado de ~400 portas. Em 8 qubits, a GPU confere
com a CPU a 1,1·10⁻⁸.

| Qubits | rqubit direto | fusão-2 | fusão-6 (f32) | cuStateVec | Melhor |
|---:|---:|---:|---:|---:|---|
| 22 | 56,7 ms | 55,1 | 90,7 | **37,7** | cuStateVec 1,6× |
| 23 | 185,5 | 171,2 | **126,5** | — | rqubit 1,47× |
| 24 | 392,8 | 362,3 | **196,4** | 367,2 | rqubit 2,00× |
| 26 | 1.758,7 | 1.636,1 | **756,2** | 1.716,2 | rqubit 2,27× |
| 26, `f16` | 925,2 | 822,4 | — | 1.716,2 | rqubit 2,09× |
| 27 | 3.745,2 | 3.480,5 | **1.539,7** | 3.685,0 | rqubit 2,39× |
| 28, `f16` | 4.031,6 | **3.763,9** | — | 7.916,4 | rqubit 1,96× |

Três leituras. Em precisão simples, **kernel a kernel é empate técnico** —
3.745 contra 3.685 ms em 27 qubits, com os dois saturando a mesma banda
(~224–228 GB/s medidos). A **fusão até 6 qubits** corta 391 portas a 82 e
dá **2,43× em tempo e 2,44× em energia** (102,0 J contra 249,0 J) — e o
cruzamento foi varrido: em 22 qubits ela perde (90,7 contra 56,7 ms, −37%,
medidos lado a lado), porque o estado de 32 MiB cabe na L2 e as passadas
ficam baratas; **em 23 qubits o estado de 64 MiB já sai da L2 e a fusão
vence (126,5 contra 185,5 ms, 1,47×)**, chegando a 2,00× em 24 e 2,43× em
27. Na energia, a fusão-6 vence em todos os tamanhos medidos — mesmo onde
perde tempo, em 22, ela gasta menos (2,12 J contra 2,93 J). E a **meia
precisão compra um qubit inteiro**: 4,03 s em 28 qubits `f16` contra 7,92 s
do cuStateVec, que não oferece `f16` — a mesma ressalva já registrada para
as portas isoladas.

Energia por circuito, com o método de cada número anotado:

| Qubits | rqubit direto | rqubit fusão-6 | cuStateVec (por fora) |
|---:|---:|---:|---:|
| 26, f32 | 120,9 J | **47,8 J** | ~65,5 J |
| 26, `f16` | **36,6 J** | — | ~65,5 J |
| 27, f32 | 249,0 J | **102,0 J** | — |
| 28, `f16` | **246,4 J** | — | ~288,3 J |

O `f16` faz **3,3× menos energia** que o f32 no mesmo circuito — mais que os
2× dos bytes, porque a potência média também cai. E a fusão de 2 qubits em
`f16` **custa energia**: 822 ms contra 925, mas 48,9 J contra 36,6 J — a
Hadamard absorvida faz aritmética extra por byte, e num regime limitado por
banda isso não compra tempo, cobra watt.

Metodologia, testes e o comparativo completo, com as ameaças à validade,
estão em [RELATORIO-QFT.tex](RELATORIO-QFT.tex); o resultado negativo dos
22 qubits, em [RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md).

### Um caso físico: o transporte de um magnon em 27 sítios

Portas e QFTs medem a pilha; um modelo físico medido contra a sua solução
exata mede a ciência. A cadeia XX de 27 sítios — o sistema-padrão de
transporte quântico em íons presos e qubits supercondutores — recebe um
magnon no centro e evolui por 100 passos de Trotter (**2.600 portas de 2
qubits**, Δt = 0,1, t final = 10). A rotação de Givens exp(−iJδ(XX+YY)/2)
preserva o número de magnons, e no setor de 1 magnon a evolução exata tem
**soma fechada sobre os 27 modos normais** — a resposta quântica exata para
o estado completo de 2²⁷ amplitudes, calculada em O(N²).

A metodologia separa dois erros que costumam ser confundidos, no mesmo
circuito em 12 qubits:

| Erro | Fonte | Medido |
|---|---|---:|
| Trotter (CPU contra exato) | do circuito | 1,66·10⁻² |
| Implementação (GPU f32 contra CPU) | do kernel | **0** |
| Precisão (GPU f16 contra CPU) | do hardware | 1,45·10⁻³ |

**O erro do f16 é 11× menor que o do Trotter**: para este observável, a
meia precisão é de graça — o circuito é o gargalo, não o hardware.

A trajetória completa em 27 qubits confirma a física: **energia conservada**
(E = 0,0000 em todo t), espalhamento balístico (σ de 1,41 a 10,2 sítios) que
dobra em t ≈ 6,5 quando a frente a v = 2J atinge as bordas — a reflexão —,
echo de Loschmidt oscilante, e vazamento de setor de 1,8·10⁻³ no `f16`,
quantificado. O perfil P(m, t = 10) confere com a solução exata dentro do
erro de Trotter.

Contra o cuStateVec, a mesma trajetória:

| Motor | Precisão | Tempo (2.600 portas) | ms/porta |
|---|---|---:|---:|
| rqubit | f32 | 25,11 s | 9,66 |
| rqubit | `f16` | **12,91 s** | 4,97 |
| cuStateVec | f32 | 24,28 s | 9,34 |

Empate técnico em f32; o `f16` — que o cuStateVec não oferece — faz a mesma
física em **1,88×** menos tempo. Metodologia e tabelas completas em
[RELATORIO-QFT.tex](RELATORIO-QFT.tex).

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

### Entre motores de deep learning — e uma armadilha no número mais alto

A comparação acima é entre fabricantes de GPU, com o mesmo código. Esta é
entre motores — `rtensor`, burn, PyTorch e TensorFlow —, mesma MLP
512→1024→1024→10, lote 512. `rtensor` usa `examples/energia.rs` (mede o laço
interno, sem o custo de processo). Os outros três são medidos por fora com
`rqubit::examples::medir_externo`, por diferença entre duas contagens de
passos (1.000 e 6.000) — a mesma técnica de `benchmarks/quantum/README.md`,
necessária porque o import de Python (~30% do tempo, ver a seção de
despacho acima) senão entraria na conta:

| Motor | ms/passo | GFLOP/s | J/passo (acima da ociosidade) | W acima da ociosidade | GFLOP/J |
|---|---:|---:|---:|---:|---:|
| burn-wgpu | 1,146 | 4.243,6 | 0,0430 | **37,6** | **112,99** |
| rtensor GPU | 1,459 | 3.333,5 | 0,0855 | 58,6 | 56,89 |
| torch-cuda | 1,483 | 3.279,3 | 0,0793 | 53,5 | 61,29 |
| tensorflow | 1,868 | 2.603,1 | 0,0163 | **8,7** | **298,7** |

**O número do TensorFlow é uma armadilha.** Isolado, parece o motor mais
eficiente por joule — 298,7 GFLOP/J, 2,6× o burn. A coluna de potência
desmente: a GPU do TensorFlow roda a **8,7 W acima da ociosidade**, contra
37–59 W dos outros três. É a mesma causa da seção de despacho: ~30% do tempo
do TF é overhead de `tf.function`, e nesse tempo a GPU fica parada. Energia é
potência × tempo — GPU quase ociosa por mais tempo dá poucos joules totais,
mas não porque o trabalho seja eficiente, e sim porque há pouco trabalho por
segundo de relógio. Um "GFLOP/J" alto aqui não é "faz mais por joule", é
"gasta pouco porque faz pouco".

**Sem essa distorção, a ordem é burn > torch > rtensor** — o burn quase 2×
mais eficiente que o `rtensor`, o mesmo "resultado mais desconfortável" já
registrado acima, agora também em energia, não só em tempo. Scripts:
`benchmarks/python/energia_tf.py`, `energia_torch.py`,
`benchmarks/rivais/src/burn_bench.rs` (com contagem de passos configurável).

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
cargo test --workspace --features rtensor/gpu   # 52 testes em 9 suítes
cargo test -p rtensor                           # núcleo, sem GPU
```

Deep learning e o GEMM:

```bash
cargo run -p rtensor --release --features gpu --example suite -- batch  # varredura de lote
cargo run -p rtensor --release --features gpu --example bench_gemm      # GEMM, com varreduras
cargo run -p rtensor --release --features gpu --example perfil          # perfilamento por kernel
cargo run -p rtensor --release --features gpu --example ocupacao         # pressão de registradores
cargo run -p rtensor --release --features gpu --example probe_coop_f16   # matriz cooperativa nas configurações anunciadas
```

Energia:

```bash
cargo run -p rtensor --release --features gpu --example eficiencia   # energia por ocupação
cargo run -p rtensor --release --features gpu --example energia      # por motor e por fabricante
sudo ./target/release/examples/frequencia                            # eficiência contra clock
```

Simulação quântica:

```bash
cargo run -p rqubit --release --example bench_porta   # banda e energia por porta
cargo run -p rqubit --release --example bench_fusao   # fusão, varrendo o limite de qubits
cargo run -p rqubit --release --example bench_qft     # QFT completa no teto, contra o cuStateVec
cargo run -p rqubit --release --example cadeia_magnon # transporte de um magnon, contra a solução exata
cargo run -p rqubit --release --example medir_externo -- <comando>   # energia de outro processo
```

## O que não funcionou

Toda otimização tentada, incluindo as revertidas, e os erros de medição que
custaram mais tempo que os erros de código, estão em
[RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md). Publicar só os acertos
falsificaria a taxa de sucesso real do trabalho.

Resumo das tentativas, com o estado atual:

| Tentativa | Resultado medido | Estado |
|---|---|---|
| **Rendeu** | | |
| Buffer duplo no GEMM | +22% em 2048³ | em produção |
| Rasterização com consciência de L2 | +5 a 6% em 4096³ | em produção |
| Split-K | **+12,6× a 21,4×** em formas K-dominantes | em produção, e por forma no treino |
| Strassen de um nível | +11 a 13% acima de 4096³; −52% abaixo | opcional, explícito |
| Cache de bind groups e uniformes | +8% | em produção |
| Fusão de portas, 1–2 qubits | 2,7× a 3,1× em tempo e energia | padrão no `rqubit` |
| Fusão, kernels especializados 3–5 qubits | 5,3× a 9,95× | em produção; o pico é 5 qubits |
| Precisão mista `f16` (simulador) | ~2× em tempo e energia | padrão no `rqubit` |
| **Não rendeu** | | |
| Fusão de kernels no epílogo do GEMM | −20% na rede profunda | registrada |
| Fundir dispatches num só compute pass | −23% na rede profunda | registrada |
| Bloco 8×8 por thread | −5% | [revertido](experimentos/gemm-8x8/) |
| Bloco 8×4 por thread | nulo | [revertido](experimentos/gemm-8x4/) |
| Leituras globais `vec4` | nulo | [revertido](experimentos/gemm-vec4/) |
| GEMM especializado para formas alinhadas | nulo | [revertido](experimentos/gemm-alinhado/) |
| Padding contra conflito de bancos | nulo | mantido, sem crédito |
| Fusão genérica em 3 e 4 qubits | −9% e −22% apesar de menos portas | substituída pelas especializadas |
| Precisão mista `f16` (GEMM) | +1 a 7% de velocidade, −5 a 11% de energia | mantida por capacidade, não por velocidade |
| GEMM com `A` via `subgroupShuffle` | sonda isolada +3 a 9% (real); kernel completo −3,4% a +2,5%, média −0,7% | [revertido](experimentos/gemm-subgrupo/) |
| **Retificado** | | |
| Matriz cooperativa (tensor cores) | "não funcional" era diagnóstico errado: **funciona**; com buffer duplo e rasterização L2, +63% sobre o escalar e 1,67× do cuBLAS | decisão pendente: exige `unsafe` e operandos `f16` |

São **onze** otimizações que não pagaram, cada uma com números e causa
identificada, ao lado das que pagaram. O código revertido fica preservado em
[`experimentos/`](experimentos/), para que ninguém refaça a tentativa e para
que quem discordar de uma rejeição possa medir.

E um registro foi **retificado**: a matriz cooperativa, arquivada como "não
funcional", funcionava — a sonda usava uma configuração que a placa não
anuncia, e configuração fora da lista é comportamento indefinido. A retificação
tem números próprios (seção "Tensor cores" acima) e é a melhor defesa deste
repositório do método: o registro é verificável, inclusive quando o erro é do
registro. Também entrou aí um achado colateral: a referência de `f16` do
`rqubit` devolvia metade dos valores nos subnormais — corrigida com teste
fixado.

O roteiro do GEMM, com o que a literatura clássica indicou e o que sobrou por
fazer, está em [ROTEIRO-GEMM.md](ROTEIRO-GEMM.md).

## Licença

MIT.
