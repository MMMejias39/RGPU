# O que não funcionou

Toda otimização tentada neste repositório, incluindo as que falharam e as que
foram revertidas. O registro existe porque o que **não** funcionou economiza
mais tempo de quem vem depois do que o que funcionou — e porque publicar só os
acertos falsifica a taxa de sucesso real do trabalho.

## Política do repositório

**Nada do que foi tentado sai daqui.** Otimização revertida tem o código
preservado em [`experimentos/`](experimentos/) com as medições no cabeçalho, e
o resultado registrado neste arquivo com números. Erro de metodologia de medição
também entra — neste projeto eles custaram mais tempo que os erros de código.

Duas regras que decorrem disso:

- **Não se reporta ganho que não foi medido.** Quando o efeito fica dentro do
  ruído, o texto diz isso, e a otimização não recebe crédito.
- **Quem discordar de uma rejeição pode medir.** É para isso que o código
  rejeitado continua publicado, e não apenas descrito.

Se você for contribuir e sua ideia não funcionar, mande mesmo assim: o resultado
negativo tem lugar neste repositório.

## Otimizações

| Tentativa | Resultado | Onde |
|---|---|---|
| Fusão de kernels no epílogo do GEMM | **−20%** na rede de 13 camadas; nulo em redes rasas | em produção |
| Buffer duplo no GEMM | **+22%** em 2048³ | em produção |
| Cache de bind groups e uniformes | +8% — muito abaixo do previsto | em produção |
| Fundir dispatches num só compute pass | **−23%** na rede profunda | em produção |
| Redução de coluna em dois estágios | ganho não isolado | em produção |
| Padding contra conflito de bancos | **nulo** | mantido |
| Bloco 8×8 por thread | **−5%** | [revertido](experimentos/gemm-8x8/) |
| Leituras globais `vec4` | **nulo** | [revertido](experimentos/gemm-vec4/) |
| GEMM especializado para formas alinhadas | **nulo** | [revertido](experimentos/gemm-alinhado/) |
| Rasterização com consciência de L2 | **+5 a 6%** em 4096³ | em produção |
| Strassen de um nível | **+11 a 13%** acima de 4096³, **−52%** em 1024³ | opcional |
| Split-K | **+12,6× a 21,4×** em formas K-dominantes | em produção, e por forma no plano de treino |
| Fusão de portas até 2 qubits (`rqubit`) | **2,7× a 3,1×** em tempo e energia | padrão |
| Fusão em 3 qubits, kernel **genérico** | **−9%** apesar de 2× menos portas | substituída |
| Fusão em 3 qubits, kernel **especializado** | **5,32×** em tempo, 4,64× em energia | em produção |
| Fusão em 4 qubits, kernel **genérico** | **−22%** apesar de 3× menos portas | substituída |
| Fusão em 4 qubits, kernel **especializado** | **7,65×** em tempo, 5,64× em energia | em produção |
| Precisão mista `f16`/`f32` | **+1 a 7%** de velocidade, **−5 a 11%** de energia | mantida por outra razão |
| Bloco `8×4` por thread | **nulo** | [revertido](experimentos/gemm-8x4/) |
| Fusão até 6 qubits em 22 qubits (QFT) | **−34%** (90,7 ms contra 67,6) | cruzamento medido entre 22 e 24 |
| Fusão de 2 qubits em `f16` (QFT) | **+11% de tempo**, mas **+34% de energia** | ganho de tempo sem ganho de energia |
| Matriz cooperativa (tensor cores) | o "não funcional" era diagnóstico errado: **funciona** nas configurações anunciadas; exige `unsafe` e operandos `f16` | sondas mantidas |
| Buffer duplo + rasterização L2 no cooperativo | **+13% (2048³) e +27% (4096³)** sobre o primeiro corte; 1,67× do cuBLAS | sonda mantida, mesma decisão pendente |
| GEMM com `A` via `subgroupShuffle` | sonda isolada: **+3 a 9%**, real; no kernel completo: **-3,4% a +2,5%** (média -0,7%) | [revertido](experimentos/gemm-subgrupo/) |

### A QFT em 22 qubits: a fusão até 6 perde, e a de 2 custa watt

A Transformada Quântica de Fourier completa (`examples/bench_qft.rs`) foi
medida de 22 a 28 qubits contra o `cuStateVec`. Dois resultados negativos
no menor tamanho:

**Fusão até 6 qubits em 22 qubits — −34%.** Os kernels especializados de 3
a 6 qubits cortam as 264 portas da QFT a 56, e mesmo assim levam 90,7 ms
contra 67,6 do caminho direto — 1,62 ms por porta original contra 0,26. A
causa é o regime: com 32 MiB, o estado inteiro cabe na L2 da placa, cada
passada custa pouco, e o ganho de cortar passadas não paga o custo maior
por dispatch. Em 24 qubits o mesmo desenho já vence (199,8 contra 392,2 ms)
e em 27 dá 2,43×. Onde está a fronteira exata não foi varrido.

**Fusão de 2 qubits em `f16` — tempo melhor, energia pior.** Em 26 qubits,
a fusão absorve as Hadamards nos CPs e corta o circuito de 364 a 338
portas: 822 ms contra 925. Mas gasta 48,9 J contra 36,6 J — **+34% de
energia para −11% de tempo**. A porta fundida faz mais aritmética por byte,
e numa carga limitada por banda isso é de graça no relógio e caro no
wattímetro — a quarta vez neste repositório que as duas medições divergem.

### Bloco 8×8 por thread — rejeitado

Ladrilho 128×128, 16 acumuladores `vec4` por thread, quatro FMAs por leitura de
memória compartilhada: a mesma intensidade aritmética do cuBLAS. Passou nos 42
casos de conferência contra a CPU e mediu pior.

```
4096³    4×4: 4.030 GFLOP/s    8×8: 3.846 GFLOP/s    −4,6%
2048³    4×4: 4.128 GFLOP/s    8×8: 3.976 GFLOP/s    −3,7%
```

A razão está em `crates/rtensor/examples/ocupacao.rs`: aritmética pura sustenta
15 a 18 TFLOP/s nesta placa, e o GEMM anda a 4. Estando 4× longe do limite das
ULAs, o gargalo não é intensidade aritmética — dobrá-la só custa ocupação.

O kernel está preservado em [`experimentos/gemm-8x8/`](experimentos/gemm-8x8/)
para que a tentativa não precise ser refeita, e para que quem discordar da
conclusão possa medir por conta própria.

### Matriz cooperativa — o diagnóstico "não funciona" estava errado

Era a última técnica da lista e a que a análise apontava: uma instrução cobrindo
um ladrilho inteiro reduziria de uma vez as leituras compartilhadas **e** as
FMAs, que é o que o diagnóstico revisado indica ser necessário.

**Primeira medição, retificada.** `crates/rtensor/examples/probe_coop.rs`
testou, em wgpu 30.0.1, naga 30.0.1, driver NVIDIA 595.91.07, e devolveu zeros.
Foi registrada como não funcional. O registro estava errado em três pontos:

1. **A configuração não existe nesta placa.** A sonda fixou `8×8 f32` sem
   consultar `Adapter::cooperative_matrix_properties()`. O RTX 4070 Laptop
   anuncia só combinações com `f16` em A e B — `16×16×16`, `16×8×16` e
   `16×8×8`, com acumulador `f16` ou `f32`. Configuração fora da lista é
   comportamento indefinido; o driver respondeu com zeros.
2. **Layout trocado.** A sonda usava `coopLoad`, que lê **column-major**
   segundo a especificação que o wgpu hoje publica
   (`docs/api-specs/cooperative_matrix.md` no repositório deles), sobre dados
   em row-major.
3. **A "marca de vida" não provava nada.** O doc afirmava que o `coopStore`
   escrevia — verificado com uma marca em `c[63]` — mas o código comitado
   inicializava tudo com zero: `c[63] = 0` não distinguia "escreveu zero" de
   "não escreveu".

**Segunda medição, que retifica a primeira**
(`crates/rtensor/examples/probe_coop_f16.rs`). Nas configurações anunciadas —
`16×16×16` com AB `f16` e acumulador `f32` (misto) ou `f16` — oito variantes do
mesmo produto 16×16 conferido contra a referência de CPU, **todas com erro
máximo zero**:

| Ponteiros | Leitura | Workgroup | C `f32` | C `f16` |
|---|---|---|---|---|
| buffer de armazenamento | `coopLoadT` | 32 | correto | correto |
| buffer de armazenamento | `coopLoadT` | 256 | correto | correto |
| memória de workgroup | `coopLoadT` | 32 | correto | correto |
| memória de workgroup | `coopLoadT` | 256 | correto | correto |

O caminho que um GEMM real usaria — estagiagem em memória de workgroup, com
workgroup de 32 (um subgrupo) ou 256 (oito) threads — funciona. O naga emite
as matrizes com escopo `Subgroup`, e o subgrupo da NVIDIA é 32. A escada de
diagnóstico da sonda (produto + marca, produto puro, só marca, zero) separa
falta de carga, falta de soma e falta de escrita; a cadeia inteira passou.

**O que resta de preço — dois, e agora são os únicos:**

1. **`unsafe`.** Habilitar a feature exige `ExperimentalFeatures::enabled()`,
   que é `unsafe fn` — o wgpu declara que estas APIs podem conter bugs que
   levam a comportamento indefinido a partir de código aparentemente seguro.
   A propriedade "zero `unsafe`" do projeto continua em jogo.
2. **Operandos em `f16`.** Todas as configurações anunciadas exigem `f16` em A
   e B — não há caminho com operandos `f32` nesta placa. É o regime da
   precisão mista, medido acima como sem ganho no GEMM convencional; com
   tensor cores a taxa de aritmética muda de base, e a conta precisa ser
   refeita.

**Medido: o kernel GEMM cooperativo** (`crates/rtensor/examples/bench_coop.rs`)
— ladrilho de saída 64×64 por workgroup de 256 threads (8 subgrupos), dois
ladrilhos cooperativos de 16×16 por subgrupo, `K` em passos de 16 com
estagiagem em memória de workgroup, acumuladores residentes entre os passos.
Mediana de 8 execuções, conferido contra referência em f64 sobre os operandos
já em `f16` (bloco 32×32 amostral, 0 posições erradas):

| `N` | cooperativo | escalar | ganho |
|---|---:|---:|---:|
| 2048³ | 4.369–4.466 | 4.386 | **par** |
| 4096³ | 5.756–5.988 | 3.804–4.243 | **+37% a +50%** |

Dois achados no caminho, ambos com registro:

- **Bug do naga 30.0.1.** `coopLoad`/`coopStore` com ponteiro dinâmico para
  buffer de **armazenamento** derrubam o compilador SPIR-V —
  `internal error: Expression is not cached!` em `back/spv/index.rs`. Ponteiro
  dinâmico para memória de workgroup compila e executa. O contorno: os
  ladrilhos de C saem e entram por slots de workgroup em base fixa por
  subgrupo, com cópias planas (que não tocam no bug) para o buffer de saída.
  Custo do contorno: ~6% contra o caminho direto (5.988 contra 6.364).
- **Artefato de medição meu.** Na primeira medição com armazenamento fixo, o
  laço de `K` ainda estava com um passo só (resto do bissecionamento) e a
  fórmula de GFLOP/s dividiu pelo trabalho completo: deu "21 TFLOP/s", número
  sem sentido. A comparação só ficou honesta com o laço inteiro — o mesmo tipo
  de erro da especialização alinhada, registrado acima.

O ganho em 4096³ vai além do que a banda explica: a precisão mista sozinha
rendera +1 a 7% no kernel escalar, e aqui são ~40% — os tensor cores estão
somando aritmética real. O primeiro corte não tem buffer duplo nem rasterização
de L2 (o escalar tem ambos); com eles, o teto do desenho comum é a próxima
medida. O `probe_coop.rs` fica como registro do diagnóstico original e o
`probe_coop_f16.rs` como prova do funcionamento.

### Buffer duplo e rasterização L2 no cooperativo — a medida prometida, feita

O parágrafo anterior deixou uma medida pendente: o primeiro corte do kernel
cooperativo não tinha buffer duplo nem rasterização L2, as duas otimizações
que o kernel escalar já usa. Implementadas — mesmo desenho do `MM_FAST`:
ladrilhos `sa`/`sb` em `f16` duplicados e alternados (`cur`/`1-cur`), leituras
globais do ladrilho `t+1` emitidas antes do `coopMultiplyAdd` sobre o ladrilho
`t`, despacho linear com o mesmo `bloco()` (grupo de 8 linhas) reconstruindo o
índice a partir de `workgroup_id.x` — em `crates/rtensor/examples/bench_coop.rs`.

**Metodologia:** `bench_coop.rs` agora roda as duas versões — a antiga
(`FONTE_ANTIGA`, preservada literalmente) e a nova (`FONTE`) — no mesmo
processo e no mesmo dispositivo, o que cancela a deriva de clock entre
execuções separadas (ver "A GPU de notebook muda de clock sozinha" abaixo).
6 execuções do binário inteiro:

| `N` | ganho sobre o 1º corte (6 execuções) | GFLOP/s, novo (min–max) |
|---|---|---:|
| 2048³ | +8,4%, +9,8%, +8,5%, +11,8%, +14,5%, +23,0% (média 12,7%) | 4.386–4.809 |
| 4096³ | +17,6%, +29,0%, +30,0%, +32,5%, +20,1%, +31,6% (média 26,8%) | 7.087–7.204 |

Positivo nas 12 medições (6 por tamanho). O erro numérico não muda —
`1,97e-4` e `5,95e-4` nos blocos amostrais de 2048³ e 4096³, idêntico ao
primeiro corte — porque nem a conversão `f16` nem a acumulação mudaram, só o
padrão de acesso à memória global e à memória de workgroup.

Contra o cuBLAS com TF32 (11.946 GFLOP/s): a distância cai de **~2×** (primeiro
corte, 5.756–5.988) para **1,67×** (7.087–7.204). Contra o kernel escalar de
produção (4.386 GFLOP/s em 4096³): **+63%**, acima dos +37–50% do primeiro
corte.

O ganho em 2048³ é mais instável (8,4% a 23,0%) que em 4096³ (17,6% a 32,5%)
— o mesmo tipo de variação de execução para execução, maior em problemas
menores, já visto em outras sondas deste repositório (a carga por workgroup é
menor, e o ruído de agendamento pesa proporcionalmente mais).

Os dois preços continuam os mesmos — `unsafe` e operandos `f16` — e não há
nada nesta medida que os elimine. O que muda é o número que a decisão pesa: a
matriz cooperativa agora está a 1,67× do cuBLAS, não a 2×.

### À procura do próximo gargalo: três hipóteses refutadas, uma inconclusiva

Com buffer duplo e rasterização L2, o GEMM cooperativo mede ~7.150 GFLOP/s
em 4096³. `examples/ocupacao_coop.rs` mediu o teto puro de `coopMultiplyAdd`
— a mesma aritmética, sem nenhuma leitura de memória nova — em **36 a 45,5
TFLOP/s** (8 execuções, faixa larga pela deriva de clock já documentada).
O GEMM real usa só **~18–20%** desse teto. É uma folga bem maior que a do
kernel escalar antes das otimizações de banda (lá, ~25–28% do teto de FMA
pura). Quatro hipóteses foram testadas para explicar a distância, cada uma
com uma sonda isolada e descartável:

**1. Memória de workgroup (`sc`, 16 KB do total de 24 KB) reduz ocupação —
refutada.** `examples/ocupacao_coop_memoria.rs` soma um array de
preenchimento ao orçamento de memória de workgroup, de 2 KB até 40 KB, sem
tocar a aritmética cronometrada. Vazão **idêntica** em toda a faixa —
inclusive nos 24 KB exatos do kernel real (45.052,6 GFLOP/s contra
45.403,1 GFLOP/s a 4 KB). Se ocupação por memória de workgroup fosse o
limite, teria de aparecer aqui, e não aparece.

**2. `coopLoadT` repetido a cada passo é caro — refutada.**
`examples/ocupacao_coop_load.rs` compara carregar `ma`/`mb` uma vez fora do
laço (como a sonda 1) contra carregar a cada iteração (como o kernel real,
do mesmo endereço fixo). Sem perda: 35.718,8 contra 43.829,7 GFLOP/s — a
segunda até mais rápida, dentro do ruído.

**3. Banda de memória global sozinha — refutada.** `examples/banda_coop.rs`
reproduz o padrão de acesso exato do kernel real — mesmos endereços, mesma
rasterização L2, mesmo buffer duplo, mesmo despacho — removendo só
`coopLoadT`/`coopMultiplyAdd`/`coopStoreT`. Tempo: **5,07 ms** em 4096³,
contra 19,1–19,4 ms do kernel completo — quase 4× mais rápido sem tensor
cores nenhum. Se a banda global sozinha fosse o teto, esse número teria de
se aproximar dos 19 ms, e fica bem abaixo.

**4. Antecipação mais profunda (3 ladrilhos em vez de 2) — inconclusiva.**
A hipótese que sobrou das três refutações: nem memória nem aritmética
isoladas explicam a distância, então é a **interação** entre elas — o
buffer duplo dá só o tempo de `coopMultiplyAdd` (agora quase instantâneo)
de folga para a busca do próximo ladrilho terminar, contra o tempo bem maior
que o FMA escalar levava. `FONTE_PROFUNDA` em `bench_coop.rs` emite a busca
do ladrilho `t+3` na iteração `t`, mas só a guarda na memória de workgroup ao
fim da iteração `t+1` — uma iteração inteira de folga, não só o tempo de
cálculo. Medido no mesmo processo contra o buffer duplo comum, 6 execuções
em 4096³:

| Execução | novo (buffer 2) | profundo (buffer 3) |
|---:|---:|---:|
| 1 | 7.263 | 6.949 |
| 2 | 7.155 | — |
| 3 | 7.172 | — |
| 4 | 7.031 | — |
| 5 | 7.488 | 6.341 |
| 6 | 7.286 | 7.191 |

Média de `novo`: 7.346. Média de `profundo`: 6.807 — **~7% pior**, não
melhor, dentro da mesma faixa de ruído de execução para execução já vista em
outras sondas. A hipótese de profundidade não se confirmou. Uma explicação
plausível: `pend_a`/`pend_b` ficam vivos por toda a iteração (registradores
extras que a versão de 2 buffers não tem), e a pressão de registradores —
um recurso diferente do que a sonda 1 testou (memória de workgroup) — pode
estar anulando o ganho de latência escondida. Isso não foi medido
diretamente; ficaria para uma sonda de ocupação por registrador, análoga a
`ocupacao.rs`, mas específica de tipos de matriz cooperativa.

**O que fica:** três causas eliminadas com clareza, uma quarta tentativa que
não rendeu. A causa exata dos ~80% de teto não usado continua em aberto.
Fixá-la provavelmente exige um perfilador de ocupação real (Nsight Compute
ou equivalente) — ferramenta que este projeto não usa, por ser específica de
fabricante. As quatro sondas ficam publicadas: `ocupacao_coop.rs`,
`ocupacao_coop_memoria.rs`, `ocupacao_coop_load.rs`, `banda_coop.rs`, e a
terceira variante em `bench_coop.rs`.

### GEMM com `A` via `subgroupShuffle` — sonda real, kernel completo nulo

A matriz cooperativa cobra dois preços: `unsafe` (a feature exige
`ExperimentalFeatures::enabled()`, que é `unsafe fn`) e operandos `f16`. A
feature `SUBGROUP` do wgpu é **estável** — não `EXPERIMENTAL_*` — e não exige
`unsafe` nem `f16`, disponível neste hardware (`subgroup_min=32,
subgroup_max=32` na RTX 4070 Laptop, confirmado também no Intel Arc iGPU via
`examples/probe_gpu.rs`). A pergunta: dá para contornar o gargalo de banda da
memória compartilhada (`ROTEIRO-GEMM.md`) sem pagar nenhum dos dois preços?

**O encaixe.** Dentro de um subgrupo de 32 lanes, o endereço de `av` no laço
interno do GEMM depende só de `ty`, que assume **2** valores distintos por
subgrupo; cruzado com os 16 passos de `K` do ladrilho, dá **32 combinações —
exatamente o número de lanes**. Cada lane passa a "possuir" um par
`(b = lane/16, kk_própria = lane%16)`, lê `av` da memória compartilhada **uma
única vez** por ladrilho (contra 16 antes) e usa `subgroupShuffle` para as
outras 15 iterações. `B` não tem esse encaixe (16 endereços × 16 passos = 256
combinações para 32 lanes) e ficou como estava.

**A sonda isolada confirmou o encaixe.** `examples/sonda_subgrupo.rs`
reproduz só o padrão de leitura de `av` — mesma aritmética, mesma contagem de
acessos — comparando a via da memória compartilhada com a via
`subgroupShuffle`:

| Execução | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ganho | +5,7% | +9,4% | +9,3% | +4,6% | +6,1% | +6,9% | +5,7% | +3,2% |

Positivo nas 8 execuções, média ~6% — um resultado real, e uma pista de que a
leitura de `av` (já quase de graça, endereço em broadcast — ver
`banda_compartilhada.rs`) tinha, mesmo assim, uma pequena margem na **contagem
de instruções** do laço, não na banda.

**Implementado no kernel de produção, o efeito desaparece.** `MM_FAST` foi
reescrito com a técnica acima (workgroup 1-D, `local_invocation_index` no
lugar de `local_invocation_id` — `@builtin(subgroup_invocation_id)` não é
aceito pelo naga 30.0.1 com workgroup multidimensional), passou pelos 12
testes de `tests/gpu.rs`, incluindo `gemm_confere_em_todas_as_variantes_e_dimensoes`
e o treino ponta a ponta. Medido no mesmo processo contra `MM_FAST`, 4096³,
grupo_l2=8, 8 execuções:

| Execução | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ganho | +2,5% | +2,0% | -1,6% | -2,3% | +1,3% | -1,7% | -2,7% | -3,4% |

Sinal alterna, cruza o zero, sem tendência — média **-0,7%**. **Dentro do
ruído**, apesar de o mesmo padrão isolado ter sido positivo nas 8 vezes em que
foi medido sozinho. A leitura mais provável: a economia de instruções de
carga é real, mas no kernel completo ela compete por registradores com o
acumulador 4×4 e o buffer duplo já existentes (`av_reg`, `base`,
`kk_propria` são registradores extras por lane), e o que se ganha de um lado
se perde do outro — o mesmo padrão de "diagnóstico certo, peso
superestimado" da tabela "O padrão" acima (cache de descritores, redução de
coluna, padding de bancos, bloco 8×8, leituras `vec4`).

Revertido de `gemm.rs`; o kernel correto e mensurável fica preservado em
[`experimentos/gemm-subgrupo/`](experimentos/gemm-subgrupo/). A sonda isolada
(`examples/sonda_subgrupo.rs`) continua no repositório — o achado dela é real,
só não se propaga ao kernel completo.

### GEMM especializado para formas alinhadas — sem efeito

Duas especializações, para `M` e `N` múltiplos de 64 e `K` de 16: o laço de `K`
desenrolado com deslocamentos literais, e as leituras globais sem guarda de
limite.

A hipótese vinha do diagnóstico — o gargalo é vazão de instruções, e estas são
instruções de puro overhead — e do que funcionara no `rqubit`, onde especializar
com índices literais rendeu 5,32× e 7,65× sobre o kernel genérico.

Quatro execuções em 4096³, três variantes cada:

| Execução | Geral | Sem guardas + laço | Sem guardas, desenrolado |
|---:|---:|---:|---:|
| 1 | 4.274,8 | 4.280,6 | 4.285,7 |
| 2 | 4.273,8 | 4.293,6 | 4.275,8 |
| 3 | 4.220,8 | 4.212,2 | 4.229,0 |
| 4 | 4.185,9 | 4.200,9 | 4.145,9 |

Dentro de cada execução as três diferem menos de 1%; entre execuções a deriva
térmica é de 3%. **São indistinguíveis.**

A explicação provável: `TK` é constante, então o compilador já desenrolava o
laço — a versão manual só engordou o código-fonte. E as guardas são 8 instruções
contra 128 leituras compartilhadas e 64 FMAs por ladrilho: 4% do trabalho,
abaixo do que a medição resolve.

**Nota de método.** A primeira medição deu −3,3% e −4,4%, e eu quase publiquei
"especialização piora o GEMM". Era deriva térmica entre as duas metades da mesma
execução, porque eu media o geral e o alinhado em sequência. Só colocar as três
variantes lado a lado, repetindo, mostrou que não há diferença nenhuma.

**Por que funcionou no `rqubit` e não aqui.** Lá o kernel genérico estagiava
amplitudes em memória compartilhada e percorria a matriz com laços de limite
**variável**, que o compilador não podia desenrolar. Aqui o limite já era
constante. Especializar só rende quando há generalidade de verdade a remover.

### Bloco 8×4 — rejeitado, e contraria o diagnóstico

Ladrilho retangular 128×64, 8 acumuladores `vec4` com índices constantes. Lê 48
bytes por 32 FMAs: **1,33 flop por byte** contra 1,0 do bloco `4×4`. Passou nas
63 combinações de conferência. E mediu neutro:

```
2048³   4×4: 4242 / 4277 / 4202     8×4: 4200 / 4271 / 4241
4096³   4×4: 4195 / 4243            8×4: 4202 / 4253
```

Isso **contraria a previsão feita a partir da sonda de banda compartilhada**,
que indicava teto de ~5 TFLOP/s com 1 flop/byte e ~6,7 com 1,33. Não é
ocupação: o `8×4` usa *menos* memória compartilhada, 12 KB contra 16 KB.

A leitura mais consistente com todos os dados é que o laço interno satura a
**vazão de instruções** do conjunto leitura-compartilhada + FMA, e mudar a
proporção entre as duas não ajuda porque ambas estão perto do limite. Na mesma
sonda, remover 7/8 das multiplicações rendeu 17% e eliminar os conflitos de
banco rendeu 28% — nenhum dos dois isoladamente domina.

Se isso estiver certo, o caminho restante é reduzir as **duas de uma vez**, que
é o que a instrução de matriz cooperativa faz: um ladrilho inteiro por
instrução. Depois de testada (ver a seção da matriz cooperativa): a placa a oferece e ela funciona nas configurações anunciadas — restam os preços do `unsafe` e do `f16`.

### Fusão em unitárias maiores — implementada, e mais lenta

Kernel genérico de até 4 qubits, mais a álgebra para compor portas em conjuntos
diferentes: expansão por produto de Kronecker via índices, e fusão greedy que
respeita a não-comutação de portas que compartilham qubit.

Reduz muito a contagem de portas. E aumenta o tempo. Circuito de 4 camadas em
26 qubits:

| Máx. qubits | Portas | ms | Ganho |
|---:|---:|---:|---:|
| sem fusão | 308 | 1.493 | — |
| **2** | 112 | **545** | **2,74×** |
| 3 | 58 | 595 | 2,51× |
| 4 | 40 | 667 | 2,24× |

Com 40 portas em vez de 112 — pouco mais de um terço — o circuito fica 22% mais
lento.

A causa é o desenho do kernel genérico, não a ideia: 64 threads por workgroup
em vez de 256, estagiagem das amplitudes em memória compartilhada, e laços com
limite variável que o compilador não desenrola. Um kernel especializado para 3
qubits, com índices constantes como nos de 1 e 2, provavelmente inverteria o
resultado — mas não foi escrito.

**O diagnóstico foi testado e confirmou-se.** Escrevendo um kernel
especializado para três qubits — 8 amplitudes em registradores, 64 elementos de
matriz por índice literal, 256 threads, sem memória compartilhada nem um único
laço — o mesmo circuito caiu de 595 para **293 ms**, e a fusão até 3 virou a
melhor opção: **5,32×** contra os 2,74× do limite de 2.

Era o desenho do kernel, não a ideia — e vale para os dois tamanhos. O kernel
especializado de quatro qubits levou o mesmo circuito de 690 para **204 ms**:
**7,65×** sobre o circuito sem fusão, com 7,70× menos portas. Proporcional até
o segundo decimal.

O genérico fica no repositório como referência e caso de comparação, mas o
caminho padrão não o usa mais.

Uma nota de método: a primeira medição deu números piores ainda, porque eu
mandava até as portas de 1 e 2 qubits para o kernel genérico. Corrigir o
despacho tirou o `max = 2` de 1,84× para 2,74× — a comparação só ficou honesta
depois disso.

### Precisão mista — não é ganho de velocidade nem de energia

Operandos em `f16` empacotado, acumulação em `f32`. A hipótese era que o GEMM
estava limitado por banda de memória, e que cortar os bytes pela metade quase
dobraria a vazão.

**A conta que sustentava a hipótese estava errada.** O `1 GB` que eu atribuí ao
tráfego de 2048³ supõe reuso zero de L2 — é um limite superior, não o tráfego
real. Com a rasterização de L2 em operação, a ida efetiva à DRAM é bem menor, e
a placa não está perto de saturada.

| Tamanho | Velocidade | Eficiência energética | Erro |
|---|---:|---:|---:|
| 2048³ | +6,6% | −10,9% | 6.827× |
| 4096³ | +1,2% | −5,4% | 5.983× |
| 6144³ | +2,3% | −10,0% | 3.864× |

O desempacotamento custa ULA por elemento lido; numa carga que não é limitada
por banda, essa conta extra gasta potência sem comprar tempo. Foi a primeira vez
que a medição de energia contradisse a de tempo — e é exatamente para isso que
ela existe.

**Mantida, mas por outra razão:** os operandos ocupam metade do espaço, o que
permite modelos ou lotes maiores quando a VRAM é o limite. É troca de precisão
por capacidade, não por velocidade, e o código diz isso.

### Leituras globais de 128 bits — rejeitado

Bindings de `A` e `B` como `array<vec4<f32>>`: uma leitura de 128 bits por
thread em vez de quatro de 32. Passou nos 42 casos de conferência e não rendeu.

```
2048³            escalar: 4.230 GFLOP/s    vec4: 4.172
4096³            escalar: 4.030            vec4: 4.044
4096×4096×64     escalar: 3.541            vec4: 3.306
```

O último formato é o teste decisivo: com `K = 64` e `M = N = 4096` há
pouquíssimo cálculo por byte lido — exatamente onde uma leitura mais larga
deveria aparecer. Não apareceu.

Revertido por não pagar a complexidade: três pipelines a mais, um módulo WGSL a
mais e lógica de alinhamento no despacho, em troca de zero.

### Padding contra conflito de bancos — mantido, sem ganho

A análise do conflito estava certa: o passo `TM = 64` fazia 16 threads de um
warp escreverem todas no banco 0. O **peso** estava errado. Por thread e por
ladrilho `K` são 8 escritas contra 128 leituras — as escritas são 6% do tráfego.
E as leituras já eram livres de conflito: sendo `vec4`, o hardware as processa
em quatro fases de 8 threads, e 8 threads × 4 bancos cobrem os 32 bancos.

Mantido porque é correto e custa 32 floats, mas sem crédito de ganho.

## Erros de medição

Estes custaram mais tempo que os erros de código.

### A partida a fria misturava imposto de import com arquitetura

O número de partida a fria do README (**0,29–0,37 s** do `rtensor` contra
**2,41 s** do PyTorch e **5,64 s** do TensorFlow) mede o processo inteiro, e
foi apresentado como vantagem do caminho CUDA-free. Não era só isso: o
`rtensor` é um binário Rust nativo, e PyTorch/TensorFlow são programas Python
que precisam primeiro importar a biblioteca — um custo de linguagem
hospedeira, não de GPU.

Decompondo com `benchmarks/python/frio_decomposto_tf.py` e
`frio_decomposto_torch.py` em três fases — import, montagem do
modelo/inicialização de GPU, primeiro passo —, mediana de 5 execuções cada
(TensorFlow 2.21.0, PyTorch 2.14.0+cu130, RTX 4070 Laptop):

| Motor | Import | Init. GPU/modelo | Primeiro passo | Total decomposto | Total medido por fora |
|---|---:|---:|---:|---:|---:|
| rtensor | — | 0,317 s | 0,0022 s | 0,319 s | 0,345 s |
| PyTorch | 1,459 s | 1,457 s | 0,229 s | 3,145 s | 4,29 s |
| TensorFlow | 1,871 s | 1,260 s | 1,431 s | 4,562 s | 5,59 s |

**Cerca de um terço do total de TensorFlow e PyTorch é só `import`** — 1,87 s
e 1,46 s respectivamente, sem nenhuma GPU envolvida ainda. Descontado isso, a
comparação que sobra é mais estreita e mais honesta:

- **Inicialização de dispositivo** (`wgpu`: instância, adaptador, device,
  compilação de shaders) contra CUDA/cuDNN se preparando e o Keras montando o
  grafo: **0,317 s contra 1,26–1,46 s — 4 a 4,6×**, e este número é
  arquitetura de verdade.
- **Primeiro passo**: o TensorFlow paga aqui o traçado do `tf.function`
  (AutoGraph + grafo de gradiente), o que infla a distância para 650×. Contra
  o PyTorch em modo eager, sem traçado, a distância cai para **104×** — o
  número mais limpo de "despachar um passo já compilado" que este repositório
  tem.
- **A diferença residual** (total medido por fora menos a soma das três
  fases: ~1,0 s no TensorFlow, ~1,15 s no PyTorch, ~26 ms no `rtensor`) é
  overhead de processo — start do interpretador, `atexit` das threads da
  biblioteca, flush de saída. Não foi decomposta mais fundo.

A primeira execução de cada processo, com cache de driver e de disco frios,
mede sistematicamente mais alto — no `rtensor`, 3,2 s contra ~0,33 s nas
seguintes; no TensorFlow, a fase de montagem chegou a 3,47 s numa execução
contra ~1,26 s nas outras quatro. As medianas acima descartam esse efeito de
cache frio, e o texto do README já assinalava isso para o `rtensor`
("0,29–0,37 s"); aqui ele é generalizado às três engines.

**A conclusão não muda, mas o crédito é redistribuído**: o `rtensor` ainda
vence com folga, mas ~35% da vantagem total é "binário nativo não paga import
de Python" — um resultado real para quem escreve um script que roda e
termina, mas que não tem relação com CUDA ser ou não contornável — e o
restante é, sim, o `wgpu` inicializando mais rápido e despachando sem
traçar grafo.

### A tabela de regime permanente também misturava despacho com arquitetura

A mesma pergunta da partida a fria — quanto é linguagem hospedeira, quanto é
GPU — se aplica à tabela "Tempo por passo de treino" do README, e por um
motivo concreto: a coluna do TensorFlow é **quase plana** (1,15 a 2,05 ms de
lote 1 a 2048), sem a escalada com o trabalho que as outras colunas mostram.
Isso é sintoma de um custo fixo dominando a medida.

`benchmarks/python/despacho_tf.py` isola o despacho: a mesma `tf.function`
traçada, mas com um corpo quase vazio (`tf.reduce_sum(x) * 0.0 +
tf.cast(y[0], ...)`), medida com o mesmo `cron()` — aquece, cronometra
`reps` chamadas, sincroniza uma vez no fim — contra o passo real, no mesmo
lote. `despacho_torch.py` e `crates/rtensor/examples/despacho_rtensor.rs`
fazem o mesmo para PyTorch (eager) e `rtensor` (um `relu` num tensor de 1
elemento, mesmo padrão de submissão: um `CommandEncoder`, um dispatch, um
`submit`).

| Motor | passo real (lote 1–2048) | despacho trivial | despacho, % do passo |
|---|---:|---:|---:|
| TensorFlow | 1,68–2,09 ms | **0,43–0,70 ms** | **23% a 36% (~29%)** |
| PyTorch | 1,01–2,27 ms | 0,04–0,06 ms | 2,0% a 4,0% |
| rtensor | 0,49–5,89 ms | 0,03–0,07 ms | 0,5% a 13%, caindo com o lote |

Números por lote (3 execuções cada, TensorFlow e PyTorch; 2 execuções,
`rtensor`):

```
TF       lote=1    real=1,71–1,93 ms  trivial=0,43–0,56 ms  despacho=25–29%
TF       lote=128  real=1,70–2,04 ms  trivial=0,56–0,65 ms  despacho=31–33%
TF       lote=2048 real=1,75–1,92 ms  trivial=0,42–0,62 ms  despacho=23–32%

torch    lote=1    real=1,01–1,48 ms  trivial=0,041 ms      despacho=2,8–4,0%
torch    lote=128  real=1,07–1,59 ms  trivial=0,041 ms      despacho=2,6–3,8%
torch    lote=2048 real=2,26–2,27 ms  trivial=0,046 ms      despacho=2,0–2,1%

rtensor  lote=1    real=0,49–0,66 ms  trivial=0,029–0,052 ms  despacho=5,3–7,9%
rtensor  lote=128  real=0,94–1,05 ms  trivial=0,029–0,030 ms  despacho=2,8–3,1%
rtensor  lote=2048 real=4,85–5,89 ms  trivial=0,029–0,030 ms  despacho=0,5–0,6%
```

**Cerca de 30% de cada número da coluna TF GPU é despachar a chamada, não
computar** — um custo fixo de ~0,5 ms, essencialmente independente do lote,
o que explica a planura da coluna. `torch` e `rtensor` têm despacho perto do
desprezível (2–4% e 1–8%, respectivamente) — a diferença entre eles na
tabela principal é computação de verdade, não overhead de linguagem.

A causa arquitetural é conhecida da literatura: o modo eager do PyTorch
despacha a operação C++ direto, enquanto o `tf.function` do TensorFlow, mesmo
já traçado, passa por uma camada de execução de grafo (`FunctionLibraryRuntime`)
a cada chamada — mais pesada que uma chamada eager, e completamente
independente de CUDA ou de qualquer coisa que este repositório meça sobre
GPU.

**O que isso não muda**: nenhum vencedor de linha na tabela principal troca
de mãos — o PyTorch já vencia todo o intervalo, e o `rtensor` já perdia em
lote grande. **O que muda**: a leitura de que o TensorFlow é uniformemente
~3× pior que o `rtensor` em lote pequeno superestima o que é arquitetura de
GPU; cerca de um terço disso é o mesmo imposto de linguagem hospedeira já
identificado na partida a fria, só que pago a cada passo em vez de uma vez
só.

### A GPU de notebook muda de clock sozinha

A mesma configuração mediu de **1,9 a 2,4 ms** entre execuções, e a placa variou
entre **38 W e 112 W** ao longo de uma sessão. Efeitos abaixo de ~20% não são
resolvíveis por cronometragem de parede aqui. O instrumento adequado é o
perfilamento por marcas de tempo do dispositivo (`examples/perfil.rs`).

### Carga de CPU derruba o clock da GPU

Medir a GPU **depois** de um benchmark pesado de CPU inflou os tempos em até
**3×**: numa Max-Q, CPU e GPU dividem orçamento de energia. Um salto de 8,6×
entre lote 128 e 512 foi atribuído por engano a um kernel de redução; a causa
era térmica. A suíte hoje mede a GPU primeiro.

### A linha de base ociosa não é uma constante

Subtrair a ociosidade da energia medida parece óbvio e é inválido quando o ponto
de operação muda. Uma GPU travada em 210 MHz consumiu **12,3 W em carga**,
abaixo dos 14,15 W medidos como ociosidade no clock padrão. A subtração ficou
negativa e foi zerada, produzindo "0,00 GFLOP/J" — um número inventado no lugar
de uma medição. Hoje `EnergiaGpu::base_suspeita` detecta o caso e cai para a
energia total.

### O `nvidia-smi` não mede a GPU da Intel

A primeira versão do comparativo entre fabricantes atribuiu **0,076 J por passo**
à Intel Arc. Era a NVIDIA ociosa. A energia da GPU integrada vem do domínio
`uncore` do RAPL, e na ausência dele o campo deve ficar vazio — não preenchido
com a leitura do sensor errado.

### Janela curta demais mede o sensor, não a carga

O `nvidia-smi` atualiza a potência a cada ~100 ms e demora a refletir mudança de
carga. Uma janela de 0,42 s leu energia zero acima da ociosidade. A janela agora
é calibrada em tempo, com alvo de 6 s.

### Dividir por energia nula não dá eficiência infinita

Dá medição falha. Hoje é descartada.

### A referência de `f16` do `rqubit` devolvia metade dos subnormais

A conversão `f16 → f32` escrita à mão no `rqubit` — para manter a crate sem
dependências — normalizava subnormais com um expoente a mais no laço: `0x0001`
virava 2⁻²⁵ em vez de 2⁻²⁴, metade do valor, em todos os subnormais. Achado por
ocasião da sonda da matriz cooperativa, cuja referência de CPU ia copiar a
função; conferido contra a crate `half` num projeto descartável fora do
repositório. Corrigido com os valores fixados em teste unitário
(`crates/rqubit/src/lib.rs`); a suíte inteira segue verde. Amplitudes quânticas
podem ser subnormais, e a referência de CPU é o árbitro dos kernels — os testes
existentes não pegaram porque as amplitudes deles eram todas normais.

## Limites de hardware encontrados

| Limite | Consequência |
|---|---|
| `nvidia-smi -pl` recusa em GPU de notebook | Varredura de potência impossível; o firmware controla o envelope, `sudo` não contorna |
| `QuerySet` limitado a 4.096 consultas | O perfilador ajusta o número de passos à capacidade |
| 65.535 workgroups por dimensão | Os kernels elementares despacham em grade 2-D |
| RAPL exige root desde 2020 | Energia de CPU e iGPU só com privilégio (mitigação do PLATYPUS) |
| Firmware recusa clocks acima de ~2.500 MHz sob carga | Pedimos 2.745 e 3.105 MHz e recebemos 2.490 e 2.520 |

## O padrão

Em seis das sete previsões confiantes feitas durante o desenvolvimento, o
gargalo foi **identificado corretamente** e o **peso dele foi superestimado**.
Foi assim com o cache de descritores, com a redução de coluna, com o padding de
bancos, com o bloco 8×8 e com as leituras `vec4`.

**Três micro-otimizações do laço interno falharam em sequência** — swizzling,
bloco 8×8 e leituras vetoriais. Isso deixou de ser azar e virou sinal: o laço
interno não é o gargalo. A sonda de ocupação confirma por outro caminho, ao
mostrar que aritmética pura sustenta 15–18 TFLOP/s nesta placa enquanto o GEMM
anda a 4.

**O gargalo foi identificado: largura de banda da memória compartilhada.**

`examples/banda_compartilhada.rs` lê os mesmos bytes no mesmo padrão de
endereços do laço interno do GEMM, variando só o que faz com eles:

| Modo | ms | TB/s lidos | GFLOP/s |
|---|---:|---:|---:|
| `fma` — as 16 FMAs por passo, como no GEMM | 17,29 | **3,98** | **3.975** |
| `soma` — mesmos bytes, 1/8 da aritmética | 14,79 | 4,65 | — |
| `broadcast` — mesmos bytes, sem conflito | 13,48 | **5,10** | — |

Três leituras deste resultado:

1. **A sonda reproduz o GEMM.** 3.975 GFLOP/s contra os 4.300 do GEMM real. O
   laço interno com suas leituras compartilhadas *é* o gargalo, isolado.
2. **Não é aritmética.** Remover 7/8 das multiplicações rende 14%.
3. **É o canal de memória compartilhada** — foi a conclusão na época, e o
   bloco `8×4` depois a enfraqueceu: aumentar a intensidade de 1,0 para 1,33
   flop/byte não rendeu nada. A leitura revisada está na seção do `8×4`.

O laço interno lê 32 bytes da memória compartilhada por 32 flops — **1 flop por
byte**. Com o canal saturando em ~5 TB/s, o teto é ~5 TFLOP/s, e estamos a 4,3.

Isso fecha a conta de todas as tentativas anteriores. E deixa uma contradição
registrada: o bloco 8×8 dobraria a intensidade para 2 flops por byte, elevando o
teto para ~10 TFLOP/s — e mediu 5% pior. A explicação provável é pressão de
registradores no kernel completo, que a sonda sintética não reproduz. Um bloco
intermediário, `8×4` com 32 acumuladores, aumentaria a intensidade em 1,5× com
metade da pressão. Não foi testado.

O que a literatura clássica indica em seguida está em
[ROTEIRO-GEMM.md](ROTEIRO-GEMM.md).

É por isso que `examples/ocupacao.rs` e `examples/perfil.rs` existem: para que
essa classe de decisão seja medida antes de ser implementada, e não depois.
