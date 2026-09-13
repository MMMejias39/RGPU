# O que a literatura clássica indica

Três micro-otimizações do laço interno falharam em sequência — swizzling de
bancos, bloco 8×8 por thread e leituras globais de 128 bits. Todas passaram nos
testes de correção e nenhuma rendeu desempenho. Isso deixou de ser azar e virou
sinal.

A sonda de ocupação confirma por outro caminho: aritmética pura sustenta **15 a
18 TFLOP/s** nesta placa, e o GEMM anda a **4**. Estando 4× longe do limite das
ULAs, não é o laço interno que prende.

Este documento reúne o que a literatura de álgebra linear numérica diz sobre
onde olhar, das bibliotecas dos anos 1970–90 até os artigos de GPU.

## A linhagem

O problema de multiplicar matrizes rápido é anterior às GPUs por décadas, e a
resposta mudou de natureza duas vezes.

**Anos 1970–80 — as bibliotecas de referência.** IMSL e as primeiras BLAS
estabeleceram a interface, não o desempenho: a implementação de referência era
correta e portátil, e deixava o desempenho para o fornecedor. O *Numerical
Recipes* segue a mesma linha — o código de multiplicação de matrizes ali é
pedagógico, escrito para ser lido, não para saturar uma máquina.

**Anos 1990 — o insight do bloco.** O LAPACK reestruturou os algoritmos para
chamar BLAS de nível 3 (matriz × matriz) em vez de nível 1 e 2 (vetor × vetor,
matriz × vetor). O argumento é de superfície contra volume: um bloco `n×n` faz
`n³` operações movendo `n²` dados, então **blocos maiores melhoram a razão entre
cálculo e tráfego**. É a mesma ideia que já usamos no ladrilho de memória
compartilhada.

**2008 — a anatomia.** Kazushige Goto e Robert van de Geijn,
[*Anatomy of High-Performance Matrix Multiplication*](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
(ACM TOMS 34(3)). É o artigo que define como toda BLAS moderna é construída — do
GotoBLAS ao OpenBLAS, ao BLIS e, adaptado para GPU, ao CUTLASS e portanto ao
cuBLAS.

A contribuição central **não é o laço interno**: é o arranjo dos laços externos e
o **empacotamento explícito** dos dados, de modo que um bloco de `A` permaneça
residente na cache L2 enquanto é reusado por muitas colunas de `B`.

O [BLIS](https://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf) refatorou isso
em **cinco laços aninhados** ao redor de um micro-kernel, um por nível da
hierarquia de memória.

## O que isso diz sobre o nosso GEMM

Temos **um** nível de blocagem: global → memória compartilhada. Não há nada
equivalente à blocagem de L2 do Goto.

Cada workgroup lê seu painel de linhas de `A` e seu painel de colunas de `B`
direto da memória global, e workgroups que compartilham os mesmos painéis não
têm nenhuma coordenação para executar próximos no tempo. Se rodarem distantes, o
painel sai da L2 entre um uso e outro e é relido.

O equivalente da blocagem de L2 numa GPU é a **rasterização de blocos**:
reordenar os workgroups — em serpentina ou em grupos de colunas — para que os
que compartilham painéis executem juntos e encontrem os dados na L2. É a única
técnica da lista que ataca a hierarquia de memória em vez do laço interno, e não
custa precisão nenhuma.

## O caminho algorítmico: Strassen

A outra vertente não otimiza a execução — reduz a conta. Strassen (1969)
multiplica dois blocos `2×2` com **sete** multiplicações em vez de oito, à custa
de mais somas. Um nível de recursão corta ~12,5% das multiplicações.

O que a literatura recente mede em GPU:

- [*Strassen's Algorithm Reloaded on GPUs*](https://dl.acm.org/doi/10.1145/3372419)
  (ACM TOMS): **32% de ganho** em precisão simples para matrizes 16384², e 20,2%
  em precisão dupla a 8192².
- [*Implementing Strassen's Algorithm with CUTLASS on NVIDIA Volta GPUs*](https://arxiv.org/pdf/1808.07984):
  um nível de Strassen dá ganho prático **mesmo em matrizes pequenas**, abaixo de
  3.000, e **sem espaço de trabalho extra** — ao contrário das implementações
  convencionais, que trocam memória por tempo.

O custo é numérico. Strassen satisfaz apenas um limite de erro por norma,
`‖C − Ĉ‖ ≤ f(n)ε‖A‖‖B‖`, e não o limite por elemento do algoritmo clássico. Na
prática, o erro máximo e o médio ficam cerca de **duas ordens de grandeza**
acima do `sgemm` em `n = 16384`, e a estabilidade degrada rápido acima de dois
níveis de recursão.

Para este repositório esse custo é mensurável, não teórico: `tests/gpu.rs` já
compara a GPU contra o motor de CPU com tolerância apertada em sete formatos. Um
Strassen de um nível apareceria ali como aumento de erro relativo, e daria para
publicar o câmbio exato entre velocidade e precisão — que é o tipo de número que
quase ninguém mede.

## Ordem sugerida

1. ~~**Rasterização de blocos com consciência de L2**~~ — **feito**, +5 a 6% em
   4096³, com curva reproduzível. Foi a primeira otimização em quatro tentativas
   a render, e a única que atacava a hierarquia de memória em vez do laço
   interno. `Gpu::set_grupo_l2` expõe o parâmetro; `1` reproduz o percurso em
   linha para comparação.
2. ~~**Strassen de um nível**~~ — **feito**. Ganho de **+11 a +13% em 4096³**,
   capturando 90% do máximo teórico, mas **negativo abaixo de 4096**: −24% em
   2048³ e −52% em 1024³. O custo de precisão ficou em 2,5× a 3,2× o erro do
   clássico, bem abaixo das duas ordens de grandeza que a literatura reporta
   para vários níveis em `n = 16384`. Não é automático, justamente por perder na
   faixa comum.
3. ~~**Split-K**~~ — **feito**, e foi o maior ganho de toda a sequência:
   **12,6× a 21,4×** nas formas K-dominantes. Não melhora o caso comum, como
   previsto; corrige o patológico.

O que **não** fazer: mais micro-otimização do laço interno. Três tentativas, três
resultados nulos ou negativos, e uma sonda que explica por quê.

## O gargalo, finalmente medido

`examples/banda_compartilhada.rs` isolou a resposta: **largura de banda da
memória compartilhada**. O padrão exato do laço interno entrega 3,98 TB/s contra
um teto de 5,10 TB/s sem conflito de banco, e rende 3.975 GFLOP/s — praticamente
os 4.300 do GEMM completo.

Remover 7/8 das multiplicações melhora só 14%, então não é aritmética.

A aritmética por byte lido é o número que governa tudo: o laço interno faz
**1 flop por byte** de memória compartilhada, e o canal satura em ~5 TB/s. É o
teto de ~5 TFLOP/s que estamos encostando.

## O que resta

A matriz cooperativa era a última técnica da lista e a que o diagnóstico
revisado indicava. O registro anterior dizia **não funciona**; o diagnóstico
estava errado: a sonda fixava `8×8 f32`, configuração que a placa não anuncia —
comportamento indefinido, e os zeros vinham daí. Nas configurações anunciadas
(`16×16`, operandos `f16`), a cadeia inteira funciona com erro zero, inclusive
com estagiagem em memória de workgroup (`examples/probe_coop_f16.rs`). Restam
dois preços: `ExperimentalFeatures::enabled()` é `unsafe fn` — custaria a
propriedade "zero `unsafe`" do projeto — e os tensor cores desta placa só
aceitam operandos em `f16`. O veredito numérico já existe: o kernel GEMM
cooperativo (`examples/bench_coop.rs`) mede **5.756–5.988 GFLOP/s em 4096³**
contra 3.804–4.243 do escalar — **+37% a +50%** — e empata em 2048³. Pagar os
dois preços (`unsafe` e `f16`) por isso é uma decisão que agora tem número.
Detalhes em [RESULTADOS-NEGATIVOS.md](RESULTADOS-NEGATIVOS.md).

O roteiro clássico está esgotado dentro do WGSL padrão — o GEMM ficou em
~4.300 GFLOP/s, ou ~4.900 com Strassen acima de 4096³, contra 3.290 no início,
**+33% a +49%**. O que resta fora do padrão é o caminho experimental dos tensor
cores, agora funcional e esperando medição.

## O saldo

Os três itens que a literatura indicou renderam; as três micro-otimizações do
laço interno não. O padrão é consistente: **o que paga é mexer em como os dados
se movem e em quanta conta se faz, não em como o laço interno a executa.**
