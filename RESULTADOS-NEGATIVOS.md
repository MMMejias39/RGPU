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
| Rasterização com consciência de L2 | **+5 a 6%** em 4096³ | em produção |
| Strassen de um nível | **+11 a 13%** acima de 4096³, **−52%** em 1024³ | opcional |
| Split-K | **+12,6× a 21,4×** em formas K-dominantes | opcional |

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

O que a literatura clássica indica em seguida está em
[ROTEIRO-GEMM.md](ROTEIRO-GEMM.md).

É por isso que `examples/ocupacao.rs` e `examples/perfil.rs` existem: para que
essa classe de decisão seja medida antes de ser implementada, e não depois.
