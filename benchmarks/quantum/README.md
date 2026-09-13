# Comparação com Qiskit Aer e cuQuantum

Os scripts que produziram os números de simulação quântica do
[README da raiz](../../README.md). Mesma carga do
`rqubit/examples/bench_porta.rs`: rotações `RY` com ângulos distintos,
percorrendo os qubits em rodízio.

## Ambiente

O venv fica **fora** do diretório temporário: o scratchpad da sessão é `tmpfs`,
ou seja, RAM, e estes pacotes passam de 3 GB.

```bash
uv venv --python 3.12 ~/.cache/rgpu-benchmarks/quantum
VIRTUAL_ENV=~/.cache/rgpu-benchmarks/quantum uv pip install \
  qiskit qiskit-aer cuquantum-python-cu12 \
  nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12

export LD_LIBRARY_PATH=$(find ~/.cache/rgpu-benchmarks/quantum/lib/python3.12/site-packages/nvidia \
  -name lib -maxdepth 2 -type d | tr '\n' ':')
```

`qiskit-aer-gpu` **não** foi usado: a versão no PyPI (0.15.1) é incompatível com
o Qiskit 2.5 e quebra a instalação. O Aer aqui é CPU; o cuStateVec cobre a GPU.

## Duas armadilhas que invalidam a medição

**Portas que se cancelam.** Hadamard aplicada duas vezes no mesmo qubit é a
identidade, e o transpilador do Qiskit a remove. A primeira medição deu
1.196 GB/s numa CPU de ~100 GB/s — não era velocidade, era o circuito sendo
apagado. Os scripts usam `RY` com ângulos distintos e `optimization_level=0`.

**Fusão de portas.** O Aer funde portas consecutivas em unitárias maiores por
padrão, o que dá 5,8×. É otimização legítima, mas compara o otimizador de
circuito dele com o nosso kernel. `fusion_enable` permite medir os dois casos.

O `rqubit` passou a ter fusão própria — ver `Circuito::fundir` —, então a
comparação com `fusion_enable=on` deixou de ser assimétrica.

**Precisão.** O Aer usa `complex128` por padrão, o dobro dos bytes do nosso
`complex64`. `precision='single'` iguala.

## Scripts

| Arquivo | Uso |
|---|---|
| `bench_aer.py` | `bench_aer.py <qubits> <portas> <CPU\|GPU> <on\|off> <single\|double>` |
| `bench_custatevec.py` | `bench_custatevec.py <qubits> <portas> <single\|double>` |
| `bench_aer_camadas.py` | Circuito em camadas, igual ao de `bench_fusao.rs` |
| `bench_custatevec_camadas.py` | O mesmo circuito, no cuStateVec |

## Energia

Os simuladores existentes reportam tempo, nunca joules. Como a potência da GPU
é medida por fora do processo, dá para instrumentá-los sem tocar no código:

```bash
cargo run -p rqubit --release --example medir_externo -- \
  ~/.cache/rgpu-benchmarks/quantum/bin/python bench_custatevec.py 26 400 single
```

A energia do processo inclui a partida do Python. Para isolar o custo marginal
por porta, mede-se com dois números de portas e toma-se a diferença.
