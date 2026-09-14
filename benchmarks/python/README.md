# Benchmarks em TensorFlow e PyTorch

Os scripts que produziram os números de TensorFlow e PyTorch citados no
[README da raiz](../../README.md). Mesma carga, mesma máquina, mesmas formas de
matriz que os benchmarks em Rust.

## Ambiente

O TensorFlow não tem roda para Python 3.14, e a leitura de energia da GPU exige
as bibliotecas CUDA. O ambiente usado foi:

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV=$PWD/venv uv pip install "tensorflow[and-cuda]" torch
```

O TensorFlow instalado por `pip` não encontra sozinho as bibliotecas CUDA que
ele mesmo empacota. Sem isto ele cai silenciosamente para CPU, dizendo apenas
*"Cannot dlopen some GPU libraries"*:

```bash
export LD_LIBRARY_PATH=$(find venv/lib/python3.12/site-packages/nvidia \
  -name lib -maxdepth 2 -type d | tr '\n' ':')
```

## Scripts

| Arquivo | O que mede |
|---|---|
| `suite_tf.py` | `gemm`, `batch`, `arch`, `infer`, `frio` — a contraparte de `examples/suite.rs`. Usa `tf.function` cru em vez de `model.fit`, para medir o TensorFlow no seu melhor |
| `bench_tf.py` | A tarefa das espirais, lendo os CSV que `examples/bench.rs` exporta — garante que os dois motores treinam exatamente as mesmas amostras |
| `bench_grande_tf.py` | Carga pesada parametrizável: `bench_grande_tf.py N D H LOTE` |
| `bench_torch.py` | Varredura de lote em PyTorch: `bench_torch.py [cpu\|cuda]` |
| `frio_torch.py` | Partida a frio do PyTorch; meça o processo inteiro com `time` |
| `frio_decomposto_tf.py` | Partida a fria do TensorFlow, separada em import / montagem do modelo e GPU / primeiro passo |
| `frio_decomposto_torch.py` | O mesmo para PyTorch — usado para separar imposto de import de arquitetura na comparação de partida a fria |
| `tf32_probe.py` | Liga e desliga os tensor cores TF32 para medir quanto eles valem |

## Uso

```bash
python suite_tf.py batch gpu
TF_FORCE_CPU=1 python suite_tf.py batch cpu
python bench_torch.py cuda
python tf32_probe.py on ; python tf32_probe.py off
```
