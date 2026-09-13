# rtensor

Um framework de deep learning em **Rust puro** — sem `unsafe`, sem FFI, e sem
dependências no núcleo. Nasceu como um exercício de reescrever o TensorFlow em
Rust e cresceu além disso: hoje tem backend próprio de GPU via `wgpu` que
ultrapassa o TensorFlow com cuBLAS em várias faixas de carga.

O objetivo nunca foi reimplementar os ~3 milhões de linhas do TensorFlow, e sim
conter em poucos milhares de linhas legíveis as peças de que todo framework de
deep learning é feito — e mostrar que a matemática delas cabe num projeto que
uma pessoa consegue ler inteiro.

## As quatro peças

| Peça | Arquivo | Equivalente no TensorFlow |
|---|---|---|
| Tensor denso com broadcasting NumPy | `src/tensor.rs` | `tf.Tensor` |
| Autodiff reverso em fita | `src/tape.rs` | `tf.GradientTape` |
| Camadas, ativações, modelo sequencial | `src/nn.rs` | `tf.keras.layers`, `Sequential` |
| Otimizadores (SGD+momentum, Adam) | `src/optim.rs` | `tf.keras.optimizers` |
| Perdas e métricas | `src/losses.rs` | `tf.keras.losses` |
| Backend de GPU (wgpu/WGSL) | `src/gpu/` | kernels CUDA/cuDNN |

## Correspondência de API

```python
# TensorFlow / Keras
with tf.GradientTape() as tape:
    logits = model(x)
    loss = tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True)
grads = tape.gradient(loss, model.trainable_variables)
opt.apply_gradients(zip(grads, model.trainable_variables))
```

```rust
// rtensor
let tape = Tape::new();
let entrada = constant(&tape, x.clone());
let logits = model.forward(&entrada);
let loss = losses::softmax_cross_entropy(&logits, &y);
let grads = loss.backward();
opt.step(&params, &grads);
```

| TensorFlow | rtensor |
|---|---|
| `tf.constant(v)` | `constant(&tape, t)` |
| `tf.Variable(v)` | `Param::new("W", t)` |
| `tape.watch(v)` | `param.watch(&tape)` |
| `tape.gradient(loss, vs)` | `loss.backward()` → `grads.of(&param)` |
| `tf.keras.layers.Dense(n, activation=...)` | `Dense::new(inputs, n, Activation::_, &mut rng)` |
| `model.summary()` | `model.summary()` |
| `model.predict(x)` | `model.predict(&x)` |

## Como funciona o autodiff

Cada operação grava na fita um nó com seus pais e uma closure que converte o
gradiente da saída nos gradientes das entradas. Como os nós nascem em ordem
topológica, o backward é **uma única varredura de trás para frente**, acumulando
gradientes — o mesmo princípio do modo eager do TensorFlow 2.

Gradientes de operandos difundidos são somados de volta ao shape original por
`Tensor::reduce_to`, o que faz o gradiente de um bias `[1, m]` sair correto num
lote `[n, m]` sem nenhum tratamento especial na camada `Dense`.

## GPU

A feature `gpu` acrescenta um backend via [`wgpu`](https://github.com/gfx-rs/wgpu),
com kernels em WGSL. Roda sobre Vulkan, Metal, DX12 ou WebGPU — NVIDIA, AMD,
Intel e Apple, sem CUDA e sem `unsafe`.

```bash
cargo run --release --features gpu --example probe_gpu   # lista os adaptadores
cargo run --release --features gpu --example bench_gpu   # benchmark na GPU
cargo test --features gpu                                # + 6 testes CPU vs GPU
```

Os pesos ficam residentes no dispositivo: um passo inteiro — avanço,
retropropagação e Adam — é gravado num único `CommandEncoder` e submetido sem
sincronizar. Os kernels são `mm`/`mm_atb`/`mm_abt` (ver GEMM abaixo),
`bias_add`, `relu`, `relu_bwd`, `softmax_xent` (fundido, devolvendo `(P−Y)/n`),
`colsum` (`1ₙᵀδ`) e `adam` (fundido sobre `θ`, `m`, `v`).

### O GEMM

`src/gpu/gemm.rs` traz ladrilhamento em dois níveis, o desenho do cuBLAS:

- **workgroup → memória compartilhada:** ladrilho de saída 64×64, passo `K` de
  16, 8 KB de memória compartilhada;
- **thread → registradores:** cada uma das 256 threads calcula um bloco 4×4,
  acumulado em `array<vec4<f32>, 4>`.

O ganho vem da intensidade aritmética: o kernel ingênuo faz 1 FMA por 2 leituras
da memória compartilhada; este faz 16 FMAs por 8 leituras — 4× mais. Medido em
`C[1024³]`: **762 → 2.713 GFLOP/s, 3,56×**. `Gpu::set_fast_gemm(false)` volta ao
ingênuo, e `tests/gpu.rs` confere os dois contra a CPU em 7 formatos × 3
variantes.

Os kernels elementares despacham numa grade 2-D: um dispatch 1-D estoura o teto
de 65.535 workgroups por dimensão já numa camada 4096×4096.

O caminho de CPU deriva os gradientes pela fita; o de GPU traz a
retropropagação escrita à mão, como nos kernels fundidos do TensorFlow.
`tests/gpu.rs` confere uma contra a outra: mesmos pesos iniciais, 120 passos, e
as duas trajetórias têm de coincidir.

## Rodando

```bash
cargo test                                 # 13 testes, incl. conferência numérica dos gradientes
cargo test --features gpu                  # 19 testes, incl. CPU vs GPU e os dois GEMMs
cargo run --release --example xor          # XOR: o problema não-linear canônico
cargo run --release --example espiral      # 3 classes em espirais, ~99% no teste
```

O `cargo test` inclui verificação de gradiente por diferenças centrais
`(f(x+h) − f(x−h)) / 2h` comparada ao autodiff para MLP+MSE, softmax+entropia
cruzada, sigmoid+BCE e softmax explícito. É a prova de que a matemática do
backward está certa, não apenas de que a rede converge.

## O que deliberadamente não está aqui

XLA/MLIR, grafos estáticos e serialização, convoluções, RNNs, paralelismo de
dados, quantização, SavedModel/protobuf. E, no backend de GPU, autodiff: a fita
vive na CPU, e o caminho de dispositivo cobre a MLP com retropropagação
explícita. Para produção em Rust,
use [`candle`](https://github.com/huggingface/candle) ou
[`burn`](https://github.com/tracel-ai/burn); para rodar o TensorFlow real por
baixo, a crate `tensorflow` (bindings FFI para `libtensorflow`).
