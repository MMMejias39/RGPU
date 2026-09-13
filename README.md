# RGPU

Computação em GPU em **Rust puro**, independente de fabricante — sem CUDA, sem
`unsafe`, sem FFI. Os kernels são escritos em WGSL e executam sobre Vulkan,
Metal, DX12 ou WebGPU, então o mesmo código roda em NVIDIA, AMD, Intel e Apple.

## Crates

| Crate | O que é |
|---|---|
| [`rtensor`](crates/rtensor) | Framework de deep learning: tensor com broadcasting, autodiff reverso, camadas, otimizadores e backend de GPU |

## Estado

O `rtensor` está funcional e medido. Em benchmarks contra o TensorFlow 2.21
numa RTX 4070 Laptop, ele **vence** em lote pequeno, matrizes magras,
inferência de latência única e partida a frio; e **perde** em GEMM quadrado
grande, onde o cuBLAS usa tensor cores.

| Teste | rtensor GPU | TensorFlow GPU |
|---|---|---|
| Treino, lote 1 | **0,89 ms** | 1,93 ms |
| Treino, lote 2048 | 7,30 ms | **2,37 ms** |
| Inferência, lote 1 | **0,49 ms** | 0,88 ms |
| GEMM 8192×64×64 | **952 GFLOP/s** | 104 GFLOP/s |
| GEMM 2048³ | 3.290 GFLOP/s | **11.946 GFLOP/s** |
| Partida a frio | **0,40 s** | 5,64 s |

O padrão é consistente: RGPU ganha onde o trabalho é pequeno, irregular ou
raro, porque paga pouco overhead fixo; perde onde o trabalho é grande e
regular, porque ainda não usa tensor cores.

## Rodando

```bash
cargo test -p rtensor                    # núcleo, sem GPU
cargo test -p rtensor --features gpu     # + testes CPU vs GPU
cargo run -p rtensor --release --features gpu --example perfil   # perfilamento por kernel
```

## Licença

MIT.
