# RGPU

Computação em GPU em **Rust puro**, independente de fabricante — sem CUDA, sem
`unsafe`, sem FFI. Os kernels são escritos em WGSL e executam sobre Vulkan,
Metal, DX12 ou WebGPU, então o mesmo código roda em NVIDIA, AMD, Intel e Apple.

## Crates

| Crate | O que é |
|---|---|
| [`rtensor`](crates/rtensor) | Framework de deep learning: tensor com broadcasting, autodiff reverso, camadas, otimizadores e backend de GPU |
| [`rgpu-power`](crates/rgpu-power) | Medição de energia: potência da GPU por `nvidia-smi`, energia da CPU e da GPU integrada por RAPL |

## Por que energia, e não só tempo

Tempo diz quão rápido; energia diz quanto custou. Um passo duas vezes mais
rápido consumindo três vezes mais potência é um retrocesso de eficiência, e só
a medição separa os dois casos.

Isso importa porque o desperdício costuma estar na **pilha de software**, não
no silício. Dois exemplos medidos neste repositório:

- Na tarefa das espirais, o TensorFlow na RTX 4070 levou **9,29 s**; o
  `rtensor` em **um núcleo de CPU** levou **0,11 s**. A placa de 80 W foi 84×
  mais lenta que um núcleo.
- Num GEMM com `K` dominante, rodamos a 54 GFLOP/s numa placa capaz de
  ~15.000 — **0,4% da capacidade**, com a placa ligada e consumindo.

Uma GPU subutilizada consome quase como uma GPU ocupada. Medir joules por
unidade de trabalho é o que torna esse desperdício visível.

### A medição

Mesmo modelo, mesma placa, mesmo código — só muda quanto trabalho chega por
vez. RTX 4070 Laptop, MLP 512→1024→1024→10, ociosidade de 14,1 W:

| Lote | Potência média | µJ por amostra | GFLOP/J |
|---:|---:|---:|---:|
| 1 | 38,1 W | **12.793,5** | 0,7 |
| 8 | 46,0 W | 2.119,2 | 4,5 |
| 32 | 47,2 W | 599,8 | 15,8 |
| 128 | 48,5 W | 290,7 | 32,7 |
| 512 | 66,8 W | 196,6 | 48,3 |
| 2048 | 69,4 W | **164,3** | 57,8 |

A potência média sobe menos de **2×** entre o lote 1 e o lote 2048 — de 38 W
para 69 W. A vazão sobe **182×**, de 1.827 para 333.412 amostras por segundo.
O resultado é **78× de diferença na energia por amostra treinada**.

O chip custa quase o mesmo ligado, fazendo muito ou pouco trabalho. Quem treina
com lote pequeno nesta placa desperdiça cerca de 98% da energia que gasta — e
nenhuma troca de hardware corrige isso, porque o problema não está no hardware.

```bash
cargo run -p rtensor --release --features gpu --example eficiencia   # a tabela acima
cargo run -p rtensor --release --features gpu --example energia      # joules por motor e por fabricante
```

### Entre fabricantes, o mesmo código

Mesmo WGSL nas duas GPUs, energia acima da ociosidade, lote 512:

| Motor | ms/passo | J/passo | GFLOP/s | GFLOP/J |
|---|---:|---:|---:|---:|
| RTX 4070 Laptop | 1,92 | 0,0874 | 2.535 | **55,64** |
| Intel Arc (iGPU) | 12,69 | 0,1038 | 383 | **46,85** |
| CPU, 1 núcleo | 156,48 | 2,2574 | 31 | **2,15** |

A placa dedicada é **6,6× mais rápida** que a gráfica integrada, e apenas
**19% mais eficiente por joule**. A vantagem da GPU discreta é velocidade,
quase nada é eficiência energética — e ela ainda consome **11,89 W só por
estar ligada**, sem trabalho algum.

O salto de eficiência acontece ao sair da CPU para *qualquer* GPU: a iGPU é
26× mais eficiente que um núcleo de CPU. O salto da integrada para a dedicada
é de velocidade.

*Ressalva:* o `nvidia-smi` mede a placa inteira (GPU, VRAM, regulação); o
domínio `uncore` do RAPL cobre só a iGPU dentro do SoC, sem sua parcela de
controlador de memória e LPDDR. O número da Arc está provavelmente
subestimado, o que reforça a conclusão em vez de enfraquecê-la.

### Limites desta máquina

Varrer o **limite de potência** (`nvidia-smi -pl`) não é possível em GPU de
notebook: o driver responde *"not supported in current scope"*, porque quem
controla o envelope é o firmware do fabricante, com o Dynamic Boost ativo.
`sudo` não contorna isso. Já **travar o clock** (`nvidia-smi -lgc`) falha por
permissão, e portanto funciona com `sudo` — é o caminho para varrer frequência.

A energia da CPU e da GPU integrada vem dos contadores RAPL, restritos a `root`
desde a mitigação do PLATYPUS (2020).

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
