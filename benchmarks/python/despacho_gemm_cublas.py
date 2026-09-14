"""O teto de cuBLAS citado no README (11.946 GFLOP/s, C[2048³], TF32) foi
medido via `tf.matmul` num laço de 20 chamadas — a mesma metodologia que já
mostrou ~30% de overhead de despacho nos passos de treino. Este script mede
a mesma multiplicação via PyTorch (overhead de despacho quase nulo, já
medido) para checar se o teto real do cuBLAS é mais alto do que o número
citado.

Uso: despacho_gemm_cublas.py
"""
import time
import torch

N = 2048
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

a = torch.randn(N, N, device=dev)
b = torch.randn(N, N, device=dev)

reps = 20
for _ in range(3):
    torch.matmul(a, b)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(reps):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / reps

flop = 2.0 * N * N * N
print("motor=torch-tf32|N=%d|ms=%.4f|gflops=%.1f" % (N, dt * 1e3, flop / dt / 1e9))
