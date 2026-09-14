"""A mesma QFT do bench_qft.rs, no cuStateVec da NVIDIA.

Uso: bench_custatevec_qft.py [qubits] [reps] [single|double]

Mesma sequência de portas: H em cada qubit, CP(2π/2^(k−j)) entre todos os
pares com k > j, e a reversão final. As matrizes vão pré-carregadas para o
dispositivo — montá-las não faz parte do que se quer medir.
"""
import sys, time

import cupy as cp
import numpy as np
from cuquantum.bindings import custatevec as cusv
from cuquantum import cudaDataType, ComputeType

qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 26
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
precisao = sys.argv[3] if len(sys.argv) > 3 else "single"

if precisao == "single":
    dtype = cp.complex64
    tipo_sv = cudaDataType.CUDA_C_32F
    tipo_calc = ComputeType.COMPUTE_32F
    bytes_amp = 8
else:
    dtype = cp.complex128
    tipo_sv = cudaDataType.CUDA_C_64F
    tipo_calc = ComputeType.COMPUTE_64F
    bytes_amp = 16

TAU = 2 * np.pi
n = 1 << qubits
sv = cp.zeros(n, dtype=dtype)
sv[0] = 1.0

handle = cusv.create()

# Uma matriz por ângulo distinto: H, e CP(2π/2^d) para d = 1..qubits−1, e a
# SWAP. O circuito referencia-as por índice, sem construir nada no laço.
h = cp.asarray(np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2))
cp_mats = {}
for d in range(1, qubits):
    m = np.eye(4, dtype=dtype)
    m[3, 3] = np.exp(1j * TAU / 2**d)
    cp_mats[d] = cp.asarray(m)
swap = cp.asarray(np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=dtype))

def portas_qft(n_q):
    # (op, q0, q1, matriz) — a mesma sequência do bench_qft.rs
    seq = []
    for j in range(n_q):
        seq.append(("h", j, None, h))
        for k in range(j + 1, n_q):
            seq.append(("cp", k, j, cp_mats[k - j]))
    for q in range(n_q // 2):
        seq.append(("swap", q, n_q - 1 - q, swap))
    return seq

portas = portas_qft(qubits)

def aplicar():
    for op, q0, q1, m in portas:
        if op == "h":
            cusv.apply_matrix(
                handle, sv.data.ptr, tipo_sv, qubits,
                m.data.ptr, tipo_sv, cusv.MatrixLayout.ROW, 0,
                [q0], 1, [], [], 0, tipo_calc, 0, 0,
            )
        else:
            cusv.apply_matrix(
                handle, sv.data.ptr, tipo_sv, qubits,
                m.data.ptr, tipo_sv, cusv.MatrixLayout.ROW, 0,
                [q0, q1], 2, [], [], 0, tipo_calc, 0, 0,
            )
    cp.cuda.Stream.null.synchronize()

aplicar()  # aquece
if reps == 0:
    cusv.destroy(handle)
    print("motor=custatevec|qubits=%d|reps=0|apenas_partida" % qubits)
    sys.exit(0)

t0 = time.perf_counter()
for _ in range(reps):
    aplicar()
dt = (time.perf_counter() - t0) / reps

cusv.destroy(handle)

total = len(portas)
por_circuito = dt * 1e3
print(
    "motor=custatevec|prec=%s|qubits=%d|portas=%d|ms_por_circuito=%.1f|ms_por_porta=%.4f"
    % (precisao, qubits, total, por_circuito, por_circuito / total)
)
