"""Mesma carga do rqubit, no cuStateVec da NVIDIA.

Uso: bench_custatevec.py [qubits] [portas] [single|double]

Aplica rotações RY com ângulos distintos percorrendo os qubits em rodízio —
as mesmas do benchmark do Aer e do rqubit.
"""
import sys, time

import cupy as cp
import numpy as np
from cuquantum.bindings import custatevec as cusv
from cuquantum import cudaDataType, ComputeType

qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 24
portas = int(sys.argv[2]) if len(sys.argv) > 2 else 100
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

n = 1 << qubits
sv = cp.zeros(n, dtype=dtype)
sv[0] = 1.0

handle = cusv.create()

# Uma matriz RY por porta, todas pré-carregadas para o dispositivo: montar a
# matriz não faz parte do que se quer medir.
matrizes = []
for i in range(portas):
    t = 0.21 + i * 0.017
    c, s = np.cos(t / 2), np.sin(t / 2)
    matrizes.append(cp.asarray(np.array([[c, -s], [s, c]], dtype=dtype)))

def aplicar():
    for i in range(portas):
        alvo = [i % qubits]
        cusv.apply_matrix(
            handle, sv.data.ptr, tipo_sv, qubits,
            matrizes[i].data.ptr, tipo_sv, cusv.MatrixLayout.ROW, 0,
            alvo, 1, [], [], 0, tipo_calc, 0, 0,
        )
    cp.cuda.Stream.null.synchronize()

aplicar()  # aquece
t0 = time.perf_counter()
aplicar()
dt = time.perf_counter() - t0

cusv.destroy(handle)

bytes_por_porta = n * 2 * bytes_amp
print(
    "motor=custatevec|prec=%s|qubits=%d|ms_por_porta=%.4f|GBs=%.1f"
    % (precisao, qubits, dt / portas * 1e3, bytes_por_porta / (dt / portas) / 1e9)
)
