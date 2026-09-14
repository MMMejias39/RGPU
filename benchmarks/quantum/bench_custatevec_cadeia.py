"""A mesma cadeia XX do cadeia_magnon.rs, no cuStateVec da NVIDIA.

Uso: bench_custatevec_cadeia.py [qubits] [passos] [delta]

Trotter de 1ª ordem: por passo, as ligações pares e depois as ímpares com
exp(-iJδ(XX+YY)/2) — a mesma rotação de Givens em todas as ligações, uma
única matriz pré-carregada no dispositivo.
"""
import sys, time

import cupy as cp
import numpy as np
from cuquantum.bindings import custatevec as cusv
from cuquantum import cudaDataType, ComputeType

qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 27
passos = int(sys.argv[2]) if len(sys.argv) > 2 else 100
J = 1.0
DT = 0.1

dtype = cp.complex64
tipo_sv = cudaDataType.CUDA_C_32F
tipo_calc = ComputeType.COMPUTE_32F

n = 1 << qubits
sv = cp.zeros(n, dtype=dtype)
sv[1 << (qubits // 2)] = 1.0  # magnon no centro

handle = cusv.create()

c, s = np.cos(J * DT), np.sin(J * DT)
ligacao = cp.asarray(np.array([
    [1, 0, 0, 0],
    [0, c, -1j * s, 0],
    [0, -1j * s, c, 0],
    [0, 0, 0, 1],
], dtype=dtype))

pares = [[m, m + 1] for m in range(0, qubits - 1, 2)] + \
        [[m, m + 1] for m in range(1, qubits - 1, 2)]

def passo():
    for alvos in pares:
        cusv.apply_matrix(
            handle, sv.data.ptr, tipo_sv, qubits,
            ligacao.data.ptr, tipo_sv, cusv.MatrixLayout.ROW, 0,
            alvos, 2, [], [], 0, tipo_calc, 0, 0,
        )

def trajetoria(passos):
    for _ in range(passos):
        passo()
    cp.cuda.Stream.null.synchronize()

trajetoria(1)  # aquece
t0 = time.perf_counter()
trajetoria(passos)
dt = time.perf_counter() - t0

cusv.destroy(handle)

portas = (qubits - 1) * passos
print(
    "motor=custatevec|prec=single|qubits=%d|passos=%d|portas=%d|s_por_trajetoria=%.2f|ms_por_porta=%.4f"
    % (qubits, passos, portas, dt, dt / portas * 1e3)
)
