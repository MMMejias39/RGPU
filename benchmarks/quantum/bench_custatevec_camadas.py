"""Circuito em camadas no cuStateVec, idêntico ao de `bench_fusao.rs`.

Uso: bench_custatevec_camadas.py [qubits] [camadas]

O cuStateVec é uma API de baixo nível: não funde portas, porque isso é
responsabilidade de quem chama. Serve como a linha de base de GPU sem fusão.
"""
import sys, time

import cupy as cp
import numpy as np
from cuquantum.bindings import custatevec as cusv
from cuquantum import cudaDataType, ComputeType

qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 24
camadas = int(sys.argv[2]) if len(sys.argv) > 2 else 4

dtype, tipo_sv, tipo_calc = cp.complex64, cudaDataType.CUDA_C_32F, ComputeType.COMPUTE_32F
sv = cp.zeros(1 << qubits, dtype=dtype)
sv[0] = 1.0
handle = cusv.create()

# Mesmo circuito: RY e fase em todos os qubits, depois CNOTs em cadeia.
ops = []
for camada in range(camadas):
    for q in range(qubits):
        t = 0.19 + camada * 0.07 + q * 0.023
        c, s = np.cos(t / 2), np.sin(t / 2)
        ops.append((cp.asarray(np.array([[c, -s], [s, c]], dtype=dtype)), [q], []))
        f = 0.11 + q * 0.017
        ops.append((
            cp.asarray(np.array([[1, 0], [0, np.exp(1j * f)]], dtype=dtype)), [q], []
        ))
    for q in range(qubits - 1):
        # CNOT com controle q+1 e alvo q: matriz X no alvo, controlada.
        ops.append((cp.asarray(np.array([[0, 1], [1, 0]], dtype=dtype)), [q], [q + 1]))

def rodar():
    for m, alvos, controles in ops:
        cusv.apply_matrix(
            handle, sv.data.ptr, tipo_sv, qubits,
            m.data.ptr, tipo_sv, cusv.MatrixLayout.ROW, 0,
            alvos, len(alvos), controles, [], len(controles), tipo_calc, 0, 0,
        )
    cp.cuda.Stream.null.synchronize()

rodar()
t0 = time.perf_counter()
rodar()
dt = time.perf_counter() - t0
cusv.destroy(handle)

print("motor=custatevec|qubits=%d|camadas=%d|portas=%d|ms=%.3f"
      % (qubits, camadas, len(ops), dt * 1e3))
