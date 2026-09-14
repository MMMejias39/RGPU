"""Isola o custo de despacho Python/cuquantum do custo real de kernel, no
padrão exato de `bench_custatevec.py`: mesma chamada `cusv.apply_matrix`,
mesmo laço, mesma sincronização única no fim. A diferença é só o tamanho do
estado — pequeno o bastante para o kernel ser desprezível, ou o tamanho real
medido no README.

Uso: despacho_custatevec.py [qubits] [portas]
"""
import sys, time

import cupy as cp
import numpy as np
from cuquantum.bindings import custatevec as cusv
from cuquantum import cudaDataType, ComputeType

qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 22
portas = int(sys.argv[2]) if len(sys.argv) > 2 else 100

dtype = cp.complex64
tipo_sv = cudaDataType.CUDA_C_32F
tipo_calc = ComputeType.COMPUTE_32F


def medir(qubits, portas):
    n = 1 << qubits
    sv = cp.zeros(n, dtype=dtype)
    sv[0] = 1.0
    handle = cusv.create()
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

    aplicar()
    t0 = time.perf_counter()
    aplicar()
    dt = time.perf_counter() - t0
    cusv.destroy(handle)
    return dt / portas


ms_real = medir(qubits, portas) * 1e3
# Estado mínimo: o kernel de aplicar a porta é desprezível (poucos
# amplitudes), então o tempo que sobra é quase só despacho Python/cuquantum.
ms_trivial = medir(4, portas) * 1e3

print("motor=custatevec|qubits=%d|real_ms_por_porta=%.4f|trivial_ms_por_porta=%.4f|despacho_pct=%.1f"
      % (qubits, ms_real, ms_trivial, 100.0 * ms_trivial / ms_real))
