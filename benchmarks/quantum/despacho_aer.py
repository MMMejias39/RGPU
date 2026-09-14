"""Isola o custo fixo de despacho Python/Aer do custo real de kernel, no
padrão exato de `bench_aer.py`: circuito inteiro montado e transpilado uma
vez, um único `sim.run()` medido — diferente do cuStateVec, que chama a API
uma vez por porta. A pergunta é se o `sim.run()` único tem custo fixo
relevante quando dividido pelas `portas` gates.

Uso: despacho_aer.py [qubits] [portas] [CPU|GPU]
"""
import sys, time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 24
portas = int(sys.argv[2]) if len(sys.argv) > 2 else 100
dispositivo = sys.argv[3] if len(sys.argv) > 3 else "CPU"


def medir(qubits, portas):
    sim = AerSimulator(method="statevector", device=dispositivo, fusion_enable=True, precision="single")
    qc = QuantumCircuit(qubits)
    for i in range(portas):
        qc.ry(0.21 + i * 0.017, i % qubits)
    qc.save_statevector()
    qc = transpile(qc, sim, optimization_level=0)
    sim.run(qc, shots=1).result()
    t0 = time.perf_counter()
    sim.run(qc, shots=1).result()
    return (time.perf_counter() - t0) / portas


ms_real = medir(qubits, portas) * 1e3
ms_trivial = medir(4, portas) * 1e3

print("motor=aer-%s|qubits=%d|real_ms_por_porta=%.4f|trivial_ms_por_porta=%.4f|despacho_pct=%.1f"
      % (dispositivo, qubits, ms_real, ms_trivial, 100.0 * ms_trivial / ms_real))
