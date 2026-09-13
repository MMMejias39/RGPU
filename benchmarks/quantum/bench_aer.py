"""Mesma carga do rqubit, no Qiskit Aer.

Uso: bench_aer.py [qubits] [portas] [CPU|GPU]

Aplica `portas` Hadamards percorrendo os qubits em rodízio, exatamente como
`rqubit/examples/bench_porta.rs`, e mede o tempo do simulador — sem o custo de
montar e transpilar o circuito, que não faz parte da conta.
"""
import sys, time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 24
portas = int(sys.argv[2]) if len(sys.argv) > 2 else 100
dispositivo = sys.argv[3] if len(sys.argv) > 3 else "CPU"
# O Aer funde portas consecutivas em unitárias maiores por padrão. É uma
# otimização real e legítima, mas compara o otimizador dele com o nosso kernel,
# não os dois kernels. Medimos os dois casos.
fusao = (sys.argv[4].lower() == "on") if len(sys.argv) > 4 else True
# O Aer usa complex128 por padrão — o dobro dos bytes do nosso complex64. Para
# comparar kernel com kernel é preciso igualar a precisão.
precisao = sys.argv[5] if len(sys.argv) > 5 else "single"

try:
    sim = AerSimulator(
        method="statevector", device=dispositivo,
        fusion_enable=fusao, precision=precisao,
    )
except Exception as e:
    print(f"motor=aer-{dispositivo}|erro={e}")
    sys.exit(0)

# Rotações com ângulos distintos, e não Hadamards: duas Hadamards no mesmo
# qubit se cancelam, e o transpilador as remove — o que mediria o otimizador de
# circuito, não o simulador. Com `optimization_level=0` nada mais é removido.
qc = QuantumCircuit(qubits)
for i in range(portas):
    qc.ry(0.21 + i * 0.017, i % qubits)
qc.save_statevector()
qc = transpile(qc, sim, optimization_level=0)

sim.run(qc, shots=1).result()          # aquece
t0 = time.perf_counter()
r = sim.run(qc, shots=1).result()
dt = time.perf_counter() - t0

# 2ⁿ amplitudes lidas e reescritas por porta.
bytes_por_amplitude = 8 if precisao == "single" else 16
bytes_por_porta = (1 << qubits) * 2 * bytes_por_amplitude
print(
    "motor=aer-%s|prec=%s|fusao=%s|qubits=%d|ms_por_porta=%.4f|GBs=%.1f"
    % (dispositivo, precisao, "on" if fusao else "off", qubits,
       dt / portas * 1e3, bytes_por_porta / (dt / portas) / 1e9)
)
