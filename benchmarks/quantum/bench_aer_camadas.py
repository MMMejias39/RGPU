"""Circuito em camadas no Qiskit Aer, idêntico ao de `rqubit/examples/bench_fusao.rs`.

Uso: bench_aer_camadas.py [qubits] [camadas] [on|off] [single|double]

Rotações RY e de fase em todos os qubits, depois CNOTs em cadeia. É o formato
de algoritmo variacional, e é onde a fusão de portas tem o que fazer.
"""
import sys, time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 24
camadas = int(sys.argv[2]) if len(sys.argv) > 2 else 4
fusao = (sys.argv[3].lower() == "on") if len(sys.argv) > 3 else True
precisao = sys.argv[4] if len(sys.argv) > 4 else "single"
# Limite de qubits por unitária fundida. O padrão do Aer é 5; varremos para dar
# a ele a mesma chance de ajuste que damos ao rqubit.
max_fundido = int(sys.argv[5]) if len(sys.argv) > 5 else 5

sim = AerSimulator(
    method="statevector", device="CPU", fusion_enable=fusao,
    precision=precisao, fusion_max_qubit=max_fundido,
)

qc = QuantumCircuit(qubits)
for camada in range(camadas):
    for q in range(qubits):
        qc.ry(0.19 + camada * 0.07 + q * 0.023, q)
        qc.p(0.11 + q * 0.017, q)
    for q in range(qubits - 1):
        # No rqubit, `duas(cnot, q, q+1)` tem q+1 como controle e q como alvo.
        qc.cx(q + 1, q)
portas = qc.size()
qc.save_statevector()
qc = transpile(qc, sim, optimization_level=0)

sim.run(qc, shots=1).result()          # aquece
t0 = time.perf_counter()
sim.run(qc, shots=1).result()
dt = time.perf_counter() - t0

print(
    "motor=aer-CPU|prec=%s|fusao=%s|max=%d|qubits=%d|portas=%d|ms=%.3f"
    % (precisao, "on" if fusao else "off", max_fundido, qubits, portas, dt * 1e3)
)
