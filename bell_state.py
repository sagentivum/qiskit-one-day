"""
Bell state preparation and measurement demonstration.

This module creates a Bell state (quantum entanglement) between two qubits
and measures the result multiple times.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

backend = AerSimulator()
result = backend.run(qc, shots=2000).result()
counts = result.get_counts()
print(counts)

