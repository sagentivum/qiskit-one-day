from __future__ import annotations

import time
from qiskit_optimization.converters import QuadraticProgramToQubo

def diagnose(qp):
    """
    Diagnose QuadraticProgram complexity to understand computational cost.
    
    Args:
        qp: QuadraticProgram to analyze.
    
    Returns:
        The Ising operator (for further analysis if needed).
    """
    print("=== DIAGNOSE QP ===")
    print("vars:", qp.get_num_vars())
    print("linear constraints:", len(qp.linear_constraints))
    print("objective sense:", qp.objective.sense)

    # Count quadratic terms
    quad_terms = len(qp.objective.quadratic.to_dict())
    lin_terms = len(qp.objective.linear.to_dict())
    print("objective linear terms:", lin_terms)
    print("objective quadratic terms:", quad_terms)

    print("\n=== QUBO + ISING ===")
    t0 = time.time()
    qubo = QuadraticProgramToQubo().convert(qp)
    t1 = time.time()
    op, offset = qubo.to_ising()
    t2 = time.time()

    # Count Pauli terms in the operator
    try:
        n_paulis = op.num_qubits
        n_terms = len(op.paulis)  # SparsePauliOp
        print("ising qubits:", n_paulis)
        print("ising pauli terms:", n_terms)
    except Exception:
        print("operator type:", type(op))

    print(f"QP->QUBO: {t1-t0:.3f}s, QUBO->Ising: {t2-t1:.3f}s")
    return op

