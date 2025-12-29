from __future__ import annotations
import networkx as nx
import numpy as np

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def solve_maxcut_qaoa(g: nx.Graph, reps: int = 2, shots: int = 2048, seed: int = 7):
    # Build the MaxCut problem -> QUBO -> Ising internally via Qiskit Optimization
    maxcut = Maxcut(g)
    qp = maxcut.to_quadratic_program()

    sampler = Sampler(options={"shots": shots, "seed": seed})
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=200), reps=reps)

    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)

    # result.x is array of 0/1
    bitstring = "".join(str(int(b)) for b in result.x)
    # Maxcut app defines objective (may be negative depending on formulation); compute cut directly
    return {
        "bitstring": bitstring,
        "fval": float(result.fval),
        "x": [int(b) for b in result.x],
        "status": str(result.status),
    }

