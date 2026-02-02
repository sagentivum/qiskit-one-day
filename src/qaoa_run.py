from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_optimization.applications import Tsp
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


class AerTranspiler:
    """Transpiler wrapper to decompose circuits without backend constraints."""
    
    def run(self, circuits, **kwargs):
        """Transpile circuits for gate decomposition only (no backend coupling_map)."""
        # Decompose gates without backend constraints - use modern basis gates
        # Conservative basis that Aer supports and forces unrolling of composite gates
        basis_gates = ["rz", "rx", "ry", "x", "sx", "cx", "id", "measure"]
        return transpile(circuits, basis_gates=basis_gates, optimization_level=0, **kwargs)


@dataclass(frozen=True)
class TspQaoaResult:
    """
    Result from QAOA-based TSP solver.
    
    Attributes:
        bitstring: Binary solution string from Qiskit optimization.
        x: List representation of the binary solution.
        fval: Objective function value from the optimizer.
        status: Optimization status string.
        tour: Decoded tour as list of city indices, or None if infeasible.
        tour_length_km: Total tour length in kilometers, or None if tour is infeasible.
        per_param_stats: Optional list of dicts with per-parameter-set statistics.
                        Each dict has keys: 'param_set', 'best_length', 'valid_count', 'running_best'.
    """
    bitstring: str
    x: List[int]
    fval: float
    status: str
    tour: Optional[List[int]]
    tour_length_km: Optional[float]
    per_param_stats: Optional[List[Dict]] = None


def _build_complete_graph_from_distance_matrix(
    D: np.ndarray, city_names: Optional[List[str]] = None
) -> nx.Graph:
    """
    Build a complete weighted undirected graph for TSP from a symmetric distance matrix.
    
    Args:
        D: Symmetric distance matrix (NxN) with zeros on diagonal.
        city_names: Optional list of city names for node labels.
    
    Returns:
        Complete weighted undirected NetworkX graph with edge weights in kilometers.
    """
    n = D.shape[0]
    g = nx.Graph()

    for i in range(n):
        g.add_node(i, name=(city_names[i] if city_names else str(i)))

    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, weight=float(D[i, j]))

    return g


def build_tsp_qp_fixed_start(D: np.ndarray, start: int = 0) -> tuple[QuadraticProgram, list[int], list[int]]:
    """
    Build a TSP QuadraticProgram with fixed start city at position 0.
    Variables y_{i,p} for i in cities!=start and p in positions 1..n-1.
    
    Returns:
        Tuple of (qp, rem_cities, rem_positions) where:
        - qp: QuadraticProgram with (n-1)×(n-1) variables for n cities with fixed start
        - rem_cities: List of city indices excluding start
        - rem_positions: List of position indices (1..n-1)
    """
    n = D.shape[0]
    rem_cities = [i for i in range(n) if i != start]  # 7 cities
    rem_positions = list(range(1, n))  # positions 1..7

    qp = QuadraticProgram(name="TSP_fixed_start")

    # Binary vars y_{i,p}: city i is at position p
    for i in rem_cities:
        for p in rem_positions:
            qp.binary_var(name=f"y_{i}_{p}")

    # Constraints:
    # 1) each remaining city appears exactly once
    for i in rem_cities:
        qp.linear_constraint(
            linear={f"y_{i}_{p}": 1 for p in rem_positions},
            sense="==",
            rhs=1,
            name=f"city_once_{i}",
        )

    # 2) each remaining position filled exactly once
    for p in rem_positions:
        qp.linear_constraint(
            linear={f"y_{i}_{p}": 1 for i in rem_cities},
            sense="==",
            rhs=1,
            name=f"pos_once_{p}",
        )

    # Objective: tour length
    # Includes:
    # start -> position 1
    # position p -> p+1 for p=1..n-2
    # position n-1 -> start
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}

    # start -> first (p=1)
    p1 = 1
    for j in rem_cities:
        linear[f"y_{j}_{p1}"] = linear.get(f"y_{j}_{p1}", 0.0) + float(D[start, j])

    # middle edges p -> p+1
    for p in range(1, n - 1):
        if p == n - 1:
            break
        if p + 1 > n - 1:
            break
        for i in rem_cities:
            for j in rem_cities:
                if i == j:
                    continue
                a = f"y_{i}_{p}"
                b = f"y_{j}_{p+1}"
                quadratic[(a, b)] = quadratic.get((a, b), 0.0) + float(D[i, j])

    # last -> start (p = n-1)
    plast = n - 1
    for i in rem_cities:
        linear[f"y_{i}_{plast}"] = linear.get(f"y_{i}_{plast}", 0.0) + float(D[i, start])

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp, rem_cities, rem_positions


def decode_fixed_start_solution(x: list[int], n: int, start: int = 0) -> Optional[List[int]]:
    """
    Decode 49-variable solution back to full 8-city tour.
    
    Args:
        x: Binary solution list of length 49 (7×7)
        n: Total number of cities (8)
        start: Start city index (0)
    
    Returns:
        Full tour as list of city indices, or None if infeasible.
    """
    rem_cities = [i for i in range(n) if i != start]
    rem_positions = list(range(1, n))  # 1..n-1
    expected = len(rem_cities) * len(rem_positions)
    if len(x) != expected:
        return None

    mat = np.array(x, dtype=int).reshape((len(rem_cities), len(rem_positions)))  # (city_idx, pos_idx)

    # each city once
    if not np.all(mat.sum(axis=1) == 1):
        return None
    # each pos once
    if not np.all(mat.sum(axis=0) == 1):
        return None

    tour = [start]
    for pos_col, p in enumerate(rem_positions):
        city_row = int(np.argmax(mat[:, pos_col]))
        tour.append(rem_cities[city_row])
    return tour


def _decode_tsp_bitstring_to_tour(bitstring: str, n: int) -> Optional[List[int]]:
    """
    Decode Qiskit TSP binary solution into a tour.
    
    Qiskit TSP uses binary variables x_{i,p} meaning: city i is visited at position p.
    Decodes a length n*n bitstring into a tour (list of city indices length n).
    
    Args:
        bitstring: Binary string of length n*n representing the solution.
        n: Number of cities.
    
    Returns:
        Tour as list of city indices, or None if infeasible (not exactly one '1' 
        per position and per city).
    """
    if len(bitstring) != n * n:
        return None

    x = np.array([int(b) for b in bitstring], dtype=int).reshape((n, n))  # (city, position)

    # Each city appears exactly once
    if not np.all(x.sum(axis=1) == 1):
        return None

    # Each position filled exactly once
    if not np.all(x.sum(axis=0) == 1):
        return None

    # For each position p, find the city i with x[i,p] = 1
    tour = [int(np.argmax(x[:, p])) for p in range(n)]
    return tour


def _tour_length_km_from_matrix(tour: List[int], D: np.ndarray) -> float:
    """
    Compute total tour length from distance matrix.
    
    Args:
        tour: List of city indices forming a closed tour.
        D: Distance matrix (NxN) in kilometers.
    
    Returns:
        Total tour length in kilometers (includes return to start city).
    """
    total = 0.0
    n = len(tour)
    for k in range(n):
        a = tour[k]
        b = tour[(k + 1) % n]
        total += float(D[a, b])
    return total


def _fix_start_city_constraints(qp, start: int, n: int) -> None:
    """
    Add constraints to fix the start city at position 0, matching classical baseline.
    
    Args:
        qp: QuadraticProgram to modify.
        start: Index of the start city (0 for Sydney).
        n: Number of cities.
    """
    def eq(var_name: str, rhs: int) -> None:
        qp.linear_constraint(linear={var_name: 1}, sense="==", rhs=rhs, name=f"fix_{var_name}_{rhs}")

    # Force start city at position 0
    eq(f"x_{start}_0", 1)
    # Forbid start city at other positions
    for p in range(1, n):
        eq(f"x_{start}_{p}", 0)
    # Forbid other cities at position 0
    for i in range(n):
        if i == start:
            continue
        eq(f"x_{i}_0", 0)


def _qaoa_callback(eval_count: int, params: np.ndarray, mean: float, metadata: dict) -> None:
    """Callback function to log QAOA optimization progress."""
    print(f"  QAOA evaluation {eval_count}: energy = {mean:.4f}")
    # Flush to ensure output appears immediately
    import sys
    sys.stdout.flush()


def solve_tsp_classical_random(
    D: np.ndarray,
    num_samples: int = 1000,
    seed: int = 7,
) -> TspQaoaResult:
    """
    Fast classical random sampling of TSP tours (for comparison with QAOA).
    
    Randomly samples valid tours and returns the best one found.
    This is fast and completes in seconds, useful for baseline comparison.
    
    Args:
        D: Symmetric distance matrix (NxN) with zeros on diagonal, units in kilometers.
        num_samples: Number of random tours to sample (default: 1000).
        seed: Random seed for reproducibility (default: 7).
    
    Returns:
        TspQaoaResult containing the best valid tour found from random sampling.
    """
    n = D.shape[0]
    print(f"Classical random sampling: {num_samples} random tours...")
    
    rng = np.random.RandomState(seed)
    best_tour = None
    best_length = float("inf")
    
    for i in range(num_samples):
        # Generate random permutation of remaining cities (start is fixed at 0)
        nodes = list(range(1, n))
        rng.shuffle(nodes)
        tour = [0] + nodes
        
        length = _tour_length_km_from_matrix(tour, D)
        if length < best_length:
            best_length = length
            best_tour = tour
            if (i + 1) % 100 == 0:
                print(f"  Sampled {i+1}/{num_samples}, best so far: {best_length:.1f} km")
    
    if best_tour is None:
        return TspQaoaResult(
            bitstring="",
            x=[],
            fval=float("inf"),
            status="NO_VALID_SOLUTION",
            tour=None,
            tour_length_km=None,
            per_param_stats=None,
        )
    
    print(f"Best tour found: length = {best_length:.1f} km")
    return TspQaoaResult(
        bitstring="",  # Not applicable for classical sampling
        x=[],  # Not applicable
        fval=best_length,
        status="SUCCESS",
        tour=best_tour,
        tour_length_km=best_length,
        per_param_stats=None,
    )


def solve_tsp_qaoa_sampling(
    D: np.ndarray,
    reps: int = 1,
    shots: int = 512,
    seed: int = 7,
    num_param_sets: int = 30,
    city_names: Optional[List[str]] = None,
) -> TspQaoaResult:
    """
    Solve TSP using QAOA as a parameterized sampler (not energy minimizer).
    
    This avoids expensive energy evaluation by:
    1. Building QAOA ansatz circuit from Ising operator
    2. Sampling bitstrings for a few parameter sets
    3. Decoding and scoring tours classically
    4. Returning the best valid tour
    
    Args:
        D: Symmetric distance matrix (NxN) with zeros on diagonal, units in kilometers.
        reps: Number of QAOA layers (default: 1).
        shots: Number of measurement shots per parameter set (default: 512).
        seed: Random seed for reproducibility (default: 7).
        num_param_sets: Number of random parameter sets to try (default: 30).
        city_names: Optional list of city names for graph construction.

    Returns:
        TspQaoaResult containing the best valid tour found from sampling.
    """
    n = D.shape[0]
    print(f"Building TSP problem for {n} cities with fixed start (reduced to {(n-1)*(n-1)} qubits)...")
    
    # Build reduced (n-1)×(n-1) variable QP
    qp, rem_cities, rem_positions = build_tsp_qp_fixed_start(D, start=0)
    print(f"Quadratic program created with {qp.get_num_vars()} variables")
    
    # Convert to Ising operator
    print("Converting QP to Ising operator...")
    qubo = QuadraticProgramToQubo().convert(qp)
    op, offset = qubo.to_ising()
    print(f"Ising operator: {op.num_qubits} qubits, {len(op.paulis)} Pauli terms")
    
    # Build QAOA ansatz directly from Ising operator
    print(f"Building QAOA ansatz (reps={reps})...")
    import sys
    sys.stdout.flush()
    from qiskit.circuit.library import QAOAAnsatz
    ansatz = QAOAAnsatz(op, reps=reps)
    print(f"Ansatz created: {ansatz.num_qubits} qubits, {ansatz.num_parameters} parameters")
    sys.stdout.flush()
    
    # Initialize sampler
    print(f"Initializing sampler (shots={shots})...")
    sys.stdout.flush()
    sampler = SamplerV2(
        default_shots=shots,
        seed=seed,
        options={
            "backend_options": {
                "coupling_map": None,
                "method": "matrix_product_state",
            },
        },
    )
    sampler._backend.set_options(method="matrix_product_state", coupling_map=None)
    
    # Transpiler for gate decomposition
    transpiler = AerTranspiler()
    
    # Try multiple parameter sets with biased ranges
    # For reps=1: params = [gamma, beta]
    # Typical good ranges: gamma ∈ [0, π], beta ∈ [0, π/2]
    print(f"Sampling {num_param_sets} parameter sets (gamma ∈ [0, π], beta ∈ [0, π/2])...")
    sys.stdout.flush()
    best_tour = None
    best_length = float("inf")
    best_bitstring = None
    best_x = None
    valid_tours_found = 0
    
    # Track per-parameter-set statistics
    per_param_stats = []
    
    num_qubits = qp.get_num_vars()
    rng = np.random.RandomState(seed)
    
    for param_idx in range(num_param_sets):
        # Sample parameters with biased ranges
        # For reps=1: [gamma, beta] where gamma ∈ [0, π], beta ∈ [0, π/2]
        params = np.zeros(2 * reps)
        for i in range(reps):
            # gamma (even indices): [0, π]
            params[2 * i] = rng.uniform(0.0, np.pi)
            # beta (odd indices): [0, π/2]
            params[2 * i + 1] = rng.uniform(0.0, np.pi / 2)
        
        print(f"  Parameter set {param_idx + 1}/{num_param_sets}: building circuit...")
        sys.stdout.flush()
        # Build circuit with parameters
        circuit = ansatz.assign_parameters(params)
        circuit.measure_all()
        
        print(f"    Transpiling circuit...")
        sys.stdout.flush()
        # Transpile
        circuit = transpiler.run([circuit])[0]
        
        print(f"    Running sampler ({shots} shots)...")
        sys.stdout.flush()
        # Sample
        job = sampler.run([circuit], shots=shots)
        print(f"    Waiting for results...")
        sys.stdout.flush()
        result = job.result()
        print(f"    Processing results...")
        sys.stdout.flush()
        # SamplerV2 returns counts in data.meas.get_counts() format
        meas_data = result[0].data.meas
        if hasattr(meas_data, 'get_counts'):
            counts = meas_data.get_counts()
        else:
            # Fallback: convert from bit_array format if needed
            counts = meas_data
        print(f"    Got {len(counts)} unique bitstrings, checking for valid tours...")
        sys.stdout.flush()
        
        # Try each sampled bitstring
        param_valid_count = 0
        for bitstring, count in counts.items():
            # Convert bitstring to binary list
            # Qiskit bitstrings are in measurement order (q0, q1, ..., qn-1)
            if len(bitstring) != num_qubits:
                continue
            x_list = [int(b) for b in bitstring]
            
            # Decode to tour
            tour = decode_fixed_start_solution(x_list, n=n, start=0)
            if tour is not None:
                param_valid_count += 1
                valid_tours_found += 1
                length = _tour_length_km_from_matrix(tour, D)
                if length < best_length:
                    best_length = length
                    best_tour = tour
                    best_bitstring = bitstring
                    best_x = x_list
                    print(f"    Found valid tour: length = {length:.1f} km (new best)")
        
        if param_valid_count > 0:
            print(f"    Parameter set {param_idx + 1}: {param_valid_count} valid tours found")
        
        # Record statistics for this parameter set
        per_param_stats.append({
            'param_set': param_idx + 1,
            'best_length': best_length if best_tour is not None else None,
            'valid_count': param_valid_count,
            'running_best': best_length if best_tour is not None else float('inf'),
        })
    
    if best_tour is None:
        print(f"\nNo valid tours found in {num_param_sets} parameter sets")
        return TspQaoaResult(
            bitstring="",
            x=[],
            fval=float("inf"),
            status="NO_VALID_SOLUTION",
            tour=None,
            tour_length_km=None,
            per_param_stats=per_param_stats,
        )
    
    print(f"\nBest tour found: length = {best_length:.1f} km")
    print(f"Total valid tours found: {valid_tours_found} across {num_param_sets} parameter sets")
    return TspQaoaResult(
        bitstring=best_bitstring or "",
        x=best_x or [],
        fval=best_length,
        status="SUCCESS",
        tour=best_tour,
        tour_length_km=best_length,
        per_param_stats=per_param_stats,
    )


def solve_tsp_qaoa(
    D: np.ndarray,
    reps: int = 1,
    shots: int = 2,
    seed: int = 7,
    maxiter: int = 1,
    city_names: Optional[List[str]] = None,
) -> TspQaoaResult:
    """
    Solve TSP using Qiskit Optimization's Tsp -> QuadraticProgram -> MinimumEigenOptimizer(QAOA).
    
    Args:
        D: Symmetric distance matrix (NxN) with zeros on diagonal, units in kilometers.
        reps: Number of QAOA layers (default: 1).
        shots: Number of measurement shots (default: 2).
        seed: Random seed for reproducibility (default: 7).
        maxiter: Maximum optimizer iterations (default: 1).
        city_names: Optional list of city names for graph construction.

    Returns:
        TspQaoaResult containing raw binary solution (bitstring, x), decoded tour
        (if feasible), and tour length computed from D (if feasible).
    """
    n = D.shape[0]
    print(f"Building TSP problem for {n} cities with fixed start (reduced to {(n-1)*(n-1)} qubits)...")
    
    # Build reduced (n-1)×(n-1) variable QP
    qp, rem_cities, rem_positions = build_tsp_qp_fixed_start(D, start=0)
    print(f"Quadratic program created with {qp.get_num_vars()} variables (should be {(n-1)*(n-1)})")

    print(f"Initializing QAOA (reps={reps}, shots={shots}, maxiter={maxiter})...")
    sampler = SamplerV2(
        default_shots=shots,
        seed=seed,
        options={
            "backend_options": {
                # Critical: avoid a constrained coupling map
                "coupling_map": None,
                # Force matrix_product_state for 49-qubit circuits
                "method": "matrix_product_state",
            },
            "run_options": {
                # nothing required; keep for clarity / future tweaks
            },
        },
    )
    # Ensure backend options are actually applied (Aer quirk workaround)
    sampler._backend.set_options(method="matrix_product_state", coupling_map=None)
    
    # Transpiler needed for gate decomposition (not routing)
    transpiler = AerTranspiler()
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        transpiler=transpiler,
        callback=_qaoa_callback,
    )

    print("Starting QAOA optimization...")
    print("  (This may take several minutes for the first evaluation - MPS on 49 qubits is computationally expensive)")
    import sys
    sys.stdout.flush()
    
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    print("QAOA optimization completed")

    x_list = [int(v) for v in result.x]
    bitstring = "".join(str(v) for v in x_list)

    # Decode 49-variable solution back to full 8-city tour
    tour = decode_fixed_start_solution(x_list, n=n, start=0)
    length = _tour_length_km_from_matrix(tour, D) if tour is not None else None

    return TspQaoaResult(
        bitstring=bitstring,
        x=x_list,
        fval=float(result.fval),
        status=str(result.status),
        tour=tour,
        tour_length_km=length,
        per_param_stats=None,
    )
