# Qiskit TSP-QAOA Experiment

A hybrid quantum-classical implementation of the Traveling Salesman Problem (TSP) using QAOA (Quantum Approximate Optimization Algorithm) with Qiskit.

## Overview

This project demonstrates solving the TSP for 5 Australian capital cities using:
- **Classical brute force** for ground truth (optimal solution)
- **QAOA sampling** as a quantum-classical hybrid approach

**Key result:** QAOA found a tour within 0.3% of optimal (3,811 km vs 3,798 km optimal).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and the following packages:
- qiskit >= 1.0.0
- qiskit-aer >= 0.14.0
- qiskit-algorithms >= 0.3.0
- qiskit-optimization >= 0.6.0
- networkx >= 3.0
- numpy >= 1.24.0
- pandas >= 2.0

## Usage

### Run the TSP Solver

```bash
python -m src.main
```

This will:
1. Select 5 cities: Sydney (fixed start), Melbourne, Canberra, Brisbane, Hobart
2. Compute distances using the haversine formula
3. Solve via brute force (24 possible tours)
4. Solve via QAOA sampling (30 parameter sets × 512 shots)
5. Compare results and save ground truth

### Visualize Results

```bash
# Compare optimal vs QAOA tours on a map
python visualize_tours.py

# Plot QAOA convergence statistics
python visualize_qaoa_stats.py
```

## Project Structure

```
├── src/
│   ├── main.py           # Main entry point
│   ├── problem.py        # City data and distance matrix
│   ├── classical.py      # Brute force TSP solver
│   ├── qaoa_run.py       # QAOA implementation
│   └── diagnose_qaoa.py  # QP/Ising diagnostics
├── results/
│   ├── classical_baseline_5_city.json  # Ground truth solution
│   ├── ground_truth.json               # 8-city reference
│   └── *.png                           # Visualization outputs
├── docs/
│   └── CODE_ANNOTATION_GUIDE.md        # Documentation standards
├── TECHNICAL_WALKTHROUGH.md            # Mathematical details
├── EXPLANATION_GUIDE.md                # Talking points for presentations
├── visualize_tours.py                  # Tour comparison visualization
└── visualize_qaoa_stats.py             # QAOA performance plots
```

## How It Works

### Problem Encoding

The TSP is encoded as a Quadratic Program with:
- **16 binary variables** `y_{i,p}`: city `i` at position `p` (4 cities × 4 positions, with Sydney fixed at position 0)
- **Constraints**: Each city appears once, each position filled once
- **Objective**: Minimize total tour length (quadratic in variables)

### Transformation Chain

```
TSP → Quadratic Program → QUBO → Ising Hamiltonian → QAOA Circuit
```

### QAOA as Sampler

Instead of using QAOA for optimization (expensive energy evaluations), we use it as a **parameterized sampler**:
1. Sample bitstrings from QAOA circuits with random parameters
2. Decode bitstrings to tours
3. Validate constraints classically
4. Score valid tours and track the best

This hybrid approach leverages quantum exploration with classical validation.

## Results

| Metric | Value |
|--------|-------|
| Optimal tour length | 3,798.3 km |
| Best QAOA tour | 3,811.1 km |
| Gap | 0.3% |
| Parameter sets tested | 30 |
| Total bitstrings sampled | 15,360 |

## Documentation

- **[TECHNICAL_WALKTHROUGH.md](TECHNICAL_WALKTHROUGH.md)** - Detailed mathematical derivation of the encoding and algorithm
- **[EXPLANATION_GUIDE.md](EXPLANATION_GUIDE.md)** - Talking points for explaining the experiment to different audiences

## Key Insights

1. **Hybrid architecture works**: Quantum sampling + classical validation is effective
2. **Encoding matters**: The QP→QUBO→Ising transformation adds significant overhead
3. **Small scale for validation**: 5 cities allows exact ground truth comparison
4. **No quantum advantage claimed**: This is a proof-of-concept, not a demonstration of quantum speedup

## License

MIT
