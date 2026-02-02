# Max Flow / Min Cut Project Setup Guide

This document outlines the components and setup required for a new QAOA-based max flow/min cut experiment project.

---

## 1. Project Structure

### Recommended Directory Layout

```
qiskit-maxflow/
├── README.md                    # Project overview and usage
├── requirements.txt             # Python dependencies
├── docs/
│   └── CODE_ANNOTATION_GUIDE.md # Code documentation standards (copy from TSP project)
├── src/
│   ├── __init__.py
│   ├── main.py                  # Main entry point
│   ├── problem.py               # Graph generation and problem definition
│   ├── classical.py             # Classical max flow / min cut solvers
│   ├── qaoa_run.py              # QAOA implementation for max flow/min cut
│   ├── diagnose_qaoa.py         # QP/Ising diagnostics (reusable concept)
│   ├── report.py                # Result reporting and visualization
│   └── runtime_run.py           # Optional: runtime analysis utilities
├── results/                     # Output directory (gitignored)
│   ├── classical_baseline.json  # Classical optimal solution
│   ├── results.json             # QAOA results
│   ├── results.csv              # Tabular results
│   └── *.png                    # Visualization outputs
├── TECHNICAL_WALKTHROUGH.md     # Mathematical and technical details
├── EXPLANATION_GUIDE.md         # High-level explanations for presentations
└── visualize_*.py               # Visualization scripts (if needed)
```

---

## 2. Dependencies

### Core Requirements (`requirements.txt`)

```
# Quantum computing
qiskit>=1.0.0
qiskit-aer>=0.14.0
qiskit-algorithms>=0.3.0
qiskit-optimization>=0.6.0

# Graph algorithms (for classical baseline)
networkx>=3.0

# Numerical computing
numpy>=1.24.0
pandas>=2.0

# Optional: visualization
matplotlib>=3.7.0  # If visualization scripts needed
```

**Notes:**
- NetworkX includes max flow algorithms: `nx.maximum_flow()`, `nx.minimum_cut()`
- Qiskit Optimization handles QUBO conversion and QAOA execution
- Same dependency set as TSP project - can reuse structure

---

## 3. Core Modules

### 3.1 `src/problem.py` - Problem Definition

**Purpose:** Graph generation, problem instance creation, validation

**Key Components:**
- Graph generation utilities
  - Directed graph construction with capacities
  - Source/sink node designation
  - Capacity validation
- Problem instance representation
  - Graph structure (NetworkX `DiGraph`)
  - Source and sink node indices
  - Edge capacities
- Validation functions
  - Graph connectivity checks
  - Capacity positivity
  - Source/sink validity

**Example Structure:**
```python
@dataclass(frozen=True)
class FlowInstance:
    graph: nx.DiGraph
    source: int
    sink: int
    capacities: Dict[Tuple[int, int], float]
    
def generate_test_graph(...) -> FlowInstance:
    """Generate a test max flow problem instance."""
    
def validate_flow_instance(instance: FlowInstance) -> dict:
    """Validate graph structure and capacities."""
```

---

### 3.2 `src/classical.py` - Classical Solvers

**Purpose:** Exact classical solutions for ground truth comparison

**Key Functions:**
- `max_flow_edmonds_karp()` or `max_flow_dinic()` - exact max flow
  - Use NetworkX: `nx.maximum_flow()`, `nx.maximum_flow_value()`
- `min_cut_value()` - min cut value (equals max flow by max-flow min-cut theorem)
- `min_cut_edges()` - actual min cut edges

**Why Both?**
- Max flow algorithm finds the maximum flow value
- Min cut finds the cut with minimum capacity
- By the max-flow min-cut theorem, these are equal
- For QAOA encoding, min cut may be more natural (binary edge variables)

**Example Structure:**
```python
def max_flow_classical(
    graph: nx.DiGraph, 
    source: int, 
    sink: int
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """
    Solve max flow using classical algorithm.
    
    Returns:
        Tuple of (max_flow_value, flow_dict) where flow_dict maps 
        (u, v) -> flow value on edge (u, v).
    """
    
def min_cut_classical(
    graph: nx.DiGraph, 
    source: int, 
    sink: int
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Solve min cut using classical algorithm.
    
    Returns:
        Tuple of (min_cut_value, cut_edges) where cut_edges is list 
        of (u, v) tuples forming the min cut.
    """
```

---

### 3.3 `src/qaoa_run.py` - QAOA Implementation

**Purpose:** Quantum encoding and QAOA execution

**Key Functions:**
- `build_maxflow_qp()` or `build_mincut_qp()` - convert problem to QuadraticProgram
  - **Encoding choice:** Max flow or min cut?
  - **Min cut encoding:** Binary variables for each edge (0 = in cut, 1 = not in cut)
    - Objective: Minimize sum of capacities of cut edges
    - Constraint: Cut separates source from sink
  - **Max flow encoding:** More complex (flow conservation constraints)
- `solve_mincut_qaoa_sampling()` - QAOA sampling approach
  - Similar to TSP: sample from parameterized circuits
  - Decode bitstrings to cuts
  - Validate cuts (separate source from sink)
  - Score cut capacity

**Encoding Considerations:**

**Option A: Min Cut Encoding (Recommended)**
- Binary variable `x_e ∈ {0,1}` for each edge `e`
- `x_e = 1` means edge `e` is in the cut
- Objective: `minimize Σ(capacity_e · x_e)`
- Constraints: Cut must separate source `s` from sink `t`
  - Requires: All paths from `s` to `t` have at least one edge with `x_e = 1`
  - **Challenge:** This is an exponential constraint (must check all paths)
  - **Workaround:** Use penalty-based QUBO encoding or alternative formulation

**Option B: Alternative Min Cut Formulation**
- Use node variables instead: `y_v ∈ {0,1}` means node `v` is in source side
- `y_s = 0`, `y_t = 1` (fixed)
- Edge `(u,v)` is cut if `y_u ≠ y_v`
- Objective: `minimize Σ(capacity_{u,v} · (y_u ⊕ y_v))`
- Simpler constraints: Just fix source and sink

**Result Data Structure:**
```python
@dataclass(frozen=True)
class MaxFlowQaoaResult:
    bitstring: str
    x: List[int]
    fval: float
    status: str
    cut_edges: Optional[List[Tuple[int, int]]]  # Decoded cut edges
    cut_value: Optional[float]  # Sum of capacities of cut edges
    is_valid: bool  # Does cut separate source from sink?
    per_param_stats: Optional[List[Dict]] = None
```

---

### 3.4 `src/diagnose_qaoa.py` - Diagnostics

**Purpose:** Analyze QuadraticProgram complexity

**Can reuse concept from TSP project:**
- QP variable count
- Constraint count
- Objective term count (linear vs quadratic)
- QUBO conversion time
- Ising operator size (qubits, Pauli terms)

**Implementation:** Nearly identical to TSP version - just different QP input.

---

### 3.5 `src/main.py` - Main Entry Point

**Purpose:** Orchestrate the experiment pipeline

**Workflow:**
1. Generate or load a test graph
2. Validate problem instance
3. Solve classically (max flow / min cut) → ground truth
4. Build QP for QAOA
5. Run QAOA sampling (or optimization)
6. Compare results
7. Save ground truth and QAOA results

**Ground Truth Structure:**
```json
{
  "problem": "MAXFLOW-TEST-GRAPH",
  "graph": {
    "num_nodes": 5,
    "num_edges": 7,
    "source": 0,
    "sink": 4
  },
  "classical_max_flow_value": 10.0,
  "classical_min_cut_value": 10.0,
  "classical_min_cut_edges": [[0, 1], [2, 4]],
  "search_space_size": 128  // 2^num_edges for min cut
}
```

---

### 3.6 `src/report.py` - Results and Visualization

**Purpose:** Generate reports and visualizations

**Functions:**
- Save results to JSON/CSV
- Visualize graph with flow/cut highlighted
- Plot QAOA performance (gap vs optimal, valid cut rate, etc.)

**Visualizations:**
- Graph with min cut edges highlighted (red) and flow paths (green)
- QAOA cut capacity vs optimal (bar chart or scatter)
- Valid vs invalid cuts over parameter sweeps

---

## 4. Problem-Specific Considerations

### 4.1 Graph Size

**Recommendation:** Start small
- 5-10 nodes, 10-20 edges
- Allows exhaustive enumeration for ground truth
- Search space: 2^num_edges (if using edge-based min cut encoding)
- Example: 10 edges = 1024 possible cuts

### 4.2 Encoding Choice

**Min Cut vs Max Flow:**
- **Min Cut:** Binary edge/node variables → natural QUBO
- **Max Flow:** Continuous flow variables + conservation → harder for QUBO
- **Recommendation:** Start with **min cut encoding** using node variables

### 4.3 Test Graph Generation

**Options:**
1. **Simple hand-crafted graph** (good for validation)
   - Example: 5 nodes, clear source/sink, obvious min cut
2. **Random directed graph** (good for generalization tests)
   - Use NetworkX: `nx.gnp_random_graph()` then make directed
   - Assign random capacities
3. **Well-known examples** (benchmark)
   - Small examples from literature
   - Known optimal values for validation

---

## 5. Reusable Concepts from TSP Project

### ✅ Can Reuse Conceptually (Adapt to Max Flow):
- **QAOA sampling strategy** - Parameter sweeps instead of optimization
- **Hybrid architecture** - Quantum sampling + classical validation
- **Result data structures** - Similar structure, different content
- **Diagnostics** - QP/Ising analysis
- **Ground truth saving** - Same pattern, different fields
- **Regression checks** - Verify classical baseline doesn't change

### ❌ Must Rebuild (Problem-Specific):
- **Problem encoding** - Completely different (cuts vs tours)
- **Classical solver** - Network flow algorithms (not permutation brute force)
- **Validation logic** - Check cuts separate source/sink (not permutation validity)
- **Visualization** - Graph flows/cuts (not tour paths)
- **Problem instance generation** - Graphs with capacities (not cities/distances)

---

## 6. Setup Checklist

### Phase 1: Project Initialization
- [ ] Create new project directory
- [ ] Initialize git repository (if using version control)
- [ ] Copy `docs/CODE_ANNOTATION_GUIDE.md` from TSP project (if following same standards)
- [ ] Create `requirements.txt` with dependencies
- [ ] Create `src/` directory with `__init__.py`
- [ ] Create `results/` directory (add to `.gitignore`)

### Phase 2: Core Implementation
- [ ] Implement `src/problem.py`
  - [ ] Graph data structures (`FlowInstance` dataclass)
  - [ ] Test graph generation
  - [ ] Validation functions
- [ ] Implement `src/classical.py`
  - [ ] Max flow solver (NetworkX)
  - [ ] Min cut solver (NetworkX)
  - [ ] Ground truth computation
- [ ] Implement `src/diagnose_qaoa.py` (copy from TSP or recreate)

### Phase 3: Quantum Implementation
- [ ] Implement `src/qaoa_run.py`
  - [ ] QP builder for min cut (or max flow) encoding
  - [ ] QAOA sampling function
  - [ ] Bitstring decoding to cuts
  - [ ] Cut validation (separates source from sink)
  - [ ] Result data structure (`MaxFlowQaoaResult`)

### Phase 4: Integration
- [ ] Implement `src/main.py`
  - [ ] Problem generation
  - [ ] Classical baseline
  - [ ] QAOA execution
  - [ ] Comparison and reporting
  - [ ] Ground truth persistence
- [ ] Implement `src/report.py` (optional, for advanced reporting)

### Phase 5: Testing and Validation
- [ ] Test classical solver on known examples
- [ ] Verify max flow = min cut (max-flow min-cut theorem)
- [ ] Test QP encoding on small graph
- [ ] Verify QAOA bitstring decoding
- [ ] Run end-to-end experiment
- [ ] Validate results against ground truth

### Phase 6: Documentation
- [ ] Write `README.md` with setup and usage
- [ ] Create `TECHNICAL_WALKTHROUGH.md` (after implementation, documenting the math)
- [ ] Create `EXPLANATION_GUIDE.md` (for presentations/demos)

---

## 7. Key Differences from TSP Project

| Aspect | TSP Project | Max Flow/Min Cut Project |
|--------|-------------|-------------------------|
| **Problem Type** | Permutation/ordering | Graph cut/flow |
| **Encoding** | Positional variables `y_{i,p}` | Edge/node variables `x_e` or `y_v` |
| **Constraints** | Each city once, each position once | Cut separates source from sink |
| **Classical Solver** | Brute force permutation (O(n!)) | Network flow algorithm (polynomial) |
| **Ground Truth** | Optimal tour length | Max flow value = min cut value |
| **Validation** | Check permutation + tour length | Check cut separates s from t + cut capacity |
| **Visualization** | Tour paths on map | Graph with cut edges/flow paths |

---

## 8. First Steps

1. **Create project structure** with empty modules
2. **Implement a simple test graph** in `problem.py` (e.g., 5 nodes, clear source/sink)
3. **Solve it classically** in `classical.py` - verify you get expected result
4. **Design the QP encoding** - decide on min cut with node variables
5. **Build QP** in `qaoa_run.py` - convert graph to QuadraticProgram
6. **Run QAOA** - sample with parameter sweeps
7. **Decode and validate** - convert bitstrings to cuts, check validity
8. **Compare** - gap between QAOA cut and optimal

---

## 9. Questions to Resolve

Before starting implementation, consider:

1. **Which encoding?** Min cut (recommended) or max flow?
2. **Node or edge variables?** Node variables (`y_v`) are simpler for constraints
3. **Test graph?** Hand-crafted, random, or known example?
4. **Problem size?** Start with 5-10 nodes for tractable search space
5. **QAOA strategy?** Sampling (like TSP) or optimization?
6. **Validation metric?** Gap to optimal, valid cut rate, etc.

---

## 10. References

### Max Flow / Min Cut Theory
- Max-flow min-cut theorem: Max flow = min cut
- Classical algorithms: Edmonds-Karp, Dinic, Push-Relabel
- NetworkX implementation: `nx.maximum_flow()`, `nx.minimum_cut()`

### QAOA for Graph Problems
- Max Cut is well-studied (this is similar but directed)
- Min Cut can be encoded as QUBO with appropriate constraints
- Node-based encoding avoids exponential path constraints

---

**Ready to start!** Once you have the project created and basic structure in place, we can dive into implementation details.