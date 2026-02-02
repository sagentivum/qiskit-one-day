# Technical Walkthrough: TSP with QAOA

## Overview

This document provides a detailed mathematical and technical walkthrough of the quantum-classical hybrid TSP solver experiment.

---

## 1. Problem Setup

### 1.1 Geographic Data

We selected 5 Australian capital cities:
- **Sydney** (start city, fixed at position 0)
- Melbourne
- Canberra
- Brisbane
- Hobart

Each city has coordinates `(lat, lon)` in decimal degrees.

### 1.2 Distance Matrix Construction

We compute pairwise distances using the **haversine formula** for great-circle distances:

```
R = 6371.0088 km  (mean Earth radius)
φ₁, λ₁ = latitude, longitude of city 1 (in radians)
φ₂, λ₂ = latitude, longitude of city 2 (in radians)

Δφ = φ₂ - φ₁
Δλ = λ₂ - λ₁

a = sin²(Δφ/2) + cos(φ₁) · cos(φ₂) · sin²(Δλ/2)
c = 2 · arcsin(√a)
distance = R · c
```

This gives us a **symmetric distance matrix** `D` where:
- `D[i,j] = D[j,i]` (symmetric)
- `D[i,i] = 0` (zero diagonal)
- Units: kilometers

For 5 cities, `D` is a 5×5 matrix.

### 1.3 Classical Baseline

With Sydney fixed at position 0, we have `(n-1)! = 4! = 24` possible tours to check.

**Brute force algorithm:**
1. Generate all permutations of `[Melbourne, Canberra, Brisbane, Hobart]`
2. For each permutation `π`, construct tour: `[Sydney, π[0], π[1], π[2], π[3]]`
3. Compute tour length: `L = D[Sydney, π[0]] + D[π[0], π[1]] + D[π[1], π[2]] + D[π[2], π[3]] + D[π[3], Sydney]`
4. Return the tour with minimum `L`

**Result:** Optimal tour length = **3,798.3 km**

---

## 2. Quantum Encoding: TSP → Quadratic Program

### 2.1 Positional Encoding with Fixed Start

We use a **positional encoding** where binary variables indicate which city occupies which position.

**Key insight:** Since Sydney is fixed at position 0, we only need variables for:
- Remaining cities: `{1, 2, 3, 4}` (indices of Melbourne, Canberra, Brisbane, Hobart)
- Remaining positions: `{1, 2, 3, 4}` (positions 1-4 in the tour)

**Binary variables:** `y_{i,p}` where:
- `i ∈ {1, 2, 3, 4}` (city index, excluding Sydney at 0)
- `p ∈ {1, 2, 3, 4}` (position in tour, excluding position 0)

`y_{i,p} = 1` means "city `i` is at position `p` in the tour"

**Total variables:** `(n-1) × (n-1) = 4 × 4 = 16` binary variables

### 2.2 Constraints

We need two types of constraints to ensure valid tours:

#### Constraint 1: Each city appears exactly once
For each city `i ∈ {1, 2, 3, 4}`:
```
∑_{p=1}^{4} y_{i,p} = 1
```

This ensures each city (except Sydney) appears in exactly one position.

#### Constraint 2: Each position is filled exactly once
For each position `p ∈ {1, 2, 3, 4}`:
```
∑_{i=1}^{4} y_{i,p} = 1
```

This ensures each position (except 0) is occupied by exactly one city.

### 2.3 Objective Function: Tour Length

The tour length is the sum of edge distances. With Sydney fixed at position 0:

**Edges in the tour:**
1. **Start → Position 1:** `D[0, j]` where city `j` is at position 1
   - Contribution: `∑_{j=1}^{4} D[0, j] · y_{j,1}`

2. **Position p → Position p+1:** For `p ∈ {1, 2, 3}`
   - If city `i` is at position `p` and city `j` is at position `p+1`, add `D[i, j]`
   - Contribution: `∑_{i=1}^{4} ∑_{j=1}^{4} D[i, j] · y_{i,p} · y_{j,p+1}` (quadratic terms)

3. **Position 4 → Start:** `D[i, 0]` where city `i` is at position 4
   - Contribution: `∑_{i=1}^{4} D[i, 0] · y_{i,4}`

**Complete objective:**
```
minimize:  ∑_{j=1}^{4} D[0, j] · y_{j,1}
         + ∑_{p=1}^{3} ∑_{i=1}^{4} ∑_{j=1}^{4} D[i, j] · y_{i,p} · y_{j,p+1}
         + ∑_{i=1}^{4} D[i, 0] · y_{i,4}
```

This is a **quadratic objective** with:
- **Linear terms:** `y_{j,1}` and `y_{i,4}` (start/end edges)
- **Quadratic terms:** `y_{i,p} · y_{j,p+1}` (middle edges)

### 2.4 Quadratic Program Formulation

The complete Quadratic Program (QP) is:

```
minimize:  c^T y + y^T Q y
subject to:
  ∑_{p=1}^{4} y_{i,p} = 1  ∀i ∈ {1,2,3,4}    (city constraints)
  ∑_{i=1}^{4} y_{i,p} = 1  ∀p ∈ {1,2,3,4}    (position constraints)
  y_{i,p} ∈ {0, 1}                            (binary variables)
```

Where:
- `y` is a 16-dimensional binary vector (flattened `y_{i,p}`)
- `c` is a 16-dimensional vector of linear coefficients
- `Q` is a 16×16 symmetric matrix of quadratic coefficients

---

## 3. QP → QUBO Conversion

### 3.1 Constraint Penalty Method

To convert the constrained QP to an unconstrained QUBO, we use **penalty terms**.

For each constraint, we add a penalty that is zero when satisfied and large when violated:

**City constraint penalty** (for city `i`):
```
P_city(i) = λ · (∑_{p=1}^{4} y_{i,p} - 1)²
```

**Position constraint penalty** (for position `p`):
```
P_pos(p) = λ · (∑_{i=1}^{4} y_{i,p} - 1)²
```

Where `λ` is a large penalty coefficient (typically chosen to dominate the objective).

**Expanding the squared terms:**
```
(∑_{p} y_{i,p} - 1)² = (∑_{p} y_{i,p})² - 2·∑_{p} y_{i,p} + 1
                     = ∑_{p} y_{i,p}² + 2·∑_{p<q} y_{i,p}·y_{i,q} - 2·∑_{p} y_{i,p} + 1
```

Since `y_{i,p}² = y_{i,p}` (binary variables), this becomes:
```
= ∑_{p} y_{i,p} + 2·∑_{p<q} y_{i,p}·y_{i,q} - 2·∑_{p} y_{i,p} + 1
= -∑_{p} y_{i,p} + 2·∑_{p<q} y_{i,p}·y_{i,q} + 1
```

The constant term `+1` can be dropped (doesn't affect optimization).

### 3.2 QUBO Form

The QUBO (Quadratic Unconstrained Binary Optimization) problem is:

```
minimize:  y^T Q_qubo y
subject to:  y ∈ {0, 1}¹⁶
```

Where `Q_qubo` combines:
- Original objective coefficients (from distance matrix)
- Penalty terms from constraints (linear and quadratic)

**Key property:** Valid solutions (satisfying all constraints) have low QUBO values. Invalid solutions have high penalty terms.

---

## 4. QUBO → Ising Hamiltonian

### 4.1 Variable Transformation

We transform binary variables `y ∈ {0, 1}` to spin variables `z ∈ {-1, +1}`:

```
z = 2y - 1    ⟹    y = (z + 1)/2
```

This maps:
- `y = 0` → `z = -1` (spin down)
- `y = 1` → `z = +1` (spin up)

### 4.2 Ising Form

The QUBO `y^T Q_qubo y` transforms to an Ising Hamiltonian:

```
H = ∑_{i} h_i · σ_i^z + ∑_{i<j} J_{ij} · σ_i^z · σ_j^z + offset
```

Where:
- `σ_i^z` is the Pauli-Z operator on qubit `i` (eigenvalues ±1)
- `h_i` are local field coefficients
- `J_{ij}` are coupling coefficients
- `offset` is a constant (doesn't affect eigenstates)

**Conversion formulas:**
- Linear terms: `Q_{ii} · y_i` → `h_i · σ_i^z` (with appropriate scaling)
- Quadratic terms: `Q_{ij} · y_i · y_j` → `J_{ij} · σ_i^z · σ_j^z` (with appropriate scaling)

### 4.3 Pauli Operator Representation

The Ising Hamiltonian is represented as a **SparsePauliOp** (sum of Pauli strings):

```
H = ∑_{k} c_k · P_k
```

Where each `P_k` is a tensor product of Pauli operators:
- `I` (identity)
- `Z` (Pauli-Z)

For example: `Z ⊗ I ⊗ Z ⊗ I ⊗ ...` means qubits 0 and 2 have `Z`, others have `I`.

**For our 16-qubit problem:** The Ising operator has many Pauli terms (typically hundreds to thousands, depending on constraint penalties and objective structure).

---

## 5. QAOA Ansatz Construction

### 5.1 QAOA Circuit Structure

The QAOA (Quantum Approximate Optimization Algorithm) ansatz alternates between two types of unitaries:

**Problem Hamiltonian evolution:**
```
U_C(γ) = exp(-iγ H_C)
```

Where `H_C` is the Ising Hamiltonian (cost function).

**Mixer Hamiltonian evolution:**
```
U_B(β) = exp(-iβ H_B)
```

Where `H_B = -∑_{i} X_i` is the transverse field mixer (sum of Pauli-X operators).

**For `p=1` layer (reps=1):**
```
|ψ(γ, β)⟩ = U_B(β) · U_C(γ) · |+⟩^⊗n
```

Where `|+⟩ = (|0⟩ + |1⟩)/√2` is the uniform superposition state.

### 5.2 Circuit Implementation

1. **Initialization:** Apply Hadamard gates to all qubits → `|+⟩^⊗16`

2. **Problem layer:** 
   - Decompose `exp(-iγ H_C)` into gates
   - Since `H_C = ∑_k c_k P_k`, we have:
     ```
     exp(-iγ H_C) ≈ ∏_k exp(-iγ c_k P_k)
     ```
   - Each `exp(-iγ c_k P_k)` is implemented using CNOTs and rotation gates

3. **Mixer layer:**
   - `exp(-iβ H_B) = exp(iβ ∑_i X_i) = ∏_i exp(iβ X_i)`
   - Each `exp(iβ X_i)` is `R_X(-2β)` on qubit `i`

4. **Measurement:** Measure all qubits in computational basis

### 5.3 Parameter Ranges

We used:
- `γ ∈ [0, π]` (problem parameter)
- `β ∈ [0, π/2]` (mixer parameter)

These ranges are typical for QAOA and cover the relevant parameter space.

---

## 6. Sampling Strategy (Not Optimization)

### 6.1 Key Design Decision

Instead of using QAOA as an **optimizer** (which would require expensive energy evaluations), we use it as a **parameterized sampler**.

**Why?**
- Energy evaluation requires computing `⟨ψ(γ,β)| H_C |ψ(γ,β)⟩` for many parameter sets
- This is computationally expensive (especially with many Pauli terms)
- For small problems, sampling and classical validation is more efficient

### 6.2 Sampling Workflow

For each parameter set `(γ, β)`:

1. **Build circuit:** Construct QAOA ansatz with parameters `(γ, β)`
2. **Transpile:** Decompose into basis gates (`rz`, `rx`, `ry`, `x`, `sx`, `cx`)
3. **Simulate:** Run on Qiskit Aer simulator with `shots=512`
4. **Sample bitstrings:** Get 512 measurement outcomes (each is a 16-bit string)
5. **Decode:** Convert each bitstring to a tour:
   - Map 16 bits → 4×4 matrix `y_{i,p}`
   - Check constraints (each row/column sums to 1)
   - If valid, construct tour: `[Sydney, city_at_pos1, city_at_pos2, city_at_pos3, city_at_pos4]`
6. **Score:** Compute tour length using distance matrix `D`
7. **Track best:** Keep the shortest valid tour found so far

### 6.3 Parameter Exploration

We tested **30 random parameter sets**:
- `γ ~ Uniform(0, π)`
- `β ~ Uniform(0, π/2)`

**Total samples:** 30 parameter sets × 512 shots = **15,360 bitstrings**

**Not all bitstrings decode to valid tours!** Many violate constraints and are discarded.

---

## 7. Decoding and Validation

### 7.1 Bitstring to Tour Mapping

Given a 16-bit string `b₀b₁...b₁₅`, we reshape it into a 4×4 matrix:

```
y = [b₀  b₁  b₂  b₃ ]   (city 1 at positions 1,2,3,4)
    [b₄  b₅  b₆  b₇ ]   (city 2 at positions 1,2,3,4)
    [b₈  b₉  b₁₀ b₁₁]   (city 3 at positions 1,2,3,4)
    [b₁₂ b₁₃ b₁₄ b₁₅]   (city 4 at positions 1,2,3,4)
```

### 7.2 Constraint Checking

**Row constraint (each city once):**
```
For each row i: ∑_{p=1}^{4} y_{i,p} == 1
```

**Column constraint (each position once):**
```
For each column p: ∑_{i=1}^{4} y_{i,p} == 1
```

If both constraints pass, the bitstring represents a valid tour.

### 7.3 Tour Construction

If valid, construct the tour:
1. Position 0: Sydney (fixed)
2. Position 1: Find `i` where `y_{i,1} = 1`
3. Position 2: Find `i` where `y_{i,2} = 1`
4. Position 3: Find `i` where `y_{i,3} = 1`
5. Position 4: Find `i` where `y_{i,4} = 1`

Result: `tour = [0, i₁, i₂, i₃, i₄]`

### 7.4 Tour Length Calculation

```
L = D[0, i₁] + D[i₁, i₂] + D[i₂, i₃] + D[i₃, i₄] + D[i₄, 0]
```

---

## 8. Results

### 8.1 Parameter Search Progression

As we tested more parameter sets:
- **Few parameter sets:** Best tour ~10% longer than optimal
- **30 parameter sets:** Best tour **0.3% longer than optimal**

**Best result:**
- Optimal: **3,798.3 km**
- QAOA: **3,811.1 km**
- Gap: **12.8 km (0.3%)**

### 8.2 Why This Approach Works

1. **Exploration:** QAOA circuits with different parameters explore different regions of the solution space
2. **Classical validation:** We efficiently check validity and score tours classically
3. **Best-of-many:** Sampling many parameter sets increases the chance of finding good solutions

### 8.3 Computational Cost Breakdown

**Quantum component:**
- Circuit construction: Fast (polynomial in problem size)
- Simulation: Moderate (exponential in qubits, but manageable for 16 qubits with MPS)
- Sampling: Fast (512 shots per parameter set)

**Classical component:**
- Constraint checking: O(n²) per bitstring
- Tour scoring: O(n) per valid tour
- Total: Dominated by the number of samples

**Key insight:** The bottleneck is not the quantum circuit itself, but the encoding, constraint handling, and validation.

---

## 9. Why This Matters

### 9.1 Hybrid Architecture

This experiment demonstrates a **hybrid quantum-classical pipeline**:
- **Quantum:** Explores solution space via parameterized sampling
- **Classical:** Validates constraints, scores solutions, tracks best

Each component does what it's best at.

### 9.2 Scaling Considerations

**At small scale (5 cities):**
- Classical brute force: 24 tours, instant
- Quantum simulation: Slower than brute force

**At large scale (20+ cities):**
- Classical brute force: Infeasible (factorial growth)
- Classical heuristics: Fast but may miss optimal
- Quantum-assisted: Could provide better exploration than random sampling

**The question:** Does quantum sampling provide better candidates than classical heuristics, given the same computational budget?

### 9.3 Where Complexity Lives

The experiment reveals that computational cost lives in:
1. **Encoding:** Converting TSP to QUBO/Ising (polynomial but can be large)
2. **Constraint handling:** Penalty terms increase problem complexity
3. **Validation:** Checking solution validity (polynomial per sample)
4. **Not primarily in the quantum algorithm itself**

This suggests that **architecture and orchestration** matter more than any single algorithm component.

---

## 10. Mathematical Summary

**Problem:** TSP with 5 cities, fixed start

**Encoding:**
- 16 binary variables `y_{i,p}` (4 cities × 4 positions)
- Constraints: `∑_p y_{i,p} = 1`, `∑_i y_{i,p} = 1`
- Objective: Minimize tour length (quadratic in `y`)

**Transformation chain:**
```
TSP → QP → QUBO → Ising Hamiltonian → QAOA Circuit
```

**Algorithm:**
- Sample from QAOA circuits with random `(γ, β)`
- Decode bitstrings to tours
- Validate and score classically
- Return best valid tour

**Result:** 0.3% gap from optimal using 15,360 samples across 30 parameter sets.
