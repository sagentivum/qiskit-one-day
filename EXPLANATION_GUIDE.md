# How to Explain the TSP-QAOA Experiment

This guide provides talking points and explanations for articulating the experiment clearly.

---

## The Elevator Pitch (30 seconds)

"We tested quantum optimization on a 5-city Traveling Salesman Problem. Instead of using quantum algorithms to directly optimize, we used them as **exploratory samplers** - generating candidate solutions that we then validated and scored classically. With 30 different parameter settings, we found a route within 0.3% of optimal. The key insight: quantum methods work best as components in hybrid systems, not as replacements for classical computation."

---

## The Problem Setup (2 minutes)

### What We Did
- Selected 5 Australian capital cities: Sydney (fixed start), Melbourne, Canberra, Brisbane, Hobart
- Computed real geographic distances using the **haversine formula** (great-circle distances on Earth's surface)
- Created a symmetric distance matrix where `D[i,j]` = distance from city `i` to city `j` in kilometers

### Why This Size?
- With Sydney fixed at position 0, there are only `4! = 24` possible tours
- This allows us to compute the **exact optimal solution** classically via brute force
- **Ground truth:** Optimal tour = 3,798.3 km
- This gives us a benchmark to measure quantum performance against

### Key Insight
"If you can solve a problem by brute force, you should. Clever methods aren't better than exhaustive search when exhaustive search is practical."

---

## The Encoding: TSP → Quantum Circuit (5 minutes)

### Step 1: Positional Binary Encoding

**The idea:** Instead of encoding tours directly, we encode which city occupies which position.

**Variables:** `y_{i,p}` = binary variable meaning "city `i` is at position `p`"

**With fixed start:**
- Sydney is always at position 0 (no variable needed)
- We only need variables for the remaining 4 cities and 4 positions
- **Total: 4 × 4 = 16 binary variables**

**Example:** If `y_{Melbourne,1} = 1`, that means Melbourne is visited first (after Sydney).

### Step 2: Constraints

We need two types of constraints to ensure valid tours:

1. **Each city appears exactly once:**
   ```
   For each city i: ∑_{positions p} y_{i,p} = 1
   ```

2. **Each position is filled exactly once:**
   ```
   For each position p: ∑_{cities i} y_{i,p} = 1
   ```

These constraints ensure we get a valid permutation (every city visited once, every position occupied).

### Step 3: Objective Function (Tour Length)

The tour length is the sum of edge distances:

- **Start → Position 1:** `D[Sydney, city_at_pos1]` → linear term
- **Position p → Position p+1:** `D[city_at_pos_p, city_at_pos_{p+1}]` → **quadratic term** (product of two variables)
- **Position 4 → Start:** `D[city_at_pos4, Sydney]` → linear term

**Why quadratic?** To express "if city A is at position 1 AND city B is at position 2, add distance D[A,B]", we need `y_{A,1} · y_{B,2}`.

### Step 4: Quadratic Program → QUBO → Ising

**Quadratic Program (QP):** Minimize quadratic objective subject to linear constraints

**QUBO conversion:** Convert constraints to penalty terms:
- Add `λ · (constraint - 1)²` for each constraint
- Large penalty `λ` ensures valid solutions have low cost
- Result: Unconstrained quadratic optimization problem

**Ising Hamiltonian:** Transform binary variables `{0,1}` to spin variables `{-1,+1}`:
- `y = 0` → `z = -1` (spin down)
- `y = 1` → `z = +1` (spin up)
- Result: `H = ∑ h_i σ_i^z + ∑ J_{ij} σ_i^z σ_j^z` (Ising model)

**Pauli representation:** The Ising Hamiltonian becomes a sum of Pauli-Z operators:
- Each term is a tensor product like `Z ⊗ I ⊗ Z ⊗ I ⊗ ...`
- For 16 qubits, this typically produces hundreds to thousands of Pauli terms

---

## The QAOA Algorithm (3 minutes)

### What is QAOA?

**Quantum Approximate Optimization Algorithm** - a parameterized quantum circuit designed to find low-energy states of a problem Hamiltonian.

### Circuit Structure

For `p=1` layer (one repetition):

1. **Initialize:** Apply Hadamard gates → uniform superposition `|+⟩^⊗16`
2. **Problem layer:** `exp(-iγ H_C)` where `H_C` is the Ising Hamiltonian
   - This "rotates" the state based on the problem structure
   - Parameter `γ` controls how much
3. **Mixer layer:** `exp(-iβ H_B)` where `H_B = -∑ X_i` (transverse field)
   - This "mixes" states to explore the solution space
   - Parameter `β` controls exploration vs exploitation
4. **Measure:** All qubits in computational basis → get a 16-bit string

### Parameters

- `γ ∈ [0, π]` - problem parameter (how much to follow the cost function)
- `β ∈ [0, π/2]` - mixer parameter (how much to explore)

Different `(γ, β)` pairs explore different regions of the solution space.

---

## The Sampling Strategy (3 minutes)

### Key Design Decision

**We used QAOA as a sampler, not an optimizer.**

**Traditional QAOA approach:**
- Optimize parameters `(γ, β)` to minimize expected energy
- Requires many expensive energy evaluations
- Each evaluation: `⟨ψ(γ,β)| H_C |ψ(γ,β)⟩`

**Our approach:**
- Sample from QAOA circuits with random `(γ, β)` pairs
- Decode bitstrings to tours
- Validate and score classically
- Keep the best valid tour

**Why?** For small problems, sampling + classical validation is more efficient than quantum optimization.

### The Workflow

For each parameter set `(γ, β)`:

1. **Build circuit:** Construct QAOA ansatz with these parameters
2. **Simulate:** Run on Qiskit Aer simulator with 512 shots
3. **Sample:** Get 512 measurement outcomes (each is a 16-bit string)
4. **Decode:** Convert bitstring → 4×4 matrix → check constraints → construct tour
5. **Score:** Compute tour length using distance matrix
6. **Track:** Keep the shortest valid tour found

**Total exploration:** 30 parameter sets × 512 shots = **15,360 bitstrings sampled**

**Not all valid!** Many bitstrings violate constraints and are discarded. Only valid tours are scored.

---

## The Results (2 minutes)

### Performance

- **Optimal tour:** 3,798.3 km (classical brute force)
- **Best QAOA tour:** 3,811.1 km
- **Gap:** 12.8 km (0.3% longer)

### Parameter Search Progression

- **Few parameter sets:** Best tour ~10% longer than optimal
- **30 parameter sets:** Best tour 0.3% longer than optimal

**Interpretation:** More exploration (more parameter sets) naturally increases the chance of finding good solutions.

### Validity Rate

Not all sampled bitstrings decode to valid tours. The constraint penalties in the QUBO formulation help, but don't guarantee validity. This is why classical validation is essential.

---

## Key Insights (2 minutes)

### 1. Hybrid Architecture

**Quantum + Classical, each doing what it's best at:**
- **Quantum:** Explores solution space via parameterized sampling
- **Classical:** Validates constraints, scores solutions, tracks best

This is the emerging pattern in quantum optimization: hybrid systems, not pure quantum solutions.

### 2. Where Complexity Lives

The computational cost is not primarily in the quantum algorithm itself, but in:
- **Encoding:** Converting TSP to QUBO/Ising (polynomial but can be large)
- **Constraint handling:** Penalty terms increase problem complexity
- **Validation:** Checking solution validity (polynomial per sample)

**Implication:** Architecture and orchestration matter more than any single algorithm component.

### 3. Scale Considerations

**At small scale (5 cities):**
- Classical brute force: 24 tours, instant
- Quantum simulation: Slower than brute force

**At large scale (20+ cities):**
- Classical brute force: Infeasible (factorial growth)
- Classical heuristics: Fast but may miss optimal
- Quantum-assisted: Could provide better exploration than random sampling

**The open question:** Does quantum sampling provide better candidates than classical heuristics, given the same computational budget?

---

## Common Questions & Answers

### Q: Why not use QAOA as an optimizer?

**A:** Energy evaluation is expensive. For small problems, sampling + classical validation is more efficient. For larger problems, this trade-off may change.

### Q: Why only 0.3% gap? Is that good or bad?

**A:** It's neither "good" nor "bad" - it's a data point. The experiment shows that quantum sampling can find near-optimal solutions, but doesn't prove quantum advantage. The value is in understanding the hybrid architecture.

### Q: Why 5 cities? That's too small.

**A:** Correct - it's intentionally small. This allows:
1. Exact optimal solution (ground truth)
2. Complete verification of the pipeline
3. Understanding of where complexity lives

Scaling up is a separate question that requires different experiments.

### Q: What about quantum advantage?

**A:** This experiment doesn't demonstrate quantum advantage. It demonstrates that quantum methods can work as components in hybrid systems. Quantum advantage would require showing quantum methods outperform classical methods on problems where classical methods struggle.

### Q: How does this compare to classical random sampling?

**A:** Good question! We didn't explicitly compare, but the code includes a classical random sampler. The key difference: QAOA sampling is **structured** - different parameters explore different regions. Random sampling is unstructured. Whether this structure helps is an empirical question.

---

## Mathematical Formulation Summary

**Problem:** Minimize tour length subject to permutation constraints

**Variables:** `y_{i,p} ∈ {0,1}` for `i,p ∈ {1,2,3,4}` (16 variables)

**Constraints:**
- `∑_p y_{i,p} = 1` ∀i (each city once)
- `∑_i y_{i,p} = 1` ∀p (each position once)

**Objective:**
```
minimize:  ∑_j D[0,j]·y_{j,1} 
         + ∑_{p=1}^{3} ∑_i ∑_j D[i,j]·y_{i,p}·y_{j,p+1}
         + ∑_i D[i,0]·y_{i,4}
```

**Transformation:** QP → QUBO (penalty method) → Ising Hamiltonian → QAOA circuit

**Algorithm:** Sample from QAOA circuits, decode, validate, score, keep best

---

## Talking Points for Different Audiences

### For Technical/Research Audience

- Emphasize the encoding details (positional, fixed start, constraint handling)
- Discuss the QUBO→Ising transformation and Pauli operator structure
- Highlight the sampling vs optimization design choice
- Mention computational cost breakdown

### For Business/Management Audience

- Focus on hybrid architecture and orchestration
- Emphasize that quantum is a component, not a replacement
- Discuss where complexity actually lives (encoding, constraints, validation)
- Frame as "proof of concept" for hybrid quantum-classical systems

### For General/Public Audience

- Start with the problem (TSP) and why it matters
- Explain quantum as "exploration tool" in simple terms
- Emphasize the honest results (0.3% gap, small problem, no claims of advantage)
- Focus on the architectural insight: quantum + classical working together

---

## Key Phrases to Use

- "Hybrid quantum-classical pipeline"
- "Quantum as exploratory sampler, not optimizer"
- "Classical validation and scoring"
- "Architecture and orchestration matter more than any single algorithm"
- "Structured exploration vs random sampling"
- "Where computational complexity actually lives"
- "Proof of concept, not proof of advantage"

---

## What NOT to Claim

- ❌ "Quantum computers solve TSP better than classical"
- ❌ "This demonstrates quantum advantage"
- ❌ "Quantum optimization is ready for production"
- ✅ "Quantum methods can work as components in hybrid systems"
- ✅ "This is a small, auditable experiment with clear results"
- ✅ "The architecture pattern (quantum exploration + classical validation) is promising"
