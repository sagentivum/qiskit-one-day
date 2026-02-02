from __future__ import annotations

import json
import math
from pathlib import Path

from .problem import australia_capitals, distance_matrix, validate_distance_matrix, City, select_cities
from .classical import brute_force_tsp
from .qaoa_run import solve_tsp_qaoa


# Regression check: expected optimal tour length (km) for 8-city problem
# If this changes, coordinates, distance formula, or problem definition was modified
EXPECTED_OPTIMAL_LENGTH_KM_8 = 10719.527607325244
EXPECTED_OPTIMAL_LENGTH_KM_5 = None  # Will be computed and stored after first run
REGRESSION_TOLERANCE_KM = 1.0


def save_ground_truth(
    cities: list[City],
    optimal_tour_indices: list[int],
    optimal_length_km: float,
    search_space_size: int,
    distance_matrix_checks: dict,
    problem_name: str = "TSP-AU-CAPITALS-8",
    outdir: str = "results",
) -> None:
    """
    Persist the optimal TSP solution as ground truth reference.
    
    This is treated as immutable - future steps compare against it.
    
    Args:
        cities: List of City objects representing the problem instance.
        optimal_tour_indices: List of city indices forming the optimal tour.
        optimal_length_km: Length of the optimal tour in kilometers.
        search_space_size: Size of the search space explored.
        distance_matrix_checks: Dictionary with distance matrix validation results.
        problem_name: Problem identifier (default: "TSP-AU-CAPITALS-8").
        outdir: Output directory path (default: "results").
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    ground_truth = {
        "problem": problem_name,
        "cities": [{"name": c.name, "lat": c.lat, "lon": c.lon} for c in cities],
        "distance_metric": "haversine_km",
        "start_city_index": 0,
        "optimal_tour_indices": optimal_tour_indices,
        "optimal_length_km": optimal_length_km,
        "search_space_size": search_space_size,
        "distance_matrix_checks": distance_matrix_checks,
    }
    
    # Use different filenames for different problem sizes
    if problem_name == "TSP-AU-CAPITALS-5":
        ground_truth_path = Path(outdir) / "classical_baseline_5_city.json"
    else:
        ground_truth_path = Path(outdir) / "ground_truth.json"
    
    with open(ground_truth_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\nGround truth saved to {ground_truth_path}")


def main() -> None:
    """
    Main entry point: solve TSP for 5-city subset of Australia capitals using brute force and QAOA.
    
    Performs validation, solves via brute force, compares with QAOA solution,
    and saves ground truth results.
    """
    # Select 5-city subset: Sydney (start), Melbourne, Canberra, Brisbane, Hobart
    all_cities = australia_capitals()
    selected_city_names = ["Sydney", "Melbourne", "Canberra", "Brisbane", "Hobart"]
    cities, index_map = select_cities(all_cities, selected_city_names)
    
    print(f"Selected {len(cities)} cities from {len(all_cities)} total cities")
    print("City selection (new_index -> name):")
    for i, c in enumerate(cities):
        print(f"  {i}: {c.name}")
    
    # Build distance matrix for selected cities
    D = distance_matrix(cities)
    
    # Validate distance matrix (symmetric, zero diagonal)
    print("\nValidating distance matrix...")
    checks = validate_distance_matrix(D)
    print("✓ Distance matrix validation passed")
    
    # Solve TSP exactly via brute force
    n = len(cities)
    search_space_size = 1  # For n=1, but for n>1 it's (n-1)!
    if n > 1:
        search_space_size = math.factorial(n - 1)
    
    print(f"\nSolving TSP via brute force (search space: {search_space_size} tours)...")
    best_tour, best_len = brute_force_tsp(D, start=0)  # Sydney is at index 0
    
    # Load or compute expected optimal length for regression check
    baseline_path = Path("results") / "classical_baseline_5_city.json"
    expected_length = None
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
            expected_length = baseline.get("optimal_length_km")
    
    if expected_length is not None:
        # Regression check: fail loudly if optimal length changed
        if abs(best_len - expected_length) > REGRESSION_TOLERANCE_KM:
            raise RuntimeError(
                f"Baseline drift: expected ~{expected_length}, got {best_len}"
            )
        print(f"✓ Regression check passed (optimal length: {best_len:.1f} km)")
    else:
        print(f"✓ First run - optimal length: {best_len:.1f} km (will be saved as baseline)")
    
    # Print results
    print("\nBest tour (indices):", best_tour)
    print("Best tour (names):", " -> ".join(cities[i].name for i in best_tour) + " -> " + cities[best_tour[0]].name)
    print(f"Length (km): {best_len:.1f}")
    
    # Run QAOA on 5-city subset (16 variables for fixed-start)
    print(f"\nSolving TSP via QAOA sampling (reps=1, shots=512, num_param_sets=30)...")
    print(f"Problem size: {n} cities -> {(n-1)*(n-1)} variables ({n-1}×{n-1})")
    print("Parameter ranges: gamma ∈ [0, π], beta ∈ [0, π/2]")
    
    from .qaoa_run import build_tsp_qp_fixed_start, solve_tsp_qaoa_sampling
    from .diagnose_qaoa import diagnose
    
    # Build QP and diagnose
    qp, _, _ = build_tsp_qp_fixed_start(D, start=0)
    diagnose(qp)
    
    # Run QAOA sampling with optimized parameters (defaults: shots=512, num_param_sets=30)
    city_names = [c.name for c in cities]
    q = solve_tsp_qaoa_sampling(D, reps=1, shots=512, num_param_sets=30, seed=7, city_names=city_names)
    
    print("\nQAOA status:", q.status)
    print("QAOA tour:", q.tour)
    if q.tour:
        print("QAOA tour (names):", " -> ".join(cities[i].name for i in q.tour) + " -> " + cities[q.tour[0]].name)
        print("QAOA length (km):", f"{q.tour_length_km:.1f}")
        gap = q.tour_length_km - best_len
        print(f"Gap vs optimal: {gap:.1f} km ({gap/best_len*100:.1f}%)")
    else:
        print("QAOA tour (names): INVALID")
        print("QAOA length (km): INVALID")
    
    # Persist ground truth solution for 5-city problem
    save_ground_truth(cities, best_tour, best_len, search_space_size, checks, problem_name="TSP-AU-CAPITALS-5")


if __name__ == "__main__":
    main()
