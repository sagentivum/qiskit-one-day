from __future__ import annotations

import json
import math
from pathlib import Path

from .problem import australia_capitals, distance_matrix, validate_distance_matrix, City
from .classical import brute_force_tsp


# Regression check: expected optimal tour length (km)
# If this changes, coordinates, distance formula, or problem definition was modified
EXPECTED_OPTIMAL_LENGTH_KM = 10719.527607325244
REGRESSION_TOLERANCE_KM = 1.0


def save_ground_truth(
    cities: list[City],
    optimal_tour_indices: list[int],
    optimal_length_km: float,
    search_space_size: int,
    distance_matrix_checks: dict,
    outdir: str = "results",
) -> None:
    """
    Persist the optimal TSP solution as ground truth reference.
    This is treated as immutable - future steps compare against it.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    ground_truth = {
        "problem": "TSP-AU-CAPITALS-8",
        "cities": [{"name": c.name, "lat": c.lat, "lon": c.lon} for c in cities],
        "distance_metric": "haversine_km",
        "start_city_index": 0,
        "optimal_tour_indices": optimal_tour_indices,
        "optimal_length_km": optimal_length_km,
        "search_space_size": search_space_size,
        "distance_matrix_checks": distance_matrix_checks,
    }
    
    ground_truth_path = Path(outdir) / "ground_truth.json"
    with open(ground_truth_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"\nGround truth saved to {ground_truth_path}")


def main():
    cities = australia_capitals()
    D = distance_matrix(cities)
    
    # Validate distance matrix (symmetric, zero diagonal)
    print("Validating distance matrix...")
    checks = validate_distance_matrix(D)
    print("✓ Distance matrix validation passed")
    
    # Solve TSP exactly via brute force
    n = len(cities)
    search_space_size = 1  # For n=1, but for n>1 it's (n-1)!
    if n > 1:
        search_space_size = math.factorial(n - 1)
    
    print(f"\nSolving TSP via brute force (search space: {search_space_size} tours)...")
    best_tour, best_len = brute_force_tsp(D, start=0)
    
    # Regression check: fail loudly if optimal length changed
    if abs(best_len - EXPECTED_OPTIMAL_LENGTH_KM) > REGRESSION_TOLERANCE_KM:
        raise RuntimeError(
            f"Baseline drift: expected ~{EXPECTED_OPTIMAL_LENGTH_KM}, got {best_len}"
        )
    print(f"✓ Regression check passed (optimal length: {best_len:.1f} km)")
    
    # Print results
    print("\nCities (index -> name):")
    for i, c in enumerate(cities):
        print(f"  {i}: {c.name}")
    
    print("\nBest tour (indices):", best_tour)
    print("Best tour (names):", " -> ".join(cities[i].name for i in best_tour) + " -> " + cities[best_tour[0]].name)
    print(f"Length (km): {best_len:.1f}")
    
    # Persist ground truth solution
    save_ground_truth(cities, best_tour, best_len, search_space_size, checks)


if __name__ == "__main__":
    main()
