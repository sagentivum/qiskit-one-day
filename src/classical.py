from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np

from .problem import tour_length_km


def brute_force_tsp(D: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    Exact TSP solver by brute force.
    
    Fixes the start city to remove rotational symmetry.

    Args:
        D: Symmetric distance matrix (NxN) with zeros on diagonal, units in kilometers.
        start: Index of the starting city (default: 0).

    Returns:
        Tuple of (best_tour, best_len) where:
        - best_tour: list of city indices (length N) forming the optimal tour
        - best_len: closed tour length in kilometers
    """
    n = D.shape[0]
    if n < 2:
        return [start], 0.0

    nodes = list(range(n))
    nodes.remove(start)

    best_tour: List[int] = []
    best_len = float("inf")

    for perm in itertools.permutations(nodes):
        tour = [start] + list(perm)
        length = tour_length_km(tour, D)
        if length < best_len:
            best_len = length
            best_tour = tour

    return best_tour, best_len
