from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, cos, asin, sqrt
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class City:
    """
    Represents a city with geographic coordinates.
    
    Attributes:
        name: City name.
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
    """
    name: str
    lat: float  # decimal degrees
    lon: float  # decimal degrees


def australia_capitals() -> List[City]:
    """
    Return list of Australian capital cities with coordinates.
    
    City coordinates (lat, lon) are in decimal degrees.

    Sources (one per city cluster):
    - Sydney: -33.865143, 151.209900  (latlong.net)
    - Melbourne: -37.840935, 144.946457 (latlong.net)
    - Perth: -31.953512, 115.857048 (latlong.net)
    - Adelaide: -34.921230, 138.599503 (latlong.net)
    - Canberra: -35.282001, 149.128998 (latlong.net)
    - Brisbane: -27.470125, 153.021072 (latlong.net)
    - Hobart: -42.880554, 147.324997 (latlong.net)
    - Darwin: -12.462827, 130.841782 (latlong.net)
    
    Returns:
        List of City objects representing Australian capital cities.
    """
    return [
        City("Sydney",    -33.865143, 151.209900),
        City("Melbourne", -37.840935, 144.946457),
        City("Perth",     -31.953512, 115.857048),
        City("Adelaide",  -34.921230, 138.599503),
        City("Canberra",  -35.282001, 149.128998),
        City("Brisbane",  -27.470125, 153.021072),
        City("Hobart",    -42.880554, 147.324997),
        City("Darwin",    -12.462827, 130.841782),
    ]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points on Earth.
    
    Args:
        lat1: Latitude of first point in decimal degrees.
        lon1: Longitude of first point in decimal degrees.
        lat2: Latitude of second point in decimal degrees.
        lon2: Longitude of second point in decimal degrees.
    
    Returns:
        Great-circle distance in kilometers.
    """
    R = 6371.0088  # mean Earth radius in km
    phi1, lam1, phi2, lam2 = map(radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def distance_matrix(cities: List[City]) -> np.ndarray:
    """
    Compute NxN symmetric distance matrix from city coordinates.
    
    Units: kilometers (great-circle distances via haversine formula).
    Matrix has zeros on diagonal and is symmetric.
    
    Args:
        cities: List of City objects with geographic coordinates.
    
    Returns:
        NxN symmetric distance matrix (np.ndarray) with zeros on diagonal.
    """
    n = len(cities)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(cities[i].lat, cities[i].lon, cities[j].lat, cities[j].lon)
            D[i, j] = d
            D[j, i] = d
    return D


def validate_distance_matrix(D: np.ndarray) -> dict:
    """
    Validate distance matrix properties.
    
    Checks that diagonal is zero and matrix is symmetric.
    
    Args:
        D: Distance matrix to validate.
    
    Returns:
        Dictionary with validation flags: {"symmetric": bool, "diag_zero": bool}
    
    Raises:
        ValueError: If validation fails (matrix not symmetric or diagonal not zero).
    """
    symmetric = bool(np.allclose(D, D.T, atol=1e-9))
    diag_zero = bool(np.allclose(np.diag(D), 0.0, atol=1e-12))
    
    if not symmetric:
        raise ValueError("Distance matrix is not symmetric.")
    if not diag_zero:
        raise ValueError("Distance matrix diagonal is not zero.")
    
    return {"symmetric": True, "diag_zero": True}


def select_cities(cities: List[City], city_names: List[str]) -> Tuple[List[City], Dict[int, int]]:
    """
    Select a subset of cities from the full list and remap indices.
    
    Takes the full city list and returns a filtered list containing only
    the specified cities, with indices remapped to a compact 0..n-1 space.
    The start city (first in city_names) will be at index 0.
    
    Args:
        cities: Full list of City objects.
        city_names: List of city names to select (must be present in cities).
                    First city will be at index 0 in the result.
    
    Returns:
        Tuple of (selected_cities, original_to_new_index) where:
        - selected_cities: Filtered list of City objects in the order specified
        - original_to_new_index: Dictionary mapping original indices to new indices
    
    Raises:
        ValueError: If any city name is not found in the cities list.
    """
    # Build name-to-city mapping
    name_to_city = {city.name: (i, city) for i, city in enumerate(cities)}
    
    # Verify all requested cities exist
    missing = [name for name in city_names if name not in name_to_city]
    if missing:
        raise ValueError(f"Cities not found: {missing}")
    
    # Build selected list in the order specified, preserving start city as index 0
    selected_cities = []
    original_to_new_index = {}
    
    for new_idx, city_name in enumerate(city_names):
        original_idx, city = name_to_city[city_name]
        selected_cities.append(city)
        original_to_new_index[original_idx] = new_idx
    
    return selected_cities, original_to_new_index


def tour_length_km(tour: List[int], D: np.ndarray) -> float:
    """
    Compute closed tour length from distance matrix.
    
    Tour is a permutation of city indices. Returns total length including
    return to the starting city.
    
    Args:
        tour: List of city indices forming a tour (permutation of [0..n-1]).
        D: Distance matrix (NxN) in kilometers.
    
    Returns:
        Total closed tour length in kilometers (includes return to start city).
    """
    total = 0.0
    n = len(tour)
    for k in range(n):
        a = tour[k]
        b = tour[(k + 1) % n]
        total += D[a, b]
    return total
