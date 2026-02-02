#!/usr/bin/env python3
"""
Visualize TSP tours on a map of Australia.

Shows optimal classical tour and best QAOA tour side-by-side.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available. Using simple coordinate plot.")
    print("Install with: pip install cartopy")


def load_tours():
    """Load optimal tour from baseline JSON."""
    baseline_path = Path("results") / "classical_baseline_5_city.json"
    with open(baseline_path, "r") as f:
        data = json.load(f)
    
    cities = data["cities"]
    optimal_tour_indices = data["optimal_tour_indices"]
    optimal_length = data["optimal_length_km"]
    
    return cities, optimal_tour_indices, optimal_length


def get_qaoa_tour():
    """Run QAOA to get the best tour found."""
    import sys
    import os
    sys.path.insert(0, "src")
    
    # Suppress verbose output during QAOA run
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from src.problem import australia_capitals, select_cities, distance_matrix
        from src.qaoa_run import solve_tsp_qaoa_sampling
        
        # Select same 5 cities
        all_cities = australia_capitals()
        cities_5, _ = select_cities(all_cities, ["Sydney", "Melbourne", "Canberra", "Brisbane", "Hobart"])
        D = distance_matrix(cities_5)
        
        # Run QAOA (with optimized settings)
        result = solve_tsp_qaoa_sampling(D, reps=1, shots=512, num_param_sets=30, seed=7)
        
        if result.tour is None:
            return None, None
        
        return result.tour, result.tour_length_km
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def plot_tour_map(cities, tour_indices, tour_length, title, ax, color='blue', use_map=True):
    """Plot a single tour on a map."""
    # Extract coordinates
    lons = [city["lon"] for city in cities]
    lats = [city["lat"] for city in cities]
    names = [city["name"] for city in cities]
    
    if use_map and HAS_CARTOPY:
        # Use cartopy for proper map projection
        # Longitude: 140E to 155E, Latitude: -45S to -10S (10S)
        ax.set_extent([140, 155, -45, -10], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, alpha=0.8, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, alpha=0.6, zorder=1)
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f5f5dc', alpha=0.5, zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#cce5ff', alpha=0.5, zorder=0)
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=2)
        
        # Use PlateCarree projection for plotting
        crs = ccrs.PlateCarree()
    else:
        # Simple coordinate system without map background
        # Longitude: 140E to 155E, Latitude: -45S to -10S (10S)
        ax.set_xlim(140, 155)
        ax.set_ylim(-45, -10)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
        crs = None
    
    # Plot cities as points
    if crs:
        ax.scatter(lons, lats, s=250, c='darkred', zorder=5, edgecolors='black', 
                  linewidths=2, transform=crs)
    else:
        ax.scatter(lons, lats, s=250, c='darkred', zorder=5, edgecolors='black', linewidths=2)
    
    # Add city labels
    for i, (lon, lat, name) in enumerate(zip(lons, lats, names)):
        if crs:
            ax.annotate(name, (lon, lat), xytext=(7, 7), textcoords='offset points',
                       fontsize=11, fontweight='bold', zorder=6, transform=crs,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.annotate(name, (lon, lat), xytext=(7, 7), textcoords='offset points',
                       fontsize=11, fontweight='bold', zorder=6,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Plot tour path
    tour_lons = [lons[i] for i in tour_indices] + [lons[tour_indices[0]]]
    tour_lats = [lats[i] for i in tour_indices] + [lats[tour_indices[0]]]
    
    if crs:
        ax.plot(tour_lons, tour_lats, color=color, linewidth=3, alpha=0.9, zorder=3, 
               linestyle='-', transform=crs)
        ax.plot(tour_lons, tour_lats, color=color, marker='o', markersize=10, zorder=4, 
               linestyle='None', transform=crs)
    else:
        ax.plot(tour_lons, tour_lats, color=color, linewidth=3, alpha=0.9, zorder=3, linestyle='-')
        ax.plot(tour_lons, tour_lats, color=color, marker='o', markersize=10, zorder=4, linestyle='None')
    
    # Set title with length
    ax.set_title(f"{title}\nLength: {tour_length:.1f} km", fontsize=13, fontweight='bold', pad=10)
    
    if not crs:
        ax.set_aspect('equal', adjustable='box')


def main():
    """Generate tour visualization."""
    print("Loading optimal tour from baseline...")
    cities, optimal_tour, optimal_length = load_tours()
    
    print("Running QAOA to get best tour...")
    qaoa_tour, qaoa_length = get_qaoa_tour()
    
    if qaoa_tour is None:
        print("ERROR: QAOA did not find a valid tour")
        return
    
    # Create figure with two subplots
    if HAS_CARTOPY:
        # Use cartopy projection for map background
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        use_map = True
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        use_map = False
    
    # Plot optimal tour (green for optimal)
    plot_tour_map(cities, optimal_tour, optimal_length, "Optimal Classical Tour", ax1, 
                 color='#2E7D32', use_map=use_map)
    
    # Plot QAOA tour (blue for quantum)
    plot_tour_map(cities, qaoa_tour, qaoa_length, "Best QAOA Tour", ax2, 
                 color='#1565C0', use_map=use_map)
    
    # Add overall title
    gap = qaoa_length - optimal_length
    gap_pct = (gap / optimal_length) * 100
    fig.suptitle(
        f"TSP Tour Comparison: 5 Australian Capitals\n"
        f"QAOA Gap: {gap:.1f} km ({gap_pct:.1f}%)",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure with fixed aspect ratio (maintain 2:1 ratio from figsize)
    output_path = Path("results") / "tsp_tours_comparison.png"
    # Don't use bbox_inches='tight' to preserve the figure's 2:1 aspect ratio
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also show it
    plt.show()


if __name__ == "__main__":
    main()

