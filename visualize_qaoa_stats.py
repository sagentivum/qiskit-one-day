#!/usr/bin/env python3
"""
Generate Figures 2 and 3 for the blog article.

Figure 2: Best gap achieved by parameter set number (running best)
Figure 3: Valid tours per parameter set
"""

import sys
import os
sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.problem import australia_capitals, select_cities, distance_matrix
from src.qaoa_run import solve_tsp_qaoa_sampling


def load_optimal_length():
    """Load optimal tour length from baseline."""
    import json
    baseline_path = Path("results") / "classical_baseline_5_city.json"
    with open(baseline_path, "r") as f:
        data = json.load(f)
    return data["optimal_length_km"]


def run_qaoa_and_capture_stats():
    """Run QAOA and capture per-parameter-set statistics."""
    # Suppress verbose output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # Select same 5 cities
        all_cities = australia_capitals()
        cities_5, _ = select_cities(all_cities, ["Sydney", "Melbourne", "Canberra", "Brisbane", "Hobart"])
        D = distance_matrix(cities_5)
        
        # Run QAOA with optimized settings
        result = solve_tsp_qaoa_sampling(D, reps=1, shots=512, num_param_sets=30, seed=7)
        
        return result, D
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def figure2_best_gap(result, optimal_length):
    """Figure 2: Best gap achieved by parameter set number (running best)."""
    if result.per_param_stats is None:
        print("ERROR: No per-parameter statistics available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    param_sets = [s['param_set'] for s in result.per_param_stats]
    running_bests = []
    current_best = float('inf')
    
    for stat in result.per_param_stats:
        if stat['running_best'] < current_best:
            current_best = stat['running_best']
        running_bests.append(current_best if current_best != float('inf') else None)
    
    # Plot running best
    valid_mask = [b is not None for b in running_bests]
    valid_params = [p for p, m in zip(param_sets, valid_mask) if m]
    valid_bests = [b for b, m in zip(running_bests, valid_mask) if m]
    
    ax.plot(valid_params, valid_bests, 'o-', linewidth=2.5, markersize=8, 
           color='#1565C0', label='Running Best QAOA Tour', zorder=3)
    
    # Highlight the drop at parameter set 11 (if applicable)
    # Find where we hit 3811.1 km or close
    target = 3811.1
    for i, (p, b) in enumerate(zip(valid_params, valid_bests)):
        if abs(b - target) < 1.0:
            ax.annotate(f'Best: {b:.1f} km\n(Set {p})', 
                       xy=(p, b), xytext=(p + 2, b + 100),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       zorder=5)
            break
    
    # Draw optimum line
    ax.axhline(y=optimal_length, color='#2E7D32', linestyle='--', linewidth=2, 
              label=f'Optimal: {optimal_length:.1f} km', zorder=2, alpha=0.8)
    ax.text(len(param_sets) * 0.98, optimal_length, f'  Optimal: {optimal_length:.1f} km',
           verticalalignment='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
           zorder=4)
    
    ax.set_xlabel('Parameter Set Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Tour Length (km)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Best Gap Achieved by Parameter Set\n(Running Best)', 
                fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xlim(0.5, len(param_sets) + 0.5)
    
    # Set y-axis to show the range nicely
    y_min = min([b for b in valid_bests] + [optimal_length])
    y_max = max([b for b in valid_bests] + [optimal_length])
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.1)
    
    plt.tight_layout()
    output_path = Path("results") / "figure2_best_gap.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Figure 2 saved to: {output_path}")
    plt.close()


def figure3_valid_tours(result):
    """Figure 3: Valid tours per parameter set."""
    if result.per_param_stats is None:
        print("ERROR: No per-parameter statistics available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    param_sets = [s['param_set'] for s in result.per_param_stats]
    valid_counts = [s['valid_count'] for s in result.per_param_stats]
    
    # Create bar chart
    colors = ['#4CAF50' if c > 0 else '#E0E0E0' for c in valid_counts]
    bars = ax.bar(param_sets, valid_counts, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    
    # Highlight bars with valid tours
    for i, (p, c) in enumerate(zip(param_sets, valid_counts)):
        if c > 0:
            bars[i].set_color('#4CAF50')
            bars[i].set_alpha(0.7)
        else:
            bars[i].set_color('#E0E0E0')
            bars[i].set_alpha(0.5)
    
    ax.set_xlabel('Parameter Set Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Valid Tours Found', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: Valid Tours per Parameter Set', 
                fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=1)
    ax.set_xlim(0.5, len(param_sets) + 0.5)
    ax.set_ylim(-0.5, max(valid_counts) + 0.5)
    
    # Add statistics text
    total_valid = sum(valid_counts)
    param_sets_with_valid = sum(1 for c in valid_counts if c > 0)
    stats_text = f'Total valid tours: {total_valid}\nParameter sets with valid tours: {param_sets_with_valid}/{len(param_sets)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
           zorder=4)
    
    plt.tight_layout()
    output_path = Path("results") / "figure3_valid_tours.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Figure 3 saved to: {output_path}")
    plt.close()


def main():
    """Generate both figures."""
    print("Loading optimal length...")
    optimal_length = load_optimal_length()
    
    print("Running QAOA to capture statistics...")
    result, D = run_qaoa_and_capture_stats()
    
    if result.per_param_stats is None:
        print("ERROR: QAOA did not return per-parameter statistics")
        return
    
    print(f"\nGenerating Figure 2: Best gap achieved...")
    figure2_best_gap(result, optimal_length)
    
    print(f"Generating Figure 3: Valid tours per parameter set...")
    figure3_valid_tours(result)
    
    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()

