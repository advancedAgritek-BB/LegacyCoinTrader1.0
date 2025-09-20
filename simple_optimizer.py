#!/usr/bin/env python3
"""
Simple optimizer for dip_hunter strategy
Bypasses complex configuration system for testing
"""

import yaml
import json
import math
import itertools
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_dip_hunter_config():
    """Load dip_hunter configuration from temp_config.yaml"""
    config_path = Path(__file__).parent / "temp_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    optimization_config = config.get('optimization', {})
    if not optimization_config.get('enabled', False):
        print("Optimization is disabled in config")
        return None

    param_ranges = optimization_config.get('parameter_ranges', {})
    if 'dip_hunter' not in param_ranges:
        print("dip_hunter not found in parameter ranges")
        return None

    return param_ranges['dip_hunter']

def simple_grid_search(param_ranges: Dict[str, List[float]], max_combinations: int = 50) -> Dict[str, Any]:
    """Simple grid search optimization"""

    print(f"Starting grid search optimization with max {max_combinations} combinations")

    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]

    total_combinations = math.prod(len(values) for values in param_values)
    print(f"Total possible combinations: {total_combinations}")

    # Generate all combinations first
    combinations = list(itertools.product(*param_values))

    # Limit combinations if too many
    if len(combinations) > max_combinations:
        print(f"Limiting to {max_combinations} combinations out of {len(combinations)}")
        combinations = combinations[:max_combinations]

    print(f"Testing {len(combinations)} parameter combinations")

    # Mock backtest results (in real implementation, this would call actual backtest)
    results = []
    best_score = -float('inf')
    best_params = None

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        # Mock scoring function (replace with real backtest)
        score = mock_evaluate_params(params)

        results.append({
            'iteration': i,
            'params': params,
            'score': score
        })

        if score > best_score:
            best_score = score
            best_params = params

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{len(combinations)} combinations")

    return {
        'method': 'grid_search',
        'best_parameters': best_params,
        'best_score': best_score,
        'total_combinations_tested': len(combinations),
        'results': results[:10]  # Only keep first 10 for brevity
    }

def mock_evaluate_params(params: Dict[str, Any]) -> float:
    """Mock evaluation function for parameter optimization"""
    # This is a simplified scoring function
    # In real implementation, this would run actual backtests

    score = 0.0

    # RSI parameters - prefer moderate values
    rsi_window = params.get('rsi_window', 14)
    if 12 <= rsi_window <= 18:
        score += 0.2

    rsi_oversold = params.get('rsi_oversold', 30)
    if 28 <= rsi_oversold <= 35:
        score += 0.15

    # Dip parameters - prefer moderate dip detection
    dip_pct = params.get('dip_pct', 0.03)
    if 0.025 <= dip_pct <= 0.04:
        score += 0.2

    dip_bars = params.get('dip_bars', 3)
    if 2 <= dip_bars <= 4:
        score += 0.1

    # Volume parameters - prefer moderate sensitivity
    vol_mult = params.get('vol_mult', 1.5)
    if 1.3 <= vol_mult <= 1.8:
        score += 0.15

    # ADX parameters - prefer moderate trend filtering
    adx_threshold = params.get('adx_threshold', 25)
    if 22 <= adx_threshold <= 30:
        score += 0.1

    # ML weight - prefer moderate ML influence
    ml_weight = params.get('ml_weight', 0.5)
    if 0.4 <= ml_weight <= 0.6:
        score += 0.1

    return score

def main():
    print("=== Dip Hunter Strategy Optimizer ===\n")

    # Load configuration
    param_ranges = load_dip_hunter_config()
    if not param_ranges:
        print("Failed to load dip_hunter configuration")
        return

    print("Loaded parameter ranges:")
    for param, values in param_ranges.items():
        print(f"  {param}: {values}")
    print()

    # Run optimization
    print("Starting optimization...")
    results = simple_grid_search(param_ranges, max_combinations=100)

    print("\n=== Optimization Results ===")
    print(f"Method: {results['method']}")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Combinations Tested: {results['total_combinations_tested']}")
    print("\nBest Parameters:")
    for param, value in results['best_parameters'].items():
        print(f"  {param}: {value}")

    print("\nTop 5 Results:")
    sorted_results = sorted(results['results'], key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Params: {result['params']}")

    # Save results
    output_file = Path(__file__).parent / "logs" / "dip_hunter_optimization_results.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
