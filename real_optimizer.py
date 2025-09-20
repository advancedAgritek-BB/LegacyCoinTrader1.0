#!/usr/bin/env python3
"""
Real optimizer for dip_hunter strategy with actual backtesting
Bypasses strict configuration validation for optimization
"""

import yaml
import json
import math
import itertools
import os
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_dip_hunter_config():
    """Load dip_hunter configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'optimization' not in config or not config['optimization'].get('enabled', False):
        print("❌ Optimization not enabled in config")
        return None

    param_ranges = config['optimization'].get('parameter_ranges', {})
    if 'dip_hunter' not in param_ranges:
        print("❌ dip_hunter not found in parameter ranges")
        return None

    return param_ranges['dip_hunter']

def create_backtest_function():
    """Create a backtest function for parameter optimization with real Kraken data."""
    def backtest_function(params: Dict[str, float]) -> Dict[str, Any]:
        """Backtest function for optimization algorithms using real market data."""
        try:
            import ccxt
            import pandas as pd
            import numpy as np
            from crypto_bot.strategy.dip_hunter import generate_signal

            # Initialize Kraken exchange directly
            exchange = ccxt.kraken({
                'apiKey': os.getenv('KRAKEN_API_KEY', os.getenv('API_KEY')),
                'secret': os.getenv('KRAKEN_API_SECRET', os.getenv('API_SECRET')),
                'enableRateLimit': True,
            })

            print(f"📊 Fetching real BTC/USD data from Kraken...")

            # Fetch real historical data (last 30 days, 1h timeframe)
            # Use simpler approach without specific timestamp to avoid API issues
            ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', limit=720)  # 30 days * 24 hours

            if not ohlcv or len(ohlcv) < 100:
                print(f"⚠️  Insufficient data from Kraken: {len(ohlcv) if ohlcv else 0} records")
                return {"sharpe": 0.0, "pnl": 0.0, "max_drawdown": 1.0}

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            print(f"✅ Loaded {len(df)} real OHLCV records from Kraken")
            print(f"Latest price: \${df.iloc[-1]['close']:.2f}")
            # Create config for dip_hunter with optimized parameters
            config = {
                'dip_hunter': {
                    'rsi_window': int(params.get('rsi_window', 14)),
                    'rsi_oversold': float(params.get('rsi_oversold', 30.0)),
                    'dip_pct': float(params.get('dip_pct', 0.03)),
                    'dip_bars': int(params.get('dip_bars', 3)),
                    'vol_window': int(params.get('vol_window', 20)),
                    'vol_mult': float(params.get('vol_mult', 1.5)),
                    'adx_window': int(params.get('adx_window', 14)),
                    'adx_threshold': float(params.get('adx_threshold', 25.0)),
                    'bb_window': int(params.get('bb_window', 20)),
                    'ema_trend': int(params.get('ema_trend', 200)),
                    'ml_weight': float(params.get('ml_weight', 0.5)),
                    'atr_normalization': True,
                    'ema_slow': int(params.get('ema_slow', 20))
                }
            }

            # Run strategy on real data
            signals = []
            position = None
            entry_price = 0
            trades = []

            # Walk through data and generate signals
            for i in range(100, len(df) - 1):  # Start after enough history
                subset = df.iloc[max(0, i-200):i+1]  # Use last 200 bars for context

                score, direction = generate_signal(subset, symbol='BTC/USD', timeframe='1h', config=config)

                current_price = subset.iloc[-1]['close']

                # Trading logic
                if direction == 'long' and position is None:
                    # Enter long position
                    position = 'long'
                    entry_price = current_price
                    signals.append({'type': 'entry', 'price': current_price, 'index': i})

                elif direction == 'none' and position == 'long':
                    # Check for exit conditions (simplified)
                    if current_price >= entry_price * (1 + params.get('take_profit_pct', 0.04)) or \
                       current_price <= entry_price * (1 - params.get('stop_loss_pct', 0.02)):
                        # Exit position
                        exit_price = current_price
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'bars_held': i - signals[-1]['index']
                        })
                        position = None
                        entry_price = 0

            # Calculate performance metrics
            if trades:
                pnl_series = [trade['pnl'] for trade in trades]
                cumulative_returns = np.cumprod([1 + pnl for pnl in pnl_series]) - 1

                # Sharpe ratio (simplified)
                if len(pnl_series) > 1:
                    avg_return = np.mean(pnl_series)
                    std_return = np.std(pnl_series)
                    sharpe = avg_return / std_return * np.sqrt(365 * 24) if std_return > 0 else 0
                else:
                    sharpe = 0

                # Max drawdown (simplified)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = running_max - cumulative_returns
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

                win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
                total_pnl = sum(pnl_series)

                print(f"   Performance: Sharpe={sharpe:.3f}, PnL={total_pnl:.3f}, Win Rate={win_rate:.1%}")
                return {
                    "sharpe": float(sharpe),
                    "pnl": float(total_pnl),
                    "max_drawdown": float(max_drawdown),
                    "win_rate": float(win_rate),
                    "total_trades": len(trades)
                }
            else:
                print(f"📊 No trades generated with current parameters")
                return {"sharpe": 0.0, "pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_trades": 0}

        except Exception as e:
            print(f"⚠️  Backtest function failed: {e}")
            import traceback
            traceback.print_exc()
            return {"sharpe": 0.0, "pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_trades": 0}

    return backtest_function

def bayesian_optimization(param_ranges: Dict[str, List[float]], max_evaluations: int = 100) -> Dict[str, Any]:
    """Bayesian optimization for efficient parameter search"""
    print(f"🎯 Starting BAYESIAN optimization with {max_evaluations} evaluations")

    try:
        import optuna
        OPTUNA_AVAILABLE = True
    except ImportError:
        print("⚠️  Optuna not available, falling back to smart sampling")
        OPTUNA_AVAILABLE = False

    if OPTUNA_AVAILABLE:
        return _run_optuna_optimization(param_ranges, max_evaluations)
    else:
        return _run_smart_sampling(param_ranges, max_evaluations)

def _run_optuna_optimization(param_ranges: Dict[str, List[float]], max_evaluations: int) -> Dict[str, Any]:
    """Run optimization using Optuna (Bayesian optimization)"""
    from crypto_bot.auto_optimizer import create_backtest_function

    def objective(trial):
        params = {}
        for param_name, value_range in param_ranges.items():
            if len(value_range) == 2:
                # Continuous range
                params[param_name] = trial.suggest_float(param_name, value_range[0], value_range[1])
            else:
                # Discrete choices
                params[param_name] = trial.suggest_categorical(param_name, value_range)

        try:
            backtest_func = create_backtest_function("dip_hunter")
            result = backtest_func(params)
            score = result.get("sharpe", 0.0)
            return -score  # Negative for minimization
        except Exception as e:
            print(f"⚠️  Backtest failed for params {params}: {e}")
            return float('inf')

    try:
        import optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=max_evaluations)

        best_params = study.best_params
        backtest_func = create_backtest_function("dip_hunter")
        best_result = backtest_func(best_params)

        return {
            'method': 'bayesian_optuna',
            'best_parameters': best_params,
            'best_score': -study.best_value,  # Convert back to positive
            'total_combinations_tested': len(study.trials),
            'results': [
                {
                    'iteration': i,
                    'params': t.params,
                    'score': -t.value,  # Convert back to positive
                    'backtest_result': backtest_func(t.params) if i < 5 else None
                }
                for i, t in enumerate(study.trials[:20])  # Keep first 20
            ]
        }

    except Exception as e:
        print(f"⚠️  Optuna optimization failed: {e}")
        return _run_smart_sampling(param_ranges, max_evaluations)

def _run_smart_sampling(param_ranges: Dict[str, List[float]], max_evaluations: int) -> Dict[str, Any]:
    """Smart sampling when Optuna is not available"""
    print(f"🧠 Using smart sampling with {max_evaluations} evaluations")

    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]

    # Use Latin Hypercube-like sampling for better coverage
    import random
    random.seed(42)  # For reproducibility

    results = []
    best_score = -float('inf')
    best_params = None

    backtest_func = create_backtest_function()

    for i in range(max_evaluations):
        # Sample parameters (could be improved with better sampling)
        params = {}
        for j, param_name in enumerate(param_names):
            values = param_ranges[param_name]
            params[param_name] = random.choice(values)

        print(f"🧪 Testing combination {i+1}/{max_evaluations}")
        result = backtest_func(params)
        score = result.get("sharpe", 0.0)

        results.append({
            'iteration': i,
            'params': params,
            'score': score,
            'backtest_result': result
        })

        if score > best_score:
            best_score = score
            best_params = params

        print(".3f")

    return {
        'method': 'smart_sampling',
        'best_parameters': best_params,
        'best_score': best_score,
        'total_combinations_tested': max_evaluations,
        'results': results[:20]  # Keep first 20
    }

def real_grid_search(param_ranges: Dict[str, List[float]], max_combinations: int = 50) -> Dict[str, Any]:
    """Real grid search optimization with actual backtesting"""

    print(f"🔍 Starting REAL grid search optimization with max {max_combinations} combinations")

    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]

    total_combinations = math.prod(len(values) for values in param_values)
    print(f"📊 Total possible combinations: {total_combinations}")

    # Generate all combinations first
    combinations = list(itertools.product(*param_values))

    # Limit combinations if too many
    if len(combinations) > max_combinations:
        print(f"✂️  Limiting to {max_combinations} combinations out of {len(combinations)}")
        combinations = combinations[:max_combinations]

    print(f"🎯 Testing {len(combinations)} parameter combinations")

    # Get backtest function
    backtest_func = create_backtest_function()

    # Mock backtest results (in real implementation, this would call actual backtest)
    results = []
    best_score = -float('inf')
    best_params = None

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        # Run actual backtest
        print(f"🧪 Testing combination {i+1}/{len(combinations)}: {params}")
        result = backtest_func(params)
        score = result.get("sharpe", 0.0)

        results.append({
            'iteration': i,
            'params': params,
            'score': score,
            'backtest_result': result
        })

        if score > best_score:
            best_score = score
            best_params = params

        print(f"   Sharpe: {score:.3f}")
        if (i + 1) % 5 == 0:
            print(f"📈 Completed {i + 1}/{len(combinations)} combinations")

    return {
        'method': 'real_grid_search',
        'best_parameters': best_params,
        'best_score': best_score,
        'total_combinations_tested': len(combinations),
        'results': results[:20]  # Only keep first 20 for brevity
    }

def main():
    print("🚀 === Real Dip Hunter Strategy Optimizer ===\n")

    # Load configuration
    param_ranges = load_dip_hunter_config()
    if not param_ranges:
        print("❌ Failed to load dip_hunter configuration")
        return

    print("✅ Loaded parameter ranges:")
    for param, values in param_ranges.items():
        print(f"   {param}: {values}")
    print()

    # Choose optimization method
    param_values = [param_ranges[name] for name in param_ranges.keys()]
    total_combinations = math.prod(len(values) for values in param_values)
    print(f"📊 Total possible combinations: {total_combinations:,}")

    # Use Bayesian optimization for large parameter spaces
    if total_combinations > 10000:
        print("🎯 Using BAYESIAN optimization (more efficient for large parameter spaces)")
        results = bayesian_optimization(param_ranges, max_evaluations=100)
    elif total_combinations > 1000:
        print("🧠 Using SMART SAMPLING (efficient for medium parameter spaces)")
        results = bayesian_optimization(param_ranges, max_evaluations=200)
    else:
        print("🔍 Using GRID SEARCH (feasible for small parameter spaces)")
        results = real_grid_search(param_ranges, max_combinations=min(1000, total_combinations))

    print("\n🎉 === Optimization Results ===")
    print(f"📊 Method: {results['method']}")
    print(".4f")
    print(f"🔢 Combinations Tested: {results['total_combinations_tested']}")
    print("\n🏆 Best Parameters:")
    if results['best_parameters']:
        for param, value in results['best_parameters'].items():
            print(f"   {param}: {value}")
    else:
        print("   No valid parameters found")

    print("\n📋 Top 5 Results:")
    sorted_results = sorted(results['results'], key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        print(f"   Score: {result['score']:.4f}")
        print(f"   Params: {result['params']}")

    # Save results
    output_file = Path(__file__).parent / "logs" / "dip_hunter_real_optimization_results.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
