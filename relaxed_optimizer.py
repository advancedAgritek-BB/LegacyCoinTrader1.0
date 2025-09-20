#!/usr/bin/env python3
"""
Relaxed Dip Hunter Optimizer - Tests with parameters that can generate signals even in trending markets.
"""

import sys
import os
sys.path.insert(0, '.')

import ccxt
import pandas as pd
import numpy as np
import math
from crypto_bot.strategy.dip_hunter import generate_signal
from dotenv import load_dotenv

load_dotenv('.env.local')

def load_relaxed_dip_hunter_config():
    """Load relaxed parameters that can generate signals in trending markets."""
    param_ranges = {
        'rsi_window': [5, 10, 14],  # Shorter RSI for more signals
        'rsi_oversold': [35.0, 40.0, 45.0],  # Higher oversold threshold
        'dip_pct': [0.01, 0.015, 0.02],  # Smaller dip requirements
        'dip_bars': [2, 3],  # Shorter dip windows
        'vol_window': [10, 15, 20],  # Shorter volume windows
        'vol_mult': [1.2, 1.5],  # Lower volume multipliers
        'adx_window': [10, 14],  # Shorter ADX windows
        'adx_threshold': [30.0, 35.0, 40.0],  # Higher ADX thresholds (allow more trending)
        'bb_window': [10, 15, 20],  # Shorter BB windows
        'ema_trend': [50, 100, 150],  # Shorter trend EMAs
        'ml_weight': [0.0, 0.3, 0.5],  # Include no ML option
        'ema_slow': [10, 15, 20],  # Shorter slow EMAs
        'take_profit_pct': [0.02, 0.04, 0.06],
        'stop_loss_pct': [0.01, 0.02, 0.03]
    }

    print("🔧 Using RELAXED parameters for signal generation:")
    for param, values in param_ranges.items():
        print(f"   {param}: {values}")

    return param_ranges

def create_relaxed_backtest_function():
    """Create a backtest function with relaxed parameters for trending markets."""
    def backtest_function(params: dict) -> dict:
        """Backtest function using longer historical data."""
        try:
            import ccxt

            # Initialize Kraken exchange
            exchange = ccxt.kraken({
                'apiKey': os.getenv('KRAKEN_API_KEY', os.getenv('API_KEY')),
                'secret': os.getenv('KRAKEN_API_SECRET', os.getenv('API_SECRET')),
                'enableRateLimit': True,
            })

            print(f"📊 Fetching BTC/USD data from Kraken (extended history)...")

            # Fetch longer historical data to find more varied market conditions
            ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', limit=1500)  # Last ~60 days

            if not ohlcv or len(ohlcv) < 200:
                print(f"⚠️  Insufficient data from Kraken: {len(ohlcv) if ohlcv else 0} records")
                return {"sharpe": 0.0, "pnl": 0.0, "max_drawdown": 0.0, "total_trades": 0}

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            print(f"✅ Loaded {len(df)} real OHLCV records")
            print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")

            # Create config with relaxed parameters
            config = {
                'dip_hunter': {
                    'rsi_window': int(params.get('rsi_window', 10)),
                    'rsi_oversold': float(params.get('rsi_oversold', 40.0)),  # More relaxed
                    'dip_pct': float(params.get('dip_pct', 0.015)),  # Smaller dips
                    'dip_bars': int(params.get('dip_bars', 2)),  # Shorter windows
                    'vol_window': int(params.get('vol_window', 15)),
                    'vol_mult': float(params.get('vol_mult', 1.2)),  # Lower volume req
                    'adx_window': int(params.get('adx_window', 10)),
                    'adx_threshold': float(params.get('adx_threshold', 35.0)),  # Higher threshold
                    'bb_window': int(params.get('bb_window', 15)),
                    'ema_trend': int(params.get('ema_trend', 100)),
                    'ml_weight': float(params.get('ml_weight', 0.0)),  # Start with no ML
                    'atr_normalization': True,
                    'ema_slow': int(params.get('ema_slow', 15))
                }
            }

            # Run strategy on historical data
            signals = []
            position = None
            entry_price = 0
            trades = []

            # Walk through data and generate signals (skip most recent data for backtesting)
            start_idx = 200  # Skip first 200 bars for indicator warmup
            end_idx = len(df) - 24  # Skip last 24 hours for forward testing

            for i in range(start_idx, end_idx):
                subset = df.iloc[max(0, i-100):i+1]  # Use last 100 bars for context

                score, direction = generate_signal(subset, symbol='BTC/USD', timeframe='1h', config=config)

                current_price = subset.iloc[-1]['close']

                # Trading logic
                if direction == 'long' and position is None and score > 0:
                    # Enter long position
                    position = 'long'
                    entry_price = current_price
                    signals.append({'type': 'entry', 'price': current_price, 'index': i, 'score': score})

                elif position == 'long':
                    # Check for exit conditions
                    take_profit_price = entry_price * (1 + params.get('take_profit_pct', 0.04))
                    stop_loss_price = entry_price * (1 - params.get('stop_loss_pct', 0.02))

                    if current_price >= take_profit_price or current_price <= stop_loss_price:
                        # Exit position
                        exit_price = current_price
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'bars_held': i - signals[-1]['index'],
                            'entry_score': signals[-1]['score']
                        })
                        position = None
                        entry_price = 0

            # Calculate performance metrics
            if trades:
                pnl_series = [trade['pnl'] for trade in trades]
                cumulative_returns = np.cumprod([1 + pnl for pnl in pnl_series]) - 1

                # Sharpe ratio
                if len(pnl_series) > 1:
                    avg_return = np.mean(pnl_series)
                    std_return = np.std(pnl_series)
                    sharpe = avg_return / std_return * np.sqrt(365 * 24) if std_return > 0 else 0
                else:
                    sharpe = 0

                # Max drawdown
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = running_max - cumulative_returns
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

                win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
                total_pnl = sum(pnl_series)

                print(f"   📊 Results: {len(trades)} trades, Win Rate: {win_rate:.1%}, PnL: {total_pnl:.3f}, Sharpe: {sharpe:.3f}")

                return {
                    "sharpe": float(sharpe),
                    "pnl": float(total_pnl),
                    "max_drawdown": float(max_drawdown),
                    "win_rate": float(win_rate),
                    "total_trades": len(trades)
                }
            else:
                print(f"   📊 No trades generated with current parameters")
                return {"sharpe": 0.0, "pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_trades": 0}

        except Exception as e:
            print(f"⚠️  Backtest function failed: {e}")
            import traceback
            traceback.print_exc()
            return {"sharpe": 0.0, "pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_trades": 0}

    return backtest_function

def smart_relaxed_optimization(param_ranges, max_evaluations=50):
    """Smart sampling optimization with relaxed parameters."""
    print(f"🎯 Starting RELAXED smart sampling with {max_evaluations} evaluations")

    backtest_func = create_relaxed_backtest_function()
    results = []

    # Generate parameter combinations intelligently
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]

    # Sample diverse combinations
    sampled_combinations = []
    for _ in range(max_evaluations):
        # Random sampling with some constraints
        combination = {}
        for name, values in param_ranges.items():
            # Bias toward more relaxed values
            if name == 'rsi_oversold':
                combination[name] = np.random.choice(values, p=[0.2, 0.5, 0.3])  # Favor higher thresholds
            elif name == 'dip_pct':
                combination[name] = np.random.choice(values, p=[0.4, 0.4, 0.2])  # Favor smaller dips
            elif name == 'adx_threshold':
                combination[name] = np.random.choice(values, p=[0.3, 0.4, 0.3])  # Favor higher ADX
            else:
                combination[name] = np.random.choice(values)

        sampled_combinations.append(combination)

    print(f"🧪 Testing {len(sampled_combinations)} parameter combinations...")

    for i, params in enumerate(sampled_combinations):
        print(f"🧪 Testing combination {i+1}/{len(sampled_combinations)}")
        print(f"   Params: rsi_oversold={params['rsi_oversold']}, dip_pct={params['dip_pct']}, adx_threshold={params['adx_threshold']}")

        result = backtest_func(params)
        result['params'] = params
        results.append(result)

        # Show some results
        if (i + 1) % 5 == 0:
            print(f"   Progress: {i+1}/{len(sampled_combinations)} combinations tested")

    # Sort and return best results
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    return {
        "method": "relaxed_smart_sampling",
        "total_combinations_tested": len(results),
        "best_result": results[0] if results else None,
        "top_5_results": results[:5] if len(results) >= 5 else results,
        "all_results": results
    }

def main():
    print("🚀 === RELAXED DIP HUNTER OPTIMIZER ===\n")
    print("🎯 Using relaxed parameters to find signals even in trending markets\n")

    param_ranges = load_relaxed_dip_hunter_config()
    print()

    # Use relaxed smart sampling
    print("🧠 Using RELAXED SMART SAMPLING (optimized for signal generation)")
    results = smart_relaxed_optimization(param_ranges, max_evaluations=50)

    print("\n🎉 === OPTIMIZATION RESULTS ===")
    print(f"📊 Method: {results['method']}")
    print(f"🔢 Combinations Tested: {results['total_combinations_tested']}")

    if results['best_result']:
        best = results['best_result']
        print("\n🏆 BEST RESULT:")
        print(f"   Sharpe: {best['sharpe']:.3f}")
        print(f"   PnL: {best['pnl']:.3f}")
        print(f"   Win Rate: {best['win_rate']:.1%}")
        print(f"   Max Drawdown: {best['max_drawdown']:.3f}")
        print(f"📊 Total Trades: {best['total_trades']}")
        print("📋 Parameters:")
        for key, value in best['params'].items():
            print(f"   {key}: {value}")

    print("\n🔝 TOP 5 RESULTS:")
    for i, result in enumerate(results['top_5_results'][:5]):
        print(f"{i+1}. Sharpe: {result['sharpe']:.3f}, PnL: {result['pnl']:.3f}, "
              f"Win Rate: {result['win_rate']:.1%}, Trades: {result['total_trades']}")

    # Save results
    import json
    from datetime import datetime

    results_file = f"logs/dip_hunter_relaxed_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {
            "method": results["method"],
            "total_combinations_tested": results["total_combinations_tested"],
            "best_result": {
                "sharpe": float(results["best_result"]["sharpe"]) if results["best_result"] else None,
                "pnl": float(results["best_result"]["pnl"]) if results["best_result"] else None,
                "max_drawdown": float(results["best_result"]["max_drawdown"]) if results["best_result"] else None,
                "win_rate": float(results["best_result"]["win_rate"]) if results["best_result"] else None,
                "total_trades": results["best_result"]["total_trades"] if results["best_result"] else None,
                "params": results["best_result"]["params"] if results["best_result"] else None
            },
            "top_5_results": [
                {
                    "sharpe": float(r["sharpe"]),
                    "pnl": float(r["pnl"]),
                    "max_drawdown": float(r["max_drawdown"]),
                    "win_rate": float(r["win_rate"]),
                    "total_trades": r["total_trades"],
                    "params": r["params"]
                } for r in results["top_5_results"][:5]
            ]
        }
        json.dump(serializable_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
