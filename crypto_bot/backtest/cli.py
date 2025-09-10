"""
Command-line interface for the enhanced backtesting system.

This module provides a CLI for running backtests, viewing results,
and managing the continuous backtesting engine.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from crypto_bot.backtest.enhanced_backtester import (
    create_enhanced_backtester,
    run_backtest_analysis,
    get_strategy_performance_summary
)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('backtest.log')
        ]
    )

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def print_results_summary(results: dict):
    """Print a summary of backtesting results."""
    print("\n" + "="*80)
    print("BACKTESTING RESULTS SUMMARY")
    print("="*80)
    
    total_tests = 0
    successful_tests = 0
    
    for pair, pair_results in results.items():
        print(f"\n{pair}:")
        for strategy, strategy_results in pair_results.items():
            for timeframe, timeframe_results in strategy_results.items():
                total_tests += 1
                if timeframe_results:
                    successful_tests += 1
                    print(f"  {strategy:20s} ({timeframe:5s}): {len(timeframe_results)} results")
                else:
                    print(f"  {strategy:20s} ({timeframe:5s}): No results")
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")

def print_strategy_rankings(rankings: dict):
    """Print current strategy rankings."""
    print("\n" + "="*80)
    print("STRATEGY RANKINGS")
    print("="*80)
    
    for i, (strategy, score) in enumerate(rankings.items(), 1):
        print(f"{i:2d}. {strategy:25s} - Score: {score:.4f}")

def print_performance_summary(summary_df: pd.DataFrame):
    """Print performance summary statistics."""
    if summary_df.empty:
        print("No performance data available.")
        return
        
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY BY STRATEGY")
    print("="*80)
    
    # Flatten column names for better display
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    
    # Round numeric columns
    numeric_cols = summary_df.select_dtypes(include=['float64', 'int64']).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    print(summary_df.to_string())

async def run_single_backtest(args):
    """Run a single backtest for specified parameters."""
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.pairs:
        config['top_pairs_count'] = len(args.pairs)
    
    engine = create_enhanced_backtester(config)
    
    pairs = args.pairs or ["BTC/USDT", "ETH/USDT", "SOL/USDC"]
    strategies = args.strategies or ["trend_bot", "momentum_bot"]
    timeframes = args.timeframes or ["1h", "4h"]
    
    print(f"Running backtest for:")
    print(f"  Pairs: {pairs}")
    print(f"  Strategies: {strategies}")
    print(f"  Timeframes: {timeframes}")
    print(f"  GPU acceleration: {'Enabled' if config.get('use_gpu', True) else 'Disabled'}")
    
    results = await run_backtest_analysis(pairs, strategies, timeframes, config)
    print_results_summary(results)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

async def run_continuous_backtesting(args):
    """Run the continuous backtesting engine."""
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.pairs:
        config['top_pairs_count'] = len(args.pairs)
    if args.strategies:
        config['strategies_to_test'] = args.strategies
    if args.timeframes:
        config['timeframes'] = args.timeframes
    
    engine = create_enhanced_backtester(config)
    
    print("Starting continuous backtesting engine...")
    print(f"  Top pairs: {config.get('top_pairs_count', 20)}")
    print(f"  Strategies: {len(engine.get_all_strategies())}")
    print(f"  Timeframes: {config.get('timeframes', ['1h', '4h', '1d'])}")
    print(f"  GPU acceleration: {'Enabled' if config.get('use_gpu', True) else 'Disabled'}")
    print(f"  Refresh interval: {config.get('refresh_interval_hours', 6)} hours")
    print(f"  Batch size: {config.get('batch_size', 100)}")
    print(f"  Max workers: {config.get('max_workers', 8)}")
    
    try:
        await engine.run_continuous_backtesting()
    except KeyboardInterrupt:
        print("\nStopping continuous backtesting engine...")
        engine.stop()
    except Exception as e:
        print(f"Error in continuous backtesting: {e}")
        engine.stop()

def view_results(args):
    """View and analyze existing backtesting results."""
    cache_dir = args.cache_dir or "crypto_bot/logs/backtest_results"
    
    print(f"Loading results from: {cache_dir}")
    
    # Get performance summary
    summary_df = get_strategy_performance_summary(cache_dir)
    print_performance_summary(summary_df)
    
    # Show detailed results for specific strategy if requested
    if args.strategy:
        strategy = args.strategy
        cache_path = Path(cache_dir)
        
        # Find CSV files for this strategy
        csv_files = list(cache_path.glob(f"{strategy}_*.csv"))
        
        if not csv_files:
            print(f"\nNo results found for strategy: {strategy}")
            return
            
        print(f"\nDetailed results for {strategy}:")
        print("-" * 60)
        
        for csv_file in sorted(csv_files, reverse=True)[:5]:  # Show last 5 results
            try:
                df = pd.read_csv(csv_file)
                print(f"\nFile: {csv_file.name}")
                print(f"Records: {len(df)}")
                if not df.empty:
                    print(f"Average Sharpe: {df['sharpe'].mean():.4f}")
                    print(f"Average PnL: {df['pnl'].mean():.4f}")
                    print(f"Max Drawdown: {df['max_drawdown'].max():.4f}")
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

def optimize_strategies(args):
    """Run strategy optimization based on backtesting results."""
    config = load_config(args.config)
    cache_dir = args.cache_dir or "crypto_bot/logs/backtest_results"
    
    print("Running strategy optimization...")
    
    # Load existing results
    summary_df = get_strategy_performance_summary(cache_dir)
    
    if summary_df.empty:
        print("No backtesting results found. Run backtests first.")
        return
    
    # Filter strategies based on performance thresholds
    config_thresholds = config.get('risk', {})
    min_sharpe = config_thresholds.get('min_sharpe_threshold', 0.5)
    max_dd = config_thresholds.get('max_drawdown_threshold', 0.5)
    
    # Filter by performance
    filtered = summary_df[
        (summary_df[('sharpe', 'mean')] >= min_sharpe) &
        (summary_df[('max_drawdown', 'mean')] <= max_dd)
    ]
    
    print(f"\nStrategies meeting criteria (Sharpe >= {min_sharpe}, Max DD <= {max_dd}):")
    print("-" * 80)
    
    if filtered.empty:
        print("No strategies meet the performance criteria.")
        return
    
    # Sort by Sharpe ratio
    filtered_sorted = filtered.sort_values(('sharpe', 'mean'), ascending=False)
    
    for i, (strategy, row) in enumerate(filtered_sorted.iterrows(), 1):
        print(f"{i:2d}. {strategy:25s}")
        print(f"    Sharpe: {row[('sharpe', 'mean')]:.4f} ± {row[('sharpe', 'std')]:.4f}")
        print(f"    PnL: {row[('pnl', 'mean')]:.4f}")
        print(f"    Max DD: {row[('max_drawdown', 'mean')]:.4f}")
        print()
    
    # Save optimization results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        optimization_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'criteria': {
                'min_sharpe': min_sharpe,
                'max_drawdown': max_dd
            },
            'top_strategies': filtered_sorted.head(10).to_dict('index'),
            'all_strategies': summary_df.to_dict('index')
        }
        
        with open(output_path, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        print(f"Optimization results saved to: {output_path}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Backtesting System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single backtest
  python -m crypto_bot.backtest.cli run --pairs BTC/USDT ETH/USDT --strategies trend_bot momentum_bot
  
  # Run continuous backtesting
  python -m crypto_bot.backtest.cli continuous --config backtest_config.yaml
  
  # View results
  python -m crypto_bot.backtest.cli view --strategy trend_bot
  
  # Optimize strategies
  python -m crypto_bot.backtest.cli optimize --output optimization_results.json
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path (YAML)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run single backtest
    run_parser = subparsers.add_parser('run', help='Run single backtest')
    run_parser.add_argument('--pairs', nargs='+', help='Trading pairs to test')
    run_parser.add_argument('--strategies', nargs='+', help='Strategies to test')
    run_parser.add_argument('--timeframes', nargs='+', help='Timeframes to test')
    run_parser.add_argument('--output', help='Output file for results')
    
    # Run continuous backtesting
    continuous_parser = subparsers.add_parser('continuous', help='Run continuous backtesting')
    continuous_parser.add_argument('--pairs', nargs='+', help='Trading pairs to test')
    continuous_parser.add_argument('--strategies', nargs='+', help='Strategies to test')
    continuous_parser.add_argument('--timeframes', nargs='+', help='Timeframes to test')
    
    # View results
    view_parser = subparsers.add_parser('view', help='View backtesting results')
    view_parser.add_argument('--cache-dir', help='Results cache directory')
    view_parser.add_argument('--strategy', help='Specific strategy to view')
    
    # Optimize strategies
    optimize_parser = subparsers.add_parser('optimize', help='Optimize strategies based on results')
    optimize_parser.add_argument('--cache-dir', help='Results cache directory')
    optimize_parser.add_argument('--output', help='Output file for optimization results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        if args.command == 'run':
            asyncio.run(run_single_backtest(args))
        elif args.command == 'continuous':
            asyncio.run(run_continuous_backtesting(args))
        elif args.command == 'view':
            view_results(args)
        elif args.command == 'optimize':
            optimize_strategies(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
