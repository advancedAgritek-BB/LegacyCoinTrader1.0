"""
Integrated CLI for Enhanced Backtesting + coinTrader_Trainer ML System

This CLI provides unified access to both the enhanced backtesting system
and the coinTrader_Trainer ML system, allowing seamless integration
between continuous backtesting and machine learning model training.
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
from crypto_bot.backtest.ml_integration import (
    create_ml_integration,
    run_integrated_system,
    get_integration_config_template
)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('integrated_backtest.log')
        ]
    )

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def print_integration_status(integration):
    """Print ML integration status."""
    print("\n" + "="*80)
    print("ML INTEGRATION STATUS")
    print("="*80)
    
    status = integration.get_integration_status()
    
    print(f"ML Integration Enabled: {status['ml_integration_enabled']}")
    print(f"Auto Retrain: {status['auto_retrain_enabled']}")
    print(f"Last Retrain: {status['last_retrain'] or 'Never'}")
    print(f"Retrain Interval: {status['retrain_interval_hours']} hours")
    print(f"Supabase Configured: {status['supabase_configured']}")
    print(f"coinTrader_Trainer Available: {status['cointrainer_available']}")
    
    gpu_info = status['gpu_acceleration']
    print(f"\nGPU Acceleration:")
    print(f"  Status: {gpu_info['status']}")
    if gpu_info['status'] != 'no_gpu':
        print(f"  Type: {gpu_info['gpu_type']}")
        print(f"  Memory: {gpu_info['memory_gb']:.1f} GB")
        print(f"  Compute Units: {gpu_info['compute_units']}")

async def run_integrated_backtesting(args):
    """Run integrated backtesting with ML training."""
    config = load_config(args.config)
    
    print("Starting integrated backtesting with ML training...")
    print(f"  GPU acceleration: {'Enabled' if config.get('use_gpu', True) else 'Disabled'}")
    print(f"  ML integration: {'Enabled' if config.get('ml_integration', {}).get('enabled', True) else 'Disabled'}")
    
    try:
        # Create ML integration
        integration = create_ml_integration(config)
        
        # Show integration status
        print_integration_status(integration)
        
        # Run integrated system
        await integration.run_integrated_backtesting()
        
    except KeyboardInterrupt:
        print("\nStopping integrated backtesting...")
    except Exception as e:
        print(f"Error in integrated backtesting: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

async def run_ml_training_only(args):
    """Run only ML training using coinTrader_Trainer."""
    config = load_config(args.config)
    
    print("Running ML training with coinTrader_Trainer...")
    
    try:
        integration = create_ml_integration(config)
        
        # Check if coinTrader_Trainer is available
        if not integration._check_cointrainer_availability():
            print("❌ coinTrader_Trainer not available. Please install it first.")
            print("   Install with: pip install cointrader-trainer")
            return
        
        # Get latest backtesting results
        backtest_results = integration._get_latest_backtest_results()
        
        if not backtest_results:
            print("⚠️  No backtesting results available. Run backtesting first.")
            return
        
        # Prepare training data
        training_data = await integration.prepare_ml_training_data(backtest_results)
        
        if training_data.empty:
            print("⚠️  No training data could be prepared.")
            return
        
        # Train ML models
        symbols = args.symbols or training_data['pair'].unique().tolist()
        await integration.train_ml_models(training_data, symbols)
        
        print("✅ ML training completed successfully!")
        
    except Exception as e:
        print(f"❌ ML training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

async def run_backtest_only(args):
    """Run only enhanced backtesting without ML training."""
    config = load_config(args.config)
    
    # Disable ML integration for this run
    config['ml_integration'] = {'enabled': False}
    
    engine = create_enhanced_backtester(config)
    
    print("Running enhanced backtesting only...")
    print(f"  Top pairs: {config.get('top_pairs_count', 20)}")
    print(f"  Strategies: {len(engine.get_all_strategies())}")
    print(f"  Timeframes: {config.get('timeframes', ['1h', '4h', '1d'])}")
    print(f"  GPU acceleration: {'Enabled' if config.get('use_gpu', True) else 'Disabled'}")
    
    try:
        await engine.run_continuous_backtesting()
    except KeyboardInterrupt:
        print("\nStopping enhanced backtesting...")
    except Exception as e:
        print(f"Error in enhanced backtesting: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def validate_ml_models(args):
    """Validate ML model predictions against backtesting results."""
    config = load_config(args.config)
    
    print("Validating ML model predictions...")
    
    try:
        integration = create_ml_integration(config)
        
        # Create sample features for validation
        sample_features = pd.DataFrame({
            'rsi_14': [30, 50, 70],
            'atr_14': [0.02, 0.015, 0.025],
            'ema_8': [100, 101, 99],
            'ema_21': [100.5, 100.8, 99.5]
        })
        
        # Validate for each symbol
        symbols = args.symbols or ['BTC/USDT', 'ETH/USDT']
        
        for symbol in symbols:
            print(f"\nValidating {symbol}...")
            
            # Run validation
            validation_result = asyncio.run(
                integration.validate_ml_predictions(symbol, sample_features)
            )
            
            if validation_result['valid']:
                print(f"✅ {symbol}: Validation successful")
                print(f"   Predictions: {validation_result['total_predictions']}")
                print(f"   Avg Confidence: {validation_result['avg_confidence']:.3f}")
            else:
                print(f"❌ {symbol}: Validation failed - {validation_result['error']}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def setup_environment(args):
    """Setup environment for coinTrader_Trainer integration."""
    print("Setting up environment for coinTrader_Trainer integration...")
    
    try:
        # Check if coinTrader_Trainer is available
        import subprocess
        
        result = subprocess.run(['cointrainer', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ coinTrader_Trainer is available")
        else:
            print("❌ coinTrader_Trainer not found")
            print("   Install with: pip install cointrader-trainer")
            return
        
        # Create necessary directories
        directories = [
            "data/ml_training",
            "local_models",
            "out/backtests",
            "out/opt"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        # Create sample configuration
        config_template = get_integration_config_template()
        
        config_file = Path("config/ml_integration_template.yaml")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_template, f, default_flow_style=False, indent=2)
        
        print(f"✅ Created configuration template: {config_file}")
        print("\nNext steps:")
        print("1. Edit config/ml_integration_template.yaml with your Supabase credentials")
        print("2. Copy to config/backtest_config.yaml")
        print("3. Run: python -m crypto_bot.backtest.integrated_cli integrated")
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Integrated Backtesting + ML Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup environment for coinTrader_Trainer
  python -m crypto_bot.backtest.integrated_cli setup
  
  # Run integrated system (backtesting + ML training)
  python -m crypto_bot.backtest.integrated_cli integrated --config config/backtest_config.yaml
  
  # Run only enhanced backtesting
  python -m crypto_bot.backtest.integrated_cli backtest-only --config config/backtest_config.yaml
  
  # Run only ML training
  python -m crypto_bot.backtest.integrated_cli ml-training --symbols BTC/USDT ETH/USDT
  
  # Validate ML models
  python -m crypto_bot.backtest.integrated_cli validate --symbols BTC/USDT
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
    
    # Setup environment
    setup_parser = subparsers.add_parser('setup', help='Setup environment for coinTrader_Trainer')
    
    # Run integrated system
    integrated_parser = subparsers.add_parser('integrated', help='Run integrated backtesting + ML training')
    
    # Run backtesting only
    backtest_parser = subparsers.add_parser('backtest-only', help='Run enhanced backtesting only')
    
    # Run ML training only
    ml_parser = subparsers.add_parser('ml-training', help='Run ML training only')
    ml_parser.add_argument('--symbols', nargs='+', help='Symbols to train models for')
    
    # Validate ML models
    validate_parser = subparsers.add_parser('validate', help='Validate ML model predictions')
    validate_parser.add_argument('--symbols', nargs='+', help='Symbols to validate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        if args.command == 'setup':
            setup_environment(args)
        elif args.command == 'integrated':
            asyncio.run(run_integrated_backtesting(args))
        elif args.command == 'backtest-only':
            asyncio.run(run_backtest_only(args))
        elif args.command == 'ml-training':
            asyncio.run(run_ml_training_only(args))
        elif args.command == 'validate':
            validate_ml_models(args)
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
