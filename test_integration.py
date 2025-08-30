#!/usr/bin/env python3
"""
Integration Test Script for Enhanced Backtesting + coinTrader_Trainer ML

This script tests the integration between the enhanced backtesting system
and the coinTrader_Trainer ML system to ensure they work together properly.
"""

import asyncio
import logging
import sys
import pytest
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_ml_integration_imports():
    """Test ML integration module imports."""
    print("Testing ML integration imports...")
    
    try:
        from crypto_bot.backtest.ml_integration import (
            create_ml_integration,
            MLBacktestingIntegration,
            get_integration_config_template
        )
        
        print("✅ ML integration imports working")
        return True
        
    except Exception as e:
        print(f"❌ ML integration imports failed: {e}")
        return False

def test_integration_config():
    """Test integration configuration template."""
    print("\nTesting integration configuration...")
    
    try:
        from crypto_bot.backtest.ml_integration import get_integration_config_template
        
        config = get_integration_config_template()
        
        # Check required keys
        required_keys = ['ml_integration']
        ml_keys = ['enabled', 'auto_retrain', 'supabase', 'features']
        
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False
                
        for key in ml_keys:
            if key not in config['ml_integration']:
                print(f"❌ Missing required ML config key: {key}")
                return False
        
        print("✅ Integration configuration template working")
        return True
        
    except Exception as e:
        print(f"❌ Integration configuration failed: {e}")
        return False

def test_ml_integration_creation():
    """Test ML integration instance creation."""
    print("\nTesting ML integration creation...")
    
    try:
        from crypto_bot.backtest.ml_integration import create_ml_integration
        
        # Create minimal config
        config = {
            'use_gpu': False,  # Disable GPU for testing
            'ml_integration': {
                'enabled': True,
                'auto_retrain': False,
                'cointrainer_path': 'cointrainer'
            }
        }
        
        integration = create_ml_integration(config)
        
        if not isinstance(integration, MLBacktestingIntegration):
            print("❌ Integration creation returned wrong type")
            return False
            
        print("✅ ML integration creation working")
        return True
        
    except Exception as e:
        print(f"❌ ML integration creation failed: {e}")
        return False

def test_integration_status():
    """Test integration status reporting."""
    print("\nTesting integration status...")
    
    try:
        from crypto_bot.backtest.ml_integration import create_ml_integration
        
        config = {
            'use_gpu': False,
            'ml_integration': {
                'enabled': True,
                'auto_retrain': False,
                'cointrainer_path': 'cointrainer'
            }
        }
        
        integration = create_ml_integration(config)
        status = integration.get_integration_status()
        
        # Check required status keys
        required_status_keys = [
            'ml_integration_enabled',
            'auto_retrain_enabled',
            'gpu_acceleration',
            'supabase_configured',
            'cointrainer_available'
        ]
        
        for key in required_status_keys:
            if key not in status:
                print(f"❌ Missing status key: {key}")
                return False
        
        print("✅ Integration status working")
        return True
        
    except Exception as e:
        print(f"❌ Integration status failed: {e}")
        return False

@pytest.mark.asyncio
async def test_ml_training_data_preparation():
    """Test ML training data preparation."""
    print("\nTesting ML training data preparation...")
    
    try:
        from crypto_bot.backtest.ml_integration import create_ml_integration
        
        config = {
            'use_gpu': False,
            'ml_integration': {
                'enabled': True,
                'auto_retrain': False,
                'cointrainer_path': 'cointrainer'
            }
        }
        
        integration = create_ml_integration(config)
        
        # Create sample backtesting results
        sample_results = {
            'BTC/USDT': {
                'trend_bot': {
                    '1h': [
                        {
                            'sharpe': 0.5,
                            'pnl': 0.02,
                            'max_drawdown': 0.15,
                            'stop_loss_pct': 0.02,
                            'take_profit_pct': 0.04
                        },
                        {
                            'sharpe': 0.6,
                            'pnl': 0.03,
                            'max_drawdown': 0.12,
                            'stop_loss_pct': 0.015,
                            'take_profit_pct': 0.045
                        }
                    ]
                }
            }
        }
        
        # Prepare training data
        training_data = await integration.prepare_ml_training_data(sample_results)
        
        if training_data.empty:
            print("❌ No training data generated")
            return False
            
        # Check required columns
        required_columns = [
            'pair', 'strategy', 'timeframe', 'avg_sharpe', 'avg_pnl',
            'max_drawdown', 'win_rate', 'total_trades'
        ]
        
        for col in required_columns:
            if col not in training_data.columns:
                print(f"❌ Missing required column: {col}")
                return False
        
        print(f"✅ Training data preparation working - {len(training_data)} samples")
        return True
        
    except Exception as e:
        print(f"❌ ML training data preparation failed: {e}")
        return False

def test_cointrainer_availability():
    """Test coinTrader_Trainer availability check."""
    print("\nTesting coinTrader_Trainer availability...")
    
    try:
        from crypto_bot.backtest.ml_integration import create_ml_integration
        
        config = {
            'use_gpu': False,
            'ml_integration': {
                'enabled': True,
                'auto_retrain': False,
                'cointrainer_path': 'cointrainer'
            }
        }
        
        integration = create_ml_integration(config)
        available = integration._check_cointrainer_availability()
        
        if available:
            print("✅ coinTrader_Trainer is available")
        else:
            print("⚠️  coinTrader_Trainer not available (this is expected if not installed)")
            
        return True  # This test should always pass
        
    except Exception as e:
        print(f"❌ coinTrader_Trainer availability check failed: {e}")
        return False

def test_integrated_cli_imports():
    """Test integrated CLI imports."""
    print("\nTesting integrated CLI imports...")
    
    try:
        from crypto_bot.backtest.integrated_cli import (
            create_ml_integration,
            run_integrated_system,
            get_integration_config_template
        )
        
        print("✅ Integrated CLI imports working")
        return True
        
    except Exception as e:
        print(f"❌ Integrated CLI imports failed: {e}")
        return False

def test_config_file_loading():
    """Test configuration file loading."""
    print("\nTesting configuration file loading...")
    
    try:
        config_path = Path("config/backtest_config.yaml")
        
        if not config_path.exists():
            print("⚠️  Configuration file not found - this is expected if not set up yet")
            return True
            
        from crypto_bot.backtest.integrated_cli import load_config
        
        config = load_config(str(config_path))
        
        if not config:
            print("⚠️  Configuration file is empty or invalid")
            return True
            
        # Check for ML integration config
        if 'ml_integration' in config:
            print("✅ Configuration file loaded with ML integration settings")
        else:
            print("⚠️  Configuration file loaded but missing ML integration settings")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration file loading failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("Enhanced Backtesting + coinTrader_Trainer ML Integration Test Suite")
    print("=" * 70)
    
    tests = [
        ("ML Integration Imports", test_ml_integration_imports),
        ("Integration Configuration", test_integration_config),
        ("ML Integration Creation", test_ml_integration_creation),
        ("Integration Status", test_integration_status),
        ("ML Training Data Preparation", test_ml_training_data_preparation),
        ("coinTrader_Trainer Availability", test_cointrainer_availability),
        ("Integrated CLI Imports", test_integrated_cli_imports),
        ("Configuration File Loading", test_config_file_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Integration Test Results Summary")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:35s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! The systems are ready to work together.")
        print("\nNext steps:")
        print("1. Install coinTrader_Trainer: pip install cointrader-trainer")
        print("2. Setup environment: python -m crypto_bot.backtest.integrated_cli setup")
        print("3. Configure Supabase in config/backtest_config.yaml")
        print("4. Run integrated system: python -m crypto_bot.backtest.integrated_cli integrated")
    else:
        print("⚠️  Some integration tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("1. Ensure all required packages are installed")
        print("2. Check that the crypto_bot package is in your Python path")
        print("3. Verify that all import paths are correct")
    
    return passed == total

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nIntegration test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nIntegration test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
