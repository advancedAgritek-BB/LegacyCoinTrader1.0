#!/usr/bin/env python3
"""
Test script for the Momentum-Aware Exit System

This script tests the basic functionality of the momentum-aware exit system
to ensure it's working correctly.
"""

import asyncio
import sys
import os

try:
    from risk.momentum_position_manager import MomentumPositionManager
    from risk.exit_manager import (
        detect_momentum_continuation,
        calculate_momentum_scaled_take_profit,
        MomentumExitConfig
    )
    print("✅ Successfully imported momentum system components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the crypto_bot directory")
    sys.exit(1)


async def test_momentum_system():
    """Test the momentum-aware exit system."""
    
    print("\n🚀 Testing Momentum-Aware Exit System")
    print("=" * 50)
    
    # Test 1: Configuration
    print("\n📋 Test 1: Configuration")
    config = {
        'exit_strategy': {
            'momentum_aware_exits': True,
            'momentum_tp_scaling': True,
            'momentum_trail_adjustment': True,
            'momentum_partial_exits': True,
            'momentum_continuation': {
                'rsi_momentum_threshold': 65.0,
                'volume_momentum_threshold': 1.5,
                'price_acceleration_threshold': 0.002,
                'macd_momentum_threshold': 0.001
            }
        }
    }
    
    try:
        momentum_manager = MomentumPositionManager(config)
        print("✅ MomentumPositionManager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize MomentumPositionManager: {e}")
        return False
    
    # Test 2: Add position
    print("\n📈 Test 2: Adding position")
    try:
        position = await momentum_manager.add_position(
            position_id="test_001",
            symbol="SOL/USD",
            side="buy",
            entry_price=100.0,
            size=1.0
        )
        print(f"✅ Position added: {position.symbol} at ${position.entry_price}")
        print(f"   Take Profit Target: ${position.take_profit_target:.2f}")
        print(f"   Trailing Stop: ${position.trailing_stop:.2f}")
    except Exception as e:
        print(f"❌ Failed to add position: {e}")
        return False
    
    # Test 3: Update position
    print("\n🔄 Test 3: Updating position")
    try:
        # Create simple market data
        import pandas as pd
        import numpy as np
        
        # Simple market data
        dates = pd.date_range('2024-01-01', periods=20, freq='1min')
        prices = [100 + i * 0.5 for i in range(20)]  # Upward trend
        volumes = [1000 + np.random.normal(0, 100) for i in range(20)]
        
        market_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Update position
        updated_position = await momentum_manager.update_position(
            position_id="test_001",
            new_price=105.0,  # 5% profit
            market_data=market_data
        )
        
        if updated_position:
            print(f"✅ Position updated successfully")
            print(f"   Current Price: ${updated_position.current_price}")
            print(f"   PnL: {updated_position.unrealized_pnl_pct*100:.2f}%")
            print(f"   Momentum Strength: {updated_position.momentum_strength:.3f}")
        else:
            print("❌ Position update failed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to update position: {e}")
        return False
    
    # Test 4: Check exit conditions
    print("\n🚪 Test 4: Checking exit conditions")
    try:
        should_exit, reason, value = momentum_manager.should_exit_position(position)
        print(f"✅ Exit check completed")
        print(f"   Should Exit: {should_exit}")
        print(f"   Reason: {reason}")
        print(f"   Value: {value}")
    except Exception as e:
        print(f"❌ Failed to check exit conditions: {e}")
        return False
    
    # Test 5: Get position summary
    print("\n📊 Test 5: Getting position summary")
    try:
        summary = momentum_manager.get_position_summary("test_001")
        if summary:
            print("✅ Position summary retrieved")
            print(f"   Symbol: {summary['symbol']}")
            print(f"   Side: {summary['side']}")
            print(f"   Entry Price: ${summary['entry_price']:.2f}")
            print(f"   Current Price: ${summary['current_price']:.2f}")
            print(f"   PnL: {summary['unrealized_pnl_pct']*100:.2f}%")
            print(f"   Momentum Strength: {summary['momentum_strength']:.3f}")
        else:
            print("❌ Failed to get position summary")
            return False
    except Exception as e:
        print(f"❌ Failed to get position summary: {e}")
        return False
    
    # Test 6: Performance metrics
    print("\n📈 Test 6: Performance metrics")
    try:
        metrics = momentum_manager.get_performance_metrics()
        print("✅ Performance metrics retrieved")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Failed to get performance metrics: {e}")
        return False
    
    # Test 7: Close position
    print("\n🔒 Test 7: Closing position")
    try:
        await momentum_manager.close_position("test_001", "test_complete")
        print("✅ Position closed successfully")
    except Exception as e:
        print(f"❌ Failed to close position: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    return True


async def main():
    """Main test function."""
    print("🧪 Testing Momentum-Aware Exit System")
    print("=" * 50)
    
    try:
        success = await test_momentum_system()
        if success:
            print("\n✅ All tests passed! The momentum system is working correctly.")
            return 0
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite crashed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run the tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
