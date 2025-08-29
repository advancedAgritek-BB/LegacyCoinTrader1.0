"""
Example: Using the Momentum-Aware Exit System

This script demonstrates how to use the enhanced momentum-aware exit system
to allow coins with momentum to keep running while protecting profits.
"""

import asyncio
import time
from typing import Dict, Any
import pandas as pd
import numpy as np

# Import the momentum-aware components
from crypto_bot.risk.momentum_position_manager import MomentumPositionManager
from crypto_bot.risk.exit_manager import (
    detect_momentum_continuation,
    calculate_momentum_scaled_take_profit,
    MomentumExitConfig
)


async def example_momentum_aware_trading():
    """Example of momentum-aware trading with dynamic exits."""
    
    # Configuration for momentum-aware exits
    config = {
        'exit_strategy': {
            'momentum_aware_exits': True,
            'momentum_tp_scaling': True,
            'momentum_trail_adjustment': True,
            'momentum_partial_exits': True,
            'momentum_exit_delays': {
                'enabled': True,
                'strong_momentum_delay_seconds': 30,
                'very_strong_momentum_delay_seconds': 60
            },
            'breakout_momentum': {
                'enabled': True,
                'breakout_threshold': 0.015,
                'momentum_extension_multiplier': 2.5,
                'volume_breakout_multiplier': 3.0
            },
            'momentum_continuation': {
                'rsi_momentum_threshold': 65.0,
                'volume_momentum_threshold': 1.5,
                'price_acceleration_threshold': 0.002,
                'macd_momentum_threshold': 0.001
            }
        }
    }
    
    # Initialize the momentum position manager
    momentum_manager = MomentumPositionManager(config)
    
    print("🚀 Momentum-Aware Trading Example")
    print("=" * 50)
    
    # Example 1: Add a position with strong momentum
    print("\n📈 Example 1: Adding a position with strong momentum")
    position1 = await momentum_manager.add_position(
        position_id="pos_001",
        symbol="SOL/USD",
        side="buy",
        entry_price=100.0,
        size=1.0
    )
    
    # Simulate market data showing strong momentum
    strong_momentum_data = create_sample_market_data(
        base_price=100.0,
        trend="strong_up",
        volume_spike=True,
        rsi_high=True
    )
    
    # Update position with strong momentum
    await momentum_manager.update_position(
        position_id="pos_001",
        new_price=105.0,  # 5% profit
        market_data=strong_momentum_data
    )
    
    # Show position summary
    summary1 = momentum_manager.get_position_summary("pos_001")
    print(f"Position Summary:")
    print(f"  Symbol: {summary1['symbol']}")
    print(f"  Entry Price: ${summary1['entry_price']:.2f}")
    print(f"  Current Price: ${summary1['current_price']:.2f}")
    print(f"  PnL: {summary1['unrealized_pnl_pct']*100:.2f}%")
    print(f"  Momentum Strength: {summary1['momentum_strength']:.3f}")
    print(f"  Take Profit Target: ${summary1['take_profit_target']:.2f}")
    print(f"  Trailing Stop: ${summary1['trailing_stop']:.2f}")
    
    # Example 2: Add a position with weak momentum
    print("\n📉 Example 2: Adding a position with weak momentum")
    position2 = await momentum_manager.add_position(
        position_id="pos_002",
        symbol="BTC/USD",
        side="buy",
        entry_price=50000.0,
        size=0.1
    )
    
    # Simulate market data showing weak momentum
    weak_momentum_data = create_sample_market_data(
        base_price=50000.0,
        trend="weak_down",
        volume_spike=False,
        rsi_high=False
    )
    
    # Update position with weak momentum
    await momentum_manager.update_position(
        position_id="pos_002",
        new_price=50200.0,  # 0.4% profit
        market_data=weak_momentum_data
    )
    
    # Show position summary
    summary2 = momentum_manager.get_position_summary("pos_002")
    print(f"Position Summary:")
    print(f"  Symbol: {summary2['symbol']}")
    print(f"  Entry Price: ${summary2['entry_price']:.2f}")
    print(f"  Current Price: ${summary2['current_price']:.2f}")
    print(f"  PnL: {summary2['unrealized_pnl_pct']*100:.2f}%")
    print(f"  Momentum Strength: {summary2['momentum_strength']:.3f}")
    print(f"  Take Profit Target: ${summary2['take_profit_target']:.2f}")
    print(f"  Trailing Stop: ${summary2['trailing_stop']:.2f}")
    
    # Example 3: Simulate price movement and check exit conditions
    print("\n🔄 Example 3: Simulating price movement and exit conditions")
    
    # Simulate SOL continuing to run with momentum
    print("\n  SOL/USD continuing to run with momentum:")
    for price in [110.0, 115.0, 120.0, 125.0]:
        await momentum_manager.update_position(
            position_id="pos_001",
            new_price=price,
            market_data=strong_momentum_data
        )
        
        summary = momentum_manager.get_position_summary("pos_001")
        should_exit, reason, value = momentum_manager.should_exit_position(position1)
        
        print(f"    Price: ${price:.2f}, PnL: {summary['unrealized_pnl_pct']*100:.2f}%")
        print(f"    Exit Decision: {reason} (value: {value})")
        print(f"    Momentum: {summary['momentum_strength']:.3f}")
        
        if should_exit:
            print(f"    🚨 EXIT SIGNAL: {reason}")
            break
    
    # Example 4: Show performance metrics
    print("\n📊 Performance Metrics:")
    metrics = momentum_manager.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Example 5: Demonstrate momentum-based partial exits
    print("\n💰 Example 5: Momentum-based partial exits")
    
    # Check if partial exit should be taken
    position = momentum_manager.positions["pos_001"]
    if position.unrealized_pnl_pct * 100 >= 5.0:  # 5% profit
        print(f"  Position has {position.unrealized_pnl_pct*100:.2f}% profit")
        print(f"  Momentum strength: {position.momentum_strength:.3f}")
        
        # This would normally trigger a partial exit
        print("  ✅ Partial exit conditions met (would execute in live trading)")
    
    print("\n🎯 Key Benefits of Momentum-Aware Exits:")
    print("  1. Strong momentum coins get higher take profit targets")
    print("  2. Trailing stops tighten with momentum strength")
    print("  3. Breakouts automatically extend profit targets")
    print("  4. Partial exits scale with momentum strength")
    print("  5. Exit delays prevent premature exits on strong momentum")
    
    return momentum_manager


def create_sample_market_data(
    base_price: float,
    trend: str = "neutral",
    volume_spike: bool = False,
    rsi_high: bool = False
) -> pd.DataFrame:
    """Create sample market data for testing."""
    
    # Generate price data
    if trend == "strong_up":
        prices = [base_price * (1 + i * 0.02) for i in range(20)]
    elif trend == "weak_down":
        prices = [base_price * (1 - i * 0.005) for i in range(20)]
    else:
        prices = [base_price * (1 + np.random.normal(0, 0.01)) for i in range(20)]
    
    # Generate volume data
    if volume_spike:
        volumes = [1000 * (1 + np.random.normal(0, 0.1)) for i in range(20)]
        volumes[-3:] = [vol * 2.5 for vol in volumes[-3:]]  # Volume spike
    else:
        volumes = [1000 * (1 + np.random.normal(0, 0.1)) for i in range(20)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    # Add technical indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    
    # Adjust RSI based on trend
    if rsi_high:
        df['rsi'] = df['rsi'] * 1.2 + 10  # Higher RSI
    else:
        df['rsi'] = df['rsi'] * 0.8 + 10  # Lower RSI
    
    # Ensure RSI is within bounds
    df['rsi'] = df['rsi'].clip(0, 100)
    
    return df


async def main():
    """Main function to run the example."""
    try:
        await example_momentum_aware_trading()
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
