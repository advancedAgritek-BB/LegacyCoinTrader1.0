#!/usr/bin/env python3
"""
Fix script to set up stop losses and take profits for existing positions.

This script will:
1. Load the current TradeManager state
2. Set up proper stop losses and take profits for all open positions
3. Configure trailing stops based on the config
4. Save the updated state
"""

import sys
from pathlib import Path
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from crypto_bot.utils.trade_manager import get_trade_manager

def setup_position_stops():
    """Set up stop losses and take profits for existing positions."""

    # Load configuration directly from config.yaml
    import yaml
    config_path = Path(__file__).parent / "crypto_bot" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    risk_cfg = config.get("risk", {})
    exit_cfg = config.get("exit_strategy", {})

    # Get exit strategy parameters
    take_profit_pct = exit_cfg.get("take_profit_pct", risk_cfg.get("take_profit_pct", 0.04))
    stop_loss_pct = exit_cfg.get("stop_loss_pct", risk_cfg.get("stop_loss_pct", 0.01))
    trailing_stop_pct = exit_cfg.get("trailing_stop_pct", risk_cfg.get("trailing_stop_pct", 0.008))
    min_gain_to_trail = exit_cfg.get("min_gain_to_trail", 0.005)

    print("🔧 Setting up stop losses for existing positions...")
    print(f"Configuration: SL={stop_loss_pct*100}%, TP={take_profit_pct*100}%, TS={trailing_stop_pct*100}%, MinGain={min_gain_to_trail*100}%")

    # Get trade manager
    tm = get_trade_manager()

    # Get all open positions
    positions = tm.get_all_positions()
    print(f"Found {len(positions)} open positions")

    updated_count = 0

    for position in positions:
        # Skip if already has stops configured
        if position.stop_loss_price is not None:
            print(f"⚠️  {position.symbol}: Already has stop loss configured")
            continue

        # Calculate stop loss price (use configured stop_loss_pct, not trailing)
        if position.side == 'long':
            # Long position: stop loss below entry price
            stop_loss_price = position.average_price * Decimal(str(1 - stop_loss_pct))
            take_profit_price = position.average_price * Decimal(str(1 + take_profit_pct))
        else:
            # Short position: stop loss above entry price
            stop_loss_price = position.average_price * Decimal(str(1 + stop_loss_pct))
            take_profit_price = position.average_price * Decimal(str(1 - take_profit_pct))

        # Set up the position stops
        position.stop_loss_price = stop_loss_price
        position.take_profit_price = take_profit_price
        position.trailing_stop_pct = Decimal(str(trailing_stop_pct))

        print(f"✅ {position.symbol}: Set SL=${stop_loss_price:.6f}, TP=${take_profit_price:.6f}, TS={trailing_stop_pct*100}%")

        # Update highest/lowest prices for trailing stop calculation
        current_price = tm.price_cache.get(position.symbol)
        if current_price:
            position.update_price_levels(Decimal(str(current_price)))

        updated_count += 1

    # Save updated state
    if updated_count > 0:
        tm.save_state()
        print(f"\n💾 Saved updated state with {updated_count} positions configured")
    else:
        print("\nℹ️  No positions needed updating")

    return updated_count

def verify_position_monitoring():
    """Verify that position monitoring is working."""

    tm = get_trade_manager()
    positions = tm.get_all_positions()

    print("\n🔍 Verifying position monitoring setup:")

    for position in positions:
        should_exit, exit_reason = position.should_exit(position.average_price)
        print(f"  {position.symbol}: SL=${position.stop_loss_price}, TP=${position.take_profit_price}, ExitCheck={should_exit} ({exit_reason})")

if __name__ == "__main__":
    try:
        updated = setup_position_stops()
        verify_position_monitoring()

        if updated > 0:
            print(f"\n🎉 Successfully configured {updated} positions with stop losses!")
            print("Position monitoring should now work correctly.")
        else:
            print("\n✅ All positions already have stops configured.")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
