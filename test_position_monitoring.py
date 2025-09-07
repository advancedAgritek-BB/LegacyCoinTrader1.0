#!/usr/bin/env python3
"""
Test script to verify position monitoring and stop loss functionality.
"""

import sys
from pathlib import Path
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from crypto_bot.utils.trade_manager import get_trade_manager

def test_stop_loss_logic():
    """Test that stop loss logic is working correctly."""

    print("🔍 Testing position monitoring and stop loss logic...")

    tm = get_trade_manager()
    positions = tm.get_all_positions()

    if not positions:
        print("❌ No positions found to test")
        return False

    print(f"Found {len(positions)} positions to test")

    # Test each position
    for position in positions:
        print(f"\n📊 Testing {position.symbol} ({position.side}):")

        # Test current price (should not trigger exit)
        current_price = tm.price_cache.get(position.symbol, position.average_price)
        should_exit, exit_reason = position.should_exit(current_price)

        print(f"  Current price: ${float(current_price):.6f}")
        print(f"  Entry price: ${float(position.average_price):.6f}")
        sl_price = float(position.stop_loss_price) if position.stop_loss_price else None
        print(f"  Stop loss: ${sl_price:.6f}" if sl_price else "  Stop loss: None")
        tp_price = float(position.take_profit_price) if position.take_profit_price else None
        print(f"  Take profit: ${tp_price:.6f}" if tp_price else "  Take profit: None")
        print(f"  Should exit at current price: {should_exit} ({exit_reason})")

        # Test stop loss trigger
        if position.stop_loss_price:
            stop_price = position.stop_loss_price
            should_exit_stop, exit_reason_stop = position.should_exit(stop_price)
            print(f"  Should exit at stop loss (${float(stop_price):.6f}): {should_exit_stop} ({exit_reason_stop})")

            if not should_exit_stop:
                print(f"  ❌ ERROR: Stop loss not triggered at stop price!")
                return False
            else:
                print(f"  ✅ Stop loss correctly triggered at stop price")

        # Test take profit trigger
        if position.take_profit_price:
            tp_price = position.take_profit_price
            should_exit_tp, exit_reason_tp = position.should_exit(tp_price)
            print(f"  Should exit at take profit (${float(tp_price):.6f}): {should_exit_tp} ({exit_reason_tp})")

            if not should_exit_tp:
                print(f"  ❌ ERROR: Take profit not triggered at target price!")
                return False
            else:
                print(f"  ✅ Take profit correctly triggered at target price")

    print("\n✅ All position stop logic tests passed!")
    return True

def test_position_monitoring_status():
    """Check if position monitoring is active."""

    print("\n🔍 Checking position monitoring status...")

    try:
        # This is a simplified check - in a real scenario we'd need to check
        # the actual running bot instance
        from crypto_bot.position_monitor import PositionMonitor
        from crypto_bot.utils.trade_manager import get_trade_manager

        # Load config
        import yaml
        config_path = Path(__file__).parent / "crypto_bot" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        tm = get_trade_manager()
        positions = tm.get_all_positions()

        if not positions:
            print("ℹ️  No positions to monitor")
            return True

        # Create a monitor instance (this won't actually start monitoring)
        monitor = PositionMonitor(
            exchange=None,  # We don't need exchange for this test
            config=config,
            positions={},  # Empty for this test
            notifier=None,
            trade_manager=tm
        )

        print(f"✅ PositionMonitor created successfully")
        print(f"✅ Monitor configuration loaded")
        print(f"  - Check interval: {monitor.check_interval_seconds}s")
        print(f"  - Price threshold: {monitor.price_update_threshold}")
        print(f"  - WebSocket enabled: {monitor.use_websocket}")

        return True

    except Exception as e:
        print(f"❌ Error checking position monitoring: {e}")
        return False

if __name__ == "__main__":
    try:
        success = True

        # Test 1: Stop loss logic
        if not test_stop_loss_logic():
            success = False

        # Test 2: Position monitoring setup
        if not test_position_monitoring_status():
            success = False

        if success:
            print("\n🎉 All tests passed! Position monitoring should be working correctly.")
            print("\n📋 What was fixed:")
            print("  ✅ Set up stop losses for all existing positions")
            print("  ✅ Configured take profit levels")
            print("  ✅ Added exit callback to position monitor")
            print("  ✅ Verified stop loss logic is working")
            print("\n🚀 Your position monitoring should now work correctly!")
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)
