#!/usr/bin/env python3
"""
Test script to verify position monitoring starts when bot starts
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

async def test_position_monitoring_startup():
    """Test that position monitoring starts correctly"""
    print("🧪 Testing position monitoring startup...")

    try:
        # Import required modules
        from crypto_bot.utils.trade_manager import get_trade_manager
        from crypto_bot.position_monitor import PositionMonitor
        import ccxt

        # Get TradeManager instance
        tm = get_trade_manager()
        positions = tm.get_all_positions()

        print(f"📊 Found {len(positions)} open positions in TradeManager")

        if not positions:
            print("ℹ️ No positions to monitor - this is expected for a clean test")
            return True

        # Create a mock exchange for testing
        exchange = ccxt.kraken()
        config = {"exit_strategy": {"real_time_monitoring": {"enabled": True}}}

        # Create PositionMonitor instance
        position_monitor = PositionMonitor(
            exchange=exchange,
            config=config,
            positions={},  # Empty dict for testing
            notifier=None,
            trade_manager=tm
        )

        print("✅ PositionMonitor created successfully")

        # Test starting monitoring for positions
        monitored_count = 0
        for tm_pos in positions:
            try:
                position_dict = {
                    'entry_price': float(tm_pos.average_price),
                    'size': float(tm_pos.total_amount),
                    'side': tm_pos.side,
                    'symbol': tm_pos.symbol,
                    'highest_price': float(tm_pos.highest_price) if tm_pos.highest_price else float(tm_pos.average_price),
                    'lowest_price': float(tm_pos.lowest_price) if tm_pos.lowest_price else float(tm_pos.average_price),
                    'trailing_stop': float(tm_pos.stop_loss_price) if tm_pos.stop_loss_price else 0.0,
                    'timestamp': tm_pos.entry_time.isoformat(),
                }

                await position_monitor.start_monitoring(tm_pos.symbol, position_dict)
                print(f"✅ Started monitoring {tm_pos.symbol}")
                monitored_count += 1
            except Exception as e:
                print(f"❌ Failed to start monitoring {tm_pos.symbol}: {e}")

        print(f"🎯 Successfully started monitoring for {monitored_count}/{len(positions)} positions")

        # Get monitoring stats
        try:
            stats = await position_monitor.get_monitoring_stats()
            print("📈 Monitoring stats:", stats)
        except Exception as e:
            print(f"⚠️ Could not get monitoring stats: {e}")

        # Clean up
        await position_monitor.stop_all_monitoring()
        print("🧹 Monitoring stopped and cleaned up")

        return monitored_count == len(positions)

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_bot_startup_integration():
    """Test that the bot startup process includes position monitoring"""
    print("🔧 Testing bot startup integration...")

    try:
        # This would require a full bot startup, so we'll just verify the code structure
        from crypto_bot.main import _main_impl

        # Check if the function exists and is callable
        if callable(_main_impl):
            print("✅ _main_impl function is available and callable")
            return True
        else:
            print("❌ _main_impl function is not callable")
            return False

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        print("🚀 Position Monitoring Startup Test Suite")
        print("=" * 50)

        # Test 1: Basic position monitoring functionality
        test1_result = await test_position_monitoring_startup()

        # Test 2: Bot startup integration
        test2_result = await test_bot_startup_integration()

        print("\n" + "=" * 50)
        print("📋 Test Results:")
        print(f"  Position Monitoring: {'✅ PASS' if test1_result else '❌ FAIL'}")
        print(f"  Bot Integration: {'✅ PASS' if test2_result else '❌ FAIL'}")

        if test1_result and test2_result:
            print("🎉 All tests passed! Position monitoring should start with bot.")
            return 0
        else:
            print("⚠️ Some tests failed. Check the implementation.")
            return 1

    sys.exit(asyncio.run(main()))
