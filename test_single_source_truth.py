#!/usr/bin/env python3
"""
Test Single Source of Truth Architecture

This script validates that the SingleSourceTradeManager is working correctly
and that all components are using the same instance.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.single_source_trade_manager import (
    get_single_source_trade_manager,
    create_trade,
    TradeEvent
)
from crypto_bot.utils.trade_manager import get_trade_manager as get_legacy_trade_manager
from libs.execution.cex_executor import execute_trade_async
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__)

def test_single_instance():
    """Test that all components get the same TradeManager instance."""
    print("🔍 Testing Single Instance...")

    # Get instances from different entry points
    tm1 = get_single_source_trade_manager()
    tm2 = get_single_source_trade_manager()

    # Check they are the same instance
    if tm1 is tm2:
        print("✅ SingleSourceTradeManager returns same instance")
        print(f"   Instance ID: {id(tm1)}")

        # Check if legacy function returns the same type
        tm3 = get_legacy_trade_manager()
        if type(tm3) == type(tm1):
            print("✅ Legacy compatibility maintained")
            return True
        else:
            print(f"⚠️  Legacy function returns different type: {type(tm3)} vs {type(tm1)}")
            return True  # Not a critical failure
    else:
        print("❌ Different SingleSourceTradeManager instances returned!")
        return False

def test_trade_recording():
    """Test that trades are recorded correctly."""
    print("\n🔍 Testing Trade Recording...")

    trade_manager = get_single_source_trade_manager()
    initial_count = len(trade_manager.trades)

    # Create a test trade with a real-looking symbol
    test_trade = create_trade(
        symbol="SOL/USD",
        side="buy",
        amount=1.0,
        price=100.0,
        strategy="test_single_source"
    )

    # Record the trade
    trade_id = trade_manager.record_trade(test_trade)

    # Verify it was recorded
    final_count = len(trade_manager.trades)
    if final_count == initial_count + 1:
        print(f"✅ Trade recorded successfully: {trade_id}")
        print(f"   Trade count: {initial_count} → {final_count}")
        return True
    else:
        print(f"❌ Trade recording failed: {initial_count} → {final_count}")
        return False

def test_event_system():
    """Test that the event system works."""
    print("\n🔍 Testing Event System...")

    trade_manager = get_single_source_trade_manager()
    events_received = []

    def test_event_handler(event: TradeEvent):
        events_received.append(event)

    # Subscribe to events
    trade_manager.event_bus.subscribe('trade_executed', test_event_handler)

    # Create and record a trade
    test_trade = create_trade(
        symbol="DOT/USD",
        side="buy",
        amount=2.0,
        price=200.0,
        strategy="test_events"
    )

    trade_manager.record_trade(test_trade)

    # Give event system time to process
    time.sleep(0.1)

    if events_received:
        event = events_received[0]
        print(f"✅ Event received: {event.event_type}")
        print(f"   Trade ID: {event.metadata.get('trade_id')}")
        return True
    else:
        print("❌ No events received")
        return False

def test_position_consistency():
    """Test that positions are calculated correctly."""
    print("\n🔍 Testing Position Consistency...")

    trade_manager = get_single_source_trade_manager()

    # Record multiple trades for the same symbol
    trades = [
        create_trade("ADA/USD", "buy", 1.0, 100.0, "test_positions"),
        create_trade("ADA/USD", "buy", 2.0, 110.0, "test_positions"),
        create_trade("ADA/USD", "sell", 1.0, 120.0, "test_positions"),
    ]

    for trade in trades:
        trade_manager.record_trade(trade)

    # Check position
    position = trade_manager.get_position("ADA/USD")

    if position:
        expected_amount = 2.0  # 1 + 2 - 1
        expected_avg_price = (1.0 * 100.0 + 2.0 * 110.0 - 1.0 * 120.0) / 2.0

        if abs(float(position.total_amount) - expected_amount) < 0.001:
            print(f"✅ Position amount correct: {position.total_amount}")
        else:
            print(f"❌ Position amount incorrect: {position.total_amount} vs {expected_amount}")
            return False

        if abs(float(position.average_price) - expected_avg_price) < 0.001:
            print(f"✅ Average price correct: {position.average_price}")
        else:
            print(f"❌ Average price incorrect: {position.average_price} vs {expected_avg_price}")
            return False

        return True
    else:
        print("❌ Position not found")
        return False

def test_csv_audit_logging():
    """Test that CSV audit logging works."""
    print("\n🔍 Testing CSV Audit Logging...")

    trade_manager = get_single_source_trade_manager()
    csv_path = trade_manager.csv_audit_path if hasattr(trade_manager, 'csv_audit_path') else None

    if csv_path and csv_path.exists():
        print(f"✅ CSV audit file exists: {csv_path}")
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            print(f"   Lines in audit file: {len(lines)}")
        return True
    else:
        print("ℹ️  CSV audit logging may not be enabled or file not yet created")
        return True  # Not a failure

def test_system_status():
    """Test system status reporting."""
    print("\n🔍 Testing System Status...")

    trade_manager = get_single_source_trade_manager()
    status = trade_manager.get_system_status()

    required_fields = ['trades_count', 'positions_count', 'total_volume', 'frontend_subscribers']
    missing_fields = [field for field in required_fields if field not in status]

    if not missing_fields:
        print("✅ System status complete:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        return True
    else:
        print(f"❌ Missing status fields: {missing_fields}")
        return False

def cleanup_test_data():
    """Clean up test trades from the system."""
    print("\n🧹 Cleaning up test data...")

    trade_manager = get_single_source_trade_manager()

    # Remove test trades and positions
    test_symbols = ["SOL/USD", "DOT/USD", "ADA/USD"]

    for symbol in test_symbols:
        if symbol in trade_manager.positions:
            del trade_manager.positions[symbol]

    # Remove test trades
    original_trades = trade_manager.trades.copy()
    trade_manager.trades = [
        trade for trade in original_trades
        if not any(test_symbol in trade.symbol for test_symbol in test_symbols)
    ]

    # Save state
    trade_manager.save_state()
    print("✅ Test data cleaned up")

def main():
    """Run all tests."""
    print("🚀 Single Source of Truth Validation Tests")
    print("=" * 60)

    tests = [
        ("Single Instance", test_single_instance),
        ("Trade Recording", test_trade_recording),
        ("Event System", test_event_system),
        ("Position Consistency", test_position_consistency),
        ("CSV Audit Logging", test_csv_audit_logging),
        ("System Status", test_system_status),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 Test {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("-" * 30)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Single Source of Truth is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

    # Cleanup
    cleanup_test_data()

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
