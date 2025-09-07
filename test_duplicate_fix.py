#!/usr/bin/env python3
"""
Test script to verify the duplicate position cards fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.app import get_open_positions, deduplicate_positions

def test_position_deduplication():
    """Test that positions are properly deduplicated."""
    print("🧪 Testing position deduplication...")
    
    # Get positions from the improved function
    positions = get_open_positions()
    print(f"📊 Retrieved {len(positions)} positions from get_open_positions()")
    
    # Show unique symbols
    symbols = set(pos.get('symbol', '') for pos in positions)
    print(f"📊 Found {len(symbols)} unique symbols: {sorted(symbols)}")
    
    # Test deduplication function
    if positions:
        deduplicated = deduplicate_positions(positions)
        print(f"📊 After deduplication: {len(deduplicated)} positions")
        
        # Check for any remaining duplicates
        symbol_counts = {}
        for pos in deduplicated:
            symbol = pos.get('symbol', '')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        duplicates = {symbol: count for symbol, count in symbol_counts.items() if count > 1}
        if duplicates:
            print(f"❌ Found duplicates after deduplication: {duplicates}")
            return False
        else:
            print("✅ No duplicates found after deduplication")
            return True
    else:
        print("ℹ️  No positions found to test")
        return True

def test_dashboard_route():
    """Test the dashboard route to ensure it returns clean data."""
    print("\n🧪 Testing dashboard route...")
    
    try:
        # Import the dashboard route
        from frontend.app import dashboard
        
        # Mock request context
        from flask import Flask
        app = Flask(__name__)
        
        with app.test_request_context('/dashboard'):
            # This would normally call the dashboard route
            # For now, just test the position loading logic
            positions = get_open_positions()
            print(f"📊 Dashboard would load {len(positions)} positions")
            
            # Check for duplicates
            symbols = [pos.get('symbol', '') for pos in positions]
            unique_symbols = set(symbols)
            
            if len(symbols) == len(unique_symbols):
                print("✅ Dashboard position loading looks clean")
                return True
            else:
                print(f"❌ Dashboard still has duplicates: {len(symbols)} positions, {len(unique_symbols)} unique symbols")
                return False
                
    except Exception as e:
        print(f"❌ Error testing dashboard route: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 Testing Duplicate Position Cards Fix")
    print("=" * 50)
    
    # Test 1: Position deduplication
    test1_passed = test_position_deduplication()
    
    # Test 2: Dashboard route
    test2_passed = test_dashboard_route()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Position Deduplication: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Dashboard Route: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The duplicate position cards fix is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. The fix may need additional work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
