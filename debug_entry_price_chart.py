#!/usr/bin/env python3
"""
Debug script to analyze entry price positioning on charts for long positions with positive P&L.
"""

import json
import os
from pathlib import Path
from decimal import Decimal
from datetime import datetime

def load_trade_manager_state():
    """Load the current trade manager state."""
    state_file = Path("crypto_bot/logs/trade_manager_state.json")
    if not state_file.exists():
        print("❌ Trade manager state file not found")
        return None

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)

        positions = state.get('positions', {})
        price_cache = state.get('price_cache', {})

        print("📊 Trade Manager State Loaded:")
        print(f"   Positions: {len(positions)}")
        print(f"   Price Cache: {len(price_cache)}")

        return positions, price_cache
    except Exception as e:
        print(f"❌ Error loading trade manager state: {e}")
        return None

def analyze_position_chart_data(positions, price_cache):
    """Analyze position data and chart positioning logic."""
    print("\n🔍 Analyzing Position Chart Data:\n")

    for symbol, pos_data in positions.items():
        if pos_data.get('total_amount', 0) <= 0:
            continue  # Skip closed positions

        amount = pos_data['total_amount']
        avg_price = pos_data['average_price']
        side = pos_data['side']
        current_price = price_cache.get(symbol, avg_price)

        print(f"📈 Position: {symbol} ({side})")
        print(f"   Entry Price: ${avg_price}")
        print(f"   Current Price: ${current_price}")
        print(f"   Amount: {amount}")

        # Calculate P&L
        if side == 'long':
            pnl = (current_price - avg_price) * amount
            pnl_type = "LONG"
        else:
            pnl = (avg_price - current_price) * amount
            pnl_type = "SHORT"

        pnl_pct = (pnl / (avg_price * amount)) * 100 if avg_price > 0 else 0

        print(f"   P&L ({pnl_type}): ${pnl:.2f} ({pnl_pct:.2f}%)")
        print(f"   Status: {'PROFIT' if pnl > 0 else 'LOSS' if pnl < 0 else 'BREAK-EVEN'}")

        # Simulate chart coordinate calculation
        print("\n📊 Chart Coordinate Analysis:")

        # Mock chart data for testing
        min_price = min(avg_price, current_price) * 0.95  # 5% below lowest
        max_price = max(avg_price, current_price) * 1.05  # 5% above highest
        price_range = max_price - min_price
        canvas_height = 120  # Chart height from dashboard

        print(f"   Chart Range: ${min_price:.6f} - ${max_price:.6f}")
        print(f"   Price Range: ${price_range:.6f}")
        print(f"   Canvas Height: {canvas_height}px")

        # Calculate Y coordinates (higher prices = lower Y values)
        current_y = canvas_height - ((current_price - min_price) / price_range) * canvas_height
        entry_y = canvas_height - ((avg_price - min_price) / price_range) * canvas_height

        print("\n📍 Y-Coordinates (0 = top, 120 = bottom):")
        print(f"   Current Price Y: {current_y:.1f}px")
        print(f"   Entry Price Y: {entry_y:.1f}px")

        # Analysis
        if side == 'long' and pnl > 0:
            print("\n🎯 LONG POSITION PROFIT ANALYSIS:")
            if current_y < entry_y:
                print("   ✅ CORRECT: Current price above entry price")
                print("   ✅ Entry price line should be below current price")
            else:
                print("   ❌ ISSUE: Current price below entry price despite positive P&L")
                print("   ❌ This suggests a chart rendering problem")

        elif side == 'long' and pnl < 0:
            print("\n📉 LONG POSITION LOSS ANALYSIS:")
            if current_y > entry_y:
                print("   ✅ CORRECT: Current price below entry price")
                print("   ✅ Entry price line should be above current price")
            else:
                print("   ❌ ISSUE: Current price above entry price despite negative P&L")

        print("   " + "="*60 + "\n")

def analyze_chart_api_data():
    """Analyze chart data from API endpoints."""
    print("\n🔍 Analyzing Chart API Data:\n")

    # Try to load trend data if available
    try:
        from frontend.app import calculate_trend_line
        print("✅ Trend calculation function available")

        # Mock candle data for testing
        mock_candles = [
            {'close': 50000, 'timestamp': '2024-01-01T00:00:00Z'},
            {'close': 50100, 'timestamp': '2024-01-01T01:00:00Z'},
            {'close': 50200, 'timestamp': '2024-01-01T02:00:00Z'},
            {'close': 50300, 'timestamp': '2024-01-01T03:00:00Z'},
            {'close': 50400, 'timestamp': '2024-01-01T04:00:00Z'},
        ]

        trend_data = calculate_trend_line(mock_candles)
        print("📈 Mock Trend Analysis:")
        print(f"   Direction: {trend_data['direction']}")
        print(f"   Strength: {trend_data['strength']}")
        print(f"   R²: {trend_data['r_squared']}")
        print(f"   Slope: {trend_data['slope']}")
        print(f"   Intercept: {trend_data['intercept']}")

        print("   Trend Points:")
        for i, point in enumerate(trend_data['trend_points'][:3]):  # Show first 3
            print(f"     {i}: ${point['price']:.2f}")

    except Exception as e:
        print(f"❌ Error analyzing trend data: {e}")

def main():
    """Main analysis function."""
    print("🔧 Entry Price Chart Analysis Tool")
    print("="*50)

    # Load and analyze trade manager data
    state_data = load_trade_manager_state()
    if state_data:
        positions, price_cache = state_data
        analyze_position_chart_data(positions, price_cache)

    # Analyze chart API data
    analyze_chart_api_data()

    print("\n✅ Analysis Complete")
    print("\n💡 If you see positioning issues, the problem may be in:")
    print("   1. Chart Y-coordinate calculation")
    print("   2. Price range scaling")
    print("   3. Trend line calculation")
    print("   4. Canvas rendering logic")

if __name__ == "__main__":
    main()
