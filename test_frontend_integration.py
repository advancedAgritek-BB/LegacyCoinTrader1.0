#!/usr/bin/env python3
"""
Test Frontend TradeManager Integration

This script simulates the frontend API call to test TradeManager integration.
"""

import sys
from pathlib import Path

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

def test_wallet_pnl():
    """Simulate the frontend wallet-pnl API call."""
    try:
        # Try to get data from TradeManager first
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        print(f"DEBUG: TradeManager has {len(trade_manager.trades)} trades")
        print(f"DEBUG: TradeManager has {len(trade_manager.positions)} positions")

        summary = trade_manager.get_portfolio_summary()
        print(f"DEBUG: Portfolio summary: {summary}")

        # Convert to the expected format
        pnl_data = {
            'total_pnl': float(summary['total_pnl']),
            'pnl_percentage': float(summary['total_unrealized_pnl_pct']),
            'realized_pnl': float(summary['total_realized_pnl']),
            'unrealized_pnl': float(summary['total_unrealized_pnl']),
            'balance': 10000.0,  # Default initial balance
            'initial_balance': 10000.0
        }

        print(f"DEBUG: Returning PnL data: {pnl_data}")
        return pnl_data

    except Exception as e:
        print(f"DEBUG: Exception in test_wallet_pnl: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == "__main__":
    print("🧪 Testing Frontend TradeManager Integration")
    print("=" * 50)
    
    result = test_wallet_pnl()
    print(f"\n✅ Result: {result}")
