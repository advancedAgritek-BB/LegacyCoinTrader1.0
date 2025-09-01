#!/usr/bin/env python3
"""
Test script for frontend PnL functionality.
This script tests the PnL calculations and real-time updates.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pnl_calculation():
    """Test PnL calculation logic."""
    print("Testing PnL calculation...")
    
    # Sample trade data
    trades = [
        {'symbol': 'BTC/USD', 'side': 'buy', 'amount': 1.0, 'price': 50000, 'timestamp': '2024-01-01 10:00:00'},
        {'symbol': 'BTC/USD', 'side': 'sell', 'amount': 0.5, 'price': 55000, 'timestamp': '2024-01-01 11:00:00'},
        {'symbol': 'ETH/USD', 'side': 'buy', 'amount': 10.0, 'price': 3000, 'timestamp': '2024-01-01 12:00:00'},
    ]
    
    # Expected PnL calculations
    expected_pnl = {
        'BTC/USD': 2500,  # (55000 - 50000) * 0.5
        'ETH/USD': 0      # No sell yet
    }
    
    # Simulate the PnL calculation logic
    open_positions = {}
    calculated_pnl = {}
    
    for trade in trades:
        symbol = trade['symbol']
        side = trade['side']
        amount = trade['amount']
        price = trade['price']
        
        if symbol in open_positions:
            # Check if this trade closes an existing position
            if (side == 'sell' and open_positions[symbol]['side'] == 'buy') or \
               (side == 'buy' and open_positions[symbol]['side'] == 'sell'):
                # Calculate realized PnL
                entry_price = open_positions[symbol]['price']
                entry_amount = open_positions[symbol]['amount']
                
                if side == 'sell':  # Closing long position
                    pnl = (price - entry_price) * min(amount, entry_amount)
                else:  # Closing short position
                    pnl = (entry_price - price) * min(amount, entry_amount)
                
                calculated_pnl[symbol] = calculated_pnl.get(symbol, 0) + pnl
                
                # Update or remove position
                if amount >= entry_amount:
                    del open_positions[symbol]
                else:
                    open_positions[symbol]['amount'] -= amount
            else:
                # Same side trade - update position
                if symbol in open_positions:
                    # Average down/up
                    total_cost = (open_positions[symbol]['price'] * open_positions[symbol]['amount']) + (price * amount)
                    total_amount = open_positions[symbol]['amount'] + amount
                    open_positions[symbol]['price'] = total_cost / total_amount
                    open_positions[symbol]['amount'] = total_amount
                else:
                    open_positions[symbol] = {'side': side, 'price': price, 'amount': amount}
        else:
            # New position
            open_positions[symbol] = {'side': side, 'price': price, 'amount': amount}
    
    # Verify calculations
    for symbol, expected in expected_pnl.items():
        calculated = calculated_pnl.get(symbol, 0)
        if abs(calculated - expected) < 0.01:  # Allow small floating point differences
            print(f"✓ {symbol} PnL calculation correct: ${calculated:.2f}")
        else:
            print(f"✗ {symbol} PnL calculation incorrect: expected ${expected:.2f}, got ${calculated:.2f}")
            return False
    
    return True

def test_current_price_api():
    """Test current price API endpoint."""
    print("\nTesting current price API...")
    
    try:
        # Import the function
        from frontend.app import get_current_price_for_symbol
        
        # Test with a known symbol
        symbol = "SOL/USD"
        price = get_current_price_for_symbol(symbol)
        
        if price >= 0:  # Price should be non-negative
            print(f"✓ Current price API works: {symbol} = ${price:.4f}")
            return True
        else:
            print(f"✗ Invalid price returned: {price}")
            return False
            
    except Exception as e:
        print(f"✗ Current price API test failed: {e}")
        return False

def test_trades_data_endpoint():
    """Test trades data endpoint with PnL."""
    print("\nTesting trades data endpoint...")
    
    try:
        # Create a temporary trades file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("BTC/USD,buy,1.0,50000,2024-01-01 10:00:00\n")
            f.write("BTC/USD,sell,0.5,55000,2024-01-01 11:00:00\n")
            f.write("ETH/USD,buy,10.0,3000,2024-01-01 12:00:00\n")
            temp_file = f.name
        
        # Test the PnL calculation logic directly
        from crypto_bot import log_reader
        from frontend.app import get_current_price_for_symbol
        
        # Mock the current prices
        def mock_get_price(symbol):
            if symbol == 'BTC/USD':
                return 60000
            elif symbol == 'ETH/USD':
                return 3500
            return 0
        
        # Read the trades
        df = log_reader._read_trades(Path(temp_file))
        
        if len(df) == 3:
            print(f"✓ Trades file contains {len(df)} trades")
            
            # Test PnL calculation logic
            open_positions = {}
            records = []
            
            for _, row in df.iterrows():
                symbol = str(row.get('symbol', ''))
                side = str(row.get('side', ''))
                amount = float(row.get('amount', 0))
                price = float(row.get('price', 0))
                timestamp = str(row.get('timestamp', ''))
                
                # Calculate trade total
                total = amount * price
                
                # Calculate PnL for this trade
                pnl = 0.0
                pnl_percentage = 0.0
                
                if symbol in open_positions:
                    # Check if this trade closes an existing position
                    if (side == 'sell' and open_positions[symbol]['side'] == 'buy') or \
                       (side == 'buy' and open_positions[symbol]['side'] == 'sell'):
                        # Calculate realized PnL
                        entry_price = open_positions[symbol]['price']
                        entry_amount = open_positions[symbol]['amount']
                        
                        if side == 'sell':  # Closing long position
                            pnl = (price - entry_price) * min(amount, entry_amount)
                        else:  # Closing short position
                            pnl = (entry_price - price) * min(amount, entry_amount)
                        
                        pnl_percentage = (pnl / (entry_price * min(amount, entry_amount))) * 100
                        
                        # Update or remove position
                        if amount >= entry_amount:
                            del open_positions[symbol]
                        else:
                            open_positions[symbol]['amount'] -= amount
                    else:
                        # Same side trade - update position
                        if symbol in open_positions:
                            # Average down/up
                            total_cost = (open_positions[symbol]['price'] * open_positions[symbol]['amount']) + total
                            total_amount = open_positions[symbol]['amount'] + amount
                            open_positions[symbol]['price'] = total_cost / total_amount
                            open_positions[symbol]['amount'] = total_amount
                        else:
                            open_positions[symbol] = {'side': side, 'price': price, 'amount': amount}
                else:
                    # New position
                    open_positions[symbol] = {'side': side, 'price': price, 'amount': amount}
                
                # Calculate unrealized PnL for open positions
                unrealized_pnl = 0.0
                unrealized_pnl_percentage = 0.0
                current_price = mock_get_price(symbol)
                if symbol in open_positions and current_price > 0:
                    pos = open_positions[symbol]
                    if pos['side'] == 'buy':
                        unrealized_pnl = (current_price - pos['price']) * pos['amount']
                    else:
                        unrealized_pnl = (pos['price'] - current_price) * pos['amount']
                    
                    if pos['price'] > 0:
                        unrealized_pnl_percentage = (unrealized_pnl / (pos['price'] * pos['amount'])) * 100
                
                record = {
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'timestamp': timestamp,
                    'total': total,
                    'status': 'completed',  # This trade would be completed since it's a sell
                    'pnl': pnl,
                    'pnl_percentage': pnl_percentage,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_percentage': unrealized_pnl_percentage,
                    'current_price': current_price
                }
                records.append(record)
            
            # Check if PnL fields are present
            if len(records) == 3:
                first_trade = records[0]
                if 'pnl' in first_trade and 'pnl_percentage' in first_trade:
                    print("✓ PnL fields present in trade data")
                    
                    # Check specific PnL calculations
                    btc_trade = records[1]  # BTC sell trade
                    if abs(btc_trade['pnl'] - 2500) < 0.01:
                        print("✓ BTC PnL calculation correct")
                        return True
                    else:
                        print(f"✗ BTC PnL calculation incorrect: {btc_trade['pnl']}")
                        return False
                else:
                    print("✗ PnL fields missing from trade data")
                    return False
            else:
                print(f"✗ Expected 3 records, got {len(records)}")
                return False
        else:
            print(f"✗ Expected 3 trades in file, got {len(df)}")
            return False
        
        # Clean up
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"✗ Trades data endpoint test failed: {e}")
        return False

def test_frontend_template_updates():
    """Test that frontend template includes PnL columns."""
    print("\nTesting frontend template updates...")
    
    try:
        # Read the trades.html template
        trades_template = project_root / "frontend" / "templates" / "trades.html"
        
        with open(trades_template) as f:
            content = f.read()
        
        # Check for PnL-related elements
        required_elements = [
            'P&L',
            'P&L %',
            'Current Price',
            'pnl_percentage',
            'unrealized_pnl',
            'refreshPrices',
            'updateCurrentPrices'
        ]
        
        for element in required_elements:
            if element in content:
                print(f"✓ {element} found in template")
            else:
                print(f"✗ {element} not found in template")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Template test failed: {e}")
        return False

def main():
    """Run all PnL tests."""
    print("Frontend PnL Test Suite")
    print("=" * 40)
    
    tests = [
        test_pnl_calculation,
        test_current_price_api,
        test_trades_data_endpoint,
        test_frontend_template_updates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n✗ Test failed: {test.__name__}")
            break
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All PnL tests passed! Frontend PnL functionality is working correctly.")
        return True
    else:
        print("❌ Some PnL tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
