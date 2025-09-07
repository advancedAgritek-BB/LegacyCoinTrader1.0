#!/usr/bin/env python3
"""
Proper Wallet Balance Analysis
This script correctly analyzes the wallet state including open positions.
"""

import yaml
from pathlib import Path

def analyze_wallet_state():
    """Analyze the current wallet state properly."""
    
    print("💰 PROPER WALLET BALANCE ANALYSIS")
    print("=" * 50)
    
    # Load current wallet state
    wallet_state_path = Path("crypto_bot/logs/paper_wallet_state.yaml")
    
    if not wallet_state_path.exists():
        print("❌ Wallet state file not found")
        return
    
    with open(wallet_state_path, 'r') as f:
        state = yaml.safe_load(f)
    
    # Extract key values
    cash_balance = state.get('balance', 0.0)
    initial_balance = state.get('initial_balance', 10000.0)
    positions = state.get('positions', {})
    
    print(f"\n📊 WALLET STATE:")
    print(f"   Initial Balance: ${initial_balance:.2f}")
    print(f"   Cash Balance: ${cash_balance:.2f}")
    print(f"   Open Positions: {len(positions)}")
    
    # Calculate position values
    total_position_value = 0.0
    print(f"\n📈 POSITION BREAKDOWN:")
    
    for trade_id, pos in positions.items():
        symbol = pos.get('symbol', trade_id)
        side = pos.get('side', 'Unknown')
        
        # Get amount/size
        if 'size' in pos:
            amount = pos['size']
        elif 'amount' in pos:
            amount = pos['amount']
        else:
            amount = 0
        
        entry_price = pos.get('entry_price', 0)
        position_value = amount * entry_price
        total_position_value += position_value
        
        print(f"   {symbol}: {side.upper()} {amount:.4f} @ ${entry_price:.6f} = ${position_value:.2f}")
    
    # Calculate total portfolio value
    total_portfolio_value = cash_balance + total_position_value
    
    print(f"\n💡 PORTFOLIO ANALYSIS:")
    print(f"   Cash Balance: ${cash_balance:.2f}")
    print(f"   Position Value: ${total_position_value:.2f}")
    print(f"   Total Portfolio Value: ${total_portfolio_value:.2f}")
    
    # Calculate metrics
    if initial_balance > 0:
        portfolio_return = ((total_portfolio_value - initial_balance) / initial_balance) * 100
        leverage_ratio = total_position_value / initial_balance
        
        print(f"   Portfolio Return: {portfolio_return:+.2f}%")
        print(f"   Leverage Ratio: {leverage_ratio:.2f}x")
    
    # Status assessment
    print(f"\n✅ STATUS ASSESSMENT:")
    if total_portfolio_value >= 0:
        print(f"   ✅ Portfolio is healthy (total value: ${total_portfolio_value:.2f})")
    else:
        print(f"   ❌ Portfolio has negative total value (${total_portfolio_value:.2f})")
    
    if cash_balance < 0 and len(positions) > 0:
        print(f"   ℹ️  Negative cash balance (${cash_balance:.2f}) is normal with open positions")
        print(f"   ℹ️  This represents the cost of your open positions")
    
    if len(positions) == 0 and cash_balance < 0:
        print(f"   ❌ Negative cash balance with no positions is problematic")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    if total_portfolio_value < initial_balance * 0.8:
        print(f"   ⚠️  Consider reducing position sizes (portfolio down {((total_portfolio_value - initial_balance) / initial_balance) * 100:.1f}%)")
    
    if leverage_ratio > 2.0:
        print(f"   ⚠️  High leverage detected ({leverage_ratio:.2f}x) - consider reducing positions")
    
    if cash_balance < -initial_balance * 0.5:
        print(f"   ⚠️  Large negative cash balance - ensure you have enough margin for position management")
    
    print(f"\n" + "=" * 50)

if __name__ == "__main__":
    analyze_wallet_state()
