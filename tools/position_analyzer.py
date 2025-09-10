#!/usr/bin/env python3
"""
Position Analyzer

This script analyzes your current trading positions and explains P&L calculations.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "crypto_bot" / "logs"
TRADES_FILE = LOGS_DIR / "trades.csv"


def analyze_positions():
    """Analyze current positions and P&L status using TradeManager as primary source."""
    # Try to analyze from TradeManager first
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        positions = trade_manager.get_all_positions()
        if positions:
            logger.info(f"Analyzing {len(positions)} positions from TradeManager")
            logger.info("\n📊 POSITION ANALYSIS (TradeManager)")
            logger.info("=" * 50)

            total_realized_pnl = float(trade_manager.total_realized_pnl)
            total_unrealized_pnl = 0.0
            total_open_positions = 0

            for pos in positions:
                if pos.is_open:
                    total_open_positions += 1
                    current_price = float(trade_manager.price_cache.get(pos.symbol, pos.average_price))

                    if current_price > 0:
                        from decimal import Decimal
                        pnl, pnl_pct = pos.calculate_unrealized_pnl(Decimal(str(current_price)))
                        total_unrealized_pnl += float(pnl)

                        logger.info(f"Symbol: {pos.symbol}")
                        logger.info(f"  Side: {pos.side}")
                        logger.info(f"  Amount: {pos.total_amount}")
                        logger.info(f"  Entry Price: ${pos.average_price}")
                        logger.info(f"  Current Price: ${current_price}")
                        logger.info(f"  P&L: ${float(pnl):.2f} ({float(pnl_pct):.2f}%)")
                        logger.info(f"  Entry Time: {pos.entry_time}")
                        logger.info("")

            logger.info("SUMMARY:")
            logger.info(f"  Open Positions: {total_open_positions}")
            logger.info(f"  Realized P&L: ${total_realized_pnl:.2f}")
            logger.info(f"  Unrealized P&L: ${total_unrealized_pnl:.2f}")
            logger.info(f"  Total P&L: ${total_realized_pnl + total_unrealized_pnl:.2f}")

            return positions

    except Exception as e:
        logger.warning(f"Failed to analyze from TradeManager: {e}, falling back to CSV")

    # Fallback to CSV analysis (deprecated)
    logger.info("\n📄 FALLBACK: POSITION ANALYSIS FROM CSV (DEPRECATED)")
    logger.info("=" * 50)

    if not TRADES_FILE.exists():
        logger.error(f"Trades file not found: {TRADES_FILE}")
        return None

    try:
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"Loaded {len(df)} trades from {TRADES_FILE}")

        if df.empty:
            logger.info("No trades found")
            return None

        # Analyze each trade
        logger.info("\n📊 POSITION ANALYSIS FROM CSV")
        logger.info("=" * 50)
        
        open_positions = {}
        closed_trades = []
        total_invested = 0
        
        for idx, row in df.iterrows():
            symbol = row['symbol']
            side = row['side']
            amount = float(row['amount'])
            price = float(row['price'])
            timestamp = row['timestamp']
            
            trade_value = amount * price
            
            logger.info(f"\n🔸 Trade {idx + 1}:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Side: {side.upper()}")
            logger.info(f"   Amount: {amount:,.6f}")
            logger.info(f"   Price: ${price:,.6f}")
            logger.info(f"   Value: ${trade_value:,.2f}")
            logger.info(f"   Date: {timestamp}")
            
            if side.lower() == 'buy':
                # This is an open position
                if symbol not in open_positions:
                    open_positions[symbol] = []
                open_positions[symbol].append({
                    'amount': amount,
                    'price': price,
                    'value': trade_value,
                    'timestamp': timestamp
                })
                total_invested += trade_value
                logger.info(f"   Status: ✅ OPEN POSITION")
                
            elif side.lower() == 'sell':
                # This would close a position (calculate P&L)
                closed_trades.append(row)
                logger.info(f"   Status: ✅ CLOSED POSITION")
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📈 POSITION SUMMARY")
        logger.info("=" * 50)
        
        logger.info(f"\n💰 Open Positions: {len(open_positions)}")
        for symbol, positions in open_positions.items():
            total_amount = sum(p['amount'] for p in positions)
            avg_price = sum(p['value'] for p in positions) / total_amount if total_amount > 0 else 0
            total_value = sum(p['value'] for p in positions)
            logger.info(f"   {symbol}: {total_amount:,.6f} @ ${avg_price:,.6f} = ${total_value:,.2f}")
        
        logger.info(f"\n🔄 Closed Trades: {len(closed_trades)}")
        if not closed_trades:
            logger.info("   None - No sell trades found")
        
        logger.info(f"\n💵 Total Invested: ${total_invested:,.2f}")
        
        # P&L Analysis
        logger.info("\n" + "=" * 50)
        logger.info("📊 P&L ANALYSIS")
        logger.info("=" * 50)
        
        if not closed_trades:
            logger.info("\n❌ NO REALIZED P&L")
            logger.info("   Reason: No sell trades found")
            logger.info("   All trades are buy orders creating open positions")
            logger.info("   P&L can only be calculated when positions are closed with sell orders")
            
            logger.info("\n💡 To calculate P&L, you need:")
            logger.info("   1. Sell orders to close positions")
            logger.info("   2. The sell price compared to buy price determines profit/loss")
            logger.info("   3. P&L = (Sell Price - Buy Price) × Amount")
        else:
            logger.info(f"\n✅ Found {len(closed_trades)} closed trades")
            # Calculate P&L for closed trades
            # (This would be implemented if there were sell trades)
        
        # Current Status
        logger.info("\n" + "=" * 50)
        logger.info("🎯 CURRENT STATUS")
        logger.info("=" * 50)
        logger.info(f"   • You have {len(open_positions)} open positions")
        logger.info(f"   • You have {len(closed_trades)} closed trades")
        logger.info(f"   • Total capital deployed: ${total_invested:,.2f}")
        logger.info("   • All positions are currently unrealized (open)")
        
        if len(closed_trades) == 0:
            logger.info("\n⚠️  P&L EXPLANATION:")
            logger.info("   Your P&L shows as $0.00 because you have no completed trades.")
            logger.info("   P&L only shows when you sell positions, creating realized gains/losses.")
            logger.info("   Current positions represent unrealized P&L (paper profits/losses).")
        
        return {
            'open_positions': open_positions,
            'closed_trades': closed_trades,
            'total_invested': total_invested
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze positions: {e}")
        return None


def check_trading_system():
    """Check if the trading system has proper sell functionality."""
    logger.info("\n" + "=" * 50)
    logger.info("🔧 TRADING SYSTEM CHECK")
    logger.info("=" * 50)
    
    # Check execution log for sell order attempts
    execution_log = LOGS_DIR / "execution.log"
    if execution_log.exists():
        with open(execution_log, 'r') as f:
            content = f.read()
        
        sell_attempts = content.count('side": "sell"')
        sell_errors = content.count('Order missing required fields')
        sell_successes = content.count('Order executed.*sell')
        
        logger.info(f"\n📋 Sell Order Analysis:")
        logger.info(f"   • Sell order attempts: {sell_attempts}")
        logger.info(f"   • Sell order errors: {sell_errors}")
        logger.info(f"   • Successful sells: {sell_successes}")
        
        if sell_errors > 0:
            logger.info(f"\n⚠️  Found {sell_errors} sell order errors")
            logger.info("   This suggests the bot tried to sell but failed due to missing data")
            logger.info("   Common issues: missing amount, invalid symbol, insufficient balance")


def main():
    """Main analysis function."""
    logger.info("🔍 Starting position analysis...")
    
    # Analyze positions
    result = analyze_positions()
    
    # Check trading system
    check_trading_system()
    
    logger.info("\n" + "=" * 50)
    logger.info("✅ Analysis complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
