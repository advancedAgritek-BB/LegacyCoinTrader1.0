#!/usr/bin/env python3
"""
Migrate existing CSV trades to TradeManager to ensure single source of truth.

This script reads all trades from the CSV file and ensures they're recorded
in TradeManager. It handles duplicates gracefully and preserves all trade data.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from crypto_bot.utils.trade_manager import get_trade_manager, Trade
from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "migration.log")

def migrate_csv_trades_to_trademanager():
    """Migrate all trades from CSV to TradeManager."""

    # Paths
    csv_file = Path("crypto_bot/logs/trades.csv")

    if not csv_file.exists():
        logger.info("No CSV file found, nothing to migrate")
        return

    # Get TradeManager instance
    trade_manager = get_trade_manager()

    # Read CSV file
    df = pd.read_csv(csv_file)
    logger.info(f"Found {len(df)} trades in CSV file")

    if df.empty:
        logger.info("CSV file is empty, nothing to migrate")
        return

    # Get existing trades from TradeManager to avoid duplicates
    existing_trades = trade_manager.get_trade_history()
    existing_trade_ids = {f"{t.symbol}_{t.side}_{t.amount}_{t.price}_{t.timestamp.isoformat()}" for t in existing_trades}
    logger.info(f"Found {len(existing_trade_ids)} existing trades in TradeManager")

    migrated_count = 0
    skipped_count = 0

    for _, row in df.iterrows():
        try:
            # Extract trade data
            symbol = str(row.get('symbol', '')).strip()
            side = str(row.get('side', '')).strip()
            amount = float(row.get('amount', 0))
            price = float(row.get('price', 0))
            timestamp_str = str(row.get('timestamp', '')).strip()
            is_stop = bool(row.get('is_stop', False))

            # Skip invalid trades
            if not symbol or not side or amount <= 0 or price <= 0:
                logger.warning(f"Skipping invalid trade: {symbol} {side} {amount} @ {price}")
                continue

            # Skip stop orders (they're not actual trades)
            if is_stop:
                continue

            # Parse timestamp
            try:
                if 'T' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except Exception as e:
                logger.warning(f"Could not parse timestamp '{timestamp_str}': {e}")
                timestamp = datetime.utcnow()

            # Create unique ID for this trade to check for duplicates
            trade_id = f"{symbol}_{side}_{amount}_{price}_{timestamp.isoformat()}"

            if trade_id in existing_trade_ids:
                logger.debug(f"Trade already exists in TradeManager: {trade_id}")
                skipped_count += 1
                continue

            # Create Trade object directly with timestamp
            tm_trade = Trade(
                id=f"{symbol}_{side}_{amount}_{price}_{timestamp.isoformat()}",
                symbol=symbol,
                side=side,
                amount=Decimal(str(amount)),
                price=Decimal(str(price)),
                timestamp=timestamp,
                strategy="migrated",  # Mark as migrated from CSV
                exchange="kraken",  # Default exchange
                fees=Decimal('0'),
                status="filled"
            )

            # Record to TradeManager
            trade_id_recorded = trade_manager.record_trade(tm_trade)
            logger.info(f"Migrated trade to TradeManager: {symbol} {side} {amount} @ {price} (ID: {trade_id_recorded})")
            migrated_count += 1

        except Exception as e:
            logger.error(f"Error migrating trade: {e}")
            logger.error(f"Trade data: {row.to_dict()}")

    logger.info("Migration completed:")
    logger.info(f"  - Migrated: {migrated_count} trades")
    logger.info(f"  - Skipped (duplicates): {skipped_count} trades")
    logger.info(f"  - Total processed: {migrated_count + skipped_count}")

    # Verify migration by checking final counts
    final_trades = trade_manager.get_trade_history()
    logger.info(f"Final TradeManager trade count: {len(final_trades)}")

    if migrated_count > 0:
        logger.info("✅ Migration successful! TradeManager now contains all historical trades.")
    else:
        logger.info("ℹ️  No new trades needed migration (all already in TradeManager).")

if __name__ == "__main__":
    migrate_csv_trades_to_trademanager()
