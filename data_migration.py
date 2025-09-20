#!/usr/bin/env python3
"""
Data Migration Script for LegacyCoinTrader Microservice Architecture

This script migrates data from legacy CSV files and JSON state files
to the new PostgreSQL-based microservice architecture.

Supported data sources:
- CSV trade files (trades.csv)
- TradeManager state files (trade_manager_state.json)
- Paper wallet state files
- Performance logs
- Configuration files

Migration targets:
- Portfolio service (trades, positions, balances)
- Monitoring service (metrics, logs)
- Configuration service (settings)

Usage:
    python data_migration.py --source-type csv --source-file /path/to/trades.csv
    python data_migration.py --source-type json --source-file /path/to/state.json
    python data_migration.py --scan-directory /path/to/logs --dry-run
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import argparse

import httpx
from pydantic import BaseModel, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """Configuration for data migration."""

    portfolio_service_url: str = "http://localhost:8003"
    monitoring_service_url: str = "http://localhost:8007"
    api_gateway_url: str = "http://localhost:8000"
    dry_run: bool = False
    batch_size: int = 100
    timeout: float = 30.0


class TradeCreate(BaseModel):
    """Schema for creating trades in portfolio service."""
    id: str
    symbol: str
    side: str
    amount: Decimal
    price: Decimal
    timestamp: datetime
    strategy: Optional[str] = None
    exchange: Optional[str] = None
    fees: Decimal = Decimal("0")
    status: str = "filled"
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class PositionCreate(BaseModel):
    """Schema for creating positions in portfolio service."""
    symbol: str
    side: str
    total_amount: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    entry_time: datetime
    last_update: datetime
    highest_price: Optional[Decimal] = None
    lowest_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    trailing_stop_pct: Optional[Decimal] = None
    metadata: Dict[str, Any] = {}


class PortfolioState(BaseModel):
    """Complete portfolio state for migration."""
    trades: List[TradeCreate]
    positions: List[PositionCreate]
    closed_positions: List[PositionCreate]


class DataMigrationService:
    """Service for migrating legacy data to microservices."""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.http_client = httpx.AsyncClient(timeout=config.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()

    async def migrate_csv_trades(self, csv_path: Path) -> Dict[str, Any]:
        """Migrate trades from CSV file to portfolio service."""

        logger.info(f"Migrating trades from CSV: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        trades = []
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                try:
                    trade = self._parse_csv_trade(row, row_num)
                    if trade:
                        trades.append(trade)
                except Exception as e:
                    logger.warning(f"Failed to parse trade at row {row_num}: {e}")
                    continue

        if not trades:
            logger.warning("No valid trades found in CSV")
            return {"status": "no_data", "trades_processed": 0}

        logger.info(f"Parsed {len(trades)} trades from CSV")

        # Migrate in batches
        return await self._migrate_trades_batch(trades)

    async def migrate_json_state(self, json_path: Path) -> Dict[str, Any]:
        """Migrate TradeManager state from JSON file."""

        logger.info(f"Migrating state from JSON: {json_path}")

        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            state_data = json.load(f)

        # Parse trades
        trades = []
        raw_trades = state_data.get('trades', [])
        for trade_data in raw_trades:
            try:
                trade = self._parse_json_trade(trade_data)
                if trade:
                    trades.append(trade)
            except Exception as e:
                logger.warning(f"Failed to parse trade: {e}")
                continue

        # Parse positions
        positions = []
        raw_positions = state_data.get('positions', {})
        for symbol, pos_data in raw_positions.items():
            try:
                position = self._parse_json_position(symbol, pos_data)
                if position:
                    positions.append(position)
            except Exception as e:
                logger.warning(f"Failed to parse position {symbol}: {e}")
                continue

        logger.info(f"Parsed {len(trades)} trades and {len(positions)} positions from JSON")

        # Create portfolio state
        portfolio_state = PortfolioState(
            trades=trades,
            positions=positions,
            closed_positions=[]
        )

        return await self._migrate_portfolio_state(portfolio_state)

    async def scan_and_migrate_directory(self, directory: Path) -> Dict[str, Any]:
        """Scan directory for data files and migrate them."""

        logger.info(f"Scanning directory for data files: {directory}")

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        results = {
            "csv_files_processed": 0,
            "json_files_processed": 0,
            "total_trades_migrated": 0,
            "total_positions_migrated": 0,
            "errors": []
        }

        # Process CSV files
        for csv_file in directory.glob("*.csv"):
            if "trades" in csv_file.name.lower():
                try:
                    csv_result = await self.migrate_csv_trades(csv_file)
                    results["csv_files_processed"] += 1
                    results["total_trades_migrated"] += csv_result.get("trades_processed", 0)
                except Exception as e:
                    error_msg = f"Failed to migrate CSV {csv_file}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)

        # Process JSON files
        for json_file in directory.glob("*.json"):
            if "trade_manager" in json_file.name.lower() or "state" in json_file.name.lower():
                try:
                    json_result = await self.migrate_json_state(json_file)
                    results["json_files_processed"] += 1
                    results["total_trades_migrated"] += json_result.get("trades_processed", 0)
                    results["total_positions_migrated"] += json_result.get("positions_processed", 0)
                except Exception as e:
                    error_msg = f"Failed to migrate JSON {json_file}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)

        return results

    def _parse_csv_trade(self, row: Dict[str, str], row_num: int) -> Optional[TradeCreate]:
        """Parse a trade from CSV row."""

        try:
            # Handle different timestamp formats
            timestamp_str = row.get('timestamp', '').strip()
            if not timestamp_str:
                logger.warning(f"Row {row_num}: Missing timestamp")
                return None

            # Parse timestamp
            if 'T' in timestamp_str:
                # ISO format
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Assume YYYY-MM-DD HH:MM:SS format
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Parse required fields
            symbol = row.get('symbol', '').strip()
            side = row.get('side', '').strip().lower()
            amount_str = row.get('amount', '').strip()
            price_str = row.get('price', '').strip()

            if not all([symbol, side, amount_str, price_str]):
                logger.warning(f"Row {row_num}: Missing required fields")
                return None

            if side not in ['buy', 'sell']:
                logger.warning(f"Row {row_num}: Invalid side '{side}'")
                return None

            # Parse amounts
            try:
                amount = Decimal(amount_str)
                price = Decimal(price_str)
                fees = Decimal(row.get('fees', '0'))
            except Exception as e:
                logger.warning(f"Row {row_num}: Invalid numeric values: {e}")
                return None

            if amount <= 0 or price <= 0:
                logger.warning(f"Row {row_num}: Invalid amounts or prices")
                return None

            # Generate ID if not present
            trade_id = row.get('id', f"csv_{row_num}")

            # Create trade object
            trade = TradeCreate(
                id=trade_id,
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                timestamp=timestamp,
                strategy=row.get('strategy', 'imported'),
                exchange=row.get('exchange', 'kraken'),
                fees=fees,
                order_id=row.get('order_id'),
                client_order_id=row.get('client_order_id'),
                metadata={
                    'source': 'csv_import',
                    'row_number': row_num,
                    'imported_at': datetime.now(timezone.utc).isoformat()
                }
            )

            return trade

        except Exception as e:
            logger.warning(f"Row {row_num}: Parse error: {e}")
            return None

    def _parse_json_trade(self, trade_data: Dict[str, Any]) -> Optional[TradeCreate]:
        """Parse a trade from JSON data."""

        try:
            # Handle timestamp conversion
            timestamp = trade_data.get('timestamp')
            if isinstance(timestamp, str):
                if 'T' in timestamp:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            # Parse amounts
            amount = Decimal(str(trade_data.get('amount', 0)))
            price = Decimal(str(trade_data.get('price', 0)))
            fees = Decimal(str(trade_data.get('fees', 0)))

            trade = TradeCreate(
                id=trade_data.get('id', str(trade_data.get('timestamp', 'unknown'))),
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                amount=amount,
                price=price,
                timestamp=timestamp,
                strategy=trade_data.get('strategy', 'imported'),
                exchange=trade_data.get('exchange', 'kraken'),
                fees=fees,
                order_id=trade_data.get('order_id'),
                client_order_id=trade_data.get('client_order_id'),
                metadata={
                    'source': 'json_import',
                    'imported_at': datetime.now(timezone.utc).isoformat()
                }
            )

            return trade

        except Exception as e:
            logger.warning(f"Failed to parse JSON trade: {e}")
            return None

    def _parse_json_position(self, symbol: str, pos_data: Dict[str, Any]) -> Optional[PositionCreate]:
        """Parse a position from JSON data."""

        try:
            # Handle entry time
            entry_time = pos_data.get('entry_time')
            if isinstance(entry_time, str):
                if 'T' in entry_time:
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                else:
                    entry_time = datetime.fromisoformat(entry_time)
            elif isinstance(entry_time, (int, float)):
                entry_time = datetime.fromtimestamp(entry_time, tz=timezone.utc)
            else:
                entry_time = datetime.now(timezone.utc)

            # Handle last update
            last_update = pos_data.get('last_update', entry_time)
            if isinstance(last_update, str):
                if 'T' in last_update:
                    last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                else:
                    last_update = datetime.fromisoformat(last_update)
            elif isinstance(last_update, (int, float)):
                last_update = datetime.fromtimestamp(last_update, tz=timezone.utc)

            position = PositionCreate(
                symbol=symbol,
                side=pos_data.get('side', 'long'),
                total_amount=Decimal(str(pos_data.get('total_amount', 0))),
                average_price=Decimal(str(pos_data.get('average_price', 0))),
                realized_pnl=Decimal(str(pos_data.get('realized_pnl', 0))),
                fees_paid=Decimal(str(pos_data.get('fees_paid', 0))),
                entry_time=entry_time,
                last_update=last_update,
                highest_price=Decimal(str(pos_data['highest_price'])) if pos_data.get('highest_price') else None,
                lowest_price=Decimal(str(pos_data['lowest_price'])) if pos_data.get('lowest_price') else None,
                stop_loss_price=Decimal(str(pos_data['stop_loss_price'])) if pos_data.get('stop_loss_price') else None,
                take_profit_price=Decimal(str(pos_data['take_profit_price'])) if pos_data.get('take_profit_price') else None,
                trailing_stop_pct=Decimal(str(pos_data['trailing_stop_pct'])) if pos_data.get('trailing_stop_pct') else None,
                metadata={
                    'source': 'json_import',
                    'imported_at': datetime.now(timezone.utc).isoformat()
                }
            )

            return position

        except Exception as e:
            logger.warning(f"Failed to parse JSON position {symbol}: {e}")
            return None

    async def _migrate_trades_batch(self, trades: List[TradeCreate]) -> Dict[str, Any]:
        """Migrate trades to portfolio service in batches."""

        if self.config.dry_run:
            logger.info(f"DRY RUN: Would migrate {len(trades)} trades")
            return {
                "status": "dry_run",
                "trades_processed": len(trades),
                "trades_migrated": 0
            }

        successful = 0
        failed = 0

        for i in range(0, len(trades), self.config.batch_size):
            batch = trades[i:i + self.config.batch_size]
            logger.info(f"Migrating batch {i//self.config.batch_size + 1} with {len(batch)} trades")

            try:
                # Use portfolio service API to create trades
                url = f"{self.config.portfolio_service_url}/trades/batch"
                payload = {"trades": [trade.model_dump() for trade in batch]}

                response = await self.http_client.post(url, json=payload)
                response.raise_for_status()

                successful += len(batch)
                logger.info(f"Successfully migrated batch of {len(batch)} trades")

            except Exception as e:
                failed += len(batch)
                logger.error(f"Failed to migrate batch: {e}")
                continue

        return {
            "status": "completed",
            "trades_processed": len(trades),
            "trades_migrated": successful,
            "trades_failed": failed
        }

    async def _migrate_portfolio_state(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Migrate complete portfolio state."""

        if self.config.dry_run:
            logger.info(f"DRY RUN: Would migrate portfolio with {len(portfolio_state.trades)} trades and {len(portfolio_state.positions)} positions")
            return {
                "status": "dry_run",
                "trades_processed": len(portfolio_state.trades),
                "positions_processed": len(portfolio_state.positions),
                "trades_migrated": 0,
                "positions_migrated": 0
            }

        # Migrate trades first
        trade_result = await self._migrate_trades_batch(portfolio_state.trades)

        # Migrate positions
        positions_migrated = 0
        for position in portfolio_state.positions:
            try:
                url = f"{self.config.portfolio_service_url}/positions"
                payload = position.model_dump()

                response = await self.http_client.post(url, json=payload)
                response.raise_for_status()

                positions_migrated += 1
                logger.info(f"Migrated position: {position.symbol}")

            except Exception as e:
                logger.error(f"Failed to migrate position {position.symbol}: {e}")
                continue

        return {
            "status": "completed",
            "trades_processed": len(portfolio_state.trades),
            "positions_processed": len(portfolio_state.positions),
            "trades_migrated": trade_result.get("trades_migrated", 0),
            "positions_migrated": positions_migrated
        }


async def main():
    """Main migration function."""

    parser = argparse.ArgumentParser(description="Migrate legacy data to microservices")
    parser.add_argument(
        "--source-type",
        choices=["csv", "json", "directory"],
        help="Type of data source to migrate"
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        help="Path to source file (CSV or JSON)"
    )
    parser.add_argument(
        "--scan-directory",
        type=Path,
        help="Directory to scan for data files"
    )
    parser.add_argument(
        "--portfolio-service-url",
        default="http://localhost:8003",
        help="Portfolio service URL"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for migration operations"
    )

    args = parser.parse_args()

    config = MigrationConfig(
        portfolio_service_url=args.portfolio_service_url,
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )

    async with DataMigrationService(config) as migrator:
        try:
            if args.source_type == "csv":
                if not args.source_file:
                    parser.error("--source-file required for CSV migration")
                result = await migrator.migrate_csv_trades(args.source_file)

            elif args.source_type == "json":
                if not args.source_file:
                    parser.error("--source-file required for JSON migration")
                result = await migrator.migrate_json_state(args.source_file)

            elif args.source_type == "directory":
                if not args.scan_directory:
                    parser.error("--scan-directory required for directory scan")
                result = await migrator.scan_and_migrate_directory(args.scan_directory)

            else:
                parser.error("Must specify --source-type")

            logger.info("Migration completed successfully")
            logger.info(f"Result: {result}")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
