"""Migration helper to push legacy JSON state into the portfolio service."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

from ..clients.interface import PortfolioServiceClient
from ..schemas import (
    PortfolioState,
    PortfolioStatistics,
    PositionRead,
    PriceCacheEntry,
    TradeRead,
)

LEGACY_STATE_PATH = Path("crypto_bot/logs/trade_manager_state.json")


def _load_legacy_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Legacy state file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_trade(trade: Dict[str, Any]) -> TradeRead:
    return TradeRead(
        id=trade["id"],
        symbol=trade["symbol"],
        side=trade["side"],
        amount=Decimal(str(trade["amount"])),
        price=Decimal(str(trade["price"])),
        timestamp=datetime.fromisoformat(trade["timestamp"]),
        strategy=trade.get("strategy"),
        exchange=trade.get("exchange"),
        fees=Decimal(str(trade.get("fees", 0))),
        status=trade.get("status", "filled"),
        order_id=trade.get("order_id"),
        client_order_id=trade.get("client_order_id"),
        metadata=trade.get("metadata", {}),
        position_symbol=trade.get("position_symbol"),
    )


def _parse_position(symbol: str, position: Dict[str, Any]) -> PositionRead:
    trades = [_parse_trade(t) for t in position.get("trades", [])]
    return PositionRead(
        symbol=symbol,
        side=position.get("side", "long"),
        total_amount=Decimal(str(position.get("total_amount", 0))),
        average_price=Decimal(str(position.get("average_price", 0))),
        realized_pnl=Decimal(str(position.get("realized_pnl", 0))),
        fees_paid=Decimal(str(position.get("fees_paid", 0))),
        entry_time=datetime.fromisoformat(position.get("entry_time")),
        last_update=datetime.fromisoformat(position.get("last_update")),
        highest_price=Decimal(str(position.get("highest_price"))) if position.get("highest_price") is not None else None,
        lowest_price=Decimal(str(position.get("lowest_price"))) if position.get("lowest_price") is not None else None,
        stop_loss_price=Decimal(str(position.get("stop_loss_price"))) if position.get("stop_loss_price") is not None else None,
        take_profit_price=Decimal(str(position.get("take_profit_price"))) if position.get("take_profit_price") is not None else None,
        trailing_stop_pct=Decimal(str(position.get("trailing_stop_pct"))) if position.get("trailing_stop_pct") is not None else None,
        metadata=position.get("metadata", {}),
        mark_price=Decimal(str(position.get("mark_price"))) if position.get("mark_price") is not None else None,
        is_open=position.get("is_open", True),
        trades=trades,
    )


def migrate_state(path: Path = LEGACY_STATE_PATH) -> PortfolioState:
    legacy = _load_legacy_state(path)
    trades = [_parse_trade(t) for t in legacy.get("trades", [])]
    positions = [
        _parse_position(symbol, data)
        for symbol, data in (legacy.get("positions", {}) or {}).items()
    ]
    closed_positions = [
        _parse_position(f"closed::{idx}", data)
        for idx, data in enumerate(legacy.get("closed_positions", []) or [])
    ]
    price_cache = [
        PriceCacheEntry(
            symbol=symbol,
            price=Decimal(str(price)),
            updated_at=datetime.utcnow(),
        )
        for symbol, price in (legacy.get("price_cache", {}) or {}).items()
    ]
    stats = legacy.get("statistics", {}) or {}
    statistics = PortfolioStatistics(
        total_trades=int(stats.get("total_trades", 0)),
        total_volume=Decimal(str(stats.get("total_volume", 0))),
        total_fees=Decimal(str(stats.get("total_fees", 0))),
        total_realized_pnl=Decimal(str(stats.get("total_realized_pnl", 0))),
        last_updated=datetime.utcnow(),
    )
    state = PortfolioState(
        trades=trades,
        positions=positions,
        closed_positions=closed_positions,
        price_cache=price_cache,
        statistics=statistics,
    )
    client = PortfolioServiceClient()
    client.put_state(state)
    return state


if __name__ == "__main__":
    migrated = migrate_state()
    print(
        f"Migrated {len(migrated.trades)} trades and {len(migrated.positions)} open positions to the portfolio service"
    )
