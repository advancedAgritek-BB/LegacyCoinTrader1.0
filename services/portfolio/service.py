"""Business logic for the portfolio service."""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy.orm import Session

from .config import PortfolioConfig
from .database import get_session
from .models import PositionModel, TradeModel
from .repository import PortfolioRepository
from .schemas import (
    PnlBreakdown,
    PortfolioState,
    PortfolioStatistics,
    PositionRead,
    PriceCacheEntry,
    RiskCheckResult,
    TradeCreate,
)

logger = logging.getLogger(__name__)


class PortfolioService:
    """High level domain service that mirrors the TradeManager API."""

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig.from_env()
        self.repository = PortfolioRepository(self.config)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def get_state(self) -> PortfolioState:
        return self.repository.load_state()

    def replace_state(self, state: PortfolioState) -> PortfolioState:
        logger.info(
            "Persisting portfolio state: %s trades, %s open positions",
            len(state.trades),
            len(state.positions),
        )
        return self.repository.replace_state(state)

    # ------------------------------------------------------------------
    # Trade & position updates
    # ------------------------------------------------------------------
    def record_trade(self, trade: TradeCreate) -> PositionRead:
        """Record a trade and return the updated position."""

        with get_session(self.config) as session:
            position = session.get(PositionModel, trade.symbol)
            stats = self.repository.load_statistics(session=session)

            trade_value = trade.amount * trade.price
            stats.total_trades += 1
            stats.total_volume += trade_value
            stats.total_fees += trade.fees

            if position is None or not position.is_open:
                position = self._open_new_position(session, trade)
            else:
                position = self._update_existing_position(session, position, trade, stats)

            trade_record = self.repository.upsert_trade(
                trade, session=session, position_symbol=position.symbol
            )

            trade_model = session.get(TradeModel, trade_record.id)
            if trade_model is not None:
                trade_model.position_symbol = position.symbol
                if trade_model not in position.trades:
                    position.trades.append(trade_model)

            self.repository.update_statistics(stats, session=session)

            session.flush()
            session.refresh(position)
            return self.repository._model_to_position(position)

    def update_price(self, symbol: str, price: Decimal) -> PositionRead | None:
        """Update cached price and position metrics."""

        with get_session(self.config) as session:
            position = session.get(PositionModel, symbol)
            if position is None:
                self.repository.upsert_price(
                    PriceCacheEntry(symbol=symbol, price=price, updated_at=datetime.utcnow()),
                    session=session,
                )
                return None

            position.mark_price = price
            position.last_update = datetime.utcnow()
            position.highest_price = max(position.highest_price or price, price)
            position.lowest_price = min(position.lowest_price or price, price)

            # Persist price cache entry
            self.repository.upsert_price(
                PriceCacheEntry(symbol=symbol, price=price, updated_at=datetime.utcnow()),
                session=session,
            )

            session.add(position)
            session.flush()
            session.refresh(position)
            return self.repository._model_to_position(position)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def compute_pnl(self, symbol: Optional[str] = None) -> PnlBreakdown:
        """Compute realized/unrealized PnL for the portfolio or a specific symbol."""

        state = self.get_state()
        realized = state.statistics.total_realized_pnl
        unrealized = Decimal("0")

        positions = state.positions
        if symbol:
            positions = [pos for pos in positions if pos.symbol == symbol]

        price_lookup = {entry.symbol: entry.price for entry in state.price_cache}

        for position in positions:
            price = position.mark_price or price_lookup.get(position.symbol)
            if price is None:
                continue
            if position.side == "long":
                pnl = (price - position.average_price) * position.total_amount
            else:
                pnl = (position.average_price - price) * position.total_amount
            unrealized += pnl

        total = realized + unrealized
        return PnlBreakdown(realized=realized, unrealized=unrealized, total=total)

    def check_risk_limits(self) -> List[RiskCheckResult]:
        """Evaluate configured risk limits against current positions."""

        limits = self.repository.list_risk_limits()
        if not limits:
            return []

        positions = self.repository.list_positions(include_closed=False)
        price_cache = {entry.symbol: entry.price for entry in self.repository.list_prices()}
        results: List[RiskCheckResult] = []

        for limit in limits:
            passed = True
            messages: List[str] = []

            if limit.max_position_size is not None:
                for position in positions:
                    if position.total_amount > limit.max_position_size:
                        passed = False
                        messages.append(
                            f"Position {position.symbol} exceeds max size {limit.max_position_size}"
                        )

            if limit.max_daily_loss is not None:
                pnl = Decimal("0")
                for position in positions:
                    price = price_cache.get(position.symbol) or position.mark_price
                    if not price:
                        continue
                    if position.side == "long":
                        pnl += (price - position.average_price) * position.total_amount
                    else:
                        pnl += (position.average_price - price) * position.total_amount
                if pnl < -limit.max_daily_loss:
                    passed = False
                    messages.append(
                        f"Aggregate unrealized loss {pnl} breaches daily loss limit {limit.max_daily_loss}"
                    )

            message = "; ".join(messages) if messages else "Within configured limits"
            results.append(
                RiskCheckResult(name=limit.name, passed=passed, message=message)
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_new_position(self, session: Session, trade: TradeCreate) -> PositionModel:
        position = PositionModel(symbol=trade.symbol)
        position.side = "long" if trade.side == "buy" else "short"
        position.total_amount = trade.amount
        position.average_price = trade.price
        position.entry_time = trade.timestamp
        position.last_update = trade.timestamp
        position.fees_paid = trade.fees
        position.highest_price = trade.price
        position.lowest_price = trade.price
        position.realized_pnl = Decimal("0")
        position.is_open = True
        session.add(position)
        session.flush()
        return position

    def _update_existing_position(
        self,
        session: Session,
        position: PositionModel,
        trade: TradeCreate,
        stats: PortfolioStatistics,
    ) -> PositionModel:
        trade_direction = "long" if trade.side == "buy" else "short"

        if trade.amount == 0:
            return position

        if position.side == trade_direction:
            total_value = position.total_amount * position.average_price + trade.amount * trade.price
            total_amount = position.total_amount + trade.amount
            if total_amount > 0:
                position.average_price = total_value / total_amount
            position.total_amount = total_amount
            position.fees_paid += trade.fees
            position.last_update = trade.timestamp
            position.highest_price = max(position.highest_price or trade.price, trade.price)
            position.lowest_price = min(position.lowest_price or trade.price, trade.price)
        else:
            realized_pnl, remaining = self._calculate_position_closure(position, trade)
            position.realized_pnl += realized_pnl
            stats.total_realized_pnl += realized_pnl
            trade_remaining = trade.amount - position.total_amount

            if trade_remaining > 0:
                # position reversal
                position.is_open = False
                position.last_update = trade.timestamp
                session.flush()

                position = PositionModel(symbol=trade.symbol)
                position.side = "short" if trade_direction == "short" else "long"
                position.total_amount = trade_remaining
                position.average_price = trade.price
                position.entry_time = trade.timestamp
                position.last_update = trade.timestamp
                position.fees_paid = trade.fees
                position.realized_pnl = Decimal("0")
                position.highest_price = trade.price
                position.lowest_price = trade.price
                position.is_open = True
                session.add(position)
            else:
                position.total_amount = remaining
                position.fees_paid += trade.fees
                position.last_update = trade.timestamp
                if position.total_amount <= Decimal("0.00000001"):
                    position.total_amount = Decimal("0")
                    position.is_open = False

        session.add(position)
        return position

    def _calculate_position_closure(
        self, position: PositionModel, closing_trade: TradeCreate
    ) -> tuple[Decimal, Decimal]:
        if position.side == "long":
            pnl_per_unit = closing_trade.price - position.average_price
        else:
            pnl_per_unit = position.average_price - closing_trade.price
        close_amount = min(position.total_amount, closing_trade.amount)
        realized_pnl = pnl_per_unit * close_amount
        remaining = position.total_amount - close_amount
        return realized_pnl, remaining
