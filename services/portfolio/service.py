"""Business logic for the portfolio service."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from sqlalchemy.orm import Session

from libs.bootstrap import load_config as load_bot_config

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
        try:
            self._bot_config = load_bot_config()
        except Exception as exc:  # pragma: no cover - config load failure should not break trades
            logger.warning("Failed to load trading configuration for risk settings: %s", exc)
            self._bot_config = {}

    # ------------------------------------------------------------------
    # Internal helpers - risk configuration
    # ------------------------------------------------------------------
    def _exit_settings(self) -> tuple[Decimal, Decimal, Optional[Decimal]]:
        """Return (stop_loss_pct, take_profit_pct, trailing_stop_pct)."""

        risk_cfg = self._bot_config.get("risk", {}) if isinstance(self._bot_config, dict) else {}
        exit_cfg = self._bot_config.get("exit_strategy", {}) if isinstance(self._bot_config, dict) else {}

        stop_loss_pct = exit_cfg.get("stop_loss_pct", risk_cfg.get("stop_loss_pct", 0.01))
        take_profit_pct = exit_cfg.get("take_profit_pct", risk_cfg.get("take_profit_pct", 0.04))
        trailing_stop_pct = exit_cfg.get("trailing_stop_pct", risk_cfg.get("trailing_stop_pct", 0.0))

        try:
            stop_loss_dec = Decimal(str(stop_loss_pct))
        except Exception:  # pragma: no cover - defensive default
            stop_loss_dec = Decimal("0.01")
        try:
            take_profit_dec = Decimal(str(take_profit_pct))
        except Exception:  # pragma: no cover - defensive default
            take_profit_dec = Decimal("0.04")

        trailing_dec: Optional[Decimal]
        try:
            trailing_value = Decimal(str(trailing_stop_pct))
            trailing_dec = trailing_value if trailing_value > 0 else None
        except Exception:  # pragma: no cover - optional config
            trailing_dec = None

        return stop_loss_dec, take_profit_dec, trailing_dec

    def _set_protective_levels(self, position: PositionModel) -> None:
        """Apply stop loss, take profit, and trailing configuration to *position*."""

        stop_loss_pct, take_profit_pct, trailing_pct = self._exit_settings()
        average_price = position.average_price or Decimal("0")
        if average_price <= 0:
            return

        if position.side == "long":
            base_stop = average_price * (Decimal("1") - stop_loss_pct)
            base_tp = average_price * (Decimal("1") + take_profit_pct)
            if position.stop_loss_price is None:
                position.stop_loss_price = base_stop
            else:
                position.stop_loss_price = max(position.stop_loss_price, base_stop)
            position.take_profit_price = base_tp
        else:
            base_stop = average_price * (Decimal("1") + stop_loss_pct)
            base_tp = average_price * (Decimal("1") - take_profit_pct)
            if position.stop_loss_price is None:
                position.stop_loss_price = base_stop
            else:
                position.stop_loss_price = min(position.stop_loss_price, base_stop)
            position.take_profit_price = base_tp

        position.trailing_stop_pct = trailing_pct
        self._update_trailing_stop(position)

    def _update_trailing_stop(self, position: PositionModel) -> None:
        """Adjust the trailing stop based on recorded highs/lows."""

        trailing_pct = position.trailing_stop_pct
        if trailing_pct is None or trailing_pct <= 0:
            return

        if position.side == "long" and position.highest_price:
            new_stop = position.highest_price * (Decimal("1") - trailing_pct)
            if position.stop_loss_price is None or new_stop > position.stop_loss_price:
                position.stop_loss_price = new_stop
        elif position.side == "short" and position.lowest_price:
            new_stop = position.lowest_price * (Decimal("1") + trailing_pct)
            if position.stop_loss_price is None or new_stop < position.stop_loss_price:
                position.stop_loss_price = new_stop

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
            stats.last_updated = trade.timestamp or datetime.utcnow()

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
            if position.is_open:
                self._update_trailing_stop(position)

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

    def get_statistics_summary(self) -> dict:
        """Return enriched statistics used by UI dashboards."""

        now = datetime.utcnow()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        last_24h = now - timedelta(hours=24)

        with get_session(self.config) as session:
            stats_model = self.repository.load_statistics(session=session)

            open_positions = (
                session.query(PositionModel)
                .filter(PositionModel.is_open.is_(True))
                .all()
            )
            closed_positions = (
                session.query(PositionModel)
                .filter(PositionModel.is_open.is_(False))
                .all()
            )

            winning_positions = sum(
                1 for position in closed_positions if position.realized_pnl > 0
            )
            losing_positions = sum(
                1 for position in closed_positions if position.realized_pnl < 0
            )
            competitive_positions = winning_positions + losing_positions
            win_rate = (
                winning_positions / competitive_positions
                if competitive_positions
                else 0.0
            )

            trades_today = (
                session.query(TradeModel)
                .filter(TradeModel.timestamp >= start_of_day)
                .count()
            )
            trades_last_24h = (
                session.query(TradeModel)
                .filter(TradeModel.timestamp >= last_24h)
                .count()
            )

            latest_trade = (
                session.query(TradeModel)
                .order_by(TradeModel.timestamp.desc())
                .first()
            )

        pnl_breakdown = self.compute_pnl()

        return {
            "total_trades": stats_model.total_trades,
            "total_volume": float(stats_model.total_volume or Decimal("0")),
            "total_fees": float(stats_model.total_fees or Decimal("0")),
            "total_realized_pnl": float(stats_model.total_realized_pnl or Decimal("0")),
            "last_updated": stats_model.last_updated.isoformat()
            if stats_model.last_updated
            else None,
            "open_positions": len(open_positions),
            "closed_positions": len(closed_positions),
            "winning_positions": winning_positions,
            "losing_positions": losing_positions,
            "win_rate": win_rate,
            "trades_today": trades_today,
            "trades_last_24h": trades_last_24h,
            "last_trade_at": latest_trade.timestamp.isoformat()
            if latest_trade
            else None,
            "unrealized_pnl": float(pnl_breakdown.unrealized),
            "realized_pnl": float(pnl_breakdown.realized),
            "total_pnl": float(pnl_breakdown.total),
        }

    def close_stale_positions(
        self, max_age_hours: int = 24, symbols: Optional[List[str]] = None
    ) -> dict:
        """Force close positions either older than ``max_age_hours`` or by symbol."""

        effective_age = max(0, max_age_hours or 0)
        cutoff = datetime.utcnow() - timedelta(hours=effective_age)

        symbol_filter: Optional[List[str]] = None
        if symbols:
            symbol_filter = [str(symbol).strip().upper() for symbol in symbols if symbol]
            symbol_filter = [symbol for symbol in symbol_filter if symbol]
            if not symbol_filter:
                return {
                    "closed": 0,
                    "symbols": [],
                    "cutoff": cutoff.isoformat(),
                    "mode": "symbols",
                }

        with get_session(self.config) as session:
            query = session.query(PositionModel).filter(PositionModel.is_open.is_(True))
            if symbol_filter is not None:
                query = query.filter(PositionModel.symbol.in_(symbol_filter))
            else:
                query = query.filter(PositionModel.last_update < cutoff)

            candidates = query.all()
            closed_symbols: List[str] = []
            timestamp = datetime.utcnow()

            for position in candidates:
                closed_symbols.append(position.symbol)
                position.is_open = False
                position.total_amount = Decimal("0")
                position.last_update = timestamp

            session.flush()

        return {
            "closed": len(closed_symbols),
            "symbols": closed_symbols,
            "cutoff": cutoff.isoformat(),
            "mode": "symbols" if symbol_filter is not None else "age",
        }

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
        self._set_protective_levels(position)
        logger.info(
            "Initialized protective stops for %s: stop=%.8f take_profit=%.8f trailing=%s",
            position.symbol,
            float(position.stop_loss_price) if position.stop_loss_price else 0.0,
            float(position.take_profit_price) if position.take_profit_price else 0.0,
            (float(position.trailing_stop_pct) if position.trailing_stop_pct is not None else None),
        )
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
            else:
                position.total_amount = remaining
                position.fees_paid += trade.fees
                position.last_update = trade.timestamp
                if position.total_amount <= Decimal("0.00000001"):
                    position.total_amount = Decimal("0")
                position.is_open = False

        if position.is_open:
            self._set_protective_levels(position)

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
