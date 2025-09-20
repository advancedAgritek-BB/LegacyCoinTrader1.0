"""Database repository for the portfolio service."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import PortfolioConfig
from .database import Base, get_engine, get_session
from .models import (
    BalanceModel,
    PortfolioStatisticModel,
    PositionModel,
    PriceCacheModel,
    RiskLimitModel,
    TradeModel,
)
from .schemas import (
    BalanceRead,
    PortfolioState,
    PortfolioStatistics,
    PositionRead,
    PriceCacheEntry,
    RiskLimitRead,
    TradeCreate,
    TradeRead,
)


class PortfolioRepository:
    """Encapsulates database access patterns for the portfolio service."""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        engine = get_engine(self.config)
        Base.metadata.create_all(engine)

    # ------------------------------------------------------------------
    # Trade helpers
    # ------------------------------------------------------------------
    def list_trades(self, session: Session | None = None) -> List[TradeRead]:
        def _list(sess: Session) -> List[TradeRead]:
            stmt = select(TradeModel).order_by(TradeModel.timestamp)
            trades = sess.scalars(stmt).all()
            return [self._model_to_trade(trade) for trade in trades]

        if session is not None:
            return _list(session)

        with get_session(self.config) as sess:
            return _list(sess)

    def upsert_trade(
        self,
        trade: TradeCreate,
        session: Session | None = None,
        position_symbol: str | None = None,
    ) -> TradeRead:
        def _upsert(sess: Session) -> TradeRead:
            model = sess.get(TradeModel, trade.id)
            if model is None:
                model = TradeModel(id=trade.id)
            self._apply_trade(model, trade)
            if position_symbol is not None:
                model.position_symbol = position_symbol
            sess.add(model)
            sess.flush()
            return self._model_to_trade(model)

        if session is not None:
            return _upsert(session)

        with get_session(self.config) as sess:
            return _upsert(sess)

    def delete_trades_not_in(self, trade_ids: Iterable[str], session: Session) -> None:
        ids = set(trade_ids)
        existing = {
            row[0]
            for row in session.execute(select(TradeModel.id)).all()
        }
        to_delete = existing - ids
        if to_delete:
            session.query(TradeModel).filter(TradeModel.id.in_(to_delete)).delete(
                synchronize_session=False
            )

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    def list_positions(self, include_closed: bool = False, session: Session | None = None) -> List[PositionRead]:
        def _list(sess: Session) -> List[PositionRead]:
            stmt = select(PositionModel)
            if not include_closed:
                stmt = stmt.where(PositionModel.is_open.is_(True))
            stmt = stmt.order_by(PositionModel.symbol)
            models = sess.scalars(stmt).all()
            return [self._model_to_position(model) for model in models]

        if session is not None:
            return _list(session)

        with get_session(self.config) as sess:
            return _list(sess)

    def upsert_position(self, position: PositionRead, session: Session) -> None:
        model = session.get(PositionModel, position.symbol)
        if model is None:
            model = PositionModel(symbol=position.symbol)
        self._apply_position(model, position)
        session.add(model)

    def delete_positions_not_in(self, symbols: Iterable[str], session: Session) -> None:
        existing = {
            row[0]
            for row in session.execute(select(PositionModel.symbol)).all()
        }
        to_delete = existing - set(symbols)
        if to_delete:
            session.query(PositionModel).filter(
                PositionModel.symbol.in_(to_delete)
            ).delete(synchronize_session=False)

    # ------------------------------------------------------------------
    # Price cache helpers
    # ------------------------------------------------------------------
    def upsert_price(self, entry: PriceCacheEntry, session: Session) -> None:
        model = (
            session.query(PriceCacheModel)
            .filter(PriceCacheModel.symbol == entry.symbol)
            .one_or_none()
        )
        if model is None:
            model = PriceCacheModel(symbol=entry.symbol)
        model.price = entry.price
        model.updated_at = entry.updated_at
        session.add(model)

    def list_prices(self, session: Session | None = None) -> List[PriceCacheEntry]:
        def _list(sess: Session) -> List[PriceCacheEntry]:
            stmt = select(PriceCacheModel).order_by(PriceCacheModel.symbol)
            models = sess.scalars(stmt).all()
            return [
                PriceCacheEntry(symbol=m.symbol, price=m.price, updated_at=m.updated_at)
                for m in models
            ]

        if session is not None:
            return _list(session)

        with get_session(self.config) as sess:
            return _list(sess)

    # ------------------------------------------------------------------
    # Balances & risk limits
    # ------------------------------------------------------------------
    def list_balances(self) -> List[BalanceRead]:
        with get_session(self.config) as sess:
            stmt = select(BalanceModel).order_by(BalanceModel.currency)
            return [
                BalanceRead(currency=row.currency, amount=row.amount, updated_at=row.updated_at)
                for row in sess.scalars(stmt).all()
            ]

    def list_risk_limits(self) -> List[RiskLimitRead]:
        with get_session(self.config) as sess:
            stmt = select(RiskLimitModel).order_by(RiskLimitModel.name)
            return [
                RiskLimitRead(
                    id=row.id,
                    name=row.name,
                    max_position_size=row.max_position_size,
                    max_drawdown=row.max_drawdown,
                    max_daily_loss=row.max_daily_loss,
                    value_at_risk=row.value_at_risk,
                    metadata=row.metadata or {},
                )
                for row in sess.scalars(stmt).all()
            ]

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------
    def load_statistics(self, session: Session | None = None) -> PortfolioStatistics:
        def _load(sess: Session) -> PortfolioStatistics:
            model = sess.query(PortfolioStatisticModel).first()
            if model is None:
                return PortfolioStatistics()
            return PortfolioStatistics(
                total_trades=model.total_trades,
                total_volume=model.total_volume,
                total_fees=model.total_fees,
                total_realized_pnl=model.total_realized_pnl,
                last_updated=model.last_updated,
            )

        if session is not None:
            return _load(session)

        with get_session(self.config) as sess:
            return _load(sess)

    def update_statistics(self, stats: PortfolioStatistics, session: Session) -> None:
        model = session.query(PortfolioStatisticModel).first()
        if model is None:
            model = PortfolioStatisticModel()
        model.total_trades = stats.total_trades
        model.total_volume = stats.total_volume
        model.total_fees = stats.total_fees
        model.total_realized_pnl = stats.total_realized_pnl
        model.last_updated = stats.last_updated or datetime.utcnow()
        session.add(model)

    # ------------------------------------------------------------------
    # Whole state helpers
    # ------------------------------------------------------------------
    def load_state(self) -> PortfolioState:
        with get_session(self.config) as sess:
            trades = self.list_trades(session=sess)
            positions = self.list_positions(include_closed=False, session=sess)
            closed_stmt = select(PositionModel).where(PositionModel.is_open.is_(False))
            closed_positions = [
                self._model_to_position(model)
                for model in sess.scalars(closed_stmt).all()
            ]
            price_cache = self.list_prices(session=sess)
            statistics = self.load_statistics(session=sess)
            return PortfolioState(
                trades=trades,
                positions=positions,
                closed_positions=closed_positions,
                price_cache=price_cache,
                statistics=statistics,
            )

    def replace_state(self, state: PortfolioState) -> PortfolioState:
        with get_session(self.config) as sess:
            # Trades
            for trade in state.trades:
                trade_payload = trade.model_dump(exclude={"position_symbol"})
                self.upsert_trade(
                    TradeCreate(**trade_payload),
                    session=sess,
                    position_symbol=trade.position_symbol,
                )
            self.delete_trades_not_in((trade.id for trade in state.trades), session=sess)

            # Positions (open + closed share same table)
            all_symbols: list[str] = []
            for position in state.positions:
                self.upsert_position(position, session=sess)
                all_symbols.append(position.symbol)
            for closed in state.closed_positions:
                closed_payload = closed.model_dump()
                closed_payload["is_open"] = False
                closed_copy = PositionRead(**closed_payload)
                self.upsert_position(closed_copy, session=sess)
                all_symbols.append(closed.symbol)
            self.delete_positions_not_in(all_symbols, session=sess)

            # Price cache
            seen_symbols = set()
            for entry in state.price_cache:
                self.upsert_price(entry, session=sess)
                seen_symbols.add(entry.symbol)
            if seen_symbols:
                sess.query(PriceCacheModel).filter(
                    ~PriceCacheModel.symbol.in_(seen_symbols)
                ).delete(synchronize_session=False)
            else:
                sess.query(PriceCacheModel).delete()

            # Statistics
            self.update_statistics(state.statistics, session=sess)

            sess.flush()

            return state

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _apply_trade(self, model: TradeModel, trade: TradeCreate) -> None:
        payload = trade.model_dump()
        metadata = payload.pop("metadata", None)

        model.symbol = payload["symbol"]
        model.side = payload["side"]
        model.amount = payload["amount"]
        model.price = payload["price"]
        model.timestamp = payload["timestamp"]
        model.strategy = payload.get("strategy")
        model.exchange = payload.get("exchange")
        model.fees = payload["fees"]
        model.status = payload["status"]
        model.order_id = payload.get("order_id")
        model.client_order_id = payload.get("client_order_id")
        model.extra_metadata = dict(metadata or {})

    def _model_to_trade(self, model: TradeModel) -> TradeRead:
        return TradeRead(
            id=model.id,
            symbol=model.symbol,
            side=model.side,
            amount=model.amount,
            price=model.price,
            timestamp=model.timestamp,
            strategy=model.strategy,
            exchange=model.exchange,
            fees=model.fees,
            status=model.status,
            order_id=model.order_id,
            client_order_id=model.client_order_id,
            metadata=dict(model.extra_metadata or {}),
            position_symbol=model.position_symbol,
        )

    def _apply_position(self, model: PositionModel, position: PositionRead) -> None:
        dump = position.model_dump(exclude={"trades"})
        metadata = dump.pop("metadata", None)
        for field, value in dump.items():
            setattr(model, field, value)
        model.last_update = position.last_update
        model.entry_time = position.entry_time
        model.extra_metadata = dict(metadata or {})

        # Synchronise trades relationship
        model.trades.clear()
        for trade in position.trades:
            trade_model = TradeModel(id=trade.id)
            payload = trade.model_dump(exclude={"position_symbol"})
            self._apply_trade(trade_model, TradeCreate(**payload))
            trade_model.position_symbol = position.symbol
            model.trades.append(trade_model)

    def _model_to_position(self, model: PositionModel) -> PositionRead:
        trades = [self._model_to_trade(trade) for trade in model.trades]
        return PositionRead(
            symbol=model.symbol,
            side=model.side,
            total_amount=model.total_amount,
            average_price=model.average_price,
            realized_pnl=model.realized_pnl,
            fees_paid=model.fees_paid,
            entry_time=model.entry_time,
            last_update=model.last_update,
            highest_price=model.highest_price,
            lowest_price=model.lowest_price,
            stop_loss_price=model.stop_loss_price,
            take_profit_price=model.take_profit_price,
            trailing_stop_pct=model.trailing_stop_pct,
            metadata=dict(model.extra_metadata or {}),
            mark_price=model.mark_price,
            is_open=model.is_open,
            trades=trades,
        )
