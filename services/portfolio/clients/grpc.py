"""gRPC client for the portfolio service."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from typing import Iterator, Optional

import grpc
from google.protobuf import empty_pb2

from ..config import get_grpc_target
from ..schemas import (
    PnlBreakdown,
    PortfolioState,
    PortfolioStatistics,
    PositionRead,
    PriceCacheEntry,
    RiskCheckResult,
    TradeCreate,
    TradeRead,
)
from ..protos import portfolio_pb2, portfolio_pb2_grpc


def _trade_to_proto(trade: TradeCreate) -> portfolio_pb2.Trade:
    return portfolio_pb2.Trade(
        id=trade.id,
        symbol=trade.symbol,
        side=trade.side,
        amount=str(trade.amount),
        price=str(trade.price),
        timestamp=trade.timestamp.isoformat(),
        strategy=trade.strategy or "",
        exchange=trade.exchange or "",
        fees=str(trade.fees),
        status=trade.status,
        order_id=trade.order_id or "",
        client_order_id=trade.client_order_id or "",
        metadata_json=json.dumps(trade.metadata or {}),
    )


def _proto_to_trade(trade_proto: portfolio_pb2.Trade) -> TradeRead:
    return TradeRead(
        id=trade_proto.id,
        symbol=trade_proto.symbol,
        side=trade_proto.side,
        amount=Decimal(trade_proto.amount or "0"),
        price=Decimal(trade_proto.price or "0"),
        timestamp=datetime.fromisoformat(trade_proto.timestamp),
        strategy=trade_proto.strategy or None,
        exchange=trade_proto.exchange or None,
        fees=Decimal(trade_proto.fees or "0"),
        status=trade_proto.status,
        order_id=trade_proto.order_id or None,
        client_order_id=trade_proto.client_order_id or None,
        metadata=json.loads(trade_proto.metadata_json or "{}"),
        position_symbol=trade_proto.position_symbol or None,
    )


def _proto_to_position(position_proto: portfolio_pb2.Position) -> PositionRead:
    trades = [_proto_to_trade(trade) for trade in position_proto.trades]
    return PositionRead(
        symbol=position_proto.symbol,
        side=position_proto.side,
        total_amount=Decimal(position_proto.total_amount or "0"),
        average_price=Decimal(position_proto.average_price or "0"),
        realized_pnl=Decimal(position_proto.realized_pnl or "0"),
        fees_paid=Decimal(position_proto.fees_paid or "0"),
        entry_time=datetime.fromisoformat(position_proto.entry_time)
        if position_proto.entry_time
        else datetime.utcnow(),
        last_update=datetime.fromisoformat(position_proto.last_update)
        if position_proto.last_update
        else datetime.utcnow(),
        highest_price=Decimal(position_proto.highest_price)
        if position_proto.highest_price
        else None,
        lowest_price=Decimal(position_proto.lowest_price)
        if position_proto.lowest_price
        else None,
        stop_loss_price=Decimal(position_proto.stop_loss_price)
        if position_proto.stop_loss_price
        else None,
        take_profit_price=Decimal(position_proto.take_profit_price)
        if position_proto.take_profit_price
        else None,
        trailing_stop_pct=Decimal(position_proto.trailing_stop_pct)
        if position_proto.trailing_stop_pct
        else None,
        metadata=json.loads(position_proto.metadata_json or "{}"),
        mark_price=Decimal(position_proto.mark_price) if position_proto.mark_price else None,
        is_open=position_proto.is_open,
        trades=trades,
    )


def _proto_to_state(state_proto: portfolio_pb2.PortfolioState) -> PortfolioState:
    positions = [_proto_to_position(pos) for pos in state_proto.positions]
    closed = [_proto_to_position(pos) for pos in state_proto.closed_positions]
    trades = [_proto_to_trade(trade) for trade in state_proto.trades]
    price_cache = [
        PriceCacheEntry(
            symbol=entry.symbol,
            price=Decimal(entry.price),
            updated_at=datetime.fromisoformat(entry.updated_at)
            if entry.updated_at
            else datetime.utcnow(),
        )
        for entry in state_proto.price_cache
    ]
    stats = state_proto.statistics
    statistics = PortfolioStatistics(
        total_trades=stats.total_trades,
        total_volume=Decimal(stats.total_volume or "0"),
        total_fees=Decimal(stats.total_fees or "0"),
        total_realized_pnl=Decimal(stats.total_realized_pnl or "0"),
        last_updated=datetime.fromisoformat(stats.last_updated)
        if stats.last_updated
        else None,
    )
    return PortfolioState(
        trades=trades,
        positions=positions,
        closed_positions=closed,
        price_cache=price_cache,
        statistics=statistics,
    )


class PortfolioGrpcClient:
    """Typed gRPC client for the portfolio service."""

    def __init__(self, target: Optional[str] = None):
        self.target = target or get_grpc_target()

    @contextmanager
    def channel(self) -> Iterator[grpc.Channel]:
        channel = grpc.insecure_channel(self.target)
        try:
            yield channel
        finally:
            channel.close()

    @contextmanager
    def stub(self) -> Iterator[portfolio_pb2_grpc.PortfolioStub]:
        with self.channel() as channel:
            yield portfolio_pb2_grpc.PortfolioStub(channel)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_state(self) -> PortfolioState:
        with self.stub() as stub:
            response = stub.GetState(empty_pb2.Empty())
            return _proto_to_state(response.state)

    def put_state(self, state: PortfolioState) -> PortfolioState:
        with self.stub() as stub:
            request = portfolio_pb2.PortfolioStateRequest(state=self._state_to_proto(state))
            response = stub.UpdateState(request)
            return _proto_to_state(response.state)

    def record_trade(self, trade: TradeCreate) -> PositionRead:
        with self.stub() as stub:
            request = portfolio_pb2.TradeRequest(trade=_trade_to_proto(trade))
            response = stub.RecordTrade(request)
            return _proto_to_position(response.position)

    def update_price(self, symbol: str, price: Decimal) -> Optional[PositionRead]:
        with self.stub() as stub:
            request = portfolio_pb2.PriceUpdateRequest(symbol=symbol, price=str(price))
            response = stub.UpdatePrice(request)
            if not response.has_position:
                return None
            return _proto_to_position(response.position)

    def compute_pnl(self, symbol: Optional[str] = None) -> PnlBreakdown:
        with self.stub() as stub:
            request = portfolio_pb2.PnlRequest(symbol=symbol or "")
            response = stub.ComputePnl(request)
            return PnlBreakdown(
                realized=Decimal(response.realized or "0"),
                unrealized=Decimal(response.unrealized or "0"),
                total=Decimal(response.total or "0"),
            )

    def check_risk(self) -> list[RiskCheckResult]:
        with self.stub() as stub:
            response = stub.CheckRisk(empty_pb2.Empty())
            return [
                RiskCheckResult(name=entry.name, passed=entry.passed, message=entry.message)
                for entry in response.results
            ]

    def _state_to_proto(self, state: PortfolioState) -> portfolio_pb2.PortfolioState:
        return portfolio_pb2.PortfolioState(
            trades=[_trade_to_proto(TradeCreate(**trade.model_dump(exclude={"position_symbol"}))) for trade in state.trades],
            positions=[self._position_to_proto(pos) for pos in state.positions],
            closed_positions=[self._position_to_proto(pos) for pos in state.closed_positions],
            price_cache=[
                portfolio_pb2.PriceEntry(
                    symbol=entry.symbol,
                    price=str(entry.price),
                    updated_at=entry.updated_at.isoformat(),
                )
                for entry in state.price_cache
            ],
            statistics=portfolio_pb2.Statistics(
                total_trades=state.statistics.total_trades,
                total_volume=str(state.statistics.total_volume),
                total_fees=str(state.statistics.total_fees),
                total_realized_pnl=str(state.statistics.total_realized_pnl),
                last_updated=state.statistics.last_updated.isoformat()
                if state.statistics.last_updated
                else "",
            ),
        )

    def _position_to_proto(self, position: PositionRead) -> portfolio_pb2.Position:
        return portfolio_pb2.Position(
            symbol=position.symbol,
            side=position.side,
            total_amount=str(position.total_amount),
            average_price=str(position.average_price),
            realized_pnl=str(position.realized_pnl),
            fees_paid=str(position.fees_paid),
            entry_time=position.entry_time.isoformat(),
            last_update=position.last_update.isoformat(),
            highest_price=str(position.highest_price) if position.highest_price is not None else "",
            lowest_price=str(position.lowest_price) if position.lowest_price is not None else "",
            stop_loss_price=str(position.stop_loss_price) if position.stop_loss_price is not None else "",
            take_profit_price=str(position.take_profit_price) if position.take_profit_price is not None else "",
            trailing_stop_pct=str(position.trailing_stop_pct) if position.trailing_stop_pct is not None else "",
            metadata_json=json.dumps(position.metadata or {}),
            mark_price=str(position.mark_price) if position.mark_price is not None else "",
            is_open=position.is_open,
            trades=[_trade_to_proto(TradeCreate(**trade.model_dump(exclude={"position_symbol"}))) for trade in position.trades],
        )
