"""gRPC server exposing the portfolio service."""

from __future__ import annotations

import json
import logging
from concurrent import futures
from datetime import datetime
from decimal import Decimal
from typing import Optional

import grpc
from google.protobuf import empty_pb2

from .config import PortfolioConfig
from .service import PortfolioService
from .schemas import (
    PortfolioState,
    PortfolioStatistics,
    PositionRead,
    PriceCacheEntry,
    RiskCheckResult,
    TradeCreate,
    TradeRead,
)
from .protos import portfolio_pb2, portfolio_pb2_grpc

logger = logging.getLogger(__name__)


def decimal_to_str(value: Optional[Decimal]) -> str:
    return "" if value is None else format(value, "f")


def datetime_to_str(value: Optional[datetime]) -> str:
    return "" if value is None else value.isoformat()


class PortfolioGrpcService(portfolio_pb2_grpc.PortfolioServicer):
    """Implementation of the gRPC service."""

    def __init__(self, service: PortfolioService):
        self.service = service

    # ------------------------------------------------------------------
    # Utility conversions
    # ------------------------------------------------------------------
    def _position_to_proto(self, position: PositionRead) -> portfolio_pb2.Position:
        trades = [self._trade_to_proto(trade) for trade in position.trades]
        return portfolio_pb2.Position(
            symbol=position.symbol,
            side=position.side,
            total_amount=str(position.total_amount),
            average_price=str(position.average_price),
            realized_pnl=str(position.realized_pnl),
            fees_paid=str(position.fees_paid),
            entry_time=position.entry_time.isoformat(),
            last_update=position.last_update.isoformat(),
            highest_price=decimal_to_str(position.highest_price),
            lowest_price=decimal_to_str(position.lowest_price),
            stop_loss_price=decimal_to_str(position.stop_loss_price),
            take_profit_price=decimal_to_str(position.take_profit_price),
            trailing_stop_pct=decimal_to_str(position.trailing_stop_pct),
            metadata_json=json.dumps(position.metadata or {}),
            mark_price=decimal_to_str(position.mark_price),
            is_open=position.is_open,
            trades=trades,
        )

    def _trade_to_proto(self, trade) -> portfolio_pb2.Trade:
        metadata = getattr(trade, "metadata", None) or {}
        position_symbol = getattr(trade, "position_symbol", "")
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
            metadata_json=json.dumps(metadata),
            position_symbol=position_symbol or "",
        )

    def _proto_to_trade_create(self, trade_proto: portfolio_pb2.Trade) -> TradeCreate:
        metadata = json.loads(trade_proto.metadata_json or "{}")
        return TradeCreate(
            id=trade_proto.id,
            symbol=trade_proto.symbol,
            side=trade_proto.side,
            amount=Decimal(trade_proto.amount),
            price=Decimal(trade_proto.price),
            timestamp=datetime.fromisoformat(trade_proto.timestamp),
            strategy=trade_proto.strategy or None,
            exchange=trade_proto.exchange or None,
            fees=Decimal(trade_proto.fees or "0"),
            status=trade_proto.status,
            order_id=trade_proto.order_id or None,
            client_order_id=trade_proto.client_order_id or None,
            metadata=metadata,
        )

    def _proto_to_trade_read(self, trade_proto: portfolio_pb2.Trade) -> TradeRead:
        metadata = json.loads(trade_proto.metadata_json or "{}")
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
            metadata=metadata,
            position_symbol=trade_proto.position_symbol or None,
        )

    # ------------------------------------------------------------------
    # RPC implementations
    # ------------------------------------------------------------------
    def GetState(self, request: empty_pb2.Empty, context) -> portfolio_pb2.PortfolioStateResponse:
        state = self.service.get_state()
        return portfolio_pb2.PortfolioStateResponse(state=self._state_to_proto(state))

    def UpdateState(self, request: portfolio_pb2.PortfolioStateRequest, context) -> portfolio_pb2.PortfolioStateResponse:
        state = self._proto_to_state(request.state)
        updated = self.service.replace_state(state)
        return portfolio_pb2.PortfolioStateResponse(state=self._state_to_proto(updated))

    def RecordTrade(self, request: portfolio_pb2.TradeRequest, context) -> portfolio_pb2.PositionResponse:
        trade = self._proto_to_trade_create(request.trade)
        position = self.service.record_trade(trade)
        return portfolio_pb2.PositionResponse(position=self._position_to_proto(position), has_position=True)

    def UpdatePrice(self, request: portfolio_pb2.PriceUpdateRequest, context) -> portfolio_pb2.PositionResponse:
        position = self.service.update_price(request.symbol, Decimal(request.price))
        has_position = position is not None
        proto_position = self._position_to_proto(position) if position else portfolio_pb2.Position()
        return portfolio_pb2.PositionResponse(position=proto_position, has_position=has_position)

    def ComputePnl(self, request: portfolio_pb2.PnlRequest, context) -> portfolio_pb2.PnlResponse:
        symbol = request.symbol or None
        pnl = self.service.compute_pnl(symbol)
        return portfolio_pb2.PnlResponse(
            realized=str(pnl.realized),
            unrealized=str(pnl.unrealized),
            total=str(pnl.total),
        )

    def CheckRisk(self, request: empty_pb2.Empty, context) -> portfolio_pb2.RiskCheckResponse:
        results = self.service.check_risk_limits()
        return portfolio_pb2.RiskCheckResponse(
            results=[
                portfolio_pb2.RiskCheckEntry(
                    name=result.name, passed=result.passed, message=result.message
                )
                for result in results
            ]
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _state_to_proto(self, state: PortfolioState) -> portfolio_pb2.PortfolioState:
        return portfolio_pb2.PortfolioState(
            trades=[self._trade_to_proto(trade) for trade in state.trades],
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

    def _proto_to_state(self, state_proto: portfolio_pb2.PortfolioState) -> PortfolioState:
        trades = [self._proto_to_trade_read(trade) for trade in state_proto.trades]
        positions = [self._proto_to_position(pos) for pos in state_proto.positions]
        closed = [self._proto_to_position(pos) for pos in state_proto.closed_positions]
        price_cache = [
            PriceCacheEntry(
                symbol=entry.symbol,
                price=Decimal(entry.price),
                updated_at=datetime.fromisoformat(entry.updated_at) if entry.updated_at else datetime.utcnow(),
            )
            for entry in state_proto.price_cache
        ]
        stats_proto = state_proto.statistics
        statistics = PortfolioStatistics(
            total_trades=stats_proto.total_trades,
            total_volume=Decimal(stats_proto.total_volume or "0"),
            total_fees=Decimal(stats_proto.total_fees or "0"),
            total_realized_pnl=Decimal(stats_proto.total_realized_pnl or "0"),
            last_updated=datetime.fromisoformat(stats_proto.last_updated)
            if stats_proto.last_updated
            else None,
        )
        return PortfolioState(
            trades=trades,
            positions=positions,
            closed_positions=closed,
            price_cache=price_cache,
            statistics=statistics,
        )

    def _proto_to_position(self, proto: portfolio_pb2.Position) -> PositionRead:
        trades = [self._proto_to_trade_read(trade) for trade in proto.trades]
        metadata = json.loads(proto.metadata_json or "{}")
        return PositionRead(
            symbol=proto.symbol,
            side=proto.side,
            total_amount=Decimal(proto.total_amount or "0"),
            average_price=Decimal(proto.average_price or "0"),
            realized_pnl=Decimal(proto.realized_pnl or "0"),
            fees_paid=Decimal(proto.fees_paid or "0"),
            entry_time=datetime.fromisoformat(proto.entry_time) if proto.entry_time else datetime.utcnow(),
            last_update=datetime.fromisoformat(proto.last_update) if proto.last_update else datetime.utcnow(),
            highest_price=Decimal(proto.highest_price) if proto.highest_price else None,
            lowest_price=Decimal(proto.lowest_price) if proto.lowest_price else None,
            stop_loss_price=Decimal(proto.stop_loss_price) if proto.stop_loss_price else None,
            take_profit_price=Decimal(proto.take_profit_price) if proto.take_profit_price else None,
            trailing_stop_pct=Decimal(proto.trailing_stop_pct) if proto.trailing_stop_pct else None,
            metadata=metadata,
            mark_price=Decimal(proto.mark_price) if proto.mark_price else None,
            is_open=proto.is_open,
            trades=trades,
        )


def serve(config: Optional[PortfolioConfig] = None, max_workers: int = 10) -> grpc.Server:
    config = config or PortfolioConfig.from_env()
    service = PortfolioService(config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    portfolio_pb2_grpc.add_PortfolioServicer_to_server(
        PortfolioGrpcService(service), server
    )
    server.add_insecure_port(f"{config.grpc_host}:{config.grpc_port}")
    logger.info(
        "Starting portfolio gRPC server on %s:%s", config.grpc_host, config.grpc_port
    )
    server.start()
    return server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    srv = serve()
    try:
        srv.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down portfolio gRPC server")
        srv.stop(grace=None)
