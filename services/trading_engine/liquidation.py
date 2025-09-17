"""Helpers used to perform emergency liquidation of open positions."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from services.execution import ExecutionService, ExecutionServiceConfig, OrderRequest
from services.portfolio.clients.interface import PortfolioServiceClient
from services.portfolio.schemas import PositionRead


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PositionLiquidation:
    """Outcome of attempting to close a single open position."""

    symbol: str
    side: str
    amount: float
    status: str
    error: Optional[str] = None


@dataclass(slots=True)
class LiquidationReport:
    """Aggregated results for an emergency liquidation run."""

    positions: list[PositionLiquidation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_positions(self) -> int:
        return len(self.positions)

    @property
    def closed_positions(self) -> int:
        return sum(1 for position in self.positions if position.status == "closed")

    @property
    def failed_positions(self) -> int:
        return sum(1 for position in self.positions if position.status == "failed")

    @property
    def status(self) -> str:
        if self.total_positions == 0:
            return "completed" if not self.errors else "failed"
        if self.failed_positions == 0 and not self.errors:
            return "completed"
        if self.closed_positions > 0:
            return "partial"
        return "failed"


class LiquidationHelper:
    """Coordinate liquidation of positions via portfolio and execution services."""

    def __init__(
        self,
        *,
        portfolio_client: Optional[PortfolioServiceClient] = None,
        execution_service: Optional[ExecutionService] = None,
        execution_config: Optional[Mapping[str, Any]] = None,
        ack_timeout: float = 15.0,
        fill_timeout: float = 60.0,
    ) -> None:
        self._portfolio_client = portfolio_client or PortfolioServiceClient()
        self._execution_service = execution_service
        self._execution_config = dict(execution_config or {})
        self._ack_timeout = float(ack_timeout)
        self._fill_timeout = float(fill_timeout)
        self._dry_run = bool(self._execution_config.get("dry_run", True))
        execution_mode = self._execution_config.get("execution_mode")
        if execution_mode is not None:
            self._dry_run = str(execution_mode).lower() == "dry_run"

    async def close_all_positions(self) -> LiquidationReport:
        """Liquidate all open positions tracked by the portfolio service."""

        report = LiquidationReport()
        try:
            positions = await asyncio.to_thread(self._portfolio_client.list_positions)
        except Exception as exc:
            logger.exception("Failed to fetch positions for liquidation: %s", exc)
            report.errors.append(f"portfolio_service_error: {exc}")
            return report

        open_positions = [
            position
            for position in positions
            if getattr(position, "is_open", False)
            and float(position.total_amount or 0) > 0
        ]

        if not open_positions:
            logger.info("No open positions found during liquidation request")
            return report

        for position in open_positions:
            outcome = PositionLiquidation(
                symbol=position.symbol,
                side=position.side,
                amount=float(position.total_amount or 0),
                status="failed",
            )
            success, error = await self._close_position(position)
            if success:
                outcome.status = "closed"
                logger.info("Liquidated position %s amount=%s", outcome.symbol, outcome.amount)
            else:
                outcome.error = error or "unknown_error"
                logger.warning(
                    "Failed to liquidate position %s: %s", outcome.symbol, outcome.error
                )
            report.positions.append(outcome)

        return report

    async def _close_position(self, position: PositionRead) -> tuple[bool, Optional[str]]:
        amount = float(position.total_amount or 0)
        if amount <= 0:
            return True, None

        side = "sell" if str(position.side).lower() == "long" else "buy"

        try:
            execution_service = self._get_execution_service()
        except Exception as exc:
            logger.exception("Execution service unavailable for liquidation: %s", exc)
            return False, f"execution_service_error: {exc}"

        ack_subscription = execution_service.subscribe_acks()
        fill_subscription = execution_service.subscribe_fills()

        client_order_id = execution_service.generate_client_order_id(prefix="liq")
        order_request = OrderRequest(
            symbol=position.symbol,
            side=side,
            amount=amount,
            client_order_id=client_order_id,
            dry_run=self._dry_run,
            use_websocket=bool(self._execution_config.get("use_websocket", False)),
            config=dict(self._execution_config),
            metadata={
                "trigger": "close_all_positions",
                "source": "trading_engine",
                "position_side": position.side,
            },
        )

        try:
            await execution_service.submit_order(order_request)
            ack = await self._await_event(ack_subscription, client_order_id, self._ack_timeout)
            if ack is None:
                return False, "ack_timeout"
            if not ack.accepted:
                return False, ack.reason or "order_rejected"

            fill = await self._await_event(
                fill_subscription, client_order_id, self._fill_timeout
            )
            if fill is None:
                return False, "fill_timeout"
            if not fill.success:
                return False, fill.error or "order_failed"
            return True, None
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "Liquidation order for %s failed unexpectedly: %s", position.symbol, exc
            )
            return False, str(exc)
        finally:
            ack_subscription.close()
            fill_subscription.close()

    async def _await_event(self, subscription, client_order_id: str, timeout: float):
        try:
            while True:
                event = await asyncio.wait_for(subscription.get(), timeout=timeout)
                if getattr(event, "client_order_id", None) == client_order_id:
                    return event
        except asyncio.TimeoutError:
            return None

    def _get_execution_service(self) -> ExecutionService:
        if self._execution_service is None:
            config = ExecutionServiceConfig.from_mapping(self._execution_config)
            self._execution_service = ExecutionService(config)
        return self._execution_service


__all__ = [
    "LiquidationHelper",
    "LiquidationReport",
    "PositionLiquidation",
]

