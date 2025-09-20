"""FastAPI REST API exposing the portfolio service."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Optional

import logging

from fastapi import Body, Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging_compat import configure_logging

from .config import PortfolioConfig
from .service import PortfolioService
from .schemas import PnlBreakdown, PortfolioState, PositionRead, TradeCreate

monitoring_settings = get_monitoring_settings().for_service("portfolio-service")
monitoring_settings.metrics.default_labels.setdefault("component", "portfolio")
configure_logging(monitoring_settings)

logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio Service", version="1.0.0")
instrument_fastapi_app(app, settings=monitoring_settings)


class ClosePositionsPayload(BaseModel):
    max_age_hours: int = Field(default=24, ge=0, description="Close positions older than this many hours")
    symbols: Optional[list[str]] = Field(
        default=None,
        description="Optional explicit list of symbols to close regardless of age",
    )


@lru_cache(maxsize=1)
def get_service() -> PortfolioService:
    config = PortfolioConfig.from_env()
    logger.debug(
        "Initialising portfolio service", extra={"host": config.rest_host, "port": config.rest_port}
    )
    return PortfolioService(config)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/state", response_model=PortfolioState)
def get_state(service: PortfolioService = Depends(get_service)) -> PortfolioState:
    return service.get_state()


@app.put("/state", response_model=PortfolioState)
def put_state(
    state: PortfolioState, service: PortfolioService = Depends(get_service)
) -> PortfolioState:
    return service.replace_state(state)


@app.post("/trades", response_model=PositionRead, status_code=status.HTTP_201_CREATED)
def post_trade(
    trade: TradeCreate, service: PortfolioService = Depends(get_service)
) -> PositionRead:
    return service.record_trade(trade)


@app.post("/trades/batch")
def post_trades_batch(
    trades: list[TradeCreate], service: PortfolioService = Depends(get_service)
) -> dict:
    """Batch create multiple trades."""
    created_count = 0
    errors = []

    for i, trade in enumerate(trades):
        try:
            service.record_trade(trade)
            created_count += 1
        except Exception as e:
            logger.error(f"Failed to create trade {trade.id}: {e}")
            errors.append({
                "index": i,
                "trade_id": getattr(trade, 'id', f"trade_{i}"),
                "error": str(e)
            })

    response = {
        "created": created_count,
        "total": len(trades),
        "errors": errors
    }

    # Return appropriate status code based on results
    if errors and created_count == 0:
        # All trades failed
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response
        )
    elif errors:
        # Some trades failed, some succeeded - use 207 Multi-Status
        response["status"] = "partial_success"
        # FastAPI doesn't directly support 207, so we'll use 200 with status info
        return response
    else:
        # All trades succeeded
        return response


@app.get("/positions", response_model=list[PositionRead])
def list_positions(service: PortfolioService = Depends(get_service)) -> list[PositionRead]:
    return service.get_state().positions


@app.get("/positions/{symbol}", response_model=PositionRead)
def get_position(
    symbol: str, service: PortfolioService = Depends(get_service)
) -> PositionRead:
    state = service.get_state()
    for position in state.positions + state.closed_positions:
        if position.symbol == symbol:
            return position
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Position not found")


@app.post("/prices", response_model=Optional[PositionRead])
def post_price(
    symbol: str,
    price: Decimal,
    service: PortfolioService = Depends(get_service),
) -> Optional[PositionRead]:
    return service.update_price(symbol, price)


@app.get("/pnl", response_model=PnlBreakdown)
def get_pnl(
    symbol: Optional[str] = None, service: PortfolioService = Depends(get_service)
) -> PnlBreakdown:
    return service.compute_pnl(symbol)


@app.get("/statistics")
def get_statistics(service: PortfolioService = Depends(get_service)) -> dict:
    """Get portfolio statistics."""
    return service.get_statistics_summary()


@app.get("/risk", response_model=list)
def get_risk(service: PortfolioService = Depends(get_service)) -> list:
    return [result.model_dump() for result in service.check_risk_limits()]


@app.post("/positions/close-stale")
def close_stale_positions(
    payload: ClosePositionsPayload = Body(default=None),
    service: PortfolioService = Depends(get_service),
) -> dict:
    if payload is None:
        payload = ClosePositionsPayload()

    result = service.close_stale_positions(
        max_age_hours=payload.max_age_hours,
        symbols=payload.symbols,
    )
    return {"success": True, **result}
