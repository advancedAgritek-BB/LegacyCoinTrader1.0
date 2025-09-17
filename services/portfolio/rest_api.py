"""FastAPI REST API exposing the portfolio service."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status

from .config import PortfolioConfig
from .service import PortfolioService
from .schemas import PnlBreakdown, PortfolioState, PositionRead, TradeCreate

app = FastAPI(title="Portfolio Service", version="1.0.0")


def get_service() -> PortfolioService:
    config = PortfolioConfig.from_env()
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


@app.get("/risk", response_model=list)
def get_risk(service: PortfolioService = Depends(get_service)) -> list:
    return [result.model_dump() for result in service.check_risk_limits()]
