import logging
import math
import random
import time
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Literal
from .base import CallableStrategy

logger = logging.getLogger(__name__)


@dataclass
class Fees:
    """Simple container for maker fee information."""

    maker_bp: float


@dataclass
class MakerSpreadConfig:
    """Configuration for the maker spread strategy."""

    size_usd: float
    tick_size: float
    max_spread_bp: float = 10.0
    edge_margin_bp: float = 0.0
    queue_timeout_ms: int = 5_000
    max_live_quotes: int = 2


@dataclass
class QuotePlan:
    side: Literal["buy", "sell"]
    price: float
    size_usd: float
    post_only: bool = True


def _compute_obi(snapshot) -> float:
    bid_vol = max(getattr(snapshot, "bid_size", 0.0), 0.0)
    ask_vol = max(getattr(snapshot, "ask_size", 0.0), 0.0)
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def compute_edge(snapshot) -> float:
    """Return expected edge in basis points from microprice vs mid."""

    best_bid = getattr(snapshot, "best_bid", 0.0)
    best_ask = getattr(snapshot, "best_ask", 0.0)
    bid_vol = max(getattr(snapshot, "bid_size", 0.0), 0.0)
    ask_vol = max(getattr(snapshot, "ask_size", 0.0), 0.0)

    mid = (best_bid + best_ask) / 2
    vol_total = bid_vol + ask_vol
    if vol_total <= 0 or mid <= 0:
        return 0.0

    micro = (best_ask * bid_vol + best_bid * ask_vol) / vol_total
    edge_bp = 10_000 * (micro - mid) / mid
    if not math.isfinite(edge_bp):
        return 0.0
    return max(min(edge_bp, 1_000), -1_000)


def should_quote(snapshot, fees: Fees, cfg: MakerSpreadConfig) -> Optional[QuotePlan]:
    """Decide whether to quote and return a :class:`QuotePlan` when suitable."""

    spread_bp = getattr(snapshot, "spread_bp", 0.0)
    if spread_bp > cfg.max_spread_bp:
        logger.info("suppressed_spread")
        return None

    edge_raw = compute_edge(snapshot)
    edge_bp = abs(edge_raw)
    if edge_bp <= fees.maker_bp + cfg.edge_margin_bp:
        logger.info("suppressed_fee")
        return None

    rv_pct = getattr(snapshot, "rv_short_pct", 0.0)
    if rv_pct > 95:
        logger.info("suppressed_vol")
        return None

    side = "sell" if edge_raw > 0 else "buy"
    improve = getattr(snapshot, "tick_size", cfg.tick_size) * random.uniform(0.25, 0.5)
    if side == "sell":
        price = getattr(snapshot, "best_ask", 0.0) - improve
    else:
        price = getattr(snapshot, "best_bid", 0.0) + improve

    logger.info("kept")
    return QuotePlan(side=side, price=price, size_usd=cfg.size_usd)


class MakerQuoter:
    """Simple maker quote manager."""

    def __init__(self, exchange, fees: Fees, cfg: MakerSpreadConfig):
        self.exchange = exchange
        self.fees = fees
        self.cfg = cfg
        self.live_quotes: Dict[str, Dict[str, object]] = {}

    def _place(self, plan: QuotePlan) -> str:
        if self.exchange:
            return self.exchange.place_order(
                plan.side, plan.price, plan.size_usd, post_only=plan.post_only
            )
        return f"{plan.side}-{time.time()}"

    def _cancel(self, order_id: str) -> None:
        if self.exchange:
            try:
                self.exchange.cancel_order(order_id)
            except Exception:  # pragma: no cover - best effort
                pass

    def on_snapshot(self, snapshot) -> None:
        now = time.time() * 1_000
        current_obi = _compute_obi(snapshot)

        for side, info in list(self.live_quotes.items()):
            if now - info["ts"] > self.cfg.queue_timeout_ms:
                self._cancel(info["id"])
                del self.live_quotes[side]
                logger.info("queue_timeout")
                continue
            if info["obi"] * current_obi < 0:
                self._cancel(info["id"])
                del self.live_quotes[side]
                logger.info("obi_flip")

        plan = should_quote(snapshot, self.fees, self.cfg)
        if not plan:
            return
        if plan.side in self.live_quotes:
            return
        if len(self.live_quotes) >= self.cfg.max_live_quotes:
            return

        order_id = self._place(plan)
        self.live_quotes[plan.side] = {"id": order_id, "ts": now, "obi": current_obi}


def generate_signal(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **kwargs,
) -> Tuple[float, str]:
    """Maker spread signal generator for sideways markets."""
    if df.empty or len(df) < 20:
        return 0.0, "none"

    # Simple spread-based signal for sideways markets
    # Calculate recent volatility to ensure we're in a ranging market
    returns = df["close"].pct_change()
    volatility = returns.rolling(20).std().iloc[-1]

    # In sideways markets, we want low volatility
    if pd.isna(volatility) or volatility > 0.02:  # More than 2% daily volatility
        return 0.0, "none"

    # Check if price is within a reasonable range (not trending strongly)
    sma_20 = df["close"].rolling(20).mean().iloc[-1]
    current_price = df["close"].iloc[-1]
    deviation = abs(current_price - sma_20) / sma_20

    # Price should be within 3% of 20-period moving average
    if deviation > 0.03:
        return 0.0, "none"

    # Generate a moderate confidence signal for maker spread
    score = 0.6

    # Boost score if volume is present and reasonable
    if "volume" in df.columns:
        vol_sma = df["volume"].rolling(20).mean().iloc[-1]
        if not pd.isna(vol_sma) and vol_sma > 0:
            score = min(score * 1.2, 0.8)  # Boost but cap at 0.8

    return score, "maker_spread"


class regime_filter:
    """Match sideways regime for maker strategies."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "sideways"

strategy = CallableStrategy('maker_spread', generate_signal)
