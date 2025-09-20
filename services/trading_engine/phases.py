"""Default phases executed by the trading engine service."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import pandas as pd

from crypto_bot.utils.symbol_validator import validate_symbols_production
from crypto_bot.utils.trade_manager import create_trade
from crypto_bot.utils.pipeline_logger import get_pipeline_logger

from libs.services.interfaces import (
    CacheUpdateResponse,
    LoadSymbolsRequest,
    MultiTimeframeOHLCVRequest,
    RegimeCacheRequest,
    StrategyBatchRequest,
    StrategyBatchResponse,
    StrategyEvaluationPayload,
    StrategyEvaluationResult,
    TokenDiscoveryRequest,
    TokenDiscoveryResponse,
    TradeExecutionRequest,
)

logger = logging.getLogger(__name__)
pipeline_logger = get_pipeline_logger()


def _unique_symbols(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def _resolve_order_price(order: Mapping[str, Any] | None, fallback: float) -> float:
    """Extract a sensible fill price from an order payload."""

    if not isinstance(order, Mapping):
        return fallback

    price_keys = (
        "average",
        "avg_price",
        "price",
        "fill_price",
        "fillPrice",
    )
    for key in price_keys:
        value = order.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    info = order.get("info")
    if isinstance(info, Mapping):
        for key in ("price", "avg_fill_price", "limit_price"):
            value = info.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return fallback


async def _record_trade_and_sync(
    context: Any,
    *,
    symbol: str,
    side: str,
    amount: float,
    price: float,
    order: Mapping[str, Any] | None,
    strategy: Optional[str],
    score: Optional[float],
    dry_run: bool,
) -> None:
    """Persist trade details so protective stops can be applied immediately."""

    trade_manager = getattr(context, "trade_manager", None)
    if trade_manager is None or price <= 0 or amount <= 0:
        return

    try:
        decimal_amount = Decimal(str(amount))
        decimal_price = Decimal(str(price))
    except Exception as exc:  # pragma: no cover - defensive against malformed data
        logger.warning("Unable to record trade for %s: invalid amount/price (%s)", symbol, exc)
        return

    order_id = None
    client_order_id = None
    exchange_name = None
    fee_amount = Decimal("0")
    metadata: dict[str, Any] = {"source": "trading-engine", "dry_run": dry_run}

    if isinstance(order, Mapping):
        order_id = order.get("id") or order.get("order_id") or order.get("txid")
        client_order_id = (
            order.get("client_order_id")
            or order.get("clientOrderId")
            or order.get("clientOrderID")
        )
        exchange_name = order.get("exchange") or order.get("venue")
        metadata["status"] = order.get("status")
        metadata["score"] = score
        info = order.get("info")
        if isinstance(info, Mapping):
            metadata.setdefault("status", info.get("status"))
            exchange_name = exchange_name or info.get("exchange")
        fee = order.get("fee")
        if isinstance(fee, Mapping):
            fee_value = fee.get("cost") or fee.get("amount")
            if fee_value is not None:
                try:
                    fee_amount = Decimal(str(fee_value))
                except Exception:  # pragma: no cover - optional path
                    fee_amount = Decimal("0")

    metadata.setdefault("score", score)

    try:
        trade = create_trade(
            symbol=symbol,
            side=side,
            amount=decimal_amount,
            price=decimal_price,
            strategy=strategy or "",
            exchange=str(exchange_name or context.config.get("exchange", "")),
            fees=fee_amount,
            order_id=order_id,
            client_order_id=client_order_id,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
    except Exception as exc:  # pragma: no cover - construction failure should not stop trading
        logger.warning("Failed to build trade payload for %s: %s", symbol, exc)
        return

    try:
        await asyncio.to_thread(trade_manager.record_trade, trade)
        if hasattr(context, "sync_positions_from_trade_manager"):
            context.sync_positions_from_trade_manager()
        logger.info(
            "Recorded trade for %s via portfolio service (side=%s amount=%.8f price=%.6f)",
            symbol,
            side,
            amount,
            price,
        )
    except Exception as exc:  # pragma: no cover - remote service failure should not halt trading
        logger.warning("Failed to persist trade for %s: %s", symbol, exc)


def _looks_like_symbol(symbol: object) -> bool:
    if not isinstance(symbol, str):
        return False
    if "/" not in symbol:
        return False
    base, quote = symbol.split("/", 1)
    return bool(base) and bool(quote) and base.isalnum() and quote.isalnum()


async def _score_symbols_for_batching(
    symbols: list[str],
    config: Mapping[str, Any],
    services: Any
) -> list[str]:
    """
    Score symbols based on market metrics to prioritize best trading opportunities.

    Scoring factors:
    - Volume (higher volume = higher score)
    - Price volatility (higher volatility = higher score)
    - Market cap proxy (larger base assets = higher score)
    - Random factor (ensures diversity)
    """
    if not symbols:
        return []

    import random
    from typing import Dict

    # Get market data service
    market_data = getattr(services, "market_data", None)
    if not market_data:
        # Fallback to alphabetical with some randomization
        random.shuffle(symbols)
        return symbols

    scored_symbols: list[tuple[float, str]] = []

    for symbol in symbols:
        try:
            # Get recent OHLCV data for scoring
            ohlcv_data = await _fetch_symbol_metrics(symbol, market_data, config)

            # Calculate score based on multiple factors
            volume_score = min(ohlcv_data.get("volume_score", 0.5), 1.0)
            volatility_score = min(ohlcv_data.get("volatility_score", 0.5), 1.0)
            market_size_score = _calculate_market_size_score(symbol)

            # Get scoring weights from config
            weights_config = config.get("batching_score_weights", {})
            volume_weight = weights_config.get("volume_weight", 0.4)
            volatility_weight = weights_config.get("volatility_weight", 0.4)
            market_size_weight = weights_config.get("market_size_weight", 0.1)
            random_weight = weights_config.get("random_weight", 0.1)

            # Add recency penalty to encourage exploration of new symbols
            import time
            symbol_age_penalty = _calculate_symbol_age_penalty(symbol, config)

            # Weighted scoring
            total_score = (
                volume_score * volume_weight +
                volatility_score * volatility_weight +
                market_size_score * market_size_weight +
                random.random() * random_weight -
                symbol_age_penalty  # Penalty for recently processed symbols
            )

            scored_symbols.append((total_score, symbol))

        except Exception as exc:
            # If scoring fails, give medium score
            logger.debug(f"Failed to score {symbol}: {exc}")
            scored_symbols.append((0.5 + random.random() * 0.1, symbol))

    # Sort by score (highest first) and return symbols
    scored_symbols.sort(key=lambda x: x[0], reverse=True)
    return [symbol for _, symbol in scored_symbols]


async def _fetch_symbol_metrics(
    symbol: str,
    market_data: Any,
    config: Mapping[str, Any]
) -> Dict[str, float]:
    """Fetch basic market metrics for symbol scoring."""
    try:
        # Try to get ticker data for volume information
        ticker_data = await market_data.get_ticker(symbol)

        if ticker_data:
            # Extract volume from ticker data if available
            volume = 0.0
            if hasattr(ticker_data, 'quoteVolume'):
                volume = float(ticker_data.quoteVolume)
            elif hasattr(ticker_data, 'volume'):
                volume = float(ticker_data.volume)
            elif isinstance(ticker_data, dict):
                volume = float(ticker_data.get('quoteVolume', ticker_data.get('volume', 0)))

            # Calculate volume score (normalized 0-1)
            volume_score = min(volume / 1000000, 1.0)  # Normalize to millions

            # For volatility, we'll use a simpler approach since we don't have OHLCV
            # Use price change as a proxy for volatility
            if hasattr(ticker_data, 'percentage'):
                price_change_pct = abs(float(ticker_data.percentage))
                volatility_score = min(price_change_pct / 10.0, 1.0)  # Normalize price change %
            elif isinstance(ticker_data, dict) and 'percentage' in ticker_data:
                price_change_pct = abs(float(ticker_data['percentage']))
                volatility_score = min(price_change_pct / 10.0, 1.0)
            else:
                volatility_score = 0.5  # Default if no change data

            return {
                "volume_score": volume_score,
                "volatility_score": volatility_score,
            }

    except Exception as exc:
        logger.debug(f"Failed to fetch metrics for {symbol}: {exc}")

    # Return default scores if data fetch fails
    return {"volume_score": 0.5, "volatility_score": 0.5}


def _calculate_market_size_score(symbol: str) -> float:
    """Calculate market size score based on base asset."""
    base_asset = symbol.split("/")[0].upper()

    # Major cryptos get higher scores
    major_cryptos = {
        "BTC": 1.0, "ETH": 0.9, "BNB": 0.8, "SOL": 0.8,
        "ADA": 0.7, "DOT": 0.7, "LINK": 0.7, "UNI": 0.6,
        "AAVE": 0.6, "AVAX": 0.6, "MATIC": 0.6, "LTC": 0.5,
        "XRP": 0.5, "DOGE": 0.4, "SHIB": 0.3
    }

    return major_cryptos.get(base_asset, 0.2)  # Default score for unknown assets


def _calculate_symbol_age_penalty(symbol: str, config: Mapping[str, Any]) -> float:
    """
    Calculate a penalty for symbols that were recently processed.
    This encourages exploration of new symbols over time.
    """
    # Simple implementation: use symbol hash to create pseudo-random age
    # In a real implementation, you'd track actual processing timestamps
    import hashlib

    # Create a hash of the symbol + current time window
    current_window = int(time.time() // 3600)  # 1-hour windows
    symbol_hash = hashlib.md5(f"{symbol}_{current_window}".encode()).hexdigest()

    # Convert hash to a number between 0 and 1
    hash_int = int(symbol_hash[:8], 16)
    age_penalty = (hash_int % 1000) / 1000.0  # 0.0 to 1.0

    # Scale the penalty (0.0 to 0.3 to not completely override other scores)
    return age_penalty * 0.3


def _normalize_usd_symbol(symbol: object) -> Optional[str]:
    """Convert exchange-specific notation to ``BASE/USD`` where possible."""

    if not isinstance(symbol, str):
        return None

    raw = symbol.strip()
    if not raw:
        return None

    upper = raw.upper()
    base: Optional[str] = None

    if "/" in upper:
        maybe_base, maybe_quote = upper.split("/", 1)
        if maybe_quote == "USD":
            base = maybe_base
    elif upper.endswith("ZUSD") and len(upper) > 4:
        base = upper[:-4]
    elif upper.endswith("USD") and len(upper) > 3:
        base = upper[:-3]

    if not base:
        return None

    if base.startswith("X") and len(base) > 3:
        base = base[1:]
    if base.endswith("Z") and len(base) > 3:
        base = base[:-1]
    base = base.replace("XBT", "BTC")

    if not base or not base.isalnum():
        return None

    return f"{base}/USD"


def _normalize_exchange_symbol(symbol: object) -> Optional[str]:
    """Strip exchange prefixes (e.g. ``KRAKEN:``) and normalise case."""

    if not isinstance(symbol, str):
        return None

    cleaned = symbol.strip()
    if not cleaned:
        return None

    if ":" in cleaned:
        _, _, cleaned = cleaned.partition(":")
        cleaned = cleaned.strip()

    if not cleaned:
        return None

    normalised_usd = _normalize_usd_symbol(cleaned)
    if normalised_usd:
        return normalised_usd

    if "/" in cleaned:
        base, _, quote = cleaned.partition("/")
        if not base or not quote:
            return None
        return f"{base.upper()}/{quote.upper()}"

    return cleaned.upper()


def _select_dataframe(
    df_cache: Mapping[str, Mapping[str, pd.DataFrame]], symbol: str
) -> Optional[pd.DataFrame]:
    for timeframe, entries in df_cache.items():
        frame = entries.get(symbol)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            logger.debug("Using %s timeframe for %s", timeframe, symbol)
            return frame
    return None


def _infer_regime(
    regime_cache: Mapping[str, Mapping[str, object]], symbol: str, default: str
) -> str:
    for entries in regime_cache.values():
        data = entries.get(symbol)
        if isinstance(data, Mapping):
            regime = data.get("regime") or data.get("label")
            if isinstance(regime, str) and regime:
                return regime
    return default


async def prepare_cycle(context: Any) -> None:
    """Initialise cycle metadata and refresh cached positions."""

    # Start pipeline logging for this cycle
    cycle_id = context.metadata.get("cycle_id") or context.state.get("cycles", 0)
    pipeline_logger.start_pipeline(cycle_id=cycle_id)

    context.metadata.setdefault("events", []).append("prepare")
    context.metadata["cycle_started_at"] = (
        datetime.now(timezone.utc).isoformat()
    )
    context.metadata.setdefault(
        "cycle_sequence", context.state.get("cycles", 0)
    )

    # Skip position sync for now to avoid coroutine issues
    # will fix after basic scanning works
    pass

    # Handle positions safely - check for coroutine first
    positions_attr = getattr(context, "positions", {})

    if asyncio.iscoroutine(positions_attr):
        logger.info("Positions data is being loaded asynchronously")
        positions = await positions_attr
    else:
        positions = positions_attr

    if not isinstance(positions, dict):
        logger.warning(
            f"Expected positions dict, got {type(positions)}, using empty dict"
        )
        positions = {}

    open_positions = sorted(positions.keys())
    position_value = sum(float(pos.get("amount", 0) * pos.get("entry_price", 0))
                        for pos in positions.values() if isinstance(pos, dict))

    logger.info(
        f"📊 Cycle #{cycle_id} initialized: {len(open_positions)} open positions "
        f"worth ${position_value:.2f}"
    )
    context.metadata["open_positions"] = open_positions
    context.metadata["portfolio_value"] = position_value


async def discover_markets(context: Any) -> None:
    """Build the symbol batch using static config and discovery results."""

    # Start discovery phase
    pipeline_logger.start_phase("discovery")

    config = getattr(context, "config", {}) or {}
    static_symbols = list(config.get("symbols", []))
    discovery_cfg = (
        config.get("enhanced_scanning")
        or config.get("token_discovery")
        or {}
    )

    services = getattr(context, "services", None)
    market_data = getattr(services, "market_data", None)

    exchange_symbols: list[str] = []
    discovery_start = time.time()

    if market_data is not None:
        exchange_id = str(config.get("exchange", "kraken"))
        loader_cfg = {}
        raw_loader_cfg = (
            config.get("symbol_loader") or config.get("market_data") or {}
        )
        if isinstance(raw_loader_cfg, Mapping):
            loader_cfg = dict(raw_loader_cfg)
        exclude: Iterable[str] = config.get("symbol_exclude") or ()
        try:
            request = LoadSymbolsRequest(
                exchange_id=exchange_id,
                exclude=tuple(exclude),
                config=loader_cfg,
            )
            symbols_response = await market_data.load_symbols(request)
            discovery_time = time.time() - discovery_start
            pipeline_logger.log_performance("Exchange symbol loading", discovery_time)
        except Exception:  # pragma: no cover - external dependency
            logger.debug("Failed to load symbols from %s", exchange_id, exc_info=True)
        else:
            normalised = (_normalize_usd_symbol(symbol) for symbol in (symbols_response.symbols or []))
            exchange_symbols = [symbol for symbol in normalised if symbol]
            if not exchange_symbols:
                logger.debug("No USD pairs discovered from %s", exchange_id)

    # Separate discovery for DEX and CEX tokens
    dex_tokens: list[str] = []
    cex_tokens: list[str] = []
    cex_symbol_sources: dict[str, str] = {}
    discovery = getattr(services, "token_discovery", None)
    if discovery is not None:
        try:
            request = TokenDiscoveryRequest(config=discovery_cfg)
            response: TokenDiscoveryResponse = await discovery.discover_tokens(request)

            # Get separate token lists
            dex_tokens = list(response.dex_tokens or [])
            raw_cex_tokens = list(response.cex_tokens or [])
            for raw_symbol in raw_cex_tokens:
                normalised = _normalize_exchange_symbol(raw_symbol)
                if not normalised:
                    continue
                cex_symbol_sources.setdefault(normalised, str(raw_symbol))
                cex_tokens.append(normalised)

            # Fallback to combined list if separate lists not provided
            if not dex_tokens and not cex_tokens:
                discovered = list(response.tokens or [])
                # Try to separate them based on format
                for token in discovered:
                    normalised = _normalize_exchange_symbol(token)
                    if not normalised:
                        continue

                    raw_value = str(token)

                    if ":" in raw_value:
                        cex_symbol_sources.setdefault(normalised, raw_value)
                        cex_tokens.append(normalised)
                        continue

                    if "/" in normalised:
                        if normalised.upper().endswith("/USDC"):
                            dex_tokens.append(normalised)
                        else:
                            cex_symbol_sources.setdefault(normalised, raw_value)
                            cex_tokens.append(normalised)
                    else:
                        dex_tokens.append(raw_value)

        except Exception:  # pragma: no cover - discovery is optional
            logger.debug("Token discovery failed", exc_info=True)

    # Apply randomization to discovered tokens before logging
    import random
    all_discovered_tokens = dex_tokens + cex_tokens
    if all_discovered_tokens:
        # Use cycle ID + timestamp for deterministic but varying randomization
        cycle_seed = hash(f"{context.metadata.get('cycle_id') or context.metadata.get('cycle_sequence', 0)}_{int(time.time() // 60)}") % 1000000
        random.seed(cycle_seed)
        random.shuffle(all_discovered_tokens)

        # Re-separate after shuffling
        dex_tokens = [s for s in all_discovered_tokens if s in dex_tokens]
        cex_tokens = [s for s in all_discovered_tokens if s in cex_tokens]

    # Log discovery results
    for symbol in exchange_symbols[:10]:  # Log first 10 to avoid spam
        pipeline_logger.log_discovery(symbol, "exchange", confidence=0.9)

    for symbol in dex_tokens[:10]:
        pipeline_logger.log_discovery(symbol, "dex_scanner", confidence=0.7)

    for symbol in cex_tokens[:10]:
        pipeline_logger.log_discovery(symbol, "cex_scanner", confidence=0.8)

    # Keep exchange symbols separate from discovered tokens
    combined = _unique_symbols(exchange_symbols + static_symbols + dex_tokens + cex_tokens)
    context.metadata["exchange_symbol_candidates"] = exchange_symbols
    context.metadata["discovered_dex_tokens"] = dex_tokens
    context.metadata["discovered_cex_tokens"] = cex_tokens
    if cex_symbol_sources:
        context.metadata["cex_symbol_sources"] = dict(cex_symbol_sources)
    context.metadata["dex_token_count"] = len(dex_tokens)
    context.metadata["cex_token_count"] = len(cex_tokens)

    # Separate DEX and CEX tokens based on their source and format
    dex_candidates: list[str] = []
    cex_candidates: list[str] = []
    static_candidates: list[str] = []

    for symbol in combined:
        # Keep discovered DEX tokens even if they do not follow BASE/QUOTE format
        if symbol in dex_tokens:
            dex_candidates.append(symbol)
            continue

        is_cex_symbol = symbol in cex_tokens
        base, _, quote = symbol.partition("/")
        if not base or not quote:
            if is_cex_symbol:
                cex_candidates.append(symbol)
            else:
                logger.debug("Skipping non-symbol entry %s", symbol)
            continue

        if is_cex_symbol:
            cex_candidates.append(symbol)
        else:
            static_candidates.append(symbol)

    # Spot candidates represent tradable exchange pairs (CEX + static)
    spot_candidates = [*cex_candidates, *static_candidates]

    if not dex_candidates and not spot_candidates:
        fallback = [
            "BTC/USD",
            "ETH/USD",
            "SOL/USD",
            "ADA/USD",
            "MATIC/USD",
            "DOGE/USD",
        ]
        spot_candidates = list(fallback)
        static_candidates = list(spot_candidates)
        context.metadata["symbol_batch_fallback"] = "default_usd"

    validator_cfg = config.get("symbol_validation") or {}
    validated: list[str] = []
    validation_start = time.time()

    if spot_candidates:
        try:
            validated = await validate_symbols_production(spot_candidates, config=validator_cfg)
            validation_time = time.time() - validation_start
            pipeline_logger.log_performance("Symbol validation", validation_time)
        except Exception:  # pragma: no cover - validation failure should not stop trading
            logger.exception("Symbol validation failed; using unfiltered spot batch")
            validated = [symbol for symbol in spot_candidates if _looks_like_symbol(symbol)]

        if validated:
            valid_set = set(validated)
            invalid = [symbol for symbol in spot_candidates if symbol not in valid_set]
            for symbol in invalid[:5]:  # Log only first 5 invalid symbols
                logger.warning("Filtered invalid or unsupported symbol from batch: %s", symbol)

            if not validated:
                logger.warning(
                    "Symbol validation removed all spot candidates; falling back to format-checked set"
                )
                validated = [symbol for symbol in spot_candidates if _looks_like_symbol(symbol)]

    if validated:
        valid_set = set(validated)
        cex_candidates = [symbol for symbol in cex_candidates if symbol in valid_set]
        static_candidates = [symbol for symbol in static_candidates if symbol in valid_set]
        spot_candidates = validated

    # Store separate batches in context for separate processing
    context.dex_batch = dex_candidates
    context.cex_batch = cex_candidates
    context.static_batch = static_candidates

    logger.info(
        f"🎯 Market discovery complete: {len(dex_candidates)} DEX, "
        f"{len(cex_candidates)} CEX, and {len(static_candidates)} static symbols"
    )

    # Apply randomization before combining to ensure diversity
    import random

    all_candidates = dex_candidates + cex_candidates + static_candidates
    if all_candidates:
        # Use cycle ID + timestamp for deterministic but varying randomization
        cycle_seed = hash(f"{context.metadata.get('cycle_id') or context.state.get('cycles', 0)}_{int(time.time() // 60)}") % 1000000
        random.seed(cycle_seed)

        # Shuffle all candidates to ensure different selection each cycle
        random.shuffle(all_candidates)

        # Re-separate after shuffling while preserving original categorization
        dex_candidates = [s for s in all_candidates if s in dex_tokens or s.split('/')[1].upper() == "USDC"]
        cex_candidates = [s for s in all_candidates if s in cex_tokens or s.split('/')[1].upper() in ("USD", "EUR", "GBP")]
        static_candidates = [s for s in all_candidates if s in static_symbols]

    combined = _unique_symbols([*dex_candidates, *cex_candidates, *static_candidates])
    batch_size = int(config.get("symbol_batch_size") or len(combined) or 0)
    if batch_size > 0:
        combined = combined[:batch_size]

    context.current_batch = combined
    context.metadata["symbol_batch"] = combined

    # End discovery phase
    pipeline_logger.end_phase("discovery")


async def load_market_data(context: Any) -> None:
    """Populate OHLCV and regime caches for all symbol batches."""

    # Start market data phase
    pipeline_logger.start_phase("market_data")

    config = getattr(context, "config", {}) or {}

    def _filter_spot_symbols(symbols: Iterable[str]) -> list[str]:
        return [symbol for symbol in symbols if _looks_like_symbol(symbol)]

    cex_symbols = list(getattr(context, "cex_batch", []) or [])
    static_symbols = list(getattr(context, "static_batch", []) or [])
    current_batch = list(getattr(context, "current_batch", []) or [])

    spot_candidates = list(dict.fromkeys(_filter_spot_symbols([*cex_symbols, *static_symbols])))

    logger.info(f"📊 Market data phase: {len(cex_symbols)} CEX, {len(static_symbols)} static, {len(current_batch)} current batch")
    logger.info(f"🎯 Initial spot_candidates: {len(spot_candidates)} symbols")

    if not spot_candidates:
        spot_candidates = list(dict.fromkeys(_filter_spot_symbols(current_batch)))
        logger.info(f"🔄 Fallback to current_batch: {len(spot_candidates)} symbols")

    if not spot_candidates:
        logger.debug("No symbols selected for market data update")
        pipeline_logger.end_phase("market_data")
        return

    max_symbols = int(config.get("symbol_batch_size", 10))

    # Enhanced priority system with market-aware scoring
    priority_symbols = [
        "BTC/USD",   # Always include major cryptos
        "ETH/USD",
        "SOL/USD",
        "ADA/USD",
        "DOT/USD",
        "LINK/USD",
        "UNI/USD",
        "AAVE/USD",
        "AVAX/USD",
        "MATIC/USD",
    ]

    prioritized_symbols: list[str] = []
    remaining_symbols: list[str] = []

    # Get services from context
    services = getattr(context, "services", None)

    # Add priority symbols first
    for symbol in priority_symbols:
        if symbol in spot_candidates:
            prioritized_symbols.append(symbol)

    # Collect remaining symbols for scoring with randomization
    all_available = list(dict.fromkeys(current_batch + spot_candidates))
    import random

    # Use cycle ID + timestamp for deterministic but varying randomization
    cycle_id = context.metadata.get("cycle_id") or context.metadata.get("cycle_sequence", 0)
    cycle_seed = hash(f"{cycle_id}_{int(time.time() // 60)}") % 1000000
    random.seed(cycle_seed)

    # Shuffle to ensure different ordering each cycle
    shuffled_available = all_available.copy()
    random.shuffle(shuffled_available)

    for symbol in shuffled_available:
        if (_looks_like_symbol(symbol) and
            symbol not in prioritized_symbols):
            remaining_symbols.append(symbol)

    # Score and prioritize remaining symbols based on strategy
    logger.info(f"🔄 Batching {len(remaining_symbols)} remaining symbols from {len(current_batch)} total")
    if remaining_symbols:
        batching_strategy = config.get("symbol_batching_strategy", "smart_scoring")
        logger.info(f"🎯 Using batching strategy: {batching_strategy}")

        if batching_strategy == "alphabetical":
            # Simple alphabetical sorting
            prioritized_symbols.extend(sorted(remaining_symbols))
        elif batching_strategy == "random":
            # Random ordering for diversity
            import random
            random_symbols = remaining_symbols.copy()
            random.shuffle(random_symbols)
            prioritized_symbols.extend(random_symbols)
        elif batching_strategy == "smart_scoring":
            # Smart scoring based on market metrics
            scored_symbols = await _score_symbols_for_batching(
                remaining_symbols, config, services
            )
            prioritized_symbols.extend(scored_symbols)
        elif batching_strategy == "rotating_priority":
            # Rotating priority: mix of smart scoring + forced diversity
            scored_symbols = await _score_symbols_for_batching(
                remaining_symbols, config, services
            )
            # Take top 20 from scoring, then add diversity from the rest
            top_scored = scored_symbols[:20]
            remaining_pool = scored_symbols[20:]

            import random
            # Add 10 randomly selected symbols for diversity
            diversity_count = min(10, len(remaining_pool))
            if diversity_count > 0:
                diversity_symbols = random.sample(remaining_pool, diversity_count)
                top_scored.extend(diversity_symbols)

            prioritized_symbols.extend(top_scored)
        else:
            # Default to rotating priority for better diversity
            scored_symbols = await _score_symbols_for_batching(
                remaining_symbols, config, services
            )
            # Take top 20 from scoring, then add diversity from the rest
            top_scored = scored_symbols[:20]
            remaining_pool = scored_symbols[20:]

            import random
            # Add 10 randomly selected symbols for diversity
            diversity_count = min(10, len(remaining_pool))
            if diversity_count > 0:
                diversity_symbols = random.sample(remaining_pool, diversity_count)
                top_scored.extend(diversity_symbols)

            prioritized_symbols.extend(top_scored)

    all_symbols = prioritized_symbols[:max_symbols]

    # Log batching strategy and sample of selected symbols
    batching_strategy = config.get("symbol_batching_strategy", "rotating_priority")
    logger.info(
        f"📊 Loading market data for {len(all_symbols)} symbols using '{batching_strategy}' strategy "
        f"(DEX: {len(getattr(context, 'dex_batch', []) or [])} skipped)"
    )

    # Log first few symbols to verify diversity
    if all_symbols:
        sample_symbols = all_symbols[:5] + (["..."] if len(all_symbols) > 5 else [])
        logger.info(f"🎯 Selected symbols sample: {', '.join(sample_symbols)}")
        logger.info(f"🔄 Total available: {len(all_available)} symbols, prioritized: {len(prioritized_symbols)}")

    services = getattr(context, "services", None)
    market_data = getattr(services, "market_data", None)
    if market_data is None:
        logger.debug("Market data service unavailable; skipping cache refresh")
        pipeline_logger.end_phase("market_data")
        return

    exchange_id = str(config.get("exchange", "kraken"))
    market_cfg = config.get("market_data", {})
    primary_timeframe = market_cfg.get("timeframe", config.get("timeframe", "1h"))
    extra_timeframes = market_cfg.get("timeframes") or config.get("additional_timeframes", [])
    limit = int(market_cfg.get("limit", config.get("ohlcv_limit", 250)))

    market_data_start = time.time()

    request = MultiTimeframeOHLCVRequest(
        exchange_id=exchange_id,
        cache=context.df_cache,
        symbols=all_symbols,
        config={
            "timeframe": primary_timeframe,
            "timeframes": [primary_timeframe, *extra_timeframes],
        },
        limit=limit,
        use_websocket=bool(config.get("use_websocket")),
        force_websocket_history=bool(config.get("force_websocket_history")),
        max_concurrent=config.get("max_concurrent_ohlcv"),
        notifier=getattr(context, "notifier", None),
    )

    try:
        logger.info(f"Fetching OHLCV data for {len(all_symbols)} symbols on {exchange_id}")
        response: CacheUpdateResponse = await market_data.update_multi_tf_cache(request)
        context.df_cache = response.cache
        symbols_loaded = len(response.cache.get(primary_timeframe, {}))
        load_time = time.time() - market_data_start
        pipeline_logger.log_performance("OHLCV data loading", load_time)
        logger.info(f"✅ Successfully loaded market data for {symbols_loaded} symbols in {load_time:.2f}s")
    except Exception as exc:
        logger.warning("Failed to update OHLCV cache: %s", exc)
        pipeline_logger.log_error("OHLCV cache update failed", phase="market_data")
        # Fallback: try with just the first 10 symbols to get some data
        if len(all_symbols) > 10:
            logger.info("Attempting fallback with first 10 symbols")
            fallback_request = MultiTimeframeOHLCVRequest(
                exchange_id=exchange_id,
                cache=context.df_cache,
                symbols=all_symbols[:10],
                config={
                    "timeframe": primary_timeframe,
                    "timeframes": [primary_timeframe, *extra_timeframes],
                },
                limit=limit,
                use_websocket=bool(config.get("use_websocket")),
                force_websocket_history=bool(config.get("force_websocket_history")),
                max_concurrent=config.get("max_concurrent_ohlcv"),
                notifier=getattr(context, "notifier", None),
            )
            try:
                response: CacheUpdateResponse = await market_data.update_multi_tf_cache(fallback_request)
                context.df_cache = response.cache
                symbols_loaded = len(response.cache.get(primary_timeframe, {}))
                logger.info(f"Fallback: Successfully loaded market data for {symbols_loaded} symbols")
            except Exception:
                logger.warning("Fallback market data request also failed")
                # Ensure df_cache exists even if empty
                if not hasattr(context, 'df_cache'):
                    context.df_cache = {}

    regime_start = time.time()
    regime_timeframes = config.get("regime_timeframes")
    if regime_timeframes:
        regime_request = RegimeCacheRequest(
            exchange_id=exchange_id,
            cache=context.regime_cache,
            symbols=all_symbols,
            config={"timeframes": list(regime_timeframes)},
            limit=limit,
            use_websocket=bool(config.get("use_websocket")),
            force_websocket_history=bool(config.get("force_websocket_history")),
            max_concurrent=config.get("max_concurrent_ohlcv"),
            notifier=getattr(context, "notifier", None),
            df_map=context.df_cache,
        )
        try:
            regime_response: CacheUpdateResponse = await market_data.update_regime_cache(regime_request)
            context.regime_cache = regime_response.cache
            regime_time = time.time() - regime_start
            pipeline_logger.log_performance("Regime analysis", regime_time)
            logger.info(f"✅ Regime analysis completed for {len(all_symbols)} symbols in {regime_time:.2f}s")
        except Exception:  # pragma: no cover - advisory
            logger.debug("Regime cache update failed", exc_info=True)

    context.metadata["market_data_updated"] = len(all_symbols)
    context.metadata["market_data_symbol_sources"] = {
        "cex": len(cex_symbols),
        "static": len(static_symbols),
        "dex_skipped": len(getattr(context, "dex_batch", []) or []),
    }

    # End market data phase
    pipeline_logger.end_phase("market_data")


async def evaluate_batch_signals(context: Any, symbol_batch: list[str], batch_type: str = "mixed") -> list[StrategyEvaluationResult]:
    """Evaluate signals for a specific batch of symbols."""

    services = getattr(context, "services", None)
    strategy_service = getattr(services, "strategy", None)
    if strategy_service is None:
        logger.debug(f"Strategy service unavailable; no {batch_type} signal evaluation performed")
        return []

    config = getattr(context, "config", {}) or {}
    default_regime = config.get("default_regime", "trending")  # Use trending regime with working strategies
    payloads: list[StrategyEvaluationPayload] = []

    for symbol in symbol_batch:
        df = _select_dataframe(context.df_cache, symbol)
        if df is None:
            continue
        regime = _infer_regime(context.regime_cache, symbol, default_regime)
        timeframes: MutableMapping[str, pd.DataFrame] = {}
        for timeframe, entries in context.df_cache.items():
            frame = entries.get(symbol)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                timeframes[timeframe] = frame
        if not timeframes:
            continue
        payloads.append(
            StrategyEvaluationPayload(
                symbol=symbol,
                regime=regime,
                mode=str(config.get("mode", "auto")),
                timeframes=timeframes,
                config=config,
                metadata={
                    "cycle": context.metadata.get("cycle_sequence"),
                    "batch_type": batch_type
                },
            )
        )

    if not payloads:
        logger.debug(f"No valid {batch_type} symbols for evaluation")
        return []

    request = StrategyBatchRequest(items=tuple(payloads), metadata={"cycle": context.metadata.get("cycle_sequence"), "batch_type": batch_type})
    try:
        response: StrategyBatchResponse = await strategy_service.evaluate_batch(request)
        results = list(response.results or [])
        if response.errors:
            logger.warning(f"{batch_type} evaluation errors: {response.errors}")
        return results
    except Exception:
        logger.exception(f"{batch_type} strategy evaluation failed")
        return []


async def evaluate_signals(context: Any) -> None:
    """Invoke the strategy service to generate trading signals for all batches."""

    # Start evaluation phase
    pipeline_logger.start_phase("evaluation")

    evaluation_start = time.time()

    # Evaluate DEX tokens separately
    dex_results = []
    if hasattr(context, 'dex_batch') and context.dex_batch:
        logger.info(f"🧠 Evaluating {len(context.dex_batch)} DEX tokens")
        dex_results = await evaluate_batch_signals(context, context.dex_batch, "dex")
        context.metadata["dex_analysis_results"] = len(dex_results)

    # Evaluate CEX tokens separately
    cex_results = []
    if hasattr(context, 'cex_batch') and context.cex_batch:
        logger.info(f"🧠 Evaluating {len(context.cex_batch)} CEX tokens")
        cex_results = await evaluate_batch_signals(context, context.cex_batch, "cex")
        context.metadata["cex_analysis_results"] = len(cex_results)

    # Evaluate static/exchange tokens
    static_results = []
    if hasattr(context, 'static_batch') and context.static_batch:
        logger.info(f"🧠 Evaluating {len(context.static_batch)} static/exchange tokens")
        static_results = await evaluate_batch_signals(context, context.static_batch, "static")
        context.metadata["static_analysis_results"] = len(static_results)

    # Combine all results
    all_results = dex_results + cex_results + static_results
    context.analysis_results = all_results
    context.metadata["analysis_results"] = len(all_results)

    evaluation_time = time.time() - evaluation_start
    pipeline_logger.log_performance("Strategy evaluation", evaluation_time)

    # Log individual evaluation results
    for result in all_results[:20]:  # Log first 20 to avoid spam
        direction = (result.direction or "").lower()
        if direction in ["long", "short"]:
            pipeline_logger.log_evaluation(
                result.symbol,
                result.strategy or "unknown",
                float(result.score or 0.0),
                direction
            )

    logger.info(
        f"🎯 Evaluation complete: {len(all_results)} signals generated "
        f"from {len(dex_results) + len(cex_results) + len(static_results)} symbols in {evaluation_time:.2f}s"
    )

    # End evaluation phase
    pipeline_logger.end_phase("evaluation")


async def execute_signals(context: Any) -> None:
    """Run risk checks and dispatch executable orders."""

    # Start execution phase
    pipeline_logger.start_phase("execution")

    execution_start = time.time()

    results: Iterable[StrategyEvaluationResult] = getattr(context, "analysis_results", []) or []
    if not results:
        logger.info("🚫 No analysis results to execute")
        context.metadata["executed_trades"] = []
        pipeline_logger.end_phase("execution")
        return

    services = getattr(context, "services", None)
    execution = getattr(services, "execution", None)
    if execution is None:
        logger.debug("Execution service unavailable; no trades placed")
        context.metadata["executed_trades"] = []
        pipeline_logger.end_phase("execution")
        return

    config = getattr(context, "config", {}) or {}
    risk_manager = getattr(context, "risk_manager", None)
    position_guard = getattr(context, "position_guard", None)
    balance = float(getattr(context, "balance", 0.0))
    execution_mode = str(config.get("execution_mode", "dry_run"))
    dry_run = execution_mode.lower() != "live"
    
    # Initialize default balance for dry run mode (like legacy system)
    if dry_run and balance <= 0.0:
        balance = 10000.0  # Default testing balance like legacy system
        logger.info(f"Initialized dry run balance to ${balance:.2f}")
        context.balance = balance
    
    executed: list[dict[str, object]] = []

    logger.info(f"Processing {len(results)} analysis results for trade execution")
    
    for i, result in enumerate(results):
        direction = (result.direction or "").lower()
        signal_score = float(result.score or 0.0)
        signal_strategy = result.strategy or ""

        logger.info(
            f"Result {i+1}: {result.symbol} direction='{direction}' score={signal_score} strategy='{signal_strategy}'"
        )

        if direction not in {"long", "short"}:
            logger.info(f"Primary direction invalid for {result.symbol}: '{direction}'")
            fallback = (result.fused_direction or "").lower()
            if fallback in {"long", "short"}:
                direction = fallback
                fused_score = float(result.fused_score or signal_score or 0.0)
                if fused_score > 0:
                    signal_score = fused_score
                if not signal_strategy:
                    signal_strategy = "fused_signal"
                logger.info(
                    f"Using fused direction '{direction}' for {result.symbol}"
                )
            else:
                for ranked_signal in result.ranked_signals:
                    ranked_direction = (ranked_signal.direction or "").lower()
                    if ranked_direction in {"long", "short"}:
                        direction = ranked_direction
                        ranked_score = float(ranked_signal.score or signal_score or 0.0)
                        if ranked_score > 0:
                            signal_score = ranked_score
                        if ranked_signal.strategy:
                            signal_strategy = ranked_signal.strategy
                        logger.info(
                            f"Using ranked signal direction '{direction}' for {result.symbol}"
                        )
                        break

        if direction not in {"long", "short"}:
            logger.info(f"Skipping {result.symbol}: no actionable direction after fallbacks")
            continue

        if signal_score <= 0 and result.ranked_signals:
            for ranked_signal in result.ranked_signals:
                if (ranked_signal.direction or "").lower() == direction:
                    ranked_score = float(ranked_signal.score or 0.0)
                    if ranked_score > 0:
                        signal_score = ranked_score
                        if ranked_signal.strategy:
                            signal_strategy = ranked_signal.strategy
                    break

        if signal_score <= 0 and result.fused_score:
            fused_score = float(result.fused_score or 0.0)
            if fused_score > 0:
                signal_score = fused_score

        signal_score = max(0.0, min(signal_score, 1.0))
        active_strategy = signal_strategy or result.strategy or ""

        df = _select_dataframe(context.df_cache, result.symbol)
        if df is None or df.empty:
            continue
        price = float(df["close"].iloc[-1])

        if position_guard is not None:
            try:
                can_open = await position_guard.can_open(getattr(context, "positions", {}))
            except Exception:  # pragma: no cover - defensive
                logger.debug("Position guard check failed for %s", result.symbol, exc_info=True)
                can_open = True
            if not can_open:
                context.metadata.setdefault("skipped_trades", []).append(
                    {"symbol": result.symbol, "reason": "position_guard"}
                )
                continue

        allowed = True
        reason = ""
        if risk_manager is not None:
            try:
                allowed, reason = await risk_manager.allow_trade(df, active_strategy or None)
            except Exception:  # pragma: no cover - best effort
                logger.debug("Risk allow_trade failed for %s", result.symbol, exc_info=True)
                allowed = True
        if not allowed:
            context.metadata.setdefault("skipped_trades", []).append({"symbol": result.symbol, "reason": reason})
            continue

        # Calculate position size
        size = 0.0
        if risk_manager is not None:
            try:
                size = float(
                    await risk_manager.position_size(
                        signal_score,
                        float(balance),
                        df=df,
                        atr=result.atr,
                        price=price,
                    )
                )
            except Exception:  # pragma: no cover - fallback sizing
                logger.debug("Risk position_size failed for %s", result.symbol, exc_info=True)
                size = float(balance) * 0.05
        else:
            # Default position sizing: 5% of balance per trade
            size = float(balance) * 0.05

        if size <= 0:
            continue

        if risk_manager is not None and hasattr(risk_manager, "can_allocate") and hasattr(
            risk_manager, "allocate_capital"
        ):
            try:
                if not await risk_manager.can_allocate(active_strategy or "", size, balance):
                    continue
                await risk_manager.allocate_capital(active_strategy or "", size)
            except Exception:  # pragma: no cover - advisory
                logger.debug("Capital allocation failed for %s", result.symbol, exc_info=True)

        amount = size / price if price else 0.0
        trade_request = TradeExecutionRequest(
            exchange=getattr(context, "exchange", None),
            ws_client=getattr(context, "ws_client", None),
            symbol=result.symbol,
            side="buy" if direction == "long" else "sell",
            amount=float(amount),
            notifier=getattr(context, "notifier", None),
            dry_run=dry_run,
            use_websocket=bool(context.config.get("use_websocket")),
            config=context.config,
            score=signal_score,
        )

        try:
            # Log trade attempt
            pipeline_logger.log_trade_attempt(
                result.symbol,
                trade_request.side,
                trade_request.amount,
                price,
                f"Strategy: {active_strategy or 'unknown'}, Score: {signal_score:.3f}"
            )

            from libs.execution import execute_trade_async
            response = await execute_trade_async(
                exchange=trade_request.exchange,
                ws_client=trade_request.ws_client,
                symbol=trade_request.symbol,
                side=trade_request.side,
                amount=trade_request.amount,
                notifier=trade_request.notifier,
                dry_run=trade_request.dry_run,
                use_websocket=trade_request.use_websocket,
                config=trade_request.config,
                score=trade_request.score,
            )

            # Check if execution failed (empty order indicates failure)
            order_data = response if isinstance(response, dict) else getattr(response, "order", {})
            logger.info(
                f"Trade execution response for {result.symbol}: order_data={order_data}, dry_run={dry_run}"
            )
            normalized_status = ""
            mapping_success = False
            list_success = isinstance(order_data, list) and len(order_data) > 0

            if isinstance(order_data, Mapping):
                status_value = order_data.get("status") or order_data.get("state")
                if isinstance(status_value, str):
                    normalized_status = status_value.lower()
                mapping_success = any(
                    [
                        normalized_status in {"filled", "closed", "completed", "done"},
                        bool(order_data.get("dry_run")),
                        bool(order_data.get("orders")),
                    ]
                )

            is_successful_order = list_success or mapping_success

            if not is_successful_order:
                # Execution failed - treat as dry run failure
                pipeline_logger.log_error(f"Trade execution failed for {result.symbol}", result.symbol, "execution")
                if dry_run:
                    mock_order = {
                        "symbol": result.symbol,
                        "side": trade_request.side,
                        "amount": trade_request.amount,
                        "price": price,
                        "size_usd": size,
                        "dry_run": True,
                        "client_order_id": f"MOCK_{result.symbol}_{trade_request.side}",
                        "status": "filled",
                        "strategy": active_strategy,
                        "score": signal_score,
                        "timestamp": context.metadata.get("cycle_started_at"),
                    }
                    executed.append(
                        {
                            "symbol": result.symbol,
                            "side": trade_request.side,
                            "amount": trade_request.amount,
                            "order": mock_order,
                            "strategy": active_strategy,
                        }
                    )
                    logger.info(f"Created mock dry run trade for {result.symbol}: {trade_request.side} {trade_request.amount} at ${price:.2f}")

                    await _record_trade_and_sync(
                        context,
                        symbol=result.symbol,
                        side=trade_request.side,
                        amount=trade_request.amount,
                        price=price,
                        order=mock_order,
                        strategy=active_strategy,
                        score=signal_score,
                        dry_run=dry_run,
                    )

                    # For dry run failures, still update paper wallet with mock trade
                    paper_wallet = getattr(context, "paper_wallet", None)
                    logger.info(f"Paper wallet for {result.symbol}: {paper_wallet is not None}")
                    if paper_wallet is not None:
                        try:
                            logger.info(f"Calling paper wallet for {result.symbol}: direction={direction}, amount={amount}, price={price}")
                            if direction == "long":
                                await paper_wallet.buy(result.symbol, amount, price)
                            else:
                                await paper_wallet.sell(result.symbol, amount, price)
                            logger.info(f"Paper wallet call completed for {result.symbol}")
                        except Exception as e:  # pragma: no cover - optional path
                            logger.error(f"Paper wallet trade failed for {result.symbol}: {e}", exc_info=True)
                else:
                    # Live execution failed
                    executed.append(
                        {
                            "symbol": result.symbol,
                            "side": trade_request.side,
                            "amount": trade_request.amount,
                            "order": order_data,
                            "strategy": active_strategy,
                        }
                    )
            else:
                # Successful execution
                executed.append(
                    {
                        "symbol": result.symbol,
                        "side": trade_request.side,
                        "amount": trade_request.amount,
                        "order": order_data,
                        "strategy": active_strategy,
                    }
                )

                # Log successful trade execution
                fill_price = _resolve_order_price(order_data, price)
                usd_value = amount * fill_price
                order_id = order_data.get("id") if isinstance(order_data, dict) else None

                pipeline_logger.log_trade_execution(
                    result.symbol,
                    trade_request.side,
                    amount,
                    fill_price,
                    order_id,
                    0.0,  # P&L calculation would be done separately
                    {"strategy": active_strategy, "score": signal_score}
                )

                # For successful execution, update paper wallet
                paper_wallet = getattr(context, "paper_wallet", None)
                if paper_wallet is not None:
                    try:
                        if direction == "long":
                            await paper_wallet.buy(result.symbol, amount, price)
                        else:
                            await paper_wallet.sell(result.symbol, amount, price)
                    except Exception:  # pragma: no cover - optional path
                        logger.debug("Paper wallet trade failed for %s", result.symbol, exc_info=True)

                await _record_trade_and_sync(
                    context,
                    symbol=result.symbol,
                    side=trade_request.side,
                    amount=trade_request.amount,
                    price=fill_price,
                    order=order_data,
                    strategy=active_strategy,
                    score=signal_score,
                    dry_run=dry_run,
                )
        except Exception as exc:
            logger.warning("Trade execution failed for %s: %s", result.symbol, exc)
            # In dry run mode, create mock trade like legacy system
            if dry_run:
                mock_order = {
                    "symbol": result.symbol,
                    "side": trade_request.side,
                    "amount": trade_request.amount,
                    "price": price,
                    "size_usd": size,
                    "dry_run": True,
                    "client_order_id": f"MOCK_{result.symbol}_{trade_request.side}",
                    "status": "filled",
                    "strategy": active_strategy,
                    "score": signal_score,
                    "timestamp": context.metadata.get("cycle_started_at"),
                }
                executed.append(
                    {
                        "symbol": result.symbol,
                        "side": trade_request.side,
                        "amount": trade_request.amount,
                        "order": mock_order,
                        "strategy": active_strategy,
                    }
                )
                logger.info(f"Created mock dry run trade for {result.symbol}: {trade_request.side} {trade_request.amount} at ${price:.2f}")

                # For dry run failures, still update paper wallet with mock trade
                paper_wallet = getattr(context, "paper_wallet", None)
                if paper_wallet is not None:
                    try:
                        if direction == "long":
                            await paper_wallet.buy(result.symbol, amount, price)
                        else:
                            await paper_wallet.sell(result.symbol, amount, price)
                    except Exception:  # pragma: no cover - optional path
                        logger.debug("Paper wallet trade failed for %s", result.symbol, exc_info=True)
                await _record_trade_and_sync(
                    context,
                    symbol=result.symbol,
                    side=trade_request.side,
                    amount=trade_request.amount,
                    price=price,
                    order=mock_order,
                    strategy=active_strategy,
                    score=signal_score,
                    dry_run=dry_run,
                )
            else:
                continue
        balance = max(balance - size, 0.0)
        context.balance = balance

    context.metadata["executed_trades"] = executed
    context.metadata["executed_trade_count"] = len(executed)

    execution_time = time.time() - execution_start
    pipeline_logger.log_performance("Trade execution", execution_time)

    logger.info(
        f"💰 Execution complete: {len(executed)} trades processed "
        f"from {len(results)} signals in {execution_time:.2f}s"
    )

    # End execution phase
    pipeline_logger.end_phase("execution")


async def finalize_cycle(context: Any) -> None:
    """Persist summary metadata and refresh derived state."""

    # Start finalization phase
    pipeline_logger.start_phase("finalization")

    context.metadata["cycle_completed_at"] = datetime.now(timezone.utc).isoformat()
    context.metadata["balance"] = float(getattr(context, "balance", 0.0))

    if getattr(context, "trade_manager", None) is not None:
        try:
            context.sync_positions_from_trade_manager()
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Post-cycle position sync failed", exc_info=True)

    # Handle positions safely in finalize_cycle too
    positions_attr = getattr(context, "positions", {})
    if asyncio.iscoroutine(positions_attr):
        positions = await positions_attr
    else:
        positions = positions_attr
    if not isinstance(positions, dict):
        positions = {}

    # Calculate final portfolio metrics
    final_positions = sorted(positions.keys())
    final_portfolio_value = sum(
        float(pos.get("amount", 0) * pos.get("entry_price", 0))
        for pos in positions.values() if isinstance(pos, dict)
    )

    context.metadata["open_positions"] = final_positions
    context.metadata["final_portfolio_value"] = final_portfolio_value

    # Calculate P&L for the cycle
    initial_value = context.metadata.get("portfolio_value", 0.0)
    cycle_pnl = final_portfolio_value - initial_value

    logger.info(
        f"📋 Cycle #{context.metadata.get('cycle_id', 'unknown')} finalized: "
        f"{len(final_positions)} positions, "
        f"Portfolio: ${final_portfolio_value:.2f} "
        f"(P&L: ${cycle_pnl:+.2f})"
    )

    # Complete pipeline with final metrics
    pipeline_logger.complete_pipeline(success=True, final_pnl=cycle_pnl)

    # End finalization phase
    pipeline_logger.end_phase("finalization")


DEFAULT_PHASES = [
    prepare_cycle,
    discover_markets,
    load_market_data,
    evaluate_signals,
    execute_signals,
    finalize_cycle,
]


__all__ = [
    "prepare_cycle",
    "discover_markets",
    "load_market_data",
    "evaluate_signals",
    "execute_signals",
    "finalize_cycle",
    "DEFAULT_PHASES",
]
