import os
import asyncio
import contextlib
import time
import warnings
from pathlib import Path
from datetime import datetime
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
import inspect
import random
# Suppress urllib3 OpenSSL warning
warnings.filterwarnings(
    'ignore',
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    category=UserWarning
)
import aiohttp
import pandas as pd
import numpy as np
import yaml
from dotenv import dotenv_values
from decimal import Decimal
import sys
import gc
# Track WebSocket ping tasks
WS_PING_TASKS: set[asyncio.Task] = set()
try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types
    ccxt = types.SimpleNamespace()
# Add the project root to the Python path for schema imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# All imports consolidated at the top
from crypto_bot.utils.telegram import TelegramNotifier, send_test_message
from crypto_bot.utils.logger import (
    LOG_DIR,
    setup_logger
)
from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.wallet_manager import load_or_create
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.cooldown_manager import (
    configure as cooldown_configure,
)
from crypto_bot.phase_runner import BotContext, PhaseRunner
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.risk.exit_manager import (
    should_exit,
)
from crypto_bot.open_position_guard import OpenPositionGuard
from crypto_bot import console_monitor, console_control
from crypto_bot.utils.position_logger import log_position, log_balance
from crypto_bot.utils.market_loader import (
    configure as market_loader_configure,
)
from crypto_bot.utils.eval_queue import build_priority_queue
from crypto_bot.utils.symbol_utils import fix_symbol
from crypto_bot.utils.metrics_logger import log_cycle as log_cycle_metrics
from crypto_bot.utils.pnl_logger import log_pnl
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.auto_optimizer import optimize_strategies
from crypto_bot.utils.telemetry import write_cycle_metrics
from crypto_bot.balance_management import (
    ensure_paper_wallet_sync,
    fetch_balance,
    get_paper_wallet_status,
    notify_balance_change,
    sync_paper_wallet_balance,
    sync_paper_wallet_with_positions_log,
    update_position_pnl,
)
from crypto_bot.enhanced_scan_integration import (
    get_enhanced_scan_integration,
    start_enhanced_scan_integration,
    stop_enhanced_scan_integration,
)
from crypto_bot.evaluation_pipeline_integration import (
    get_evaluation_pipeline_integration,
    initialize_evaluation_pipeline,
    get_tokens_for_evaluation,
)
from crypto_bot.position_monitor import PositionMonitor
from crypto_bot.fund_manager import (
    auto_convert_funds,
    check_wallet_balances,
    detect_non_trade_tokens,
)
from crypto_bot.regime.regime_classifier import classify_regime_async
from crypto_bot.volatility_filter import calc_atr
from crypto_bot.solana.exit import monitor_price
from crypto_bot.utils.memory_manager import get_memory_manager
from crypto_bot.runtime_signals import (
    check_existing_instance,
    cleanup_pid_file,
    install_signal_handlers,
    write_pid_file,
)
from crypto_bot.startup_utils import (
    CONFIG_PATH,
    create_service_container,
    flatten_config,
    load_config,
    maybe_reload_config,
    reload_config,
    set_last_config_mtime,
)
from crypto_bot.services.interfaces import (
    CreateTradeRequest,
    ExchangeRequest,
    LoadSymbolsRequest,
    MultiTimeframeOHLCVRequest,
    OHLCVCacheRequest,
    OrderBookRequest,
    RecordScannerMetricsRequest,
    RegimeCacheRequest,
    ServiceContainer,
    TimeframeRequest,
    TokenDiscoveryRequest,
    TradeExecutionRequest,
)
# Backwards compatibility for tests - function defined later
# Log monitoring setup will be done after logger is initialized
# Memory management utilities
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
class MemoryManager:
    """Memory management utilities for the trading bot."""
    def __init__(self, memory_threshold: float = 0.8):
        """
        Initialize the memory manager.
        Args:
            memory_threshold: Memory usage threshold (0.0-1.0) above which optimization is triggered
        """
        self.memory_threshold = memory_threshold
        self.optimization_count = 0
        self.last_optimization = time.time()
    def check_memory_pressure(self) -> bool:
        """
        Check if memory usage is above the threshold.
        Returns:
            True if memory pressure is high, False otherwise
        """
        if not PSUTIL_AVAILABLE:
            return False
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0 > self.memory_threshold
        except Exception:
            return False
    def optimize_cache_sizes(self, df_cache: dict, regime_cache: dict) -> None:
        """
        Optimize cache sizes to reduce memory usage.
        Args:
            df_cache: DataFrame cache dictionary
            regime_cache: Regime cache dictionary
        """
        if not self.check_memory_pressure():
            return
        self.optimization_count += 1
        self.last_optimization = time.time()
        # Reduce cache sizes by 20%
        for tf, cache in df_cache.items():
            if hasattr(cache, 'maxlen') and cache.maxlen > 100:
                new_maxlen = max(100, int(cache.maxlen * 0.8))
                # Create new deque with reduced maxlen
                new_cache = deque(list(cache)[-new_maxlen:], maxlen=new_maxlen)
                df_cache[tf] = new_cache
        for tf, cache in regime_cache.items():
            if hasattr(cache, 'maxlen') and cache.maxlen > 100:
                new_maxlen = max(100, int(cache.maxlen * 0.8))
                new_cache = deque(list(cache)[-new_maxlen:], maxlen=new_maxlen)
                regime_cache[tf] = new_cache
    def force_garbage_collection(self) -> int:
        """
        Force garbage collection and return the number of collected objects.
        Returns:
            Number of objects collected
        """
        collected = gc.collect()
        return collected
    def get_memory_stats(self) -> dict:
        """
        Get current memory statistics.
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "optimization_count": self.optimization_count,
            "last_optimization": self.last_optimization
        }
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                stats.update({
                    "total_mb": memory.total // (1024 * 1024),
                    "available_mb": memory.available // (1024 * 1024),
                    "used_mb": memory.used // (1024 * 1024),
                    "percent": memory.percent
                })
            except Exception:
                pass
        else:
            stats.update({
                "total_mb": 0,
                "available_mb": 0,
                "used_mb": 0,
                "percent": 0
            })
        return stats
def _fix_symbol(sym: str) -> str:
    """Internal wrapper for tests to normalize symbols."""
    return fix_symbol(sym)
ENV_PATH = Path(__file__).resolve().parent / ".env"
logger = setup_logger("bot", LOG_DIR / "bot.log", to_console=False)
# Start continuous log monitoring in background (after logger is initialized)
try:
    from tools.log_monitor import LogMonitor
    import threading
    def start_log_monitoring():
        """Start continuous log monitoring in a background thread."""
        try:
            monitor = LogMonitor(repo_root=Path(__file__).resolve().parent.parent, flush_interval=60)
            monitor.run()
        except Exception as e:
            logger.warning(f"Failed to start log monitoring: {e}")
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=start_log_monitoring, daemon=True, name="LogMonitor")
    monitor_thread.start()
    logger.info("Continuous log monitoring started in background")
except ImportError as e:
    logger.warning(f"Could not start log monitoring: {e}")
# Queue of symbols awaiting evaluation across loops
symbol_priority_queue: deque[str] = deque()
# Queue tracking symbols evaluated across cycles
SYMBOL_EVAL_QUEUE: deque[str] = deque()
# Protects shared queues for future multi-tasking scenarios
QUEUE_LOCK = asyncio.Lock()
# Retry parameters for the initial symbol scan
MAX_SYMBOL_SCAN_ATTEMPTS = 3
SYMBOL_SCAN_RETRY_DELAY = 10
MAX_SYMBOL_SCAN_DELAY = 60
# Maximum number of symbols per timeframe to keep in the OHLCV cache
DF_CACHE_MAX_SIZE = 500
# Track regime analysis statistics
UNKNOWN_COUNT = 0
TOTAL_ANALYSES = 0
@dataclass
class SessionState:
    """Runtime session state shared across tasks."""
    positions: dict[str, dict] = field(default_factory=dict)
    df_cache: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    regime_cache: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    last_balance: Union[float, None] = None
    scan_task: Union[asyncio.Task, None] = None
def update_df_cache(
    cache: dict[str, dict[str, pd.DataFrame]],
    timeframe: str,
    symbol: str,
    df: pd.DataFrame,
    max_size: int = DF_CACHE_MAX_SIZE,
) -> None:
    """Update an OHLCV cache with LRU eviction."""
    tf_cache = cache.setdefault(timeframe, OrderedDict())
    if not isinstance(tf_cache, OrderedDict):
        tf_cache = OrderedDict(tf_cache)
        cache[timeframe] = tf_cache
    tf_cache[symbol] = df
    tf_cache.move_to_end(symbol)
    if len(tf_cache) > max_size:
        tf_cache.popitem(last=False)
def compute_average_atr(symbols: list[str], df_cache: dict, timeframe: str) -> float:
    """Return the average ATR for symbols present in ``df_cache``."""
    atr_values: list[float] = []
    tf_cache = df_cache.get(timeframe, {})
    for sym in symbols:
        df = tf_cache.get(sym)
        if df is None or df.empty:
            continue
        atr_values.append(calc_atr(df))
    return sum(atr_values) / len(atr_values) if atr_values else 0.0


def _timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe strings like ``1h`` or ``15m`` to seconds."""

    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    if unit == "w":
        return value * 604800
    if unit == "M":
        return value * 2592000
    raise ValueError(f"Unknown timeframe {timeframe}")


def is_market_pumping(
    symbols: list[str], df_cache: dict, timeframe: str = "1h", lookback_hours: int = 24
) -> bool:
    """Return ``True`` when the average % change over ``lookback_hours`` exceeds ~10%."""
    tf_cache = df_cache.get(timeframe, {})
    if not tf_cache:
        return False
    try:
        sec = _timeframe_to_seconds(timeframe)
    except Exception:
        return False
    candles = int(lookback_hours * 3600 / sec) if sec else 0
    changes: list[float] = []
    for sym in symbols:
        df = tf_cache.get(sym)
        if df is None or df.empty or "close" not in df:
            continue
        closes = df["close"]
        if len(closes) == 0:
            continue
        start_idx = -candles - 1 if candles and len(closes) > candles else 0
        try:
            start = float(closes[start_idx])
            end = float(closes[-1])
        except Exception:
            continue
        if start == 0:
            continue
        changes.append((end - start) / start)
    avg_change = sum(changes) / len(changes) if changes else 0.0
    return avg_change >= 0.10
def direction_to_side(direction: str) -> str:
    """Translate strategy direction to trade side."""
    return "buy" if direction == "long" else "sell"
def opposite_side(side: str) -> str:
    """Return the opposite trading side."""
    return "sell" if side == "buy" else "buy"
def _closest_wall_distance(book: dict, entry: float, side: str) -> Optional[float]:
    """Return distance to the nearest bid/ask wall from ``entry``."""
    if not isinstance(book, dict):
        return None
    levels = book.get("asks") if side == "buy" else book.get("bids")
    if not levels:
        return None
    dists = []
    for price, _amount in levels:
        if side == "buy" and price > entry:
            dists.append(price - entry)
        elif side == "sell" and price < entry:
            dists.append(entry - price)
    if not dists:
        return None
    return min(dists)
def _emit_timing(
    symbol_t: float,
    ohlcv_t: float,
    analyze_t: float,
    total_t: float,
    metrics_path: Union[Path, None] = None,
    ohlcv_fetch_latency: float = 0.0,
    execution_latency: float = 0.0,
) -> None:
    """Log timing information and optionally append to metrics CSV."""
    logger.info(
        "\u23f1\ufe0f Cycle timing - Symbols: %.2fs, OHLCV: %.2fs, Analyze: %.2fs, Total: %.2fs",
        symbol_t,
        ohlcv_t,
        analyze_t,
        total_t,
    )
    if metrics_path:
        log_cycle_metrics(
            symbol_t,
            ohlcv_t,
            analyze_t,
            total_t,
            ohlcv_fetch_latency,
            execution_latency,
            metrics_path,
        )
async def _ws_ping_loop(exchange: object, interval: float) -> None:
    """Periodically send WebSocket ping messages."""
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                ping = getattr(exchange, "ping", None)
                if ping is None:
                    continue
                is_coro = asyncio.iscoroutinefunction(ping)
                clients = getattr(exchange, "clients", None)
                if isinstance(clients, dict):
                    if clients:
                        for client in clients.values():
                            if is_coro:
                                await ping(client)
                            else:
                                await asyncio.to_thread(ping, client)
                    else:
                        continue
                else:
                    if is_coro:
                        await ping()
                    else:
                        await asyncio.to_thread(ping)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - ping failures
                logger.error("WebSocket ping failed: %s", exc, exc_info=True)
    except asyncio.CancelledError:
        pass
async def initial_scan(
    exchange: object,
    config: dict,
    state: SessionState,
    services: ServiceContainer,
    notifier: Union[TelegramNotifier, None] = None,
) -> None:
    """Populate OHLCV and regime caches before trading begins."""
    # Build an exchange-aware, pre-filtered symbol list
    raw_symbols = config.get("symbols") or [config.get("symbol")]
    if not raw_symbols:
        return
    # Use the same filtering pathway as runtime to avoid unsupported pairs
    try:
        from crypto_bot.utils.symbol_utils import get_filtered_symbols
        scored = await get_filtered_symbols(exchange, config)
        symbols = [s for s, _ in scored]
    except Exception:
        symbols = [s for s in raw_symbols if s]
    exchange_id = getattr(exchange, "id", "") or str(config.get("exchange", "kraken"))
    batch_size = int(config.get("symbol_batch_size", 10))
    total = len(symbols)
    processed = 0
    # Populate OHLCV and regime caches before first loop
    if not state.df_cache:
        state.df_cache = {}
    if not state.regime_cache:
        state.regime_cache = {}
    logger.info("Initial scan starting: warming OHLCV/regime caches")
    # Respect configured timeframes, chunk symbols to batches
    tfs = config.get("timeframes", [config.get("timeframe", "15m")])
    # Simple batching to avoid timeouts
    batch_size = max(1, int(config.get("symbol_batch_size", 10)))
    for i in range(0, total, batch_size):
        batch = symbols[i:i+batch_size]
        loader_cfg = {**config, "timeframes": tfs}
        try:
            df_response = await services.market_data.update_multi_tf_cache(
                MultiTimeframeOHLCVRequest(
                    exchange_id=str(exchange_id),
                    cache=state.df_cache,
                    symbols=batch,
                    config=loader_cfg,
                )
            )
            state.df_cache = df_response.cache
            regime_response = await services.market_data.update_regime_cache(
                RegimeCacheRequest(
                    exchange_id=str(exchange_id),
                    cache=state.regime_cache,
                    symbols=batch,
                    config=config,
                )
            )
            state.regime_cache = regime_response.cache
        except Exception as exc:
            logger.error("Initial scan batch failed: %s", exc)
        processed += len(batch)
        logger.info("Initial scan progress: %d/%d symbols cached", processed, total)
    logger.info("Initial scan completed: OHLCV/regime caches ready")
    if notifier and config.get("telegram", {}).get("status_updates", True) and notifier.enabled:
        notifier.notify("Initial scan completed: caches ready")
    return
async def fetch_candidates(ctx: BotContext) -> None:
    """Gather symbols for this cycle and build the evaluation batch using robust pipeline."""
    logger.info("PHASE: fetch_candidates starting")
    t0 = time.perf_counter()
    # Get mode and exchange ID for routing logic
    ex_id = getattr(ctx.exchange, "id", "").lower()
    mode = ctx.config.get("mode", "cex")
    # Use the robust evaluation pipeline integration
    pipeline_integration = get_evaluation_pipeline_integration(ctx.config)
    # Determine batch size based on volatility and configuration
    base_size = ctx.config.get("symbol_batch_size", 10)
    # Try to get ATR for adaptive sizing (fallback to base_size if fails)
    try:
        # Get some initial symbols to compute ATR
        initial_symbols = ctx.config.get("symbols", [ctx.config.get("symbol")])[:10]
        avg_atr = compute_average_atr(
            initial_symbols, ctx.df_cache, ctx.config.get("timeframe", "1h")
        )
        adaptive_cfg = ctx.config.get("adaptive_scan", {})
        if adaptive_cfg.get("enabled"):
            baseline = adaptive_cfg.get("atr_baseline", 0.01)
            max_factor = adaptive_cfg.get("max_factor", 2.0)
            volatility_factor = min(max_factor, max(1.0, avg_atr / baseline))
        else:
            volatility_factor = 1.0
        ctx.volatility_factor = volatility_factor
        batch_size = int(base_size * volatility_factor)
    except Exception as e:
        logger.warning(f"Failed to compute adaptive batch size: {e}, using base size")
        batch_size = base_size
        ctx.volatility_factor = 1.0
    # Check if we should process ALL available symbols (now default)
    process_all_symbols = ctx.config.get("process_all_symbols", True)
    if process_all_symbols:
        logger.info("🔄 Processing ALL available symbols on exchange (process_all_symbols=True)")
        try:
            # Get all available USD symbols from the exchange
            if hasattr(ctx.exchange, 'symbols') and ctx.exchange.symbols:
                all_symbols = [s for s in ctx.exchange.symbols if s.endswith('/USD')]
                logger.info(f"📊 Found {len(all_symbols)} USD symbols on exchange")
                # Apply batch size limit for performance
                tokens = all_symbols[:batch_size]
                logger.info(
                f"🎯 Processing batch of {len(tokens)} symbols (limited by batch_size={batch_size})"
            )
                # Apply chunking for memory optimization if enabled
                comp_opts = ctx.config.get("comprehensive_mode_optimization", {})
                if comp_opts.get("enable_memory_optimization", False):
                    chunk_size = comp_opts.get("batch_chunk_size", 25)
                    if len(tokens) > chunk_size:
                        logger.info(
                f"📦 Memory optimization: Processing in chunks of {chunk_size} symbols"
            )
                        # We'll implement chunked processing in the analysis phase
            else:
                logger.warning("⚠️ Exchange symbols not available, falling back to evaluation pipeline")
                process_all_symbols = False
        except Exception as e:
            logger.error(
                f"❌ Failed to get all symbols: {e}, falling back to evaluation pipeline"
            )
            process_all_symbols = False
    if not process_all_symbols:
        logger.info(f"🎯 Requesting {batch_size} tokens from evaluation pipeline")
        try:
            # Get tokens from the robust evaluation pipeline
            tokens = await get_tokens_for_evaluation(ctx.config, batch_size, ctx.exchange)
            logger.info(f"✅ Evaluation pipeline returned {len(tokens)} tokens")
            if not tokens:
                logger.warning("⚠️ Evaluation pipeline returned no tokens, using fallback")
                # Fallback to basic symbols if pipeline fails
                tokens = ctx.config.get("symbols", [ctx.config.get("symbol")])[:batch_size]
                logger.info(f"🔄 Using fallback tokens: {len(tokens)}")
        except Exception as e:
            logger.error(f"❌ Failed to get tokens from evaluation pipeline: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Ultimate fallback
            tokens = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD"]
            logger.info(f"🚨 Using ultimate fallback tokens: {len(tokens)}")
    ctx.timing["symbol_time"] = time.perf_counter() - t0
    # Handle Solana candidates separately if needed
    solana_tokens: list[str] = []
    sol_cfg = ctx.config.get("solana_scanner", {})
    if sol_cfg.get("enabled"):
        try:
            # Get additional Solana tokens for separate processing
            solana_tokens = await get_solana_new_tokens(sol_cfg)
            # Keep aside for potential onchain executor, avoid CEX fetches
            if solana_tokens and mode in ("onchain", "auto"):
                setattr(ctx, "solana_candidates", list(solana_tokens))
                logger.info(
                f"Captured {len(solana_tokens)} Solana tokens (kept separate from CEX batch)"
            )
            elif solana_tokens:
                logger.info(
                f"Found {len(solana_tokens)} Solana tokens but mode='{mode}' - processing through CEX pipeline"
            )
        except Exception as exc:
            logger.error("Solana scanner failed: %s", exc)
    # Set up symbol priority queue with the tokens from pipeline
    global symbol_priority_queue
    async with QUEUE_LOCK:
        # Clear priority queue if using Kraken in CEX mode
        if mode == "cex" and ex_id == "kraken":
            symbol_priority_queue.clear()
            logger.info("Cleared priority queue for Kraken CEX mode")
        else:
            logger.info(
                f"Not clearing priority queue - mode: {mode}, exchange: {ex_id}"
            )
        # Build priority queue from pipeline tokens
        if not symbol_priority_queue:
            # Convert tokens to (symbol, score) format for priority queue
            symbols_with_scores = [(token, 1.0) for token in tokens]
            symbol_priority_queue = build_priority_queue(symbols_with_scores)
        # NEVER add Solana tokens to CEX priority queue
        # Solana tokens must be processed through their own pipeline (scanning → evaluation → execution)
        # They should never mix with CEX processing
        if solana_tokens and not hasattr(ctx, 'solana_candidates'):
            logger.warning(
                f"Found {len(solana_tokens)} Solana tokens that were not set aside for separate processing. "
                f"This indicates a configuration issue - Solana tokens should never be processed through CEX pipeline."
            )
            # Set them aside to ensure they don't get processed through CEX
            if not hasattr(ctx, 'solana_candidates'):
                setattr(ctx, "solana_candidates", list(solana_tokens))
                logger.info(f"Corrected: Set {len(solana_tokens)} Solana tokens aside for separate processing")
        # Ensure we have enough tokens in queue
        if len(symbol_priority_queue) < batch_size:
            # Add more from pipeline if needed
            additional_tokens = await get_tokens_for_evaluation(ctx.config, batch_size * 2)
            additional_symbols = [(token, 0.5) for token in additional_tokens if token not in tokens]
            symbol_priority_queue.extend(build_priority_queue(additional_symbols))
        # Get final batch
        ctx.current_batch = [
            symbol_priority_queue.popleft()
            for _ in range(min(batch_size, len(symbol_priority_queue)))
        ]
    # Update timing metrics
    total_available = len(ctx.config.get("symbols") or [ctx.config.get("symbol")])
    ctx.timing["symbol_filter_ratio"] = (
        len(ctx.current_batch) / total_available if total_available else 1.0
    )
    logger.info("PHASE: fetch_candidates completed. Current batch size: %d, symbols: %s",
                len(ctx.current_batch), ctx.current_batch[:5])
    # Log pipeline status for monitoring
    try:
        pipeline_status = pipeline_integration.get_pipeline_status()
        logger.info(f"Evaluation pipeline status: {pipeline_status['status']} "
                   f"(received: {pipeline_status['metrics']['tokens_received']}, "
                   f"processed: {pipeline_status['metrics']['tokens_processed']})")
    except Exception as e:
        logger.debug(f"Could not get pipeline status: {e}")
async def process_solana_candidates(ctx: BotContext) -> None:
    """Process Solana candidates that are kept separate from CEX batch."""
    logger.info("PHASE: process_solana_candidates starting")
    # Check if we have Solana candidates to process
    if not hasattr(ctx, 'solana_candidates') or not ctx.solana_candidates:
        logger.info("PHASE: process_solana_candidates - no Solana candidates to process")
        return
    solana_candidates = ctx.solana_candidates
    logger.info(
                f"PHASE: process_solana_candidates - processing {len(solana_candidates)} Solana candidates"
            )
    # Process each Solana candidate
    for token_mint in solana_candidates:
        try:
            logger.info(f"Processing Solana candidate: {token_mint}")
            # Here you would add the logic to:
            # 1. Analyze the Solana token using Solana-specific data sources
            # 2. Evaluate trading opportunities
            # 3. Execute trades using Solana DEX execution
            # For now, just log that we're processing it
            logger.info(f"Solana candidate {token_mint} processed (placeholder)")
        except Exception as exc:
            logger.error(f"Error processing Solana candidate {token_mint}: {exc}")
    # Clear the candidates after processing
    ctx.solana_candidates = []
    logger.info("PHASE: process_solana_candidates completed")
async def scan_arbitrage(exchange: object, config: dict) -> list[str]:
    """Return symbols with profitable Solana arbitrage opportunities."""
    pairs: list[str] = config.get("arbitrage_pairs", [])
    if not pairs:
        return []
    try:
        from crypto_bot.utils.market_loader import fetch_geckoterminal_ohlcv
    except Exception:
        fetch_geckoterminal_ohlcv = None
    gecko_prices: dict[str, float] = {}
    if fetch_geckoterminal_ohlcv:
        for sym in pairs:
            try:
                res = await fetch_geckoterminal_ohlcv(sym, limit=1, return_price=True)
            except Exception:
                res = None
            if res:
                _data, _vol, price = res
                gecko_prices[sym] = price
    remaining = [s for s in pairs if s not in gecko_prices]
    dex_prices: dict[str, float] = gecko_prices.copy()
    if remaining:
        try:
            from crypto_bot.solana import fetch_solana_prices
        except Exception:
            fetch_solana_prices = None
        if fetch_solana_prices:
            dex_prices.update(await fetch_solana_prices(remaining))
    results: list[str] = []
    threshold = float(config.get("arbitrage_threshold", 0.0))
    for sym in pairs:
        dex_price = dex_prices.get(sym)
        if not dex_price:
            continue
        try:
            if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                ticker = await exchange.fetch_ticker(sym)
            else:
                ticker = await asyncio.to_thread(exchange.fetch_ticker, sym)
        except Exception:
            continue
        cex_price = ticker.get("last") or ticker.get("close")
        if cex_price is None:
            continue
        try:
            cex_val = float(cex_price)
        except Exception:
            continue
        if cex_val <= 0:
            continue
        diff = abs(dex_price - cex_val) / cex_val
        if diff >= threshold:
            results.append(sym)
    return results
async def update_caches(ctx: BotContext) -> None:
    """Update OHLCV and regime caches for the current symbol batch."""
    logger.info("PHASE: update_caches starting with batch size %d", len(ctx.current_batch))
    if ctx.services is None:
        raise RuntimeError("Market data service unavailable in context")
    exchange_id = getattr(ctx.exchange, "id", "") if ctx.exchange else ctx.config.get("exchange", "kraken")
    batch = ctx.current_batch
    if not batch:
        logger.info("PHASE: update_caches - empty batch, skipping")
        return
    start = time.perf_counter()
    # Update OHLCV cache for the current batch
    logger.info("Updating OHLCV cache for %d symbols", len(batch))
    # Initialize empty caches if they don't exist
    if not ctx.df_cache:
        ctx.df_cache = {}
    if not ctx.regime_cache:
        ctx.regime_cache = {}
    # Update OHLCV caches for analysis
    try:
        logger.info("Updating OHLCV cache for %d symbols", len(batch))
        # Get regime timeframes to include in the consolidated cache update
        regime_timeframes = ctx.config.get("regime_timeframes", [])
        # Update both main and regime caches in a single consolidated operation
        multi_response = await ctx.services.market_data.update_multi_tf_cache(
            MultiTimeframeOHLCVRequest(
                exchange_id=str(exchange_id),
                cache=ctx.df_cache,
                symbols=batch,
                config=ctx.config,
                additional_timeframes=regime_timeframes,
            )
        )
        ctx.df_cache = multi_response.cache
        regime_response = await ctx.services.market_data.update_regime_cache(
            RegimeCacheRequest(
                exchange_id=str(exchange_id),
                cache=ctx.regime_cache,
                symbols=batch,
                config=ctx.config,
                df_map=multi_response.cache,
            )
        )
        ctx.regime_cache = regime_response.cache
        # Filter batch to only include symbols that have data in the main timeframe
        main_tf = ctx.config.get("timeframe", "1h")
        filtered_batch = []
        missing_data_symbols = []
        for sym in batch:
            main_df = ctx.df_cache.get(main_tf, {}).get(sym)
            if main_df is not None and not main_df.empty:
                filtered_batch.append(sym)
            else:
                missing_data_symbols.append(sym)
                logger.debug(
                f"Removing {sym} from batch: no data available for timeframe {main_tf}"
            )
        if missing_data_symbols:
            logger.warning(
                f"Filtered out {len(missing_data_symbols)} symbols with missing data: {missing_data_symbols}"
            )
            logger.info(
                f"Batch reduced from {len(batch)} to {len(filtered_batch)} symbols with available data"
            )
        # Update the batch to only include symbols with data
        ctx.current_batch = filtered_batch
        logger.info("OHLCV cache update completed for %d symbols (%d filtered out)",
                   len(filtered_batch), len(missing_data_symbols))
    except Exception as exc:
        logger.error("Error updating OHLCV cache: %s", exc)
        # Continue with existing cache data if update fails
    vol_thresh = ctx.config.get("bounce_scalper", {}).get("vol_zscore_threshold")
    if vol_thresh is not None:
        tf = ctx.config.get("timeframe", "1h")
        status_updates = ctx.config.get("telegram", {}).get("status_updates", True)
        for sym in batch:
            df = ctx.df_cache.get(tf, {}).get(sym)
            if df is None or df.empty or "volume" not in df:
                continue
            vols = df["volume"].to_numpy(dtype=float)
            mean = float(np.mean(vols)) if len(vols) else 0.0
            std = float(np.std(vols))
            if std <= 0:
                continue
            z_scores = (vols - mean) / std
            z_max = float(np.max(z_scores))
            if z_max > vol_thresh:
                async with QUEUE_LOCK:
                    symbol_priority_queue.appendleft(sym)
                msg = f"Volume spike priority for {sym}: z={z_max:.2f}"
                logger.info(msg)
                if status_updates and ctx.notifier and ctx.notifier.enabled:
                    ctx.notifier.notify(msg)
    ctx.timing["ohlcv_fetch_latency"] = time.perf_counter() - start
    logger.info("PHASE: update_caches completed")
async def enrich_with_pyth(ctx: BotContext) -> None:
    """Update cached OHLCV using the latest Pyth prices."""
    batch = ctx.current_batch
    if not batch:
        return
    async with aiohttp.ClientSession() as session:
        for sym in batch:
            if not sym.endswith("/USDC"):
                continue
            base = sym.split("/")[0]
            try:
                url = f"https://hermes.pyth.network/v2/price_feeds?query={base}"
                async with session.get(url, timeout=10) as resp:
                    feeds = await resp.json()
            except Exception:
                continue
            feed_id = None
            for item in feeds:
                attrs = item.get("attributes", {})
                if attrs.get("base") == base and attrs.get("quote_currency") == "USD":
                    feed_id = item.get("id")
                    break
            if not feed_id:
                continue
            try:
                url = (
                    "https://hermes.pyth.network/api/latest_price_feeds?ids[]="
                    f"{feed_id}"
                )
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
            except Exception:
                continue
            if not data:
                continue
            price_info = data[0].get("price")
            if not price_info:
                continue
            price = float(price_info.get("price", 0)) * (
                10 ** price_info.get("expo", 0)
            )
            for cache in ctx.df_cache.values():
                df = cache.get(sym)
                if df is not None and not df.empty:
                    df.loc[df.index[-1], "close"] = price
async def analyse_batch(ctx: BotContext) -> None:
    """Run signal analysis on the current batch with comprehensive mode optimizations."""
    logger.info("PHASE: analyse_batch starting with batch size %d", len(ctx.current_batch))
    batch = ctx.current_batch
    if not batch:
        logger.info("PHASE: analyse_batch - empty batch, setting empty results")
        ctx.analysis_results = []
        return
    # Check for comprehensive mode optimizations
    comp_opts = ctx.config.get("comprehensive_mode_optimization", {})
    enable_chunking = comp_opts.get("enable_memory_optimization", False)
    chunk_size = comp_opts.get("batch_chunk_size", 25)
    memory_cleanup_interval = comp_opts.get("memory_cleanup_interval", 50)
    enable_progress = comp_opts.get("enable_progress_tracking", False)
    mode = ctx.config.get("mode", "cex")
    all_results = []
    if enable_chunking and len(batch) > chunk_size:
        logger.info(
                f"📦 Processing {len(batch)} symbols in chunks of {chunk_size} for memory optimization"
            )
        # Process in chunks
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i + chunk_size]
            logger.info(
                f"🔄 Processing chunk {i//chunk_size + 1}/{(len(batch) + chunk_size - 1)//chunk_size} ({len(chunk)} symbols)"
            )
            chunk_tasks = []
            for sym in chunk:
                df_map = {}
                for tf, c in ctx.df_cache.items():
                    df = c.get(sym)
                    if df is not None:
                        df_map[tf] = df
                for tf, cache in ctx.regime_cache.items():
                    df = cache.get(sym)
                    if df is not None:
                        df_map[tf] = df
                chunk_tasks.append(analyze_symbol(sym, df_map, mode, ctx.config, ctx.notifier))
            logger.info(
                f"PHASE: analyse_batch - running analysis on chunk with {len(chunk_tasks)} tasks"
            )
            chunk_results = await asyncio.gather(*chunk_tasks)
            all_results.extend(chunk_results)
            # Memory cleanup if enabled
            if (i + chunk_size) % memory_cleanup_interval == 0:
                logger.debug(f"🧹 Memory cleanup after processing {i + chunk_size} symbols")
                # Force garbage collection
                import gc
                gc.collect()
            if enable_progress:
                processed = min(i + chunk_size, len(batch))
                progress = (processed / len(batch)) * 100
                logger.info(
                f"📊 Progress: {processed}/{len(batch)} symbols ({progress:.1f}%)"
            )
    else:
        # Standard processing without chunking
        tasks = []
        # Since update_caches already filtered the batch to only include symbols with data,
        # we can safely assume all symbols in the batch have the required data
        for sym in batch:
            df_map = {}
            for tf, c in ctx.df_cache.items():
                df = c.get(sym)
                if df is not None:
                    df_map[tf] = df
            for tf, cache in ctx.regime_cache.items():
                df = cache.get(sym)
                if df is not None:
                    df_map[tf] = df
            tasks.append(analyze_symbol(sym, df_map, mode, ctx.config, ctx.notifier))
        logger.info("PHASE: analyse_batch - running analysis on %d tasks", len(tasks))
        all_results = await asyncio.gather(*tasks)
    ctx.analysis_results = all_results
    logger.info("PHASE: analyse_batch - analysis completed, %d total results", len(ctx.analysis_results))
    global UNKNOWN_COUNT, TOTAL_ANALYSES
    for res in ctx.analysis_results:
        if res.get("skip"):
            continue
        TOTAL_ANALYSES += 1
        if res.get("regime") == "unknown":
            UNKNOWN_COUNT += 1
    logger.info("PHASE: analyse_batch completed - processed %d results, %d actionable signals", len(ctx.analysis_results), len([r for r in ctx.analysis_results if not r.get("skip") and r.get("direction") != "none"]))
async def execute_solana_trade(
    ctx: BotContext,
    candidate: dict,
    sym: str,
    size: float,
    price: float,
    strategy: str,
    side: str,
    sentiment_boost: float = 1.0,
) -> bool:
    """Execute a Solana trade asynchronously."""
    if ctx.services is None:
        raise RuntimeError("Portfolio service unavailable in context")
    try:
        from crypto_bot.solana import sniper_solana
        from crypto_bot.solana_trading import sniper_trade
        sol_score, _ = sniper_solana.generate_signal(candidate["df"])
        if sol_score > 0.7:
            base, quote = sym.split("/")
            # Apply sentiment boost to size
            adjusted_size = size * sentiment_boost
            # Validate Solana token exists before trading
            try:
                from crypto_bot.solana import sniper_solana
                test_score = sniper_solana.generate_signal(None)  # Basic validation
                if test_score == 0.0:  # Token not found or invalid
                    logger.error(f"Solana token validation failed for {base} - aborting trade")
                    return False
            except Exception as e:
                logger.error(
                f"Solana token validation failed for {base}: {e} - aborting trade"
            )
                return False
            await sniper_trade(
                ctx.config.get("wallet_address", ""),
                quote,
                base,
                adjusted_size,
                dry_run=ctx.config.get("execution_mode") == "dry_run",
                slippage_bps=ctx.config.get("solana_slippage_bps", 50),
                notifier=ctx.notifier,
                paper_wallet=ctx.paper_wallet if ctx.config.get("execution_mode") == "dry_run" else None,
            )
            # Update context for successful trade
            ctx.risk_manager.allocate_capital(strategy, size)
            amount = size / price if price > 0 else 0.0
            # Record trade through centralized TradeManager
            if ctx.trade_manager:
                from decimal import Decimal
                trade_response = ctx.services.portfolio.create_trade(
                    CreateTradeRequest(
                        symbol=sym,
                        side=side,
                        amount=Decimal(str(amount)),
                        price=Decimal(str(price)),
                        strategy=strategy,
                        exchange="solana",
                        metadata={
                            "regime": candidate.get("regime"),
                            "confidence": candidate.get("score", 0.0)
                        },
                    )
                )
                trade = trade_response.trade
                trade_id = ctx.trade_manager.record_trade(trade)
                logger.info(
                f"Solana trade recorded: {trade.symbol} {trade.side} {trade.amount} @ {trade.price}"
            )
                # Sync positions if using TradeManager as source
                if hasattr(ctx, 'sync_positions_from_trade_manager') and ctx.use_trade_manager_as_source:
                    ctx.sync_positions_from_trade_manager()
                # Update position in ctx.positions for backward compatibility
                position = ctx.trade_manager.get_position(sym)
                if position:
                    ctx.positions[sym] = {
                        "side": position.side,
                        "entry_price": float(position.average_price),
                        "entry_time": position.entry_time.isoformat(),
                        "regime": candidate.get("regime"),
                        "strategy": strategy,
                        "confidence": candidate.get("score", 0.0),
                        "pnl": 0.0,  # Will be updated by TradeManager
                        "size": float(position.total_amount),
                        "trailing_stop": float(position.stop_loss_price) if position.stop_loss_price else 0.0,
                        "highest_price": float(position.highest_price) if position.highest_price else price,
                        "lowest_price": float(position.lowest_price) if position.lowest_price else price,
                    }
            # Handle paper wallet updates for backward compatibility
            if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                try:
                    trade_id = ctx.paper_wallet.open(sym, side, amount, price)
                    ctx.balance = ctx.paper_wallet.balance
                    logger.info(
                f"Paper Solana trade opened: {side} {amount} {sym} @ ${price:.6f}, balance: ${ctx.balance:.2f}"
            )
                    if ctx.notifier and ctx.config.get("telegram", {}).get("trade_updates", True) and ctx.notifier.enabled:
                        paper_msg = f"📄 Paper Solana Trade Opened\n{side.upper()} {amount:.4f} {sym}\nPrice: ${price:.6f}\nBalance: ${ctx.balance:.2f}\nStrategy: {strategy}"
                        ctx.notifier.notify(paper_msg)
                except Exception as e:
                    logger.error(f"Failed to open paper Solana trade: {e}")
                    return False
            sync_paper_wallet_balance(ctx)
            update_position_pnl(ctx)
            try:
                log_position(sym, side, amount, price, price, ctx.balance)
            except Exception:
                pass
            logger.info(f"Solana trade executed successfully: {side} {amount} {sym}")
            return True
        else:
            logger.debug(f"Solana sniper score too low ({sol_score:.2f}) for {sym}")
            return False
    except Exception as e:
        logger.error(f"Error executing Solana trade for {sym}: {e}")
        return False
async def execute_cex_trade(
    ctx: BotContext,
    candidate: dict,
    sym: str,
    size: float,
    price: float,
    strategy: str,
    side: str,
    sentiment_boost: float = 1.0,
) -> bool:
    """Execute a CEX trade asynchronously."""
    if ctx.services is None:
        raise RuntimeError("Execution service unavailable in context")
    try:
        # Apply sentiment boost to size
        adjusted_size = size * sentiment_boost
        amount = adjusted_size / price if price > 0 else 0.0
        # Validate symbol exists on exchange before trading
        try:
            if asyncio.iscoroutinefunction(getattr(ctx.exchange, "fetch_ticker", None)):
                test_ticker = await ctx.exchange.fetch_ticker(sym)
            else:
                test_ticker = await asyncio.to_thread(ctx.exchange.fetch_ticker, sym)
            if not test_ticker or not test_ticker.get("last"):
                logger.error(
                f"Symbol {sym} not found on exchange or has no price data - aborting trade"
            )
                return False
        except Exception as e:
            logger.error(f"Symbol validation failed for {sym}: {e} - aborting trade")
            return False
        order_response = await ctx.services.execution.execute_trade(
            TradeExecutionRequest(
                exchange=ctx.exchange,
                ws_client=ctx.ws_client,
                symbol=sym,
                side=side,
                amount=amount,
                notifier=ctx.notifier,
                dry_run=ctx.config.get("execution_mode") == "dry_run",
                use_websocket=ctx.config.get("use_websocket", False),
                config=ctx.config,
            )
        )
        order = order_response.order
        if order:
            # Handle take profit for bounce scalper strategy
            take_profit = None
            if strategy == "bounce_scalper":
                depth = int(ctx.config.get("liquidity_depth", 10))
                exchange_id_local = getattr(ctx.exchange, "id", ctx.config.get("exchange", "kraken"))
                order_book_resp = await ctx.services.market_data.fetch_order_book(
                    OrderBookRequest(
                        exchange_id=str(exchange_id_local),
                        symbol=sym,
                        depth=depth,
                        config=ctx.config,
                    )
                )
                book = order_book_resp.order_book
                dist = _closest_wall_distance(book, price, side)
                if dist is not None:
                    take_profit = dist * 0.8
            ctx.risk_manager.register_stop_order(
                order,
                strategy=strategy,
                symbol=sym,
                entry_price=price,
                confidence=candidate.get("score", 0.0),
                direction=side,
                take_profit=take_profit,
            )
            # Place native exchange stop-loss order immediately after entry if enabled
            try:
                place_native = ctx.config.get("exit_strategy", {}).get("place_native_stop", True)
                is_dry_run = ctx.config.get("execution_mode") == "dry_run"
                if place_native:
                    # Use TradeManager-calculated stop price for precision
                    tm_position = ctx.trade_manager.get_position(sym) if hasattr(ctx, "trade_manager") and ctx.trade_manager else None
                    stop_price = float(tm_position.stop_loss_price) if tm_position and tm_position.stop_loss_price else None
                    stop_amount = float(tm_position.total_amount) if tm_position else amount
                    if stop_price and stop_amount > 0:
                        stop_side = "sell" if side == "buy" else "buy"
                        if is_dry_run:
                            logger.info(
                                f"[DRY_RUN] Would place native stop {stop_side} {stop_amount} {sym} at {stop_price:.6f}"
                            )
                        else:
                            # Avoid blocking the event loop with a sync ccxt call
                            from crypto_bot.execution import cex_executor
                            stop_order = await asyncio.to_thread(
                                cex_executor.place_stop_order,
                                ctx.exchange,
                                sym,
                                stop_side,
                                stop_amount,
                                stop_price,
                                None,
                                None,
                                ctx.notifier,
                                False,
                            )
                            if stop_order:
                                # Record the protective stop in risk manager for tracking/cancellation
                                ctx.risk_manager.register_stop_order(
                                    stop_order,
                                    strategy=strategy,
                                    symbol=sym,
                                    entry_price=price,
                                    confidence=candidate.get("score", 0.0),
                                    direction=side,
                                )
                    else:
                        logger.warning(
                            f"Native stop not placed for {sym}: missing computed stop price or amount"
                        )
            except Exception as e:
                logger.warning(f"Native stop placement failed for {sym}: {e}")
            # Update context for successful trade
            ctx.risk_manager.allocate_capital(strategy, size)
            # Record trade through centralized TradeManager
            if ctx.trade_manager:
                from decimal import Decimal
                trade_response = ctx.services.portfolio.create_trade(
                    CreateTradeRequest(
                        symbol=sym,
                        side=side,
                        amount=Decimal(str(amount)),
                        price=Decimal(str(price)),
                        strategy=strategy,
                        exchange="cex",
                        metadata={
                            "regime": candidate.get("regime"),
                            "confidence": candidate.get("score", 0.0)
                        },
                    )
                )
                trade = trade_response.trade
                trade_id = ctx.trade_manager.record_trade(trade)
                logger.info(
                f"CEX trade recorded: {trade.symbol} {trade.side} {trade.amount} @ {trade.price}"
            )
                # Sync positions if using TradeManager as source
                if hasattr(ctx, 'sync_positions_from_trade_manager') and ctx.use_trade_manager_as_source:
                    ctx.sync_positions_from_trade_manager()
                # Update position in ctx.positions for backward compatibility
                position = ctx.trade_manager.get_position(sym)
                if position:
                    ctx.positions[sym] = {
                        "side": position.side,
                        "entry_price": float(position.average_price),
                        "entry_time": position.entry_time.isoformat(),
                        "regime": candidate.get("regime"),
                        "strategy": strategy,
                        "confidence": candidate.get("score", 0.0),
                        "pnl": 0.0,  # Will be updated by TradeManager
                        "size": float(position.total_amount),
                        "trailing_stop": float(position.stop_loss_price) if position.stop_loss_price else 0.0,
                        "highest_price": float(position.highest_price) if position.highest_price else price,
                        "lowest_price": float(position.lowest_price) if position.lowest_price else price,
                    }
            # Start real-time monitoring for this position
            if hasattr(ctx, 'position_monitor') and ctx.trade_manager:
                position = ctx.trade_manager.get_position(sym)
                if position:
                    await ctx.position_monitor.start_monitoring(sym, ctx.positions[sym])
            # Handle paper wallet updates for backward compatibility
            if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                try:
                    trade_id = ctx.paper_wallet.open(sym, side, amount, price)
                    ctx.balance = ctx.paper_wallet.balance
                    logger.info(
                f"Paper CEX trade opened: {side} {amount} {sym} @ ${price:.6f}, balance: ${ctx.balance:.2f}"
            )
                    if ctx.notifier and ctx.config.get("telegram", {}).get("trade_updates", True) and ctx.notifier.enabled:
                        paper_msg = f"📄 Paper CEX Trade Opened\n{side.upper()} {amount:.4f} {sym}\nPrice: ${price:.6f}\nBalance: ${ctx.balance:.2f}\nStrategy: {strategy}"
                        ctx.notifier.notify(paper_msg)
                except Exception as e:
                    logger.error(f"Failed to open paper CEX trade: {e}")
                    return False
            sync_paper_wallet_balance(ctx)
            update_position_pnl(ctx)
            try:
                log_position(sym, side, amount, price, price, ctx.balance)
            except Exception:
                pass
            logger.info(f"CEX trade executed successfully: {side} {amount} {sym}")
            return True
        else:
            logger.warning(f"CEX trade failed for {sym}")
            return False
    except Exception as e:
        logger.error(f"Error executing CEX trade for {sym}: {e}")
        return False
class AsyncTradeManager:
    """Manages asynchronous trading tasks to ensure Solana and CEX trading run independently."""
    def __init__(self):
        self.active_tasks: set[asyncio.Task] = set()
        self.task_lock = asyncio.Lock()
    async def execute_trade_async(self, coro) -> None:
        """Execute a trade coroutine as a background task."""
        async with self.task_lock:
            task = asyncio.create_task(coro)
            self.active_tasks.add(task)
            # Remove task from active set when it completes
            def cleanup_task(task_ref):
                self.active_tasks.discard(task_ref)
            task.add_done_callback(cleanup_task)
    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        """Wait for all active trading tasks to complete or timeout."""
        if not self.active_tasks:
            return
        try:
            await asyncio.wait(self.active_tasks, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Some trading tasks timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Error waiting for trading tasks: {e}")
    def cancel_all(self) -> None:
        """Cancel all active trading tasks."""
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
    async def cleanup_completed(self) -> None:
        """Clean up any completed tasks from the active set."""
        async with self.task_lock:
            completed_tasks = {task for task in self.active_tasks if task.done()}
            self.active_tasks -= completed_tasks
            # Log any exceptions from completed tasks
            for task in completed_tasks:
                if task.exception():
                    logger.error(f"Trading task failed: {task.exception()}")
# Global trade manager instance
trade_manager = AsyncTradeManager()
async def execute_signals(ctx: BotContext) -> None:
    """Open trades for qualified analysis results using async execution."""
    logger.info("PHASE: execute_signals starting")
    results = getattr(ctx, "analysis_results", [])
    if not results:
        logger.info("PHASE: execute_signals - No analysis results to act on")
        return
    # Prioritize by score
    from crypto_bot.utils.telemetry import telemetry
    initial = len(results)
    results = [r for r in results if not r.get("skip") and r.get("direction") != "none"]
    filtered = initial - len(results)
    if filtered > 0:
        telemetry.inc("exec.filtered_non_actionable", filtered)
    if not results:
        logger.info("PHASE: execute_signals - All signals filtered out - nothing actionable")
        telemetry.inc("exec.no_actionable")
        return
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_n = ctx.config.get("top_n_symbols", 3)
    executed = 0
    for candidate in results[:top_n]:
        logger.info("Analysis result: %s", candidate)
        if not ctx.position_guard or not ctx.position_guard.can_open(ctx.positions):
            logger.info("Max open trades reached; skipping remaining signals")
            telemetry.inc("exec.blocked_max_open_trades")
            break
        sym = candidate["symbol"]
        if sym in ctx.positions:
            logger.info("Existing position for %s - skipping", sym)
            telemetry.inc("exec.skip_existing_position")
            continue
        df = candidate["df"]
        price = df["close"].iloc[-1]
        score = candidate.get("score", 0.0)
        strategy = candidate.get("name", "")
        allowed, reason = ctx.risk_manager.allow_trade(df, strategy)
        if not allowed:
            logger.info("Trade blocked for %s: %s", sym, reason)
            telemetry.inc(f"exec.blocked_risk.{str(reason).strip().replace(' ', '_').lower()}")
            continue
        probs = candidate.get("probabilities", {})
        reg_prob = float(probs.get(candidate.get("regime"), 0.0))
        # Get LunarCrush sentiment boost if available
        sentiment_boost = 1.0
        try:
            from crypto_bot.sentiment_filter import get_lunarcrush_sentiment_boost
            sentiment_boost = await get_lunarcrush_sentiment_boost(sym, candidate["direction"])
        except Exception as exc:
            logger.debug(f"Failed to get sentiment boost for {sym}: {exc}")
        base_size = ctx.risk_manager.position_size(
            reg_prob,
            ctx.balance,
            df,
            atr=candidate.get("atr"),
            price=price,
        )
        # Apply sentiment boost
        size = base_size * sentiment_boost
        if size <= 0:
            logger.info("Calculated size %.4f for %s - skipping", size, sym)
            telemetry.inc("exec.blocked_zero_size")
            continue
        if not ctx.risk_manager.can_allocate(strategy, size, ctx.balance):
            logger.info(
                "Insufficient capital to allocate %.4f for %s via %s",
                size,
                sym,
                strategy,
            )
            telemetry.inc("exec.blocked_insufficient_capital")
            continue
        side = direction_to_side(candidate["direction"])
        if side == "sell" and not ctx.config.get("allow_short", False):
            logger.info("Short selling disabled; skipping signal for %s", sym)
            telemetry.inc("exec.blocked_short_disabled")
            continue
        # Execute trades asynchronously based on symbol type
        start_exec = time.perf_counter()
        if sym.endswith("/USDC"):
            # Solana trade - execute asynchronously
            logger.info(f"Queueing Solana trade for {sym}")
            await trade_manager.execute_trade_async(
                execute_solana_trade(
                    ctx,
                    candidate,
                    sym,
                    size,
                    price,
                    strategy,
                    side,
                    sentiment_boost,
                )
            )
            executed += 1
        else:
            # CEX trade - execute asynchronously
            logger.info(f"Queueing CEX trade for {sym}")
            await trade_manager.execute_trade_async(
                execute_cex_trade(
                    ctx,
                    candidate,
                    sym,
                    size,
                    price,
                    strategy,
                    side,
                    sentiment_boost,
                )
            )
            executed += 1
        ctx.timing["execution_latency"] = max(
            ctx.timing.get("execution_latency", 0.0),
            time.perf_counter() - start_exec,
        )
        # Handle micro-scalp monitoring
        if strategy == "micro_scalp":
            asyncio.create_task(_monitor_micro_scalp_exit(ctx, sym))
    # Wait for all trading tasks to complete or timeout
    if executed > 0:
        logger.info(f"Waiting for {executed} trading tasks to complete...")
        await trade_manager.wait_for_completion(timeout=30.0)
        await trade_manager.cleanup_completed()
        logger.info("All trading tasks completed or timed out")
    logger.info("PHASE: execute_signals completed - executed %d trades from %d candidate signals", executed, len(results[:top_n]))
    if executed == 0:
        logger.info("No trades executed from %d candidate signals", len(results[:top_n]))
async def handle_exits(ctx: BotContext) -> None:
    """Check open positions for exit conditions with enhanced monitoring."""
    logger.info("PHASE: handle_exits starting with %d positions", len(ctx.positions))
    if ctx.services is None:
        raise RuntimeError("Execution service unavailable in context")
    tf = ctx.config.get("timeframe", "1h")
    tf_cache = ctx.df_cache.get(tf, {})
    # Clean up old monitors for closed positions
    if hasattr(ctx, 'position_monitor'):
        await ctx.position_monitor.cleanup_old_monitors()
    for sym, pos in list(ctx.positions.items()):
        df = tf_cache.get(sym)
        if df is None or df.empty:
            continue
        current_price = float(df["close"].iloc[-1])
        pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * (
            1 if pos["side"] == "buy" else -1
        )
        # Enhanced trailing stop logic with real-time monitoring
        if pnl_pct >= ctx.config.get("exit_strategy", {}).get("min_gain_to_trail", 0):
            if pos["side"] == "buy":  # Long position
                if current_price > pos.get("highest_price", pos["entry_price"]):
                    pos["highest_price"] = current_price
                # Calculate trailing stop from the actual highest price since entry
                pos["trailing_stop"] = pos["highest_price"] * (1 - ctx.config.get("exit_strategy", {}).get("trailing_stop_pct", 0.02))
            else:  # Short position
                if current_price < pos.get("lowest_price", pos["entry_price"]):
                    pos["lowest_price"] = current_price
                # Calculate trailing stop from the actual lowest price since entry
                pos["trailing_stop"] = pos["lowest_price"] * (1 + ctx.config.get("exit_strategy", {}).get("trailing_stop_pct", 0.02))
        # Check exit conditions with enhanced monitoring
        exit_signal, new_stop = should_exit(
            df,
            current_price,
            pos.get("trailing_stop", 0.0),
            ctx.config,
            ctx.risk_manager,
            pos["side"],  # Pass position side
            pos["entry_price"],  # Pass entry price for take profit
        )
        pos["trailing_stop"] = new_stop
        if exit_signal:
            await ctx.services.execution.execute_trade(
                TradeExecutionRequest(
                    exchange=ctx.exchange,
                    ws_client=ctx.ws_client,
                    symbol=sym,
                    side=opposite_side(pos["side"]),
                    amount=pos["size"],
                    notifier=ctx.notifier,
                    dry_run=ctx.config.get("execution_mode") == "dry_run",
                    use_websocket=ctx.config.get("use_websocket", False),
                    config=ctx.config,
                )
            )
            if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                try:
                    pnl = ctx.paper_wallet.close(sym, pos["size"], current_price)
                    ctx.balance = ctx.paper_wallet.balance
                    logger.info(
                f"Paper trade closed: {pos['side']} {pos['size']} {sym} @ ${current_price:.6f}, PnL: ${pnl:.2f}, balance: ${ctx.balance:.2f}"
            )
                    # Send Telegram notification for paper trade exit
                    if ctx.notifier and ctx.config.get("telegram", {}).get("trade_updates", True) and ctx.notifier.enabled:
                        pnl_emoji = "💰" if pnl >= 0 else "📉"
                        paper_exit_msg = f"📄 Paper Trade Closed {pnl_emoji}\n{pos['side'].upper()} {pos['size']:.4f} {sym}\nEntry: ${pos['entry_price']:.6f}\nExit: ${current_price:.6f}\nPnL: ${pnl:.2f}\nBalance: ${ctx.balance:.2f}"
                        ctx.notifier.notify(paper_exit_msg)
                except Exception as e:
                    logger.error(f"Failed to close paper trade: {e}")
                    # Continue with position closure even if paper wallet fails
            # Update position PnL before removing from ctx.positions
            if sym in ctx.positions:
                pos = ctx.positions[sym]
                if ctx.paper_wallet and sym in ctx.paper_wallet.positions:
                    final_pnl = ctx.paper_wallet.unrealized(sym, current_price)
                    pos["pnl"] = final_pnl
                    logger.info(f"Final PnL for {sym}: ${final_pnl:.2f}")
            # Ensure balance is synchronized
            sync_paper_wallet_balance(ctx)
            # Stop monitoring this position
            if hasattr(ctx, 'position_monitor'):
                await ctx.position_monitor.stop_monitoring(sym)
        ctx.risk_manager.deallocate_capital(
            pos.get("strategy", ""), pos["size"] * pos["entry_price"]
        )
        ctx.positions.pop(sym, None)
        try:
            log_position(
                sym,
                pos["side"],
                pos["size"],
                pos["entry_price"],
                current_price,
                ctx.balance,
            )
        except Exception:
            pass
async def force_exit_all(ctx: BotContext) -> None:
    """Liquidate all open positions immediately."""
    if ctx.services is None:
        raise RuntimeError("Execution service unavailable in context")
    tf = ctx.config.get("timeframe", "1h")
    tf_cache = ctx.df_cache.get(tf, {})
    for sym, pos in list(ctx.positions.items()):
        df = tf_cache.get(sym)
        exit_price = pos["entry_price"]
        if df is not None and not df.empty:
            exit_price = float(df["close"].iloc[-1])
        await ctx.services.execution.execute_trade(
            TradeExecutionRequest(
                exchange=ctx.exchange,
                ws_client=ctx.ws_client,
                symbol=sym,
                side=opposite_side(pos["side"]),
                amount=pos["size"],
                notifier=ctx.notifier,
                dry_run=ctx.config.get("execution_mode") == "dry_run",
                use_websocket=ctx.config.get("use_websocket", False),
                config=ctx.config,
            )
        )
        if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
            try:
                pnl = ctx.paper_wallet.close(sym, pos["size"], exit_price)
                ctx.balance = ctx.paper_wallet.balance
                logger.info(
                f"Paper trade force closed: {pos['side']} {pos['size']} {sym} @ ${exit_price:.6f}, PnL: ${pnl:.2f}, balance: ${ctx.balance:.2f}"
            )
                # Send Telegram notification for paper trade force exit
                if ctx.notifier and ctx.config.get("telegram", {}).get("trade_updates", True) and ctx.notifier.enabled:
                    pnl_emoji = "💰" if pnl >= 0 else "📉"
                    paper_force_exit_msg = f"📄 Paper Trade FORCE CLOSED {pnl_emoji}\n{pos['side'].upper()} {pos['size']:.4f} {sym}\nEntry: ${pos['entry_price']:.6f}\nExit: ${exit_price:.6f}\nPnL: ${pnl:.2f}\nBalance: ${ctx.balance:.2f}"
                    ctx.notifier.notify(paper_force_exit_msg)
            except Exception as e:
                logger.error(f"Failed to force close paper trade: {e}")
                # Continue with position closure even if paper wallet fails
        ctx.risk_manager.deallocate_capital(
            pos.get("strategy", ""), pos["size"] * pos["entry_price"]
        )
        ctx.positions.pop(sym, None)
        try:
            log_position(
                sym,
                pos["side"],
                pos["size"],
                pos["entry_price"],
                exit_price,
                ctx.balance,
            )
        except Exception:
            pass
async def _monitor_micro_scalp_exit(ctx: BotContext, sym: str) -> None:
    """Monitor a micro-scalp trade and exit based on :func:`monitor_price`."""
    if ctx.services is None:
        raise RuntimeError("Execution service unavailable in context")
    pos = ctx.positions.get(sym)
    if not pos:
        return
    tf = ctx.config.get("scalp_timeframe", "1m")
    def feed() -> float:
        df = ctx.df_cache.get(tf, {}).get(sym)
        if df is None or df.empty:
            return pos["entry_price"]
        return float(df["close"].iloc[-1])
    res = await monitor_price(feed, pos["entry_price"], {})
    exit_price = res.get("exit_price", feed())
    await ctx.services.execution.execute_trade(
        TradeExecutionRequest(
            exchange=ctx.exchange,
            ws_client=ctx.ws_client,
            symbol=sym,
            side=opposite_side(pos["side"]),
            amount=pos["size"],
            notifier=ctx.notifier,
            dry_run=ctx.config.get("execution_mode") == "dry_run",
            use_websocket=ctx.config.get("use_websocket", False),
            config=ctx.config,
        )
    )
    if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
        try:
            pnl = ctx.paper_wallet.close(sym, pos["size"], exit_price)
            ctx.balance = ctx.paper_wallet.balance
            logger.info(
                f"Paper trade micro-scalp closed: {pos['side']} {pos['size']} {sym} @ ${exit_price:.6f}, PnL: ${pnl:.2f}, balance: ${ctx.balance:.2f}"
            )
            # Send Telegram notification for paper trade micro-scalp exit
            if ctx.notifier and ctx.config.get("telegram", {}).get("trade_updates", True) and ctx.notifier.enabled:
                pnl_emoji = "💰" if pnl >= 0 else "📉"
                paper_scalp_exit_msg = f"📄 Paper Micro-Scalp Closed {pnl_emoji}\n{pos['side'].upper()} {pos['size']:.4f} {sym}\nEntry: ${pos['entry_price']:.6f}\nExit: ${exit_price:.6f}\nPnL: ${pnl:.2f}\nBalance: ${ctx.balance:.2f}"
                ctx.notifier.notify(paper_scalp_exit_msg)
        except Exception as e:
            logger.error(f"Failed to close paper trade micro-scalp: {e}")
            # Continue with position closure even if paper wallet fails
    ctx.risk_manager.deallocate_capital(
        pos.get("strategy", ""), pos["size"] * pos["entry_price"]
    )
    ctx.positions.pop(sym, None)
    try:
        log_position(
            sym, pos["side"], pos["size"], pos["entry_price"], exit_price, ctx.balance
        )
    except Exception:
        pass
async def _rotation_loop(
    rotator: PortfolioRotator,
    exchange: object,
    wallet: str,
    state: dict,
    notifier: Optional[TelegramNotifier],
    check_balance_change: callable,
) -> None:
    """Periodically rotate portfolio holdings."""
    interval = rotator.config.get("interval_days", 7) * 86400
    while True:
        try:
            if state.get("running") and rotator.config.get("enabled"):
                # Temporarily disable rotation to prevent nonce errors
                logger.debug("Rotation temporarily disabled to prevent nonce errors")
                await asyncio.sleep(300)  # Sleep for 5 minutes and continue
                continue
                if asyncio.iscoroutinefunction(
                    getattr(exchange, "fetch_balance", None)
                ):
                    bal = await exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(exchange.fetch_balance_with_retry)
                current_balance = (
                    bal.get("USDT", {}).get("free", 0)
                    if isinstance(bal.get("USDT"), dict)
                    else bal.get("USDT", 0)
                )
                check_balance_change(float(current_balance), "external change")
                holdings = {
                    k: (v.get("total") if isinstance(v, dict) else v)
                    for k, v in bal.items()
                }
                await rotator.rotate(exchange, wallet, holdings, notifier)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - rotation errors
            logger.error("Rotation loop error: %s", exc, exc_info=True)
        sleep_remaining = interval
        while sleep_remaining > 0:
            sleep_chunk = min(60, sleep_remaining)
            await asyncio.sleep(sleep_chunk)
            sleep_remaining -= sleep_chunk
            if not (rotator.config.get("enabled") and state.get("running")):
                break
async def _main_impl(
    *,
    clock: Optional[Callable[[], datetime]] = None,
    timer: Optional[Callable[[], float]] = None,
    rng: Optional[random.Random] = None,
    numpy_rng: Optional[Any] = None,
) -> TelegramNotifier:
    """Implementation for running the trading bot."""
    logger.info("Starting bot")
    global UNKNOWN_COUNT, TOTAL_ANALYSES
    config = load_config()
    services: ServiceContainer = create_service_container()
    sol_syms = [fix_symbol(s) for s in config.get("solana_symbols", [])]
    sol_syms = [f"{s}/USDC" if "/" not in s else s for s in sol_syms]
    if sol_syms:
        merged = list(dict.fromkeys((config.get("symbols") or []) + sol_syms))
        config["symbols"] = merged
    try:
        set_last_config_mtime(CONFIG_PATH.stat().st_mtime)
    except OSError:
        pass
    metrics_path = (
        Path(config.get("metrics_csv")) if config.get("metrics_csv") else None
    )
    async def solana_scan_loop() -> None:
        """Periodically fetch new Solana tokens and queue them using enhanced scanner."""
        cfg = config.get("solana_scanner", {})
        enhanced_cfg = config.get("enhanced_scanning", {})
        interval = cfg.get("interval_minutes", 5) * 60
        # Use enhanced scanner if configured
        if enhanced_cfg.get("enabled", False):
            logger.info("Using enhanced Solana scanner")
            from crypto_bot.solana.enhanced_scanner import get_enhanced_scanner
            scanner = get_enhanced_scanner(config)
            # Start the enhanced scanner
            await scanner.start()
            while True:
                try:
                    # Get top opportunities from enhanced scanner
                    opportunities = scanner.get_top_opportunities(limit=20)
                    if opportunities:
                        async with QUEUE_LOCK:
                            for opp in reversed(opportunities):
                                symbol = opp.get("symbol", "")
                                if symbol:
                                    symbol_priority_queue.appendleft(symbol)
                                    logger.info(
                f"Queued enhanced opportunity: {symbol} (score: {opp.get('score', 0):.2f})"
            )
                except asyncio.CancelledError:
                    await scanner.stop()
                    break
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error("Enhanced Solana scan error: %s", exc)
                await asyncio.sleep(interval)
        else:
            # Fallback to basic scanner
            logger.info("Using basic Solana scanner")
            while True:
                try:
                    response = await services.token_discovery.discover_tokens(
                        TokenDiscoveryRequest(config=cfg)
                    )
                    tokens = response.tokens
                    if tokens:
                        async with QUEUE_LOCK:
                            for sym in reversed(tokens):
                                symbol_priority_queue.appendleft(sym)
                except asyncio.CancelledError:
                    break
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error("Solana scan error: %s", exc)
                await asyncio.sleep(interval)
    # Load environment variables early to ensure API credentials are available
    # for WebSocket clients created during market loader configuration
    # Initialize production memory manager
    logger.info("Initializing production memory manager...")
    memory_manager = get_memory_manager({
        "memory_threshold": 0.8,
        "gc_threshold": 0.7,
        "model_cleanup_interval": 300,
        "enable_background_cleanup": True
    })
    logger.info("Loading environment variables from .env file...")
    secrets = dotenv_values(ENV_PATH)
    logger.info(f"Loaded {len(secrets)} environment variables from .env")
    # Debug: Check for API credentials
    api_key = secrets.get('API_KEY')
    api_secret = secrets.get('API_SECRET')
    logger.info(f"API_KEY from .env: {'SET' if api_key else 'NOT SET'}")
    logger.info(f"API_SECRET from .env: {'SET' if api_secret else 'NOT SET'}")
    flat_cfg = flatten_config(config)
    for key, val in secrets.items():
        if key in flat_cfg:
            if flat_cfg[key] != val:
                logger.info(
                    "Overriding %s from .env (config.yaml value: %s)",
                    key,
                    flat_cfg[key],
                )
                # Update the config dictionary with the override value
                config[key.lower()] = val
            else:
                logger.info("Using %s from .env (matches config.yaml)", key)
        else:
            logger.info("Setting %s from .env", key)
            # Add new keys to config as well
            config[key.lower()] = val
    os.environ.update(secrets)
    logger.info("Environment variables updated in os.environ")
    # Debug: Verify environment variables are set
    logger.info(
                f"API_KEY in os.environ: {'SET' if os.getenv('API_KEY') else 'NOT SET'}"
            )
    logger.info(
                f"API_SECRET in os.environ: {'SET' if os.getenv('API_SECRET') else 'NOT SET'}"
            )
    volume_ratio = 0.01 if config.get("testing_mode") else 1.0
    cooldown_configure(config.get("min_cooldown", 0))
    status_updates = config.get("telegram", {}).get("status_updates", True)
    market_loader_configure(
        config.get("ohlcv_timeout", 120),
        config.get("max_ohlcv_failures", 3),
        config.get("max_ws_limit", 50),
        status_updates,
        max_concurrent=config.get("max_concurrent_ohlcv"),
    )
    user = load_or_create()
    trade_updates = config.get("telegram", {}).get("trade_updates", True)
    status_updates = config.get("telegram", {}).get("status_updates", True)
    balance_updates = config.get("telegram", {}).get("balance_updates", False)
    tg_cfg = {**config.get("telegram", {})}
    if user.get("telegram_token"):
        tg_cfg["token"] = user["telegram_token"]
    if user.get("telegram_chat_id"):
        tg_cfg["chat_id"] = user["telegram_chat_id"]
    if os.getenv("TELE_CHAT_ADMINS"):
        tg_cfg["chat_admins"] = os.getenv("TELE_CHAT_ADMINS")
    trade_updates = tg_cfg.get("trade_updates", True)
    status_updates = tg_cfg.get("status_updates", status_updates)
    balance_updates = tg_cfg.get("balance_updates", balance_updates)
    # Allow environment variables to override token/chat_id if provided
    env_token = os.getenv("TELEGRAM_TOKEN")
    env_chat = os.getenv("TELEGRAM_CHAT_ID")
    if env_token:
        tg_cfg["token"] = env_token
    if env_chat:
        tg_cfg["chat_id"] = env_chat
    notifier = TelegramNotifier.from_config(tg_cfg)
    if status_updates and notifier.enabled:
        notifier.notify("🤖 CoinTrader2.0 started")
    # Initialize and start monitoring system
    logger.info("🚀 Initializing monitoring system...")
    try:
        from crypto_bot.pipeline_monitor import PipelineMonitor
        # Create monitoring instance with bot's config
        monitor = PipelineMonitor(config)
        # Start monitoring in background
        logger.info("📊 Starting pipeline monitoring...")
        asyncio.create_task(monitor.start_monitoring())
        # Notify about monitoring startup
        if status_updates and notifier.enabled:
            notifier.notify("✅ Pipeline monitoring system activated")
            notifier.notify("📊 View monitoring dashboard at: http://localhost:8000/monitoring")
        logger.info("✅ Monitoring system started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start monitoring system: {e}")
        if status_updates and notifier.enabled:
            notifier.notify(f"⚠️ Monitoring system failed to start: {e}")
        # Continue without monitoring rather than failing the bot
        monitor = None
    if notifier.token and notifier.chat_id and notifier.enabled:
        if not send_test_message(notifier.token, notifier.chat_id, "Bot started"):
            logger.warning("Telegram test message failed; check your token and chat ID")
    # allow user-configured exchange to override YAML setting
    if user.get("exchange"):
        config["exchange"] = user["exchange"]
    exchange_resp = services.execution.create_exchange(
        ExchangeRequest(config=config)
    )
    exchange = exchange_resp.exchange
    ws_client = exchange_resp.ws_client
    exchange_id = getattr(exchange, "id", "") or str(config.get("exchange", "kraken"))
    ping_interval = int(config.get("ws_ping_interval", 0) or 0)
    if ping_interval > 0 and hasattr(exchange, "ping"):
        task = asyncio.create_task(_ws_ping_loop(exchange, ping_interval))
        WS_PING_TASKS.add(task)
    if not hasattr(exchange, "load_markets"):
        logger.error("The installed ccxt package is missing or a local stub is in use.")
        if status_updates and notifier.enabled:
            notifier.notify(
                "❌ ccxt library not found or stubbed; check your installation"
            )
        # Continue startup even if ccxt is missing for testing environments
    if config.get("scan_markets", False):
        attempt = 0
        delay = SYMBOL_SCAN_RETRY_DELAY
        discovered: Union[list[str], None] = None
        while attempt < MAX_SYMBOL_SCAN_ATTEMPTS:
            start_scan = time.perf_counter()
            response = await services.market_data.load_symbols(
                LoadSymbolsRequest(
                    exchange_id=str(exchange_id),
                    exclude=config.get("excluded_symbols", []),
                    config=config,
                )
            )
            discovered = response.symbols
            latency = time.perf_counter() - start_scan
            services.monitoring.record_scanner_metrics(
                RecordScannerMetricsRequest(
                    tokens=len(discovered or []),
                    latency=latency,
                    config=config,
                )
            )
            if discovered:
                break
            attempt += 1
            if attempt >= MAX_SYMBOL_SCAN_ATTEMPTS:
                break
            logger.warning(
                "Symbol scan empty; retrying in %d seconds (attempt %d/%d)",
                delay,
                attempt + 1,
                MAX_SYMBOL_SCAN_ATTEMPTS,
            )
            if status_updates and notifier.enabled:
                notifier.notify(
                    f"Symbol scan failed; retrying in {delay}s (attempt {attempt + 1}/{MAX_SYMBOL_SCAN_ATTEMPTS})"
                )
            if inspect.iscoroutinefunction(asyncio.sleep):
                await asyncio.sleep(delay)
            else:  # pragma: no cover - compatibility with patched sleep
                asyncio.sleep(delay)
            delay = min(delay * 2, MAX_SYMBOL_SCAN_DELAY)
        if discovered:
            # Combine discovered symbols with configured symbols instead of replacing
            configured_symbols = config.get("symbols", [])
            if configured_symbols:
                # Merge discovered symbols with configured ones, avoiding duplicates
                all_symbols = list(configured_symbols)
                for sym in discovered:
                    if sym not in all_symbols:
                        all_symbols.append(sym)
                config["symbols"] = all_symbols
                logger.info(
                f"Combined {len(configured_symbols)} configured symbols with {len(discovered)} discovered symbols = {len(all_symbols)} total"
            )
                # Persist combined symbols to config file
                try:
                    import yaml
                    with open(CONFIG_PATH, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                    file_config["symbols"] = all_symbols
                    with open(CONFIG_PATH, 'w') as f:
                        yaml.dump(file_config, f, default_flow_style=False, sort_keys=False)
                    logger.info(
                f"Persisted {len(all_symbols)} combined symbols to config file"
            )
                except Exception as e:
                    logger.warning(f"Failed to persist combined symbols to config file: {e}")
            else:
                # No configured symbols, use discovered ones
                config["symbols"] = discovered
                logger.info(f"Using {len(discovered)} discovered symbols")
                # Persist discovered symbols to config file
                try:
                    import yaml
                    with open(CONFIG_PATH, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                    file_config["symbols"] = discovered
                    with open(CONFIG_PATH, 'w') as f:
                        yaml.dump(file_config, f, default_flow_style=False, sort_keys=False)
                    logger.info(
                f"Persisted {len(discovered)} discovered symbols to config file"
            )
                except Exception as e:
                    logger.warning(f"Failed to persist discovered symbols to config file: {e}")
        else:
            logger.error(
                "No symbols discovered after %d attempts; using configured symbols only",
                MAX_SYMBOL_SCAN_ATTEMPTS,
            )
            if status_updates and notifier.enabled:
                notifier.notify(
                    f"⚠️ Symbol scan failed after {MAX_SYMBOL_SCAN_ATTEMPTS} attempts; using configured symbols only"
                )
    balance_threshold = config.get("balance_change_threshold", 0.01)
    previous_balance = 0.0
    def check_balance_change(new_balance: float, reason: str) -> None:
        nonlocal previous_balance
        delta = new_balance - previous_balance
        if abs(delta) > balance_threshold and notifier and notifier.enabled:
            notifier.notify(f"Balance changed by {delta:.4f} USDT due to {reason}")
        previous_balance = new_balance
    try:
        # Skip balance fetch in dry run mode - use paper wallet instead
        if config.get("execution_mode") == "dry_run":
            logger.info("Dry run mode detected - skipping exchange balance fetch")
            init_bal = 0.0
            last_balance = 0.0
            previous_balance = 0.0
        else:
            init_bal = await fetch_balance(exchange, None, config)
            log_balance(float(init_bal))
            last_balance = float(init_bal)
            previous_balance = float(init_bal)
    except Exception as exc:  # pragma: no cover - network
        logger.error("Exchange API setup failed: %s", exc)
        if status_updates and notifier.enabled:
            err = notifier.notify(f"API error: {exc}")
            if err:
                logger.error("Failed to notify user: %s", err)
        return notifier
    risk_params = {**config.get("risk", {})}
    risk_params.update(config.get("sentiment_filter", {}))
    risk_params.update(config.get("volatility_filter", {}))
    risk_params["symbol"] = config.get("symbol", "")
    risk_params["trade_size_pct"] = config.get("trade_size_pct", 0.1)
    risk_params["strategy_allocation"] = config.get("strategy_allocation", {})
    risk_params["volume_threshold_ratio"] = config.get("risk", {}).get(
        "volume_threshold_ratio", 0.1
    )
    risk_params["atr_period"] = config.get("risk", {}).get("atr_period", 14)
    risk_params["stop_loss_atr_mult"] = config.get("risk", {}).get(
        "stop_loss_atr_mult", 2.0
    )
    risk_params["take_profit_atr_mult"] = config.get("risk", {}).get(
        "take_profit_atr_mult", 4.0
    )
    risk_params["volume_ratio"] = volume_ratio
    # Filter out unknown fields that aren't in RiskConfig
    valid_fields = {
        'max_drawdown', 'stop_loss_pct', 'take_profit_pct', 'min_fng', 'min_sentiment',
        'bull_fng', 'bull_sentiment', 'min_atr_pct', 'max_funding_rate', 'symbol',
        'trade_size_pct', 'risk_pct', 'min_volume', 'volume_threshold_ratio',
        'strategy_allocation', 'volume_ratio', 'atr_short_window', 'atr_long_window',
        'max_volatility_factor', 'min_expected_value', 'default_expected_value',
        'atr_period', 'stop_loss_atr_mult', 'take_profit_atr_mult', 'max_pair_drawdown',
        'pair_drawdown_lookback'
    }
    filtered_risk_params = {k: v for k, v in risk_params.items() if k in valid_fields}
    risk_config = RiskConfig(**filtered_risk_params)
    risk_manager = RiskManager(risk_config)
    paper_wallet = None
    if config.get("execution_mode") == "dry_run":
        # Try to read from paper wallet config file first
        # Check environment variable first
        start_bal = 10000.0  # Default fallback
        config_loaded = False
        # Check for environment variable
        env_balance = os.getenv("PAPER_WALLET_BALANCE")
        if env_balance:
            try:
                start_bal = float(env_balance)
                logger.info(
                f"Loaded paper wallet balance from environment variable: ${start_bal:.2f}"
            )
                config_loaded = True
            except ValueError:
                logger.warning(
                f"Invalid PAPER_WALLET_BALANCE environment variable: {env_balance}"
            )
        # If no environment variable, try main config file first
        if not config_loaded and config.get("paper_wallet", {}).get("initial_balance"):
            try:
                start_bal = float(config["paper_wallet"]["initial_balance"])
                logger.info(
                f"Loaded paper wallet balance from main config: ${start_bal:.2f}"
            )
                config_loaded = True
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid paper wallet balance in main config: {e}")
        # If still no config loaded, try dedicated paper wallet config files
        if not config_loaded:
            possible_paths = [
                Path("crypto_bot/paper_wallet_config.yaml"),  # Relative to current directory
                Path(__file__).parent / "paper_wallet_config.yaml",  # Relative to main.py
                Path.cwd() / "crypto_bot" / "paper_wallet_config.yaml",  # Relative to working directory
                Path.home() / "Downloads" / "LegacyCoinTrader1.0" / "crypto_bot" / "paper_wallet_config.yaml",  # Absolute path fallback
                Path("/Users/brandonburnette/Downloads/LegacyCoinTrader1.0/crypto_bot/paper_wallet_config.yaml"),  # Hardcoded fallback
            ]
            for paper_wallet_config_path in possible_paths:
                if paper_wallet_config_path.exists():
                    try:
                        import yaml
                        with open(paper_wallet_config_path, 'r') as f:
                            paper_config = yaml.safe_load(f) or {}
                            start_bal = paper_config.get('initial_balance', 10000.0)
                            logger.info(
                f"Loaded paper wallet balance from config {paper_wallet_config_path}: ${start_bal:.2f}"
            )
                            config_loaded = True
                            break
                    except Exception as e:
                        logger.warning(
                f"Failed to read paper wallet config {paper_wallet_config_path}: {e}"
            )
                        continue
        if not config_loaded:
            # Use default balance instead of prompting for user input on restart
            logger.warning("No paper wallet config found, using default balance of $10,000")
            start_bal = 10000.0
        paper_wallet = PaperWallet(
            start_bal,
            config.get("paper_wallet", {}).get("max_open_trades", config.get("max_open_trades", 1)),
            config.get("paper_wallet", {}).get("allow_short", config.get("allow_short", False)),
        )
        # Don't load state initially - we'll handle this later in BotContext creation
        # This prevents loading old data before we can reset the files
        logger.info(f"Created paper wallet with initial balance: ${start_bal:.2f}")
        # Initialize risk manager equity with current balance
        risk_manager.update_equity(paper_wallet.balance)
        logger.info(
                f"Risk manager equity initialized to: ${paper_wallet.balance:.2f}"
            )
        log_balance(paper_wallet.balance)
        last_balance = notify_balance_change(
            notifier,
            last_balance,
            float(paper_wallet.balance),
            balance_updates,
            is_paper_trading=True,
        )
    monitor_task = asyncio.create_task(
        console_monitor.monitor_loop(exchange, paper_wallet, LOG_DIR / "bot.log")
    )
    position_tasks: dict[str, asyncio.Task] = {}
    max_open_trades = config.get("max_open_trades", 1)
    position_guard = OpenPositionGuard(max_open_trades)
    rotator = PortfolioRotator()
    mode = user.get("mode", config.get("mode", "auto"))
    state = {"running": True, "mode": mode}
    # Caches for OHLCV and regime data are stored on the session_state
    session_state = SessionState(last_balance=last_balance)
    last_candle_ts: dict[str, int] = {}
    control_task = asyncio.create_task(console_control.control_loop(state))
    rotation_task = asyncio.create_task(
        _rotation_loop(
            rotator,
            exchange,
            user.get("wallet_address", ""),
            state,
            notifier,
            check_balance_change,
        )
    )
    solana_scan_task: Union[asyncio.Task, None] = None
    if config.get("solana_scanner", {}).get("enabled"):
        solana_scan_task = asyncio.create_task(solana_scan_loop())
    print("Bot running. Type 'stop' to pause, 'start' to resume, 'quit' to exit.")
    # Temporarily disable Telegram bot completely to fix conflicts
    # from crypto_bot.telegram_bot_ui import TelegramBotUI
    # telegram_bot = (
    #     TelegramBotUI(
    #         notifier,
    #         state,
    #         LOG_DIR / "bot.log",
    #         rotator,
    #         exchange,
    #         user.get("wallet_address", ""),
    #         command_cooldown=config.get("telegram", {}).get("command_cooldown", 5),
    #         paper_wallet=paper_wallet,
    #     )
    #     if notifier.enabled
    #     else None
    # )
    # if telegram_bot:
    #     telegram_bot.run_async()
    telegram_bot = None
    # Legacy meme wave sniper (keeping for backward compatibility)
    meme_wave_task = None
    if config.get("meme_wave_sniper", {}).get("enabled"):
        from crypto_bot.solana import start_runner
        # Add paper trading parameters to meme wave config
        meme_wave_cfg = config.get("meme_wave_sniper", {}).copy()
        meme_wave_cfg["execution"] = meme_wave_cfg.get("execution", {})
        meme_wave_cfg["execution"]["dry_run"] = config.get("execution_mode") == "dry_run"
        meme_wave_cfg["execution"]["paper_wallet"] = paper_wallet if config.get("execution_mode") == "dry_run" else None
        meme_wave_task = start_runner(meme_wave_cfg)
    sniper_cfg = config.get("meme_wave_sniper", {})
    sniper_task = None
    if sniper_cfg.get("enabled"):
        from crypto_bot.solana.runner import run as sniper_run
        sniper_task = asyncio.create_task(sniper_run(sniper_cfg))
    # Advanced Pump Sniper System
    pump_sniper_task = None
    if config.get("pump_sniper_orchestrator", {}).get("enabled"):
        from crypto_bot.solana.pump_sniper_integration import start_pump_sniper_system
        try:
            pump_sniper_started = await start_pump_sniper_system(
                config,
                dry_run=config.get("execution_mode") == "dry_run",
                paper_wallet=paper_wallet if config.get("execution_mode") == "dry_run" else None
            )
            if pump_sniper_started:
                logger.info("Advanced Pump Sniper System started successfully")
            else:
                logger.warning("Advanced Pump Sniper System failed to start")
        except Exception as exc:
            logger.error(f"Failed to start Advanced Pump Sniper System: {exc}")
    if config.get("scan_in_background", True):
        session_state.scan_task = asyncio.create_task(
            initial_scan(
                exchange,
                config,
                session_state,
                services,
                notifier if status_updates else None,
            )
        )
    else:
        await initial_scan(
            exchange,
            config,
            session_state,
            services,
            notifier if status_updates else None,
        )
    ctx = BotContext(
        positions=session_state.positions,
        df_cache=session_state.df_cache,
        regime_cache=session_state.regime_cache,
        config=config,
        services=services,
        rng=rng,
        numpy_rng=numpy_rng,
    )
    ctx.exchange = exchange
    ctx.ws_client = ws_client
    ctx.risk_manager = risk_manager
    ctx.notifier = notifier
    ctx.paper_wallet = paper_wallet
    # Initialize TradeManager and sync with paper wallet
    try:
        from crypto_bot.utils.price_monitor import start_price_monitoring
        tm = services.portfolio.get_trade_manager()
        # Start price monitoring service for real-time price updates
        ctx.price_monitor = start_price_monitoring(exchange, tm)
        logger.info("Price monitoring service started")
        # Sync paper wallet balance with TradeManager
        if ctx.paper_wallet:
            # Update paper wallet from TradeManager positions
            tm_positions = tm.get_all_positions()
            if tm_positions:
                # Calculate current portfolio value from TradeManager
                portfolio_value = ctx.paper_wallet.initial_balance
                for pos in tm_positions:
                    current_price = tm.price_cache.get(pos.symbol, pos.average_price)
                    if pos.side == 'long':
                        portfolio_value += (current_price - pos.average_price) * pos.total_amount
                    else:  # short
                        portfolio_value += (pos.average_price - current_price) * pos.total_amount
                    portfolio_value -= pos.fees_paid
                ctx.paper_wallet.balance = portfolio_value
                ctx.balance = portfolio_value
                logger.info(
                f"Synchronized paper wallet balance with TradeManager: ${portfolio_value:.2f}"
            )
            else:
                # No positions, use initial balance
                ctx.paper_wallet.balance = ctx.paper_wallet.initial_balance
                ctx.balance = ctx.paper_wallet.balance
        logger.info("✅ TradeManager synchronization completed")
    except Exception as e:
        logger.warning(f"Could not synchronize with TradeManager: {e}")
    ctx.position_guard = position_guard
    # Initialize TradeManager if not already available
    if not hasattr(ctx, 'trade_manager') or ctx.trade_manager is None:
        ctx.trade_manager = services.portfolio.get_trade_manager()
        logger.info("Initialized TradeManager in context for position monitor")
    # Initialize real-time position monitor
    # Convert TradeManager positions to the format expected by PositionMonitor
    tm_positions = {}
    for pos in ctx.trade_manager.get_all_positions():
        tm_positions[pos.symbol] = {
            'symbol': pos.symbol,
            'side': 'long' if pos.side == 'long' else 'short',
            'amount': float(pos.total_amount),
            'entry_price': float(pos.average_price),
            'current_price': float(ctx.trade_manager.price_cache.get(pos.symbol, pos.average_price)),
            'pnl': 0.0,  # Will be updated by position monitor
            'highest_price': float(pos.highest_price) if pos.highest_price else float(pos.average_price),
            'lowest_price': float(pos.lowest_price) if pos.lowest_price else float(pos.average_price),
            'stop_loss_price': float(pos.stop_loss_price) if pos.stop_loss_price else None,
            'take_profit_price': float(pos.take_profit_price) if pos.take_profit_price else None,
        }
    ctx.position_monitor = PositionMonitor(
        exchange=exchange,
        config=config,
        positions=tm_positions,  # Use TradeManager positions instead of session_state
        notifier=notifier,
        trade_manager=ctx.trade_manager  # Pass TradeManager for centralized calculations
    )
    # Set up exit callback for position monitor
    async def handle_position_exit(symbol: str, position: dict, current_price: float, exit_reason: str) -> None:
        """Handle position exit triggered by position monitor."""
        logger.info(
                f"🚨 Position monitor triggered exit for {symbol}: {exit_reason} at {current_price:.6f}"
            )
        if ctx.services is None:
            raise RuntimeError("Execution service unavailable in context")
        try:
            # Execute the exit trade
            await ctx.services.execution.execute_trade(
                TradeExecutionRequest(
                    exchange=ctx.exchange,
                    ws_client=ctx.ws_client,
                    symbol=symbol,
                    side=opposite_side(position["side"]),
                    amount=position["size"],
                    notifier=ctx.notifier,
                    dry_run=ctx.config.get("execution_mode") == "dry_run",
                    use_websocket=ctx.config.get("use_websocket", False),
                    config=ctx.config,
                )
            )
            # Handle paper wallet if in dry run mode
            if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                try:
                    pnl = ctx.paper_wallet.close(symbol, position["size"], current_price)
                    ctx.balance = ctx.paper_wallet.balance
                    logger.info(
                f"Paper trade closed: {position['side']} {position['size']} {symbol} @ ${current_price:.6f}, PnL: ${pnl:.2f}, balance: ${ctx.balance:.2f}"
            )
                    # Send Telegram notification for paper trade exit
                    if ctx.notifier and ctx.config.get("telegram", {}).get("trade_updates", True) and ctx.notifier.enabled:
                        pnl_emoji = "💰" if pnl >= 0 else "📉"
                        paper_exit_msg = f"📄 Paper Trade Closed {pnl_emoji}\n{position['side'].upper()} {position['size']} {symbol}\nEntry: ${position['entry_price']:.6f}\nExit: ${current_price:.6f}\nPnL: ${pnl:.2f}\nBalance: ${ctx.balance:.2f}\nReason: {exit_reason}"
                        ctx.notifier.notify(paper_exit_msg)
                except Exception as e:
                    logger.error(f"Failed to close paper trade: {e}")
            # Handle TradeManager position closure first (for dashboard sync)
            if hasattr(ctx, 'trade_manager') and ctx.trade_manager:
                try:
                    # Check if TradeManager has this position
                    tm_position = ctx.trade_manager.get_position(symbol)
                    if tm_position and tm_position.is_open:
                        logger.info(f"Closing TradeManager position for {symbol} via {exit_reason}")
                        closing_side = 'sell' if tm_position.side == 'long' else 'buy'
                        closing_trade_response = ctx.services.portfolio.create_trade(
                            CreateTradeRequest(
                                symbol=symbol,
                                side=closing_side,
                                amount=Decimal(str(position["size"])),
                                price=Decimal(str(current_price)),
                                strategy=exit_reason,  # Use exit_reason as strategy
                                exchange='kraken',
                            )
                        )
                        closing_trade = closing_trade_response.trade
                        # Record the closing trade
                        trade_id = ctx.trade_manager.record_trade(closing_trade)
                        logger.info(f"TradeManager position closed for {symbol} with trade ID: {trade_id}")
                        # Save TradeManager state
                        ctx.trade_manager.save_state()
                        # Sync positions if needed
                        if hasattr(ctx, 'sync_positions_from_trade_manager') and ctx.use_trade_manager_as_source:
                            ctx.sync_positions_from_trade_manager()
                except Exception as e:
                    logger.error(f"Failed to close TradeManager position for {symbol}: {e}")

            # Update legacy position tracking
            if symbol in ctx.positions:
                pos = ctx.positions[symbol]
                if ctx.paper_wallet and symbol in ctx.paper_wallet.positions:
                    final_pnl = ctx.paper_wallet.unrealized(symbol, current_price)
                    pos["pnl"] = final_pnl
                    logger.info(f"Final PnL for {symbol}: ${final_pnl:.2f}")
            # Clean up position tracking
            sync_paper_wallet_balance(ctx)
            if hasattr(ctx, 'position_monitor'):
                await ctx.position_monitor.stop_monitoring(symbol)
            ctx.risk_manager.deallocate_capital(
                position.get("strategy", ""), position["size"] * position["entry_price"]
            )
            ctx.positions.pop(symbol, None)
            # Log the exit
            try:
                log_position(
                    symbol,
                    position["side"],
                    position["size"],
                    position["entry_price"],
                    current_price,
                    ctx.balance,
                    exit_reason=exit_reason
                )
                log_pnl(
                    position.get("strategy", "unknown"),
                    symbol,
                    position["entry_price"],
                    current_price,
                    (current_price - position["entry_price"]) * (1 if position["side"] == "buy" else -1),
                    0.0,  # confidence
                    position["side"]
                )
            except Exception as e:
                logger.error(f"Failed to log position exit: {e}")
        except Exception as e:
            logger.error(f"Error handling position monitor exit for {symbol}: {e}")
    # Set the exit callback
    ctx.position_monitor.on_exit_triggered = handle_position_exit

    # Early position monitoring startup - ensure monitoring starts before main loop
    logger.info("🚨 Performing early position monitoring startup...")
    try:
        # Get all positions from TradeManager for early monitoring
        tm_positions = []
        if hasattr(ctx, 'trade_manager') and ctx.trade_manager:
            tm_positions = ctx.trade_manager.get_all_positions()

        if tm_positions:
            logger.info(f"Starting early monitoring for {len(tm_positions)} TradeManager positions")
            for tm_pos in tm_positions:
                position_dict = {
                    'entry_price': float(tm_pos.average_price),
                    'size': float(tm_pos.total_amount),
                    'side': tm_pos.side,
                    'symbol': tm_pos.symbol,
                    'highest_price': float(tm_pos.highest_price) if tm_pos.highest_price else float(tm_pos.average_price),
                    'lowest_price': float(tm_pos.lowest_price) if tm_pos.lowest_price else float(tm_pos.average_price),
                    'trailing_stop': float(tm_pos.stop_loss_price) if tm_pos.stop_loss_price else 0.0,
                    'timestamp': tm_pos.entry_time.isoformat(),
                }
                await ctx.position_monitor.start_monitoring(tm_pos.symbol, position_dict)
            logger.info("✅ Early position monitoring startup completed")
        else:
            logger.info("ℹ️ No positions found for early monitoring startup")
    except Exception as e:
        logger.warning(f"⚠️ Early position monitoring startup failed: {e}")

    # Initialize evaluation pipeline integration
    logger.info("🔗 Initializing evaluation pipeline integration...")
    try:
        pipeline_initialized = await initialize_evaluation_pipeline(config)
        if pipeline_initialized:
            logger.info("✅ Evaluation pipeline integration initialized successfully")
            if status_updates and notifier.enabled:
                notifier.notify("🚀 Evaluation pipeline integration activated")
        else:
            logger.warning("⚠️ Evaluation pipeline integration failed to initialize")
            if status_updates and notifier.enabled:
                notifier.notify("⚠️ Evaluation pipeline integration failed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize evaluation pipeline: {e}")
        if status_updates and notifier.enabled:
            notifier.notify(f"❌ Pipeline initialization error: {e}")
    # Start enhanced scanning integration if enabled
    try:
        if config.get("enhanced_scanning", {}).get("enabled", False):
            await start_enhanced_scan_integration(config, notifier)
            logger.info("✅ Enhanced scanning integration started")
            if status_updates and notifier.enabled:
                notifier.notify("🔍 Enhanced scanning integration started")
    except Exception as e:
        logger.error(f"❌ Failed to start enhanced scanning integration: {e}")
    async def monitor_positions_phase(ctx: BotContext) -> None:
        """Monitor positions for stop loss triggers."""
        if hasattr(ctx, 'position_monitor'):
            active_count = len(ctx.position_monitor.active_monitors) if hasattr(ctx.position_monitor, 'active_monitors') else 0

            if active_count > 0:
                logger.debug(f"PositionMonitor actively monitoring {active_count} positions")

                # Get monitoring stats if available
                try:
                    if hasattr(ctx.position_monitor, 'get_monitoring_stats'):
                        stats = await ctx.position_monitor.get_monitoring_stats()
                        if stats and stats.get('price_updates', 0) > 0:
                            logger.debug(f"Position monitoring stats: {stats.get('price_updates', 0)} price updates, {stats.get('trailing_stop_triggers', 0)} triggers")
                except Exception as e:
                    logger.debug(f"Could not get monitoring stats: {e}")

                # Check if we need to restart monitoring for any positions that might be missing
                if hasattr(ctx, 'trade_manager') and ctx.trade_manager:
                    tm_positions = ctx.trade_manager.get_all_positions()
                    tm_symbols = {pos.symbol for pos in tm_positions}

                    # Check for TradeManager positions not being monitored
                    for tm_pos in tm_positions:
                        if hasattr(ctx.position_monitor, 'active_monitors') and tm_pos.symbol not in ctx.position_monitor.active_monitors:
                            logger.warning(f"Position {tm_pos.symbol} not being monitored - restarting monitoring")
                            try:
                                position_dict = {
                                    'entry_price': float(tm_pos.average_price),
                                    'size': float(tm_pos.total_amount),
                                    'side': tm_pos.side,
                                    'symbol': tm_pos.symbol,
                                    'highest_price': float(tm_pos.highest_price) if tm_pos.highest_price else float(tm_pos.average_price),
                                    'lowest_price': float(tm_pos.lowest_price) if tm_pos.lowest_price else float(tm_pos.average_price),
                                    'trailing_stop': float(tm_pos.stop_loss_price) if tm_pos.stop_loss_price else 0.0,
                                    'timestamp': tm_pos.entry_time.isoformat(),
                                }
                                await ctx.position_monitor.start_monitoring(tm_pos.symbol, position_dict)
                                logger.info(f"Restarted monitoring for {tm_pos.symbol}")
                            except Exception as e:
                                logger.error(f"Failed to restart monitoring for {tm_pos.symbol}: {e}")
            else:
                logger.warning("PositionMonitor has no active monitors - attempting to restart")
                # Try to restart monitoring for all positions
                try:
                    tm_positions = []
                    if hasattr(ctx, 'trade_manager') and ctx.trade_manager:
                        tm_positions = ctx.trade_manager.get_all_positions()

                    if tm_positions:
                        logger.info(f"Restarting position monitoring for {len(tm_positions)} positions")
                        for tm_pos in tm_positions:
                            position_dict = {
                                'entry_price': float(tm_pos.average_price),
                                'size': float(tm_pos.total_amount),
                                'side': tm_pos.side,
                                'symbol': tm_pos.symbol,
                                'highest_price': float(tm_pos.highest_price) if tm_pos.highest_price else float(tm_pos.average_price),
                                'lowest_price': float(tm_pos.lowest_price) if tm_pos.lowest_price else float(tm_pos.average_price),
                                'trailing_stop': float(tm_pos.stop_loss_price) if tm_pos.stop_loss_price else 0.0,
                                'timestamp': tm_pos.entry_time.isoformat(),
                            }
                            await ctx.position_monitor.start_monitoring(tm_pos.symbol, position_dict)
                        logger.info("Position monitoring restarted successfully")
                except Exception as e:
                    logger.error(f"Failed to restart position monitoring: {e}")
    runner = PhaseRunner(
        [
            fetch_candidates,
            process_solana_candidates,
            update_caches,
            enrich_with_pyth,
            analyse_batch,
            execute_signals,
            handle_exits,
            monitor_positions_phase,
        ],
        clock=clock,
        timer=timer,
        rng=rng,
        numpy_rng=numpy_rng,
    )
    # Start PositionMonitor for real-time stop loss monitoring
    if hasattr(ctx, 'position_monitor'):
        logger.info("🚨 Starting PositionMonitor for real-time stop loss monitoring...")
        try:
            # Get all positions from TradeManager (single source of truth)
            tm_positions = []
            if hasattr(ctx, 'trade_manager') and ctx.trade_manager:
                tm_positions = ctx.trade_manager.get_all_positions()

            # Start monitoring TradeManager positions
            monitored_count = 0
            for tm_pos in tm_positions:
                # Convert TradeManager position to dict format for PositionMonitor
                position_dict = {
                    'entry_price': float(tm_pos.average_price),
                    'size': float(tm_pos.total_amount),
                    'side': tm_pos.side,
                    'symbol': tm_pos.symbol,
                    'highest_price': float(tm_pos.highest_price) if tm_pos.highest_price else float(tm_pos.average_price),
                    'lowest_price': float(tm_pos.lowest_price) if tm_pos.lowest_price else float(tm_pos.average_price),
                    'trailing_stop': float(tm_pos.stop_loss_price) if tm_pos.stop_loss_price else 0.0,
                    'timestamp': tm_pos.entry_time.isoformat(),
                }

                await ctx.position_monitor.start_monitoring(tm_pos.symbol, position_dict)
                logger.info(f"Started monitoring {tm_pos.symbol} ({tm_pos.side} {tm_pos.total_amount})")
                monitored_count += 1

            # Also start monitoring any positions in ctx.positions that aren't already monitored
            for symbol, position in ctx.positions.items():
                if not hasattr(ctx.position_monitor, 'active_monitors') or symbol not in ctx.position_monitor.active_monitors:
                    await ctx.position_monitor.start_monitoring(symbol, position)
                    logger.info(f"Started monitoring legacy position {symbol}")
                    monitored_count += 1

            logger.info(f"✅ PositionMonitor started successfully - monitoring {monitored_count} positions")
            if status_updates and notifier.enabled:
                notifier.notify(f"🚨 Real-time stop loss monitoring activated for {monitored_count} positions")
        except Exception as e:
            logger.error(f"❌ Failed to start PositionMonitor: {e}")
            if status_updates and notifier.enabled:
                notifier.notify(f"⚠️ Stop loss monitoring failed to start: {e}")
    loop_count = 0
    last_weight_update = last_optimize = 0.0
    try:
        while True:
            maybe_reload_config(state, config)
            reload_config(
                config,
                ctx,
                risk_manager,
                rotator,
                position_guard,
                force=state.get("reload", False),
            )
            state["reload"] = False
            if state.get("liquidate_all"):
                await force_exit_all(ctx)
                state["liquidate_all"] = False
            if config.get("arbitrage_enabled", True):
                try:
                    arb_syms = await scan_arbitrage(exchange, config)
                    if arb_syms:
                        async with QUEUE_LOCK:
                            for sym in reversed(arb_syms):
                                symbol_priority_queue.appendleft(sym)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error("Arbitrage scan error: %s", exc)
            cycle_start = time.perf_counter()
            logger.info("=== Starting trading cycle %d ===", loop_count)
            try:
                ctx.timing = await runner.run(ctx)
                logger.info("Trading cycle completed. Timing: %s", ctx.timing)
            except Exception as exc:
                logger.error("Error during cycle execution: %s", exc, exc_info=True)
                # Continue with a minimal timing object to prevent further errors
                ctx.timing = {"fetch_candidates": 0.0, "update_caches": 0.0, "analyse_batch": 0.0}
            # Clean up any completed trading tasks from previous cycle
            try:
                await trade_manager.cleanup_completed()
            except Exception as exc:
                logger.error("Error during trading task cleanup: %s", exc)
            # Process any pending sell requests from the frontend
            try:
                await process_sell_requests(ctx, notifier)
            except Exception as exc:
                logger.error("Error processing sell requests: %s", exc)
            # Synchronize paper wallet with positions.log to prevent desynchronization
            try:
                await sync_paper_wallet_with_positions_log(ctx)
            except Exception as exc:
                logger.error("Error synchronizing paper wallet: %s", exc)
            # Update PnL for all positions after each trading cycle
            try:
                update_position_pnl(ctx)
            except Exception as exc:
                logger.error("Error updating position PnL: %s", exc)
            loop_count += 1
            # Add cycle delay to prevent infinite loops and allow time for data processing
            cycle_delay = config.get("cycle_delay_seconds", 30)  # Default 30 seconds between cycles
            cycle_elapsed = time.perf_counter() - cycle_start
            if cycle_elapsed < cycle_delay:
                sleep_time = cycle_delay - cycle_elapsed
                logger.info(
                f"Cycle completed in {cycle_elapsed:.2f}s, sleeping for {sleep_time:.2f}s before next cycle"
            )
                await asyncio.sleep(sleep_time)
            else:
                logger.info(
                f"Cycle completed in {cycle_elapsed:.2f}s, proceeding to next cycle"
            )
            try:
                if time.time() - last_weight_update >= 86400:
                    weights = compute_strategy_weights()
                    if weights:
                        logger.info("Updating strategy allocation to %s", weights)
                        risk_manager.update_allocation(weights)
                        config["strategy_allocation"] = weights
                    last_weight_update = time.time()
            except Exception as exc:
                logger.error("Error updating strategy weights: %s", exc)
            try:
                if config.get("optimization", {}).get("enabled"):
                    if (
                        time.time() - last_optimize
                        >= config["optimization"].get("interval_days", 7) * 86400
                    ):
                        optimize_strategies()
                        last_optimize = time.time()
            except Exception as exc:
                logger.error("Error during strategy optimization: %s", exc)
            if not state.get("running"):
                await asyncio.sleep(1)
                continue
            try:
                balances = await asyncio.to_thread(
                    check_wallet_balances, user.get("wallet_address", "")
                )
                for token in detect_non_trade_tokens(balances):
                    amount = balances[token]
                    logger.info("Converting %s %s to USDC", amount, token)
                    await auto_convert_funds(
                        user.get("wallet_address", ""),
                        token,
                        "USDC",
                        amount,
                        dry_run=config["execution_mode"] == "dry_run",
                        slippage_bps=config.get("solana_slippage_bps", 50),
                        notifier=notifier,
                        paper_wallet=paper_wallet if config["execution_mode"] == "dry_run" else None,
                    )
            except Exception as exc:
                logger.error("Error during wallet balance checking: %s", exc)
                if asyncio.iscoroutinefunction(
                    getattr(exchange, "fetch_balance", None)
                ):
                    bal = await exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(exchange.fetch_balance_with_retry)
                bal_val = (
                    bal.get("USDT", {}).get("free", 0)
                    if isinstance(bal.get("USDT"), dict)
                    else bal.get("USDT", 0)
                )
                check_balance_change(float(bal_val), "funds converted")
            # Refresh OHLCV for open positions if a new candle has formed
            try:
                tf = config.get("timeframe", "1h")
                tf_sec = _timeframe_to_seconds(tf)
                open_syms: list[str] = []
                for sym in ctx.positions:
                    last_ts = last_candle_ts.get(sym, 0)
                    if time.time() - last_ts >= tf_sec:
                        open_syms.append(sym)
                # Ensure paper wallet balance is synchronized
                if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                    ensure_paper_wallet_sync(ctx)
                # Log comprehensive wallet status every cycle
                if loop_count % 10 == 0:  # Log every 10 cycles
                    status = get_paper_wallet_status(ctx)
                    if status:
                        logger.info(f"Cycle {loop_count} - Paper wallet summary: {status}")
                    # Log position monitoring statistics
                    if hasattr(ctx, 'position_monitor'):
                        monitor_stats = await ctx.position_monitor.get_monitoring_stats()
                        logger.info(f"Cycle {loop_count} - Position monitoring: {monitor_stats}")
                if open_syms:
                    tf_cache = ctx.df_cache.get(tf, {})
                    # Check if enhanced fetcher is enabled in config
                    use_enhanced_fetcher = config.get("use_enhanced_ohlcv_fetcher", False)
                    if use_enhanced_fetcher:
                        # Use enhanced fetcher for open positions
                        from crypto_bot.utils.enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher
                        enhanced_fetcher = EnhancedOHLCVFetcher(exchange, config)
                        tf_cache = await enhanced_fetcher.update_cache(
                            tf_cache,
                            open_syms,
                            tf,
                            limit=2,
                            since_map={}
                        )
                    else:
                        # Use legacy fetcher for open positions
                        response = await ctx.services.market_data.update_ohlcv_cache(
                            OHLCVCacheRequest(
                                exchange_id=str(exchange_id),
                                cache=tf_cache,
                                symbols=open_syms,
                                timeframe=tf,
                                limit=2,
                                use_websocket=False,
                                max_concurrent=config.get("max_concurrent_ohlcv", 3),
                                config=config,
                            )
                        )
                        tf_cache = response.cache
                    ctx.df_cache[tf] = tf_cache
                    session_state.df_cache[tf] = tf_cache
                    for sym in open_syms:
                        df = tf_cache.get(sym)
                        if df is not None and not df.empty:
                            last_candle_ts[sym] = int(df["timestamp"].iloc[-1])
                            higher_df = ctx.df_cache.get(
                                config.get("higher_timeframe", "1d"), {}
                            ).get(sym)
                            regime, _ = await classify_regime_async(df, higher_df)
                            ctx.positions[sym]["regime"] = regime
                            TOTAL_ANALYSES += 1
                            if regime == "unknown":
                                UNKNOWN_COUNT += 1
            except Exception as exc:
                logger.error("Error during position refresh: %s", exc)
            total_time = time.perf_counter() - cycle_start
            timing = getattr(ctx, "timing", {})
            _emit_timing(
                timing.get("fetch_candidates", 0.0),
                timing.get("update_caches", 0.0),
                timing.get("analyse_batch", 0.0),
                total_time,
                metrics_path,
                timing.get("ohlcv_fetch_latency", 0.0),
                timing.get("execution_latency", 0.0),
            )
            if config.get("metrics_enabled") and config.get("metrics_backend") == "csv":
                metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "ticker_fetch_time": timing.get("fetch_candidates", 0.0),
                    "symbol_filter_ratio": timing.get("symbol_filter_ratio", 1.0),
                    "ohlcv_fetch_latency": timing.get("ohlcv_fetch_latency", 0.0),
                    "execution_latency": timing.get("execution_latency", 0.0),
                    "unknown_regimes": sum(
                        1
                        for r in getattr(ctx, "analysis_results", [])
                        if r.get("regime") == "unknown"
                    ),
                }
                write_cycle_metrics(metrics, config)
            unknown_rate = UNKNOWN_COUNT / max(TOTAL_ANALYSES, 1)
            if unknown_rate > 0.2 and ctx.notifier and ctx.notifier.enabled:
                ctx.notifier.notify(f"⚠️ Unknown regime rate {unknown_rate:.1%}")
            # Adaptive loop interval based on volatility and active positions
            base_interval = config.get("loop_interval_minutes", 0.5)
            # Speed up if we have active positions that need monitoring
            if hasattr(ctx, 'position_monitor') and ctx.position_monitor.active_monitors:
                active_positions_factor = 0.5  # 2x faster when monitoring positions
            else:
                active_positions_factor = 1.0
            # Adjust based on volatility
            volatility_factor = max(ctx.volatility_factor, 1e-6)
            # Final delay calculation
            delay = (base_interval * active_positions_factor) / volatility_factor
            # Ensure minimum and maximum bounds
            delay = max(0.1, min(delay, 5.0))  # Between 6 seconds and 5 minutes
            logger.info("Sleeping for %.2f minutes (volatility: %.2f, active positions: %d)", 
                       delay, ctx.volatility_factor, 
                       len(ctx.position_monitor.active_monitors) if hasattr(ctx, 'position_monitor') else 0)
            await asyncio.sleep(delay * 60)
    finally:
        if hasattr(exchange, "close"):
            if asyncio.iscoroutinefunction(getattr(exchange, "close")):
                with contextlib.suppress(Exception):
                    await exchange.close()
            else:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(exchange.close)
        if solana_scan_task:
            solana_scan_task.cancel()
            try:
                await solana_scan_task
            except asyncio.CancelledError:
                pass
        if session_state.scan_task:
            session_state.scan_task.cancel()
            try:
                await session_state.scan_task
            except asyncio.CancelledError:
                pass
        monitor_task.cancel()
        control_task.cancel()
        rotation_task.cancel()
        if sniper_task:
            sniper_task.cancel()
        for task in list(position_tasks.values()):
            task.cancel()
        for task in list(position_tasks.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
        position_tasks.clear()
        if meme_wave_task:
            meme_wave_task.cancel()
            try:
                await meme_wave_task
            except asyncio.CancelledError:
                pass
        # Stop all position monitoring
        if hasattr(ctx, 'position_monitor'):
            await ctx.position_monitor.stop_all_monitoring()
        # Cancel and cleanup any remaining trading tasks
        trade_manager.cancel_all()
        try:
            await trade_manager.wait_for_completion(timeout=5.0)
        except Exception as e:
            logger.warning(f"Error during trading task cleanup: {e}")
        if telegram_bot:
            telegram_bot.stop()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        try:
            await rotation_task
        except asyncio.CancelledError:
            pass
        # Stop enhanced scanning integration
        try:
            await stop_enhanced_scan_integration()
            logger.info("✅ Enhanced scanning integration stopped")
        except Exception as e:
            logger.warning(f"Failed to stop enhanced scanning integration: {e}")
        if sniper_task:
            try:
                await sniper_task
            except asyncio.CancelledError:
                pass
        try:
            await control_task
        except asyncio.CancelledError:
                pass
        for task in list(WS_PING_TASKS):
            task.cancel()
        for task in list(WS_PING_TASKS):
            try:
                await task
            except asyncio.CancelledError:
                pass
        WS_PING_TASKS.clear()
    return notifier


async def run_bot(
    *,
    clock: Optional[Callable[[], datetime]] = None,
    timer: Optional[Callable[[], float]] = None,
    rng: Optional[random.Random] = None,
    numpy_rng: Optional[Any] = None,
) -> TelegramNotifier:
    """Public entry point for running the bot with injectable dependencies."""

    return await _main_impl(clock=clock, timer=timer, rng=rng, numpy_rng=numpy_rng)
async def main() -> None:
    """Entry point for running the trading bot with error handling."""
    bot_pid_file = Path("bot_pid.txt")
    # Check for existing instance
    if check_existing_instance(bot_pid_file):
        logger.error("Another bot instance is already running! Please stop it first or remove bot_pid.txt if it's stale.")
        logger.error("To stop the existing bot, you can:")
        logger.error("1. Use the Telegram bot's /stop command")
        logger.error("2. Kill the process manually: pkill -f 'crypto_bot'")
        logger.error("3. Remove the PID file if you're sure no bot is running: rm bot_pid.txt")
        sys.exit(1)
    # Write our PID file
    write_pid_file(bot_pid_file)
    # Set up signal handlers for cleanup
    install_signal_handlers(bot_pid_file)
    notifier: Union[TelegramNotifier, None] = None
    try:
        notifier = await run_bot()
    except Exception as exc:  # pragma: no cover - error path
        logger.exception("Unhandled error in main: %s", exc)
        # Check if it's a rate limiting error
        if "Too many requests" in str(exc) or "rate limit" in str(exc).lower():
            logger.error("Rate limiting detected - this is expected with aggressive scanning")
            if notifier and notifier.enabled:
                notifier.notify("⚠️ Rate limiting detected - consider increasing loop_interval_minutes")
            # Add a delay before restarting to respect rate limits
            logger.info("Waiting 30 seconds before restarting to respect rate limits...")
            await asyncio.sleep(30)
        elif "EGeneral" in str(exc):
            logger.error("Kraken API error detected - this may be temporary")
            if notifier and notifier.enabled:
                notifier.notify("⚠️ Kraken API error - this may be temporary")
            # Add a delay for API errors too
            logger.info("Waiting 15 seconds before restarting due to API error...")
            await asyncio.sleep(15)
        if notifier and notifier.enabled:
            notifier.notify(f"❌ Bot stopped: {exc}")
    finally:
        # Stop price monitoring service
        try:
            from crypto_bot.utils.price_monitor import stop_price_monitoring
            stop_price_monitoring()
            logger.info("Price monitoring service stopped")
        except Exception as e:
            logger.error(f"Error stopping price monitoring: {e}")
        if notifier and notifier.enabled:
            notifier.notify("Bot shutting down")
        logger.info("Bot shutting down")
        cleanup_pid_file(bot_pid_file)
if __name__ == "__main__":  # pragma: no cover - manual execution
    asyncio.run(main())
