from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


from crypto_bot.config import load_config as load_bot_config, resolve_config_path
from crypto_bot.utils.symbol_utils import fix_symbol

from crypto_bot.backtest.backtest_runner import BacktestRunner, BacktestConfig
from crypto_bot.utils.logger import LOG_DIR, setup_logger

CONFIG_PATH = resolve_config_path()
LOG_FILE = LOG_DIR / "optimized_params.json"

logger = setup_logger(__name__, LOG_DIR / "optimizer.log")


def _load_config() -> dict:
    data = load_bot_config(CONFIG_PATH)
    if "symbol" in data:
        data["symbol"] = fix_symbol(data["symbol"])
    if "symbols" in data:
        data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]
    return data


def optimize_strategies() -> Dict[str, Dict[str, float]]:
    """Run backtests for each configured strategy and store best params."""
    cfg = _load_config().get("optimization", {})
    if not cfg.get("enabled"):
        logger.info("Optimization disabled")
        return {}

    param_ranges = cfg.get("parameter_ranges", {})
    bot_cfg = _load_config()
    results: Dict[str, Dict[str, float]] = {}

    # Check for required dependencies
    try:
        import skopt  # noqa: F401
        import joblib  # noqa: F401
        from tqdm import tqdm  # noqa: F401
        logger.info("All optimization dependencies are available")
    except ImportError as e:
        logger.error(f"Missing optimization dependencies: {e}. Please install scikit-optimize, joblib, and tqdm.")
        return {}

    # Initialize exchange based on config using get_exchange for nonce improvements
    exchange = None
    try:
        from crypto_bot.execution.cex_executor import get_exchange
        exchange, _ = get_exchange(bot_cfg)
        logger.info(f"Using {exchange.id} exchange for backtesting with nonce improvements")
    except Exception as e:
        logger.error(f"Failed to initialize exchange: {e}")
        # Continue without exchange - will use simulated data
        exchange = None

    for name, ranges in param_ranges.items():
        sl_range: Iterable[float] = ranges.get("stop_loss", [])
        tp_range: Iterable[float] = ranges.get("take_profit", [])

        if not sl_range or not tp_range:
            logger.warning(f"Skipping {name}: missing stop_loss or take_profit ranges")
            continue

        config = BacktestConfig(
            symbol=bot_cfg.get("symbol", "BTC/USD"),
            timeframe=bot_cfg.get("timeframe", "1h"),
            since=0,
            limit=1000,
            mode=bot_cfg.get("mode", "cex"),
            stop_loss_range=sl_range,
            take_profit_range=tp_range,
        )

        try:
            logger.info(f"Starting optimization for {name} with {len(sl_range)}x{len(tp_range)} parameter combinations")
            runner = BacktestRunner(config, exchange=exchange)
            df = runner.run_grid()
            logger.info(f"Completed optimization for {name} - got {len(df)} results")
        except ImportError as e:
            logger.error(f"Missing dependency for backtesting {name}: {e}")
            continue
        except Exception as exc:
            logger.error(f"Backtest failed for {name}: {exc}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

        if df.empty:
            logger.warning(f"No results from backtest for {name}")
            continue

        df = df.sort_values(["sharpe", "max_drawdown"], ascending=[False, True])
        best = df.iloc[0]
        results[name] = {
            "stop_loss_pct": float(best["stop_loss_pct"]),
            "take_profit_pct": float(best["take_profit_pct"]),
            "sharpe": float(best["sharpe"]),
            "max_drawdown": float(best["max_drawdown"]),
        }
        logger.info(
            "Best for %s: sl=%.4f, tp=%.4f, sharpe=%.2f, dd=%.2f",
            name,
            results[name]["stop_loss_pct"],
            results[name]["take_profit_pct"],
            results[name]["sharpe"],
            results[name]["max_drawdown"],
        )

    if results:
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            LOG_FILE.write_text(json.dumps(results, indent=2))
            logger.info("Wrote optimized params to %s", LOG_FILE)
        except Exception as e:
            logger.error(f"Failed to write optimization results: {e}")
    else:
        logger.warning("No optimization results to save")

    return results
