from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Any, Optional, Tuple

from crypto_bot.config import load_config as load_bot_config, resolve_config_path
from crypto_bot.utils.symbol_utils import fix_symbol

from crypto_bot.backtest.backtest_runner import BacktestRunner, BacktestConfig
from crypto_bot.utils.logger import LOG_DIR, setup_logger

# Advanced optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skopt import gp_minimize, forest_minimize
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

CONFIG_PATH = resolve_config_path()
LOG_FILE = LOG_DIR / "optimized_params.json"

logger = setup_logger(__name__, LOG_DIR / "optimizer.log")


class AdvancedParameterOptimizer:
    """Advanced parameter optimization using multiple algorithms."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimization_method = self.config.get("method", "bayesian")

        # Check available libraries
        if self.optimization_method == "bayesian" and not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to genetic algorithm")
            self.optimization_method = "genetic"
        elif self.optimization_method == "genetic" and not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, falling back to grid search")
            self.optimization_method = "grid"

    def optimize_strategy_parameters(
        self,
        strategy_name: str,
        param_ranges: Dict[str, List[float]],
        backtest_function: callable,
        max_evaluations: int = 100
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using advanced algorithms."""

        logger.info(f"Optimizing {strategy_name} using {self.optimization_method} method")

        if self.optimization_method == "bayesian" and OPTUNA_AVAILABLE:
            return self._optimize_bayesian(strategy_name, param_ranges, backtest_function, max_evaluations)
        elif self.optimization_method == "genetic" and SCIPY_AVAILABLE:
            return self._optimize_genetic(strategy_name, param_ranges, backtest_function, max_evaluations)
        else:
            return self._optimize_grid(strategy_name, param_ranges, backtest_function)

    def _optimize_bayesian(
        self,
        strategy_name: str,
        param_ranges: Dict[str, List[float]],
        backtest_function: callable,
        max_evaluations: int
    ) -> Dict[str, Any]:
        """Bayesian optimization using Optuna."""

        def objective(trial):
            params = {}
            for param_name, value_range in param_ranges.items():
                if len(value_range) == 2:
                    # Continuous range
                    params[param_name] = trial.suggest_float(
                        param_name, value_range[0], value_range[1]
                    )
                else:
                    # Discrete choices
                    params[param_name] = trial.suggest_categorical(param_name, value_range)

            try:
                result = backtest_function(params)
                score = -result.get("sharpe", 0)  # Negative for minimization
                return score
            except Exception as e:
                logger.warning(f"Backtest failed for params {params}: {e}")
                return float('inf')

        try:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=max_evaluations)

            best_params = study.best_params
            best_result = backtest_function(best_params)

            return {
                "method": "bayesian",
                "best_parameters": best_params,
                "best_score": -study.best_value,  # Convert back to positive Sharpe
                "optimization_history": [
                    {"trial": i, "params": t.params, "value": t.value}
                    for i, t in enumerate(study.trials)
                ],
                "backtest_result": best_result
            }

        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return self._optimize_grid(strategy_name, param_ranges, backtest_function)

    def _optimize_genetic(
        self,
        strategy_name: str,
        param_ranges: Dict[str, List[float]],
        backtest_function: callable,
        max_evaluations: int
    ) -> Dict[str, Any]:
        """Genetic algorithm optimization using differential evolution."""

        # Convert parameter ranges to bounds
        bounds = []
        param_names = []
        discrete_choices: Dict[str, Optional[List[float]]] = {}
        for param_name, value_range in param_ranges.items():
            if not value_range:
                logger.warning(f"Skipping {param_name}: empty range provided")
                continue

            try:
                numeric_values = sorted({float(v) for v in value_range})
            except (TypeError, ValueError):
                logger.warning(f"Non-numeric values for {param_name} are not supported in genetic optimization")
                continue

            if len(numeric_values) == 1:
                logger.warning(f"Skipping {param_name}: only a single value provided")
                continue

            bounds.append((numeric_values[0], numeric_values[-1]))
            param_names.append(param_name)
            # Retain the discrete grid so we can snap results back if needed
            discrete_choices[param_name] = numeric_values if len(numeric_values) > 2 else None

        def objective_function(params):
            param_dict: Dict[str, float] = {}
            for name, value in zip(param_names, params):
                choices = discrete_choices.get(name)
                if choices:
                    param_dict[name] = min(choices, key=lambda option: abs(option - value))
                else:
                    param_dict[name] = float(value)
            try:
                result = backtest_function(param_dict)
                return -result.get("sharpe", 0)  # Negative for minimization
            except Exception as e:
                logger.warning(f"Backtest failed for params {param_dict}: {e}")
                return float('inf')

        try:
            if not bounds:
                logger.warning("No continuous parameter bounds available; falling back to grid search")
                return self._optimize_grid(strategy_name, param_ranges, backtest_function)

            iterations = max(1, max_evaluations // max(len(bounds), 1))

            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=iterations,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42
            )

            if result.success:
                best_params: Dict[str, float] = {}
                for name, value in zip(param_names, result.x):
                    choices = discrete_choices.get(name)
                    if choices:
                        # Snap to the closest option from the original discrete grid
                        best_params[name] = min(choices, key=lambda option: abs(option - value))
                    else:
                        best_params[name] = float(value)
                best_result = backtest_function(best_params)

                return {
                    "method": "genetic",
                    "best_parameters": best_params,
                    "best_score": -result.fun,  # Convert back to positive Sharpe
                    "convergence_info": {
                        "nfev": result.nfev,
                        "success": result.success,
                        "message": result.message
                    },
                    "backtest_result": best_result
                }
            else:
                logger.warning("Genetic optimization did not converge, falling back to grid search")
                return self._optimize_grid(strategy_name, param_ranges, backtest_function)

        except Exception as e:
            logger.error(f"Genetic optimization failed: {e}")
            return self._optimize_grid(strategy_name, param_ranges, backtest_function)

    def _optimize_grid(
        self,
        strategy_name: str,
        param_ranges: Dict[str, List[float]],
        backtest_function: callable
    ) -> Dict[str, Any]:
        """Fallback grid search optimization."""

        logger.info(f"Using grid search optimization for {strategy_name}")

        # Generate all parameter combinations
        import itertools

        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        if not param_values:
            logger.warning("No parameter ranges provided for grid optimization")
            baseline = backtest_function({})
            return {
                "method": "grid",
                "best_parameters": {},
                "best_score": baseline.get("sharpe", float("-inf")),
                "optimization_history": [],
                "backtest_result": baseline,
            }

        combinations = itertools.product(*param_values)
        total_combinations = math.prod((len(values) for values in param_values), start=1)

        logger.info(f"Testing {total_combinations} parameter combinations")

        best_score = float('-inf')
        best_params = None
        best_result = None
        optimization_history = []

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            try:
                result = backtest_function(params)
                score = result.get("sharpe", 0)

                optimization_history.append({
                    "iteration": i,
                    "params": params,
                    "score": score
                })

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result

                if total_combinations and (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{total_combinations} combinations")

            except Exception as e:
                logger.warning(f"Backtest failed for combination {i}: {e}")
                continue

        return {
            "method": "grid",
            "best_parameters": best_params,
            "best_score": best_score,
            "optimization_history": optimization_history,
            "backtest_result": best_result
        }


def _load_config() -> dict:
    data = load_bot_config(CONFIG_PATH)
    if "symbol" in data:
        data["symbol"] = fix_symbol(data["symbol"])
    if "symbols" in data:
        data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]
    return data


def create_backtest_function(strategy_name: str, exchange=None):
    """Create a backtest function for parameter optimization."""
    def backtest_function(params: Dict[str, float]) -> Dict[str, Any]:
        """Backtest function for optimization algorithms."""
        try:
            config = BacktestConfig(
                symbol="BTC/USD",  # Default symbol for optimization
                timeframe="1h",
                since=0,
                limit=1000,
                mode="cex",
                stop_loss_range=[params.get("stop_loss_pct", 0.02)],
                take_profit_range=[params.get("take_profit_pct", 0.04)],
            )

            runner = BacktestRunner(config, exchange=exchange)
            df = runner.run_grid()

            if df.empty:
                return {"sharpe": 0.0, "pnl": 0.0, "max_drawdown": 1.0}

            # Get best result
            best = df.iloc[0]
            return {
                "sharpe": float(best.get("sharpe", 0.0)),
                "pnl": float(best.get("pnl", 0.0)),
                "max_drawdown": float(best.get("max_drawdown", 0.0)),
                "win_rate": float(best.get("win_rate", 0.5))
            }

        except Exception as e:
            logger.warning(f"Backtest function failed: {e}")
            return {"sharpe": 0.0, "pnl": 0.0, "max_drawdown": 1.0}

    return backtest_function


def optimize_strategies() -> Dict[str, Dict[str, float]]:
    """Run advanced parameter optimization for each configured strategy."""
    cfg = _load_config().get("optimization", {})
    if not cfg.get("enabled"):
        logger.info("Optimization disabled")
        return {}

    param_ranges = cfg.get("parameter_ranges", {})
    bot_cfg = _load_config()
    results: Dict[str, Dict[str, float]] = {}

    # Initialize exchange
    exchange = None
    try:
        from crypto_bot.execution.cex_executor import get_exchange
        exchange, _ = get_exchange(bot_cfg)
        logger.info(f"Using {exchange.id} exchange for backtesting")
    except Exception as e:
        logger.error(f"Failed to initialize exchange: {e}")
        exchange = None

    # Initialize advanced optimizer
    optimizer_config = cfg.get("advanced_config", {})
    optimizer = AdvancedParameterOptimizer(optimizer_config)

    for name, ranges in param_ranges.items():
        sl_range = ranges.get("stop_loss", [])
        tp_range = ranges.get("take_profit", [])

        if not sl_range or not tp_range:
            logger.warning(f"Skipping {name}: missing stop_loss or take_profit ranges")
            continue

        # Convert ranges to parameter format
        param_ranges_dict = {
            "stop_loss_pct": sl_range,
            "take_profit_pct": tp_range
        }

        # Create backtest function for this strategy
        backtest_func = create_backtest_function(name, exchange)

        # Run advanced optimization
        try:
            logger.info(f"Starting advanced optimization for {name}")
            optimization_result = optimizer.optimize_strategy_parameters(
                name,
                param_ranges_dict,
                backtest_func,
                max_evaluations=cfg.get("max_evaluations", 50)
            )

            if optimization_result["best_parameters"]:
                results[name] = {
                    "stop_loss_pct": optimization_result["best_parameters"]["stop_loss_pct"],
                    "take_profit_pct": optimization_result["best_parameters"]["take_profit_pct"],
                    "sharpe": optimization_result["best_score"],
                    "optimization_method": optimization_result["method"],
                    "max_drawdown": optimization_result["backtest_result"].get("max_drawdown", 0.0) if optimization_result["backtest_result"] else 0.0,
                }

                logger.info(
                    f"Best for {name}: sl={results[name]['stop_loss_pct']:.4f}, "
                    f"tp={results[name]['take_profit_pct']:.4f}, "
                    f"sharpe={results[name]['sharpe']:.2f}, "
                    f"method={results[name]['optimization_method']}"
                )
            else:
                logger.warning(f"No optimization results for {name}")

        except Exception as exc:
            logger.error(f"Advanced optimization failed for {name}: {exc}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

    # Save results
    if results:
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info("Wrote optimized params to %s", LOG_FILE)
        except Exception as e:
            logger.error(f"Failed to write optimization results: {e}")
    else:
        logger.warning("No optimization results to save")

    return results


def optimize_strategies_legacy() -> Dict[str, Dict[str, float]]:
    """Legacy optimization function for backward compatibility."""
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
        logger.info("Legacy optimization dependencies available")
    except ImportError as e:
        logger.error(f"Missing optimization dependencies: {e}")
        return {}

    # Initialize exchange
    exchange = None
    try:
        from crypto_bot.execution.cex_executor import get_exchange
        exchange, _ = get_exchange(bot_cfg)
        logger.info(f"Using {exchange.id} exchange for backtesting")
    except Exception as e:
        logger.error(f"Failed to initialize exchange: {e}")
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
            logger.info(f"Starting legacy optimization for {name}")
            runner = BacktestRunner(config, exchange=exchange)
            df = runner.run_grid()
            logger.info(f"Completed optimization for {name} - got {len(df)} results")
        except Exception as exc:
            logger.error(f"Backtest failed for {name}: {exc}")
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
