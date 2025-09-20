from dataclasses import dataclass
import dataclasses
from typing import Any, Optional, Mapping, Tuple, Dict

import pandas as pd
from math import isnan

from crypto_bot.capital_tracker import CapitalTracker

from crypto_bot.sentiment_filter import boost_factor, too_bearish
from crypto_bot.volatility_filter import too_flat, too_hot, calc_atr

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils import trade_memory
from crypto_bot.utils import ev_tracker
from crypto_bot.utils.strategy_utils import compute_drawdown

# Portfolio optimization imports
try:
    import cvxpy as cp
    import numpy as np
    from scipy.optimize import minimize
    PORTFOLIO_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PORTFOLIO_OPTIMIZATION_AVAILABLE = False


# Log to the main bot file so risk messages are consolidated
logger = setup_logger(__name__, LOG_DIR / "bot.log")


@dataclass
class RiskConfig:
    """Configuration values governing risk limits."""

    max_drawdown: float
    stop_loss_pct: float
    take_profit_pct: float
    min_fng: int = 0
    min_sentiment: int = 0
    bull_fng: int = 101
    bull_sentiment: int = 101
    min_atr_pct: float = 0.0
    max_funding_rate: float = 1.0
    symbol: str = ""
    trade_size_pct: float = 0.1
    risk_pct: float = 0.01
    min_volume: float = 0.0
    volume_threshold_ratio: float = 0.1
    strategy_allocation: Optional[dict] = None
    volume_ratio: float = 1.0
    atr_short_window: int = 14
    atr_long_window: int = 50
    max_volatility_factor: float = 1.5
    min_expected_value: float = 0.0
    default_expected_value: Optional[float] = None
    atr_period: int = 14
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0
    max_pair_drawdown: float = 0.0
    pair_drawdown_lookback: int = 20

    # Portfolio optimization settings
    enable_portfolio_optimization: bool = True
    max_portfolio_allocation: float = 0.2  # Max allocation per asset
    min_portfolio_allocation: float = 0.01  # Min allocation per asset
    rebalance_threshold: float = 0.05  # Rebalance when deviation > 5%
    risk_parity_weighting: bool = False


class RiskManager:
    """Utility class for evaluating account and trade level risk."""

    def __init__(self, config: RiskConfig) -> None:
        """Create a new manager with the given risk configuration."""
        self.config = config
        self.capital_tracker = CapitalTracker(config.strategy_allocation or {})
        self.equity = 1.0
        self.peak_equity = 1.0
        self.stop_orders: Dict[str, dict] = {}
        self.stop_order: Optional[dict] = None
        # Track protective stop orders for each open trade by symbol
        self.boost = 1.0

    @classmethod
    def from_config(cls, cfg: Mapping) -> "RiskManager":
        """Instantiate ``RiskManager`` from a configuration mapping.

        Parameters
        ----------
        cfg : Mapping
            Dictionary with keys corresponding to :class:`RiskConfig` fields.

        Returns
        -------
        RiskManager
            Newly created instance using the provided configuration.
        """
        params = {}
        for f in dataclasses.fields(RiskConfig):
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
                default = f.default_factory()
            else:
                default = None
            params[f.name] = cfg.get(f.name, default)
        config = RiskConfig(**params)
        return cls(config)

    def get_stop_order(self, symbol: str) -> Optional[dict]:
        """Return the stop order for ``symbol`` if present."""
        return self.stop_orders.get(symbol)

    def update_allocation(self, weights: dict) -> None:
        """Update strategy allocation weights at runtime."""
        self.config.strategy_allocation = weights
        self.capital_tracker.update_allocation(weights)

    def update_equity(self, new_equity: float) -> bool:
        """Update current equity and evaluate drawdown limit.

        Parameters
        ----------
        new_equity : float
            The account equity after the most recent trade.

        Returns
        -------
        bool
            ``True`` if drawdown remains under ``max_drawdown``.
        """
        self.equity = new_equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        drawdown = 1 - self.equity / self.peak_equity
        logger.info(
            "Equity updated to %.2f (drawdown %.2f)",
            self.equity,
            drawdown,
        )
        return drawdown < self.config.max_drawdown

    def calculate_position_size(
        self,
        confidence: float,
        balance: float,
        price: float,
        stop_loss_price: Optional[float] = None,
        risk_per_trade: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk management rules."""
        if risk_per_trade is None:
            risk_per_trade = self.config.risk_pct
        
        # Calculate risk amount
        risk_amount = balance * risk_per_trade
        
        # If stop loss is provided, calculate position size based on risk
        if stop_loss_price is not None:
            risk_per_share = abs(price - stop_loss_price)
            if risk_per_share > 0:
                return risk_amount / risk_per_share
        
        # Otherwise, use fixed percentage of balance
        position_value = balance * self.config.trade_size_pct * confidence
        return position_value / price

    def position_size(
        self,
        confidence: float,
        balance: float,
        df: Optional[pd.DataFrame] = None,
        stop_distance: Optional[float] = None,
        atr: Optional[float] = None,
        price: Optional[float] = None,
    ) -> float:
        """Return the trade value for a signal.

        When ``stop_distance`` or ``atr`` is provided the size is calculated
        using ``risk_pct`` relative to that distance.  Otherwise the fixed
        ``trade_size_pct`` is scaled by volatility and current drawdown.
        """

        volatility_factor = 1.0
        if df is not None and not df.empty:
            short_atr = calc_atr(df, window=self.config.atr_short_window)
            long_atr = calc_atr(df, window=self.config.atr_long_window)
            if long_atr > 0 and not isnan(short_atr) and not isnan(long_atr):
                volatility_factor = min(
                    short_atr / long_atr,
                    self.config.max_volatility_factor,
                )

        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = 1 - self.equity / self.peak_equity
        if self.config.max_drawdown > 0:
            capital_risk_factor = max(
                0.0, 1 - drawdown / self.config.max_drawdown
            )
        else:
            capital_risk_factor = 1.0

        if stop_distance is not None or atr is not None:
            risk_value = balance * self.config.risk_pct * confidence
            stop_loss_distance = atr if atr and atr > 0 else stop_distance
            trade_price = price or 1.0
            if stop_loss_distance and stop_loss_distance > 0:
                size = risk_value * trade_price / stop_loss_distance
            else:
                size = balance * confidence * self.config.trade_size_pct
            max_size = balance * self.config.trade_size_pct
            if size > max_size:
                size = max_size
        else:
            size = (
                balance
                * self.config.trade_size_pct
                * confidence
                * volatility_factor
                * capital_risk_factor
            )

        logger.info(
            "Calculated position size: %.4f (vol %.2f risk %.2f)",
            size,
            volatility_factor,
            capital_risk_factor,
        )
        
        # Apply minimum position size check
        min_position_size = getattr(self.config, 'min_position_size_usd', 10.0)
        if size < min_position_size:
            logger.warning(
                "Position size %.4f below minimum %.4f (confidence: %.2f, balance: %.2f) - using minimum",
                size,
                min_position_size,
                confidence,
                balance
            )
            # Only use minimum if the calculated size is at least 25% of minimum
            # This prevents tiny confidence scores from always defaulting to minimum
            if size >= min_position_size * 0.25:
                size = min_position_size
            else:
                logger.info(
                    "Calculated size %.4f too small (< 25%% of min) - skipping trade",
                    size
                )
                return 0.0
            
        return size

    def allow_trade(self, df: Any, strategy: Optional[str] = None) -> Tuple[bool, str]:
        """Assess whether market conditions merit taking a trade.

        Parameters
        ----------
        df : Any
            DataFrame of OHLCV data.

        Returns
        -------
        Tuple[bool, str]
            ``True``/``False`` along with the reason for the decision.
        """
        df_len = len(df)
        logger.info("[EVAL] Data length: %d", df_len)

        if df_len < 20:
            reason = "Not enough data to trade"
            logger.info("[EVAL] %s (len=%d)", reason, df_len)
            return False, reason

        if self.config.symbol and trade_memory.should_avoid(self.config.symbol):
            reason = "Symbol blocked by trade memory"
            logger.info("[EVAL] %s", reason)
            return False, reason

        if too_bearish(self.config.min_fng, self.config.min_sentiment):
            reason = "Sentiment too bearish"
            logger.info("[EVAL] %s", reason)
            return False, reason

        if too_flat(df, self.config.min_atr_pct):
            reason = "Market volatility too low"
            logger.info("[EVAL] %s", reason)
            return False, reason

        if self.config.symbol and too_hot(self.config.symbol, self.config.max_funding_rate):
            reason = "Funding rate too high"
            logger.info("[EVAL] %s", reason)
            return False, reason

        last_close = df["close"].iloc[-1]
        last_time = str(df.index[-1])
        logger.info(
            f"{self.config.symbol} | Last close: {last_close:.2f}, Time: {last_time}"
        )

        vol_mean = df["volume"].rolling(20).mean().iloc[-1]
        # Use most recent complete candle for volume (not incomplete current candle)
        current_volume_idx = -2 if len(df) >= 2 else -1
        current_volume = df["volume"].iloc[current_volume_idx]
        vol_threshold = vol_mean * self.config.volume_threshold_ratio
        logger.info(
            (
                "[EVAL] len=%d volume=%.4f (mean %.4f | min %.4f | threshold %.4f)"
            ),
            df_len,
            current_volume,
            vol_mean,
            self.config.min_volume,
            vol_threshold,
        )

        # Volume checks using configured thresholds
        if current_volume < self.config.min_volume:
            reason = "Volume < min volume threshold"
            logger.info(
                "[EVAL] %s (%.2f < %.2f)",
                reason,
                current_volume,
                self.config.min_volume,
            )
            return False, reason

        if current_volume < vol_threshold:
            percent = self.config.volume_threshold_ratio * 100
            reason = f"Volume < {percent:.0f}% of mean volume"
            logger.info(
                "[EVAL] %s (%.2f < %.2f)",
                reason,
                current_volume,
                vol_threshold,
            )
            return False, reason
        vol_std = df["close"].rolling(20).std().iloc[-1]
        prev_period_std = (
            df["close"].iloc[-21:-1].std() if len(df) >= 21 else float("nan")
        )
        if not isnan(prev_period_std) and vol_std < prev_period_std * 0.5:
            reason = "Volatility too low"
            logger.info(
                "[EVAL] %s (%.4f < %.4f)",
                reason,
                vol_std,
                prev_period_std * 0.5,
            )
            return False, reason

        if strategy is not None:
            ev = ev_tracker.get_expected_value(strategy)
            if ev == 0.0:
                stats = ev_tracker._load_stats().get(strategy, {})
                if not stats:
                    if self.config.default_expected_value is not None:
                        ev = self.config.default_expected_value
                    else:
                        ev = None
            if ev is not None and ev < self.config.min_expected_value:
                reason = (
                    f"Expected value {ev:.4f} below {self.config.min_expected_value}"
                )
                logger.info("[EVAL] %s", reason)
                return False, reason

        drawdown = compute_drawdown(
            df, lookback=self.config.pair_drawdown_lookback
        )
        if (
            self.config.max_pair_drawdown > 0
            and abs(drawdown) > self.config.max_pair_drawdown
        ):
            reason = (
                f"Pair drawdown {abs(drawdown):.2f} exceeds {self.config.max_pair_drawdown}"
            )
            logger.info("[EVAL] %s", reason)
            return False, reason

        self.boost = boost_factor(self.config.bull_fng, self.config.bull_sentiment)
        logger.info(
            f"[EVAL] Trade allowed for {self.config.symbol} – Volume {current_volume:.4f} >= {self.config.volume_threshold_ratio*100}% of mean {vol_mean:.4f}"
        )
        reason = f"Trade allowed (boost {self.boost:.2f})"
        logger.info("[EVAL] %s", reason)
        return True, reason

    def register_stop_order(
        self,
        order: dict,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        entry_price: Optional[float] = None,
        confidence: Optional[float] = None,
        direction: Optional[str] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Store the protective stop order and related trade info."""
        order = dict(order)
        if strategy is not None:
            order["strategy"] = strategy
        if symbol is None:
            symbol = order.get("symbol")
        if symbol is None:
            raise ValueError("Symbol required to register stop order")
        order["symbol"] = symbol
        if entry_price is not None:
            order["entry_price"] = entry_price
        if confidence is not None:
            order["confidence"] = confidence
        if direction is not None:
            order["direction"] = direction
        if take_profit is not None:
            order["take_profit"] = take_profit
        self.stop_order = order
        if symbol is not None:
            self.stop_orders[symbol] = order
        logger.info("Registered stop order %s", order)

    def update_stop_order(self, new_amount: float, symbol: Optional[str] = None) -> None:
        """Update the stored stop order amount."""
        order = self.stop_orders.get(symbol) if symbol else self.stop_order
        if not order:
            return
        order["amount"] = new_amount
        if symbol:
            self.stop_orders[symbol] = order
        else:
            self.stop_order = order
        logger.info("Updated stop order amount to %.4f", new_amount)

    def cancel_stop_order(self, exchange, symbol: Optional[str] = None) -> None:
        """Cancel the existing stop order if needed."""
        order = self.stop_orders.get(symbol) if symbol else self.stop_order
        if not order:
            return
        if not order.get("dry_run") and "id" in order:
            try:
                exchange.cancel_order(order["id"], order.get("symbol"))
                logger.info("Cancelled stop order %s", order.get("id"))
            except Exception as e:
                logger.error("Failed to cancel stop order: %s", e)
        if symbol:
            self.stop_orders.pop(symbol, None)
        else:
            self.stop_order = None

    def can_allocate(self, strategy: str, amount: float, balance: float) -> bool:
        """Check if ``strategy`` can use additional ``amount`` capital."""
        return self.capital_tracker.can_allocate(strategy, amount, balance)

    def allocate_capital(self, strategy: str, amount: float) -> None:
        """Record capital allocation for a strategy."""
        self.capital_tracker.allocate(strategy, amount)

    def deallocate_capital(self, strategy: str, amount: float) -> None:
        """Release previously allocated capital."""
        self.capital_tracker.deallocate(strategy, amount)

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) for a series of returns."""
        if returns.empty:
            return 0.0

        try:
            # Historical VaR
            return -np.percentile(returns.values, (1 - confidence_level) * 100)
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.0

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)."""
        if returns.empty:
            return 0.0

        try:
            var_threshold = -self.calculate_var(returns, confidence_level)
            tail_losses = returns[returns <= var_threshold]

            if len(tail_losses) == 0:
                return var_threshold

            return -tail_losses.mean()
        except Exception as e:
            logger.error(f"CVaR calculation error: {e}")
            return 0.0

    def calculate_portfolio_var(
        self,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate portfolio VaR using covariance matrix approach."""
        try:
            if asset_returns.empty or not weights:
                return 0.0

            # Calculate covariance matrix
            cov_matrix = asset_returns.cov()

            # Convert weights to array in correct order
            asset_list = list(weights.keys())
            weight_array = np.array([weights[asset] for asset in asset_list])

            # Portfolio volatility
            portfolio_volatility = np.sqrt(weight_array.T @ cov_matrix.values @ weight_array)

            # Portfolio VaR (assuming normal distribution)
            # Z-score for confidence level
            from scipy.stats import norm
            z_score = norm.ppf(confidence_level)

            portfolio_var = -z_score * portfolio_volatility

            return portfolio_var

        except Exception as e:
            logger.error(f"Portfolio VaR calculation error: {e}")
            return 0.0

    def calculate_portfolio_cvar(
        self,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate portfolio CVaR using Monte Carlo simulation."""
        try:
            if asset_returns.empty or not weights:
                return 0.0

            # Monte Carlo simulation
            n_simulations = 10000

            # Calculate mean returns and covariance matrix
            mean_returns = asset_returns.mean()
            cov_matrix = asset_returns.cov()

            # Generate random scenarios
            scenarios = np.random.multivariate_normal(
                mean_returns.values,
                cov_matrix.values,
                n_simulations
            )

            # Calculate portfolio returns for each scenario
            asset_list = list(weights.keys())
            weight_array = np.array([weights[asset] for asset in asset_list])
            portfolio_returns = scenarios @ weight_array

            # Calculate CVaR
            var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            tail_losses = portfolio_returns[portfolio_returns <= var_threshold]

            if len(tail_losses) == 0:
                return -var_threshold

            return -tail_losses.mean()

        except Exception as e:
            logger.error(f"Portfolio CVaR calculation error: {e}")
            return 0.0

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        try:
            if returns.empty:
                return {
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "var_95": 0.0,
                    "cvar_95": 0.0,
                    "volatility": 0.0,
                    "skewness": 0.0,
                    "kurtosis": 0.0
                }

            # Basic metrics
            mean_return = returns.mean()
            volatility = returns.std()
            risk_free_rate = 0.02  # Assume 2% risk-free rate

            # Sharpe ratio
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

            # VaR and CVaR
            var_95 = self.calculate_var(returns, confidence_level)
            cvar_95 = self.calculate_cvar(returns, confidence_level)

            # Higher moments
            skewness = returns.skew() if len(returns) > 2 else 0
            kurtosis = returns.kurtosis() if len(returns) > 3 else 0

            return {
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "volatility": volatility,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "expected_return": mean_return,
                "total_return": (1 + returns).prod() - 1
            }

        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "volatility": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "expected_return": 0.0,
                "total_return": 0.0
            }

    def assess_risk_limits(
        self,
        current_metrics: Dict[str, float],
        position_size: float,
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Assess if current risk metrics are within acceptable limits."""

        try:
            # Check individual risk limits
            violations = []

            # Maximum drawdown limit
            if current_metrics.get("max_drawdown", 0) > self.config.max_drawdown:
                violations.append({
                    "type": "max_drawdown",
                    "current": current_metrics["max_drawdown"],
                    "limit": self.config.max_drawdown,
                    "breached": True
                })

            # VaR limit (if configured)
            var_limit = getattr(self.config, 'max_var_limit', 0.1)  # 10% default
            if current_metrics.get("var_95", 0) > var_limit:
                violations.append({
                    "type": "var_95",
                    "current": current_metrics["var_95"],
                    "limit": var_limit,
                    "breached": True
                })

            # Position size limit
            max_position_pct = getattr(self.config, 'max_position_size_pct', 0.1)  # 10% default
            position_pct = position_size / portfolio_value if portfolio_value > 0 else 0
            if position_pct > max_position_pct:
                violations.append({
                    "type": "position_size",
                    "current": position_pct,
                    "limit": max_position_pct,
                    "breached": True
                })

            # Volatility limit
            max_volatility = getattr(self.config, 'max_volatility_limit', 0.5)  # 50% default
            if current_metrics.get("volatility", 0) > max_volatility:
                violations.append({
                    "type": "volatility",
                    "current": current_metrics["volatility"],
                    "limit": max_volatility,
                    "breached": True
                })

            return {
                "within_limits": len(violations) == 0,
                "violations": violations,
                "overall_risk_score": self._calculate_risk_score(current_metrics)
            }

        except Exception as e:
            logger.error(f"Risk limit assessment error: {e}")
            return {
                "within_limits": False,
                "violations": [{"type": "calculation_error", "error": str(e)}],
                "overall_risk_score": 1.0
            }

    def _calculate_risk_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall risk score (0-1, higher = riskier)."""
        try:
            # Normalize each risk metric to 0-1 scale
            normalized_metrics = []

            # Sharpe ratio (lower is riskier)
            sharpe = metrics.get("sharpe_ratio", 0)
            sharpe_score = max(0, min(1, 1 / (1 + sharpe)))  # Invert and normalize
            normalized_metrics.append(sharpe_score)

            # Maximum drawdown (higher is riskier)
            max_dd = metrics.get("max_drawdown", 0)
            dd_score = max(0, min(1, max_dd / self.config.max_drawdown))
            normalized_metrics.append(dd_score)

            # VaR (higher is riskier)
            var = metrics.get("var_95", 0)
            var_score = max(0, min(1, var / 0.1))  # Assume 10% is max acceptable
            normalized_metrics.append(var_score)

            # Volatility (higher is riskier)
            vol = metrics.get("volatility", 0)
            vol_score = max(0, min(1, vol / 0.5))  # Assume 50% is max acceptable
            normalized_metrics.append(vol_score)

            # Average of all normalized metrics
            return np.mean(normalized_metrics) if normalized_metrics else 0.5

        except Exception as e:
            logger.error(f"Risk score calculation error: {e}")
            return 0.5


class PortfolioOptimizer:
    """Modern portfolio theory implementation for optimal asset allocation."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.available = PORTFOLIO_OPTIMIZATION_AVAILABLE

        if not self.available:
            logger.warning("Portfolio optimization libraries not available (cvxpy, scipy)")
            return

    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """Optimize portfolio using modern portfolio theory (Markowitz optimization)."""

        if not self.available:
            return self._fallback_equal_weight(expected_returns)

        try:
            assets = expected_returns.index.tolist()
            n_assets = len(assets)

            # Convert to numpy arrays
            mu = expected_returns.values
            Sigma = covariance_matrix.values

            # Define optimization variables
            weights = cp.Variable(n_assets)

            # Define constraints
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= self.config.min_portfolio_allocation,  # Minimum allocation
                weights <= self.config.max_portfolio_allocation   # Maximum allocation
            ]

            # Portfolio return and risk
            portfolio_return = mu.T @ weights
            portfolio_risk = cp.quad_form(weights, Sigma)

            # Optimization objectives
            if self.config.risk_parity_weighting:
                # Risk parity optimization
                risk_contributions = cp.multiply(weights, Sigma @ weights) / portfolio_risk
                objective = cp.sum_squares(risk_contributions - 1/n_assets)
            else:
                # Mean-variance optimization (maximizing Sharpe ratio)
                objective = cp.Minimize(portfolio_risk - portfolio_return)

            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status != cp.OPTIMAL:
                logger.warning(f"Portfolio optimization failed: {problem.status}")
                return self._fallback_equal_weight(expected_returns)

            # Extract optimal weights
            optimal_weights = weights.value

            # Calculate portfolio metrics
            portfolio_return_opt = float(mu.T @ optimal_weights)
            portfolio_volatility = float(np.sqrt(optimal_weights.T @ Sigma @ optimal_weights))
            sharpe_ratio = (portfolio_return_opt - risk_free_rate) / portfolio_volatility

            return {
                "weights": dict(zip(assets, optimal_weights)),
                "expected_return": portfolio_return_opt,
                "volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "optimization_method": "mean_variance" if not self.config.risk_parity_weighting else "risk_parity",
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return self._fallback_equal_weight(expected_returns)

    def _fallback_equal_weight(self, expected_returns: pd.Series) -> Dict[str, Any]:
        """Fallback to equal weight allocation when optimization fails."""
        n_assets = len(expected_returns)
        equal_weight = 1.0 / n_assets

        weights = {asset: equal_weight for asset in expected_returns.index}

        return {
            "weights": weights,
            "expected_return": expected_returns.mean(),
            "volatility": expected_returns.std(),
            "sharpe_ratio": expected_returns.mean() / expected_returns.std() if expected_returns.std() > 0 else 0,
            "optimization_method": "equal_weight",
            "status": "fallback"
        }

    def calculate_efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_points: int = 50
    ) -> pd.DataFrame:
        """Calculate the efficient frontier for portfolio optimization."""

        if not self.available:
            return pd.DataFrame()

        try:
            assets = expected_returns.index.tolist()
            n_assets = len(assets)
            mu = expected_returns.values
            Sigma = Sigma = covariance_matrix.values

            # Generate range of target returns
            min_return = np.min(mu)
            max_return = np.max(mu)
            target_returns = np.linspace(min_return, max_return, n_points)

            frontier_points = []

            for target_return in target_returns:
                # Define optimization
                weights = cp.Variable(n_assets)

                constraints = [
                    cp.sum(weights) == 1,
                    weights >= self.config.min_portfolio_allocation,
                    weights <= self.config.max_portfolio_allocation,
                    mu.T @ weights >= target_return
                ]

                portfolio_risk = cp.quad_form(weights, Sigma)
                objective = cp.Minimize(portfolio_risk)
                problem = cp.Problem(objective, constraints)

                try:
                    problem.solve()
                    if problem.status == cp.OPTIMAL:
                        optimal_weights = weights.value
                        volatility = float(np.sqrt(optimal_weights.T @ Sigma @ optimal_weights))

                        frontier_points.append({
                            "return": target_return,
                            "volatility": volatility,
                            "sharpe_ratio": target_return / volatility if volatility > 0 else 0,
                            "weights": dict(zip(assets, optimal_weights))
                        })
                except:
                    continue

            return pd.DataFrame(frontier_points)

        except Exception as e:
            logger.error(f"Efficient frontier calculation error: {e}")
            return pd.DataFrame()

    def rebalance_portfolio(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate rebalancing trades to achieve target weights."""

        try:
            # Calculate current values
            current_values = {
                asset: current_weights.get(asset, 0) * portfolio_value
                for asset in target_weights.keys()
            }

            # Calculate target values
            target_values = {
                asset: target_weights[asset] * portfolio_value
                for asset in target_weights.keys()
            }

            # Calculate required trades
            trades = {}
            for asset in target_weights.keys():
                current_value = current_values.get(asset, 0)
                target_value = target_values[asset]
                value_difference = target_value - current_value

                # Only trade if difference exceeds threshold
                if abs(value_difference / portfolio_value) > self.config.rebalance_threshold:
                    price = current_prices.get(asset, 1.0)
                    trades[asset] = value_difference / price

            return trades

        except Exception as e:
            logger.error(f"Portfolio rebalancing error: {e}")
            return {}

    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""

        try:
            weight_array = np.array([weights.get(asset, 0) for asset in expected_returns.index])
            mu = expected_returns.values
            Sigma = covariance_matrix.values

            # Expected return
            expected_return = weight_array.T @ mu

            # Volatility (standard deviation)
            volatility = np.sqrt(weight_array.T @ Sigma @ weight_array)

            # Sharpe ratio
            sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0

            # Value at Risk (95% confidence)
            var_95 = np.percentile(
                np.random.multivariate_normal(mu, Sigma, 10000) @ weight_array,
                5
            )

            # Conditional VaR (Expected Shortfall)
            portfolio_returns = np.random.multivariate_normal(mu, Sigma, 10000) @ weight_array
            tail_returns = portfolio_returns[portfolio_returns <= var_95]
            cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95

            # Maximum drawdown (simplified)
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (running_max - cumulative_returns) / running_max
            max_drawdown = np.max(drawdowns)

            return {
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "max_drawdown": max_drawdown
            }

        except Exception as e:
            logger.error(f"Portfolio metrics calculation error: {e}")
            return {
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "max_drawdown": 0.0
            }


class EnhancedRiskManager(RiskManager):
    """Enhanced risk manager with portfolio optimization capabilities."""

    def __init__(self, config: RiskConfig) -> None:
        super().__init__(config)
        self.portfolio_optimizer = PortfolioOptimizer(config) if config.enable_portfolio_optimization else None

    def optimize_portfolio_allocation(
        self,
        strategy_returns: Dict[str, pd.Series],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation across strategies."""

        if not self.portfolio_optimizer:
            # Fallback to equal weighting
            n_strategies = len(strategy_returns)
            equal_weight = 1.0 / n_strategies if n_strategies > 0 else 0

            return {
                "weights": {strategy: equal_weight for strategy in strategy_returns.keys()},
                "method": "equal_weight",
                "status": "fallback"
            }

        try:
            # Calculate expected returns for each strategy
            expected_returns = pd.Series({
                strategy: returns.mean() for strategy, returns in strategy_returns.items()
            })

            # Calculate covariance matrix
            returns_df = pd.DataFrame(strategy_returns)
            if correlation_matrix is None:
                covariance_matrix = returns_df.cov()
            else:
                # Use correlation matrix to construct covariance matrix
                volatilities = returns_df.std()
                covariance_matrix = correlation_matrix * np.outer(volatilities, volatilities)

            # Optimize portfolio
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                expected_returns, covariance_matrix
            )

            return optimization_result

        except Exception as e:
            logger.error(f"Portfolio allocation optimization error: {e}")

            # Fallback to equal weighting
            n_strategies = len(strategy_returns)
            equal_weight = 1.0 / n_strategies if n_strategies > 0 else 0

            return {
                "weights": {strategy: equal_weight for strategy in strategy_returns.keys()},
                "method": "equal_weight_fallback",
                "status": "error",
                "error": str(e)
            }
