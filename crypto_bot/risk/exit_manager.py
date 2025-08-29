"""Position exit helpers with improved trailing stop and take profit logic."""

from typing import Tuple, Dict, List, Optional
import pandas as pd
import ta
import numpy as np
from dataclasses import dataclass

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.volatility_filter import calc_atr
from pathlib import Path


# Use the main bot log for exit messages
logger = setup_logger(__name__, LOG_DIR / "bot.log")


@dataclass
class MomentumExitConfig:
    """Configuration for momentum-aware exit strategy."""
    
    # Momentum thresholds
    weak_momentum: float = 0.3
    moderate_momentum: float = 0.6
    strong_momentum: float = 0.8
    very_strong_momentum: float = 0.9
    
    # Take profit scaling
    momentum_tp_multipliers: Dict[str, float] = None
    
    # Partial exit settings
    partial_exit_momentum_thresholds: List[Dict] = None
    
    # Trailing stop adjustments
    momentum_trailing_adjustments: Dict[str, float] = None
    
    # Momentum continuation
    rsi_momentum_threshold: float = 65.0
    volume_momentum_threshold: float = 1.5
    price_acceleration_threshold: float = 0.002
    macd_momentum_threshold: float = 0.001
    
    # Breakout detection
    breakout_threshold: float = 0.015
    momentum_extension_multiplier: float = 2.5
    volume_breakout_multiplier: float = 3.0
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.momentum_tp_multipliers is None:
            self.momentum_tp_multipliers = {
                'weak_momentum': 1.0,
                'moderate_momentum': 1.5,
                'strong_momentum': 2.0,
                'very_strong_momentum': 3.0
            }
        
        if self.partial_exit_momentum_thresholds is None:
            self.partial_exit_momentum_thresholds = [
                {'momentum': 0.3, 'exit_pct': 10, 'profit_pct': 0.02},
                {'momentum': 0.5, 'exit_pct': 15, 'profit_pct': 0.03},
                {'momentum': 0.7, 'exit_pct': 20, 'profit_pct': 0.05},
                {'momentum': 0.85, 'exit_pct': 25, 'profit_pct': 0.08}
            ]
        
        if self.momentum_trailing_adjustments is None:
            self.momentum_trailing_adjustments = {
                'weak_momentum': 1.0,
                'moderate_momentum': 0.8,
                'strong_momentum': 0.6,
                'very_strong_momentum': 0.4
            }


def calculate_trailing_stop(
    price_series: pd.Series, trail_pct: float = 0.1
) -> float:
    """Return a trailing stop from the high of ``price_series``.

    Parameters
    ----------
    price_series : pd.Series
        Series of closing prices.
    trail_pct : float, optional
        Percentage to trail below the maximum price.

    Returns
    -------
    float
        Calculated trailing stop value.
    """
    highest = price_series.max()
    stop = highest * (1 - trail_pct)
    logger.info("Calculated trailing stop %.4f from high %.4f", stop, highest)
    return stop


def calculate_trailing_stop_short(
    price_series: pd.Series, trail_pct: float = 0.1
) -> float:
    """Return a trailing stop from the low of ``price_series`` for short positions.

    Parameters
    ----------
    price_series : pd.Series
        Series of closing prices.
    trail_pct : float, optional
        Percentage to trail above the minimum price.

    Returns
    -------
    float
        Calculated trailing stop value.
    """
    lowest = price_series.min()
    stop = lowest * (1 + trail_pct)
    logger.info("Calculated trailing stop %.4f from low %.4f", stop, lowest)
    return stop


def calculate_atr_trailing_stop(
    df: pd.DataFrame, atr_factor: float = 2.0
) -> float:
    """Calculate trailing stop using ATR for dynamic adjustment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data and ATR column.
    atr_factor : float
        Multiplier for ATR-based trailing stop.

    Returns
    -------
    float
        ATR-based trailing stop value.
    """
    if "atr" not in df.columns:
        df = df.copy()
        df["atr"] = calc_atr(df, window=14)
    
    if df["atr"].iloc[-1] > 0:
        atr_value = df["atr"].iloc[-1]
        highest = df["high"].max()
        stop = highest - (atr_value * atr_factor)
        logger.info("ATR trailing stop: %.4f (ATR: %.4f, factor: %.2f)", stop, atr_value, atr_factor)
        return stop
    else:
        # Fallback to percentage-based trailing stop
        highest = df["high"].max()
        stop = highest * 0.95
        logger.info("Fallback trailing stop: %.4f", stop)
        return stop


def calculate_atr_trailing_stop_short(
    df: pd.DataFrame, atr_factor: float = 2.0
) -> float:
    """Calculate trailing stop using ATR for short positions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data and ATR column.
    atr_factor : float
        Multiplier for ATR-based trailing stop.

    Returns
    -------
    float
        ATR-based trailing stop value for shorts.
    """
    if "atr" not in df.columns:
        df = df.copy()
        df["atr"] = calc_atr(df, window=14)
    
    if df["atr"].iloc[-1] > 0:
        atr_value = df["atr"].iloc[-1]
        lowest = df["low"].min()
        stop = lowest + (atr_value * atr_factor)
        logger.info("ATR trailing stop (short): %.4f (ATR: %.4f, factor: %.2f)", stop, atr_value, atr_factor)
        return stop
    else:
        # Fallback to percentage-based trailing stop
        lowest = df["low"].min()
        stop = lowest * 1.05
        logger.info("Fallback trailing stop (short): %.4f", stop)
        return stop


def momentum_healthy(df: pd.DataFrame) -> bool:
    """Check RSI, MACD and volume to gauge trend health.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data used to compute indicators.

    Returns
    -------
    bool
        ``True`` if the momentum indicators confirm strength.
    """
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    vol_avg = df['volume'].rolling(3).mean()
    # Ensure at least two non-null volume averages exist before comparing
    if vol_avg.dropna().shape[0] < 2:
        return False
    vol_rising = vol_avg.iloc[-1] > vol_avg.iloc[-2]

    latest = df.iloc[-1]
    # Verify momentum indicators have valid values
    if (
        pd.isna(latest.get('rsi'))
        or pd.isna(latest.get('macd'))
        or pd.isna(latest.get('macd_signal'))
    ):
        return False

    return bool(
        latest['rsi'] > 55
        and latest['macd'] > latest['macd_signal']
        and vol_rising
    )


def _assess_momentum_strength(df: pd.DataFrame) -> float:
    """Assess momentum strength on a scale of 0-1, less restrictive than momentum_healthy."""
    try:
        df = df.copy()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        latest = df.iloc[-1]
        
        # Check if indicators have valid values
        if (pd.isna(latest.get('rsi')) or 
            pd.isna(latest.get('macd')) or 
            pd.isna(latest.get('macd_signal'))):
            return 0.5  # Neutral if data unavailable
        
        # Calculate momentum score components
        rsi_score = min(1.0, max(0.0, (latest['rsi'] - 30) / 40))  # 30-70 range
        macd_score = 1.0 if latest['macd'] > latest['macd_signal'] else 0.0
        
        # Volume trend (less strict)
        vol_avg = df['volume'].rolling(3).mean()
        if vol_avg.dropna().shape[0] >= 2:
            vol_score = 1.0 if vol_avg.iloc[-1] > vol_avg.iloc[-2] else 0.5
        else:
            vol_score = 0.5
        
        # Weighted average
        momentum_strength = (rsi_score * 0.4 + macd_score * 0.4 + vol_score * 0.2)
        
        return momentum_strength
        
    except Exception:
        return 0.5  # Neutral on error


def detect_momentum_continuation(df: pd.DataFrame, config: MomentumExitConfig) -> Dict[str, any]:
    """Detect if momentum is likely to continue based on multiple indicators."""
    try:
        df = df.copy()
        
        # Calculate indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        # Price acceleration (second derivative)
        df['price_change'] = df['close'].pct_change()
        df['price_acceleration'] = df['price_change'].pct_change()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        latest = df.iloc[-1]
        
        # Check momentum continuation signals
        rsi_strong = latest['rsi'] > config.rsi_momentum_threshold
        macd_strong = (latest['macd'] - latest['macd_signal']) > config.macd_momentum_threshold
        volume_strong = latest['volume_ratio'] > config.volume_momentum_threshold
        acceleration_strong = latest['price_acceleration'] > config.price_acceleration_threshold
        
        # Calculate continuation probability
        signals = [rsi_strong, macd_strong, volume_strong, acceleration_strong]
        continuation_probability = sum(signals) / len(signals)
        
        # Detect breakouts
        recent_high = df['high'].tail(20).max()
        current_price = latest['close']
        breakout_detected = (current_price - recent_high) / recent_high > config.breakout_threshold
        
        return {
            'continuation_probability': continuation_probability,
            'rsi_strong': rsi_strong,
            'macd_strong': macd_strong,
            'volume_strong': volume_strong,
            'acceleration_strong': acceleration_strong,
            'breakout_detected': breakout_detected,
            'volume_spike': latest['volume_ratio'] > config.volume_breakout_multiplier
        }
        
    except Exception as e:
        logger.error(f"Error detecting momentum continuation: {e}")
        return {
            'continuation_probability': 0.5,
            'rsi_strong': False,
            'macd_strong': False,
            'volume_strong': False,
            'acceleration_strong': False,
            'breakout_detected': False,
            'volume_spike': False
        }


def calculate_momentum_scaled_take_profit(
    base_tp_pct: float, 
    momentum_strength: float, 
    config: MomentumExitConfig,
    momentum_continuation: Dict[str, any] = None
) -> float:
    """Calculate take profit scaled by momentum strength and continuation signals."""
    
    # Base momentum scaling
    if momentum_strength < config.weak_momentum:
        multiplier = config.momentum_tp_multipliers['weak_momentum']
    elif momentum_strength < config.moderate_momentum:
        multiplier = config.momentum_tp_multipliers['moderate_momentum']
    elif momentum_strength < config.strong_momentum:
        multiplier = config.momentum_tp_multipliers['strong_momentum']
    else:
        multiplier = config.momentum_tp_multipliers['very_strong_momentum']
    
    # Additional scaling for momentum continuation
    if momentum_continuation and momentum_continuation.get('continuation_probability', 0) > 0.75:
        multiplier *= 1.2  # 20% additional scaling for strong continuation
    
    # Breakout momentum scaling
    if momentum_continuation and momentum_continuation.get('breakout_detected', False):
        multiplier *= config.momentum_extension_multiplier
    
    # Volume spike scaling
    if momentum_continuation and momentum_continuation.get('volume_spike', False):
        multiplier *= config.volume_breakout_multiplier
    
    scaled_tp = base_tp_pct * multiplier
    logger.info(f"Momentum-scaled take profit: {base_tp_pct:.4f} -> {scaled_tp:.4f} (multiplier: {multiplier:.2f})")
    
    return scaled_tp


def calculate_momentum_adjusted_trailing_stop(
    base_trail_pct: float,
    momentum_strength: float,
    config: MomentumExitConfig
) -> float:
    """Calculate trailing stop adjusted by momentum strength."""
    
    if momentum_strength < config.weak_momentum:
        adjustment = config.momentum_trailing_adjustments['weak_momentum']
    elif momentum_strength < config.moderate_momentum:
        adjustment = config.momentum_trailing_adjustments['moderate_momentum']
    elif momentum_strength < config.strong_momentum:
        adjustment = config.momentum_trailing_adjustments['strong_momentum']
    else:
        adjustment = config.momentum_trailing_adjustments['very_strong_momentum']
    
    adjusted_trail = base_trail_pct * adjustment
    logger.info(f"Momentum-adjusted trailing stop: {base_trail_pct:.4f} -> {adjusted_trail:.4f} (adjustment: {adjustment:.2f})")
    
    return adjusted_trail


def should_exit(
    df: pd.DataFrame,
    current_price: float,
    trailing_stop: float,
    config: dict,
    risk_manager=None,
    position_side: str = "buy",  # Add position side parameter
    entry_price: float = None,  # Add entry price for take profit calculations
) -> Tuple[bool, float]:
    """Determine whether to exit a position and update trailing stop.

    Parameters
    ----------
    df : pd.DataFrame
        Recent market data.
    current_price : float
        Latest traded price.
    trailing_stop : float
        Current trailing stop value.
    config : dict
        Strategy configuration.
    risk_manager : object, optional
        Risk manager instance for stop order handling.
    position_side : str
        Position side: "buy" for long, "sell" for short.
    entry_price : float, optional
        Entry price for take profit calculations.

    Returns
    -------
    Tuple[bool, float]
        Flag indicating whether to exit and the updated stop price.
    """
    exit_signal = False
    new_stop = trailing_stop
    
    # Get exit strategy configuration
    exit_cfg = config.get('exit_strategy', {})
    
    # Check if momentum-aware exits are enabled
    momentum_aware = exit_cfg.get('momentum_aware_exits', False)
    
    # Check take profit first (if configured)
    if entry_price is not None:
        take_profit_pct = exit_cfg.get('take_profit_pct', 0.0)
        
        if take_profit_pct > 0:
            # Apply momentum scaling if enabled
            if momentum_aware and exit_cfg.get('momentum_tp_scaling', False):
                momentum_strength = _assess_momentum_strength(df)
                momentum_continuation = detect_momentum_continuation(df, MomentumExitConfig(**exit_cfg.get('momentum_continuation', {})))
                
                # Create config object for momentum calculations
                momentum_config = MomentumExitConfig(**exit_cfg.get('momentum_continuation', {}))
                scaled_tp = calculate_momentum_scaled_take_profit(
                    take_profit_pct, 
                    momentum_strength, 
                    momentum_config,
                    momentum_continuation
                )
                take_profit_pct = scaled_tp
            
            if position_side == "buy":  # Long position
                take_profit_price = entry_price * (1 + take_profit_pct)
                if current_price >= take_profit_price:
                    logger.info(
                        "Take profit hit at %.4f (target: %.4f) for long position",
                        current_price,
                        take_profit_price,
                    )
                    return True, new_stop
            else:  # Short position
                take_profit_price = entry_price * (1 - take_profit_pct)
                if current_price <= take_profit_price:
                    logger.info(
                        "Take profit hit at %.4f (target: %.4f) for short position",
                        current_price,
                        take_profit_price,
                    )
                    return True, new_stop
    
    # Check if price hit trailing stop based on position side
    if position_side == "buy":  # Long position
        stop_hit = current_price < trailing_stop
    else:  # Short position
        stop_hit = current_price > trailing_stop
    
    if stop_hit and trailing_stop > 0:
        # Enhanced momentum check for exit decisions
        momentum_strength = _assess_momentum_strength(df)
        
        if momentum_aware:
            # Get momentum continuation data
            momentum_continuation = detect_momentum_continuation(df, MomentumExitConfig(**exit_cfg.get('momentum_continuation', {})))
            
            # More sophisticated momentum-based exit logic
            should_block_exit = False
            
            # Block exit if momentum is very strong and continuing
            if (momentum_strength > 0.8 and 
                momentum_continuation.get('continuation_probability', 0) > 0.75):
                should_block_exit = True
                logger.info("Blocking exit due to very strong continuing momentum")
            
            # Block exit if breakout detected with strong volume
            if (momentum_continuation.get('breakout_detected', False) and 
                momentum_continuation.get('volume_spike', False)):
                should_block_exit = True
                logger.info("Blocking exit due to breakout with volume spike")
            
            if not should_block_exit:
                exit_signal = True
                logger.info(
                    "Price %.4f hit trailing stop %.4f for %s position (momentum: %.2f)",
                    current_price,
                    trailing_stop,
                    "long" if position_side == "buy" else "short",
                    momentum_strength
                )
        else:
            # Legacy logic - only block exit if momentum is very strong
            if momentum_strength < 0.7:  # Allow exit unless momentum is very strong
                exit_signal = True
                logger.info(
                    "Price %.4f hit trailing stop %.4f for %s position",
                    current_price,
                    trailing_stop,
                    "long" if position_side == "buy" else "short",
                )
        
        # Handle exit signal
        if exit_signal:
            if risk_manager and getattr(risk_manager, "stop_order", None):
                order = risk_manager.stop_order
                entry = order.get("entry_price")
                direction = order.get("direction")
                strategy = order.get("strategy", "")
                symbol = order.get("symbol", config.get("symbol", ""))
                confidence = order.get("confidence", 0.0)
                if entry is not None and direction:
                    pnl = (current_price - entry) * (
                        1 if direction == "buy" else -1
                    )
                    from crypto_bot.utils.pnl_logger import log_pnl

                    log_pnl(
                        strategy,
                        symbol,
                        entry,
                        current_price,
                        pnl,
                        confidence,
                        direction,
                    )
    else:
        if trailing_stop > 0:
            # Update trailing stop with momentum adjustments
            if momentum_aware and exit_cfg.get('momentum_trail_adjustment', False):
                momentum_strength = _assess_momentum_strength(df)
                base_trail_pct = exit_cfg.get('trailing_stop_pct', 0.008)
                
                # Calculate momentum-adjusted trailing stop
                adjusted_trail_pct = calculate_momentum_adjusted_trailing_stop(
                    base_trail_pct, momentum_strength, 
                    MomentumExitConfig(**exit_cfg.get('momentum_continuation', {}))
                )
                
                if 'trailing_stop_factor' in exit_cfg:
                    if position_side == "buy":
                        trailed = calculate_atr_trailing_stop(
                            df,
                            exit_cfg['trailing_stop_factor'],
                        )
                    else:  # Short position
                        trailed = calculate_atr_trailing_stop_short(
                            df,
                            exit_cfg['trailing_stop_factor'],
                        )
                else:
                    if position_side == "buy":
                        trailed = calculate_trailing_stop(
                            df['close'],
                            adjusted_trail_pct,
                        )
                    else:  # Short position
                        trailed = calculate_trailing_stop_short(
                            df['close'],
                            adjusted_trail_pct,
                        )
            else:
                # Legacy trailing stop logic
                if 'trailing_stop_factor' in exit_cfg:
                    if position_side == "buy":
                        trailed = calculate_atr_trailing_stop(
                            df,
                            exit_cfg['trailing_stop_factor'],
                        )
                    else:  # Short position
                        trailed = calculate_atr_trailing_stop_short(
                            df,
                            exit_cfg['trailing_stop_factor'],
                        )
                else:
                    if position_side == "buy":
                        trailed = calculate_trailing_stop(
                            df['close'],
                            exit_cfg['trailing_stop_pct'],
                        )
                    else:  # Short position
                        trailed = calculate_trailing_stop_short(
                            df['close'],
                            exit_cfg['trailing_stop_pct'],
                        )
            
            # Update stop only if it's better (higher for long, lower for short)
            if position_side == "buy" and trailed > trailing_stop:
                new_stop = trailed
                logger.info("Trailing stop moved to %.4f", new_stop)
            elif position_side == "sell" and trailed < trailing_stop:
                new_stop = trailed
                logger.info("Trailing stop moved to %.4f", new_stop)
    
    return exit_signal, new_stop


def get_partial_exit_percent(pnl_pct: float) -> int:
    """Return percent of position to close based on profit.

    Parameters
    ----------
    pnl_pct : float
        Unrealized profit or loss percentage.

    Returns
    -------
    int
        Portion of the position to close expressed as a percentage.
    """
    if pnl_pct > 100:
        return 50
    if pnl_pct > 50:
        return 30
    if pnl_pct > 25:
        return 20
    return 0


def get_momentum_based_partial_exit(
    pnl_pct: float, 
    momentum_strength: float, 
    config: dict
) -> Tuple[int, float]:
    """Get momentum-based partial exit percentage and profit target."""
    
    exit_cfg = config.get('exit_strategy', {})
    
    if not exit_cfg.get('momentum_partial_exits', False):
        # Fall back to standard partial exit logic
        exit_pct = get_partial_exit_percent(pnl_pct)
        return exit_pct, 0.0
    
    # Get momentum-based partial exit thresholds
    thresholds = exit_cfg.get('partial_exit_momentum_thresholds', [])
    
    for threshold in thresholds:
        if momentum_strength >= threshold['momentum'] and pnl_pct >= threshold['profit_pct'] * 100:
            return threshold['exit_pct'], threshold['profit_pct']
    
    return 0, 0.0


def should_delay_exit_for_momentum(
    momentum_strength: float, 
    config: dict
) -> Tuple[bool, int]:
    """Determine if exit should be delayed due to strong momentum."""
    
    exit_cfg = config.get('exit_strategy', {})
    
    if not exit_cfg.get('momentum_exit_delays', {}).get('enabled', False):
        return False, 0
    
    delays = exit_cfg['momentum_exit_delays']
    
    if momentum_strength > 0.85:
        delay_seconds = delays.get('very_strong_momentum_delay_seconds', 60)
        return True, delay_seconds
    elif momentum_strength > 0.7:
        delay_seconds = delays.get('strong_momentum_delay_seconds', 30)
        return True, delay_seconds
    
    return False, 0
