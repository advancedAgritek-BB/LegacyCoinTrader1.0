"""Position exit helpers with improved trailing stop and take profit logic."""

from typing import Tuple

import pandas as pd
import ta

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.volatility_filter import calc_atr
from pathlib import Path


# Use the main bot log for exit messages
logger = setup_logger(__name__, LOG_DIR / "bot.log")


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
        Calculated trailing stop value above the low.
    """
    lowest = price_series.min()
    stop = lowest * (1 + trail_pct)
    logger.info("Calculated short trailing stop %.4f from low %.4f", stop, lowest)
    return stop


def calculate_atr_trailing_stop(df: pd.DataFrame, atr_factor: float = 2.0) -> float:
    """Return an ATR based trailing stop.

    The stop is calculated as ``highest_price_since_entry - ATR * atr_factor``.
    ``df`` should contain the OHLC data from trade entry to the current bar.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing ``high``, ``low`` and ``close`` columns.
    atr_factor : float, optional
        Multiplier applied to the ATR value.

    Returns
    -------
    float
        Calculated trailing stop using ATR.
    """
    try:
        highest = df["close"].max()
        atr = calc_atr(df)
        
        # Fallback ATR calculation if the cached version fails
        if pd.isna(atr) or atr == 0:
            logger.warning("Cached ATR failed, using fallback calculation")
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
        
        if pd.isna(atr) or atr == 0:
            logger.warning("Fallback ATR also failed, using percentage-based stop")
            return highest * 0.95  # Fallback to 5% trailing stop
        
        stop = highest - atr * atr_factor
        logger.info(
            "Calculated ATR trailing stop %.4f using high %.4f and ATR %.4f",
            stop,
            highest,
            atr,
        )
        return stop
    except Exception as e:
        logger.error(f"Error calculating ATR trailing stop: {e}")
        # Fallback to percentage-based trailing stop
        highest = df["close"].max()
        return highest * 0.95


def calculate_atr_trailing_stop_short(df: pd.DataFrame, atr_factor: float = 2.0) -> float:
    """Return an ATR based trailing stop for short positions.

    The stop is calculated as ``lowest_price_since_entry + ATR * atr_factor``.
    ``df`` should contain the OHLC data from trade entry to the current bar.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing ``high``, ``low`` and ``close`` columns.
    atr_factor : float, optional
        Multiplier applied to the ATR value.

    Returns
    -------
    float
        Calculated trailing stop using ATR for short positions.
    """
    try:
        lowest = df["close"].min()
        atr = calc_atr(df)
        
        # Fallback ATR calculation if the cached version fails
        if pd.isna(atr) or atr == 0:
            logger.warning("Cached ATR failed, using fallback calculation")
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
        
        if pd.isna(atr) or atr == 0:
            logger.warning("Fallback ATR also failed, using percentage-based stop")
            return lowest * 1.05  # Fallback to 5% trailing stop
        
        stop = lowest + atr * atr_factor
        logger.info(
            "Calculated ATR short trailing stop %.4f using low %.4f and ATR %.4f",
            stop,
            lowest,
            atr,
        )
        return stop
    except Exception as e:
        logger.error(f"Error calculating ATR short trailing stop: {e}")
        # Fallback to percentage-based trailing stop
        lowest = df["close"].min()
        return lowest * 1.05


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
    
    # Check take profit first (if configured)
    if entry_price is not None:
        exit_cfg = config.get('exit_strategy', {})
        take_profit_pct = exit_cfg.get('take_profit_pct', 0.0)
        
        if take_profit_pct > 0:
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
        # Make momentum check less restrictive - only block exit if momentum is very strong
        momentum_strength = _assess_momentum_strength(df)
        if momentum_strength < 0.7:  # Allow exit unless momentum is very strong
            logger.info(
                "Price %.4f hit trailing stop %.4f for %s position",
                current_price,
                trailing_stop,
                "long" if position_side == "buy" else "short",
            )
            exit_signal = True
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
            exit_cfg = config.get('exit_strategy', {})
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
