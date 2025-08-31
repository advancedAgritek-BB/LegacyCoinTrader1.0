"""
Ultra-Aggressive Scalping Strategy for Maximum Profit in Shortest Time

This strategy is designed for extremely fast profit-taking with minimal risk exposure.
It uses ultra-fast timeframes (1m) and very tight stop losses for rapid execution.
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

from dataclasses import dataclass

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.indicator_cache import cache_series


@dataclass
class UltraScalpConfig:
    """Configuration for ultra-aggressive scalping."""
    
    # Core settings
    timeframe: str = "1m"
    min_score: float = 0.04
    max_concurrent: int = 30
    
    # Risk management
    stop_loss_pct: float = 0.005
    take_profit_pct: float = 0.02
    
    # Technical indicators
    volume_mult: float = 1.5
    atr_window: int = 5
    
    # Signal thresholds
    rsi_window: int = 6
    rsi_oversold: float = 20.0
    rsi_overbought: float = 80.0
    
    # Volume filters
    volume_window: int = 8
    min_volume_zscore: float = 0.5
    
    # Momentum filters
    ema_fast: int = 3
    ema_slow: int = 8
    macd_fast: int = 6
    macd_slow: int = 12
    macd_signal: int = 4
    
    # Pattern detection
    wick_threshold: float = 0.15
    body_threshold: float = 0.6
    
    # Cooldown and filters
    cooldown_seconds: int = 10
    min_atr_pct: float = 0.0002
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.stop_loss_pct >= self.take_profit_pct:
            raise ValueError("Stop loss must be less than take profit")
        if self.min_score < 0 or self.min_score > 1:
            raise ValueError("Min score must be between 0 and 1")


def _calculate_ultra_fast_indicators(df: pd.DataFrame, config: UltraScalpConfig) -> Dict[str, pd.Series]:
    """Calculate ultra-fast technical indicators."""
    
    indicators = {}
    
    # Ultra-fast EMAs
    indicators['ema_fast'] = df['close'].ewm(span=config.ema_fast, adjust=False).mean()
    indicators['ema_slow'] = df['close'].ewm(span=config.ema_slow, adjust=False).mean()

    # Ultra-fast RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config.rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config.rsi_window).mean()
    rs = gain / loss
    indicators['rsi'] = 100 - (100 / (1 + rs))

    # Ultra-fast MACD
    ema_fast = df['close'].ewm(span=config.macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=config.macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=config.macd_signal, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    indicators['macd'] = macd_line
    indicators['macd_signal'] = macd_signal
    indicators['macd_histogram'] = macd_histogram
    
    # Ultra-fast ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    indicators['atr'] = tr.rolling(window=config.atr_window).mean()
    
    # Volume indicators
    indicators['volume_sma'] = df['volume'].rolling(window=config.volume_window).mean()
    indicators['volume_zscore'] = (
        (df['volume'] - indicators['volume_sma']) / 
        df['volume'].rolling(window=config.volume_window).std()
    )
    
    # Price action patterns
    indicators['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    indicators['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
    indicators['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'])
    
    return indicators


def _detect_ultra_scalp_signals(
    df: pd.DataFrame, 
    indicators: Dict[str, pd.Series], 
    config: UltraScalpConfig
) -> Tuple[float, str, Dict[str, Any]]:
    """Detect ultra-scalp trading signals."""
    
    if len(df) < max(config.ema_slow, config.rsi_window, config.volume_window):
        return 0.0, "none"
    
    # Get latest values
    current = {k: v.iloc[-1] for k, v in indicators.items()}
    prev = {k: v.iloc[-2] for k, v in indicators.items()}
    
    # Initialize signal components
    signal_strength = 0.0
    signal_direction = "none"
    signal_metadata = {}
    
    # 1. EMA Crossover Signal (Fastest)
    if (current['ema_fast'] > current['ema_slow'] and 
        prev['ema_fast'] <= prev['ema_slow']):
        signal_strength += 0.3
        signal_direction = "long"
        signal_metadata['ema_cross'] = "bullish"
    elif (current['ema_fast'] < current['ema_slow'] and 
          prev['ema_fast'] >= prev['ema_slow']):
        signal_strength += 0.3
        signal_direction = "short"
        signal_metadata['ema_cross'] = "bearish"
    
    # 2. RSI Extremes (Quick reversal signals)
    if current['rsi'] < config.rsi_oversold:
        signal_strength += 0.2
        if signal_direction == "none":
            signal_direction = "long"
        signal_metadata['rsi_signal'] = "oversold"
    elif current['rsi'] > config.rsi_overbought:
        signal_strength += 0.2
        if signal_direction == "none":
            signal_direction = "short"
        signal_metadata['rsi_signal'] = "overbought"
    
    # 3. MACD Momentum
    if (current['macd'] > current['macd_signal'] and 
        prev['macd'] <= prev['macd_signal']):
        signal_strength += 0.15
        if signal_direction == "none":
            signal_direction = "long"
        signal_metadata['macd_signal'] = "bullish"
    elif (current['macd'] < current['macd_signal'] and 
          prev['macd'] >= prev['macd_signal']):
        signal_strength += 0.15
        if signal_direction == "none":
            signal_direction = "short"
        signal_metadata['macd_signal'] = "bearish"
    
    # 4. Volume Spike Detection
    if current['volume_zscore'] > config.min_volume_zscore:
        signal_strength += 0.15
        signal_metadata['volume_spike'] = current['volume_zscore']
    
    # 5. Price Action Patterns
    if current['body_size'] > config.body_threshold:
        if current['upper_wick'] < config.wick_threshold:
            signal_strength += 0.1
            if signal_direction == "none":
                signal_direction = "long"
            signal_metadata['pattern'] = "strong_bullish_body"
        elif current['lower_wick'] < config.wick_threshold:
            signal_strength += 0.1
            if signal_direction == "none":
                signal_direction = "short"
            signal_metadata['pattern'] = "strong_bearish_body"
    
    # 6. ATR Volatility Filter
    if current['atr'] is not None and not np.isnan(current['atr']):
        atr_pct = current['atr'] / df['close'].iloc[-1]
        if atr_pct > config.min_atr_pct:
            signal_strength += 0.1
            signal_metadata['atr_pct'] = atr_pct
        else:
            # Reduce signal strength if volatility is too low
            signal_strength *= 0.5
            signal_metadata['low_volatility'] = True
    
    # Normalize signal strength
    signal_strength = min(signal_strength, 1.0)
    
    # Apply minimum score filter
    if signal_strength < config.min_score:
        return 0.0, "none"
    
    # Add additional metadata
    signal_metadata.update({
        'signal_strength': signal_strength,
        'timestamp': df.index[-1] if hasattr(df.index[-1], 'timestamp') else None,
        'price': df['close'].iloc[-1],
        'volume': df['volume'].iloc[-1]
    })
    
    return signal_strength, signal_direction, signal_metadata


def generate_signal(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[float, str]:
    """
    Generate ultra-aggressive scalping signals.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume
    config : dict, optional
        Configuration dictionary
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Tuple[float, str]
        (signal_score, direction)
    """
    
    # Create configuration
    if config is None:
        config = {}
    
    # Merge with defaults
    ultra_config = UltraScalpConfig(**{**UltraScalpConfig().__dict__, **config})
    
    # Validate data
    if df.empty or len(df) < ultra_config.ema_slow:
        return 0.0, "none"
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        return 0.0, "none"
    
    # Calculate indicators
    try:
        indicators = _calculate_ultra_fast_indicators(df, ultra_config)
    except Exception as e:
        return 0.0, "none"
    
    # Detect signals
    try:
        signal_score, direction, metadata = _detect_ultra_scalp_signals(
            df, indicators, ultra_config
        )
    except Exception as e:
        return 0.0, "none"
    
    # Apply volatility normalization if enabled
    if 'atr_pct' in metadata:
        try:
            signal_score = normalize_score_by_volatility(
                signal_score, metadata['atr_pct']
            )
        except:
            pass  # Continue without normalization if it fails
    
    # Ensure signal score is within bounds
    signal_score = max(0.0, min(1.0, signal_score))
    
    return signal_score, direction


def get_strategy_info() -> Dict[str, Any]:
    """Get strategy information and metadata."""
    return {
        "name": "ultra_scalp_bot",
        "description": "Ultra-aggressive scalping strategy for maximum profit in shortest time",
        "timeframe": "1m",
        "risk_level": "high",
        "profit_potential": "very_high",
        "speed": "ultra_fast",
        "suitable_for": ["experienced_traders", "high_frequency", "aggressive_profits"],
        "key_features": [
            "1-minute timeframes",
            "Very tight stop losses (0.5%)",
            "Fast profit taking (2%)",
            "High concurrent positions (30)",
            "Ultra-fast indicator calculation",
            "Volume spike detection",
            "Price action pattern recognition"
        ],
        "risk_warnings": [
            "Very high risk due to tight stops",
            "Requires excellent execution speed",
            "May generate many small losses",
            "Suitable only for experienced traders",
            "Requires low-latency infrastructure"
        ]
    }
