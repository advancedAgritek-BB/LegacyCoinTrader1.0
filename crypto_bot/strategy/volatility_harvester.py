"""
Volatility Harvesting Strategy for Maximum Profit in Shortest Time

This strategy identifies and exploits periods of high volatility, using ATR-based
signals and volume spikes to capture quick profits from market turbulence.
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

from dataclasses import dataclass

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.indicator_cache import cache_series


@dataclass
class VolatilityHarvesterConfig:
    """Configuration for volatility harvesting strategy."""
    
    # Core settings
    atr_threshold: float = 0.002
    volume_spike: float = 2.0
    max_positions: int = 20
    
    # Risk management
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.04
    
    # Volatility detection
    atr_window: int = 14
    atr_multiplier: float = 1.5
    volatility_lookback: int = 20
    
    # Volume analysis
    volume_window: int = 15
    volume_zscore_threshold: float = 1.8
    
    # Price action
    price_range_window: int = 10
    range_expansion_threshold: float = 0.8
    
    # Bollinger Bands
    bb_window: int = 20
    bb_std_dev: float = 2.0
    
    # Filters
    min_price_change: float = 0.001
    max_spread_pct: float = 0.3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.stop_loss_pct >= self.take_profit_pct:
            raise ValueError("Stop loss must be less than take profit")
        if self.atr_threshold < 0:
            raise ValueError("ATR threshold must be positive")


def _calculate_volatility_indicators(df: pd.DataFrame, config: VolatilityHarvesterConfig) -> Dict[str, pd.Series]:
    """Calculate volatility-related technical indicators."""
    
    indicators = {}
    
    # Calculate ATR manually
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    indicators['atr'] = tr.rolling(window=config.atr_window).mean()
    indicators['atr_pct'] = indicators['atr'] / df['close']
    
    # ATR moving averages for volatility trend
    indicators['atr_sma'] = indicators['atr'].rolling(window=config.volatility_lookback).mean()
    indicators['atr_ratio'] = indicators['atr'] / indicators['atr_sma']
    
    # Price range analysis
    indicators['price_range'] = df['high'] - df['low']
    indicators['price_range_pct'] = indicators['price_range'] / df['close']
    indicators['range_sma'] = indicators['price_range_pct'].rolling(window=config.price_range_window).mean()
    indicators['range_expansion'] = indicators['price_range_pct'] / indicators['range_sma']
    
    # Volume volatility
    indicators['volume_sma'] = df['volume'].rolling(window=config.volume_window).mean()
    indicators['volume_std'] = df['volume'].rolling(window=config.volume_window).std()
    indicators['volume_zscore'] = (
        (df['volume'] - indicators['volume_sma']) / indicators['volume_std']
    )
    
    # Calculate Bollinger Bands manually
    bb_mid = df['close'].rolling(config.bb_window).mean()
    bb_std = df['close'].rolling(config.bb_window).std()
    bb_width = (bb_std * config.bb_std_dev * 2) / bb_mid
    indicators['bb_upper'] = bb_mid + (bb_std * config.bb_std_dev)
    indicators['bb_lower'] = bb_mid - (bb_std * config.bb_std_dev)
    indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / df['close']
    indicators['bb_position'] = (df['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
    
    # Calculate Keltner Channels manually
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    # Use the already calculated ATR
    indicators['kc_upper'] = typical_price + indicators['atr'] * 2
    indicators['kc_lower'] = typical_price - indicators['atr'] * 2
    indicators['kc_width'] = (indicators['kc_upper'] - indicators['kc_lower']) / df['close']
    
    # Price momentum in volatile conditions
    indicators['price_change'] = df['close'].pct_change()
    indicators['price_momentum'] = df['close'].pct_change(periods=5)
    
    # Volatility-adjusted RSI
    # Calculate RSI manually
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['rsi'] = 100 - (100 / (1 + rs))
    
    return indicators


def _detect_volatility_signals(
    df: pd.DataFrame, 
    indicators: Dict[str, pd.Series], 
    config: VolatilityHarvesterConfig
) -> Tuple[float, str, Dict[str, Any]]:
    """Detect volatility harvesting trading signals."""
    
    if len(df) < max(config.atr_window, config.volatility_lookback, config.volume_window):
        return 0.0, "none"
    
    # Get latest values
    current = {k: v.iloc[-1] for k, v in indicators.items()}
    prev = {k: v.iloc[-2] for k, v in indicators.items()}
    
    # Initialize signal components
    signal_strength = 0.0
    signal_direction = "none"
    signal_metadata = {}

    # Precompute rolling comparisons for band/channel width checks
    bb_width_series = indicators.get('bb_width')
    kc_width_series = indicators.get('kc_width')
    bb_width_mean_series = None
    kc_width_mean_series = None
    if bb_width_series is not None:
        bb_width_mean_series = bb_width_series.rolling(window=20).mean()
    if kc_width_series is not None:
        kc_width_mean_series = kc_width_series.rolling(window=20).mean()
    
    # 1. ATR Volatility Detection (Primary Signal)
    if current['atr_pct'] > config.atr_threshold:
        signal_strength += 0.3
        signal_metadata['volatility_type'] = "high_atr"
        signal_metadata['atr_pct'] = current['atr_pct']
        
        # Determine direction based on price action
        if current['price_momentum'] > config.min_price_change:
            signal_direction = "long"
            signal_metadata['direction_reason'] = "bullish_momentum"
        elif current['price_momentum'] < -config.min_price_change:
            signal_direction = "short"
            signal_metadata['direction_reason'] = "bearish_momentum"
    
    # 2. ATR Ratio Expansion
    if current['atr_ratio'] > config.atr_multiplier:
        signal_strength += 0.25
        signal_metadata['atr_expansion'] = current['atr_ratio']
        
        if signal_direction == "none":
            # Use price momentum for direction if not set
            if current['price_momentum'] > 0:
                signal_direction = "long"
            else:
                signal_direction = "short"
    
    # 3. Price Range Expansion
    if current['range_expansion'] > config.range_expansion_threshold:
        signal_strength += 0.2
        signal_metadata['range_expansion'] = current['range_expansion']
        
        # Range expansion often precedes breakouts
        if signal_direction == "none":
            if current['price_momentum'] > 0:
                signal_direction = "long"
            else:
                signal_direction = "short"
    
    # 4. Volume Spike Confirmation
    if current['volume_zscore'] > config.volume_zscore_threshold:
        signal_strength += 0.2
        signal_metadata['volume_spike'] = current['volume_zscore']
        
        # Volume-price alignment check
        if (signal_direction == "long" and current['price_change'] > 0) or \
           (signal_direction == "short" and current['price_change'] < 0):
            signal_strength += 0.1
            signal_metadata['volume_price_aligned'] = True
    
    # 5. Bollinger Band Volatility
    if (
        bb_width_series is not None
        and bb_width_mean_series is not None
        and not pd.isna(bb_width_series.iloc[-1])
        and not pd.isna(bb_width_mean_series.iloc[-1])
        and bb_width_series.iloc[-1] > bb_width_mean_series.iloc[-1]
    ):
        signal_strength += 0.15
        signal_metadata['bb_expansion'] = True
        
        # BB position for direction
        if current['bb_position'] < 0.2:  # Near lower band
            if signal_direction == "none":
                signal_direction = "long"
            signal_metadata['bb_signal'] = "near_lower_band"
        elif current['bb_position'] > 0.8:  # Near upper band
            if signal_direction == "none":
                signal_direction = "short"
            signal_metadata['bb_signal'] = "near_upper_band"
    
    # 6. Keltner Channel Expansion
    if (
        kc_width_series is not None
        and kc_width_mean_series is not None
        and not pd.isna(kc_width_series.iloc[-1])
        and not pd.isna(kc_width_mean_series.iloc[-1])
        and kc_width_series.iloc[-1] > kc_width_mean_series.iloc[-1]
    ):
        signal_strength += 0.1
        signal_metadata['kc_expansion'] = True
        
        # KC position for direction
        if df['close'].iloc[-1] < current['kc_lower']:
            if signal_direction == "none":
                signal_direction = "long"
            signal_metadata['kc_signal'] = "below_lower_channel"
        elif df['close'].iloc[-1] > current['kc_upper']:
            if signal_direction == "none":
                signal_direction = "short"
            signal_metadata['kc_signal'] = "above_upper_channel"
    
    # 7. RSI Extremes in Volatile Conditions
    if current['atr_pct'] > config.atr_threshold:
        if current['rsi'] < 30:
            signal_strength += 0.1
            if signal_direction == "none":
                signal_direction = "long"
            signal_metadata['rsi_signal'] = "oversold_in_volatility"
        elif current['rsi'] > 70:
            signal_strength += 0.1
            if signal_direction == "none":
                signal_direction = "short"
            signal_metadata['rsi_signal'] = "overbought_in_volatility"
    
    # 8. Price Action Confirmation
    if signal_direction == "long" and current['price_change'] > 0:
        signal_strength += 0.1
        signal_metadata['price_action'] = "bullish"
    elif signal_direction == "short" and current['price_change'] < 0:
        signal_strength += 0.1
        signal_metadata['price_action'] = "bearish"
    
    # Normalize signal strength
    signal_strength = min(signal_strength, 1.0)
    
    # Apply minimum volatility filter
    if current['atr_pct'] < config.atr_threshold:
        signal_strength *= 0.5  # Reduce strength if volatility is too low
        signal_metadata['low_volatility_penalty'] = True
    
    # Apply volume filter
    if current['volume_zscore'] < config.volume_spike:
        signal_strength *= 0.7  # Reduce strength if volume is not spiking
        signal_metadata['low_volume_penalty'] = True
    
    # Ensure we have a direction
    if signal_direction == "none":
        return 0.0, "none"
    
    # Add additional metadata
    signal_metadata.update({
        'signal_strength': signal_strength,
        'timestamp': df.index[-1] if hasattr(df.index[-1], 'timestamp') else None,
        'price': df['close'].iloc[-1],
        'volume': df['volume'].iloc[-1],
        'atr_pct': current['atr_pct'],
        'volume_zscore': current['volume_zscore'],
        'volatility_score': current['atr_ratio']
    })
    
    return signal_strength, signal_direction, signal_metadata


def generate_signal(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[float, str]:
    """
    Generate volatility harvesting trading signals.
    
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
    volatility_config = VolatilityHarvesterConfig(**{**VolatilityHarvesterConfig().__dict__, **config})
    
    # Validate data
    if df.empty or len(df) < volatility_config.atr_window:
        return 0.0, "none"
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        return 0.0, "none"
    
    # Calculate indicators
    try:
        indicators = _calculate_volatility_indicators(df, volatility_config)
    except Exception as e:
        return 0.0, "none"
    
    # Detect signals
    try:
        signal_score, direction, metadata = _detect_volatility_signals(
            df, indicators, volatility_config
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
        "name": "volatility_harvester",
        "description": "Volatility harvesting strategy for profiting from market turbulence",
        "timeframe": "1m-5m",
        "risk_level": "high",
        "profit_potential": "very_high",
        "speed": "fast",
        "suitable_for": ["experienced_traders", "volatility_trading", "breakout_profits"],
        "key_features": [
            "ATR-based volatility detection",
            "Volume spike confirmation",
            "Price range expansion analysis",
            "Bollinger Band volatility",
            "Keltner Channel expansion",
            "Volatility-adjusted RSI signals"
        ],
        "risk_warnings": [
            "High risk due to volatility dependency",
            "Requires quick execution",
            "May generate false signals in low volatility",
            "Suitable for experienced traders",
            "Requires good volatility timing"
        ]
    }
