import numpy as _np

def ema_indicator(series, window=14):
    try:
        values = _np.array(series, dtype=float)
        if len(values) == 0:
            return values
        alpha = 2 / (window + 1)
        ema = _np.empty_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema
    except Exception:
        return _np.array([])

def adx(high, low, close, window=14):
    # Simplified ADX placeholder returning mid value array
    n = len(close)
    return _np.array([25.0] * n)

def macd(series, fast=12, slow=26):
    values = _np.array(series, dtype=float)
    ema_fast = ema_indicator(values, fast)
    ema_slow = ema_indicator(values, slow)
    return ema_fast - ema_slow

def macd_signal(series, fast=12, slow=26, signal=9):
    macd_vals = macd(series, fast, slow)
    return ema_indicator(macd_vals, signal)

