import numpy as _np

def rsi(series, window=14):
    # Minimal, non-robust RSI for tests (returns mid value if short)
    try:
        values = list(series)
        if len(values) < 2:
            return _np.array([50.0] * len(values))
        deltas = _np.diff(values)
        up = _np.clip(deltas, 0, None)
        down = -_np.clip(deltas, None, 0)
        up_mean = _np.convolve(up, _np.ones(window), "full")[: len(up)] / window
        down_mean = _np.convolve(down, _np.ones(window), "full")[: len(down)] / window
        rs = _np.divide(up_mean, _np.maximum(down_mean, 1e-9))
        rsi_vals = 100.0 - (100.0 / (1.0 + rs))
        return _np.concatenate([[50.0], rsi_vals])
    except Exception:
        return _np.array([50.0])

