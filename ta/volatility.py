import numpy as _np

def average_true_range(high, low, close, window=14):
    high = _np.array(high, dtype=float)
    low = _np.array(low, dtype=float)
    close = _np.array(close, dtype=float)
    trs = _np.maximum(high - low, _np.maximum(_np.abs(high - _np.roll(close, 1)), _np.abs(low - _np.roll(close, 1))))
    trs[0] = high[0] - low[0]
    # Simple moving average
    kernel = _np.ones(window) / window
    atr = _np.convolve(trs, kernel, mode="same")
    return atr

class BollingerBands:
    def __init__(self, close, window=20, ndev=2):
        self._close = _np.array(close, dtype=float)
        self._window = window
        self._ndev = ndev

    def bollinger_wband(self):
        n = self._window
        close = self._close
        if len(close) < n:
            return _np.array([0.0] * len(close))
        kernel = _np.ones(n) / n
        ma = _np.convolve(close, kernel, mode="same")
        # crude rolling std approximation
        var = _np.convolve((close - ma) ** 2, kernel, mode="same")
        std = _np.sqrt(_np.maximum(var, 1e-12))
        upper = ma + self._ndev * std
        lower = ma - self._ndev * std
        width = (upper - lower) / _np.maximum(ma, 1e-9)
        return width

