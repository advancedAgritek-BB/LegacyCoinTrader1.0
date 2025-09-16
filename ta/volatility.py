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

        # Use proper rolling calculations for financial time series
        # Initialize arrays to store results
        ma = _np.full(len(close), _np.nan)
        std = _np.full(len(close), _np.nan)

        # Calculate rolling mean and std for valid windows
        for i in range(n-1, len(close)):
            window_data = close[i-n+1:i+1]
            ma[i] = _np.mean(window_data)
            std[i] = _np.std(window_data, ddof=1)  # Sample standard deviation

        # Fill initial values with the first valid calculation
        first_valid_idx = n-1
        ma[:first_valid_idx] = ma[first_valid_idx]
        std[:first_valid_idx] = std[first_valid_idx]

        upper = ma + self._ndev * std
        lower = ma - self._ndev * std
        width = (upper - lower) / _np.maximum(_np.abs(ma), 1e-9)
        return width


class KeltnerChannel:
    """Simplified Keltner Channel indicator."""
    def __init__(self, high, low, close, window=14):
        self.high = _np.array(high, dtype=float)
        self.low = _np.array(low, dtype=float)
        self.close = _np.array(close, dtype=float)
        self.window = window
    
    def keltner_channel_hband(self):
        """Return upper band."""
        atr = average_true_range(self.high, self.low, self.close, self.window)
        kernel = _np.ones(self.window) / self.window
        ma = _np.convolve(self.close, kernel, mode="same")
        return ma + 2 * atr
    
    def keltner_channel_lband(self):
        """Return lower band."""
        atr = average_true_range(self.high, self.low, self.close, self.window)
        kernel = _np.ones(self.window) / self.window
        ma = _np.convolve(self.close, kernel, mode="same")
        return ma - 2 * atr


class VolumeWeightedAveragePrice:
    """Simplified VWAP indicator."""
    def __init__(self, high, low, close, volume, window=14):
        self.high = _np.array(high, dtype=float)
        self.low = _np.array(low, dtype=float)
        self.close = _np.array(close, dtype=float)
        self.volume = _np.array(volume, dtype=float)
        self.window = window
    
    def volume_weighted_average_price(self):
        """Return VWAP values."""
        typical_price = (self.high + self.low + self.close) / 3
        kernel = _np.ones(self.window) / self.window
        vwap = _np.convolve(typical_price * self.volume, kernel, mode="same") / _np.maximum(_np.convolve(self.volume, kernel, mode="same"), 1e-9)
        return vwap

