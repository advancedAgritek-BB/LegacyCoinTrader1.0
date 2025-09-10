import numpy as _np

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
