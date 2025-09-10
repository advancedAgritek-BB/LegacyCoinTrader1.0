import pandas as pd


def zscore(series, lookback: int = 250):
    """Return z-score relative to the last ``lookback`` observations."""
    import numpy as np
    
    # Convert to numpy array if it's not already
    if hasattr(series, 'values'):
        # It's a pandas Series
        values = series.values
        original_index = series.index
    else:
        # It's already a numpy array
        values = np.array(series)
        original_index = range(len(values))
    
    if lookback <= 0:
        return pd.Series([0.0] * len(values), index=original_index)
    
    if len(values) < lookback:
        window = values
    else:
        window = values[-lookback:]
    
    std = np.std(window)
    if std == 0 or np.isnan(std):
        return pd.Series([0.0] * len(values), index=original_index)
    
    mean = np.mean(window)
    result = (values - mean) / (std if std else 1.0)
    
    return pd.Series(result, index=original_index)


def last_window_zscore(series: pd.Series, lookback: int = 250) -> float:
    """Return the last z-score value relative to the last ``lookback`` observations."""
    z_series = zscore(series, lookback)
    if z_series.empty:
        return float('nan')
    return float(z_series.iloc[-1])
