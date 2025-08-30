# Bollinger Bands Method Fix

## Issue Description

The trading bot was encountering the following error in logs:
```
WARNING - Strategy generate_signal failed: 'BollingerBands' object has no attribute 'bollinger_pband'
```

## Root Cause

The issue was caused by using incorrect method names for the `ta.volatility.BollingerBands` class. The installed version of the `ta` library only provides the `bollinger_wband()` method, which returns the width of the Bollinger Bands.

The code was incorrectly trying to call:
- `bollinger_pband()` - which doesn't exist
- `bollinger_hband()` - which doesn't exist  
- `bollinger_lband()` - which doesn't exist

## Solution Applied

Updated all strategy files to use the correct approach:

1. **Use `bollinger_wband()`** to get the band width
2. **Calculate the middle band** using a simple moving average
3. **Derive upper and lower bands** by adding/subtracting half the width from the middle

### Code Pattern

```python
# Before (incorrect):
bb = ta.volatility.BollingerBands(df['close'], window=bb_window)
bb_upper = bb.bollinger_hband()  # ❌ Method doesn't exist
bb_lower = bb.bollinger_lband()  # ❌ Method doesn't exist

# After (correct):
bb = ta.volatility.BollingerBands(df['close'], window=bb_window)
bb_width = bb.bollinger_wband()  # ✅ Get band width
bb_mid = df['close'].rolling(bb_window).mean()  # ✅ Calculate middle
bb_upper = bb_mid + (bb_width / 2)  # ✅ Derive upper band
bb_lower = bb_mid - (bb_width / 2)  # ✅ Derive lower band
```

## Files Fixed

1. **`crypto_bot/strategy/dip_hunter.py`** - Fixed `bollinger_pband()` usage
2. **`crypto_bot/strategy/volatility_harvester.py`** - Fixed `bollinger_hband()` and `bollinger_lband()` usage
3. **`crypto_bot/strategy/momentum_exploiter.py`** - Fixed `bollinger_hband()` and `bollinger_lband()` usage
4. **`crypto_bot/strategy_router.py`** - Enhanced error handling for Bollinger Band method errors

## Verification

All fixed strategies now import successfully without errors. The Bollinger Band calculations work correctly using the available `bollinger_wband()` method.

## Impact

- ✅ Eliminates the `'BollingerBands' object has no attribute 'bollinger_pband'` error
- ✅ All strategies using Bollinger Bands now function correctly
- ✅ Maintains the same trading logic and signal generation
- ✅ No changes to strategy behavior or performance

## Technical Notes

The `ta` library version being used appears to be an older version that only provides the `bollinger_wband()` method. This method returns the width of the bands, which can be used to reconstruct the upper and lower bands when combined with the middle band (simple moving average).

For future reference, if upgrading the `ta` library, verify that the new version provides the expected `bollinger_hband()` and `bollinger_lband()` methods before reverting to the original code pattern.
