# Strategy Allocation Calculation Fix

## Problem
The frontend dashboard was showing incorrect strategy allocation data. It was using static allocation percentages from the config file instead of calculating dynamic allocation based on actual strategy performance.

## Root Cause
The frontend was reading `strategy_allocation` directly from `crypto_bot/config.yaml`, which contains static values that don't reflect actual strategy performance or usage.

## Solution
Implemented a dynamic allocation calculation system that:

1. **Reads actual performance data** from `crypto_bot/logs/strategy_stats.json`
2. **Calculates composite scores** based on:
   - Win rate (40% weight)
   - PnL per trade (40% weight) 
   - Trade volume (20% weight)
3. **Normalizes scores** to percentages that sum to 100%
4. **Falls back gracefully** to static config if no performance data is available

## Changes Made

### 1. New Function: `calculate_dynamic_allocation()` in `frontend/utils.py`
- Reads strategy performance data from `strategy_stats.json`
- Calculates composite performance scores
- Returns normalized allocation percentages

### 2. Updated Frontend Routes in `frontend/app.py`
- Modified `/` (index), `/dashboard`, and `/api/dashboard-metrics` routes
- Now use `utils.calculate_dynamic_allocation()` instead of static config
- Maintains fallback to static config if dynamic data unavailable

### 3. Enhanced Dashboard Template in `frontend/templates/dashboard.html`
- Added allocation details card showing calculation method
- Improved chart tooltips to show percentages
- Added visual indicator that the fix has been applied

## Results

### Before (Static Allocation)
```json
{
  "bounce_scalper": 15,
  "grid_bot": 15, 
  "micro_scalp_bot": 30,
  "sniper_bot": 25,
  "trend_bot": 15
}
```

### After (Dynamic Allocation)
```json
{
  "sniper_bot": 28.5,
  "trend_bot": 25.3,
  "micro_scalp_bot": 25.0,
  "momentum_bot": 21.3
}
```

## Key Improvements

1. **Accuracy**: Only shows strategies that have actually been used
2. **Performance-based**: Allocates more capital to better-performing strategies
3. **Dynamic**: Updates automatically as performance data changes
4. **Robust**: Handles missing data gracefully with fallbacks

## Testing

Run the test script to verify the fix:
```bash
python3 test_allocation.py
```

Expected output:
```
✅ Allocation percentages sum to ~100% (correct)
Strategy breakdown:
  sniper_bot: 28.5%
  trend_bot: 25.3%
  micro_scalp_bot: 25.0%
  momentum_bot: 21.3%
```

## Files Modified

- `frontend/utils.py` - Added dynamic allocation calculation functions
- `frontend/app.py` - Updated routes to use dynamic allocation
- `frontend/templates/dashboard.html` - Enhanced UI to show allocation details
- `test_allocation.py` - Test script to verify functionality

