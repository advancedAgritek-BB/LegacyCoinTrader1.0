# Chart Update Optimization for 5-Minute Intervals

## Overview
This update optimizes the chart updates on open positions cards to only refresh when new 5-minute candle data is available, instead of updating every 30 seconds. This reduces unnecessary API calls and improves performance while ensuring charts are always current with the latest market data.

## Changes Made

### 1. Frontend Dashboard Template (`frontend/templates/dashboard.html`)

#### Update Intervals
- **Position refresh interval**: Changed from 30 seconds to 5 minutes (300,000ms)
- **Main data update interval**: Changed from 10 seconds to 5 minutes (300,000ms)

#### New Chart Update Logic
- Added `hasNewCandleData()` function to check if new 5-minute candle data is available
- Modified `renderChart()` function to skip updates when no new data is available
- Added visual timestamp indicators on each chart showing last update time
- Added "Charts update every 5m" indicator in the Open Positions header

#### Chart Update Tracking
- Added `window.lastChartUpdates` to track last update times per symbol
- Added `updateChartLastUpdate()` function to display update timestamps
- Charts now show "Updated: HH:MM:SS" in bottom-right corner

### 2. Frontend API (`frontend/app.py`)

#### New API Endpoint
- **`/api/candle-timestamp`**: Returns the timestamp of the most recent 5-minute candle for a symbol
- **`get_latest_candle_timestamp()`**: Fetches real market data or falls back to calculated 5-minute boundary

#### Implementation Details
- Uses the enhanced OHLCV fetcher to get real market data
- Falls back to calculated 5-minute boundary if real data unavailable
- Handles authentication errors gracefully
- Returns Unix timestamp for easy comparison

### 3. Main App Updates (`frontend/static/app.js`)

#### Update Frequency Changes
- Changed main update interval from 10 seconds to 5 minutes
- Updated comments to reflect 5-minute intervals
- Maintained single update interval to prevent conflicts

## Benefits

### Performance Improvements
- **Reduced API calls**: Charts only update when new data is available
- **Lower server load**: Fewer unnecessary data fetches
- **Better user experience**: Less frequent UI updates, more stable display

### Data Accuracy
- **Always current**: Charts update exactly when new 5-minute candles form
- **No stale data**: Prevents displaying outdated chart information
- **Consistent timing**: Aligns with actual market data intervals

### User Experience
- **Visual feedback**: Users can see when each chart was last updated
- **Clear expectations**: "Charts update every 5m" indicator sets proper expectations
- **Manual refresh**: Users can still manually refresh if needed

## Technical Implementation

### Chart Update Flow
1. **Check for new data**: `hasNewCandleData()` queries `/api/candle-timestamp`
2. **Compare timestamps**: Only update if new candle timestamp > last update
3. **Fetch trend data**: If new data available, fetch from `/api/trend-data`
4. **Render chart**: Update canvas and timestamp display
5. **Skip if no new data**: Log "SKIPPING chart update" message

### Error Handling
- **API failures**: Graceful fallback to calculated timestamps
- **Network issues**: Charts remain functional with last known data
- **Authentication errors**: Prevents fallback to inaccurate mock data

### Monitoring
- **Console logging**: Clear messages when charts are skipped vs updated
- **Visual indicators**: Timestamp display shows last update time
- **Performance tracking**: Reduced API calls are logged

## Testing

### Test Script
Created `test_chart_update_optimization.py` to verify:
- New candle timestamp API endpoint functionality
- Trend data API still works correctly
- Timestamp accuracy and recency

### Manual Testing Steps
1. Start frontend server: `python frontend/app.py`
2. Open dashboard in browser
3. Monitor console for "SKIPPING chart update" messages
4. Verify chart timestamps update every 5 minutes
5. Check that manual refresh still works

## Configuration

### Update Intervals
- **Position cards**: 5 minutes (300,000ms)
- **Main dashboard**: 5 minutes (300,000ms)
- **Chart updates**: Only when new 5-minute candle data available

### API Endpoints
- **`/api/candle-timestamp`**: Check for new data
- **`/api/trend-data`**: Fetch chart data (unchanged)
- **`/api/open-positions`**: Fetch position data (unchanged)

## Future Enhancements

### Potential Improvements
- **WebSocket integration**: Real-time updates when new candles form
- **Multiple timeframes**: Support for different chart intervals
- **Caching**: Cache chart data to reduce API calls further
- **User preferences**: Allow users to set custom update intervals

### Monitoring
- **Metrics tracking**: Log chart update frequency and performance
- **Alert system**: Notify when charts are not updating as expected
- **Health checks**: Monitor API endpoint availability

## Conclusion

This optimization significantly improves the trading dashboard's performance and user experience by ensuring charts only update when meaningful new data is available. The 5-minute interval aligns with standard market data intervals and provides a good balance between data freshness and system performance.
