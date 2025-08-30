# Monitor Backlog Fixes

This document summarizes the fixes implemented to address the errors found in `crypto_bot/logs/monitor_backlog.jsonl`.

## Issues Identified

### 1. Kraken OHLCV Loading Failures
- **Problem**: Multiple "Failed to load OHLCV" errors for various pairs and timeframes
- **Specific Issues**:
  - "Too many requests" errors
  - "Missing client certificate for mTLS" errors
  - Empty error arrays []

### 2. Twitter Sentiment API Failures
- **Problem**: DNS resolution failures for placeholder URL `api.example.com`
- **Error**: `HTTPSConnectionPool(host='api.example.com', port=443): Max retries exceeded`

### 3. Alternative.me SOPR API Failures
- **Problem**: 404 errors for `https://api.alternative.me/v2/onchain/sopr`
- **Issue**: SOPR endpoint appears to be deprecated/unavailable

## Fixes Implemented

### 1. Enhanced Retry Mechanisms (`crypto_bot/utils/market_loader.py`)

**Changes:**
- Increased max retries from 3 to 5 attempts
- Added exponential backoff with configurable delays (1.0s to 30.0s max)
- Enhanced error detection for Kraken-specific issues:
  - "too many requests"
  - "missing client certificate"
  - "rate limit"
  - "temporary"/"unavailable"
  - Network-related errors (timeout, connection, DNS)
- Added `ccxt.BadRequest` to handled exceptions
- Improved logging with attempt numbers and retry reasons

**Key Functions Modified:**
- `_call_with_retry()`: Enhanced retry logic with better error classification

### 2. Twitter Sentiment API Fixes (`crypto_bot/sentiment_filter.py`)

**Changes:**
- Changed placeholder URL from `api.example.com` to `api.twitter-sentiment.invalid`
- Added detection for placeholder/unconfigured URLs
- Improved error handling with better logging
- Added timeout increase from 5s to 10s
- Better error messages distinguishing between configuration and network issues

**Key Functions Modified:**
- `fetch_twitter_sentiment()`: Enhanced with placeholder detection and better error handling

### 3. Alternative.me SOPR API Fixes (`crypto_bot/indicators/cycle_bias.py`)

**Changes:**
- Added alternative SOPR endpoint: `realized-profit-loss`
- Implemented fallback chain:
  1. Primary SOPR endpoint
  2. Alternative endpoint (if primary fails)
  3. Fallback to Fear & Greed Index
- Added logging for endpoint switching
- Improved error handling for 404 responses

**Key Functions Modified:**
- `get_cycle_bias()`: Added multi-endpoint fallback logic

### 4. Enhanced OHLCV Fetcher (`crypto_bot/utils/enhanced_ohlcv_fetcher.py`)

**Changes:**
- Added data validation for OHLCV responses
- Better handling of empty data arrays
- Improved error logging with symbol and timeframe context
- Validation of candle data structure (must have at least 6 elements)

### 5. API Error Handling Configuration (`config/api_error_handling.yaml`)

**New Configuration File:**
```yaml
# API Error Handling Configuration
kraken:
  max_retries: 5
  base_retry_delay: 1.0
  max_retry_delay: 30.0
  retryable_status_codes: [400, 429, 520, 522]
  retryable_error_patterns:
    - "too many requests"
    - "missing client certificate"
    - "rate limit"
    - "temporary"
    - "unavailable"
    - "timeout"
    - "connection"
    - "network"
    - "dns"
    - "resolve"
    - "nodename"

alternative_me:
  primary_endpoints:
    sopr: "https://api.alternative.me/v2/onchain/sopr"
  alternative_endpoints:
    sopr: "https://api.alternative.me/v2/onchain/realized-profit-loss"
  fallback_endpoints:
    all: "https://api.alternative.me/fng/?limit=1"

twitter_sentiment:
  enabled: false
  url: "https://api.twitter-sentiment.invalid/twitter-sentiment"
  placeholder_domains:
    - "api.example.com"
    - "api.twitter-sentiment.invalid"
```

### 6. Monitor Backlog Manager (`tools/monitor_backlog_manager.py`)

**New Tool Script:**
- Analyzes error patterns in the backlog file
- Generates reports on affected symbols and common errors
- Provides suggestions for configuration fixes
- Can clear old entries from the backlog

**Usage Examples:**
```bash
# Generate error report
python3 tools/monitor_backlog_manager.py --report

# Clear entries older than 7 days
python3 tools/monitor_backlog_manager.py --clear-old 7

# Get suggested fixes
python3 tools/monitor_backlog_manager.py --suggest-fixes
```

## Configuration Integration

The system now uses the `config/api_error_handling.yaml` file to:
- Configure retry parameters dynamically
- Define retryable error patterns
- Set API endpoints and fallbacks
- Enable/disable specific services

## Error Classification

Enhanced error classification now handles:
- **Retryable Errors**: Rate limits, temporary failures, network issues
- **Non-retryable Errors**: Authentication failures, invalid requests
- **Configuration Errors**: Placeholder URLs, missing API keys

## Testing Recommendations

To verify the fixes:

1. **Monitor Error Reduction**:
   - Check that `monitor_backlog.jsonl` entries decrease over time
   - Look for reduction in "too many requests" errors

2. **API Response Validation**:
   - Verify alternative.me SOPR data is fetched from alternative endpoint
   - Confirm Twitter sentiment uses neutral values when unconfigured

3. **Retry Behavior**:
   - Simulate network failures to test exponential backoff
   - Verify Kraken rate limit handling

## Future Improvements

Consider implementing:
- Circuit breaker pattern for persistently failing APIs
- API key rotation for rate-limited services
- Enhanced metrics collection for error tracking
- Automated configuration updates based on error patterns

## Files Modified

1. `crypto_bot/utils/market_loader.py` - Enhanced retry mechanisms
2. `crypto_bot/sentiment_filter.py` - Twitter API fixes
3. `crypto_bot/indicators/cycle_bias.py` - Alternative.me API fallbacks
4. `crypto_bot/utils/enhanced_ohlcv_fetcher.py` - Data validation improvements
5. `config/api_error_handling.yaml` - New configuration file (created)
6. `tools/monitor_backlog_manager.py` - New monitoring tool (created)

## Monitoring

Use the new monitoring tool to track improvements:

```bash
# Regular monitoring
python3 tools/monitor_backlog_manager.py --report

# Weekly cleanup
python3 tools/monitor_backlog_manager.py --clear-old 7
```

The fixes should significantly reduce the error volume in `monitor_backlog.jsonl` and improve the overall stability of the trading bot's API interactions.
