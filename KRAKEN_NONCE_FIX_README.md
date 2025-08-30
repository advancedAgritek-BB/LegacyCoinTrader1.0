# 🔧 Kraken API Nonce Error Fix

## 🚨 Problem Description

The bot was experiencing "Exchange API setup failed: kraken {"error":["EAPI:Invalid nonce"]}" errors during startup. This is a common issue with Kraken's API that occurs when:

1. **System time synchronization issues** - Local time differs significantly from Kraken's server time
2. **Nonce conflicts** - Multiple API calls using the same or invalid nonce values
3. **Race conditions** - Multiple bot instances or rapid API calls
4. **Network latency** - Delays causing nonce expiration

## ✅ Solutions Implemented

### 1. Enhanced Nonce Management (`crypto_bot/execution/cex_executor.py`)

- **Custom nonce function**: Thread-safe nonce generation ensuring uniqueness
- **Time synchronization**: Automatic detection and compensation for server time differences
- **Retry mechanism**: Automatic retry for nonce errors with exponential backoff
- **Rate limit handling**: Intelligent retry for rate limit errors

### 2. Configuration Options (`crypto_bot/config.yaml`)

```yaml
kraken_settings:
  enable_nonce_improvements: true  # Enable custom nonce management
  api_retry_attempts: 3           # Number of retry attempts for API calls
  time_sync_threshold: 1000       # Time difference threshold in ms for server sync
```

### 3. Improved Error Handling (`crypto_bot/main.py`)

- **Specific error detection**: Identifies nonce errors and provides actionable guidance
- **Diagnostic information**: Logs specific error types and suggests solutions
- **User-friendly messages**: Clear explanations of what went wrong

## 🔧 Technical Details

### Nonce Generation Algorithm

```python
def custom_nonce():
    with nonce_lock:
        current_time = int(time.time() * 1000)
        if current_time <= last_nonce:
            current_time = last_nonce + 1
        last_nonce = current_time
        return current_time
```

### Time Synchronization

- Automatically detects time differences > 1 second
- Sets `timeDifference` option in CCXT for proper nonce calculation
- Logs warnings when significant time drift is detected

### Retry Logic

- **Nonce errors**: Retry with new nonce after 100ms delay
- **Rate limits**: Retry after 1 second delay
- **Authentication errors**: Immediate failure with clear error message
- **Configurable attempts**: Default 3 retries, configurable via config

## 🚀 Usage

### Enable Nonce Improvements

The improvements are enabled by default. To disable:

```yaml
kraken_settings:
  enable_nonce_improvements: false
```

### Customize Retry Behavior

```yaml
kraken_settings:
  api_retry_attempts: 5        # Increase retry attempts
  time_sync_threshold: 500     # Lower time sync threshold
```

### Monitor Logs

Watch for these log messages:

```
INFO - Time difference with Kraken server: 733ms
WARNING - Nonce error on attempt 1, retrying...
INFO - Generated nonce: 1756530517822 (last: 1756530517821)
```

## 🧪 Testing

The fixes have been tested with:

- ✅ Nonce uniqueness verification
- ✅ Time synchronization logic
- ✅ Configuration loading
- ✅ Import and module structure

## 🔍 Troubleshooting

### If Nonce Errors Persist

1. **Check system time**: Ensure your system clock is synchronized
2. **Verify API keys**: Confirm Kraken API key permissions
3. **Check for duplicates**: Ensure only one bot instance is running
4. **Network issues**: Verify stable internet connection

### Common Error Messages

- `EAPI:Invalid nonce` → Nonce management issue (usually fixed by these improvements)
- `EAPI:Rate limit exceeded` → Rate limiting (handled by retry logic)
- `EAPI:Invalid key` → API key configuration issue

### Debug Mode

Enable debug logging to see detailed nonce generation:

```yaml
logging:
  level: DEBUG
```

## 📋 Files Modified

1. **`crypto_bot/execution/cex_executor.py`** - Core nonce improvements
2. **`crypto_bot/config.yaml`** - Configuration options
3. **`crypto_bot/main.py`** - Enhanced error handling

## 🔄 Future Improvements

- **Persistent nonce storage**: Save last nonce to disk for restarts
- **Advanced time sync**: Periodic time synchronization checks
- **Metrics collection**: Track nonce error frequency and resolution
- **Circuit breaker**: Automatic fallback to paper trading on persistent failures

## 📞 Support

If you continue to experience nonce errors after implementing these fixes:

1. Check the logs for specific error messages
2. Verify your system time is accurate
3. Ensure no duplicate bot instances are running
4. Contact support with detailed error logs

---

**Note**: These improvements are backward compatible and can be safely enabled/disabled via configuration.
