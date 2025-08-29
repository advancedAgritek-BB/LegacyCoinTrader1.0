# Crypto Bot Diagnostic Tools

This directory contains tools to help diagnose and fix common issues with the crypto bot.

## Tools Overview

### 1. Configuration Validator (`validate_config.py`)

Validates your bot configuration for common issues and provides recommendations.

**Usage:**
```bash
cd crypto_bot/tools
python validate_config.py
```

**What it checks:**
- Telegram bot configuration
- WebSocket settings
- OHLCV quality parameters
- Risk management settings

### 2. Telegram Bot Tester (`test_telegram.py`)

Tests your Telegram bot connection and sends a test message to verify everything is working.

**Usage:**
```bash
cd crypto_bot/tools
python test_telegram.py
```

**What it tests:**
- Bot token validity
- Chat ID configuration
- Connection to Telegram servers
- Ability to send messages

**Prerequisites:**
```bash
pip install python-telegram-bot
```

### 3. WebSocket Health Checker (`websocket_health.py`)

Monitors WebSocket connections and identifies connectivity issues.

**Usage:**
```bash
cd crypto_bot/tools
python websocket_health.py
```

**What it checks:**
- WebSocket configuration values
- Kraken WebSocket connectivity
- OHLCV quality settings
- Connection stability recommendations

**Prerequisites:**
```bash
pip install websocket-client
```

## Common Issues and Fixes

### Telegram Bot Timeout Errors

**Symptoms:**
- Multiple "Failed to send message: Timed out" errors in logs
- Bot appears to be running but no messages are sent

**Fixes:**
1. Run `test_telegram.py` to verify connection
2. Check that bot is added to the chat
3. Verify chat ID is correct
4. Ensure bot has permission to send messages
5. Check internet connection

### WebSocket OHLCV Timeouts

**Symptoms:**
- "WS OHLCV timeout" errors with negative limit values
- Frequent fallbacks to REST API
- Incomplete candle data

**Fixes:**
1. Run `websocket_health.py` to check connectivity
2. Verify Kraken WebSocket is accessible
3. Check timeout and ping interval settings
4. Ensure proper fallback configuration

### Incomplete OHLCV Data

**Symptoms:**
- "Incomplete OHLCV" warnings
- Symbols being skipped due to insufficient data
- Analysis quality issues

**Fixes:**
1. Check `min_data_ratio` setting (should be 0.5-0.8)
2. Verify `min_required_candles` setting (should be 20-50)
3. Ensure REST fallback is enabled
4. Check exchange rate limits

## Configuration Recommendations

### WebSocket Settings
```yaml
use_websocket: true
ws_ohlcv_timeout: 15        # 15-20 seconds
ws_ping_interval: 5         # 5-8 seconds
ws_failures_before_disable: 3
ws_reconnect_delay: 10
ws_max_retries: 3
```

### OHLCV Quality Settings
```yaml
ohlcv_quality:
  fallback_to_rest: true
  min_data_ratio: 0.5       # 0.5-0.8
  min_required_candles: 20  # 20-50
  retry_incomplete: true
  max_retry_attempts: 2
  rest_fallback_timeout: 25
```

### Telegram Settings
```yaml
telegram:
  timeout_seconds: 30
  max_retries: 3
  retry_delay_seconds: 5
  fail_silently: true
  connection_pool_size: 8
  connect_timeout: 30.0
  read_timeout: 30.0
  write_timeout: 30.0
```

## Troubleshooting Workflow

1. **Start with configuration validation:**
   ```bash
   python validate_config.py
   ```

2. **Test Telegram bot if enabled:**
   ```bash
   python test_telegram.py
   ```

3. **Check WebSocket health:**
   ```bash
   python websocket_health.py
   ```

4. **Review the main bot logs:**
   ```bash
   tail -f ../logs/bot.log
   ```

5. **Apply fixes based on tool recommendations**

6. **Restart the bot and monitor for improvements**

## Getting Help

If you continue to experience issues after running these tools:

1. Check the tool output for specific error messages
2. Review the configuration recommendations
3. Verify your exchange API keys and permissions
4. Check your internet connection and firewall settings
5. Review the main bot documentation for additional troubleshooting steps

## Tool Dependencies

Make sure you have the required packages installed:

```bash
pip install pyyaml python-telegram-bot websocket-client
```

Or install from the main requirements file:

```bash
pip install -r ../../requirements.txt
```
