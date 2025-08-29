# 🌙 LunarCrush Sentiment Integration

This document describes the LunarCrush sentiment analysis integration that enhances your trading bot with real-time social intelligence without hindering trades.

## 🎯 Overview

The LunarCrush integration provides:

1. **Sentiment-Enhanced Trading**: Boosts position sizes for trades aligned with positive sentiment
2. **Solana Token Scouting**: Discover trending tokens with strong social sentiment
3. **Fail-Safe Design**: Never blocks trades due to sentiment issues

## 🔧 Configuration

### Environment Variables

```bash
export LUNARCRUSH_API_KEY="hpn7960ebtf31fplz8j0eurxqmdn418mequk61bq"
```

### Configuration Files

The integration uses `crypto_bot/config/lunarcrush_config.yaml` for detailed settings:

```yaml
lunarcrush:
  sentiment_boost:
    enabled: true
    min_galaxy_score: 60.0        # Minimum Galaxy Score for boost
    min_sentiment: 0.6            # Minimum sentiment score for boost
    max_boost: 0.5                # Maximum 50% position size increase
```

## 🚀 Features

### 1. Sentiment-Enhanced Trading

**How it works:**
- Analyzes Galaxy Score, AltRank, and sentiment direction
- Applies position size boost (1.0x to 1.5x) for aligned bullish sentiment
- **Never reduces position sizes** - only enhances or remains neutral

**Example:**
```python
# In execute_signals (main.py)
sentiment_boost = await get_lunarcrush_sentiment_boost("BTC/USD", "long")
final_size = base_size * sentiment_boost  # 1.0 to 1.5x multiplier
```

### 2. Solana Token Scouting

**Discovery Features:**
```python
from crypto_bot.solana import get_sentiment_enhanced_tokens

# Get tokens with strong bullish sentiment
tokens = await get_sentiment_enhanced_tokens(
    cfg={}, 
    min_galaxy_score=70.0, 
    min_sentiment=0.65
)

for mint_address, sentiment_data in tokens:
    print(f"Token: {sentiment_data['symbol']}")
    print(f"Galaxy Score: {sentiment_data['galaxy_score']}")
    print(f"Recommendation: {sentiment_data['recommendation']}")
```

**Scoring System:**
```python
from crypto_bot.solana import score_token_by_sentiment

score_data = await score_token_by_sentiment("SOL")
# Returns composite score (0-100) and recommendation
```

### 3. Pool Event Enhancement

```python
from crypto_bot.solana.score import score_event_with_sentiment

# Enhanced scoring for new pool events
enhanced_score = await score_event_with_sentiment(
    event, 
    config, 
    symbol="TOKEN_SYMBOL"
)
```

## 📊 Sentiment Metrics

### Galaxy Score (0-100)
- LunarCrush's proprietary social + market performance score
- Higher scores indicate stronger overall performance

### AltRank (1-infinity)
- Ranking relative to other cryptocurrencies
- Lower numbers = better performance

### Sentiment (0-1)
- Proportion of bullish vs bearish mentions
- 0.5 = neutral, >0.6 = bullish, <0.4 = bearish

### Recommendation Levels
- **STRONG_BUY**: Composite score ≥80 + bullish sentiment
- **BUY**: Composite score ≥60 + bullish sentiment  
- **HOLD**: Composite score ≥40 + not bearish
- **AVOID**: Bearish sentiment
- **NEUTRAL**: Default/insufficient data

## 🛡️ Safety Features

### 1. Fail-Safe Design
- API failures return neutral values (1.0x boost)
- Network timeouts don't block trades
- Invalid responses fall back to defaults

### 2. Trade Protection
```python
# Sentiment boost is ALWAYS ≥ 1.0
boost = max(1.0, calculated_boost)  # Never hinders trades
```

### 3. Graceful Degradation
- Works without internet connection (cached data)
- Functions with invalid API keys (neutral sentiment)
- Handles rate limits with caching

## 🔄 Integration Points

### Main Trading Loop
```python
# In crypto_bot/main.py execute_signals()
sentiment_boost = await get_lunarcrush_sentiment_boost(symbol, direction)
enhanced_size = base_position_size * sentiment_boost
```

### Strategy Integration
```python
# In strategy modules (optional)
from crypto_bot.sentiment_filter import check_sentiment_alignment

if await check_sentiment_alignment(symbol, "long", require_alignment=False):
    # Proceed with trade logic
    pass
```

### Solana Sniping
```python
# Enhanced pool scoring
from crypto_bot.solana.score import score_event_with_sentiment

score = await score_event_with_sentiment(pool_event, config, symbol)
```

## 📈 Usage Examples

### Basic Sentiment Boost
```python
# Get sentiment boost for a long position on BTC
boost = await get_lunarcrush_sentiment_boost("BTC", "long")
print(f"Position size multiplier: {boost:.2f}")
```

### Solana Token Discovery
```python
# Find trending Solana tokens with positive sentiment
enhanced_tokens = await get_sentiment_enhanced_tokens(
    cfg={},
    min_galaxy_score=75.0,
    min_sentiment=0.7,
    limit=10
)

for mint, data in enhanced_tokens:
    if data['recommendation'] in ['STRONG_BUY', 'BUY']:
        print(f"Consider: {data['symbol']} (Score: {data['galaxy_score']})")
```

### Check Token Sentiment
```python
# Evaluate individual token sentiment
sentiment_data = await score_token_by_sentiment("ETH")
if sentiment_data:
    print(f"ETH Composite Score: {sentiment_data['composite_score']:.1f}")
    print(f"Recommendation: {sentiment_data['recommendation']}")
```

## 🧪 Testing

Run the integration test to verify everything works:

```bash
python3 test_lunarcrush_integration.py
```

This tests:
- ✅ API connectivity
- ✅ Sentiment boost functionality  
- ✅ Alignment checking
- ✅ Solana scouting features
- ✅ Fail-safe behavior

## 📝 Configuration Options

### Strategy-Specific Settings
```yaml
strategy_overrides:
  meme_wave_bot:
    sentiment_boost:
      min_galaxy_score: 80.0  # Higher bar for meme tokens
      min_sentiment: 0.7
  
  sniper_bot:
    sentiment_boost:
      min_galaxy_score: 70.0
      min_sentiment: 0.65
```

### Cache Configuration
```yaml
cache:
  ttl_seconds: 300      # Cache for 5 minutes
  max_entries: 1000     # Maximum cached symbols
```

### Failsafe Settings
```yaml
failsafe:
  timeout_seconds: 10           # API timeout
  max_retries: 2               # Retry attempts
  fallback_to_neutral: true    # Use neutral on failure
  never_hinder_trades: true    # Always allow trades
```

## 🎉 Benefits

1. **Enhanced Returns**: Position size boosts for high-conviction trades
2. **Early Discovery**: Find trending tokens before they pump
3. **Risk Management**: Avoid tokens with very negative sentiment
4. **Reliability**: Fail-safe design ensures trading continues uninterrupted
5. **Configurability**: Tune sentiment thresholds per strategy

## 🔮 Future Enhancements

- Real-time sentiment alerts via Telegram
- Historical sentiment backtesting
- Custom sentiment models
- Integration with other social platforms
- Multi-timeframe sentiment analysis

---

The LunarCrush integration enhances your trading with social intelligence while maintaining the reliability and safety of your existing trading system.
