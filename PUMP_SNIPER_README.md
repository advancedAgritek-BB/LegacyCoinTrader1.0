# 🎯 Advanced Memecoin Pump Sniper System

A sophisticated, multi-signal analysis system for detecting and automatically trading high-probability memecoin pumps on Solana. This system combines advanced pattern recognition, social sentiment analysis, liquidity pool analysis, and risk management to identify and capitalize on pump opportunities.

## 🚀 Key Features

### 🧠 Multi-Signal Analysis Engine
- **Pump Detection**: Advanced algorithms analyze liquidity patterns, transaction velocity, and market microstructure
- **Pool Quality Analysis**: Real-time assessment of liquidity depth, trading activity, and pool health
- **Social Sentiment**: Integration with Twitter, Telegram, Discord, and Reddit for sentiment analysis
- **Momentum Detection**: Volume spike detection and momentum pattern recognition
- **Risk Assessment**: Comprehensive rug pull detection and risk scoring

### ⚡ Ultra-Fast Execution
- **Rapid Executor**: Multi-DEX routing with MEV protection and intelligent slippage management
- **Priority Fee Optimization**: Dynamic priority fee calculation based on urgency and network conditions
- **Order Splitting**: Large orders split across multiple smaller trades to minimize market impact
- **Execution Workers**: Parallel execution workers for maximum speed

### 🛡️ Advanced Risk Management
- **Dynamic Position Sizing**: Risk-adjusted position sizing based on confidence and volatility
- **Real-time Monitoring**: Continuous position monitoring with adaptive stop losses and take profits
- **Portfolio Risk Control**: Exposure limits, correlation analysis, and emergency liquidation
- **Multiple Risk Profiles**: Conservative, moderate, and aggressive risk profiles

### 📊 Comprehensive Analytics
- **Performance Tracking**: Detailed statistics on success rates, P&L, and execution times
- **Real-time Monitoring**: Live dashboards and alerts for all system components
- **Decision Logging**: Complete audit trail of all sniping decisions and reasoning

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pump Sniper Orchestrator                     │
├─────────────────────────────────────────────────────────────────┤
│  📊 Decision Engine  │  🎯 Execution Control  │  🛡️ Risk Mgmt   │
└─────────────────────────────────────────────────────────────────┘
           │                     │                     │
┌──────────▼──────────┐ ┌────────▼────────┐ ┌─────────▼─────────┐
│   Pool Watcher      │ │  Rapid Executor │ │  Risk Manager     │
│  - New Pool Events  │ │  - Multi-DEX    │ │  - Position Size  │
│  - Helius WebSocket │ │  - MEV Protect  │ │  - Stop/Take      │
│  - Real-time Feed   │ │  - Fast Execute │ │  - Emergency      │
└─────────────────────┘ └─────────────────┘ └───────────────────┘

┌─────────────────────┐ ┌─────────────────┐ ┌───────────────────┐
│   Pump Detector     │ │ Pool Analyzer   │ │ Sentiment Analyzer│
│  - Pattern Recog    │ │ - Liquidity     │ │ - Twitter/Social  │
│  - Signal Fusion    │ │ - Health Score  │ │ - Influencer      │
│  - Confidence       │ │ - Viability     │ │ - Buzz Detection  │
└─────────────────────┘ └─────────────────┘ └───────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Momentum Detector                            │
│         Volume Spikes  │  Technical Analysis  │  Velocity       │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Installation & Setup

### 1. Prerequisites
- Python 3.8+
- Solana CLI tools
- Helius API key (for pool monitoring)
- Twitter API access (optional, for sentiment)
- LunarCrush API key (optional, for enhanced sentiment)

### 2. Environment Setup
Create a `.env` file with your API keys:
```bash
HELIUS_KEY=your_helius_api_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token  # Optional
LUNARCRUSH_API_KEY=your_lunarcrush_api_key      # Optional
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
WALLET_PRIVATE_KEY=your_wallet_private_key
```

### 3. Configuration
Edit `config/pump_sniper_config.yaml` to customize:

```yaml
pump_sniper_orchestrator:
  enabled: true
  min_decision_confidence: 0.7    # Adjust confidence threshold
  
rapid_executor:
  default_slippage_pct: 0.03      # 3% slippage tolerance
  base_position_size_sol: 0.1     # Base position size
  
sniper_risk_manager:
  default_profile: "moderate"     # Risk profile
  
social_sentiment:
  enabled: true                   # Enable sentiment analysis
```

### 4. Integration with Main Bot
The pump sniper integrates seamlessly with your existing bot. Add to your main config:

```yaml
pump_sniper_orchestrator:
  enabled: true
  
# The system will automatically start when the main bot starts
```

## 📈 Usage

### Automatic Operation
Once configured, the system runs automatically:

1. **Pool Discovery**: Monitors Helius WebSocket for new liquidity pools
2. **Analysis**: Performs comprehensive multi-signal analysis
3. **Decision Making**: Uses weighted scoring to make snipe/monitor/ignore decisions
4. **Execution**: Rapidly executes high-confidence opportunities
5. **Monitoring**: Continuously monitors positions for optimal exits

### Manual Controls
```python
from crypto_bot.solana.pump_sniper_integration import (
    manual_evaluate_token,
    get_pump_sniper_status,
    emergency_stop_pump_sniper
)

# Manually evaluate a token
result = await manual_evaluate_token(
    "token_mint_address", 
    "TOKEN_SYMBOL"
)

# Get system status
status = get_pump_sniper_status()

# Emergency stop (if needed)
emergency_stop_pump_sniper()
```

### Telegram Integration
The system can send notifications via Telegram:

```yaml
monitoring:
  telegram_notifications: true
  large_profit_threshold: 0.50    # 50% profit alerts
  large_loss_threshold: 0.20     # 20% loss alerts
```

## 🎯 Decision Making Process

The system uses a sophisticated weighted scoring system:

### Signal Weights (Default)
- **Pump Probability** (25%): How likely is this to pump?
- **Pool Quality** (20%): Is the liquidity pool healthy?
- **Sentiment Score** (15%): What's the social buzz?
- **Momentum Score** (15%): Are there momentum indicators?
- **Risk Score** (15%): How risky is this token?
- **Timing Score** (10%): Is this the right time to enter?

### Decision Thresholds
- **Snipe**: Confidence ≥ 70% + Risk validation passed
- **Monitor**: Confidence 50-70% (watch for better entry)
- **Ignore**: Confidence < 50% or high risk

## 🛡️ Risk Management

### Risk Profiles

#### Conservative (Default for new users)
- Max 2% per position, 10% total exposure
- 15% stop loss, 50% take profit
- Requires $50K+ liquidity
- Max 3 concurrent positions

#### Moderate (Recommended for experienced users)
- Max 5% per position, 20% total exposure  
- 20% stop loss, 75% take profit
- Requires $25K+ liquidity
- Max 5 concurrent positions

#### Aggressive (For high-risk tolerance)
- Max 10% per position, 50% total exposure
- 30% stop loss, 100% take profit
- Requires $10K+ liquidity
- Max 10 concurrent positions

### Safety Features
- **Emergency Stop**: Instantly halt all operations
- **Daily Loss Limits**: Automatic pause if daily loss exceeds threshold
- **Position Monitoring**: Real-time monitoring with adaptive exits
- **Correlation Limits**: Prevents over-concentration in similar tokens
- **Rug Pull Detection**: Multiple risk indicators and blacklists

## 📊 Performance Monitoring

### Key Metrics Tracked
- **Success Rate**: Percentage of profitable trades
- **Average P&L**: Mean profit/loss per trade
- **Execution Speed**: Average time from signal to execution
- **Slippage**: Actual vs expected execution prices
- **Risk-Adjusted Returns**: Sharpe ratio and risk metrics

### Real-time Dashboards
Access system performance through:
- Telegram notifications
- Log file analysis (`crypto_bot/logs/pump_sniper.log`)
- Statistics API endpoints

## 🔍 Signal Analysis Details

### Pump Detection Signals
- **Liquidity Analysis**: Pool depth, stability, concentration
- **Transaction Velocity**: Frequency and size of transactions
- **Price Momentum**: Multi-timeframe price analysis
- **Volume Patterns**: Spike detection and trend analysis
- **Whale Activity**: Large trader behavior analysis

### Social Sentiment Indicators
- **Mention Volume**: Sudden increases in social mentions
- **Sentiment Score**: Positive/negative sentiment analysis
- **Influencer Activity**: Verified accounts and large follower mentions
- **Viral Potential**: Engagement rates and sharing patterns
- **Authenticity**: Organic vs artificial activity detection

### Technical Indicators
- **RSI Momentum**: Relative strength analysis
- **MACD Signals**: Moving average convergence/divergence
- **Bollinger Squeeze**: Volatility compression detection
- **Volume Profile**: Volume distribution analysis
- **Pattern Recognition**: Cup & handle, accumulation patterns

## 🚨 Troubleshooting

### Common Issues

#### System Not Starting
1. Check your `HELIUS_KEY` environment variable
2. Verify configuration in `config/pump_sniper_config.yaml`
3. Ensure sufficient account balance for trading
4. Check log files for specific error messages

#### No Snipe Opportunities
1. Lower the `min_decision_confidence` threshold
2. Increase `max_slippage_tolerance` if needed
3. Check if risk limits are too restrictive
4. Verify social sentiment APIs are working

#### High False Positives
1. Increase the `min_decision_confidence` threshold
2. Enable more restrictive risk filters
3. Adjust signal weights to favor quality over speed
4. Use more conservative risk profile

### Log Analysis
Check these log files for debugging:
- `crypto_bot/logs/pump_sniper.log` - Main system log
- `crypto_bot/logs/sniper_risk_state.json` - Risk manager state
- `crypto_bot/logs/pump_analysis/` - Detailed analysis results

## 🔮 Advanced Configuration

### Custom Signal Weights
Adjust the decision-making process:

```yaml
pump_sniper_orchestrator:
  decision_weights:
    pump_probability: 0.30      # Increase pump detection weight
    pool_quality: 0.25          # Increase pool quality weight
    sentiment_score: 0.10       # Decrease sentiment weight
    momentum_score: 0.20        # Momentum importance
    risk_score: 0.10            # Risk assessment weight
    timing_score: 0.05          # Timing factor weight
```

### DEX Preferences
Configure execution routing:

```yaml
rapid_executor:
  dex_preferences:
    - "raydium"    # Highest preference
    - "jupiter"    # Backup option
    - "orca"       # Alternative
    - "serum"      # Last resort
```

### Sentiment Sources
Enable/disable sentiment analysis sources:

```yaml
social_sentiment:
  platforms:
    - "twitter"     # Enable Twitter monitoring
    - "telegram"    # Enable Telegram (requires setup)
    - "discord"     # Enable Discord (requires setup)
    - "reddit"      # Enable Reddit (requires setup)
```

## 📚 API Reference

### Integration Functions
```python
# Start/stop system
await start_pump_sniper_system(config)
await stop_pump_sniper_system()

# Get status and statistics
status = get_pump_sniper_status()
stats = orchestrator.get_statistics()

# Manual evaluation
decision = await manual_evaluate_token(token_mint, symbol)

# Emergency controls
emergency_stop_pump_sniper()
resume_pump_sniper()
```

### Configuration Hot-Reloading
The system automatically reloads configuration changes without restart.

## ⚠️ Risk Warnings

1. **High-Risk Trading**: Memecoin trading is extremely risky with potential for total loss
2. **Rapid Execution**: System can execute trades very quickly; monitor position sizes
3. **API Dependencies**: Relies on external APIs that may have downtime
4. **Market Volatility**: Memecoin markets are highly volatile and unpredictable
5. **Rug Pull Risk**: Despite detection systems, some rug pulls may not be caught

## 🤝 Support & Contributing

### Getting Help
1. Check the troubleshooting section above
2. Review log files for error messages
3. Verify configuration and API keys
4. Test with paper trading first

### Contributing
The pump sniper system is modular and extensible:
- Add new signal sources in the respective analyzer modules
- Implement additional DEX integrations in the rapid executor
- Enhance risk management with new risk factors
- Contribute to pattern recognition algorithms

## 📜 License & Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies carries significant financial risk. Always:
- Start with paper trading
- Use small position sizes
- Never invest more than you can afford to lose
- Understand the risks involved
- Comply with local regulations

The authors are not responsible for any financial losses incurred through use of this software.

---

**Happy Sniping! 🎯💰**

*Remember: The best trade is often the one you don't take. This system helps you find opportunities, but always apply your own judgment and risk management.*
