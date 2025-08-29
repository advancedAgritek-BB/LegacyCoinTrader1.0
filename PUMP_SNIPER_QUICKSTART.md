# 🚀 Pump Sniper Quick Start Guide

Get your advanced memecoin pump sniper up and running in minutes!

## ⚡ 5-Minute Setup

### 1. Set Environment Variables
```bash
# Add to your .env file or export directly
export HELIUS_KEY="your_helius_api_key_here"
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
export WALLET_PRIVATE_KEY="your_wallet_private_key_base58"

# Optional but recommended
export TWITTER_BEARER_TOKEN="your_twitter_bearer_token"
export LUNARCRUSH_API_KEY="your_lunarcrush_api_key"
```

### 2. Enable in Main Config
Edit `crypto_bot/config.yaml` and add:
```yaml
# Add this section to enable the pump sniper
pump_sniper_orchestrator:
  enabled: true
```

### 3. Start with Paper Trading
The system starts in paper trading mode by default. Check `config/pump_sniper_config.yaml`:
```yaml
development:
  paper_trading: true  # Safe to test with
```

### 4. Run Your Bot
```bash
python -m crypto_bot.main
```

You should see:
```
INFO - Advanced Pump Sniper System started successfully
INFO - Pump detector started
INFO - Liquidity pool analyzer started
INFO - Rapid executor started
INFO - Risk management system started
INFO - Social sentiment analyzer started
INFO - Momentum detector started
```

## 🎯 First Successful Snipe

### Watch the Logs
```bash
tail -f crypto_bot/logs/pump_sniper.log
```

Look for entries like:
```
INFO - High-probability pump detected: 7xKXtg2C... Probability: 0.85, Timing: 0.78, Risk: 0.25
INFO - Decision for 7xKXtg2C...: snipe (confidence: 0.82) - High confidence score: 0.82, Pump probability: 0.85, Pool quality: 0.78
INFO - Snipe executed successfully for 7xKXtg2C...: Amount: 0.1000 SOL, Price: 0.001234
```

### Telegram Notifications (Optional)
If you have Telegram set up, you'll receive messages like:
```
🎯 PUMP SNIPE: 7xKXtg2C...
💪 Confidence: 82%
💰 Size: 0.100 SOL
🛑 Stop Loss: 20%
🎯 Take Profit: 50%
📝 Reasoning:
  • High confidence score: 0.82
  • Pump probability: 0.85
  • Pool quality: 0.78
```

## ⚙️ Essential Configuration Tuning

### For Conservative Trading
```yaml
pump_sniper_orchestrator:
  min_decision_confidence: 0.8    # Higher confidence required

sniper_risk_manager:
  default_profile: "conservative"

rapid_executor:
  base_position_size_sol: 0.05    # Smaller positions
```

### For More Aggressive Trading
```yaml
pump_sniper_orchestrator:
  min_decision_confidence: 0.6    # Lower confidence threshold

sniper_risk_manager:
  default_profile: "aggressive"

rapid_executor:
  base_position_size_sol: 0.2     # Larger positions
```

### For Maximum Opportunities
```yaml
pool_analyzer:
  min_liquidity_usd: 5000         # Lower liquidity requirement

safety_filters:
  min_unique_traders: 3           # Fewer traders required
```

## 🔍 Monitoring Your Performance

### Check Statistics
The system tracks detailed performance metrics:

```python
from crypto_bot.solana.pump_sniper_integration import get_pump_sniper_status

status = get_pump_sniper_status()
print(status["statistics"])
```

### Key Metrics to Watch
- **Success Rate**: Aim for >50% (memecoin trading is inherently risky)
- **Average Execution Time**: Should be <5 seconds
- **Risk-Adjusted P&L**: Total P&L considering risk taken
- **Slippage**: Should be close to your tolerance settings

## 🛠️ Common Tweaks

### Getting More Snipe Opportunities
If the system isn't finding opportunities:

1. **Lower confidence threshold**:
   ```yaml
   min_decision_confidence: 0.6  # Down from 0.7
   ```

2. **Increase slippage tolerance**:
   ```yaml
   default_slippage_pct: 0.05  # Up from 0.03
   ```

3. **Lower liquidity requirements**:
   ```yaml
   min_liquidity_usd: 5000  # Down from 10000
   ```

### Reducing False Positives
If getting too many bad trades:

1. **Increase confidence threshold**:
   ```yaml
   min_decision_confidence: 0.8  # Up from 0.7
   ```

2. **Add more risk filters**:
   ```yaml
   safety_filters:
     min_unique_traders: 10      # More traders required
     min_trading_volume: 5000    # Higher volume required
   ```

3. **Adjust signal weights** (favor quality over speed):
   ```yaml
   decision_weights:
     pump_probability: 0.20
     pool_quality: 0.30  # Increase pool quality weight
     risk_score: 0.20    # Increase risk assessment weight
   ```

## 🚨 Safety Checklist

### Before Going Live
- [ ] Test with paper trading for at least 24 hours
- [ ] Verify your API keys are working
- [ ] Check that you have sufficient SOL balance
- [ ] Set appropriate position sizes for your risk tolerance
- [ ] Enable emergency stop notifications
- [ ] Understand the risk management settings

### Daily Monitoring
- [ ] Check daily P&L and position sizes
- [ ] Review any emergency stops or alerts
- [ ] Monitor system performance metrics
- [ ] Verify API connections are stable

## 🎛️ Advanced Features

### Manual Token Evaluation
Test the system on specific tokens:
```python
from crypto_bot.solana.pump_sniper_integration import manual_evaluate_token

result = await manual_evaluate_token(
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC for testing
    "USDC"
)
print(f"Decision: {result['decision']}, Confidence: {result['confidence']}")
```

### Emergency Controls
```python
from crypto_bot.solana.pump_sniper_integration import (
    emergency_stop_pump_sniper,
    resume_pump_sniper
)

# Stop all operations immediately
emergency_stop_pump_sniper()

# Resume after emergency
resume_pump_sniper()
```

### Real-time Position Monitoring
```python
from crypto_bot.solana.pump_sniper_integration import get_pump_sniper_integration

integration = get_pump_sniper_integration(config)
positions = integration.get_active_positions()
print(f"Active positions: {len(positions)}")
```

## 🔧 Troubleshooting

### System Won't Start
1. **Check your HELIUS_KEY**: Must be valid and have sufficient credits
2. **Verify configuration**: Ensure `pump_sniper_orchestrator.enabled: true`
3. **Check logs**: Look in `crypto_bot/logs/pump_sniper.log` for errors
4. **Test API connectivity**: Try manual token evaluation first

### No Opportunities Found
1. **Market conditions**: Memecoin activity varies greatly
2. **Confidence threshold**: Try lowering to 0.6 temporarily
3. **Liquidity requirements**: Lower minimum liquidity requirements
4. **Time of day**: More activity during US trading hours

### High Slippage/Poor Execution
1. **Increase priority fees**: Higher fees for faster execution
2. **Adjust slippage tolerance**: Increase if markets are very volatile
3. **Check DEX preferences**: Ensure preferred DEXs are available
4. **Position size**: Smaller positions = less slippage

## 📊 Expected Performance

### Realistic Expectations
- **Hit Rate**: 40-60% profitable trades (memecoin trading is hard!)
- **Average Trade**: 2-8% profit when successful
- **Execution Speed**: 3-10 seconds from signal to trade
- **Daily Opportunities**: 5-20 potential snipes depending on market activity

### Performance Optimization Tips
1. **Start conservative**: Better to miss opportunities than lose money
2. **Gradually increase risk**: As you gain confidence in the system
3. **Monitor correlation**: Don't hold too many similar memecoins
4. **Take profits**: The system has take-profit targets for a reason
5. **Review decisions**: Learn from both wins and losses

## 🎯 Success Tips

1. **Paper trade first**: Get comfortable with the system behavior
2. **Start small**: Use minimal position sizes initially
3. **Monitor actively**: Watch the first few days of operation closely
4. **Adjust gradually**: Make small configuration changes based on results
5. **Understand the risks**: Memecoin trading can result in significant losses

## 📞 Getting Help

If you encounter issues:

1. **Check logs**: `crypto_bot/logs/pump_sniper.log` contains detailed information
2. **Verify setup**: Ensure all API keys and configuration are correct
3. **Test components**: Use manual evaluation to test individual components
4. **Start simple**: Disable advanced features initially, add them gradually

---

**Happy Sniping! 🎯**

*Remember: Start with paper trading, use small positions, and never risk more than you can afford to lose!*
