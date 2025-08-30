# 📄 Clear Paper Trading Cache

This document explains how to clear the trade cache in paper trading mode to start fresh.

## 🚀 Quick Start

### Option 1: Command Line Script (Basic)
```bash
# From the project root directory
python3 tools/clear_paper_cache.py
```

### Option 2: Comprehensive Cache Clear (Recommended)
```bash
# Clear ALL cache data, logs, and files
python3 tools/clear_all_cache.py
```

### Option 3: Telegram Bot Command
If you have the Telegram bot enabled, use the `/clear_cache` command.

### Option 4: Telegram Bot UI
1. Open the Telegram bot menu
2. Click "Clear Cache" button
3. Confirm the action

## 🔧 What Gets Cleared

### Basic Cache Clear (Option 1)
When you clear the basic cache, the following items are reset:
- **Paper Wallet Positions**: All open positions are closed
- **Paper Wallet Balance**: Reset to initial balance
- **Trade History**: All trade statistics are reset
- **Bot Context Positions**: Open position tracking is cleared
- **Data Cache**: OHLCV and regime analysis cache is cleared
- **Regime Cache**: Market regime classification cache is cleared

### Comprehensive Cache Clear (Option 2)
The comprehensive clear additionally removes:
- **All Log Files**: positions.log, trades.csv, bot.log
- **Asset Scores**: asset_scores.json, strategy_stats.json
- **Scan Cache**: All scanner and strategy logs
- **Market Data**: liquid_pairs.json and other cache files
- **Session State**: last_regime.json, startup_output.log
- **Strategy Logs**: All enhanced scanner and integration logs

## 📋 Prerequisites

- **Paper Trading Mode**: Must be running in `dry_run` mode
- **Bot Stopped**: It's recommended to stop the bot before clearing cache
- **Admin Access**: For Telegram commands, you need admin privileges

## 🛠️ Manual Cache Clearing

If you prefer to manually clear specific components:

### Clear Paper Wallet Only
```python
from crypto_bot.paper_wallet import PaperWallet

# Get your paper wallet instance
paper_wallet.reset()  # Resets to initial balance
```

### Clear Bot Context Only
```python
# Clear positions
ctx.positions.clear()

# Clear data cache
ctx.df_cache.clear()

# Clear regime cache
ctx.regime_cache.clear()
```

### Clear Frontend Data Sources
```bash
# Clear position logs
echo "" > crypto_bot/logs/positions.log
echo "" > crypto_bot/logs/trades.csv
echo "{}" > crypto_bot/logs/asset_scores.json
echo "{}" > crypto_bot/logs/strategy_stats.json
```

## ⚠️ Important Notes

1. **Irreversible Action**: Clearing the cache cannot be undone
2. **Stop Bot First**: Always stop the bot before clearing cache to avoid conflicts
3. **Restart Required**: After clearing cache, restart the bot to start fresh
4. **Balance Reset**: Paper wallet balance returns to initial amount
5. **Position Loss**: All open positions will be lost
6. **Frontend Data**: May need to refresh browser to see changes

## 🔄 After Clearing Cache

1. **Restart the Bot**: Start the bot again to initialize fresh state
2. **Set New Balance**: Enter your desired paper trading balance
3. **Monitor**: Watch for new trades and ensure clean state
4. **Verify**: Check that old positions are no longer showing
5. **Refresh Frontend**: Clear browser cache or refresh the page

## 🐛 Troubleshooting

### Cache Not Clearing
- Ensure bot is stopped
- Check if you're in paper trading mode
- Verify admin permissions for Telegram commands

### Partial Clear
- Some components may not clear if bot is running
- Restart bot after clearing for complete reset

### Frontend Still Shows Old Data
- Clear browser cache and refresh page
- Check if there are additional data sources
- Use comprehensive cache clear script
- Verify all log files are empty

### Balance Issues
- If balance seems incorrect, manually reset paper wallet
- Check configuration files for balance settings

## 📞 Support

If you encounter issues:
1. Check the bot logs for error messages
2. Ensure all dependencies are installed
3. Verify configuration settings
4. Restart the bot completely
5. Use the comprehensive cache clear script

## 🔗 Related Files

- `crypto_bot/utils/telegram.py` - Cache clearing utility function
- `crypto_bot/paper_wallet.py` - Paper wallet implementation
- `tools/clear_paper_cache.py` - Basic command line script
- `tools/clear_all_cache.py` - Comprehensive cache clearing script
- `crypto_bot/telegram_bot_ui.py` - Telegram bot integration
