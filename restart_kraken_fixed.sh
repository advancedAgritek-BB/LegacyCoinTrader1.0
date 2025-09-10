#!/bin/bash
echo "🔄 Restarting bot with Kraken exchange fix..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Start with Kraken exchange
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_kraken_fixed.log 2>&1 &

echo "✅ Bot restarted with Kraken exchange (PID: $!)"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_kraken_fixed.log"

# Wait and check
sleep 20
echo "🔍 Checking bot status..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Bot is running successfully"
    
    # Check for Kraken usage and no Coinbase errors
    if grep -q "coinbase.*401 Unauthorized" crypto_bot/logs/bot_kraken_fixed.log; then
        echo "⚠️ Still seeing Coinbase errors - checking config..."
        echo "📋 Current exchange config:"
        grep -A 5 -B 5 "exchange" crypto_bot/config.yaml
    elif grep -q "kraken" crypto_bot/logs/bot_kraken_fixed.log; then
        echo "✅ SUCCESS: Bot is using Kraken!"
    elif grep -q "actionable signals" crypto_bot/logs/bot_kraken_fixed.log; then
        echo "🎉 SUCCESS: Found actionable signals!"
    elif grep -q "PHASE: analyse_batch completed" crypto_bot/logs/bot_kraken_fixed.log; then
        echo "✅ SUCCESS: Analysis pipeline working!"
    else
        echo "⏳ Bot is running, waiting for signal generation..."
    fi
else
    echo "❌ Bot failed to start"
    echo "📋 Check logs for errors"
fi
