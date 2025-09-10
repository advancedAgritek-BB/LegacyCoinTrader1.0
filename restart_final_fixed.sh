#!/bin/bash
echo "🔄 Final restart with RiskConfig fix..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Start with fixed config
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_final_fixed.log 2>&1 &

echo "✅ Bot restarted with RiskConfig fix (PID: $!)"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_final_fixed.log"

# Wait and check
sleep 20
echo "🔍 Checking bot status..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Bot is running successfully"
    echo "📈 Check logs for signal generation"
    
    # Check for successful operation
    if grep -q "actionable signals" crypto_bot/logs/bot_final_fixed.log; then
        echo "🎉 SUCCESS: Found actionable signals!"
    elif grep -q "PHASE: analyse_batch completed" crypto_bot/logs/bot_final_fixed.log; then
        echo "✅ SUCCESS: Analysis pipeline working!"
    elif grep -q "Trading cycle completed" crypto_bot/logs/bot_final_fixed.log; then
        echo "✅ SUCCESS: Trading cycle completed!"
    else
        echo "⏳ Bot is running, waiting for signal generation..."
    fi
else
    echo "❌ Bot failed to start"
    echo "📋 Check logs for errors"
fi
