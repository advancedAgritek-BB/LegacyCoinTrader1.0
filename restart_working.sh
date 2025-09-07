#!/bin/bash
echo "🔄 Final restart with working solution..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true
pkill -f "python.*start_with_patch" || true

sleep 3

# Start with working solution
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 start_working.py > crypto_bot/logs/bot_working_final.log 2>&1 &

echo "✅ Bot restarted with working solution (PID: $!)"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_working_final.log"

# Wait and check
sleep 20
echo "🔍 Checking bot status..."
if pgrep -f "python.*start_working" > /dev/null; then
    echo "✅ Bot is running successfully"
    echo "📈 Check logs for signal generation"
    
    # Check for successful operation
    if grep -q "actionable signals" crypto_bot/logs/bot_working_final.log; then
        echo "🎉 SUCCESS: Found actionable signals!"
    elif grep -q "PHASE: analyse_batch completed" crypto_bot/logs/bot_working_final.log; then
        echo "✅ SUCCESS: Analysis pipeline working!"
    else
        echo "⏳ Bot is running, waiting for signal generation..."
    fi
else
    echo "❌ Bot failed to start"
    echo "📋 Check logs for errors"
fi
