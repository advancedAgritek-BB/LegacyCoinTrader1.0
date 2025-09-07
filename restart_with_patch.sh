#!/bin/bash
echo "🔄 Restarting with symbol patch..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Start with patch
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 start_with_patch.py > crypto_bot/logs/bot_patched.log 2>&1 &

echo "✅ Bot restarted with patch (PID: $!)"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_patched.log"

# Wait and check
sleep 15
echo "🔍 Checking bot status..."
if pgrep -f "python.*start_with_patch" > /dev/null; then
    echo "✅ Bot is running with patch"
    echo "📈 Check logs for signal generation"
else
    echo "❌ Bot failed to start with patch"
fi
