#!/bin/bash
echo "🔄 Comprehensive restart with stable configuration..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Clear any cached data
rm -f crypto_bot/logs/*.log.bak || true

# Start with comprehensive config
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_comprehensive.log 2>&1 &

echo "✅ Bot restarted with PID: $!"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_comprehensive.log"
echo "🌐 Dashboard: http://localhost:8000"

# Wait and check status
sleep 10
echo "🔍 Checking bot status..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Bot is running successfully"
    echo "📈 Check logs for signal generation"
else
    echo "❌ Bot failed to start"
    echo "📋 Check logs for errors"
fi
