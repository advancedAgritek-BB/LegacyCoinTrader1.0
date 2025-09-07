#!/bin/bash
echo "🔍 Restarting with enhanced scanning and token analysis..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Clear scan cache if needed
rm -f crypto_bot/logs/scan_cache.json || true

# Start with enhanced scanning
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_enhanced_scanning.log 2>&1 &

echo "✅ Bot restarted with PID: $!"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_enhanced_scanning.log"
echo "🌐 Dashboard: http://localhost:8000"

# Wait and check status
sleep 15
echo "🔍 Checking bot status..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Bot is running successfully"
    echo "🔍 Scanning and token analysis should be active"
    echo "📈 Check logs for scan results and token discoveries"
else
    echo "❌ Bot failed to start"
    echo "📋 Check logs for errors"
fi
