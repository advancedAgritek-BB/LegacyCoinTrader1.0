#!/bin/bash
echo "🔄 Final restart with stable configuration..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Start with stable config
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_final_stable.log 2>&1 &

echo "✅ Bot restarted with PID: $!"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_final_stable.log"
echo "🌐 Dashboard: http://localhost:8000"
