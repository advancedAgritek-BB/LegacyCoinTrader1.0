#!/bin/bash
echo "🔄 Restarting bot with final fixes..."

# Stop any running processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 2

# Start the bot
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_restart_final.log 2>&1 &

echo "✅ Bot restarted with PID: $!"
echo "📊 Monitor logs: tail -f crypto_bot/logs/bot_restart_final.log"
