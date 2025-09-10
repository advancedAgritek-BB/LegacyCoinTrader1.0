#!/bin/bash
# Restart script for bot with fixed stop loss configuration

echo "🔄 Restarting bot with fixed stop loss configuration..."

# Stop the current bot
if [ -f "bot_pid.txt" ]; then
    PID=$(cat bot_pid.txt)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping bot (PID: $PID)..."
        kill $PID
        sleep 5
    fi
fi

# Clear any stale PID file
rm -f bot_pid.txt

# Clear cached data that might be causing DataFrame errors
echo "Clearing cached data..."
rm -rf cache/*.json 2>/dev/null || true

# Start the bot
echo "Starting bot with fixed configuration..."
python3 -m crypto_bot.main &

# Wait a moment for bot to start
sleep 10

# Check if bot started successfully
if [ -f "bot_pid.txt" ]; then
    PID=$(cat bot_pid.txt)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Bot restarted successfully (PID: $PID)"
        echo "📊 Monitor the logs for stop loss activity:"
        echo "   tail -f crypto_bot/logs/bot.log"
    else
        echo "❌ Bot failed to start"
    fi
else
    echo "❌ Bot PID file not created"
fi
