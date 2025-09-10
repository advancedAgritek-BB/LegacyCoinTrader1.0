#!/bin/bash
"""
Script to restart the trading bot and monitor evaluation pipeline improvements.
"""

echo "🔄 Restarting trading bot with evaluation pipeline fixes..."

# Stop any running bot processes
echo "📋 Stopping existing bot processes..."
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

# Wait a moment for processes to stop
sleep 2

# Start the bot in the background
echo "🚀 Starting trading bot with fixes..."
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_restart.log 2>&1 &
BOT_PID=$!

echo "✅ Bot started with PID: $BOT_PID"

# Monitor the logs for improvements
echo "📊 Monitoring evaluation pipeline improvements..."
echo "Press Ctrl+C to stop monitoring"

# Function to monitor logs
monitor_logs() {
    local log_file="crypto_bot/logs/bot.log"
    local last_line_count=0
    
    while true; do
        if [ -f "$log_file" ]; then
            current_lines=$(wc -l < "$log_file" 2>/dev/null || echo "0")
            
            if [ "$current_lines" -gt "$last_line_count" ]; then
                # Show new lines
                tail -n $((current_lines - last_line_count)) "$log_file" | grep -E "(evaluation|analysis|signal|error|warning)" || true
                last_line_count=$current_lines
            fi
        fi
        
        sleep 5
    done
}

# Start monitoring
monitor_logs &
MONITOR_PID=$!

# Wait for user interrupt
trap "echo '🛑 Stopping monitoring...'; kill $MONITOR_PID; kill $BOT_PID; echo '✅ Bot stopped'; exit" INT

wait
