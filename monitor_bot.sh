#!/bin/bash
# Bot monitoring script
BOT_PID_FILE="bot_pid.txt"
LOG_FILE="crypto_bot/logs/bot_monitor.log"

while true; do
    if ! pgrep -f "crypto_bot" > /dev/null; then
        echo "$(date): Bot not running, restarting..." >> $LOG_FILE
        python3 start_bot_auto.py &
        echo $! > $BOT_PID_FILE
        sleep 10
    else
        echo "$(date): Bot running normally" >> $LOG_FILE
    fi
    sleep 30
done
