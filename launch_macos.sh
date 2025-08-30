#!/bin/bash

# LegacyCoinTrader macOS Launcher
# Optimized for macOS with automatic browser opening

echo "🚀 Starting LegacyCoinTrader on macOS..."

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "❌ Virtual environment not found. Please run './startup.sh setup' first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
ENV_FOUND=false
for env_path in ".env" "crypto_bot/.env"; do
    if [[ -f "$env_path" ]]; then
        echo "✅ Found .env file at: $env_path"
        # Check if it contains real API keys (not template values)
        if grep -q "your_kraken_api_key_here\|your_telegram_token_here\|your_helius_key_here" "$env_path"; then
            echo "❌ .env file contains template values. Please edit with real API keys."
            exit 1
        else
            echo "✅ .env file contains real API keys"
            ENV_FOUND=true
            break
        fi
    fi
done

if [[ "$ENV_FOUND" == "false" ]]; then
    echo "❌ No valid .env file found. Please run './startup.sh setup' first."
    exit 1
fi

echo "✅ Environment ready. Starting services..."

# Start main application
echo "📊 Starting main trading bot..."
python -m crypto_bot.main &
MAIN_PID=$!

# Start web frontend
echo "🌐 Starting web dashboard..."
python -m frontend.app &
FRONTEND_PID=$!

# Start Telegram bot
echo "📱 Starting Telegram bot..."
python telegram_ctl.py &
TELEGRAM_PID=$!

echo ""
echo "🎉 LegacyCoinTrader is now running!"
echo "📊 Main bot PID: $MAIN_PID"
echo "🌐 Web dashboard: http://localhost:8000"
echo "📱 Telegram bot PID: $TELEGRAM_PID"
echo ""

# Wait for frontend to start, then open browser
echo "⏳ Waiting 3 seconds for server to start..."
sleep 3

echo "🌐 Opening browser..."
open "http://localhost:8000"

echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping LegacyCoinTrader..."
    kill $MAIN_PID $FRONTEND_PID $TELEGRAM_PID 2>/dev/null || true
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait
