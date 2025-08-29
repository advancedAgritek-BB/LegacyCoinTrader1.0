#!/bin/bash

# LegacyCoinTrader Quick Launcher
# Simple script to start the application

echo "🚀 Starting LegacyCoinTrader..."

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "❌ Virtual environment not found. Please run './startup.sh setup' first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found. Please run './startup.sh setup' first."
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
echo "🌐 Web dashboard: http://localhost:5000"
echo "📱 Telegram bot PID: $TELEGRAM_PID"
echo ""
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
