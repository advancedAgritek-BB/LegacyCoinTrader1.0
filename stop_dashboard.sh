#!/bin/bash

# Stop Dashboard Services
echo "Stopping LegacyCoinTrader Dashboard services..."

# Stop Flask app
echo "Stopping Flask web app..."
pkill -f "python3 frontend/app.py"
sleep 1

# Stop API server
echo "Stopping API server..."
pkill -f "python3 simple_api_server.py"
sleep 1

echo "✅ All services stopped"
echo ""
echo "To restart: ./start_dashboard.sh"
