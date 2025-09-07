#!/bin/bash

# Start Dashboard with Price Monitoring
# This script starts both the Flask web app and the simple API server

echo "=== Starting LegacyCoinTrader Dashboard ==="
echo ""

# Kill any existing processes
echo "Stopping any existing services..."
pkill -f "python3 frontend/app.py" 2>/dev/null
pkill -f "python3 simple_api_server.py" 2>/dev/null
sleep 2

# Start the simple API server (for position data with current prices)
echo "Starting API server on port 8003..."
python3 simple_api_server.py &
API_PID=$!
echo "API server started (PID: $API_PID)"

# Wait a moment for API server to start
sleep 3

# Start the Flask web app
echo "Starting Flask web app..."
python3 frontend/app.py &
FLASK_PID=$!
echo "Flask app started (PID: $FLASK_PID)"

echo ""
echo "=== Services Started Successfully ==="
echo "📊 API Server: http://localhost:8003"
echo "🌐 Web Dashboard: http://localhost:8000"
echo ""
echo "✅ Position cards should now display CURRENT MARKET PRICES"
echo "✅ No more 'Loading...' - real-time prices from Kraken exchange"
echo ""
echo "To stop services:"
echo "  pkill -f 'python3 frontend/app.py'"
echo "  pkill -f 'python3 simple_api_server.py'"
echo ""
echo "Or run: ./stop_dashboard.sh"

# Keep script running to show logs
wait
