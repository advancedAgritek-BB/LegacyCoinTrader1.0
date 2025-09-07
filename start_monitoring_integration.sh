#!/bin/bash
# Start Monitoring Frontend Integration
# Ensures all monitoring components properly tie into the frontend monitoring page

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "🚀 Starting Monitoring Frontend Integration"
echo "Working directory: $WORKING_DIR"

# Change to working directory
cd "$WORKING_DIR"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check if monitoring integration is already running
if pgrep -f "integrate_monitoring_frontend.py" > /dev/null; then
    echo "⚠️  Monitoring frontend integration is already running"
    echo "Stopping existing process..."
    pkill -f "integrate_monitoring_frontend.py"
    sleep 2
fi

# Start monitoring frontend integration
echo "Starting monitoring frontend integration..."
python3 integrate_monitoring_frontend.py &

# Save PID
MONITORING_INTEGRATION_PID=$!
echo $MONITORING_INTEGRATION_PID > monitoring_integration.pid
echo "✅ Monitoring frontend integration started (PID: $MONITORING_INTEGRATION_PID)"

# Wait a moment for the integration to initialize
sleep 5

# Check if the integration started successfully
if ps -p $MONITORING_INTEGRATION_PID > /dev/null; then
    echo "✅ Monitoring frontend integration is running successfully"
    
    # Display initial status
    echo ""
    echo "📊 Initial Monitoring Status:"
    echo "============================"
    
    # Check if frontend status file was created
    if [ -f "crypto_bot/logs/frontend_monitoring_status.json" ]; then
        echo "✅ Frontend monitoring status file created"
        echo "📁 Status file: crypto_bot/logs/frontend_monitoring_status.json"
    else
        echo "⚠️  Frontend monitoring status file not yet created"
    fi
    
    # Check if health status file was created
    if [ -f "crypto_bot/logs/health_status.json" ]; then
        echo "✅ Health status file created"
        echo "📁 Health file: crypto_bot/logs/health_status.json"
    else
        echo "⚠️  Health status file not yet created"
    fi
    
    # Check if monitoring report was created
    if [ -f "crypto_bot/logs/monitoring_report.json" ]; then
        echo "✅ Monitoring report file created"
        echo "📁 Report file: crypto_bot/logs/monitoring_report.json"
    else
        echo "⚠️  Monitoring report file not yet created"
    fi
    
    echo ""
    echo "🌐 Frontend Monitoring Dashboard:"
    echo "   http://localhost:8000/monitoring"
    echo ""
    echo "📋 Available API Endpoints:"
    echo "   GET /api/monitoring/health     - System health status"
    echo "   GET /api/monitoring/metrics    - Performance metrics"
    echo "   GET /api/monitoring/alerts     - Active alerts"
    echo "   GET /api/monitoring/logs       - Monitoring logs"
    echo "   GET /api/monitoring/status     - Monitoring system status"
    echo "   GET /api/monitoring/components - Component status"
    echo ""
    echo "🔄 Monitoring integration will update every 30 seconds"
    echo "📝 Logs available in: crypto_bot/logs/"
    echo ""
    echo "To stop the integration:"
    echo "  pkill -f integrate_monitoring_frontend.py"
    echo "  or"
    echo "  kill $MONITORING_INTEGRATION_PID"
    
else
    echo "❌ Failed to start monitoring frontend integration"
    echo "Check the logs for more information"
    exit 1
fi
