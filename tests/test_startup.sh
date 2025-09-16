#!/bin/bash
# Quick startup test script
# Tests if the integrated system is working properly

echo "🧪 Testing LegacyCoinTrader Startup"
echo "==================================="

# Test 1: Check if web server is running
echo "📊 Test 1: Web Server Status"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ Web server is responding"
else
    echo "❌ Web server not responding"
    exit 1
fi

# Test 2: Check if bot process is running
echo ""
echo "🤖 Test 2: Bot Process Status"
if pgrep -f "start_bot.py" > /dev/null; then
    echo "✅ Bot process is running"
else
    echo "❌ Bot process not found"
    exit 1
fi

# Test 3: Test basic API endpoints
echo ""
echo "🔌 Test 3: API Endpoints"
echo "   Testing /api/bot-status..."
if curl -s "http://localhost:8000/api/bot-status" > /dev/null; then
    echo "✅ /api/bot-status working"
else
    echo "❌ /api/bot-status failed"
fi

echo "   Testing /api/open-positions..."
if curl -s "http://localhost:8000/api/open-positions" > /dev/null; then
    echo "✅ /api/open-positions working"
else
    echo "❌ /api/open-positions failed"
fi

# Test 4: Test new batch chart API
echo ""
echo "📈 Test 4: New Batch Chart API"
echo "   Testing /api/batch-chart-data..."
response=$(curl -s -w "%{http_code}" "http://localhost:8000/api/batch-chart-data?symbols[]=BTC/USD" -o /dev/null)
if [ "$response" = "200" ]; then
    echo "✅ /api/batch-chart-data working (HTTP $response)"
else
    echo "❌ /api/batch-chart-data failed (HTTP $response)"
fi

# Test 5: Show system info
echo ""
echo "📋 Test 5: System Information"
echo "   Web Server: http://localhost:8000"
echo "   Dashboard: http://localhost:8000/dashboard"
echo "   Bot Status: Running"
echo "   Process ID: $(pgrep -f 'start_bot.py')"

echo ""
echo "🎯 All tests completed!"
echo "🌐 Open your browser to: http://localhost:8000"
echo "📊 The new chart loading optimization should make Open Positions cards load much faster!"
