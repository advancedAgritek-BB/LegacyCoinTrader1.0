#!/bin/bash
# Final verification script

echo "🎯 Final Duplicate Verification"
echo "==============================="

echo "📊 Server-side data check..."
server_count=$(curl -s "http://localhost:8000/dashboard" | grep -c "HBAR/USD")
echo "   HBAR/USD in server data: $server_count"

echo ""
echo "🔌 API data check..."
api_data=$(curl -s "http://localhost:8000/api/open-positions")
api_count=$(echo "$api_data" | python3 -c "import json, sys; data=json.load(sys.stdin); print(len(data))")
echo "   API positions: $api_count"

echo ""
echo "🌐 Browser verification instructions..."
echo "1. Open http://localhost:8000/dashboard"
echo "2. Open browser console (F12)"
echo "3. Look for these messages:"
echo "   - '🔄 processAndDisplayPositions called with 7 positions'"
echo "   - '✅ After deduplication: 7 unique positions'"
echo "   - '🎉 Successfully displayed 7 position cards'"
echo "   - 'Initial load complete, setting up auto-refresh...'"
echo ""
echo "4. Count the position cards on screen"
echo "5. Check for any error messages"
echo ""
echo "🎯 Expected Results:"
echo "   - Should see exactly 7 position cards"
echo "   - Should see 'processAndDisplayPositions called' only ONCE"
echo "   - Should see 'Initial load complete' message"
echo ""
echo "❌ If you still see 14 cards:"
echo "   - Check console for multiple 'processAndDisplayPositions called' messages"
echo "   - Look for any error messages"
echo "   - Try hard refresh (Ctrl+F5)"

# Open browser
if command -v open >/dev/null 2>&1; then
    open "http://localhost:8000/dashboard"
fi
