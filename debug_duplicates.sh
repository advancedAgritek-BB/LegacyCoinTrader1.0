#!/bin/bash
# Debug script to check for multiple API calls and card creation

echo "🔍 Debugging Duplicate Cards Issue"
echo "=================================="

echo "📊 Checking API response..."
api_data=$(curl -s "http://localhost:8000/api/open-positions")
echo "API returns: $(echo "$api_data" | python3 -c "import json, sys; data=json.load(sys.stdin); print(len(data))") positions"

echo ""
echo "🌐 Opening browser with debug instructions..."
echo ""
echo "📋 Debug Steps:"
echo "1. Open http://localhost:8000/dashboard"
echo "2. Open browser console (F12)"
echo "3. Look for these messages:"
echo "   - '🔄 processAndDisplayPositions called with X positions'"
echo "   - '✅ After deduplication: X unique positions'"
echo "   - '🎉 Successfully displayed X position cards'"
echo ""
echo "4. Count how many times 'processAndDisplayPositions called' appears"
echo "5. Check if the numbers match what you see on screen"
echo ""
echo "🔍 Expected:"
echo "   - Should see 'processAndDisplayPositions called' only ONCE"
echo "   - Should see '7 unique positions'"
echo "   - Should see 'Successfully displayed 7 position cards'"
echo ""
echo "❌ If you see multiple calls or different numbers, that's the issue!"

# Try to open the browser
if command -v open >/dev/null 2>&1; then
    open "http://localhost:8000/dashboard"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:8000/dashboard"
else
    echo "🌐 Please manually open: http://localhost:8000/dashboard"
fi

echo ""
echo "💡 If you still see 14 cards:"
echo "   - Check browser console for multiple API calls"
echo "   - Look for any error messages"
echo "   - Try hard refresh (Ctrl+F5)"
