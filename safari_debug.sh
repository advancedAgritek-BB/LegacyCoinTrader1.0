#!/bin/bash
# Safari-specific duplicate detection and fix

echo "🦁 Safari Duplicate Detection"
echo "============================"

echo "📊 Checking if this is Safari-specific..."
echo "   - Chrome/Edge: Should show 7 cards"
echo "   - Safari: Currently showing 14 cards"
echo ""

echo "🔍 Safari-specific issues that could cause duplication:"
echo "   1. Aggressive caching of JavaScript files"
echo "   2. Different async/await behavior"
echo "   3. Different event loop timing"
echo "   4. Different Promise resolution order"
echo ""

echo "🌐 Opening Safari with cache-busting..."
SAFARI_URL="http://localhost:8000/dashboard?cb=$(date +%s)&safari=1"

echo "📱 Safari URL: $SAFARI_URL"
echo ""
echo "💡 Safari-specific fixes to try:"
echo "   1. Hard refresh: Cmd+Option+R"
echo "   2. Clear Safari cache: Safari > Preferences > Privacy > Manage Website Data"
echo "   3. Disable Safari cache: Develop > Disable Caches"
echo "   4. Check console for Safari-specific errors"
echo ""

# Try to open Safari specifically
if command -v open >/dev/null 2>&1; then
    echo "🚀 Opening Safari..."
    open -a Safari "$SAFARI_URL"
else
    echo "🌐 Please manually open Safari and go to: $SAFARI_URL"
fi

echo ""
echo "🔧 If Safari still shows duplicates:"
echo "   1. Open Safari Developer Tools (Develop > Show Web Inspector)"
echo "   2. Check Console tab for errors"
echo "   3. Look for multiple 'processAndDisplayPositions called' messages"
echo "   4. Check Network tab for multiple API calls"
echo ""
echo "📋 Safari Console Debug Messages to Look For:"
echo "   - '🔄 processAndDisplayPositions called with 7 positions'"
echo "   - '✅ After deduplication: 7 unique positions'"
echo "   - '🎉 Successfully displayed 7 position cards'"
echo "   - 'Initial load complete, setting up auto-refresh...'"
