#!/bin/bash
# Open web interface with cache-busting

echo "🌐 Opening LegacyCoinTrader Web Interface"
echo "========================================="

# Add cache-busting parameter
CACHE_BUST=$(date +%s)
URL="http://localhost:8000/dashboard?cb=$CACHE_BUST"

echo "📊 Dashboard URL: $URL"
echo ""
echo "🔧 If you're still seeing duplicate cards:"
echo "   1. Hard refresh the page (Ctrl+F5 or Cmd+Shift+R)"
echo "   2. Clear browser cache (Ctrl+Shift+Delete)"
echo "   3. Open browser console (F12) and look for:"
echo "      - 'Using initial positions data from server...'"
echo "      - 'Processing X unique positions'"
echo "      - Any error messages"
echo ""
echo "📋 Expected behavior:"
echo "   - Should see 7 unique position cards"
echo "   - Each symbol should appear only once"
echo "   - Charts should load quickly (optimized)"
echo ""

# Try to open the browser
if command -v open >/dev/null 2>&1; then
    echo "🚀 Opening browser..."
    open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then
    echo "🚀 Opening browser..."
    xdg-open "$URL"
else
    echo "🌐 Please manually open: $URL"
fi

echo ""
echo "✅ Web interface should now be open with cache-busting!"
echo "🔄 If duplicates persist, try a hard refresh (Ctrl+F5)"
