#!/bin/bash
# Comprehensive duplicate detection script

echo "🔍 Comprehensive Duplicate Detection"
echo "===================================="

# Test 1: Check server-side data
echo "📊 Test 1: Server-side Data Check"
server_data=$(curl -s "http://localhost:8000/dashboard" | grep -o "initialPositions.*" | head -1)
if echo "$server_data" | grep -q "HBAR/USD"; then
    hbar_count=$(echo "$server_data" | grep -o "HBAR/USD" | wc -l)
    echo "   HBAR/USD occurrences in server data: $hbar_count"
    if [ "$hbar_count" -eq 1 ]; then
        echo "   ✅ Server-side data is clean"
    else
        echo "   ❌ Server-side data has duplicates"
    fi
else
    echo "   ℹ️  HBAR/USD not found in server data"
fi

# Test 2: Check API data
echo ""
echo "🔌 Test 2: API Data Check"
api_data=$(curl -s "http://localhost:8000/api/open-positions")
hbar_api_count=$(echo "$api_data" | python3 -c "import json, sys; data=json.load(sys.stdin); symbols=[p['symbol'] for p in data]; print(symbols.count('HBAR/USD'))")
echo "   HBAR/USD occurrences in API: $hbar_api_count"
if [ "$hbar_api_count" -eq 1 ]; then
    echo "   ✅ API data is clean"
else
    echo "   ❌ API data has duplicates"
fi

# Test 3: Check for multiple data sources
echo ""
echo "📋 Test 3: Multiple Data Sources Check"
echo "   Checking if both server and API data are being used..."

# Count total positions in server data
server_positions=$(echo "$server_data" | python3 -c "
import re, sys
data = sys.stdin.read()
try:
    # Extract the JSON array from the string
    start = data.find('[')
    end = data.rfind(']') + 1
    json_str = data[start:end]
    import json
    positions = json.loads(json_str)
    print(len(positions))
except:
    print('0')
" 2>/dev/null)

echo "   Server-side positions: $server_positions"

# Count total positions in API
api_positions=$(echo "$api_data" | python3 -c "import json, sys; data=json.load(sys.stdin); print(len(data))" 2>/dev/null)
echo "   API positions: $api_positions"

# Test 4: Check browser console for clues
echo ""
echo "🌐 Test 4: Browser Console Check"
echo "   Open browser console and look for:"
echo "   - 'Using initial positions data from server...'"
echo "   - 'Using fallback positions data...'"
echo "   - 'Frontend: Skipping duplicate position for...'"
echo "   - Multiple 'Processing X positions' messages"

# Test 5: Manual verification
echo ""
echo "👀 Test 5: Manual Verification"
echo "   1. Open http://localhost:8000/dashboard"
echo "   2. Count the position cards"
echo "   3. Check if any symbols appear multiple times"
echo "   4. Look for duplicate cards with same symbol"

echo ""
echo "🎯 If duplicates persist, the issue might be:"
echo "   1. Browser caching the old JavaScript"
echo "   2. Multiple API calls happening"
echo "   3. Race condition between server and API data"
echo ""
echo "💡 Try:"
echo "   - Hard refresh (Ctrl+F5 or Cmd+Shift+R)"
echo "   - Clear browser cache"
echo "   - Check browser console for errors"
