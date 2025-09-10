#!/bin/bash
# Comprehensive duplicate verification script

echo "🔍 Comprehensive Duplicate Verification"
echo "======================================"

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
    echo "   ℹ️  HBAR/USD not found in server data (this is expected now)"
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

# Test 3: Count total positions
echo ""
echo "📋 Test 3: Total Position Count"
api_positions=$(echo "$api_data" | python3 -c "import json, sys; data=json.load(sys.stdin); print(len(data))" 2>/dev/null)
echo "   Total positions in API: $api_positions"

# Test 4: Check for duplicate symbols in API
echo ""
echo "🔍 Test 4: Duplicate Symbol Check"
duplicate_check=$(echo "$api_data" | python3 -c "
import json, sys
data = json.load(sys.stdin)
symbols = [p['symbol'] for p in data]
unique_symbols = set(symbols)
if len(symbols) == len(unique_symbols):
    print('✅ No duplicate symbols found')
else:
    print('❌ Duplicate symbols found:')
    for symbol in symbols:
        if symbols.count(symbol) > 1:
            print(f'   {symbol} appears {symbols.count(symbol)} times')
")

echo "   $duplicate_check"

# Test 5: Manual verification instructions
echo ""
echo "👀 Test 5: Manual Verification"
echo "   1. Open http://localhost:8000/dashboard"
echo "   2. Count the position cards (should be $api_positions)"
echo "   3. Check if any symbols appear multiple times"
echo "   4. Look for duplicate cards with same symbol"
echo "   5. Open browser console (F12) and look for:"
echo "      - 'Processing X positions from API...'"
echo "      - 'After deduplication: X unique positions'"
echo "      - 'Successfully displayed X position cards'"

echo ""
echo "🎯 Expected Results:"
echo "   - Server-side data: 0 or 1 HBAR/USD (0 is expected now)"
echo "   - API data: 1 HBAR/USD"
echo "   - Total positions: $api_positions"
echo "   - No duplicate symbols"
echo "   - Browser shows exactly $api_positions cards"

echo ""
echo "💡 If duplicates persist:"
echo "   1. Hard refresh (Ctrl+F5 or Cmd+Shift+R)"
echo "   2. Clear browser cache (Ctrl+Shift+Delete)"
echo "   3. Check browser console for multiple API calls"
echo "   4. Restart system: ./stop_integrated.sh && ./start_integrated.sh"
