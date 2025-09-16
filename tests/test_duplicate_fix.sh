#!/bin/bash
# Test script to verify duplicate position fix

echo "🧪 Testing Duplicate Position Fix"
echo "================================="

# Test 1: Check API for duplicates
echo "📊 Test 1: API Duplicate Check"
response=$(curl -s "http://localhost:8000/api/open-positions")
position_count=$(echo "$response" | python3 -c "import json, sys; data=json.load(sys.stdin); print(len(data))")
symbols=$(echo "$response" | python3 -c "import json, sys; data=json.load(sys.stdin); symbols=[p['symbol'] for p in data]; print(' '.join(symbols))")
duplicates=$(echo "$response" | python3 -c "import json, sys; data=json.load(sys.stdin); symbols=[p['symbol'] for p in data]; dups=[s for s in set(symbols) if symbols.count(s) > 1]; print(' '.join(dups))")

echo "   Positions returned: $position_count"
echo "   Symbols: $symbols"
if [ -z "$duplicates" ]; then
    echo "   ✅ No duplicates found in API"
else
    echo "   ❌ Duplicates found: $duplicates"
fi

# Test 2: Check dashboard page loads correctly
echo ""
echo "🌐 Test 2: Dashboard Page Load"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/dashboard | grep -q "200"; then
    echo "   ✅ Dashboard loads successfully"
else
    echo "   ❌ Dashboard failed to load"
fi

# Test 3: Check for any error logs
echo ""
echo "📋 Test 3: Error Log Check"
if [ -f "bot_debug.log" ]; then
    recent_errors=$(tail -20 bot_debug.log | grep -i "error\|duplicate" | wc -l)
    echo "   Recent errors in log: $recent_errors"
    if [ "$recent_errors" -eq 0 ]; then
        echo "   ✅ No recent errors found"
    else
        echo "   ⚠️  Found $recent_errors recent errors"
        tail -20 bot_debug.log | grep -i "error\|duplicate" | head -5
    fi
else
    echo "   ℹ️  No debug log found"
fi

echo ""
echo "🎯 Duplicate position fix test completed!"
echo "🌐 Open your browser to: http://localhost:8000/dashboard"
echo "📊 You should now see only unique position cards"
