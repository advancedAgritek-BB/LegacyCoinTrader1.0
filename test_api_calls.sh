#!/bin/bash
# Quick test to check if multiple API calls are happening

echo "🔍 Quick API Call Test"
echo "====================="

echo "📊 Testing API response..."
for i in {1..3}; do
    echo "Call $i:"
    curl -s "http://localhost:8000/api/open-positions" | python3 -c "import json, sys; data=json.load(sys.stdin); print(f'  Positions: {len(data)}')"
done

echo ""
echo "🌐 Now check browser console for:"
echo "   - How many times 'processAndDisplayPositions called' appears"
echo "   - How many times 'Loading positions from API...' appears"
echo ""
echo "💡 If you see multiple API calls, that's the source of duplication!"

# Open browser
if command -v open >/dev/null 2>&1; then
    open "http://localhost:8000/dashboard"
fi
