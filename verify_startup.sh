#!/bin/bash
# Quick startup verification script
# This script checks if the system is ready to start

echo "🔍 LegacyCoinTrader Startup Verification"
echo "========================================"

# Check if any processes are still running
echo "📋 Checking for running processes..."
if pgrep -f "start_bot.py\|frontend/app.py\|crypto_bot/main.py" > /dev/null; then
    echo "⚠️  Found running processes. Please stop them first:"
    echo "   ./stop_integrated.sh"
    exit 1
else
    echo "✅ No conflicting processes found"
fi

# Check if required files exist
echo ""
echo "📁 Checking required files..."
required_files=(
    "start_integrated.sh"
    "frontend/app.py"
    "crypto_bot/main.py"
    "requirements.txt"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Check if virtual environment exists
echo ""
echo "🐍 Checking Python environment..."
if [ -d "venv" ]; then
    echo "✅ Virtual environment found"
else
    echo "⚠️  Virtual environment not found"
    echo "   Run: python -m venv venv"
    echo "   Then: source venv/bin/activate"
fi

# Check if we're in the right directory
echo ""
echo "📂 Checking current directory..."
if [[ "$PWD" == *"LegacyCoinTrader"* ]]; then
    echo "✅ In LegacyCoinTrader directory"
else
    echo "⚠️  Not in LegacyCoinTrader directory"
    echo "   Current: $PWD"
fi

echo ""
echo "🎯 System ready to start!"
echo "   Run: ./start_integrated.sh"
echo ""
