#!/usr/bin/env python3
"""
Final direct fix for evaluation pipeline:
Patch the symbol loading function to only use supported symbols
"""

import os
import sys
from pathlib import Path

def create_symbol_patch():
    """Create a patch for the symbol loading function."""
    
    print("🔧 Creating symbol loading patch...")
    
    # Create a patch file that overrides the symbol loading
    patch_content = '''#!/usr/bin/env python3
"""
Symbol loading patch to ensure only supported symbols are used
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Patch the symbol loading function
def patched_get_filtered_symbols(exchange, config):
    """Patched version that only returns supported symbols."""
    
    # Define supported symbols
    supported_symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
        "LINK/USD", "UNI/USD", "AAVE/USD", "AVAX/USD",
        "BTC/EUR", "ETH/EUR", "SOL/EUR", "ADA/EUR"
    ]
    
    # Return only supported symbols with default scores
    return [(symbol, 1.0) for symbol in supported_symbols]

# Apply the patch
import crypto_bot.utils.symbol_utils
crypto_bot.utils.symbol_utils.get_filtered_symbols = patched_get_filtered_symbols

print("✅ Symbol loading patch applied")

# Start the main application
if __name__ == "__main__":
    import crypto_bot.main
'''
    
    with open("start_with_patch.py", 'w') as f:
        f.write(patch_content)
    
    os.chmod("start_with_patch.py", 0o755)
    print("✅ Created symbol patch: start_with_patch.py")
    
    # Create restart script with patch
    restart_script = """#!/bin/bash
echo "🔄 Restarting with symbol patch..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Start with patch
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 start_with_patch.py > crypto_bot/logs/bot_patched.log 2>&1 &

echo "✅ Bot restarted with patch (PID: $!)"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_patched.log"

# Wait and check
sleep 15
echo "🔍 Checking bot status..."
if pgrep -f "python.*start_with_patch" > /dev/null; then
    echo "✅ Bot is running with patch"
    echo "📈 Check logs for signal generation"
else
    echo "❌ Bot failed to start with patch"
fi
"""
    
    with open("restart_with_patch.sh", 'w') as f:
        f.write(restart_script)
    
    os.chmod("restart_with_patch.sh", 0o755)
    print("✅ Created restart script: restart_with_patch.sh")
    
    print("\n🎉 Symbol patch created!")
    print("🚀 Run './restart_with_patch.sh' to restart with the patch")

if __name__ == "__main__":
    create_symbol_patch()
