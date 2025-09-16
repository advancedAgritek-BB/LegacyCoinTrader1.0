#!/bin/bash
# Bot Status Checker Script

echo "🤖 LegacyCoinTrader Status Checker"
echo "=================================="

# Check if bot is running
if pgrep -f "start_bot.py" > /dev/null 2>&1; then
    echo "✅ Bot is RUNNING"
    echo "   Process ID: $(pgrep -f 'start_bot.py')"
    echo ""
    echo "📊 Available Interfaces:"
    echo "   • Web Dashboard: http://localhost:8000 (if web server is running)"
    echo "   • Monitoring Dashboard: http://localhost:8000/monitoring"
    echo "   • System Logs: http://localhost:8000/system_logs"
    echo ""
    echo "📋 Log Files:"
    echo "   • Main Bot Log: crypto_bot/logs/bot.log"
    echo "   • Bot Output: bot_output.log"
    echo "   • Frontend Log: frontend.log"
    echo ""
    echo "🎮 Control Options:"
    echo "   • Stop bot: pkill -f 'start_bot.py'"
    echo "   • View logs: tail -f bot_output.log"
    echo "   • Check web interface: curl http://localhost:8000"
    echo ""
    echo "💡 Note: The bot runs in the background by design."
    echo "   The terminal returning to prompt is normal behavior."
    echo "   Use the web interface or logs to monitor the bot."
else
    echo "❌ Bot is NOT RUNNING"
    echo ""
    echo "🚀 To start the bot:"
    echo "   ./start_integrated.sh"
    echo ""
    echo "📝 The bot will start in the background and return control to terminal."
    echo "   This is the expected behavior for the integrated system."
fi

echo ""
echo "🔍 Quick Commands:"
echo "   • Check this status: ./bot_status.sh"
echo "   • View live logs: tail -f bot_output.log"
echo "   • Stop bot: pkill -f 'start_bot.py'"
echo "   • Start bot (fixed): ./start_integrated_fixed.sh"
echo "   • Restart bot: ./start_integrated_fixed.sh"
