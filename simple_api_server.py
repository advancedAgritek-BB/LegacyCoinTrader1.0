#!/usr/bin/env python3
"""
Simple API server for bot status that works independently
"""

import subprocess
import json
import time
from pathlib import Path
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/bot/status')
def get_bot_status():
    """Get current bot system status."""
    try:
        status = {
            'bot_running': False,
            'execution_running': False,
            'trading_active': False,
            'processes': [],
            'last_trade': None,
            'last_execution': None
        }

        # Check if bot processes are running
        try:
            # Check for main bot process - look for multiple possible process names
            bot_process_patterns = [
                'main.py',
                'start_bot.py',
                'crypto_bot'
            ]
            
            for pattern in bot_process_patterns:
                result = subprocess.run(
                    ['pgrep', '-f', pattern],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    status['bot_running'] = True
                    processes = result.stdout.strip().split('\n')
                    status['processes'].extend([p for p in processes if p])
                    break

            # Check for execution processes
            result = subprocess.run(
                ['pgrep', '-f', 'execution'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                status['execution_running'] = True
                status['processes'].extend(result.stdout.strip().split('\n'))
        except Exception as e:
            print(f"Error in process detection: {e}")

        # Check if there are recent trades (indicating active trading)
        trades_file = Path(__file__).parent / 'crypto_bot' / 'logs' / 'trades.csv'
        if trades_file.exists():
            try:
                # Get file modification time
                mod_time = trades_file.stat().st_mtime
                # If modified within last 5 minutes, consider trading active
                if time.time() - mod_time < 300:  # 5 minutes
                    status['trading_active'] = True
                    status['last_trade'] = mod_time
            except Exception as e:
                print(f"Error checking trades file: {e}")

        # Check execution log for recent activity
        execution_log = Path(__file__).parent / 'crypto_bot' / 'logs' / 'execution.log'
        if execution_log.exists():
            try:
                mod_time = execution_log.stat().st_mtime
                status['last_execution'] = mod_time
            except Exception as e:
                print(f"Error checking execution log: {e}")

        return jsonify({
            'success': True,
            'data': status,
            'timestamp': time.time(),
            'debug': {
                'patterns_checked': bot_process_patterns,
                'api_server': 'standalone'
            }
        })

    except Exception as e:
        print(f"Error getting bot status: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        })

@app.route('/test')
def test():
    """Simple test endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'Standalone API server is running',
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("🚀 Starting standalone API server on port 8003...")
    app.run(host='0.0.0.0', port=8003, debug=False)
