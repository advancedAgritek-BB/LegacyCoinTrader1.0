#!/usr/bin/env python3
"""
Simple Flask app to test environment variable loading.
"""

import os
from pathlib import Path
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Flask test app running"

@app.route('/env')
def show_env():
    """Show environment variables."""
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')

    response = f"""
    API_KEY: {bool(api_key)} ({api_key[:8] if api_key else 'None'}...)
    API_SECRET: {bool(api_secret)} ({api_secret[:8] if api_secret else 'None'}...)
    Working directory: {os.getcwd()}
    .env file exists: {Path('.env').exists()}
    """

    # Try to load .env directly
    try:
        with open('.env', 'r') as f:
            content = f.read()
        response += f"\n.env content length: {len(content)}"
    except Exception as e:
        response += f"\n.env read error: {e}"

    return response

@app.route('/test-exchange')
def test_exchange():
    """Test exchange connection."""
    try:
        import ccxt
        exchange = ccxt.kraken({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('API_SECRET'),
            'enableRateLimit': True,
        })

        ticker = exchange.fetch_ticker('BTC/USD')
        return f"Exchange working! BTC/USD: ${ticker['last']}"
    except Exception as e:
        return f"Exchange error: {e}"

if __name__ == '__main__':
    # Load environment variables directly
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
                        print(f"Loaded {key.strip()}")
    except Exception as e:
        print(f"Failed to load .env: {e}")

    print(f"API_KEY loaded: {bool(os.getenv('API_KEY'))}")
    print(f"API_SECRET loaded: {bool(os.getenv('API_SECRET'))}")

    app.run(host='0.0.0.0', port=8001, debug=True)
