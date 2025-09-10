#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/test-candle-data')
def test_api_candle_data():
    """Test API endpoint for candle data."""
    try:
        from frontend.app import generate_candle_data
        
        symbol = request.args.get('symbol', 'BTC/USD')
        limit = int(request.args.get('limit', 5))
        
        print(f"API request: {symbol}, limit: {limit}")
        
        candle_data = generate_candle_data(symbol, limit)
        
        if candle_data and len(candle_data) > 0:
            return jsonify({
                'symbol': symbol,
                'candles': candle_data,
                'success': True,
                'count': len(candle_data)
            })
        else:
            return jsonify({
                'error': f'No candle data available for {symbol}',
                'success': False
            }), 404
            
    except Exception as e:
        print(f"API Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/')
def home():
    return "Test server running"

if __name__ == '__main__':
    print("Starting test server on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)
