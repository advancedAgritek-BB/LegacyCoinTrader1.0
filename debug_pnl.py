#!/usr/bin/env python3
"""
Debug script to test PnL calculation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.app import calculate_wallet_pnl, api_wallet_pnl
from flask import Flask
import json

def test_pnl_calculation():
    """Test the PnL calculation directly."""
    print("=== Direct Function Test ===")
    result = calculate_wallet_pnl()
    print("Function result:")
    print(json.dumps(result, indent=2))
    print(f"\nTotal PnL field exists: {'total_pnl' in result}")
    print(f"Total PnL value: {result.get('total_pnl', 'NOT FOUND')}")

def test_api_endpoint():
    """Test the API endpoint."""
    print("\n=== API Endpoint Test ===")
    
    # Create a minimal Flask app for testing
    app = Flask(__name__)
    app.add_url_rule('/test', 'test', api_wallet_pnl)
    
    with app.test_client() as client:
        response = client.get('/test')
        data = response.get_json()
        print("API response:")
        print(json.dumps(data, indent=2))
        print(f"\nTotal PnL field exists: {'total_pnl' in data}")
        print(f"Total PnL value: {data.get('total_pnl', 'NOT FOUND')}")

if __name__ == "__main__":
    test_pnl_calculation()
    test_api_endpoint()
