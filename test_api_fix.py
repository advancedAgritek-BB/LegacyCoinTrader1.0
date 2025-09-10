#!/usr/bin/env python3
"""
Test the API fixes
"""

import requests
import json

def main():
    try:
        response = requests.get("http://localhost:8000/api/open-positions", timeout=10)
        response.raise_for_status()
        data = response.json()

        if data:
            print(f'API returned {len(data)} positions:')
            total_value = 0.0
            for pos in data:
                value = pos.get('position_value', 0)
                total_value += value
                print(f'{pos["symbol"]}: position_value=${value:.2f}, current_price={pos["current_price"]}, amount={pos["amount"]}')
            print(f'\nTotal position value from API: ${total_value:.2f}')
        else:
            print('No positions returned')

    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()
