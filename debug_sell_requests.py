#!/usr/bin/env python3
"""
Debug script to test sell request processing
"""

import json
import sys
from pathlib import Path

# Add the crypto_bot directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'crypto_bot'))

def debug_sell_requests():
    # Load sell requests
    sell_requests_file = Path('crypto_bot/logs/sell_requests.json')

    if not sell_requests_file.exists():
        print("Sell requests file does not exist")
        return

    with open(sell_requests_file, 'r') as f:
        sell_requests = json.load(f)

    print(f"Found {len(sell_requests)} sell requests:")
    for i, request in enumerate(sell_requests):
        print(f"  {i+1}: {request}")

    # Check paper wallet state
    paper_wallet_state_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
    if paper_wallet_state_file.exists():
        import yaml
        with open(paper_wallet_state_file, 'r') as f:
            wallet_state = yaml.safe_load(f)
        print(f"\nPaper wallet state:")
        print(f"  Balance: ${wallet_state.get('balance', 0):.2f}")
        print(f"  Positions: {wallet_state.get('positions', {})}")
        print(f"  Realized PnL: ${wallet_state.get('realized_pnl', 0):.2f}")

    # Check if we can find GOAT/USD in positions
    positions = wallet_state.get('positions', {})
    if 'GOAT/USD' in positions:
        print(f"\nGOAT/USD position found: {positions['GOAT/USD']}")
    else:
        print("\nNo GOAT/USD position found in paper wallet")

if __name__ == '__main__':
    debug_sell_requests()
