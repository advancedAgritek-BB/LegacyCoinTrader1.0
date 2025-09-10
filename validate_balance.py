#!/usr/bin/env python3
"""
Balance Protection Script
This script validates that the paper wallet balance doesn't go negative.
"""

import yaml
from pathlib import Path

def validate_balance():
    pw_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
    if pw_file.exists():
        with open(pw_file, 'r') as f:
            state = yaml.safe_load(f)
        
        balance = state.get('balance', 0)
        if balance < 0:
            print(f"WARNING: Paper wallet balance is negative: ${balance:.2f}")
            return False
    return True

if __name__ == "__main__":
    validate_balance()
