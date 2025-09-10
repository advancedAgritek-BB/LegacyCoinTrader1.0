#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'crypto_bot'))

from paper_wallet import PaperWallet
import yaml

def check_wallet_state():
    # Load paper wallet
    try:
        with open('crypto_bot/logs/paper_wallet.yaml', 'r') as f:
            wallet_data = yaml.safe_load(f)

        wallet = PaperWallet(wallet_data['balance'])
        wallet.positions = wallet_data['positions']

        print('Paper Wallet Positions:')
        for symbol, pos in wallet.positions.items():
            print(f'  {symbol}: {pos}')

        print(f'Balance: ${wallet.balance:.2f}')

        # Also check if there are any open positions in the main context
        try:
            import json
            from frontend.app import get_open_positions
            positions = get_open_positions()
            print(f'\nOpen positions from API: {len(positions)} positions')
            for pos in positions:
                print(f'  {pos["symbol"]}: {pos["amount"]} @ ${pos["entry_price"]}')
        except Exception as e:
            print(f'Error checking API positions: {e}')

    except Exception as e:
        print(f'Error loading paper wallet: {e}')

if __name__ == '__main__':
    check_wallet_state()
