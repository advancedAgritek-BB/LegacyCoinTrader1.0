#!/usr/bin/env python3

import asyncio
from crypto_bot.bot_controller import TradingBotController

async def test_list_positions():
    print("Testing TradingBotController.list_positions()...")

    # Create bot controller
    controller = TradingBotController()

    # Test list_positions method
    positions = await controller.list_positions()

    print(f'Bot controller returned {len(positions)} positions:')
    if positions:
        print('Sample position:', positions[0])
        print('All positions:')
        for pos in positions:
            print(f'  {pos["symbol"]}: {pos["side"]} {pos["amount"]} @ ${pos["price"]} (P&L: ${pos["pnl"]:+.2f})')
    else:
        print('No positions found')

    print()
    print('SUCCESS: Bot controller list_positions method works!')

if __name__ == "__main__":
    asyncio.run(test_list_positions())
