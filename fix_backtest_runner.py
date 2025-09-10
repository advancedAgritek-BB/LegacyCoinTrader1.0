#!/usr/bin/env python3
"""Fix type annotations in backtest_runner.py"""

with open('crypto_bot/backtest/backtest_runner.py', 'r') as f:
    content = f.read()

# Fix the type annotations
content = content.replace('stop_loss_range: Iterable[float] | None = None,', 'stop_loss_range: Optional[Iterable[float]] = None,')
content = content.replace('take_profit_range: Iterable[float] | None = None,', 'take_profit_range: Optional[Iterable[float]] = None,')

with open('crypto_bot/backtest/backtest_runner.py', 'w') as f:
    f.write(content)

print("Fixed type annotations in backtest_runner.py")
