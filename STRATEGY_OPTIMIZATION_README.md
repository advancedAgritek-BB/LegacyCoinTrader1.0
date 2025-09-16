# Strategy Optimization Notes

Use this primer when refining trading strategies or evaluating new indicators. The goal is to iterate quickly without putting capital at unnecessary risk.

## Backtesting Workflow
1. Clone the baseline configuration from `config.yaml` to a strategy-specific file.
2. Run historical simulations with `backtest/run_backtest.py --config <file>`.
3. Export trade logs to `backtest/results/` and compare PnL, drawdown, and Sharpe ratio.

## Parameter Tuning
- Focus on three levers at a time (e.g., lookback, threshold, position size) to keep experiments interpretable.
- Store each experiment's parameters in `backtest/notes/` for traceability.
- Automate grid searches with `tune.py` when the search space becomes large.

## Risk Controls
- Keep max simultaneous positions aligned with wallet liquidity; update `position_limits` in the relevant config.
- Enable stop-loss rules in `set_stop_losses.py` during paper trading before promoting to production.
- Document every production deployment in `deploy_production.py` notes for postmortems.
