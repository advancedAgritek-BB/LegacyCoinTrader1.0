from __future__ import annotations

"""Utility for computing trade statistics from log files."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd


def _read_trades(path: Union[Path, str]) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        return pd.DataFrame(columns=["symbol", "side", "amount", "price", "timestamp"])

    try:
        # First try to read with header detection
        df = pd.read_csv(file, engine="python")
        
        # Check if the first row looks like a header (contains string values)
        if not df.empty and df.iloc[0].dtype == 'object':
            # First row is likely a header, check if it contains expected column names
            first_row = df.iloc[0].astype(str).str.lower()
            if any(col in first_row.values for col in ['symbol', 'side', 'amount', 'price', 'timestamp']):
                # This is a header row, skip it
                df = df.iloc[1:].reset_index(drop=True)
        
        # Ensure we have the expected columns
        expected_cols = ["symbol", "side", "amount", "price", "timestamp"]
        if not df.empty and len(df.columns) >= len(expected_cols):
            df = df.iloc[:, :len(expected_cols)]
            df.columns = expected_cols
        else:
            # Fallback to reading without header
            df = pd.read_csv(
                file,
                header=None,
                names=expected_cols,
                engine="python",
                on_bad_lines=lambda row: row[: len(expected_cols)],
            )
            df = df.iloc[:, : len(expected_cols)]
            
    except Exception:
        # If all else fails, return empty DataFrame
        df = pd.DataFrame(columns=["symbol", "side", "amount", "price", "timestamp"])
    
    return df


def trade_summary(path: Union[Path, str]) -> Dict[str, float]:
    df = _read_trades(path)
    num_trades = len(df)
    pnl = 0.0
    wins = 0
    closed = 0
    # Track open long and short trades separately
    open_longs: List[Tuple[float, float]] = []
    open_shorts: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        side = row.get("side")
        try:
            price = float(row.get("price", 0))
        except Exception:
            price = 0.0
        try:
            amount = float(row.get("amount", 0))
        except Exception:
            amount = 0.0
        if side == "buy":
            # Buy orders first close any open shorts
            if open_shorts:
                entry_price, qty = open_shorts.pop(0)
                traded = min(qty, amount)
                profit = (entry_price - price) * traded
                pnl += profit
                closed += 1
                if profit > 0:
                    wins += 1
                if qty > traded:
                    open_shorts.insert(0, (entry_price, qty - traded))
                amount -= traded
            # Remaining amount opens new long position
            if amount > 0:
                open_longs.append((price, amount))
        elif side == "sell":
            # Sell orders close longs first
            if open_longs:
                entry_price, qty = open_longs.pop(0)
                traded = min(qty, amount)
                profit = (price - entry_price) * traded
                pnl += profit
                closed += 1
                if profit > 0:
                    wins += 1
                if qty > traded:
                    open_longs.insert(0, (entry_price, qty - traded))
                amount -= traded
            # Excess sell amount opens short position
            if amount > 0:
                open_shorts.append((price, amount))
    win_rate = wins / closed if closed else 0.0
    active = sum(qty for _, qty in open_longs) + sum(qty for _, qty in open_shorts)
    return {
        "num_trades": num_trades,
        "total_pnl": pnl,
        "win_rate": win_rate,
        "active_positions": active,
    }
