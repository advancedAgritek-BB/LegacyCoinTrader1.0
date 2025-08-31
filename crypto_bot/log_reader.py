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
        # Try to read with pandas' built-in header detection first
        df = pd.read_csv(file, engine="python")

        # Check if we need to handle the header manually
        expected_cols = ["symbol", "side", "amount", "price", "timestamp"]
        actual_cols = [str(col).lower() for col in df.columns]

        # If the columns match our expected columns, use them as is
        if all(col in actual_cols for col in ['symbol', 'side', 'amount', 'price', 'timestamp']):
            # Good, columns are properly named
            pass
        elif len(df.columns) >= len(expected_cols):
            # Try to detect if first row is header by checking if it contains column names
            first_row_values = df.iloc[0].astype(str).values
            header_keywords = ['symbol', 'side', 'amount', 'price', 'timestamp']

            # Check if first row contains header-like values
            if any(keyword in ' '.join(first_row_values).lower() for keyword in header_keywords):
                # First row looks like header, skip it
                df = df.iloc[1:].reset_index(drop=True)
                df.columns = expected_cols[:len(df.columns)]
            else:
                # No header detected, set column names
                df.columns = expected_cols[:len(df.columns)]
        else:
            # Too few columns, set expected names
            df.columns = expected_cols[:len(df.columns)]

        # Ensure we only keep the expected columns
        df = df[expected_cols[:len(df.columns)]]

    except Exception as e:
        # If pandas fails, try manual reading
        try:
            with open(file, 'r') as f:
                lines = f.readlines()

            if not lines:
                return pd.DataFrame(columns=expected_cols)

            data = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        # Try to parse the row
                        try:
                            symbol = str(parts[0])
                            side = str(parts[1])
                            amount = float(parts[2]) if parts[2] else 0.0
                            price = float(parts[3]) if parts[3] else 0.0
                            timestamp = str(parts[4])
                            data.append([symbol, side, amount, price, timestamp])
                        except (ValueError, IndexError):
                            # Skip malformed rows
                            continue

            df = pd.DataFrame(data, columns=expected_cols)

        except Exception:
            # If all else fails, return empty DataFrame
            df = pd.DataFrame(columns=expected_cols)

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
