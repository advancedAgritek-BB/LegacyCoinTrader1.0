from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Optional
import pandas as pd

import yaml


def load_execution_mode(config_file: Path) -> str:
    """Return execution mode from the YAML config."""
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f).get("execution_mode", "dry_run")
    return "dry_run"


def set_execution_mode(mode: str, config_file: Path) -> None:
    """Update execution mode in the YAML config."""
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    config["execution_mode"] = mode
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)


def compute_performance(df: pd.DataFrame) -> Dict[str, float]:
    """Return performance metrics from the trades dataframe."""
    if df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_trade_size': 0.0
        }
    
    # Calculate basic metrics
    total_trades = len(df)
    
    # Calculate PnL per symbol
    perf: Dict[str, float] = {}
    open_pos: dict[str, list[Tuple[float, float]]] = {}

    for _, row in df.iterrows():
        symbol = row["symbol"] if "symbol" in row else "Unknown"
        side = row["side"] if "side" in row else "buy"
        try:
            price = float(row["price"]) if "price" in row else 0.0
        except (ValueError, TypeError):
            price = 0.0
        try:
            amount = float(row["amount"]) if "amount" in row else 0.0
        except (ValueError, TypeError):
            amount = 0.0

        if side == "buy":
            # Buys first close any existing short positions
            while amount > 0 and open_pos.get(symbol, []) and open_pos[symbol][0][1] < 0:
                entry_price, qty = open_pos[symbol].pop(0)
                qty = -qty  # convert short quantity to positive
                traded = min(qty, amount)
                perf[symbol] = perf.get(symbol, 0.0) + (entry_price - price) * traded
                if qty > traded:
                    open_pos[symbol].insert(0, (entry_price, -(qty - traded)))
                amount -= traded
            # Remaining amount opens a new long position
            if amount > 0:
                if symbol not in open_pos:
                    open_pos[symbol] = []
                open_pos[symbol].append((price, amount))

        elif side == "sell":
            # Sells first close existing long positions
            while amount > 0 and open_pos.get(symbol, []) and open_pos[symbol][0][1] > 0:
                entry_price, qty = open_pos[symbol].pop(0)
                traded = min(qty, amount)
                perf[symbol] = perf.get(symbol, 0.0) + (price - entry_price) * traded
                if qty > traded:
                    open_pos[symbol].insert(0, (entry_price, qty - traded))
                amount -= traded
            # Excess amount starts a short position
            if amount > 0:
                if symbol not in open_pos:
                    open_pos[symbol] = []
                open_pos[symbol].append((price, -amount))

    # Calculate win rate (simplified - positive PnL trades)
    winning_trades = sum(1 for pnl in perf.values() if pnl > 0)
    win_rate = winning_trades / len(perf) if perf else 0.0
    
    # Calculate total PnL
    total_pnl = sum(perf.values())
    
    # Calculate average trade size
    avg_trade_size = df['amount'].mean() if 'amount' in df.columns else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_trade_size': avg_trade_size,
        'per_symbol': perf
    }


def is_running(proc: Optional[subprocess.Popen]) -> bool:
    """Return True if the given process is running."""
    return proc is not None and proc.poll() is None


def get_uptime(start_time: Optional[float]) -> str:
    """Return human-readable uptime from a start timestamp."""
    if start_time is None:
        return "-"
    delta = int(time.time() - start_time)
    hrs, rem = divmod(delta, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def get_last_trade(trade_file: Path) -> dict:
    """Return last trade from trades CSV as a dictionary."""
    if not trade_file.exists():
        return {}
    
    try:
        lines = trade_file.read_text().strip().split('\n')
        if not lines or not lines[-1].strip():
            return {}
        
        # Get the last non-empty line
        for line in reversed(lines):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 5:
                    return {
                        'symbol': parts[0],
                        'side': parts[1],
                        'amount': float(parts[2]),
                        'price': float(parts[3]),
                        'timestamp': parts[4]
                    }
                break
    except Exception as e:
        print(f"Error reading trade file: {e}")
    
    return {}


def get_current_regime(log_file: Path) -> str:
    """Return most recent regime classification from bot log."""
    if log_file.exists():
        lines = log_file.read_text().splitlines()
        for line in reversed(lines):
            if "Market regime classified as" in line:
                return line.rsplit("Market regime classified as", 1)[1].strip()
    return "N/A"


def get_last_decision_reason(log_file: Path) -> str:
    """Return the last evaluation reason from bot log."""
    if log_file.exists():
        lines = log_file.read_text().splitlines()
        for line in reversed(lines):
            if "[EVAL]" in line:
                return line.split("[EVAL]", 1)[1].strip()
    return "N/A"
