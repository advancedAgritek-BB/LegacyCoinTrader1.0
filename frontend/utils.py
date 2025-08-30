from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Optional, Dict, Tuple
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
    try:
        if 'amount' in df.columns and not df.empty:
            # Check if the amount column contains numeric data
            numeric_amounts = pd.to_numeric(df['amount'], errors='coerce')
            avg_trade_size = numeric_amounts.mean() if not numeric_amounts.isna().all() else 0.0
        else:
            avg_trade_size = 0.0
    except Exception:
        avg_trade_size = 0.0
    
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


def calculate_dynamic_allocation() -> Dict[str, float]:
    """Calculate dynamic strategy allocation based on actual performance data."""
    import json
    from pathlib import Path
    
    # Path to strategy stats file
    # Try multiple possible paths since Flask might run from different directories
    possible_paths = [
        Path('crypto_bot/logs/strategy_stats.json'),
        Path('../crypto_bot/logs/strategy_stats.json'),
        Path('../../crypto_bot/logs/strategy_stats.json')
    ]
    
    strategy_stats_file = None
    for path in possible_paths:
        if path.exists():
            strategy_stats_file = path
            break
    
    if strategy_stats_file is None:
        return {}
    
    if not strategy_stats_file.exists():
        return {}
    
    try:
        with open(strategy_stats_file) as f:
            stats_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}
    
    # Calculate performance scores for each strategy
    strategy_scores = {}
    
    for regime, strategies in stats_data.items():
        for strategy_name, strategy_data in strategies.items():
            if not isinstance(strategy_data, dict):
                continue
                
            # Extract performance metrics
            trades = strategy_data.get('trades', 0)
            win_rate = strategy_data.get('win_rate', 0.0)
            total_pnl = strategy_data.get('total_pnl', 0.0)
            
            # Skip strategies with no trades
            if trades == 0:
                continue
            
            # Calculate composite score based on multiple factors
            # Weight factors: win rate (40%), PnL per trade (40%), trade volume (20%)
            pnl_per_trade = total_pnl / trades if trades > 0 else 0.0
            
            # Normalize PnL per trade (assuming typical range of -100 to +1000)
            normalized_pnl = max(0.0, min(1.0, (pnl_per_trade + 100) / 1100))
            
            # Normalize trade volume (assuming typical range of 0 to 100 trades)
            normalized_volume = min(1.0, trades / 100.0)
            
            # Calculate composite score
            composite_score = (
                win_rate * 0.4 +
                normalized_pnl * 0.4 +
                normalized_volume * 0.2
            )
            
            strategy_scores[strategy_name] = composite_score
    
    # Normalize scores to sum to 1.0 (convert to percentages)
    total_score = sum(strategy_scores.values())
    
    if total_score == 0:
        # If no performance data, return equal allocation
        num_strategies = len(strategy_scores)
        if num_strategies > 0:
            return {strategy: 1.0 / num_strategies for strategy in strategy_scores}
        return {}
    
    # Convert to percentages
    allocation = {
        strategy: (score / total_score) * 100 
        for strategy, score in strategy_scores.items()
    }
    
    return allocation


def get_allocation_comparison() -> Dict[str, Dict[str, float]]:
    """Get both static and dynamic allocation for comparison."""
    import yaml
    from pathlib import Path
    
    # Get dynamic allocation
    dynamic_allocation = calculate_dynamic_allocation()
    
    # Get static allocation from config
    static_allocation = {}
    config_file = Path('crypto_bot/config.yaml')
    if config_file.exists():
        try:
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
                static_allocation = cfg.get('strategy_allocation', {})
        except (yaml.YAMLError, FileNotFoundError):
            pass
    
    return {
        'dynamic': dynamic_allocation,
        'static': static_allocation
    }