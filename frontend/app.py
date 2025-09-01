"""Start the web dashboard and expose REST endpoints for the trading bot.

This module launches the Flask web server, manages the background trading
process and provides REST API routes used by the UI and tests.
"""

try:
    from flask import Flask, render_template, redirect, url_for, request, jsonify
except Exception:  # pragma: no cover - provide minimal shim for import-time tests
    class _Dummy:
        def __getattr__(self, _):
            return self
        def __call__(self, *a, **k):
            return None
    Flask = lambda *a, **k: _Dummy()  # type: ignore
    render_template = redirect = url_for = request = jsonify = _Dummy()
from pathlib import Path
import os
import signal
from typing import Optional, Dict, Any

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None

from crypto_bot.utils.logger import LOG_DIR
import subprocess
import json
import threading
import time
import yaml
import requests
from crypto_bot import log_reader
from crypto_bot import ml_signal_model as ml
import frontend.utils as utils
from typing import Dict
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Disable caching for development
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Handle the async trading bot process
bot_proc = None
bot_start_time = None
watch_thread = None

# Context processor to make bot status available to all templates
@app.context_processor
def inject_bot_status():
    return {
        'running': is_running(),
        'mode': load_execution_mode(),
        'uptime': get_uptime()
    }
LOG_FILE = LOG_DIR / 'bot.log'
STATS_FILE = LOG_DIR / 'strategy_stats.json'
SCAN_FILE = LOG_DIR / 'asset_scores.json'
MODEL_REPORT = Path('crypto_bot/ml_signal_model/models/model_report.json')
TRADE_FILE = LOG_DIR / 'trades.csv'
ERROR_FILE = LOG_DIR / 'errors.log'
CONFIG_FILE = Path('crypto_bot/config.yaml')
REGIME_FILE = LOG_DIR / 'regime_history.txt'
POSITIONS_FILE = LOG_DIR / 'positions.log'

# Load environment variables from .env file
ENV_FILE = Path('.env')
if ENV_FILE.exists() and dotenv_values:
    print("Loading environment variables from .env file")
    env_vars = dotenv_values(str(ENV_FILE))
    os.environ.update(env_vars)
    print(f"Loaded {len(env_vars)} environment variables")
else:
    print("No .env file found or dotenv not available")


def stop_conflicting_bots() -> None:
    """Stop any other bot processes that might be running to prevent conflicts."""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline'] and 'crypto_bot.main' in ' '.join(proc.info['cmdline']):
                if proc.info['pid'] != os.getpid():  # Don't kill ourselves
                    print(f"Stopping conflicting bot process (PID {proc.info['pid']})")
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
        time.sleep(2)  # Give processes time to terminate
    except ImportError:
        pass


def check_existing_bot() -> bool:
    """Check if there's already a bot process running to prevent conflicts."""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline']:
                cmdline_str = ' '.join(proc.info['cmdline'])
                # Check for various bot startup patterns
                if any(pattern in cmdline_str for pattern in [
                    'crypto_bot.main',
                    'start_bot_noninteractive.py',
                    'start_bot_auto.py'
                ]):
                    return True
        return False
    except ImportError:
        # psutil not available, use basic check
        return False


def watch_bot() -> None:
    """Monitor the trading bot and restart it if the process exits."""
    global bot_proc, bot_start_time
    while True:
        time.sleep(5)
        if bot_proc is not None and bot_proc.poll() is not None:
            # Check if there's already another bot process running to avoid conflicts
            if not check_existing_bot():
                print("Bot process exited, restarting...")
                venv_python = Path(__file__).parent.parent / 'venv' / 'bin' / 'python3'
                bot_script = Path(__file__).parent.parent / 'start_bot_noninteractive.py'
                bot_proc = subprocess.Popen([str(venv_python), str(bot_script)])
                bot_start_time = time.time()
            else:
                print("Another bot process detected, skipping restart to avoid conflicts")
                bot_proc = None



def is_running() -> bool:
    """Return True if the bot process is running."""
    # Check if we have a tracked subprocess
    if bot_proc and bot_proc.poll() is None:
        return True

    # Also check for existing bot processes
    return check_existing_bot()


def set_execution_mode(mode: str) -> None:
    """Set execution mode in config file."""
    utils.set_execution_mode(mode, CONFIG_FILE)


def load_execution_mode() -> str:
    """Load execution mode from config file."""
    return utils.load_execution_mode(CONFIG_FILE)


def get_paper_wallet_balance() -> float:
    """Get paper wallet balance from multiple sources, prioritizing the most recent."""
    try:
        # Priority 1: Check paper wallet state file (most accurate)
        paper_wallet_state_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
        if paper_wallet_state_file.exists():
            try:
                import numpy as np
                from yaml import Loader, SafeLoader

                # Custom loader to handle numpy scalars
                class NumpyLoader(SafeLoader):
                    pass

                def construct_numpy_scalar(loader, node):
                    """Construct numpy scalar from YAML node."""
                    try:
                        # Get the numpy dtype and binary data
                        if hasattr(node, 'value') and isinstance(node.value, list):
                            dtype_info = node.value[0]
                            binary_data = node.value[1]

                            # Extract binary data
                            if hasattr(binary_data, 'value'):
                                import base64
                                import struct

                                # Decode base64 binary data
                                decoded = base64.b64decode(binary_data.value)
                                # Convert to float64 (little endian)
                                value = struct.unpack('<d', decoded)[0]
                                return float(value)
                    except Exception as e:
                        print(f"Error decoding numpy scalar: {e}")
                        return 0.0

                    return 0.0

                # Add constructor for numpy scalars
                NumpyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar', construct_numpy_scalar)

                with open(paper_wallet_state_file, 'r') as f:
                    state = yaml.load(f, Loader=NumpyLoader) or {}
                    balance = state.get('balance', 0.0)

                    if isinstance(balance, (int, float)) and balance > 0:
                        print(f"Frontend got paper wallet balance from state file: ${balance:.2f}")
                        return float(balance)
            except Exception as e:
                print(f"Error reading paper wallet state file: {e}")

        # Priority 2: Check positions.log for most recent balance
        if POSITIONS_FILE.exists():
            with open(POSITIONS_FILE, 'r') as f:
                lines = f.readlines()

            # Look for the most recent balance entry
            for line in reversed(lines):
                if 'balance $' in line:
                    import re
                    balance_match = re.search(r'balance \$?([0-9.]+)', line)
                    if balance_match:
                        balance = float(balance_match.group(1))
                        print(f"Frontend got paper wallet balance from positions.log: ${balance:.2f}")
                        return balance

        # Priority 3: Check paper_wallet.yaml
        paper_wallet_file = LOG_DIR / 'paper_wallet.yaml'
        if paper_wallet_file.exists():
            try:
                with open(paper_wallet_file) as f:
                    config = yaml.safe_load(f) or {}
                    balance = float(config.get('initial_balance', 10000.0))
                    print(f"Frontend got paper wallet balance from paper_wallet.yaml: ${balance:.2f}")
                    return balance
            except Exception as e:
                print(f"Error reading paper_wallet.yaml: {e}")

        # Priority 4: Check user_config.yaml
        user_config_file = Path('crypto_bot/user_config.yaml')
        if user_config_file.exists():
            try:
                with open(user_config_file) as f:
                    config = yaml.safe_load(f) or {}
                    balance = float(config.get('paper_wallet_balance', 10000.0))
                    print(f"Frontend got paper wallet balance from user_config.yaml: ${balance:.2f}")
                    return balance
            except Exception as e:
                print(f"Error reading user_config.yaml: {e}")

        # Fallback: Default balance
        default_balance = 10000.0
        print(f"Frontend using default paper wallet balance: ${default_balance:.2f}")
        return default_balance

    except Exception as e:
        print(f"Error getting paper wallet balance: {e}")
        return 10000.0


def set_paper_wallet_balance(balance: float) -> None:
    """Set paper wallet balance in multiple locations for consistency."""
    try:
        # Update paper wallet state file (highest priority)
        paper_wallet_state_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
        if paper_wallet_state_file.exists():
            try:
                with open(paper_wallet_state_file, 'r') as f:
                    state = yaml.safe_load(f) or {}
                state['balance'] = balance
                state['initial_balance'] = balance  # Also update initial balance
                with open(paper_wallet_state_file, 'w') as f:
                    yaml.dump(state, f, default_flow_style=False)
                print(f"Frontend updated paper wallet state file: ${balance:.2f}")
            except Exception as e:
                print(f"Frontend failed to update paper wallet state file: {e}")
        else:
            # Create new state file
            state = {
                'balance': balance,
                'initial_balance': balance,
                'realized_pnl': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'positions': {}
            }
            paper_wallet_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(paper_wallet_state_file, 'w') as f:
                yaml.dump(state, f, default_flow_style=False)
            print(f"Frontend created new paper wallet state file: ${balance:.2f}")
        
        # Update paper_wallet.yaml
        paper_wallet_file = LOG_DIR / 'paper_wallet.yaml'
        paper_config = {'initial_balance': balance}
        with open(paper_wallet_file, 'w') as f:
            yaml.dump(paper_config, f, default_flow_style=False)
        print(f"Frontend updated paper_wallet.yaml: ${balance:.2f}")
        
        # Update user_config.yaml
        user_config_file = Path('crypto_bot/user_config.yaml')
        if user_config_file.exists():
            with open(user_config_file) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        config['paper_wallet_balance'] = balance
        with open(user_config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Frontend updated user_config.yaml: ${balance:.2f}")
        
        # Update legacy config if it exists
        legacy_config_path = Path('crypto_bot/paper_wallet_config.yaml')
        if legacy_config_path.exists():
            try:
                with open(legacy_config_path) as f:
                    legacy_config = yaml.safe_load(f) or {}
                legacy_config['initial_balance'] = balance
                with open(legacy_config_path, 'w') as f:
                    yaml.dump(legacy_config, f, default_flow_style=False)
                print(f"Frontend updated legacy config {legacy_config_path}: ${balance:.2f}")
            except Exception as e:
                print(f"Frontend failed to update legacy config {legacy_config_path}: {e}")
        
        print(f"Frontend successfully updated paper wallet balance to: ${balance:.2f}")
        
    except Exception as e:
        print(f"Error setting paper wallet balance: {e}")
        raise


def get_open_positions() -> list:
    """Get open positions from paper wallet state file."""
    try:
        # Try to load from paper wallet state file first (most accurate)
        paper_wallet_state_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
        if paper_wallet_state_file.exists():
            import numpy as np
            from yaml import Loader, SafeLoader

            # Custom loader to handle numpy scalars
            class NumpyLoader(SafeLoader):
                pass

            def construct_numpy_scalar(loader, node):
                """Construct numpy scalar from YAML node."""
                try:
                    # Get the numpy dtype and binary data
                    if hasattr(node, 'value') and isinstance(node.value, list):
                        dtype_info = node.value[0]
                        binary_data = node.value[1]

                        # Extract binary data
                        if hasattr(binary_data, 'value'):
                            import base64
                            import struct

                            # Decode base64 binary data
                            decoded = base64.b64decode(binary_data.value)
                            # Convert to float64 (little endian)
                            value = struct.unpack('<d', decoded)[0]
                            return float(value)
                except Exception as e:
                    print(f"Error decoding numpy scalar: {e}")
                    return 0.0

                return 0.0

            # Add constructor for numpy scalars
            NumpyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar', construct_numpy_scalar)

            with open(paper_wallet_state_file, 'r') as f:
                state = yaml.load(f, Loader=NumpyLoader) or {}

            positions = state.get('positions', {})
            if not positions:
                return []

            open_positions = []
            for pos_id, pos_data in positions.items():
                symbol = pos_data.get('symbol')
                if not symbol:
                    continue

                side = pos_data.get('side', 'buy')

                # Handle numpy scalars for amount/size
                amount = pos_data.get('size', pos_data.get('amount', 0))
                if hasattr(amount, 'item'):  # numpy scalar
                    amount = amount.item()
                elif isinstance(amount, dict) and 'item' in amount:  # numpy scalar serialized
                    try:
                        amount = float(amount.get('item', 0))
                    except:
                        amount = 0

                # Handle numpy scalars for entry_price
                entry_price = pos_data.get('entry_price', 0)
                if hasattr(entry_price, 'item'):  # numpy scalar
                    entry_price = entry_price.item()
                elif isinstance(entry_price, dict) and 'item' in entry_price:  # numpy scalar serialized
                    try:
                        entry_price = float(entry_price.get('item', 0))
                    except:
                        entry_price = 0

                # Convert to float
                amount = float(amount) if amount else 0.0
                entry_price = float(entry_price) if entry_price else 0.0

                if amount <= 0 or entry_price <= 0:
                    continue

                # Get current price
                current_price = get_current_price_for_symbol(symbol)
                if current_price <= 0:
                    # Use entry price if no current price available
                    current_price = entry_price

                # Calculate PnL
                if side == 'buy':
                    pnl = (current_price - entry_price) * amount
                else:
                    pnl = (entry_price - current_price) * amount

                open_positions.append({
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_percentage': (pnl / (entry_price * amount)) * 100 if entry_price > 0 else 0
                })

            return open_positions

        # Fallback to parsing positions.log (legacy method)
        return get_open_positions_from_log()

    except Exception as e:
        print(f"Error getting open positions: {e}")
        return []

def get_open_positions_from_log() -> list:
    """Parse open positions from positions.log file (legacy method)."""
    import re
    from datetime import datetime, timedelta
    
    if not POSITIONS_FILE.exists():
        return []
    
    positions = []
    # Updated regex pattern to handle more position formats
    pos_patterns = [
        # Pattern 1: Standard format with pnl calculation
        re.compile(
            r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
            r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
            r"pnl \$?(?P<pnl>[0-9.+-]+).*balance \$?(?P<balance>[0-9.]+)"
        ),
        # Pattern 2: Format without pnl calculation
        re.compile(
            r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
            r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+)"
        ),
        # Pattern 3: Alternative format
        re.compile(
            r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
            r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
            r"pnl \$?(?P<pnl>[0-9.+-]+)"
        )
    ]
    
    try:
        with open(POSITIONS_FILE) as f:
            lines = f.readlines()
            
        # Only process recent lines (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_positions = []
        
        for line in lines:
            # Extract timestamp from the beginning of the line
            timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
            if timestamp_match:
                try:
                    timestamp_str = timestamp_match.group(1)
                    line_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    
                    # Only include positions from the last 24 hours
                    if line_timestamp >= cutoff_time:
                        # Try each pattern
                        position_data = None
                        for pattern in pos_patterns:
                            match = pattern.search(line)
                            if match:
                                # Check if this is a real position (not just a balance update)
                                symbol = match.group('symbol')
                                side = match.group('side')
                                amount = float(match.group('amount'))
                                
                                # Filter out positions with zero amounts or very small amounts
                                if amount > 0.0001:  # Minimum threshold
                                    entry_price = float(match.group('entry'))

                                    # Get LIVE current price instead of cached price
                                    current_price = get_current_price_for_symbol(symbol)
                                    if current_price <= 0:
                                        # For unknown tokens, use entry price to show 0 PnL
                                        # This is better than using stale cached prices
                                        print(f"No live price available for {symbol}, using entry price for 0 PnL")
                                        current_price = entry_price

                                    # Calculate PnL if not provided
                                    if 'pnl' in match.groupdict() and match.group('pnl'):
                                        pnl = float(match.group('pnl'))
                                    else:
                                        # Calculate PnL manually
                                        if side == 'buy':
                                            pnl = (current_price - entry_price) * amount
                                        else:  # sell/short
                                            pnl = (entry_price - current_price) * amount
                                    
                                    # Get balance if available
                                    balance = 0.0
                                    if 'balance' in match.groupdict() and match.group('balance'):
                                        balance = float(match.group('balance'))
                                    
                                    position_data = {
                                        'symbol': symbol,
                                        'side': side,
                                        'amount': amount,
                                        'entry_price': entry_price,
                                        'current_price': current_price,
                                        'pnl': pnl,
                                        'balance': balance,
                                        'timestamp': timestamp_str
                                    }
                                    break
                        
                        if position_data:
                            recent_positions.append(position_data)
                            
                except ValueError as e:
                    print(f"Error parsing timestamp in line: {line.strip()}, error: {e}")
                    continue
        
        # Remove duplicates based on symbol and side, keeping the most recent
        seen = set()
        unique_positions = []
        for pos in reversed(recent_positions):  # Process in reverse to keep most recent
            key = f"{pos['symbol']}_{pos['side']}"
            if key not in seen:
                seen.add(key)
                unique_positions.append(pos)
        
        # Return positions in chronological order
        return list(reversed(unique_positions))
        
    except Exception as e:
        print(f"Error parsing positions from log: {e}")
        return []
    
    return []


def clear_old_positions() -> None:
    """Clear old position entries from the positions.log file."""
    if not POSITIONS_FILE.exists():
        return
    
    try:
        import re
        from datetime import datetime, timedelta
        
        # Read all lines
        with open(POSITIONS_FILE, 'r') as f:
            lines = f.readlines()
        
        # Keep only lines from the last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_lines = []
        
        for line in lines:
            # Extract timestamp from the beginning of the line
            timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
            if timestamp_match:
                try:
                    timestamp_str = timestamp_match.group(1)
                    line_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    
                    # Keep lines from the last 24 hours
                    if line_timestamp >= cutoff_time:
                        recent_lines.append(line)
                except ValueError:
                    # Keep lines that don't have valid timestamps (they might be important)
                    recent_lines.append(line)
            else:
                # Keep lines without timestamps
                recent_lines.append(line)
        
        # Write back the filtered lines
        with open(POSITIONS_FILE, 'w') as f:
            f.writelines(recent_lines)
            
    except Exception as e:
        print(f"Error clearing old positions: {e}")


def get_uptime() -> str:
    """Return human readable uptime."""
    return utils.get_uptime(bot_start_time)


def calculate_wallet_pnl() -> Dict[str, float]:
    """Calculate current wallet PnL based on paper wallet state, open positions, and trade history."""
    try:
        # Try to load paper wallet state first
        paper_wallet_state_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
        if paper_wallet_state_file.exists():
            try:
                import numpy as np
                from yaml import Loader, SafeLoader

                # Custom loader to handle numpy scalars
                class NumpyLoader(SafeLoader):
                    pass

                def construct_numpy_scalar(loader, node):
                    """Construct numpy scalar from YAML node."""
                    try:
                        # Get the numpy dtype and binary data
                        if hasattr(node, 'value') and isinstance(node.value, list):
                            dtype_info = node.value[0]
                            binary_data = node.value[1]

                            # Extract binary data
                            if hasattr(binary_data, 'value'):
                                import base64
                                import struct

                                # Decode base64 binary data
                                decoded = base64.b64decode(binary_data.value)
                                # Convert to float64 (little endian)
                                value = struct.unpack('<d', decoded)[0]
                                return float(value)
                    except Exception as e:
                        print(f"Error decoding numpy scalar: {e}")
                        return 0.0

                    return 0.0

                # Add constructor for numpy scalars
                NumpyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar', construct_numpy_scalar)

                with open(paper_wallet_state_file, 'r') as f:
                    state = yaml.load(f, Loader=NumpyLoader) or {}
                    current_balance = state.get('balance', 0.0)
                    initial_balance = state.get('initial_balance', 10000.0)
                    realized_pnl = state.get('realized_pnl', 0.0)
                    logger.info(f"Loaded paper wallet state: balance=${current_balance:.2f}, realized_pnl=${realized_pnl:.2f}")
            except Exception as e:
                logger.error(f"Error reading paper wallet state: {e}")
                current_balance = get_paper_wallet_balance()
                initial_balance = current_balance
                realized_pnl = 0.0
        else:
            # Fallback to reading from positions.log
            current_balance = get_paper_wallet_balance()
            initial_balance = current_balance
            realized_pnl = 0.0

        # Calculate realized P&L from trade history if paper wallet state doesn't have it
        if realized_pnl == 0.0:
            try:
                from crypto_bot import log_reader
                df = log_reader._read_trades(TRADE_FILE)

                if not df.empty:
                    # Track position history for realized P&L calculation
                    position_history = {}
                    calculated_realized_pnl = 0.0

                    for _, row in df.iterrows():
                        symbol = str(row.get('symbol', ''))
                        side = str(row.get('side', ''))
                        amount = float(row.get('amount', 0))
                        price = float(row.get('price', 0))

                        if symbol and amount > 0 and price > 0:
                            # Calculate trade total
                            total = amount * price

                            # Check if this trade closes an existing position
                            if symbol in position_history:
                                existing_pos = position_history[symbol]

                                # Check if this is a closing trade (opposite side)
                                if ((side == 'sell' and existing_pos['side'] == 'buy') or
                                    (side == 'buy' and existing_pos['side'] == 'sell')):

                                    # Calculate realized PnL
                                    if side == 'sell':  # Closing long position
                                        pnl = (price - existing_pos['price']) * min(amount, existing_pos['amount'])
                                    else:  # Closing short position
                                        pnl = (existing_pos['price'] - price) * min(amount, existing_pos['amount'])

                                    calculated_realized_pnl += pnl

                                    # Update or remove position
                                    if amount >= existing_pos['amount']:
                                        del position_history[symbol]
                                    else:
                                        position_history[symbol]['amount'] -= amount
                                else:
                                    # Same side trade - average the position
                                    if symbol in position_history:
                                        total_cost = (existing_pos['price'] * existing_pos['amount']) + total
                                        total_amount = existing_pos['amount'] + amount
                                        position_history[symbol] = {
                                            'side': side,
                                            'price': total_cost / total_amount,
                                            'amount': total_amount
                                        }
                                    else:
                                        position_history[symbol] = {'side': side, 'price': price, 'amount': amount}
                            else:
                                # New position
                                position_history[symbol] = {'side': side, 'price': price, 'amount': amount}

                    realized_pnl = calculated_realized_pnl
                    print(f"Calculated realized P&L from trade history: ${realized_pnl:.2f}")

            except Exception as e:
                print(f"Error calculating realized P&L from trade history: {e}")

        # Get open positions
        open_positions = get_open_positions()

        # Calculate unrealized PnL from open positions
        unrealized_pnl = 0.0
        position_details = []

        for position in open_positions:
            symbol = position['symbol']
            side = position['side']
            amount = position['amount']
            entry_price = position['entry_price']
            current_price = position['current_price']

            # Calculate position PnL
            if side == 'buy':  # Long position
                position_pnl = (current_price - entry_price) * amount
            else:  # Short position
                position_pnl = (entry_price - current_price) * amount

            unrealized_pnl += position_pnl

            position_details.append({
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl': position_pnl,
                'pnl_percentage': (position_pnl / (entry_price * amount)) * 100 if entry_price > 0 else 0
            })

        # Calculate total PnL (realized + unrealized)
        total_pnl = realized_pnl + unrealized_pnl

        # Update current balance to include unrealized PnL
        total_balance = current_balance + unrealized_pnl

        return {
            'initial_balance': initial_balance,
            'current_balance': total_balance,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'pnl_percentage': (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0,
            'open_positions': position_details,
            'position_count': len(open_positions)
        }

    except Exception as e:
        print(f"Error calculating wallet PnL: {e}")
        return {
            'initial_balance': 0.0,
            'current_balance': 0.0,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'total_pnl': 0.0,
            'pnl_percentage': 0.0,
            'open_positions': [],
            'position_count': 0,
            'error': str(e)
        }


@app.route('/api/wallet-pnl')
def api_wallet_pnl():
    """Return current wallet PnL calculation."""
    try:
        pnl_data = calculate_wallet_pnl()
        return jsonify(pnl_data)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/')
def index():
    mode = load_execution_mode()
    
    # Get performance data
    df = log_reader._read_trades(TRADE_FILE)
    perf = utils.compute_performance(df)
    
    # Get dynamic allocation data based on actual performance
    allocation = utils.calculate_dynamic_allocation()
    
    # Fallback to static config if no dynamic data available
    if not allocation and CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
            allocation = cfg.get('strategy_allocation', {})
    
    # Final fallback to weights.json if no allocation in config
    if not allocation and (LOG_DIR / 'weights.json').exists():
        with open(LOG_DIR / 'weights.json') as f:
            weights_data = json.load(f)
            # Convert decimal weights to percentages for consistency
            allocation = {strategy: weight * 100 for strategy, weight in weights_data.items()}
    
    # Get paper wallet balance (always show wallet balance)
    paper_wallet_balance = get_paper_wallet_balance()
    
    # Get open positions
    open_positions = get_open_positions()
    
    # Calculate total PnL including both realized and unrealized
    pnl_data = calculate_wallet_pnl()
    total_pnl = pnl_data.get('total_pnl', 0.0)
    initial_balance = pnl_data.get('initial_balance', 10000.0)

    return render_template(
        'index.html',
        running=is_running(),
        mode=mode,
        uptime=get_uptime(),
        last_trade=utils.get_last_trade(TRADE_FILE),
        regime=utils.get_current_regime(LOG_FILE),
        last_reason=utils.get_last_decision_reason(LOG_FILE),
        pnl=total_pnl,
        performance=perf,
        allocation=allocation,
        paper_wallet_balance=paper_wallet_balance,
        initial_balance=initial_balance,
        open_positions=open_positions,
        pnl_data=pnl_data,  # Pass the full PnL data for JavaScript
    )




@app.route('/start', methods=['POST'])
def start():
    global bot_proc, bot_start_time
    mode = request.form.get('mode', 'dry_run')
    set_execution_mode(mode)
    if not is_running() and not check_existing_bot():
        # Launch the asyncio-based trading bot using the non-interactive script
        venv_python = Path(__file__).parent.parent / 'venv' / 'bin' / 'python3'
        bot_script = Path(__file__).parent.parent / 'start_bot_noninteractive.py'
        bot_proc = subprocess.Popen([str(venv_python), str(bot_script)])
        bot_start_time = time.time()
    return redirect(url_for('index'))


@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    mode = (
        request.json.get('mode', 'dry_run') if request.is_json else request.form.get('mode', 'dry_run')
    )
    print(f"Starting bot with mode: {mode}")
    set_execution_mode(mode)
    
    # Check if we have a tracked subprocess running
    if utils.is_running(bot_proc):
        print("Bot subprocess is already running")
        return jsonify({
            'status': 'already_running',
            'running': True,
            'uptime': get_uptime(),
            'mode': mode,
        })

    # Check if there's another bot process running
    if check_existing_bot():
        print("Another bot process detected, sending start command")
        try:
            from crypto_bot.utils.logger import LOG_DIR
            control_file = LOG_DIR / "bot_control.json"
            with open(control_file, 'w') as f:
                json.dump({"command": "start"}, f)

            # Set start time if not already set (for existing processes)
            global bot_start_time
            if bot_start_time is None:
                bot_start_time = time.time()

            return jsonify({
                'status': 'started',
                'running': True,
                'uptime': get_uptime(),
                'mode': mode,
            })
        except Exception as e:
            print(f"Error sending start command: {e}")
            return jsonify({
                'status': f'error: {e}',
                'running': False,
                'uptime': get_uptime(),
                'mode': mode,
            })
    
    # Start new bot process
    print("Starting new bot process")
    try:
        venv_python = Path(__file__).parent.parent / 'venv' / 'bin' / 'python3'
        bot_script = Path(__file__).parent.parent / 'start_bot_noninteractive.py'

        print(f"Using Python: {venv_python}")
        print(f"Using script: {bot_script}")

        if not venv_python.exists():
            print(f"Python executable not found: {venv_python}")
            return jsonify({
                'status': 'error: Python executable not found',
                'running': False,
                'uptime': get_uptime(),
                'mode': mode,
            })

        if not bot_script.exists():
            print(f"Bot script not found: {bot_script}")
            return jsonify({
                'status': 'error: Bot script not found',
                'running': False,
                'uptime': get_uptime(),
                'mode': mode,
            })

        # Pass environment variables to subprocess
        env = os.environ.copy()
        bot_proc = subprocess.Popen([str(venv_python), str(bot_script)], env=env)
        bot_start_time = time.time()

        # Wait a moment to see if the process starts successfully
        time.sleep(1)

        if bot_proc.poll() is None:
            print("Bot process started successfully")
            return jsonify({
                'status': 'started',
                'running': True,
                'uptime': get_uptime(),
                'mode': mode,
            })
        else:
            print(f"Bot process failed to start, return code: {bot_proc.returncode}")
            return jsonify({
                'status': f'error: Bot process failed to start (return code: {bot_proc.returncode})',
                'running': False,
                'uptime': get_uptime(),
                'mode': mode,
            })
            
    except Exception as e:
        print(f"Error starting bot: {e}")
        return jsonify({
            'status': f'error: {e}',
            'running': False,
            'uptime': get_uptime(),
            'mode': mode,
        })


@app.route('/stop')
def stop():
    global bot_proc, bot_start_time
    if is_running():
        bot_proc.terminate()
        bot_proc.wait()
    bot_proc = None
    bot_start_time = None
    return redirect(url_for('index'))


@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    status = 'not_running'
    
    # Send stop command to running bot if it exists
    if check_existing_bot():
        try:
            from crypto_bot.utils.logger import LOG_DIR
            control_file = LOG_DIR / "bot_control.json"
            with open(control_file, 'w') as f:
                json.dump({"command": "stop"}, f)
            status = 'stopped'
        except Exception as e:
            status = f'error: {e}'
    elif is_running():
        bot_proc.terminate()
        bot_proc.wait()
        status = 'stopped'
        bot_proc = None
        bot_start_time = None
    
    return jsonify({
        'status': status,
        'running': False,
        'uptime': get_uptime(),
        'mode': load_execution_mode(),
    })


@app.route('/pause_bot', methods=['POST'])
def pause_bot():
    """Pause the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    status = 'not_running'
    if is_running():
        # Send SIGSTOP to pause the process
        bot_proc.send_signal(signal.SIGSTOP)
        status = 'paused'
    return jsonify({
        'status': status,
        'running': False,
        'uptime': get_uptime(),
        'mode': load_execution_mode(),
    })


@app.route('/resume_bot', methods=['POST'])
def resume_bot():
    """Resume the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    status = 'not_running'
    if bot_proc and bot_proc.poll() is None:
        # Send SIGCONT to resume the process
        bot_proc.send_signal(signal.SIGCONT)
        status = 'resumed'
    return jsonify({
        'status': status,
        'running': True,
        'uptime': get_uptime(),
        'mode': load_execution_mode(),
    })


@app.route('/logs')
def logs_page():
    return render_template('logs.html')


@app.route('/logs_tail')
def logs_tail():
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()[-200:]
        return '\n'.join(lines)
    return ''


@app.route('/stats')
def stats():
    data = {}
    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            data = json.load(f)
    return render_template('stats.html', stats=data)


@app.route('/scans')
def scans():
    data = {}
    if SCAN_FILE.exists():
        with open(SCAN_FILE) as f:
            data = json.load(f)
    return render_template('scans.html', scans=data)


@app.route('/cli', methods=['GET', 'POST'])
def cli():
    """Run CLI commands and display output."""
    output = None
    if request.method == 'POST':
        base = request.form.get('base', 'bot')
        cmd_args = request.form.get('command', '')
        venv_python = Path(__file__).parent.parent / 'venv' / 'bin' / 'python3'
        if base == 'backtest':
            cmd = f"{venv_python} -m crypto_bot.backtest.backtest_runner {cmd_args}"
        elif base == 'custom':
            cmd = cmd_args
        else:
            cmd = f"{venv_python} start_bot_noninteractive.py {cmd_args}"
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=False
            )
            output = proc.stdout + proc.stderr
        except Exception as exc:  # pragma: no cover - subprocess
            output = str(exc)
    return render_template('cli.html', output=output)
@app.route('/dashboard')
def dashboard():
    summary = log_reader.trade_summary(TRADE_FILE)
    df = log_reader._read_trades(TRADE_FILE)
    perf = utils.compute_performance(df)
    
    # Get dynamic allocation data based on actual performance
    allocation = utils.calculate_dynamic_allocation()
    
    # Fallback to static config if no dynamic data available
    if not allocation and CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
            allocation = cfg.get('strategy_allocation', {})
    
    # Final fallback to weights.json if no allocation in config
    if not allocation and (LOG_DIR / 'weights.json').exists():
        with open(LOG_DIR / 'weights.json') as f:
            weights_data = json.load(f)
            # Convert decimal weights to percentages for consistency
            allocation = {strategy: weight * 100 for strategy, weight in weights_data.items()}
    
    regimes = []
    if REGIME_FILE.exists():
        regimes = REGIME_FILE.read_text().splitlines()[-20:]
    
    # Use the new performance structure
    pnl = perf.get('total_pnl', 0.0)
    
    return render_template(
        'dashboard.html',
        pnl=pnl,
        performance=perf,
        allocation=allocation,
        regimes=regimes,
    )
@app.route('/model')
def model_page():
    report = {}
    if MODEL_REPORT.exists():
        with open(MODEL_REPORT) as f:
            report = json.load(f)
    return render_template('model.html', report=report)


@app.route('/train_model', methods=['POST'])
def train_model_route():
    file = request.files.get('csv')
    if file:
        tmp_path = LOG_DIR / 'upload.csv'
        file.save(tmp_path)
        ml.train_from_csv(tmp_path)
        tmp_path.unlink()
    return redirect(url_for('model_page'))


@app.route('/validate_model', methods=['POST'])
def validate_model_route():
    file = request.files.get('csv')
    tmp_path = None
    if file:
        tmp_path = LOG_DIR / 'validate.csv'
        file.save(tmp_path)
        metrics = ml.validate_from_csv(tmp_path)
        tmp_path.unlink()
    else:
        default_csv = LOG_DIR / 'trades.csv'
        if default_csv.exists():
            metrics = ml.validate_from_csv(default_csv)
        else:
            metrics = ml.validate_from_csv(default_csv)
    if metrics:
        MODEL_REPORT.write_text(json.dumps(metrics))
    return redirect(url_for('model_page'))


@app.route('/api_config')
def api_config_page():
    """API configuration page."""
    # Load current API configuration
    api_config = {}
    user_config_file = Path('crypto_bot/user_config.yaml')
    if user_config_file.exists():
        with open(user_config_file) as f:
            api_config = yaml.safe_load(f) or {}
    
    return render_template('api_config.html', api_config=api_config)


@app.route('/config_settings')
def config_settings_page():
    """General configuration settings page."""
    # Load current configuration
    config_data = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config_data = yaml.safe_load(f) or {}
    
    return render_template('config_settings.html', config_data=config_data)


@app.route('/api/save_api_config', methods=['POST'])
def save_api_config():
    """Save API configuration."""
    try:
        data = request.get_json()
        user_config_file = Path('crypto_bot/user_config.yaml')
        
        # Load existing config
        current_config = {}
        if user_config_file.exists():
            with open(user_config_file) as f:
                current_config = yaml.safe_load(f) or {}
        
        # Update with new values
        current_config.update(data)
        
        # Save back to file
        with open(user_config_file, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
        
        return jsonify({'status': 'success', 'message': 'API configuration saved successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error saving configuration: {str(e)}'}), 500


@app.route('/api/save_config_settings', methods=['POST'])
def save_config_settings():
    """Save general configuration settings."""
    try:
        data = request.get_json()
        
        # Load existing config
        current_config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                current_config = yaml.safe_load(f) or {}
        
        # Update with new values (merge nested structures)
        def deep_merge(d1, d2):
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value
            return d1
        
        updated_config = deep_merge(current_config, data)
        
        # Save back to file
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False)
        
        return jsonify({'status': 'success', 'message': 'Configuration saved successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error saving configuration: {str(e)}'}), 500


@app.route('/api/refresh_config', methods=['POST'])
def refresh_config():
    """Refresh configuration by reloading from files."""
    try:
        # Send reload command to running bot if it exists
        if check_existing_bot():
            from crypto_bot.utils.logger import LOG_DIR
            control_file = LOG_DIR / "bot_control.json"
            with open(control_file, 'w') as f:
                json.dump({"command": "reload"}, f)
            return jsonify({'status': 'success', 'message': 'Reload command sent to bot'})
        else:
            # Bot not running, just return success
            return jsonify({'status': 'success', 'message': 'Configuration refreshed successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error refreshing configuration: {str(e)}'}), 500


@app.route('/trades')
def trades_page():
    return render_template('trades.html')


@app.route('/trades_tail')
def trades_tail():
    trades = ''
    if TRADE_FILE.exists():
        trades = '\n'.join(TRADE_FILE.read_text().splitlines()[-100:])
    errors = ''
    if ERROR_FILE.exists():
        errors = '\n'.join(ERROR_FILE.read_text().splitlines()[-100:])
    return jsonify({'trades': trades, 'errors': errors})


@app.route('/api/current-prices')
def api_current_prices():
    """Return current market prices for symbols."""
    try:
        # Read trades to get unique symbols
        df = log_reader._read_trades(TRADE_FILE)
        if df.empty:
            return jsonify({})
        
        symbols = df['symbol'].unique().tolist()
        current_prices = {}
        
        # For now, we'll use a simple approach to get current prices
        # In a real implementation, you'd fetch from exchange APIs
        for symbol in symbols:
            try:
                # Try to get price from various sources
                price = get_current_price_for_symbol(symbol)
                if price > 0:
                    current_prices[symbol] = price
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                continue
        
        return jsonify(current_prices)
    except Exception as e:
        return jsonify({'error': str(e)})


# Price caching system
_price_cache = {}
_CACHE_TTL = 60  # Cache prices for 60 seconds

# Price source health monitoring
_price_source_health = {}
_HEALTH_CHECK_WINDOW = 10  # Track last 10 attempts per source
_HEALTH_DECAY_FACTOR = 0.9  # Decay factor for old health data

# Manual price overrides
_manual_prices = {}
_MANUAL_PRICES_FILE = LOG_DIR / 'manual_prices.json'

def _update_price_source_health(source_name: str, success: bool) -> None:
    """Update health status for a price source."""
    if source_name not in _price_source_health:
        _price_source_health[source_name] = {
            'successes': 0,
            'failures': 0,
            'last_success': None,
            'last_failure': None,
            'consecutive_failures': 0
        }

    health = _price_source_health[source_name]

    if success:
        health['successes'] += 1
        health['last_success'] = time.time()
        health['consecutive_failures'] = 0
    else:
        health['failures'] += 1
        health['last_failure'] = time.time()
        health['consecutive_failures'] += 1

    # Apply decay to prevent old data from dominating
    health['successes'] = int(health['successes'] * _HEALTH_DECAY_FACTOR)
    health['failures'] = int(health['failures'] * _HEALTH_DECAY_FACTOR)

def _get_price_source_health(source_name: str) -> dict:
    """Get health metrics for a price source."""
    if source_name not in _price_source_health:
        return {'health_score': 0.5, 'total_attempts': 0, 'success_rate': 0.0}

    health = _price_source_health[source_name]
    total_attempts = health['successes'] + health['failures']

    if total_attempts == 0:
        return {'health_score': 0.5, 'total_attempts': 0, 'success_rate': 0.0}

    success_rate = health['successes'] / total_attempts

    # Calculate health score (0-1, higher is better)
    # Penalize consecutive failures and recent failures
    health_score = success_rate

    if health['consecutive_failures'] > 0:
        health_score *= (0.8 ** health['consecutive_failures'])

    # Boost score if recently successful
    if health['last_success'] and health['last_failure']:
        if health['last_success'] > health['last_failure']:
            health_score *= 1.2  # Recent success bonus
        else:
            health_score *= 0.8  # Recent failure penalty

    health_score = min(1.0, max(0.0, health_score))

    return {
        'health_score': health_score,
        'total_attempts': total_attempts,
        'success_rate': success_rate,
        'consecutive_failures': health['consecutive_failures']
    }

def _get_best_price_sources() -> list:
    """Get price sources sorted by health score (best first)."""
    sources = []
    for source_name in _price_source_health.keys():
        health = _get_price_source_health(source_name)
        sources.append((source_name, health['health_score']))

    # Sort by health score (highest first)
    sources.sort(key=lambda x: x[1], reverse=True)
    return [source[0] for source in sources]


def _load_manual_prices():
    """Load manual price overrides from file."""
    global _manual_prices
    try:
        if _MANUAL_PRICES_FILE.exists():
            with open(_MANUAL_PRICES_FILE, 'r') as f:
                data = json.load(f)
                # Filter out expired manual prices (older than 24 hours)
                current_time = time.time()
                _manual_prices = {
                    symbol: price_data
                    for symbol, price_data in data.items()
                    if current_time - price_data.get('timestamp', 0) < 86400  # 24 hours
                }
    except Exception as e:
        print(f"Error loading manual prices: {e}")
        _manual_prices = {}


def _save_manual_prices():
    """Save manual price overrides to file."""
    try:
        with open(_MANUAL_PRICES_FILE, 'w') as f:
            json.dump(_manual_prices, f, indent=2)
    except Exception as e:
        print(f"Error saving manual prices: {e}")


def _get_manual_price(symbol: str) -> Optional[float]:
    """Get manual price override for a symbol."""
    if symbol in _manual_prices:
        price_data = _manual_prices[symbol]
        current_time = time.time()

        # Check if manual price is still valid (not expired)
        if current_time - price_data.get('timestamp', 0) < price_data.get('validity_hours', 24) * 3600:
            return price_data['price']
        else:
            # Remove expired manual price
            del _manual_prices[symbol]
            _save_manual_prices()

    return None


def _set_manual_price(symbol: str, price: float, validity_hours: int = 24):
    """Set manual price override for a symbol."""
    _manual_prices[symbol] = {
        'price': price,
        'timestamp': time.time(),
        'validity_hours': validity_hours
    }
    _save_manual_prices()


# Load manual prices on startup
_load_manual_prices()

def get_current_price_for_symbol(symbol: str) -> float:
    """Get current price for a symbol using available price sources with caching."""
    if not symbol or symbol.strip() == '':
        return 0.0

    symbol = symbol.strip().upper()

    # Check cache first
    now = time.time()
    if symbol in _price_cache:
        cached_price, timestamp = _price_cache[symbol]
        if now - timestamp < _CACHE_TTL:
            print(f"Using cached price for {symbol}: ${cached_price}")
            return cached_price
        else:
            # Cache expired, remove it
            del _price_cache[symbol]
    
    try:
        # Check for manual price override first
        manual_price = _get_manual_price(symbol)
        if manual_price and manual_price > 0:
            print(f"Using manual price override for {symbol}: ${manual_price}")
            _price_cache[symbol] = (manual_price, time.time())
            return manual_price

        # Try Pyth network first for Solana tokens
        try:
            from crypto_bot.utils.pyth import get_pyth_price
            price = get_pyth_price(symbol)
            if price and price > 0:
                print(f"Got price for {symbol} from Pyth: ${price}")
                _price_cache[symbol] = (price, time.time())  # Cache the price
                _update_price_source_health('Pyth', True)
                return price
            else:
                _update_price_source_health('Pyth', False)
        except Exception as e:
            print(f"Pyth price fetch failed for {symbol}: {e}")
            _update_price_source_health('Pyth', False)
            pass
        
        # Try Helius API for Solana tokens (enhanced price data)
        try:
            import requests

            # Clean symbol for Helius API
            base = symbol.split('/')[0] if '/' in symbol else symbol
            # Remove common prefixes/suffixes
            base = base.replace('USDT', '').replace('USDC', '').replace('USD', '')

            # Helius token address mapping for common tokens
            helius_token_mapping = {
                'SOL': 'So11111111111111111111111111111111111111112',
                'BTC': '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E',  # Wrapped BTC
                'ETH': '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs',  # Wrapped ETH
                'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
                'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB'
            }

            if base in helius_token_mapping:
                token_address = helius_token_mapping[base]
                # Try Helius enhanced price API
                response = requests.get(
                    f"https://mainnet.helius-rpc.com/?api-key=demo",  # Using demo key for public access
                    json={
                        "jsonrpc": "2.0",
                        "id": "1",
                        "method": "getAsset",
                        "params": {
                            "id": token_address,
                            "displayOptions": {
                                "showFungible": True
                            }
                        }
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data and data['result']:
                        price = float(data['result'].get('price', {}).get('usd', 0.0))
                        if price > 0:
                            print(f"Got price for {symbol} from Helius: ${price}")
                            _price_cache[symbol] = (price, time.time())
                            _update_price_source_health('Helius', True)
                            return price
                        else:
                            _update_price_source_health('Helius', False)
        except Exception as e:
            print(f"Helius price fetch failed for {symbol}: {e}")
            _update_price_source_health('Helius', False)
            pass

        # Try Raydium API for Solana DEX prices
        try:
            import requests

            # Clean symbol for Raydium API
            base = symbol.split('/')[0] if '/' in symbol else symbol
            base = base.replace('USDT', '').replace('USDC', '').replace('USD', '')

            # Raydium token mint mapping
            raydium_token_mapping = {
                'SOL': 'So11111111111111111111111111111111111111112',
                'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
                'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
                'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R'
            }

            if base in raydium_token_mapping:
                # Try Raydium price API
                response = requests.get(
                    "https://api.raydium.io/v2/sdk/price",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if base in data:
                        price = float(data[base])
                        if price > 0:
                            print(f"Got price for {symbol} from Raydium: ${price}")
                            _price_cache[symbol] = (price, time.time())
                            _update_price_source_health('Raydium', True)
                            return price
                        else:
                            _update_price_source_health('Raydium', False)
        except Exception as e:
            print(f"Raydium price fetch failed for {symbol}: {e}")
            _update_price_source_health('Raydium', False)
            pass

        # Try Orca API for Solana DEX prices
        try:
            import requests

            # Clean symbol for Orca API
            base = symbol.split('/')[0] if '/' in symbol else symbol
            base = base.replace('USDT', '').replace('USDC', '').replace('USD', '')

            # Try Orca Whirlpool API
            response = requests.get(
                f"https://www.orca.so/api/v1/token/{base.lower()}/info",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                price = float(data.get('price', 0.0))
                if price > 0:
                    print(f"Got price for {symbol} from Orca: ${price}")
                    _price_cache[symbol] = (price, time.time())
                    _update_price_source_health('Orca', True)
                    return price
                else:
                    _update_price_source_health('Orca', False)
        except Exception as e:
            print(f"Orca price fetch failed for {symbol}: {e}")
            _update_price_source_health('Orca', False)
            pass

        # Try Jupiter API for Solana tokens (legacy fallback)
        try:
            import requests

            # Clean symbol for Jupiter API
            base = symbol.split('/')[0] if '/' in symbol else symbol
            # Remove common prefixes/suffixes
            base = base.replace('USDT', '').replace('USDC', '').replace('USD', '')

            if base and len(base) > 0:
                response = requests.get(
                    "https://price.jup.ag/v4/price",
                    params={"ids[]": base},
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    price = float(data.get("data", {}).get(base, {}).get("price", 0.0))
                    if price > 0:
                        print(f"Got price for {symbol} from Jupiter: ${price}")
                        _price_cache[symbol] = (price, time.time())
                        _update_price_source_health('Jupiter', True)
                        return price
                    else:
                        _update_price_source_health('Jupiter', False)
        except Exception as e:
            # Jupiter API appears to be unavailable/deprecated, skip silently
            # This avoids repeated DNS error messages in the terminal
            _update_price_source_health('Jupiter', False)
            pass
        
        # Try Kraken API for major pairs
        try:
            import requests

            # Convert symbol format for Kraken
            kraken_symbol = symbol.replace('/', '')

            # Map common symbol formats to Kraken format (with proper Kraken pair names)
            kraken_mapping = {
                'BTCUSDT': 'XXBTZUSD',
                'BTCUSD': 'XXBTZUSD',
                'ETHUSDT': 'XETHZUSD',
                'ETHUSD': 'XETHZUSD',
                'SOLUSDT': 'SOLUSD',
                'SOLUSD': 'SOLUSD',
                'ADAUSDT': 'ADAUSD',
                'ADAUSD': 'ADAUSD',
                'DOTUSDT': 'DOTUSD',
                'DOTUSD': 'DOTUSD',
                'LINKUSDT': 'LINKUSD',
                'LINKUSD': 'LINKUSD',
                'UNIUSDT': 'UNIUSD',
                'UNIUSD': 'UNIUSD',
                'AAVEUSDT': 'AAVEUSD',
                'AAVEUSD': 'AAVEUSD',
                'SUSHIUSDT': 'SUSHIUSD',
                'SUSHIUSD': 'SUSHIUSD',
                'COMPUSDT': 'COMPUSD',
                'COMPUSD': 'COMPUSD',
                'MKRUSDT': 'MKRUSD',
                'MKRUSD': 'MKRUSD',
                'YFIUSDT': 'YFIUSD',
                'YFIUSD': 'YFIUSD',
                'HBARUSDT': 'HBARUSD',
                'HBARUSD': 'HBARUSD'
            }

            kraken_pair = kraken_mapping.get(kraken_symbol)
            if kraken_pair:
                response = requests.get(
                    f"https://api.kraken.com/0/public/Ticker?pair={kraken_pair}",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data and kraken_pair in data['result']:
                        price = float(data['result'][kraken_pair]['c'][0])  # Current price
                        if price > 0:
                            print(f"Got price for {symbol} from Kraken: ${price}")
                            _price_cache[symbol] = (price, time.time())
                            _update_price_source_health('Kraken', True)
                            return price
                        else:
                            _update_price_source_health('Kraken', False)
        except Exception as e:
            print(f"Kraken price fetch failed for {symbol}: {e}")
            _update_price_source_health('Kraken', False)
            pass

        # Try Binance API as additional exchange fallback
        try:
            import requests

            # Convert symbol format for Binance
            binance_symbol = symbol.replace('/', '')

            # Map common symbol formats to Binance format
            binance_mapping = {
                'BTCUSDT': 'BTCUSDT',
                'BTCUSD': 'BTCUSDT',  # Binance uses USDT pairs
                'ETHUSDT': 'ETHUSDT',
                'ETHUSD': 'ETHUSDT',
                'SOLUSDT': 'SOLUSDT',
                'SOLUSD': 'SOLUSDT',
                'ADAUSDT': 'ADAUSDT',
                'ADAUSD': 'ADAUSDT',
                'DOTUSDT': 'DOTUSDT',
                'DOTUSD': 'DOTUSDT',
                'LINKUSDT': 'LINKUSDT',
                'LINKUSD': 'LINKUSDT',
                'UNIUSDT': 'UNIUSDT',
                'UNIUSD': 'UNIUSDT',
                'HBARUSDT': 'HBARUSDT',
                'HBARUSD': 'HBARUSDT'
            }

            binance_pair = binance_mapping.get(binance_symbol)
            if binance_pair:
                response = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={binance_pair}",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    price = float(data.get('price', 0.0))
                    if price > 0:
                        print(f"Got price for {symbol} from Binance: ${price}")
                        _price_cache[symbol] = (price, time.time())
                        _update_price_source_health('Binance', True)
                        return price
                    else:
                        _update_price_source_health('Binance', False)
        except Exception as e:
            print(f"Binance price fetch failed for {symbol}: {e}")
            _update_price_source_health('Binance', False)
            pass

        # Try CoinGecko API for altcoins and smaller tokens
        try:
            import requests

            # Clean symbol for CoinGecko
            base = symbol.split('/')[0] if '/' in symbol else symbol
            base = base.replace('USDT', '').replace('USDC', '').replace('USD', '')

            # Comprehensive CoinGecko ID mapping for popular tokens and altcoins
            coingecko_mapping = {
                # Major cryptocurrencies
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'SOL': 'solana',
                'ADA': 'cardano',
                'XRP': 'ripple',
                'DOT': 'polkadot',
                'DOGE': 'dogecoin',
                'AVAX': 'avalanche-2',
                'MATIC': 'matic-network',
                'LTC': 'litecoin',
                'ETC': 'ethereum-classic',
                'BCH': 'bitcoin-cash',
                'LINK': 'chainlink',
                'UNI': 'uniswap',
                'ALGO': 'algorand',
                'ICP': 'internet-computer',
                'FIL': 'filecoin',
                'TRX': 'tron',
                'ETC': 'ethereum-classic',
                'VET': 'vechain',
                'HBAR': 'hedera-hashgraph',
                'FLOW': 'flow',
                'MANA': 'decentraland',
                'SAND': 'the-sandbox',
                'AXS': 'axie-infinity',
                'CHZ': 'chiliz',
                'ENJ': 'enjincoin',

                # DeFi tokens
                'AAVE': 'aave',
                'SUSHI': 'sushi',
                'COMP': 'compound-governance-token',
                'MKR': 'maker',
                'YFI': 'yearn-finance',
                'BAL': 'balancer',
                'CRV': 'curve-dao-token',
                'REN': 'ren',
                'BAT': 'basic-attention-token',
                'OMG': 'omisego',
                'LRC': 'loopring',
                'REP': 'augur',
                'ANT': 'aragon',
                'STORJ': 'storj',
                'GNT': 'golem',

                # Solana ecosystem
                'RAY': 'raydium',
                'SRM': 'serum',
                'FTT': 'ftx-token',
                'KIN': 'kin',
                'ORCA': 'orca',
                'COPE': 'cope',
                'SAMO': 'samoyedcoin',
                'BONK': 'bonk',
                'SHIB': 'shiba-inu',
                'AKITA': 'akita-inu',
                'Hoge': 'hoge-finance',
                'WOOF': 'woof-token',
                'ELON': 'dogelon-mars',
                'SAFEMOON': 'safemoon',
                'CUMMIES': 'cumrocket',
                'SAFEMOON': 'safemoon-2',
                'MOONSHOT': 'moonshot',
                'GOAT': 'goatseus-maximus',
                'MEW': 'cat-in-a-dogs-world',
                'WIF': 'dogwifcoin',
                'BOME': 'book-of-meme',
                'POPCAT': 'popcat',
                'CLOUD': 'cloudtx',
                'MOTHER': 'mother-iggy',

                # Meme coins and altcoins
                'PEPE': 'pepe',
                'FLOKI': 'floki',
                'BABYDOGE': 'baby-doge-coin',
                'SHIB': 'shiba-inu',
                'CUMMIES': 'cumrocket',
                'SAFEMOON': 'safemoon-2',
                'MOONSHOT': 'moonshot',
                'ELON': 'dogelon-mars',
                'AKITA': 'akita-inu',
                'Hoge': 'hoge-finance',
                'WOOF': 'woof-token',
                'RAIIN': 'raiinmaker',
                'USDUC': 'usduc',
                'FEG': 'feg-token',
                'YOOSHI': 'yooshi',
                'QUACK': 'richquack',
                'PIT': 'pitbull',
                'HOGE': 'hoge-finance',
                'CUMMIES': 'cumrocket',
                'SAFEMOON': 'safemoon-2',
                'MOONSHOT': 'moonshot',
                'ELON': 'dogelon-mars',

                # Gaming tokens
                'GAL': 'galatasaray-fan-token',
                'JUV': 'juventus-fan-token',
                'BAR': 'fc-barcelona-fan-token',
                'PSG': 'paris-saint-germain-fan-token',
                'ASR': 'as-roma-fan-token',
                'CITY': 'manchester-city-fan-token',
                'ATM': 'atletico-madrid',
                'NAP': 'napoli-fan-token',
                'ACM': 'ac-milan-fan-token',
                'IMX': 'immutable-x',
                'GALAXY': 'galaxy-fight-club',
                'GF': 'guildfi',
                'YGG': 'yield-guild-games',
                'ILV': 'illuvium',
                'REEF': 'reef',
                'SUPER': 'superfarm',
                'TLM': 'alien-worlds',
                'ATLAS': 'star-atlas',
                'POLIS': 'star-atlas-dao',

                # Layer 1 tokens
                'AVAX': 'avalanche-2',
                'FTM': 'fantom',
                'ONE': 'harmony',
                'NEAR': 'near',
                'CELO': 'celo',
                'KAVA': 'kava',
                'ONT': 'ontology',
                'QTUM': 'qtum',
                'ZIL': 'zilliqa',
                'ICX': 'icon',
                'WAVES': 'waves',
                'LSK': 'lisk',
                'ARK': 'ark',
                'STRAT': 'stratis',
                'BTG': 'bitcoin-gold',

                # Stablecoins
                'USDT': 'tether',
                'USDC': 'usd-coin',
                'BUSD': 'binance-usd',
                'DAI': 'dai',
                'FRAX': 'frax',
                'LUSD': 'liquity-usd',
                'SUSD': 'nusd',
                'USDP': 'paxos-standard',
                'GUSD': 'gemini-dollar',
                'USDN': 'neutrino',

                # Oracle tokens
                'API3': 'api3',
                'TRU': 'truefi',
                'DIA': 'dia-data',
                'OCEAN': 'ocean-protocol',
                'RLC': 'iexec-rlc',
                'GRT': 'the-graph',

                # Privacy tokens
                'XMR': 'monero',
                'ZEC': 'zcash',
                'DASH': 'dash',
                'BTCP': 'bitcoin-private',
                'XVG': 'verge',
                'PIVX': 'pivx',

                # Storage tokens
                'STORJ': 'storj',
                'FIL': 'filecoin',
                'SC': 'siacoin',
                'AR': 'arweave',
                'HOT': 'holotoken',
                'BTFS': 'bitf',
                'LPT': 'livepeer',
                'HIVE': 'hive',
                'STEEM': 'steem',

                # Exchange tokens
                'BNB': 'binancecoin',
                'CRO': 'crypto-com-chain',
                'LEO': 'leo-token',
                'OKB': 'okb',
                'HT': 'huobi-token',
                'FTT': 'ftx-token',
                'KCS': 'kucoin-shares',
                'GT': 'gatechain-token',

                # Utility tokens
                'BAT': 'basic-attention-token',
                'ANT': 'aragon',
                'GNO': 'gnosis',
                'LPT': 'livepeer',
                'REP': 'augur',
                'NMR': 'numeraire',
                'AGI': 'singularitynet',
                'FET': 'fetch-ai',
                'OCEAN': 'ocean-protocol',
                'RLC': 'iexec-rlc',
                'GRT': 'the-graph'
            }

            coin_id = coingecko_mapping.get(base.upper())
            if coin_id:
                response = requests.get(
                    f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    price = float(data.get(coin_id, {}).get('usd', 0.0))
                    if price > 0:
                        print(f"Got price for {symbol} from CoinGecko: ${price}")
                        _price_cache[symbol] = (price, time.time())
                        _update_price_source_health('CoinGecko', True)
                        return price
                    else:
                        _update_price_source_health('CoinGecko', False)
        except Exception as e:
            print(f"CoinGecko price fetch failed for {symbol}: {e}")
            _update_price_source_health('CoinGecko', False)
            pass

        # Try Coinbase Pro API as another fallback
        try:
            import requests

            # Convert symbol format for Coinbase
            coinbase_symbol = symbol.replace('/', '-').upper()

            # Map common symbol formats to Coinbase format
            coinbase_mapping = {
                'BTC-USDT': 'BTC-USD',  # Coinbase uses USD pairs
                'BTC-USD': 'BTC-USD',
                'ETH-USDT': 'ETH-USD',
                'ETH-USD': 'ETH-USD',
                'SOL-USDT': 'SOL-USD',
                'SOL-USD': 'SOL-USD',
                'ADA-USDT': 'ADA-USD',
                'ADA-USD': 'ADA-USD',
                'DOT-USDT': 'DOT-USD',
                'DOT-USD': 'DOT-USD',
                'LINK-USDT': 'LINK-USD',
                'LINK-USD': 'LINK-USD',
                'HBAR-USDT': 'HBAR-USD',
                'HBAR-USD': 'HBAR-USD'
            }

            coinbase_pair = coinbase_mapping.get(coinbase_symbol)
            if coinbase_pair:
                response = requests.get(
                    f"https://api.coinbase.com/v2/exchange-rates?currency={coinbase_pair.split('-')[0]}",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'rates' in data['data'] and 'USD' in data['data']['rates']:
                        price = float(data['data']['rates']['USD'])
                        if price > 0:
                            print(f"Got price for {symbol} from Coinbase: ${price}")
                            _price_cache[symbol] = (price, time.time())
                            _update_price_source_health('Coinbase', True)
                            return price
                        else:
                            _update_price_source_health('Coinbase', False)
        except Exception as e:
            print(f"Coinbase price fetch failed for {symbol}: {e}")
            _update_price_source_health('Coinbase', False)
            pass
        
        # Fallback: try to get from cached data if available
        try:
            scan_file = LOG_DIR / 'asset_scores.json'
            if scan_file.exists():
                with open(scan_file) as f:
                    data = json.load(f)
                    if symbol in data and 'price' in data[symbol]:
                        price = float(data[symbol]['price'])
                        if price > 0:
                            print(f"Got cached price for {symbol}: ${price}")
                            _price_cache[symbol] = (price, time.time())
                            return price
        except Exception as e:
            print(f"Cached price fetch failed for {symbol}: {e}")
            pass

        # Try CoinGecko API as additional fallback
        try:
            import requests
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            # Remove common suffixes/prefixes for CoinGecko
            base_symbol = base_symbol.replace('USDT', '').replace('USDC', '').replace('USD', '').replace('EUR', '')

            # CoinGecko ID mapping for common tokens
            coingecko_mapping = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'SOL': 'solana',
                'ADA': 'cardano',
                'DOT': 'polkadot',
                'LINK': 'chainlink',
                'UNI': 'uniswap',
                'AAVE': 'aave',
                'SUSHI': 'sushi',
                'COMP': 'compound-governance-token',
                'MKR': 'maker',
                'YFI': 'yearn-finance',
                'HBAR': 'hedera-hashgraph',
                'RAIIN': 'raiinmaker'
            }

            # Try direct mapping first
            if base_symbol in coingecko_mapping:
                coin_id = coingecko_mapping[base_symbol]
            else:
                # Fallback to lowercase base symbol
                coin_id = base_symbol.lower()

            if coin_id:
                response = requests.get(
                    f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if coin_id in data and 'usd' in data[coin_id]:
                        price = float(data[coin_id]['usd'])
                        if price > 0:
                            print(f"Got price for {symbol} from CoinGecko: ${price}")
                            _price_cache[symbol] = (price, time.time())
                            return price
        except Exception as e:
            print(f"CoinGecko price fetch failed for {symbol}: {e}")
            pass

        # Try CoinMarketCap API as final fallback
        try:
            import requests
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            # Remove common suffixes/prefixes for CoinMarketCap
            base_symbol = base_symbol.replace('USDT', '').replace('USDC', '').replace('USD', '').replace('EUR', '')

            # CoinMarketCap ID mapping for common tokens
            cmc_mapping = {
                'BTC': '1',      # Bitcoin
                'ETH': '1027',   # Ethereum
                'SOL': '5426',   # Solana
                'ADA': '2010',   # Cardano
                'DOT': '6636',   # Polkadot
                'LINK': '1975',  # Chainlink
                'UNI': '7083',   # Uniswap
                'AAVE': '7278',  # Aave
                'SUSHI': '6758', # Sushi
                'COMP': '5692',  # Compound
                'MKR': '1518',   # Maker
                'YFI': '5864',   # Yearn Finance
                'HBAR': '4642',  # Hedera Hashgraph
                'RAIIN': '10098' # Raiinmaker (if available)
            }

            # Check if we have a direct mapping
            if base_symbol in cmc_mapping:
                coin_id = cmc_mapping[base_symbol]
                # Try free tier API (limited to top 100 coins)
                response = requests.get(
                    f"https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail?id={coin_id}",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and data['data']:
                        price = float(data['data'].get('price', 0.0))
                        if price > 0:
                            print(f"Got price for {symbol} from CoinMarketCap: ${price}")
                            _price_cache[symbol] = (price, time.time())
                            return price

            # Fallback to free API for top coins
            response = requests.get(
                "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing?start=1&limit=100",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'cryptoCurrencyList' in data['data']:
                    for coin in data['data']['cryptoCurrencyList']:
                        if coin.get('symbol', '').upper() == base_symbol:
                            price = float(coin.get('quotes', [{}])[0].get('price', 0.0))
                            if price > 0:
                                print(f"Got price for {symbol} from CoinMarketCap: ${price}")
                                _price_cache[symbol] = (price, time.time())
                                return price

        except Exception as e:
            print(f"CoinMarketCap price fetch failed for {symbol}: {e}")
            pass

        # Enhanced fallback: try to get from positions.log if it's a recent position
        try:
            if POSITIONS_FILE.exists():
                with open(POSITIONS_FILE, 'r') as f:
                    lines = f.readlines()

                # Look for recent position with this symbol
                for line in reversed(lines[-50:]):  # Check last 50 lines
                    if symbol in line and 'current' in line:
                        import re
                        current_match = re.search(r'current ([0-9.]+)', line)
                        if current_match:
                            price = float(current_match.group(1))
                            if price > 0:
                                print(f"Got fallback price for {symbol} from positions log: ${price}")
                                _price_cache[symbol] = (price, time.time())
                                return price
        except Exception as e:
            print(f"Positions log fallback failed for {symbol}: {e}")
            pass

        # Last resort: try to get entry price from trades.csv
        try:
            if hasattr(TRADE_FILE, 'exists') and TRADE_FILE.exists():
                import pandas as pd
                df = pd.read_csv(TRADE_FILE)
                if not df.empty and 'symbol' in df.columns and 'price' in df.columns:
                    symbol_trades = df[df['symbol'] == symbol]
                    if not symbol_trades.empty:
                        # Get the most recent trade price for this symbol
                        latest_price = symbol_trades['price'].iloc[-1]
                        if latest_price > 0:
                            print(f"Got fallback price for {symbol} from trade history: ${latest_price}")
                            _price_cache[symbol] = (latest_price, time.time())
                            return latest_price
        except Exception as e:
            print(f"Trade history fallback failed for {symbol}: {e}")
            pass

        # Only log final failure once per symbol per session to reduce noise
        if not hasattr(get_current_price_for_symbol, '_failed_symbols'):
            get_current_price_for_symbol._failed_symbols = set()

        if symbol not in get_current_price_for_symbol._failed_symbols:
            print(f"All price sources failed for {symbol}, returning 0.0 (entry price will be used)")
            print(f"💡 Tip: You can set a manual price for {symbol} via the dashboard or API")
            get_current_price_for_symbol._failed_symbols.add(symbol)
        else:
            print(f"Price fetch still failing for {symbol} (entry price will be used)")

        return 0.0
        
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
        return 0.0


@app.route('/trades_data')
def trades_data():
    """Return full trade history as JSON records with PnL calculations."""
    if TRADE_FILE.exists():
        try:
            df = log_reader._read_trades(TRADE_FILE)
            if not df.empty:
                # Get current prices for PnL calculation
                current_prices = {}
                try:
                    # Get current prices from the new endpoint
                    import requests
                    response = requests.get('http://localhost:8000/api/current-prices', timeout=5)
                    if response.status_code == 200:
                        current_prices = response.json()
                except Exception:
                    # Fallback: try to get prices directly
                    symbols = df['symbol'].unique().tolist()
                    for symbol in symbols:
                        price = get_current_price_for_symbol(symbol)
                        if price > 0:
                            current_prices[symbol] = price
                
                # Get open positions using the same logic as console monitor
                from crypto_bot.utils.open_trades import get_open_trades
                open_trades = get_open_trades(TRADE_FILE)
                print(f"DEBUG: open_trades returned: {open_trades}")
                if not open_trades:
                    return jsonify([])
                
                # Create a mapping of open positions for quick lookup
                open_positions = {}
                for trade in open_trades:
                    symbol = trade['symbol']
                    open_positions[symbol] = {
                        'side': trade['side'],
                        'price': float(trade['price']),
                        'amount': float(trade['amount'])
                    }
                
                # Calculate PnL for each trade
                records = []
                
                # Track position history for realized P&L calculation
                position_history = {}
                
                for _, row in df.iterrows():
                    symbol = str(row.get('symbol', ''))
                    side = str(row.get('side', ''))
                    amount = float(row.get('amount', 0))
                    price = float(row.get('price', 0))
                    timestamp = str(row.get('timestamp', ''))
                    
                    # Calculate trade total
                    total = amount * price
                    
                    # Calculate realized PnL for this trade
                    pnl = 0.0
                    pnl_percentage = 0.0
                    
                    # Check if this trade closes an existing position
                    if symbol in position_history:
                        existing_pos = position_history[symbol]
                        
                        # Check if this is a closing trade (opposite side)
                        if ((side == 'sell' and existing_pos['side'] == 'buy') or 
                            (side == 'buy' and existing_pos['side'] == 'sell')):
                            
                            # Calculate realized P&L
                            if side == 'sell':  # Closing long position
                                pnl = (price - existing_pos['price']) * min(amount, existing_pos['amount'])
                            else:  # Closing short position
                                pnl = (existing_pos['price'] - price) * min(amount, existing_pos['amount'])
                            
                            # Calculate percentage
                            if existing_pos['price'] > 0:
                                pnl_percentage = (pnl / (existing_pos['price'] * min(amount, existing_pos['amount']))) * 100
                            
                            # Update or remove position
                            if amount >= existing_pos['amount']:
                                del position_history[symbol]
                            else:
                                position_history[symbol]['amount'] -= amount
                        else:
                            # Same side trade - average the position
                            total_cost = (existing_pos['price'] * existing_pos['amount']) + total
                            total_amount = existing_pos['amount'] + amount
                            position_history[symbol] = {
                                'side': side,
                                'price': total_cost / total_amount,
                                'amount': total_amount
                            }
                    else:
                        # New position
                        position_history[symbol] = {
                            'side': side,
                            'price': price,
                            'amount': amount
                        }
                    
                    # Calculate unrealized PnL for open positions only
                    unrealized_pnl = 0.0
                    unrealized_pnl_percentage = 0.0
                    
                    # Determine if this trade is still open
                    is_open = symbol in open_positions
                    
                    # Check for very small amounts that should be considered closed (precision issues)
                    if is_open and symbol in open_positions:
                        pos = open_positions[symbol]
                        if pos['amount'] < 0.0001:  # Less than 0.01% - consider closed
                            is_open = False
                    
                    # Only calculate unrealized P&L for truly open positions
                    if is_open and symbol in current_prices and symbol in open_positions:
                        current_price = current_prices[symbol]
                        if current_price > 0:
                            pos = open_positions[symbol]
                            if pos['side'] == 'long':
                                unrealized_pnl = (current_price - pos['price']) * pos['amount']
                            else:  # short
                                unrealized_pnl = (pos['price'] - current_price) * pos['amount']
                            
                            if pos['price'] > 0:
                                unrealized_pnl_percentage = (unrealized_pnl / (pos['price'] * pos['amount'])) * 100
                    
                    # Set status based on whether the position is still open
                    status = 'active' if is_open else 'completed'
                    
                    # Debug: Log P&L values for completed trades
                    if status == 'completed' and pnl != 0:
                        print(f"DEBUG: {symbol} {side} - Realized P&L: {pnl}, Unrealized: {unrealized_pnl}, Status: {status}")
                    
                    record = {
                        'symbol': symbol,
                        'side': side,
                        'amount': amount,
                        'price': price,
                        'timestamp': timestamp,
                        'total': total,
                        'status': status,
                        'pnl': pnl,
                        'pnl_percentage': pnl_percentage,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_percentage': unrealized_pnl_percentage,
                        'current_price': current_prices.get(symbol, 0.0)
                    }
                    records.append(record)
                
                return jsonify(records)
            else:
                return jsonify([])
        except Exception as e:
            print(f"Error reading trades: {e}")
            return jsonify([])
    return jsonify([])


@app.route('/api/bot-status')
def api_bot_status():
    """Return current bot status as JSON."""
    return jsonify({
        'running': is_running(),
        'mode': load_execution_mode(),
        'uptime': get_uptime(),
        'last_trade': utils.get_last_trade(TRADE_FILE),
        'regime': utils.get_current_regime(LOG_FILE),
        'last_reason': utils.get_last_decision_reason(LOG_FILE),
    })





@app.route('/api/open-positions')
def api_open_positions():
    """Return open positions as JSON."""
    try:
        from crypto_bot.utils.open_trades import get_open_trades
        
        # Get open positions
        open_trades = get_open_trades(TRADE_FILE)
        if not open_trades:
            return jsonify([])
        
        # Get current prices
        current_prices = {}
        try:
            import requests
            response = requests.get('http://localhost:8000/api/current-prices', timeout=5)
            if response.status_code == 200:
                current_prices = response.json()
        except Exception:
            # Fallback: get prices directly
            symbols = [trade['symbol'] for trade in open_trades]
            for symbol in symbols:
                price = get_current_price_for_symbol(symbol)
                if price > 0:
                    current_prices[symbol] = price
        
        # Calculate PnL for each open position
        positions = []
        for trade in open_trades:
            symbol = trade['symbol']
            entry_price = float(trade['price'])
            amount = float(trade['amount'])
            side = trade['side']
            current_price = current_prices.get(symbol, 0.0)
            
            # Calculate PnL
            if side == 'long':
                pnl = (current_price - entry_price) * amount
            else:  # short
                pnl = (entry_price - current_price) * amount
            
            pnl_percentage = 0.0
            if entry_price > 0:
                pnl_percentage = (pnl / (entry_price * amount)) * 100
            
            position = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'amount': amount,
                'current_price': current_price,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'entry_time': trade.get('entry_time', ''),
                'timestamp': trade.get('entry_time', '')
            }
            positions.append(position)
        
        return jsonify(positions)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/live-signals')
def api_live_signals():
    """Return live trading signals as JSON."""
    try:
        # Read asset scores for live signals
        if SCAN_FILE.exists():
            with open(SCAN_FILE) as f:
                data = json.load(f)
                # Return the most recent scores as live signals
                if data and isinstance(data, dict):
                    return jsonify(data)
        return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/strategy-performance')
def api_strategy_performance():
    """Return strategy performance breakdown as JSON."""
    try:
        # Read strategy stats
        if STATS_FILE.exists():
            with open(STATS_FILE) as f:
                data = json.load(f)
                return jsonify(data)
        
        # Fallback: create basic structure from trades
        if TRADE_FILE.exists():
            df = log_reader._read_trades(TRADE_FILE)
            if not df.empty:
                # Group by strategy if available, otherwise use symbol
                strategy_col = 'strategy' if 'strategy' in df.columns else 'symbol'
                if strategy_col in df.columns:
                    grouped = df.groupby(strategy_col).size().to_dict()
                    return jsonify({
                        'overall': {
                            strategy: {'trades': count} for strategy, count in grouped.items()
                        }
                    })
        
        return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/stop_conflicts', methods=['POST'])
def stop_conflicts():
    """Stop any conflicting bot processes."""
    stop_conflicting_bots()
    return jsonify({'status': 'conflicts_stopped'})


@app.route('/api/price-source-health')
def price_source_health():
    """Return health status of all price sources."""
    try:
        health_data = {}
        for source_name in _price_source_health.keys():
            health_data[source_name] = _get_price_source_health(source_name)

        # Add sources that haven't been used yet with default values
        # Focus on Kraken + complementary sources for better coverage
        all_sources = ['Pyth', 'Helius', 'Raydium', 'Orca', 'Jupiter', 'Kraken', 'CoinGecko', 'Coinbase']
        for source in all_sources:
            if source not in health_data:
                health_data[source] = {
                    'health_score': 0.5,
                    'total_attempts': 0,
                    'success_rate': 0.0,
                    'consecutive_failures': 0
                }

        return jsonify({
            'price_sources': health_data,
            'best_sources': _get_best_price_sources()[:5]  # Top 5 healthiest sources
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/manual-prices', methods=['GET'])
def get_manual_prices():
    """Get all manual price overrides."""
    try:
        return jsonify({
            'manual_prices': _manual_prices,
            'count': len(_manual_prices)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/manual-prices', methods=['POST'])
def set_manual_price_endpoint():
    """Set a manual price override."""
    try:
        data = request.get_json()
        if not data or 'symbol' not in data or 'price' not in data:
            return jsonify({'error': 'Missing symbol or price'}), 400

        symbol = data['symbol'].strip().upper()
        price = float(data['price'])
        validity_hours = int(data.get('validity_hours', 24))

        if price <= 0:
            return jsonify({'error': 'Price must be positive'}), 400

        _set_manual_price(symbol, price, validity_hours)

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'price': price,
            'validity_hours': validity_hours
        })
    except ValueError as e:
        return jsonify({'error': 'Invalid price format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/manual-prices/<symbol>', methods=['DELETE'])
def delete_manual_price(symbol):
    """Delete a manual price override."""
    try:
        symbol = symbol.strip().upper()
        if symbol in _manual_prices:
            del _manual_prices[symbol]
            _save_manual_prices()
            return jsonify({'status': 'deleted', 'symbol': symbol})
        else:
            return jsonify({'error': 'Manual price not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard-metrics')
def api_dashboard_metrics():
    """Return comprehensive dashboard metrics as JSON."""
    try:
        # Read trades and calculate performance
        df = log_reader._read_trades(TRADE_FILE)
        perf = utils.compute_performance(df)
        
        # Get dynamic allocation data based on actual performance
        allocation = utils.calculate_dynamic_allocation()
        
        # Fallback to static config if no dynamic data available
        if not allocation and CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                cfg = yaml.safe_load(f) or {}
                allocation = cfg.get('strategy_allocation', {})
        
        # Final fallback to weights.json if no allocation in config
        if not allocation and (LOG_DIR / 'weights.json').exists():
            with open(LOG_DIR / 'weights.json') as f:
                weights_data = json.load(f)
                # Convert decimal weights to percentages for consistency
                allocation = {strategy: weight * 100 for strategy, weight in weights_data.items()}
        
        # Get asset scores
        asset_scores = {}
        if SCAN_FILE.exists():
            with open(SCAN_FILE) as f:
                asset_scores = json.load(f)
        
        # Get recent trades using the log_reader function
        recent_trades = []
        if not df.empty:
            # Get last 10 trades from the DataFrame
            recent_df = df.tail(10)
            for _, row in recent_df.iterrows():
                recent_trades.append({
                    'symbol': str(row.get('symbol', '')),
                    'side': str(row.get('side', '')),
                    'amount': float(row.get('amount', 0.0)),
                    'price': float(row.get('price', 0.0)),
                    'timestamp': str(row.get('timestamp', ''))
                })
        
        # Get open positions
        open_positions = get_open_positions()
        
        return jsonify({
            'performance': perf,
            'allocation': allocation,
            'asset_scores': asset_scores,
            'recent_trades': recent_trades,
            'open_positions': open_positions,
            'bot_status': {
                'running': is_running(),
                'mode': load_execution_mode(),
                'uptime': get_uptime(),
                'regime': utils.get_current_regime(LOG_FILE),
                'last_reason': utils.get_last_decision_reason(LOG_FILE)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/paper-wallet-balance', methods=['GET', 'POST'])
def api_paper_wallet_balance():
    """Get or set paper wallet balance."""
    if request.method == 'POST':
        try:
            data = request.get_json()
            balance = float(data.get('balance', 10000.0))
            set_paper_wallet_balance(balance)
            return jsonify({'success': True, 'balance': balance})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        balance = get_paper_wallet_balance()
        return jsonify({'balance': balance})


@app.route('/api/wallet-balance')
def api_wallet_balance():
    """Get current wallet balance from bot logs."""
    try:
        # Try to get the most recent balance from positions.log
        if POSITIONS_FILE.exists():
            with open(POSITIONS_FILE, 'r') as f:
                lines = f.readlines()
                
            # Look for the most recent balance entry
            for line in reversed(lines):
                if 'balance $' in line:
                    # Extract balance using regex
                    import re
                    balance_match = re.search(r'balance \$?([0-9.]+)', line)
                    if balance_match:
                        balance = float(balance_match.group(1))
                        return jsonify({
                            'success': True,
                            'balance': balance,
                            'source': 'positions_log'
                        })
        
        # Fallback to paper wallet config
        balance = get_paper_wallet_balance()
        return jsonify({
            'success': True,
            'balance': balance,
            'source': 'paper_wallet_config'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'balance': 0.0
        })


@app.route('/api/live-updates')
def api_live_updates():
    """Return live updates for real-time dashboard."""
    try:
        # Get current bot status
        bot_status = {
            'running': is_running(),
            'mode': load_execution_mode(),
            'uptime': get_uptime(),
            'regime': utils.get_current_regime(LOG_FILE),
            'last_reason': utils.get_current_regime(LOG_FILE)
        }
        
        # Get latest performance data
        df = log_reader._read_trades(TRADE_FILE)
        perf = utils.compute_performance(df)
        
        # Get latest asset scores
        asset_scores = {}
        if SCAN_FILE.exists():
            with open(SCAN_FILE) as f:
                asset_scores = json.load(f)
        
        # Get paper wallet balance if in dry run mode
        paper_wallet_balance = None
        if bot_status['mode'] == 'dry_run':
            paper_wallet_balance = get_paper_wallet_balance()
        
        # Get open positions
        open_positions = get_open_positions()
        
        return jsonify({
            'timestamp': time.time(),
            'bot_status': bot_status,
            'performance': perf,
            'asset_scores': asset_scores,
            'paper_wallet_balance': paper_wallet_balance,
            'open_positions': open_positions
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/sell-position', methods=['POST'])
def api_sell_position():
    """Sell a specific position immediately via market order."""
    try:
        print(f"Received sell position request: {request.get_json()}")
        data = request.get_json()
        symbol = data.get('symbol')
        amount = data.get('amount')
        
        print(f"Processing sell request for {amount} {symbol}")
        
        if not symbol or not amount:
            print(f"Missing symbol or amount: symbol={symbol}, amount={amount}")
            return jsonify({'success': False, 'error': 'Missing symbol or amount'})
        
        # Write sell request to a state file that the main bot can read
        sell_request = {
            'symbol': symbol,
            'amount': float(amount),
            'timestamp': time.time()
        }
        
        sell_state_file = LOG_DIR / 'sell_requests.json'
        try:
            # Read existing requests
            if sell_state_file.exists():
                with open(sell_state_file, 'r') as f:
                    requests = json.load(f)
            else:
                requests = []
            
            # Add new request
            requests.append(sell_request)
            
            # Keep only recent requests (last 10)
            requests = requests[-10:]
            
            # Write back to file
            with open(sell_state_file, 'w') as f:
                json.dump(requests, f)
            
            print(f"Sell request saved successfully for {amount} {symbol}")
            return jsonify({
                'success': True,
                'message': f'Market sell order submitted for {amount} {symbol}',
                'symbol': symbol,
                'amount': amount
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to save sell request: {str(e)}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/clear-old-positions', methods=['POST'])
def api_clear_old_positions():
    """Clear old position entries from the log file."""
    try:
        clear_old_positions()
        return jsonify({'success': True, 'message': 'Old positions cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/generate_sample_trades', methods=['POST'])
def api_generate_sample_trades():
    """Generate sample trade data for testing and demonstration."""
    try:
        import csv
        from datetime import datetime, timedelta
        import random
        
        # Create sample trade data
        symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'MATIC/USD']
        sample_trades = []
        
        # Generate trades over the last 7 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        current_time = start_time
        while current_time <= end_time:
            # Generate 1-3 trades per day
            trades_per_day = random.randint(1, 3)
            
            for _ in range(trades_per_day):
                symbol = random.choice(symbols)
                side = random.choice(['buy', 'sell'])
                
                # Generate realistic prices based on symbol
                base_prices = {
                    'BTC/USD': 50000,
                    'ETH/USD': 3000,
                    'SOL/USD': 100,
                    'ADA/USD': 0.5,
                    'MATIC/USD': 0.8
                }
                
                base_price = base_prices.get(symbol, 100)
                price_variation = random.uniform(0.95, 1.05)
                price = base_price * price_variation
                
                # Generate realistic amounts
                if symbol == 'BTC/USD':
                    amount = random.uniform(0.001, 0.01)
                elif symbol == 'ETH/USD':
                    amount = random.uniform(0.01, 0.1)
                else:
                    amount = random.uniform(1, 100)
                
                # Add some time variation within the day
                trade_time = current_time + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                sample_trades.append([
                    symbol,
                    side,
                    f"{amount:.6f}",
                    f"{price:.6f}",
                    trade_time.strftime("%Y-%m-%d %H:%M:%S")
                ])
            
            current_time += timedelta(days=1)
        
        # Sort by timestamp
        sample_trades.sort(key=lambda x: x[4])
        
        # Write to trades.csv
        with open(TRADE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['symbol', 'side', 'amount', 'price', 'timestamp'])
            writer.writerows(sample_trades)
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(sample_trades)} sample trades',
            'count': len(sample_trades),
            'file': str(TRADE_FILE)
        })
        
    except Exception as e:
        print(f"Error generating sample trades: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/generate_scores', methods=['POST'])
def api_generate_scores():
    """Generate sample asset scores for the Scans page."""
    try:
        # Import the score generator
        import sys
        from pathlib import Path
        
        # Add tools directory to path
        tools_dir = Path(__file__).parent.parent / 'tools'
        sys.path.insert(0, str(tools_dir))
        
        # Import and run the score generator
        from generate_asset_scores import generate_sample_scores, save_asset_scores
        
        # Generate scores
        scores = generate_sample_scores()
        
        # Save to file
        save_asset_scores(scores)
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(scores)} asset scores',
            'count': len(scores)
        })
        
    except Exception as e:
        print(f"Error generating scores: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/test')
def test():
    """Simple test endpoint to verify the Flask app is running."""
    return jsonify({
        'status': 'ok',
        'message': 'Flask app is running',
        'timestamp': time.time(),
        'debug': True
    })


@app.route('/debug_market_sell')
def debug_market_sell():
    """Debug page for testing market sell functionality."""
    return render_template('debug_market_sell.html')


if __name__ == '__main__':
    print("=== FLASK APP STARTUP DEBUG ===")
    print("Starting Flask app...")
    
    watch_thread = threading.Thread(target=watch_bot, daemon=True)
    watch_thread.start()
    print("Watch thread started")
    
    # Try to find an available port starting from 8000
    import socket
    
    def find_free_port(start_port=8000, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # fallback
    
    # Configure Flask to be accessible from any host (for containerized deployments)
    # and find an available port starting from 8000 (avoiding macOS ControlCenter on port 5000)
    port = int(os.environ.get('FLASK_RUN_PORT', find_free_port()))
    print(f"Starting Flask app on port {port}")
    print("=== END FLASK APP STARTUP DEBUG ===")
    app.run(host='0.0.0.0', port=port, debug=False)
