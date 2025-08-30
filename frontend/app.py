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

app = Flask(__name__)

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
    """Get paper wallet balance from config."""
    # Try multiple possible paths for the paper wallet config (same as main bot)
    possible_paths = [
        Path("crypto_bot/paper_wallet_config.yaml"),  # Relative to current directory
        Path(__file__).parent.parent / "paper_wallet_config.yaml",  # Relative to frontend/app.py
        Path.cwd() / "crypto_bot" / "paper_wallet_config.yaml",  # Relative to working directory
        LOG_DIR / 'paper_wallet.yaml',  # Legacy location (fallback)
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    balance = config.get('initial_balance', 10000.0)
                    print(f"Frontend loaded paper wallet balance from {config_path}: ${balance:.2f}")
                    return balance
            except Exception as e:
                print(f"Frontend failed to read paper wallet config {config_path}: {e}")
                continue
    
    print("Frontend: No paper wallet config found, using default balance: $10000.0")
    return 10000.0  # Default balance


def set_paper_wallet_balance(balance: float) -> None:
    """Set paper wallet balance in config."""
    # Write to the same config file that the main bot reads from
    primary_config_path = Path("crypto_bot/paper_wallet_config.yaml")
    
    # Also write to the legacy location for backward compatibility
    legacy_config_path = LOG_DIR / 'paper_wallet.yaml'
    
    config = {'initial_balance': balance}
    
    # Write to primary location
    try:
        primary_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(primary_config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Frontend wrote paper wallet balance to {primary_config_path}: ${balance:.2f}")
    except Exception as e:
        print(f"Frontend failed to write to primary config {primary_config_path}: {e}")
    
    # Write to legacy location for backward compatibility
    try:
        legacy_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(legacy_config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Frontend wrote paper wallet balance to legacy config {legacy_config_path}: ${balance:.2f}")
    except Exception as e:
        print(f"Frontend failed to write to legacy config {legacy_config_path}: {e}")


def get_open_positions() -> list:
    """Parse open positions from positions.log file."""
    import re
    from datetime import datetime, timedelta
    
    if not POSITIONS_FILE.exists():
        return []
    
    positions = []
    pos_pattern = re.compile(
        r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
        r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
        r"pnl \$?(?P<pnl>[0-9.+-]+).*balance \$?(?P<balance>[0-9.]+)"
    )
    
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
                        match = pos_pattern.search(line)
                        if match:
                            # Check if this is a real position (not just a balance update)
                            symbol = match.group('symbol')
                            side = match.group('side')
                            amount = float(match.group('amount'))
                            
                            # Filter out positions with zero amounts or very small amounts
                            if amount > 0.0001:  # Minimum threshold
                                recent_positions.append({
                                    'symbol': symbol,
                                    'side': side,
                                    'amount': amount,
                                    'entry_price': float(match.group('entry')),
                                    'current_price': float(match.group('current')),
                                    'pnl': float(match.group('pnl')),
                                    'balance': float(match.group('balance')),
                                    'timestamp': timestamp_str
                                })
                except ValueError:
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
        print(f"Error reading positions: {e}")
    
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
    
    return render_template(
        'index.html',
        running=is_running(),
        mode=mode,
        uptime=get_uptime(),
        last_trade=utils.get_last_trade(TRADE_FILE),
        regime=utils.get_current_regime(LOG_FILE),
        last_reason=utils.get_last_decision_reason(LOG_FILE),
        pnl=perf.get('total_pnl', 0.0),
        performance=perf,
        allocation=allocation,
        paper_wallet_balance=paper_wallet_balance,
        open_positions=open_positions,
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


def get_current_price_for_symbol(symbol: str) -> float:
    """Get current price for a symbol using available price sources."""
    try:
        # Try Pyth network first
        from crypto_bot.utils.pyth import get_pyth_price
        price = get_pyth_price(symbol)
        if price and price > 0:
            return price
    except Exception:
        pass
    
    try:
        # Try Jupiter API for Solana tokens
        import requests
        
        base = symbol.split('/')[0] if '/' in symbol else symbol
        response = requests.get(
            "https://price.jup.ag/v4/price",
            params={"ids[]": base},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            price = float(data.get("data", {}).get(base, {}).get("price", 0.0))
            if price > 0:
                return price
    except Exception:
        pass
    
    # Fallback: try to get from cached data if available
    try:
        scan_file = LOG_DIR / 'asset_scores.json'
        if scan_file.exists():
            with open(scan_file) as f:
                data = json.load(f)
                if symbol in data and 'price' in data[symbol]:
                    return float(data[symbol]['price'])
    except Exception:
        pass
    
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
                
                # Calculate PnL for each trade
                records = []
                open_positions = {}  # Track open positions per symbol
                
                for _, row in df.iterrows():
                    symbol = str(row.get('symbol', ''))
                    side = str(row.get('side', ''))
                    amount = float(row.get('amount', 0))
                    price = float(row.get('price', 0))
                    timestamp = str(row.get('timestamp', ''))
                    
                    # Calculate trade total
                    total = amount * price
                    
                    # Calculate PnL for this trade
                    pnl = 0.0
                    pnl_percentage = 0.0
                    
                    if symbol in open_positions:
                        # Check if this trade closes an existing position
                        if (side == 'sell' and open_positions[symbol]['side'] == 'buy') or \
                           (side == 'buy' and open_positions[symbol]['side'] == 'sell'):
                            # Calculate realized PnL
                            entry_price = open_positions[symbol]['price']
                            entry_amount = open_positions[symbol]['amount']
                            
                            if side == 'sell':  # Closing long position
                                pnl = (price - entry_price) * min(amount, entry_amount)
                            else:  # Closing short position
                                pnl = (entry_price - price) * min(amount, entry_amount)
                            
                            pnl_percentage = (pnl / (entry_price * min(amount, entry_amount))) * 100
                            
                            # Update or remove position
                            if amount >= entry_amount:
                                del open_positions[symbol]
                            else:
                                open_positions[symbol]['amount'] -= amount
                        else:
                            # Same side trade - update position
                            if symbol in open_positions:
                                # Average down/up
                                total_cost = (open_positions[symbol]['price'] * open_positions[symbol]['amount']) + total
                                total_amount = open_positions[symbol]['amount'] + amount
                                open_positions[symbol]['price'] = total_cost / total_amount
                                open_positions[symbol]['amount'] = total_amount
                            else:
                                open_positions[symbol] = {'side': side, 'price': price, 'amount': amount}
                    else:
                        # New position
                        open_positions[symbol] = {'side': side, 'price': price, 'amount': amount}
                    
                    # Calculate unrealized PnL for open positions
                    unrealized_pnl = 0.0
                    unrealized_pnl_percentage = 0.0
                    if symbol in open_positions and symbol in current_prices:
                        current_price = current_prices[symbol]
                        if current_price > 0:
                            pos = open_positions[symbol]
                            if pos['side'] == 'buy':
                                unrealized_pnl = (current_price - pos['price']) * pos['amount']
                            else:
                                unrealized_pnl = (pos['price'] - current_price) * pos['amount']
                            
                            if pos['price'] > 0:
                                unrealized_pnl_percentage = (unrealized_pnl / (pos['price'] * pos['amount'])) * 100
                    
                    record = {
                        'symbol': symbol,
                        'side': side,
                        'amount': amount,
                        'price': price,
                        'timestamp': timestamp,
                        'total': total,
                        'status': 'completed',
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
        positions = get_open_positions()
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
        
        # Get recent trades
        recent_trades = []
        if TRADE_FILE.exists():
            lines = TRADE_FILE.read_text().strip().split('\n')
            for line in lines[-10:]:  # Last 10 trades
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 5:
                        recent_trades.append({
                            'symbol': parts[0],
                            'side': parts[1],
                            'amount': float(parts[2]),
                            'price': float(parts[3]),
                            'timestamp': parts[4]
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
