"""Start the web dashboard and expose REST endpoints for the trading bot.

This module launches the Flask web server, manages the background trading
process and provides REST API routes used by the UI and tests.
"""

from flask import Flask, render_template, redirect, url_for, request, jsonify
from pathlib import Path
import os

from crypto_bot.utils.logger import LOG_DIR
import subprocess
import json
import threading
import time
import yaml
from crypto_bot import log_reader
from crypto_bot import ml_signal_model as ml
from . import utils

app = Flask(__name__)

# Handle the async trading bot process
bot_proc = None
bot_start_time = None
watch_thread = None
LOG_FILE = LOG_DIR / 'bot.log'
STATS_FILE = LOG_DIR / 'strategy_stats.json'
SCAN_FILE = LOG_DIR / 'asset_scores.json'
MODEL_REPORT = Path('crypto_bot/ml_signal_model/models/model_report.json')
TRADE_FILE = LOG_DIR / 'trades.csv'
ERROR_FILE = LOG_DIR / 'errors.log'
CONFIG_FILE = Path('crypto_bot/config.yaml')
REGIME_FILE = LOG_DIR / 'regime_history.txt'


def watch_bot() -> None:
    """Monitor the trading bot and restart it if the process exits."""
    global bot_proc, bot_start_time
    while True:
        time.sleep(5)
        if bot_proc is not None and bot_proc.poll() is not None:
            bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
            bot_start_time = time.time()



def is_running() -> bool:
    """Return True if the bot process is running."""
    return utils.is_running(bot_proc)


def set_execution_mode(mode: str) -> None:
    """Set execution mode in config file."""
    utils.set_execution_mode(mode, CONFIG_FILE)


def load_execution_mode() -> str:
    """Load execution mode from config file."""
    return utils.load_execution_mode(CONFIG_FILE)


def get_uptime() -> str:
    """Return human readable uptime."""
    return utils.get_uptime(bot_start_time)




@app.route('/')
def index():
    mode = load_execution_mode()
    
    # Get performance data
    df = log_reader._read_trades(TRADE_FILE)
    perf = utils.compute_performance(df)
    
    # Get allocation data
    allocation = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
            allocation = cfg.get('strategy_allocation', {})
    
    # Fallback to weights.json if no allocation in config
    if not allocation and (LOG_DIR / 'weights.json').exists():
        with open(LOG_DIR / 'weights.json') as f:
            allocation = json.load(f)
    
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
    )




@app.route('/start', methods=['POST'])
def start():
    global bot_proc, bot_start_time
    mode = request.form.get('mode', 'dry_run')
    set_execution_mode(mode)
    if not is_running():
        # Launch the asyncio-based trading bot
        bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
        bot_start_time = time.time()
    return redirect(url_for('index'))


@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    mode = (
        request.json.get('mode', 'dry_run') if request.is_json else request.form.get('mode', 'dry_run')
    )
    set_execution_mode(mode)
    status = 'running'
    if not is_running():
        bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
        bot_start_time = time.time()
        status = 'started'
    return jsonify({
        'status': status,
        'running': True,
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
    if is_running():
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
        if base == 'backtest':
            cmd = f"python -m crypto_bot.backtest.backtest_runner {cmd_args}"
        elif base == 'custom':
            cmd = cmd_args
        else:
            cmd = f"python -m crypto_bot.main {cmd_args}"
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
    allocation = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
            allocation = cfg.get('strategy_allocation', {})
    
    # Fallback to weights.json if no allocation in config
    if not allocation and (LOG_DIR / 'weights.json').exists():
        with open(LOG_DIR / 'weights.json') as f:
            allocation = json.load(f)
    
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


@app.route('/trades_data')
def trades_data():
    """Return full trade history as JSON records."""
    if TRADE_FILE.exists():
        df = log_reader._read_trades(TRADE_FILE)
        return jsonify(df.to_dict(orient='records'))
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


@app.route('/api/dashboard-metrics')
def api_dashboard_metrics():
    """Return comprehensive dashboard metrics as JSON."""
    try:
        # Read trades and calculate performance
        df = log_reader._read_trades(TRADE_FILE)
        perf = utils.compute_performance(df)
        
        # Get allocation data
        allocation = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                cfg = yaml.safe_load(f) or {}
                allocation = cfg.get('strategy_allocation', {})
        
        # Fallback to weights.json if no allocation in config
        if not allocation and (LOG_DIR / 'weights.json').exists():
            with open(LOG_DIR / 'weights.json') as f:
                allocation = json.load(f)
        
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
        
        return jsonify({
            'performance': perf,
            'allocation': allocation,
            'asset_scores': asset_scores,
            'recent_trades': recent_trades,
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


if __name__ == '__main__':
    watch_thread = threading.Thread(target=watch_bot, daemon=True)
    watch_thread.start()
    
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
    app.run(host='0.0.0.0', port=port, debug=False)
