#!/usr/bin/env python3
"""Unified command-line interface for starting LegacyCoinTrader in various modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable, Dict, Iterable, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CRYPTO_ROOT = PROJECT_ROOT / "crypto_bot"
if str(CRYPTO_ROOT) not in sys.path:
    sys.path.insert(0, str(CRYPTO_ROOT))


def set_environment(auto_start: bool = True, non_interactive: bool = True, extra: Optional[Dict[str, str]] = None) -> None:
    """Set common environment variables used by the trading bot."""

    env_vars: Dict[str, str] = {}
    if auto_start:
        env_vars["AUTO_START_TRADING"] = "1"
    if non_interactive:
        env_vars["NON_INTERACTIVE"] = "1"
    if extra:
        env_vars.update(extra)

    os.environ.update(env_vars)


def force_clean_state() -> None:
    """Force a clean state by resetting caches and log files."""

    print("🧹 FORCING COMPLETE CLEAN STATE...")

    try:
        subprocess.run(["pkill", "-f", "python.*bot"], capture_output=True)
        print("✅ Killed any running bot processes")
    except Exception as exc:
        print(f"⚠️ Could not kill processes: {exc}")

    files_to_reset = [
        "crypto_bot/logs/paper_wallet_state.yaml",
        "crypto_bot/logs/trade_manager_state.json",
        "crypto_bot/logs/paper_wallet.yaml",
        "crypto_bot/paper_wallet_config.yaml",
        "crypto_bot/user_config.yaml",
        "crypto_bot/logs/positions.log",
    ]

    clean_paper_wallet_state = {
        "balance": 10000.0,
        "initial_balance": 10000.0,
        "realized_pnl": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "positions": {},
    }

    clean_trade_manager_state = {
        "trades": [],
        "positions": {},
        "price_cache": {},
        "statistics": {
            "total_trades": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "total_realized_pnl": 0.0,
        },
        "last_save_time": "2025-09-03T08:35:00.000000",
    }

    clean_paper_wallet_config = {"initial_balance": 10000.0}

    clean_user_config = {
        "coinbase_api_key": "",
        "coinbase_api_secret": "",
        "coinbase_passphrase": "",
        "exchange": "kraken",
        "mode": "cex",
        "paper_wallet_balance": 10000.0,
        "telegram_chat_id": "827777274",
        "telegram_token": "8126215032:AAEhQZLiXpssauKf0ktQsq1XqXl94QriCdE",
        "wallet_address": "EoiVpzLA6b6JBKXTB5WRFor3mPkseM6UisLHt8qK9g1c",
    }

    for file_path in files_to_reset:
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.endswith(".yaml"):
                if "user_config" in file_path:
                    with open(path, "w", encoding="utf-8") as handle:
                        yaml.dump(clean_user_config, handle, default_flow_style=False)
                elif "paper_wallet_config" in file_path or "paper_wallet.yaml" in file_path:
                    with open(path, "w", encoding="utf-8") as handle:
                        yaml.dump(clean_paper_wallet_config, handle, default_flow_style=False)
                else:
                    with open(path, "w", encoding="utf-8") as handle:
                        yaml.dump(clean_paper_wallet_state, handle, default_flow_style=False)
            elif file_path.endswith(".json"):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(clean_trade_manager_state, handle, indent=2)
            elif file_path.endswith(".log"):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("")

            print(f"✅ Reset {file_path}")
        except Exception as exc:
            print(f"❌ Failed to reset {file_path}: {exc}")

    cache_dirs = [
        "crypto_bot/__pycache__",
        "__pycache__",
        "crypto_bot/logs/__pycache__",
        "tests/__pycache__",
        "crypto_bot/utils/__pycache__",
        "crypto_bot/solana/__pycache__",
    ]

    try:
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print(f"✅ Cleared cache: {cache_dir}")
    except Exception as exc:
        print(f"⚠️ Could not clear cache: {exc}")

    try:
        backup_patterns = ["*backup*", "*migration_backup*", "*negative_balance*", "*mismatch*"]
        for pattern in backup_patterns:
            for backup_file in Path("crypto_bot/logs").glob(pattern):
                if backup_file.is_file():
                    backup_file.unlink()
                    print(f"✅ Removed backup file: {backup_file}")
    except Exception as exc:
        print(f"⚠️ Could not remove backup files: {exc}")

    try:
        modules_to_clear = [name for name in sys.modules.keys() if name.startswith("crypto_bot")]
        for module_name in modules_to_clear:
            del sys.modules[module_name]
        print(f"✅ Cleared {len(modules_to_clear)} cached Python modules")
    except Exception as exc:
        print(f"⚠️ Could not clear Python modules: {exc}")

    print("\n🎯 COMPLETE CLEAN STATE FORCED!")
    print("✅ All wallet state files reset to clean $10,000 balance")
    print("✅ All positions cleared")
    print("✅ All caches cleared")
    print("✅ All backup files removed")
    print("✅ All Python modules cleared")
    print("✅ All running processes killed")


class IsolatedPaperWallet:
    """Completely isolated paper wallet that doesn't sync with TradeManager."""

    def __init__(self, balance: float = 10000.0) -> None:
        self.balance = balance
        self.initial_balance = balance
        self.positions: Dict[str, Dict[str, float]] = {}
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0

    def open(self, symbol: str, side: str, amount: float, price: float) -> str:
        trade_id = f"trade_{len(self.positions) + 1}"
        position_value = amount * price

        if side == "buy":
            if self.balance < position_value:
                raise ValueError(f"Insufficient balance: ${self.balance:.2f} < ${position_value:.2f}")
            self.balance -= position_value
        else:
            self.balance += position_value

        self.positions[trade_id] = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "entry_price": price,
            "size": amount,
        }

        self.total_trades += 1
        return trade_id

    def close(self, trade_id: str, amount: float, price: float) -> float:
        if trade_id not in self.positions:
            raise ValueError(f"Position {trade_id} not found")

        position = self.positions[trade_id]
        entry_price = position["entry_price"]

        if position["side"] == "buy":
            pnl = (price - entry_price) * amount
            self.balance += amount * price
        else:
            pnl = (entry_price - price) * amount
            self.balance -= amount * price

        self.realized_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1

        del self.positions[trade_id]
        return pnl

    def get_position_summary(self) -> Dict[str, object]:
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "realized_pnl": self.realized_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades else 0.0,
            "open_positions": len(self.positions),
            "positions": self.positions,
        }


def start_web_server(
    *,
    debug_logging: bool = False,
    verify_main_import: bool = False,
    use_run_simple: bool = False,
    open_browser: bool = True,
    wait_seconds: float = 3.0,
) -> Tuple[Optional[threading.Thread], Optional[int]]:
    """Start the Flask web server in a separate thread."""

    prefix = "   " if debug_logging else ""
    print("🌐 Starting integrated web server...")
    if debug_logging:
        print(f"{prefix}Step 1: Importing Flask app...")

    try:
        from frontend.app import app
        print(f"{prefix}✅ Flask app imported successfully")
    except Exception as exc:
        print(f"{prefix}❌ Failed to import Flask app: {exc}")
        if debug_logging:
            import traceback

            traceback.print_exc()
        return None, None

    if verify_main_import:
        try:
            from crypto_bot.main import _main_impl  # noqa: F401

            print(f"{prefix}✅ Main bot function imported successfully")
        except Exception as exc:
            print(f"{prefix}❌ Failed to import main bot function: {exc}")
            import traceback

            traceback.print_exc()
            return None, None

    import socket

    def find_free_port(start_port: int = 8000, max_attempts: int = 10) -> int:
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("", port))
                    return port
            except OSError:
                continue
        return start_port

    port = find_free_port()
    print(f"{prefix}✅ Found free port: {port}")

    def run_flask(port_num: int) -> None:
        try:
            print(f"🌐 Web server running on http://localhost:{port_num}")
            print(f"📊 Monitoring dashboard: http://localhost:{port_num}/monitoring")
            print(f"📋 System logs: http://localhost:{port_num}/system_logs")
            print(f"🏠 Main dashboard: http://localhost:{port_num}")
            print("-" * 60)

            if use_run_simple:
                from werkzeug.serving import run_simple

                run_simple("0.0.0.0", port_num, app, use_reloader=False, threaded=True)
            else:
                from gevent.pywsgi import WSGIServer

                server = WSGIServer(("0.0.0.0", port_num), app, log=None)
                try:
                    server.serve_forever()
                finally:
                    server.stop(timeout=1.0)
        except Exception as exc:
            print(f"❌ Web server error: {exc}")
            import traceback

            traceback.print_exc()

    if debug_logging:
        print(f"{prefix}Step 2: Starting Flask thread...")
    flask_thread = threading.Thread(target=run_flask, args=(port,), daemon=True)
    flask_thread.start()
    print(f"{prefix}✅ Flask thread started")
    if debug_logging:
        print("✅ Web server started successfully")
        print(f"{prefix}Step 3: Waiting for Flask to initialize...")
    else:
        print("⏳ Waiting for web server to initialize...")

    time.sleep(wait_seconds)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result == 0:
            print(f"✅ Web server confirmed running on port {port}")
        else:
            print(f"⚠️ Web server may not be running on port {port}")
    except Exception as exc:
        print(f"⚠️ Could not verify web server: {exc}")

    if open_browser:
        try:
            import webbrowser

            url = f"http://localhost:{port}"
            print(f"🌐 Opening browser to: {url}")
            webbrowser.open(url)
            print("✅ Browser opened successfully")
        except Exception as exc:
            print(f"⚠️ Could not open browser automatically: {exc}")
            print(f"🌐 Please manually navigate to: http://localhost:{port}")

    return flask_thread, port


async def run_integrated_mode(header_lines: Iterable[str], wait_time: float, **web_config: object) -> None:
    for line in header_lines:
        print(line)

    print("Step 1: Starting web server...")
    web_thread, _ = start_web_server(**web_config)

    if web_thread is None:
        print("❌ Web server failed to start, but continuing with bot...")
    else:
        print("✅ Web server started successfully")

    print("Step 2: Waiting for web server to initialize...")
    await asyncio.sleep(wait_time)

    try:
        print("Step 3: Starting trading bot...")
        from crypto_bot.main import _main_impl

        print("🎯 Starting trading bot with integrated monitoring...")
        print("-" * 60)
        await _main_impl()
        print("✅ Bot completed successfully")
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as exc:
        print(f"❌ Bot error: {exc}")
        import traceback

        traceback.print_exc()

    print("🛑 Shutting down integrated system...")
    print("✅ Shutdown complete")


async def run_auto_mode() -> None:
    print("🚀 Starting LegacyCoinTrader - Integrated Edition")
    print("=" * 60)
    print("🤖 Trading Bot + 📊 Monitoring Dashboard + 🌐 Web Server + 📈 OHLCV Fetching")
    print("=" * 60)

    print("Step 1: Initializing OHLCV data cache...")
    try:
        from crypto_bot.utils.market_loader import load_kraken_symbols, update_multi_tf_ohlcv_cache
        from dotenv import dotenv_values
        import ccxt

        secrets = dotenv_values(".env") or dotenv_values("crypto_bot/.env")
        os.environ.update(secrets)

        if secrets.get("KRAKEN_API_KEY") and not os.environ.get("API_KEY"):
            os.environ["API_KEY"] = secrets["KRAKEN_API_KEY"]
        if secrets.get("KRAKEN_API_SECRET") and not os.environ.get("API_SECRET"):
            os.environ["API_SECRET"] = secrets["KRAKEN_API_SECRET"]

        exchange = ccxt.kraken(
            {
                "apiKey": os.environ.get("API_KEY") or secrets.get("KRAKEN_API_KEY"),
                "secret": os.environ.get("API_SECRET") or secrets.get("KRAKEN_API_SECRET"),
            }
        )

        symbols = await load_kraken_symbols(exchange, [], {})
        if symbols:
            print(f"✅ Found {len(symbols)} trading symbols")
            print("Initializing OHLCV cache for top symbols...")
            cache_config = {
                "timeframes": ["5m", "1h"],
                "ohlcv_timeout": 120,
                "max_ohlcv_failures": 3,
                "production_mode": True,
                "symbol_validation": {
                    "filter_invalid_symbols": True,
                    "min_liquidity_score": 0.6,
                    "min_volume_usd": 10000,
                    "strict_mode": True,
                },
            }
            await update_multi_tf_ohlcv_cache(exchange, symbols[:20], cache_config)
            print("✅ OHLCV cache initialized successfully")
        else:
            print("⚠️ No symbols found, cache initialization skipped")
    except Exception as exc:
        print(f"⚠️ OHLCV cache initialization failed (continuing): {exc}")
        import traceback

        traceback.print_exc()

    print("Step 2: Starting web dashboard server...")
    web_thread, _ = start_web_server()

    if web_thread is None:
        print("❌ Web server failed to start, but continuing with bot...")
    else:
        print("✅ Web server started successfully")

    print("Step 3: Waiting for web server to fully initialize...")
    await asyncio.sleep(5)

    try:
        print("Step 4: Starting trading bot with integrated OHLCV fetching...")
        from crypto_bot.main import _main_impl

        print("🎯 Starting trading bot with integrated monitoring...")
        print("📊 OHLCV fetching will run continuously as part of trading cycles")
        print("-" * 60)
        await _main_impl()
        print("✅ Bot completed successfully")
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as exc:
        print(f"❌ Bot error: {exc}")
        import traceback

        traceback.print_exc()

    print("🛑 Shutting down integrated system...")
    print("✅ Shutdown complete")


async def run_trading_bot_only(
    header_lines: Iterable[str], *, start_message: str = "🎯 Starting trading bot..."
) -> None:
    for line in header_lines:
        print(line)

    try:
        from crypto_bot.main import _main_impl

        print(start_message)
        print("-" * 60)
        await _main_impl()
        print("✅ Bot completed successfully")
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as exc:
        print(f"❌ Bot error: {exc}")
        import traceback

        traceback.print_exc()


class InteractiveBotLauncher:
    """Launches the bot with interactive shutdown capabilities."""

    def __init__(self) -> None:
        self.project_root = PROJECT_ROOT
        self.bot_process: Optional[subprocess.Popen[str]] = None
        self.shutdown_requested = False
        self.input_thread: Optional[threading.Thread] = None

    def setup_signal_handlers(self) -> None:
        def signal_handler(signum: int, _frame: Optional[object]) -> None:
            signal_name = signal.Signals(signum).name
            print(f"\n📡 Received {signal_name} signal")
            self.request_shutdown(f"Signal: {signal_name}")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal_handler)

    def start_input_monitor(self) -> None:
        def input_monitor() -> None:
            try:
                while not self.shutdown_requested:
                    try:
                        user_input = input().strip().lower()
                        if not user_input:
                            print("🛑 Enter key detected - requesting shutdown...")
                            self.request_shutdown("Enter key")
                            break
                        if user_input in {"quit", "exit", "stop", "shutdown"}:
                            print(f"🛑 Command '{user_input}' detected - requesting shutdown...")
                            self.request_shutdown(f"Command: {user_input}")
                            break
                        if user_input in {"help", "h", "?"}:
                            self.show_help()
                        elif user_input == "status":
                            self.show_status()
                        else:
                            print(f"❓ Unknown command: {user_input}")
                            print("💡 Press Enter to shutdown, or type 'help' for commands")
                    except EOFError:
                        break
            except Exception as exc:
                logging.getLogger(__name__).error(f"Input monitor error: {exc}")

        self.input_thread = threading.Thread(target=input_monitor, daemon=True)
        self.input_thread.start()

    def show_help(self) -> None:
        print("\n📖 Available commands:")
        print("  <Enter>           - Safe shutdown")
        print("  quit, exit, stop  - Safe shutdown")
        print("  status            - Show bot status")
        print("  help              - Show this help")
        print("  Ctrl+C            - Emergency shutdown")
        print()

    def show_status(self) -> None:
        if self.bot_process:
            if self.bot_process.poll() is None:
                print(f"🟢 Bot Status: Running (PID: {self.bot_process.pid})")
            else:
                print(
                    f"🔴 Bot Status: Stopped (Exit code: {self.bot_process.returncode})"
                )
        else:
            print("🔴 Bot Status: Not started")

    def request_shutdown(self, reason: str) -> None:
        if self.shutdown_requested:
            return

        self.shutdown_requested = True
        print(f"🛑 Shutdown requested: {reason}")
        print("🔄 Initiating safe shutdown...")

        if self.bot_process and self.bot_process.poll() is None:
            try:
                print("📤 Sending SIGTERM to bot process...")
                self.bot_process.terminate()
                try:
                    self.bot_process.wait(timeout=10)
                    print("✅ Bot stopped gracefully")
                except subprocess.TimeoutExpired:
                    print("⏰ Graceful shutdown timeout, force killing...")
                    self.bot_process.kill()
                    self.bot_process.wait()
                    print("💀 Bot force killed")
            except Exception as exc:
                logging.getLogger(__name__).error(f"Error shutting down bot: {exc}")

        pid_file = self.project_root / "bot_pid.txt"
        if pid_file.exists():
            try:
                pid_file.unlink()
                print("🧹 PID file cleaned up")
            except Exception as exc:
                logging.getLogger(__name__).warning(f"Failed to clean PID file: {exc}")

    def start_bot(self) -> bool:
        print("🚀 Starting LegacyCoinTrader bot...")

        python_cmd = sys.executable
        bot_script = self.project_root / "crypto_bot" / "main.py"
        if not bot_script.exists():
            print(f"❌ Bot script not found: {bot_script}")
            return False

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "crypto_bot")
            self.bot_process = subprocess.Popen(
                [python_cmd, str(bot_script)],
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            print(f"✅ Bot started (PID: {self.bot_process.pid})")
            return True
        except Exception as exc:
            print(f"❌ Failed to start bot: {exc}")
            return False

    def monitor_bot_output(self) -> None:
        if not self.bot_process or not self.bot_process.stdout:
            return
        try:
            for line in iter(self.bot_process.stdout.readline, ""):
                if self.shutdown_requested:
                    break
                print(line.rstrip())
        except Exception as exc:
            logging.getLogger(__name__).error(f"Error reading bot output: {exc}")

    def run(self) -> int:
        print("🎮 LegacyCoinTrader Interactive Launcher")
        print("=" * 50)
        print("💡 Interactive controls:")
        print("   • Press Ctrl+C for emergency shutdown")
        print("   • Press Enter for safe shutdown")
        print("   • Type 'help' for more commands")
        print("=" * 50)

        self.setup_signal_handlers()

        if not self.start_bot():
            return 1

        self.start_input_monitor()

        try:
            self.monitor_bot_output()
            if self.bot_process:
                exit_code = self.bot_process.wait()
                print(f"🏁 Bot exited with code: {exit_code}")
                return exit_code
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C detected")
            self.request_shutdown("Ctrl+C")
        except Exception as exc:
            logging.getLogger(__name__).error(f"Error in main loop: {exc}")
            self.request_shutdown(f"Error: {exc}")
        finally:
            if self.bot_process and self.bot_process.poll() is None:
                self.request_shutdown("Cleanup")

        print("👋 Interactive launcher finished")
        return 0


def run_async(coro: Callable[[], Awaitable[None]], keyboard_msg: str = "\n👋 Goodbye!") -> None:
    try:
        asyncio.run(coro())
    except KeyboardInterrupt:
        print(keyboard_msg)
    except Exception as exc:
        print(f"💥 Fatal error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def handle_auto(_: argparse.Namespace) -> None:
    set_environment(
        auto_start=True,
        non_interactive=True,
        extra={
            "PRODUCTION": "true",
            "FLASK_ENV": "production",
            "LOG_LEVEL": "INFO",
            "ENABLE_METRICS": "true",
            "ENABLE_POSITION_SYNC": "true",
            "ENABLE_MEMORY_MANAGEMENT": "true",
        },
    )
    print("🚀 Starting LegacyCoinTrader in PRODUCTION MODE")
    print("=" * 60)
    print("📊 Production Features Enabled:")
    print("  • Enhanced Symbol Validation")
    print("  • Production Memory Management")
    print("  • Position Synchronization")
    print("  • Circuit Breaker Protection")
    print("  • Production Monitoring")
    print("=" * 60)
    run_async(run_auto_mode)


def handle_clean(_: argparse.Namespace) -> None:
    set_environment(auto_start=True, non_interactive=True)

    async def _runner() -> None:
        print("🚀 Starting LegacyCoinTrader - CLEAN STATE")
        print("=" * 60)
        print("🤖 Trading Bot with FORCED CLEAN STATE")
        print("=" * 60)
        force_clean_state()
        await run_trading_bot_only(
            [], start_message="🎯 Starting trading bot with clean state..."
        )

    run_async(_runner)


def handle_debug(_: argparse.Namespace) -> None:
    set_environment(auto_start=True, non_interactive=True)
    run_async(
        lambda: run_integrated_mode(
            [
                "🚀 Starting LegacyCoinTrader - Integrated Edition (DEBUG)",
                "=" * 60,
                "🤖 Trading Bot + 📊 Monitoring Dashboard + 🌐 Web Server",
                "=" * 60,
            ],
            wait_time=2,
            debug_logging=True,
            verify_main_import=True,
        )
    )


def handle_direct(_: argparse.Namespace) -> None:
    set_environment(auto_start=True, non_interactive=True)
    run_async(
        lambda: run_trading_bot_only(
            [
                "🚀 Starting LegacyCoinTrader - Trading Bot Only",
                "=" * 60,
                "🤖 Trading Bot with TradeManager as Single Source of Truth",
                "=" * 60,
            ]
        )
    )


def handle_final(_: argparse.Namespace) -> None:
    set_environment(auto_start=True, non_interactive=True)
    run_async(
        lambda: run_integrated_mode(
            [
                "🚀 Starting LegacyCoinTrader - Integrated Edition",
                "=" * 60,
                "🤖 Trading Bot + 📊 Monitoring Dashboard + 🌐 Web Server",
                "=" * 60,
            ],
            wait_time=3,
            use_run_simple=True,
        )
    )


def handle_fixed(_: argparse.Namespace) -> None:
    set_environment(auto_start=True, non_interactive=True)
    run_async(
        lambda: run_integrated_mode(
            [
                "🚀 Starting LegacyCoinTrader - Integrated Edition",
                "=" * 60,
                "🤖 Trading Bot + 📊 Monitoring Dashboard + 🌐 Web Server",
                "=" * 60,
            ],
            wait_time=3,
            use_run_simple=True,
        )
    )


def handle_interactive(_: argparse.Namespace) -> None:
    launcher = InteractiveBotLauncher()
    sys.exit(launcher.run())


def handle_isolated(_: argparse.Namespace) -> None:
    set_environment(auto_start=True, non_interactive=True)

    async def _runner() -> None:
        print("🚀 Starting LegacyCoinTrader - ISOLATED PAPER WALLET")
        print("=" * 60)
        print("🤖 Trading Bot with ISOLATED PAPER WALLET")
        print("=" * 60)
        force_clean_state()
        await run_trading_bot_only(
            [], start_message="🎯 Starting trading bot with isolated paper wallet..."
        )

    run_async(_runner)


def handle_noninteractive(_: argparse.Namespace) -> None:
    os.environ["NON_INTERACTIVE"] = "1"
    try:
        from crypto_bot.main import main as bot_main

        asyncio.run(bot_main())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as exc:
        print(f"Bot error: {exc}")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LegacyCoinTrader unified startup CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("auto", help="Start the integrated production bot").set_defaults(func=handle_auto)
    subparsers.add_parser("clean", help="Force a clean state and start the bot").set_defaults(func=handle_clean)
    subparsers.add_parser("debug", help="Start the integrated bot with debug logging").set_defaults(func=handle_debug)
    subparsers.add_parser("direct", help="Run the trading bot without web server").set_defaults(func=handle_direct)
    subparsers.add_parser("final", help="Start the integrated bot with production web server").set_defaults(func=handle_final)
    subparsers.add_parser("fixed", help="Start the integrated bot using Werkzeug server").set_defaults(func=handle_fixed)
    subparsers.add_parser("interactive", help="Launch the bot with interactive controls").set_defaults(func=handle_interactive)
    subparsers.add_parser("isolated", help="Start the bot with isolated paper wallet state").set_defaults(func=handle_isolated)
    subparsers.add_parser("noninteractive", help="Start the bot in non-interactive mode").set_defaults(func=handle_noninteractive)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
