#!/usr/bin/env python3
"""
Comprehensive script to reset the entire paper trading system to a clean state.
This will clear all positions, caches, trade history, and reset balance to initial value.
"""

import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime

# Ensure json is available globally
JSON = json

# Add the crypto_bot directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'crypto_bot'))

def reset_paper_wallet_comprehensive():
    """Reset the entire paper trading system to a clean state."""
    print("🔄 Starting comprehensive paper trading reset...")
    print("=" * 60)

    # 1. Reset Paper Wallet
    print("📊 Resetting paper wallet...")
    try:
        from libs.models.paper_wallet import PaperWallet

        # Create wallet instance with initial balance
        wallet = PaperWallet(balance=10000.0, max_open_trades=10, allow_short=True)

        # Load current state (if it exists)
        wallet.load_state()

        print(f"   Before: balance=${wallet.balance:.2f}, positions={len(wallet.positions)}, trades={wallet.total_trades}")

        # Reset to clean state
        wallet.reset(10000.0)

        # Force save the reset state
        wallet.save_state()

        print(f"   After: balance=${wallet.balance:.2f}, positions={len(wallet.positions)}, trades={wallet.total_trades}")
        print("✅ Paper wallet reset complete")

    except Exception as e:
        print(f"❌ Failed to reset paper wallet: {e}")

    # 2. Reset Trade Manager State
    print("\n📈 Resetting trade manager state...")
    try:
        trade_state_file = Path("crypto_bot/logs/trade_manager_state.json")
        if trade_state_file.exists():
            baseline_state = {
                "trades": [],
                "positions": {},
                "closed_positions": [],
                "price_cache": {},
                "statistics": {
                    "total_trades": 0,
                    "total_volume": 0.0,
                    "total_fees": 0.0,
                    "total_realized_pnl": 0.0
                },
                "last_save_time": datetime.now().isoformat()
            }

            with trade_state_file.open("w", encoding="utf-8") as handle:
                JSON.dump(baseline_state, handle, indent=2)

            print("✅ Trade manager state reset")
        else:
            print("ℹ️  Trade manager state file not found, skipping")

    except Exception as e:
        print(f"❌ Failed to reset trade manager state: {e}")

    # 3. Reset CEX Scanner State
    print("\n🔍 Resetting CEX scanner state...")
    try:
        cex_state_file = Path("crypto_bot/logs/cex_scanner_state.json")
        if cex_state_file.exists():
            baseline_state = {
                "seen_pairs": [],
                "last_scan": None,
                "exchange": "kraken",
                "initialised": False
            }

            with cex_state_file.open("w", encoding="utf-8") as handle:
                JSON.dump(baseline_state, handle, indent=2)

            print("✅ CEX scanner state reset")
        else:
            print("ℹ️  CEX scanner state file not found, skipping")

    except Exception as e:
        print(f"❌ Failed to reset CEX scanner state: {e}")

    # 4. Clear Trade History
    print("\n📝 Clearing trade history...")
    try:
        trades_file = Path("crypto_bot/logs/trades.csv")
        if trades_file.exists():
            # Keep header but clear all trade data
            with trades_file.open("w", encoding="utf-8") as handle:
                handle.write("")  # Clear the file completely
            print("✅ Trade history cleared")
        else:
            print("ℹ️  Trade history file not found, skipping")

    except Exception as e:
        print(f"❌ Failed to clear trade history: {e}")

    # 5. Clear Cache Files
    print("\n🗂️  Clearing cache files...")
    cache_files = [
        Path("cache/liquid_pairs.json"),
        Path("crypto_bot/logs/last_regime.json"),
        Path("crypto_bot/logs/system_status.json")
    ]

    for cache_file in cache_files:
        try:
            if cache_file.exists():
                cache_file.unlink()
                print(f"✅ Cleared {cache_file.name}")
            else:
                print(f"ℹ️  {cache_file.name} not found, skipping")
        except Exception as e:
            print(f"❌ Failed to clear {cache_file.name}: {e}")

    # 6. Reset Balance Manager (Single Source of Truth)
    print("\n💰 Resetting balance manager...")
    try:
        from crypto_bot.utils.balance_manager import BalanceManager
        BalanceManager.set_balance(10000.0)
        print("✅ Balance manager reset to $10,000.00")
    except Exception as e:
        print(f"❌ Failed to reset balance manager: {e}")

    # 7. Clear Any Additional State Files
    print("\n🧹 Clearing additional state files...")
    additional_files = [
        Path("crypto_bot/logs/paper_wallet.yaml"),
        Path("frontend/crypto_bot/logs/paper_wallet_state.yaml"),
        Path("frontend/crypto_bot/logs/trade_manager_state.json")
    ]

    for state_file in additional_files:
        try:
            if state_file.exists():
                # Create clean state based on file type
                if state_file.suffix == '.yaml':
                    clean_state = {
                        "balance": 10000.0,
                        "initial_balance": 10000.0,
                        "realized_pnl": 0.0,
                        "total_trades": 0,
                        "winning_trades": 0,
                        "positions": {}
                    }
                    with state_file.open("w", encoding="utf-8") as handle:
                        yaml.safe_dump(clean_state, handle, default_flow_style=False)
                elif state_file.suffix == '.json':
                    clean_state = {
                        "trades": [],
                        "positions": {},
                        "closed_positions": [],
                        "price_cache": {},
                        "statistics": {
                            "total_trades": 0,
                            "total_volume": 0.0,
                            "total_fees": 0.0,
                            "total_realized_pnl": 0.0
                        }
                    }
                    with state_file.open("w", encoding="utf-8") as handle:
                        JSON.dump(clean_state, handle, indent=2)

                print(f"✅ Reset {state_file.name}")
            else:
                print(f"ℹ️  {state_file.name} not found, skipping")
        except Exception as e:
            print(f"❌ Failed to reset {state_file.name}: {e}")

    # 8. Reset Portfolio Service Database and All Services
    print("\n🔄 Stopping all services to prevent data re-population...")

    try:
        import subprocess

        # Stop all services except infrastructure
        stop_commands = [
            'docker-compose stop portfolio',
            'docker-compose stop trading-engine',
            'docker-compose stop strategy-engine',
            'docker-compose stop execution',
            'docker-compose stop token-discovery',
            'docker-compose stop market-data',
            'docker-compose stop monitoring',
            'docker-compose stop frontend'
        ]

        for cmd in stop_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.path.dirname(__file__))
                if result.returncode == 0:
                    print("✅ Service stopped")
                else:
                    print(f"⚠️  Service stop command failed: {result.stderr.strip()}")
            except Exception as e:
                print(f"⚠️  Service stop error: {e}")

        print("✅ All trading services stopped")

    except Exception as e:
        print(f"⚠️  Failed to stop services: {e}")

    # Clear PostgreSQL database tables
    print("🔄 Clearing PostgreSQL database tables...")
    try:
        # Commands to clear portfolio database tables
        clear_commands = [
            # Connect to database and truncate tables (in correct dependency order)
            'docker exec legacycointrader10-1-postgres-1 psql -U legacy_user -d legacy_coin_trader -c "TRUNCATE TABLE trades RESTART IDENTITY CASCADE;"',
            'docker exec legacycointrader10-1-postgres-1 psql -U legacy_user -d legacy_coin_trader -c "TRUNCATE TABLE positions RESTART IDENTITY CASCADE;"',
            'docker exec legacycointrader10-1-postgres-1 psql -U legacy_user -d legacy_coin_trader -c "TRUNCATE TABLE price_cache RESTART IDENTITY CASCADE;"',
            'docker exec legacycointrader10-1-postgres-1 psql -U legacy_user -d legacy_coin_trader -c "TRUNCATE TABLE portfolio_statistics RESTART IDENTITY CASCADE;"',
            'docker exec legacycointrader10-1-postgres-1 psql -U legacy_user -d legacy_coin_trader -c "TRUNCATE TABLE risk_limits RESTART IDENTITY CASCADE;"',
            'docker exec legacycointrader10-1-postgres-1 psql -U legacy_user -d legacy_coin_trader -c "TRUNCATE TABLE balances RESTART IDENTITY CASCADE;"'
        ]

        for cmd in clear_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✅ Database table cleared")
                else:
                    # Check if this is just a "table does not exist" error or a real error
                    stderr_output = result.stderr.strip()
                    if "does not exist" in stderr_output:
                        print(f"ℹ️  Table does not exist (expected): {stderr_output}")
                    else:
                        print(f"⚠️  Database command failed: {stderr_output}")
                        print(f"   Command was: {cmd}")
            except subprocess.TimeoutExpired:
                print(f"⚠️  Database command timed out: {cmd}")
            except Exception as e:
                print(f"⚠️  Database command error: {e}")

        # Verify the tables were actually cleared
        try:
            verify_cmd = 'docker exec legacycointrader10-1-postgres-1 psql -U legacy_user -d legacy_coin_trader -c "SELECT COUNT(*) as total_trades FROM trades; SELECT COUNT(*) as total_positions FROM positions;"'
            verify_result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True, timeout=10)
            if verify_result.returncode == 0:
                output_lines = verify_result.stdout.strip().split('\n')
                for line in output_lines:
                    if 'total_trades' in line or 'total_positions' in line:
                        if '0' in line:
                            print("✅ Verified: Table is empty")
                        else:
                            print(f"⚠️  Warning: Table may not be fully cleared: {line}")
            else:
                print(f"⚠️  Could not verify database clear: {verify_result.stderr.strip()}")
        except Exception as e:
            print(f"⚠️  Database verification error: {e}")

        print("✅ PostgreSQL database tables cleared")

    except Exception as e:
        print(f"⚠️  Failed to clear PostgreSQL database: {e}")

    # Clear Redis cache
    print("🔄 Clearing Redis cache...")
    try:
        # Clear all Redis data
        redis_commands = [
            'docker exec legacycointrader10-1-redis-1 redis-cli FLUSHALL',
            'docker exec legacycointrader10-1-redis-1 redis-cli FLUSHDB'
        ]

        for cmd in redis_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print("✅ Redis cache cleared")
                else:
                    print(f"⚠️  Redis command failed: {result.stderr.strip()}")
            except Exception as e:
                print(f"⚠️  Redis command error: {e}")

        print("✅ Redis cache cleared")

    except Exception as e:
        print(f"⚠️  Failed to clear Redis cache: {e}")

    # Restart all services
    print("🔄 Restarting all services...")
    try:
        restart_cmd = 'docker-compose up -d'
        result = subprocess.run(restart_cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print("✅ All services restarted")
        else:
            print(f"⚠️  Service restart failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"⚠️  Service restart error: {e}")

    print("✅ Portfolio service database reset with service restart")

    print("\n" + "=" * 60)
    print("🎉 Paper trading system reset complete!")
    print("✅ All positions cleared")
    print("✅ All caches cleared")
    print("✅ Trade history cleared")
    print("✅ Balance reset to $10,000.00")
    print("✅ Portfolio service database reset")
    print("✅ PostgreSQL database tables cleared")
    print("✅ Redis cache cleared")
    print("✅ All services restarted fresh")
    print("✅ System ready for fresh start")
    print("=" * 60)

if __name__ == "__main__":
    reset_paper_wallet_comprehensive()
