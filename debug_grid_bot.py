#!/usr/bin/env python3
"""
Debug grid_bot to see why it's returning 'none'.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.main import load_config
from crypto_bot.strategy.grid_bot import generate_signal as grid_signal
from crypto_bot.execution.cex_executor import get_exchange
from crypto_bot.utils.market_loader import update_multi_tf_ohlcv_cache
from crypto_bot.phase_runner import BotContext

async def debug_grid_bot():
    """Debug why grid_bot returns 'none'."""

    print("🔧 Debugging Grid Bot")
    print("=" * 50)

    try:
        # Load configuration
        print("\n📄 Loading configuration...")
        config = load_config()
        print(f"✅ Config loaded. Mode: {config.get('mode', 'cex')}")

        # Get real exchange connection
        print("\n🔗 Creating real exchange connection...")
        exchange, ws_client = get_exchange(config)
        print(f"✅ Connected to {exchange.id}")

        # Load markets
        print("\n📊 Loading exchange markets...")
        try:
            if asyncio.iscoroutinefunction(exchange.load_markets):
                await exchange.load_markets()
            else:
                exchange.load_markets()
            print(f"✅ Loaded {len(exchange.symbols)} markets")
        except Exception as e:
            print(f"❌ Failed to load markets: {e}")
            return

        # Create bot context
        print("\n🔧 Creating bot context...")
        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config=config
        )
        ctx.exchange = exchange

        # Test symbols
        test_symbols = ['BTC/USD']
        print(f"\n🎯 Testing with symbols: {test_symbols}")

        # Fetch real data
        print("\n📡 Fetching real data...")
        await update_multi_tf_ohlcv_cache(
            exchange=ctx.exchange,
            cache=ctx.df_cache,
            symbols=test_symbols,
            config=ctx.config,
            limit=100
        )

        print("✅ Data fetch completed")

        # Check data
        for symbol in test_symbols:
            if '1h' in ctx.df_cache and symbol in ctx.df_cache['1h']:
                df = ctx.df_cache['1h'][symbol]
                print(f"\n📊 {symbol} data ({len(df)} candles):")
                print(f"   Latest close: {df['close'].iloc[-1]}")
                print(f"   Latest volume: {df['volume'].iloc[-1] if 'volume' in df.columns else 'N/A'}")

                # Check volume conditions
                if 'volume' in df.columns:
                    vol_sma = df['volume'].rolling(20).mean()
                    vol_zscore = (df['volume'] - vol_sma) / df['volume'].rolling(20).std()
                    latest_vol = df['volume'].iloc[-1]
                    latest_vol_sma = vol_sma.iloc[-1]
                    latest_vol_zscore = vol_zscore.iloc[-1]

                    print(f"   Volume SMA (20): {latest_vol_sma}")
                    print(f"   Latest volume: {latest_vol}")
                    print(f"   Volume ratio: {latest_vol / latest_vol_sma if latest_vol_sma > 0 else 'N/A'}")
                    print(f"   Volume z-score: {latest_vol_zscore}")

                    # Check grid_bot volume conditions
                    volume_multiple = 1.5  # default
                    vol_zscore_threshold = 2.0  # default

                    volume_ok = (latest_vol > latest_vol_sma * volume_multiple and
                               latest_vol_zscore < vol_zscore_threshold)

                    print(f"   Volume OK: {volume_ok}")
                    print(f"     Condition 1 (vol > sma * {volume_multiple}): {latest_vol > latest_vol_sma * volume_multiple}")
                    print(f"     Condition 2 (zscore < {vol_zscore_threshold}): {latest_vol_zscore < vol_zscore_threshold}")

        # Test grid_bot with debug info
        print("\n🧪 Testing grid_bot...")

        for symbol in test_symbols:
            if '1h' in ctx.df_cache and symbol in ctx.df_cache['1h']:
                df = ctx.df_cache['1h'][symbol]
                print(f"\n   Testing {symbol}...")

                # Test with default config
                try:
                    # Let's add some debug prints to the grid_bot function temporarily
                    print("   🔧 Testing grid_step calculation...")

                    # Manually calculate what grid_bot does
                    from crypto_bot.strategy.grid_bot import GridConfig, _as_dict

                    cfg = GridConfig.from_dict(_as_dict(config))
                    price = df["close"].iloc[-1]
                    range_window = cfg.range_window
                    range_slice = df.tail(range_window)
                    high = range_slice["high"].max()
                    low = range_slice["low"].min()

                    print(f"     Price: {price}")
                    print(f"     Range: {high} - {low} = {high - low}")

                    # Calculate ATR percent
                    from crypto_bot.utils.volatility import atr_percent
                    atr_pct_1h = atr_percent(df, cfg.atr_period)
                    print(f"     ATR %: {atr_pct_1h}")

                    spacing_pct = max(0.3, 1.2 * atr_pct_1h)
                    print(f"     Spacing %: {spacing_pct}")

                    grid_step = price * spacing_pct / 100
                    print(f"     Grid step: {grid_step}")
                    print(f"     Grid step truthy: {bool(grid_step)}")

                    # Let's debug the grid bounds and position
                    print("   🔧 Debugging grid bounds and price position...")

                    # Calculate what grid_bot calculates internally
                    import numpy as np

                    # Recalculate what grid_bot does
                    num_levels = 6  # default
                    n = num_levels // 2  # 3

                    # Calculate centre (similar to grid_bot logic)
                    centre = (high + low) / 2  # fallback centre

                    # Use the grid_step we calculated earlier
                    levels = centre + np.arange(-n, n + 1) * grid_step
                    lower_bound = levels[1]  # Second level (index 1)
                    upper_bound = levels[-2]  # Second-to-last level (index -2)

                    print(f"     Centre: {centre}")
                    print(f"     Grid step: {grid_step}")
                    print(f"     Levels: {levels}")
                    print(f"     Lower bound: {lower_bound}")
                    print(f"     Upper bound: {upper_bound}")
                    print(f"     Price: {price}")
                    print(f"     Price <= lower_bound: {price <= lower_bound}")
                    print(f"     Price >= upper_bound: {price >= upper_bound}")
                    print(f"     Price in grid: {lower_bound < price < upper_bound}")

                    # Debug trend check
                    print("   🔧 Debugging trend check...")

                    # Check what 'recent' data looks like (what grid_bot actually uses)
                    range_window = 20  # default
                    atr_period = 14  # default
                    volume_ma_window = 20  # default
                    recent_len = max(range_window, atr_period, volume_ma_window)
                    recent = df.tail(recent_len)

                    print(f"     Full df length: {len(df)}")
                    print(f"     Recent length: {len(recent)}")
                    print(f"     Recent start price: {recent['close'].iloc[0]}")
                    print(f"     Recent end price: {recent['close'].iloc[-1]}")

                    # Test trend on recent data
                    fast_ema_recent = recent["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                    slow_ema_recent = recent["close"].ewm(span=200, adjust=False).mean().iloc[-1]

                    print(f"     Recent Fast EMA (50): {fast_ema_recent}")
                    print(f"     Recent Slow EMA (200): {slow_ema_recent}")
                    print(f"     Recent Fast > Slow: {fast_ema_recent > slow_ema_recent}")

                    # Also test on full data for comparison
                    fast_ema_full = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                    slow_ema_full = df["close"].ewm(span=200, adjust=False).mean().iloc[-1]

                    print(f"     Full Fast EMA (50): {fast_ema_full}")
                    print(f"     Full Slow EMA (200): {slow_ema_full}")
                    print(f"     Full Fast > Slow: {fast_ema_full > slow_ema_full}")

                    # Test the is_in_trend function directly
                    from crypto_bot.strategy.grid_bot import is_in_trend
                    trend_result = is_in_trend(recent, 50, 200, "long")
                    print(f"     is_in_trend(recent, 50, 200, 'long'): {trend_result}")

                    score, direction = grid_signal(df, config)
                    print(f"   📊 Grid Bot result: score={score:.3f}, direction='{direction}'")

                    # Additional analysis
                    if direction == 'none' and price <= lower_bound:
                        print("   ⚠️ Price is below grid but no LONG signal - trend check likely failed")
                    elif direction == 'none' and price >= upper_bound:
                        print("   ⚠️ Price is above grid but no SHORT signal - trend check likely failed")

                except Exception as e:
                    print(f"   ❌ Grid Bot failed: {e}")
                    import traceback
                    traceback.print_exc()

                # Test with disabled volume filter
                print("\n   🔧 Testing with volume filter disabled...")
                test_config = config.copy()
                test_config['volume_filter'] = False

                try:
                    score, direction = grid_signal(df, test_config)
                    print(f"   📊 Grid Bot (no volume filter): score={score:.3f}, direction='{direction}'")
                except Exception as e:
                    print(f"   ❌ Grid Bot (no volume filter) failed: {e}")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_grid_bot())
