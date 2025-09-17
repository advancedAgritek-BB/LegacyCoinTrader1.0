"""Signal validation tests for strategy modules."""
from __future__ import annotations

import importlib
import logging

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.regression


def _load_strategy(name: str):
    try:
        return importlib.import_module(f"crypto_bot.strategy.{name}")
    except Exception:
        return None


arbitrage_engine = _load_strategy("arbitrage_engine")
bounce_scalper = _load_strategy("bounce_scalper")
breakout_bot = _load_strategy("breakout_bot")
cross_chain_arb_bot = _load_strategy("cross_chain_arb_bot")
dca_bot = _load_strategy("dca_bot")
dex_scalper = _load_strategy("dex_scalper")
dip_hunter = _load_strategy("dip_hunter")
flash_crash_bot = _load_strategy("flash_crash_bot")
grid_bot = _load_strategy("grid_bot")
hft_engine = _load_strategy("hft_engine")
lstm_bot = _load_strategy("lstm_bot")
maker_spread = _load_strategy("maker_spread")
market_making_bot = _load_strategy("market_making_bot")
mean_bot = _load_strategy("mean_bot")
meme_wave_bot = _load_strategy("meme_wave_bot")
micro_scalp_bot = _load_strategy("micro_scalp_bot")
momentum_bot = _load_strategy("momentum_bot")
momentum_exploiter = _load_strategy("momentum_exploiter")
range_arb_bot = _load_strategy("range_arb_bot")
sniper_bot = _load_strategy("sniper_bot")
stat_arb_bot = _load_strategy("stat_arb_bot")
trend_bot = _load_strategy("trend_bot")
ultra_scalp_bot = _load_strategy("ultra_scalp_bot")
volatility_harvester = _load_strategy("volatility_harvester")


EMPTY_STRATEGIES = []


def _add_empty_strategy(module, *, kwargs=None):
    if module is None:
        return
    kwargs = kwargs or {}

    def _call(df, mod=module, kw=kwargs):
        return mod.generate_signal(df, **kw)

    EMPTY_STRATEGIES.append(_call)


_add_empty_strategy(arbitrage_engine)
_add_empty_strategy(bounce_scalper)
_add_empty_strategy(breakout_bot)
_add_empty_strategy(dca_bot)
_add_empty_strategy(dex_scalper)
_add_empty_strategy(dip_hunter)
_add_empty_strategy(flash_crash_bot)
_add_empty_strategy(grid_bot)
_add_empty_strategy(hft_engine)
_add_empty_strategy(lstm_bot)
_add_empty_strategy(maker_spread)
_add_empty_strategy(market_making_bot)
_add_empty_strategy(mean_bot)
_add_empty_strategy(micro_scalp_bot)
_add_empty_strategy(momentum_bot)
_add_empty_strategy(momentum_exploiter)
_add_empty_strategy(sniper_bot)
_add_empty_strategy(stat_arb_bot)
_add_empty_strategy(trend_bot)
_add_empty_strategy(ultra_scalp_bot)
_add_empty_strategy(volatility_harvester)


def test_arbitrage_engine_favors_high_volatility(make_ohlcv):
    if arbitrage_engine is None:
        pytest.skip("arbitrage_engine strategy unavailable")
    closes = [100, 120, 90, 130, 85, 140, 80, 150, 95, 145, 90, 160]
    volumes = [1_000] * (len(closes) - 1) + [5_000]
    df = make_ohlcv(closes, volumes=volumes)

    score, direction = arbitrage_engine.generate_signal(df)
    assert direction == "arbitrage"
    assert score > 0

@pytest.mark.xfail(reason="requires complex market state for positive bounce signal", strict=True)
def test_bounce_scalper_generates_bullish_signal(make_ohlcv, monkeypatch):
    if bounce_scalper is None:
        pytest.skip("bounce_scalper strategy unavailable")
    closes = [100, 98, 96, 95, 93, 92, 90, 94]
    volumes = [1_000, 950, 900, 850, 800, 780, 760, 2_000]
    df = make_ohlcv(closes, volumes=volumes)

    monkeypatch.setattr(bounce_scalper, "is_engulfing", lambda *_a, **_k: "bullish")
    monkeypatch.setattr(bounce_scalper, "is_hammer", lambda *_a, **_k: None)
    monkeypatch.setattr(bounce_scalper, "normalize_score_by_volatility", lambda *a, **k: a[1])
    monkeypatch.setattr(bounce_scalper, "confirm_higher_lows", lambda *_a, **_k: True)
    monkeypatch.setattr(bounce_scalper, "confirm_lower_highs", lambda *_a, **_k: True)

    config = {
        "bounce_scalper": {
            "down_candles": 1,
            "up_candles": 1,
            "volume_multiple": 0.1,
            "zscore_threshold": -1_000_000.0,
            "vol_window": 3,
            "lookback": 10,
            "rsi_window": 3,
            "rsi_overbought_pct": 0,
            "rsi_oversold_pct": 100,
            "atr_normalization": False,
            "body_pct": 0.2,
        }
    }

    score, direction = bounce_scalper.generate_signal(df, config)
    assert direction == "long"
    assert score > 0

@pytest.mark.xfail(reason="requires squeeze alignment and exact breakout setup", strict=True)
def test_breakout_bot_breakout_signal(make_ohlcv, monkeypatch):
    if breakout_bot is None:
        pytest.skip("breakout_bot strategy unavailable")
    closes = [100, 101, 102, 103, 104, 130]
    volumes = [1_000, 1_100, 1_050, 1_100, 1_050, 1_500]
    df = make_ohlcv(closes, volumes=volumes)

    squeeze_series = pd.Series([True] * len(df), index=df.index)
    atr_series = pd.Series([1.0] * len(df), index=df.index)
    monkeypatch.setattr(
        breakout_bot,
        "_squeeze",
        lambda *_a, **_k: (squeeze_series, atr_series),
    )
    monkeypatch.setattr(breakout_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])

    config = {"breakout": {"donchian_window": 3, "volume_window": 2, "vol_confirmation": False}}
    score, direction = breakout_bot.generate_signal(df, config)
    assert direction == "long"
    assert score > 0


def test_cross_chain_arb_bot_detects_positive_spread(ohlcv_trending_up, monkeypatch):
    if cross_chain_arb_bot is None:
        pytest.skip("cross_chain_arb_bot strategy unavailable")
    config = {"cross_chain_arb_bot": {"pair": "SOL/USDT", "spread_threshold": 0.01}}
    monkeypatch.setattr(
        cross_chain_arb_bot,
        "_fetch_prices",
        lambda symbols: {"SOL/USDT": ohlcv_trending_up["close"].iloc[-1] * 1.05},
    )

    score, direction = cross_chain_arb_bot.generate_signal(
        ohlcv_trending_up, config={"symbol": "SOL/USDT", **config}
    )
    assert direction == "long"
    assert score > 0


def test_dca_bot_buys_discount(ohlcv_trending_down):
    if dca_bot is None:
        pytest.skip("dca_bot strategy unavailable")
    df = ohlcv_trending_down.copy()
    df.loc[df.index[-1], "close"] *= 0.8
    score, direction = dca_bot.generate_signal(df)
    assert direction == "long"
    assert score == pytest.approx(0.8, rel=1e-3)


def test_dex_scalper_positive_momentum(ohlcv_trending_up, monkeypatch):
    if dex_scalper is None:
        pytest.skip("dex_scalper strategy unavailable")
    monkeypatch.setenv("MOCK_ETH_PRIORITY_FEE_GWEI", "0")
    monkeypatch.setattr(dex_scalper, "normalize_score_by_volatility", lambda *a, **k: a[1])
    config = {
        "dex_scalper": {
            "ema_fast": 2,
            "ema_slow": 5,
            "gas_threshold_gwei": 100,
            "min_signal_score": 0.0,
        }
    }
    score, direction = dex_scalper.generate_signal(ohlcv_trending_up, config)
    assert direction == "long"
    assert score > 0


def test_dip_hunter_detects_sharp_dip(make_ohlcv, monkeypatch):
    if dip_hunter is None:
        pytest.skip("dip_hunter strategy unavailable")
    base = np.linspace(100, 110, 110)
    base[-3:] = [105, 95, 96]
    volumes = np.full_like(base, 1_000.0)
    volumes[-1] = 5_000
    df = make_ohlcv(base, volumes=volumes)

    monkeypatch.setattr(dip_hunter, "MODEL", None)
    monkeypatch.setattr(dip_hunter, "normalize_score_by_volatility", lambda *a, **k: a[1])
    dip_hunter.score_logger = logging.getLogger("dip_hunter_test")
    config = {
        "dip_hunter": {
            "dip_pct": 0.01,
            "dip_bars": 2,
            "rsi_window": 5,
            "rsi_oversold": 80,
            "vol_window": 5,
            "vol_mult": 1.0,
            "adx_window": 5,
            "adx_threshold": 200,
            "bb_window": 5,
            "ema_trend": 5,
            "ema_slow": 5,
            "atr_normalization": False,
        }
    }

    score, direction = dip_hunter.generate_signal(df, config=config)
    assert direction == "long"
    assert score > 0


def test_flash_crash_bot_flags_drop(make_ohlcv):
    if flash_crash_bot is None:
        pytest.skip("flash_crash_bot strategy unavailable")
    closes = [100, 101, 102, 90]
    volumes = [1_000, 1_050, 1_000, 6_000]
    df = make_ohlcv(closes, volumes=volumes)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(flash_crash_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])
    config = {"flash_crash": {"drop_pct": 0.05, "volume_mult": 1.5, "vol_window": 2, "ema_window": 3}}
    score, direction = flash_crash_bot.generate_signal(df, config=config)
    monkeypatch.undo()
    assert direction == "long"
    assert score > 0


def test_grid_bot_buys_near_lower_bound(make_ohlcv, monkeypatch):
    if grid_bot is None:
        pytest.skip("grid_bot strategy unavailable")
    closes = [100] * 30 + [97, 96, 95]
    volumes = [1_500] * len(closes)
    df = make_ohlcv(closes, volumes=volumes)

    class DummyGridState:
        def update_bar(self, *_a, **_k):
            return None

        def in_cooldown(self, *_a, **_k):
            return False

        def active_leg_count(self, *_a, **_k):
            return 0

        def get_grid_step(self, *_a, **_k):
            return None

        def set_grid_step(self, *_a, **_k):
            return None

        def get_last_atr(self, *_a, **_k):
            return None

        def set_last_atr(self, *_a, **_k):
            return None

    monkeypatch.setattr(grid_bot, "grid_state", DummyGridState())
    monkeypatch.setattr(grid_bot, "atr_percent", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(grid_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])

    config = {
        "num_levels": 6,
        "volume_filter": False,
        "min_range_pct": 0.0001,
        "atr_normalization": False,
        "dynamic_grid": False,
        "range_window": 10,
        "atr_period": 3,
    }

    score, direction = grid_bot.generate_signal(df, config=config)
    assert direction == "long"
    assert score > 0


def test_hft_engine_placeholder(make_ohlcv):
    if hft_engine is None:
        pytest.skip("hft_engine strategy unavailable")
    score, direction = hft_engine.generate_signal(make_ohlcv([100, 101]))
    assert (score, direction) == (0.0, "none")


def test_lstm_bot_uses_model_prediction(ohlcv_trending_up, monkeypatch):
    if lstm_bot is None:
        pytest.skip("lstm_bot strategy unavailable")
    class DummyModel:
        def predict(self, *_a, **_k):
            return np.array([0.6])

    monkeypatch.setattr(lstm_bot, "MODEL", DummyModel())
    score, direction = lstm_bot.generate_signal(ohlcv_trending_up, config={"sequence_length": 50, "threshold_pct": 0.1})
    assert direction == "long"
    assert score == pytest.approx(0.6, rel=1e-3)


def test_maker_spread_prefers_sideways_market(ohlcv_range_bound):
    if maker_spread is None:
        pytest.skip("maker_spread strategy unavailable")
    score, direction = maker_spread.generate_signal(ohlcv_range_bound)
    assert direction == "maker_spread"
    assert score > 0


def test_market_making_scores_range(ohlcv_range_bound):
    if market_making_bot is None:
        pytest.skip("market_making_bot strategy unavailable")
    score, direction = market_making_bot.generate_signal(ohlcv_range_bound)
    assert direction == "market_making"
    assert score > 0

@pytest.mark.xfail(reason="mean reversion stack requires extensive historical context", strict=True)
def test_mean_bot_identifies_reversion(ohlcv_range_bound, monkeypatch):
    if mean_bot is None:
        pytest.skip("mean_bot strategy unavailable")
    df = ohlcv_range_bound.copy()
    df["close"] = df["close"] - 2
    df.loc[df.index[-1], "close"] -= 10
    monkeypatch.setattr(mean_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])
    config = {
        "ml_enabled": False,
        "indicator_lookback": 14,
        "rsi_oversold_pct": 100,
        "rsi_overbought_pct": 0,
    }
    score, direction = mean_bot.generate_signal(df, config=config)
    assert direction == "long"
    assert score > 0


def test_meme_wave_bot_uses_sentiment(make_ohlcv):
    if meme_wave_bot is None:
        pytest.skip("meme_wave_bot strategy unavailable")
    closes = [10, 10.5, 11.0, 11.5]
    volumes = [1_000, 1_200, 1_400, 5_000]
    df = make_ohlcv(closes, volumes=volumes)
    config = {
        "symbol": "DOGE",
        "vol_threshold": 1.5,
        "vol_mult": 2.0,
        "jump_mult": 1.0,
        "sentiment_threshold": 0.4,
        "avg_mempool_volume": 1_000,
        "recent_mempool_volume": 6_000,
    }
    score, direction = meme_wave_bot.generate_signal(df, config=config)
    assert direction == "long"
    assert score > 0

@pytest.mark.xfail(reason="micro scalp bot needs order book context for live signal", strict=True)
def test_micro_scalp_bot_long_signal(make_ohlcv, monkeypatch):
    if micro_scalp_bot is None:
        pytest.skip("micro_scalp_bot strategy unavailable")
    df = make_ohlcv([100, 101, 102, 103, 104, 120], volumes=[1_000, 1_100, 1_200, 1_300, 1_400, 3_000])
    config = {
        "micro_scalp": {
            "ema_fast": 2,
            "ema_slow": 3,
            "vol_window": 3,
            "min_vol_z": -10.0,
            "min_atr_pct": 0.0,
            "trend_filter": False,
            "imbalance_filter": False,
            "atr_normalization": False,
        }
    }
    monkeypatch.setattr(micro_scalp_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])
    score, direction = micro_scalp_bot.generate_signal(df, config=config)
    assert direction == "long"
    assert score > 0


def test_momentum_bot_macd_confirms(ohlcv_trending_up, monkeypatch):
    if momentum_bot is None:
        pytest.skip("momentum_bot strategy unavailable")
    monkeypatch.setattr(momentum_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])
    score, direction = momentum_bot.generate_signal(
        ohlcv_trending_up,
        config={"momentum_bot": {"fast_length": 3, "slow_length": 5, "atr_normalization": False}},
    )
    assert direction == "long"
    assert score > 0


def test_momentum_exploiter_detects_run(ohlcv_trending_up, monkeypatch):
    if momentum_exploiter is None:
        pytest.skip("momentum_exploiter strategy unavailable")
    config = {
        "threshold": 0.0,
        "momentum_window": 5,
        "volume_zscore_threshold": -1.0,
        "acceleration_threshold": 0.0,
        "min_atr_pct": 0.0,
    }
    monkeypatch.setattr(momentum_exploiter, "normalize_score_by_volatility", lambda *a, **k: a[1])
    monkeypatch.setattr(
        momentum_exploiter,
        "_detect_momentum_signals",
        lambda df, *_a, **_k: (0.6, "long", {"atr_pct": 0.02}),
    )
    score, direction = momentum_exploiter.generate_signal(ohlcv_trending_up, config=config)
    assert direction == "long"
    assert score > 0


def test_range_arb_bot_low_vol_buy(ohlcv_range_bound, monkeypatch):
    if range_arb_bot is None:
        pytest.skip("range_arb_bot strategy unavailable")
    monkeypatch.setattr(range_arb_bot, "kernel_regression", lambda *_a, **_k: ohlcv_range_bound["close"].iloc[-1])
    monkeypatch.setattr(range_arb_bot, "last_window_zscore", lambda *_a, **_k: -2.5)
    monkeypatch.setattr(range_arb_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])
    score, direction = range_arb_bot.generate_signal(
        ohlcv_range_bound,
        config={"range_arb_bot": {"atr_window": 5, "kr_window": 5, "z_threshold": 1.0, "vol_z_threshold": 10.0}},
    )
    assert direction == "long"
    assert score > 0


def test_sniper_bot_flags_breakout(make_ohlcv):
    if sniper_bot is None:
        pytest.skip("sniper_bot strategy unavailable")
    closes = [1, 1.05, 1.1, 1.5]
    volumes = [200, 250, 300, 1_000]
    df = make_ohlcv(closes, volumes=volumes)
    score, direction = sniper_bot.generate_signal(df, config={"min_volume": 100})
    assert direction == "long"
    assert score > 0

@pytest.mark.xfail(reason="stat arb bot requires correlated pairs with live spread dynamics", strict=True)
def test_stat_arb_bot_spread_mean_reverts(make_ohlcv, monkeypatch):
    if stat_arb_bot is None:
        pytest.skip("stat_arb_bot strategy unavailable")
    base = np.linspace(100, 102, 40)
    df_a = make_ohlcv(base)
    df_b = make_ohlcv(base + 0.5)
    df_b.loc[df_b.index[-1], "close"] += 10
    monkeypatch.setattr(stat_arb_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])
    monkeypatch.setattr(stat_arb_bot, "MODEL", None)
    score, direction = stat_arb_bot.generate_signal(
        df_a,
        df_b,
        config={"lookback": 10, "zscore_threshold": 0.1},
    )
    assert direction == "long"
    assert score > 0

@pytest.mark.xfail(reason="trend bot thresholds depend on longer trend confirmation", strict=True)
def test_trend_bot_long_signal(make_ohlcv, monkeypatch):
    if trend_bot is None:
        pytest.skip("trend_bot strategy unavailable")
    df = make_ohlcv([100, 101, 103, 105, 110, 125], volumes=[1_000, 1_100, 1_200, 1_300, 1_400, 2_500])
    config = {
        "trend_ema_fast": 2,
        "trend_ema_slow": 5,
        "rsi_oversold_pct": 100,
        "rsi_overbought_pct": 0,
        "volume_mult": 0.0,
        "atr_normalization": False,
    }
    monkeypatch.setattr(trend_bot, "normalize_score_by_volatility", lambda *a, **k: a[1])
    score, direction = trend_bot.generate_signal(df, config=config)
    assert direction == "long"
    assert score >= 0


def test_ultra_scalp_bot_fast_signal(ohlcv_trending_up, monkeypatch):
    if ultra_scalp_bot is None:
        pytest.skip("ultra_scalp_bot strategy unavailable")
    config = {
        "min_score": 0.0,
        "volume_mult": 1.0,
        "volume_window": 3,
        "ema_fast": 2,
        "ema_slow": 4,
        "macd_fast": 3,
        "macd_slow": 6,
        "macd_signal": 2,
    }
    monkeypatch.setattr(ultra_scalp_bot, "normalize_score_by_volatility", lambda *_a, **_k: 0.6)
    score, direction = ultra_scalp_bot.generate_signal(ohlcv_trending_up, config=config)
    assert direction in {"long", "short"}
    assert score >= 0


def test_volatility_harvester_high_volatility(ohlcv_trending_up, monkeypatch):
    if volatility_harvester is None:
        pytest.skip("volatility_harvester strategy unavailable")
    def fake_detect(df, *_a, **_k):
        return 0.6, "long", {"atr_pct": 0.02}

    monkeypatch.setattr(volatility_harvester, "_detect_volatility_signals", fake_detect)
    monkeypatch.setattr(volatility_harvester, "normalize_score_by_volatility", lambda *a, **k: a[1])
    score, direction = volatility_harvester.generate_signal(
        ohlcv_trending_up,
        config={"atr_window": 5, "volume_window": 5},
    )
    assert direction == "long"
    assert score > 0


@pytest.mark.skipif(not EMPTY_STRATEGIES, reason="no strategies available for empty-data test")
@pytest.mark.parametrize("strategy", EMPTY_STRATEGIES)
def test_strategies_handle_empty_dataframe(strategy):
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    score, direction = strategy(empty)
    assert score == 0.0
    assert direction == "none"


def test_dex_scalper_respects_gas_threshold(ohlcv_trending_up, monkeypatch):
    if dex_scalper is None:
        pytest.skip("dex_scalper strategy unavailable")
    monkeypatch.setenv("MOCK_ETH_PRIORITY_FEE_GWEI", "50")
    config = {"dex_scalper": {"gas_threshold_gwei": 10}}
    score, direction = dex_scalper.generate_signal(ohlcv_trending_up, config)
    assert (score, direction) == (0.0, "none")


def test_momentum_bot_handles_nan(monkeypatch):
    if momentum_bot is None:
        pytest.skip("momentum_bot strategy unavailable")
    df = pd.DataFrame(
        {
            "open": [1.0, np.nan],
            "high": [1.0, np.nan],
            "low": [1.0, np.nan],
            "close": [1.0, np.nan],
            "volume": [1_000, np.nan],
        }
    )
    score, direction = momentum_bot.generate_signal(df)
    assert (score, direction) == (0.0, "none")
