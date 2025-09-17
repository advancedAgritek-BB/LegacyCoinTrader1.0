import copy
from collections import defaultdict
import importlib
from decimal import Decimal
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from httpx import AsyncClient

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:  # pragma: no cover - test path bootstrap
    sys.path.insert(0, str(ROOT))

if "pydantic_settings" not in sys.modules:  # pragma: no cover - optional dependency shim
    stub = ModuleType("pydantic_settings")

    class _BaseSettings:  # pylint: disable=too-few-public-methods
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._data = kwargs

        def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return dict(self._data)

    stub.BaseSettings = _BaseSettings
    stub.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = stub

if "crypto_bot.startup_utils" not in sys.modules:  # pragma: no cover - optional dependency shim
    startup_stub = ModuleType("crypto_bot.startup_utils")
    startup_stub.create_service_container = lambda: None
    startup_stub.load_config = lambda: {}
    sys.modules["crypto_bot.startup_utils"] = startup_stub

if "services.monitoring.config" not in sys.modules:  # pragma: no cover - optional dependency shim
    monitoring_settings = SimpleNamespace()
    monitoring_settings.metrics = SimpleNamespace(default_labels={})
    monitoring_settings.for_service = lambda _name: monitoring_settings

    monitoring_config_stub = ModuleType("services.monitoring.config")
    monitoring_config_stub.get_monitoring_settings = lambda: monitoring_settings
    sys.modules["services.monitoring.config"] = monitoring_config_stub

if "services.monitoring.logging" not in sys.modules:  # pragma: no cover - optional dependency shim
    logging_stub = ModuleType("services.monitoring.logging")
    logging_stub.configure_logging = lambda _settings: None
    sys.modules["services.monitoring.logging"] = logging_stub

if "services.monitoring.instrumentation" not in sys.modules:  # pragma: no cover - optional dependency shim
    instrumentation_stub = ModuleType("services.monitoring.instrumentation")
    instrumentation_stub.instrument_fastapi_app = lambda *_args, **_kwargs: None
    instrumentation_stub.instrument_flask_app = lambda *_args, **_kwargs: None
    sys.modules["services.monitoring.instrumentation"] = instrumentation_stub

from crypto_bot.services.interfaces import (
    CacheUpdateResponse,
    ExchangeResponse,
    LoadSymbolsResponse,
    StrategyBatchResponse,
    StrategyEvaluationResult,
    TokenDiscoveryResponse,
    TradeExecutionResponse,
)


class FakeRedis:
    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    async def ping(self) -> bool:  # pragma: no cover - trivial
        return True

    async def close(self) -> None:  # pragma: no cover - trivial
        return None

    async def get(self, key: str) -> Any:
        return self._store.get(key)

    async def set(self, key: str, value: Any) -> bool:
        self._store[key] = value
        return True


@pytest.mark.asyncio
async def test_run_cycle_traverses_production_phases(monkeypatch):
    config = {
        "exchange": "kraken",
        "symbols": ["BTC/USD"],
        "mode": "auto",
        "execution_mode": "dry_run",
        "max_open_trades": 3,
        "ohlcv_limit": 50,
        "solana_scanner": {"enabled": True, "max_tokens_per_scan": 5},
        "token_discovery_feed": {"enabled": False},
        "risk": {
            "max_drawdown": 0.5,
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.02,
            "starting_balance": 10000,
            "min_volume": 0.0,
            "volume_threshold_ratio": 0.1,
        },
    }

    timestamps = pd.date_range("2024-01-01", periods=30, freq="H", tz="UTC")
    price_series = [100 + idx for idx in range(30)]
    data_frame = pd.DataFrame(
        {
            "open": price_series,
            "high": [p + 1 for p in price_series],
            "low": [p - 1 for p in price_series],
            "close": price_series,
            "volume": [1_000.0] * len(price_series),
        },
        index=timestamps,
    )

    ohlcv_cache = defaultdict(dict)
    ohlcv_cache["1h"]["BTC/USD"] = data_frame
    regime_cache = defaultdict(dict)
    regime_cache["context"]["BTC/USD"] = {"regime": "bull"}

    market_data = SimpleNamespace(
        load_symbols=AsyncMock(return_value=LoadSymbolsResponse(symbols=["BTC/USD", "ETH/USD"])),
        update_multi_tf_cache=AsyncMock(return_value=CacheUpdateResponse(cache=ohlcv_cache)),
        update_regime_cache=AsyncMock(return_value=CacheUpdateResponse(cache=regime_cache)),
    )

    strategy_result = StrategyEvaluationResult(
        symbol="BTC/USD",
        regime="bull",
        strategy="trend",
        score=0.82,
        direction="long",
        atr=1.5,
        metadata={"confidence": 0.75},
    )
    strategy = SimpleNamespace(
        evaluate_batch=AsyncMock(return_value=StrategyBatchResponse(results=(strategy_result,), errors=())),
    )

    execution = SimpleNamespace(
        create_exchange=MagicMock(return_value=ExchangeResponse(exchange="exchange", ws_client=None)),
        execute_trade=AsyncMock(return_value=TradeExecutionResponse(order={"id": "abc"})),
    )

    portfolio = SimpleNamespace(
        list_positions=MagicMock(return_value=[]),
        get_trade_manager=MagicMock(return_value=SimpleNamespace(get_all_positions=lambda: [])),
        create_trade=MagicMock(),
        check_risk=MagicMock(return_value=[]),
        compute_pnl=MagicMock(return_value=SimpleNamespace(
            realized=Decimal("0"), unrealized=Decimal("0"), total=Decimal("0")
        )),
    )

    token_discovery = SimpleNamespace(
        discover_tokens=AsyncMock(return_value=TokenDiscoveryResponse(tokens=["SOLXYZ"]))
    )
    monitoring = SimpleNamespace(record_scanner_metrics=MagicMock())

    from crypto_bot.services.interfaces import ServiceContainer

    container = ServiceContainer(
        market_data=market_data,
        strategy=strategy,
        portfolio=portfolio,
        execution=execution,
        token_discovery=token_discovery,
        monitoring=monitoring,
    )

    trading_app = importlib.import_module("services.trading_engine.app")

    class DummySettings:
        app_name = "trading-engine-service"
        default_cycle_interval = 60
        log_level = "INFO"
        redis_host = "localhost"
        redis_port = 6379
        redis_db = 0
        redis_use_ssl = False
        state_key_prefix = "trading_engine"

        def redis_dsn(self) -> str:
            return "redis://localhost:6379/0"

    monkeypatch.setattr(trading_app, "load_trading_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(trading_app, "build_service_container", lambda: container)
    monkeypatch.setattr(trading_app.redis, "from_url", lambda *_args, **_kwargs: FakeRedis())
    monkeypatch.setattr(trading_app, "get_settings", lambda: DummySettings())

    async with trading_app.app.router.lifespan_context(trading_app.app):
        async with AsyncClient(app=trading_app.app, base_url="http://test") as client:
            scheduler = getattr(trading_app.app.state, "scheduler", None)
            assert scheduler is not None, "scheduler not initialized"
            response = await client.post("/cycles/run", json={"metadata": {"cycle_id": "unit-test"}})
            payload = None
            try:
                payload = response.json()
            except Exception:  # pragma: no cover - debugging aid
                payload = response.text
            assert response.status_code == 200, payload

    data = payload
    phase_names = set(data["timings"].keys())
    expected_phases = {
        "fetch_candidates",
        "process_solana_candidates",
        "update_caches",
        "analyse_batch",
        "execute_signals",
        "handle_exits",
        "monitor_positions_phase",
    }
    assert expected_phases <= phase_names

    phase_metadata = data["metadata"]["phases"]
    assert phase_metadata["fetch_candidates"]["total"] >= 2
    assert phase_metadata["process_solana_candidates"]["tokens"] == 1
    assert phase_metadata["execute_signals"]["executed"] == 1

    market_data.load_symbols.assert_awaited()
    market_data.update_multi_tf_cache.assert_awaited()
    market_data.update_regime_cache.assert_awaited()
    strategy.evaluate_batch.assert_awaited()
    execution.create_exchange.assert_called_once()
    execution.execute_trade.assert_awaited()
    portfolio.create_trade.assert_called_once()
    token_discovery.discover_tokens.assert_awaited()
    monitoring.record_scanner_metrics.assert_called_once()
