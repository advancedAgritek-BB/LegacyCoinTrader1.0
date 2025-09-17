import sys
import types
from contextlib import asynccontextmanager

import httpx
import pytest


def _stub_monitoring_modules() -> None:
    monitoring_module = types.ModuleType("services.monitoring")
    config_module = types.ModuleType("services.monitoring.config")
    logging_module = types.ModuleType("services.monitoring.logging")
    instrumentation_module = types.ModuleType("services.monitoring.instrumentation")

    class _DummySettings:
        def __init__(self) -> None:
            self.metrics = types.SimpleNamespace(default_labels={})

        def for_service(self, _service_name: str, *, environment: str | None = None):
            return self

    def get_monitoring_settings() -> _DummySettings:
        return _DummySettings()

    def configure_logging(_settings: object) -> None:  # pragma: no cover - noop
        return None

    def instrument_fastapi_app(app, settings=None):  # pragma: no cover - noop
        return app

    config_module.get_monitoring_settings = get_monitoring_settings
    logging_module.configure_logging = configure_logging
    instrumentation_module.instrument_fastapi_app = instrument_fastapi_app

    monitoring_module.config = config_module
    monitoring_module.logging = logging_module
    monitoring_module.instrumentation = instrumentation_module

    sys.modules.setdefault("services.monitoring", monitoring_module)
    sys.modules.setdefault("services.monitoring.config", config_module)
    sys.modules.setdefault("services.monitoring.logging", logging_module)
    sys.modules.setdefault("services.monitoring.instrumentation", instrumentation_module)


_stub_monitoring_modules()


def _stub_trading_engine_config() -> None:
    config_module = types.ModuleType("services.trading_engine.config")

    class Settings:
        def __init__(self) -> None:
            self.app_name = "trading-engine-service"
            self.host = "0.0.0.0"
            self.port = 8001
            self.redis_host = "localhost"
            self.redis_port = 6379
            self.redis_db = 0
            self.state_key_prefix = "trading_engine"
            self.default_cycle_interval = 60
            self.log_level = "INFO"

        def redis_dsn(self) -> str:
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    _settings = Settings()

    def get_settings() -> Settings:
        return _settings

    config_module.Settings = Settings
    config_module.get_settings = get_settings

    sys.modules.setdefault("services.trading_engine.config", config_module)


_stub_trading_engine_config()

@asynccontextmanager
async def _noop_lifespan(_app):
    yield

from services.trading_engine.app import app, get_scheduler
from services.trading_engine.interface import TradingEngineInterface
from services.trading_engine.liquidation import LiquidationReport, PositionLiquidation


@pytest.mark.asyncio
async def test_interface_liquidation_uses_helper():
    calls: list[str] = []

    class DummyHelper:
        async def close_all_positions(self) -> LiquidationReport:
            calls.append("called")
            return LiquidationReport()

    interface = TradingEngineInterface(liquidation_helper=DummyHelper())
    report = await interface.liquidate_positions()

    assert isinstance(report, LiquidationReport)
    assert calls == ["called"]


@pytest.mark.asyncio
async def test_close_all_positions_endpoint_reports_success():
    class DummyScheduler:
        async def liquidate_positions(self) -> LiquidationReport:
            return LiquidationReport(
                positions=[
                    PositionLiquidation(
                        symbol="BTC/USD",
                        side="long",
                        amount=1.0,
                        status="closed",
                    )
                ]
            )

    app.router.lifespan_context = _noop_lifespan

    app.dependency_overrides[get_scheduler] = lambda: DummyScheduler()
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/positions/close-all")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["closed_positions"] == 1
    assert data["total_positions"] == 1
