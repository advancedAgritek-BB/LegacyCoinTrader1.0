from __future__ import annotations

import sys
import types

if "pydantic_settings" not in sys.modules:  # pragma: no cover - test scaffold
    module = types.ModuleType("pydantic_settings")

    class _BaseSettings:  # pragma: no cover - minimal shim
        model_config = {}

    module.BaseSettings = _BaseSettings
    module.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = module

from services.trading_engine.interface import CycleContext
from crypto_bot.phase_runner import BotContext


class _FakeMemoryManager:
    def perform_maintenance(self):
        return {"cleaned": True}


def _make_bot_context() -> BotContext:
    ctx = BotContext(
        positions={},
        df_cache={},
        regime_cache={},
        config={},
    )
    ctx.memory_manager = _FakeMemoryManager()
    return ctx


def test_cycle_context_delegate_attributes() -> None:
    bot = _make_bot_context()
    wrapper = CycleContext(bot, metadata={"source": "test"}, state={"cycles": 1})

    wrapper.balance = 125.0
    assert bot.balance == 125.0
    assert wrapper.metadata == {"source": "test"}

    wrapper.state["cycles"] = 2
    assert wrapper.state["cycles"] == 2

    maintenance = wrapper.perform_memory_maintenance()
    assert maintenance == {"cleaned": True}
