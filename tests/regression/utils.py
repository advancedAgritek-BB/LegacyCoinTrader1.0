from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
from unittest.mock import patch

# Provide a lightweight fallback for optional dependencies that are not
# installed in the minimal test environment.
if "pydantic_settings" not in sys.modules:  # pragma: no cover - import hook
    module = types.ModuleType("pydantic_settings")

    class _BaseSettings:  # pylint: disable=too-few-public-methods
        """Minimal stand-in that mimics pydantic's BaseSettings API."""

        model_config: Dict[str, Any] = {}

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._data = kwargs

        def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return dict(self._data)

    module.BaseSettings = _BaseSettings
    module.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = module

from crypto_bot.phase_runner import BotContext, PhaseRunner
from crypto_bot.risk.risk_manager import RiskManager


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _round_float(value: float, places: int = 8) -> float:
    """Return ``value`` rounded to ``places`` decimal places."""

    return float(round(float(value), places))


def _load_json(path: Path) -> Dict[str, Any]:
    """Return parsed JSON from ``path``."""

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _dataframe_from_rows(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Materialize a DataFrame from fixture rows."""

    df = pd.DataFrame(list(rows))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    return df


def _dataframe_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Serialize a DataFrame back into fixture-compatible rows."""

    rows: List[Dict[str, Any]] = []
    for ts, row in df.iterrows():
        record: Dict[str, Any] = {"timestamp": ts.isoformat().replace("+00:00", "Z")}
        for column in ["open", "high", "low", "close", "volume"]:
            if column in row:
                record[column] = _round_float(row[column])
        rows.append(record)
    return rows


class RecordingRiskManager:
    """Proxy risk manager that records intermediate decisions."""

    def __init__(self, config: Mapping[str, Any]):
        self._manager = RiskManager.from_config(config)
        min_size = config.get("min_position_size_usd")
        if min_size is not None:
            setattr(self._manager.config, "min_position_size_usd", float(min_size))
        self.records: List[Dict[str, Any]] = []
        self._current_symbol: Optional[str] = None

    # Delegation ---------------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        return getattr(self._manager, item)

    # Recording hooks ----------------------------------------------------
    def reset_records(self) -> None:
        self.records.clear()

    def allow_trade(self, df: Any, strategy: Optional[str] = None):
        symbol = getattr(df, "attrs", {}).get("symbol")
        allowed, reason = self._manager.allow_trade(df, strategy)
        self.records.append(
            {
                "symbol": symbol or "",
                "strategy": strategy or "",
                "allowed": bool(allowed),
                "reason": reason or "",
            }
        )
        self._current_symbol = symbol
        return allowed, reason

    def position_size(
        self,
        confidence: float,
        balance: float,
        df: Optional[pd.DataFrame] = None,
        stop_distance: Optional[float] = None,
        atr: Optional[float] = None,
        price: Optional[float] = None,
    ) -> float:
        size = self._manager.position_size(
            confidence,
            balance,
            df=df,
            stop_distance=stop_distance,
            atr=atr,
            price=price,
        )
        if self.records:
            rec = self.records[-1]
            rec["confidence"] = float(confidence)
            if atr is not None:
                rec["atr"] = float(atr)
            if price is not None:
                rec["price"] = float(price)
            rec["position_size"] = float(size)
        return size

    def can_allocate(self, strategy: str, amount: float, balance: float) -> bool:
        allowed = self._manager.can_allocate(strategy, amount, balance)
        if self.records:
            rec = self.records[-1]
            rec["can_allocate"] = bool(allowed)
            rec["allocation_request"] = float(amount)
        return allowed

    def allocate_capital(self, strategy: str, amount: float) -> None:
        self._manager.allocate_capital(strategy, amount)
        if self.records:
            rec = self.records[-1]
            rec["allocated"] = float(amount)

    # Serialization ------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        normalized: List[Dict[str, Any]] = []
        for record in self.records:
            item: Dict[str, Any] = {
                "symbol": record.get("symbol", ""),
                "strategy": record.get("strategy", ""),
                "allowed": bool(record.get("allowed", False)),
                "reason": record.get("reason", ""),
            }
            if "confidence" in record:
                item["confidence"] = _round_float(record["confidence"])
            if "price" in record:
                item["price"] = _round_float(record["price"])
            if "atr" in record:
                item["atr"] = _round_float(record["atr"])
            if "position_size" in record:
                item["position_size"] = _round_float(record["position_size"])
            if "can_allocate" in record:
                item["can_allocate"] = bool(record.get("can_allocate", False))
            if "allocation_request" in record:
                item["allocation_request"] = _round_float(record["allocation_request"])
            if "allocated" in record:
                item["allocated"] = _round_float(record["allocated"])
            normalized.append(item)
        normalized.sort(key=lambda entry: entry.get("symbol"))
        return {"decisions": normalized}


class PositionGuardStub:
    """Allow opening any position for regression replay."""

    def can_open(self, _positions: Mapping[str, Any]) -> bool:
        return True


@dataclass
class RegressionFixture:
    signal: Dict[str, Any]
    risk: Dict[str, Any]
    execution: Dict[str, Any]


class RegressionReplay:
    """Orchestrates a regression replay using stored fixtures."""

    def __init__(self, fixture_name: str):
        base = FIXTURES_DIR / fixture_name
        if not base.exists():
            raise FileNotFoundError(f"Regression fixture '{fixture_name}' not found")
        self.signal_fixture = _load_json(base / "signal_snapshot.json")
        self.risk_fixture = _load_json(base / "risk_snapshot.json")
        self.execution_fixture = _load_json(base / "execution_snapshot.json")
        self.context_cfg = self.signal_fixture.get("context", {})
        self._recorded_trades: List[Dict[str, Any]] = []
        risk_cfg = self.context_cfg.get("risk", {})
        self.risk_manager = RecordingRiskManager(risk_cfg)

    # Fixture helpers ----------------------------------------------------
    def _analysis_results(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for entry in self.signal_fixture.get("analysis_results", []):
            candidate = {k: v for k, v in entry.items() if k != "ohlcv"}
            df = _dataframe_from_rows(entry.get("ohlcv", []))
            df.attrs["symbol"] = candidate.get("symbol")
            candidate["df"] = df
            results.append(candidate)
        return results

    def _build_context(self) -> BotContext:
        config = dict(self.context_cfg.get("config", {}))
        config.setdefault("top_n_symbols", len(self.signal_fixture.get("analysis_results", [])))
        config.setdefault("execution_mode", "live")
        config.setdefault("allow_short", False)
        exit_cfg = config.setdefault("exit_strategy", {})
        exit_cfg.setdefault("place_native_stop", False)

        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config=config,
            services=None,
        )
        ctx.balance = float(self.context_cfg.get("balance", 0.0))
        ctx.analysis_results = []
        ctx.current_batch = []
        ctx.position_guard = PositionGuardStub()
        ctx.risk_manager = self.risk_manager
        ctx.timing = {}
        ctx.trade_manager = None
        return ctx

    def _serialize_signals(self, results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        serialized: List[Dict[str, Any]] = []
        for entry in results:
            df = entry.get("df")
            serialized.append(
                {
                    "symbol": entry.get("symbol"),
                    "name": entry.get("name"),
                    "direction": entry.get("direction"),
                    "score": _round_float(entry.get("score", 0.0)),
                    "probabilities": {
                        key: _round_float(val)
                        for key, val in sorted(entry.get("probabilities", {}).items())
                    },
                    "regime": entry.get("regime"),
                    "atr": _round_float(entry.get("atr", 0.0)),
                    "ohlcv": _dataframe_to_rows(df) if isinstance(df, pd.DataFrame) else [],
                }
            )
        serialized.sort(key=lambda item: item.get("symbol"))
        return {"context": self.signal_fixture.get("context", {}), "analysis_results": serialized}

    def _serialize_trades(self) -> Dict[str, Any]:
        trades = []
        for trade in self._recorded_trades:
            trades.append(
                {
                    "symbol": trade["symbol"],
                    "strategy": trade["strategy"],
                    "side": trade["side"],
                    "notional": _round_float(trade["notional"]),
                    "amount": _round_float(trade["amount"], places=10),
                    "price": _round_float(trade["price"]),
                    "confidence": _round_float(trade["confidence"]),
                    "score": _round_float(trade["score"]),
                    "regime": trade.get("regime"),
                    "sentiment_boost": _round_float(trade.get("sentiment_boost", 1.0)),
                }
            )
        trades.sort(key=lambda item: item.get("symbol"))
        return {"trades": trades}

    # Replay -------------------------------------------------------------
    async def _load_signal_phase(self, ctx: BotContext) -> None:
        ctx.analysis_results = self._analysis_results()
        ctx.current_batch = [entry.get("symbol") for entry in ctx.analysis_results]
        self.risk_manager.reset_records()

    async def _replay_execute_phase(self, ctx: BotContext) -> None:
        results = getattr(ctx, "analysis_results", [])
        if not results:
            return

        actionable = [
            candidate
            for candidate in results
            if not candidate.get("skip") and candidate.get("direction") not in (None, "none")
        ]
        actionable.sort(key=lambda cand: cand.get("score", 0.0), reverse=True)

        top_n = int(ctx.config.get("top_n_symbols", len(actionable)))
        balance = float(getattr(ctx, "balance", 0.0))

        for candidate in actionable[:top_n]:
            sym = candidate.get("symbol", "")
            df = candidate.get("df")
            if not sym or not isinstance(df, pd.DataFrame) or df.empty:
                continue

            strategy = candidate.get("name", "")
            allowed, _reason = ctx.risk_manager.allow_trade(df, strategy)
            if not allowed:
                continue

            probs = candidate.get("probabilities", {})
            regime = candidate.get("regime")
            confidence = float(probs.get(regime, 0.0))
            price = float(df["close"].iloc[-1])
            atr = candidate.get("atr")

            size = ctx.risk_manager.position_size(
                confidence,
                balance,
                df=df,
                atr=atr,
                price=price,
            )
            if size <= 0:
                continue

            if not ctx.risk_manager.can_allocate(strategy, size, balance):
                continue

            side = "buy" if candidate.get("direction") == "long" else "sell"
            self._record_trade(candidate, sym, size, price, strategy, side, confidence)
            ctx.risk_manager.allocate_capital(strategy, size)

    def _record_trade(
        self,
        candidate: Mapping[str, Any],
        symbol: str,
        size: float,
        price: float,
        strategy: str,
        side: str,
        confidence: float,
    ) -> None:
        trade = {
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "notional": float(size),
            "amount": float(size / price) if price else 0.0,
            "price": float(price),
            "confidence": _round_float(confidence),
            "score": float(candidate.get("score", 0.0)),
            "regime": candidate.get("regime"),
            "sentiment_boost": 1.0,
        }
        self._recorded_trades.append(trade)

    def _patch_environment(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, {"MOCK_FNG_VALUE": "50"}, clear=False))
        stack.enter_context(
            patch("crypto_bot.utils.trade_memory.should_avoid", lambda *_args, **_kwargs: False)
        )
        stack.enter_context(
            patch("crypto_bot.utils.ev_tracker.get_expected_value", lambda *_a, **_k: 1.0)
        )
        stack.enter_context(patch("crypto_bot.utils.ev_tracker._load_stats", lambda: {}))
        return stack

    def run(self) -> tuple[RegressionFixture, RegressionFixture]:
        ctx = self._build_context()
        phases = [self._load_signal_phase, self._replay_execute_phase]
        runner = PhaseRunner(phases)
        with self._patch_environment():
            asyncio.run(runner.run(ctx))
        observed = RegressionFixture(
            signal=self._serialize_signals(ctx.analysis_results),
            risk=self.risk_manager.snapshot(),
            execution=self._serialize_trades(),
        )
        expected = RegressionFixture(
            signal=self.signal_fixture,
            risk=self.risk_fixture,
            execution=self.execution_fixture,
        )
        return observed, expected


def run_regression_cycle(fixture_name: str) -> tuple[RegressionFixture, RegressionFixture]:
    """Execute a regression replay for ``fixture_name`` and return (observed, expected)."""

    replay = RegressionReplay(fixture_name)
    return replay.run()
