from __future__ import annotations

import json
from typing import Any, Dict, List

from click.testing import CliRunner

from crypto_bot.main import TradingEngineClient, cli


def _patch_requests(monkeypatch) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    def fake_request(self, method: str, path: str, payload: Union[Dict[str, Any], None] = None) -> Dict[str, Any]:
        record = {
            "method": method,
            "path": path,
            "payload": payload,
        }
        calls.append(record)
        return {"method": method, "path": path, "payload": payload, "status": "ok"}

    monkeypatch.setattr(TradingEngineClient, "_request", fake_request, raising=True)
    return calls


def test_cli_start_triggers_gateway_request(monkeypatch) -> None:
    calls = _patch_requests(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--gateway-url",
            "http://gateway.local",
            "start",
            "--interval",
            "120",
            "--metadata",
            "mode=paper",
            "--metadata",
            "trigger=manual",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/trading-engine/cycles/start"
    assert call["payload"] == {
        "immediate": True,
        "interval_seconds": 120,
        "metadata": {"mode": "paper", "trigger": "manual"},
    }

    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["payload"]["metadata"]["mode"] == "paper"


def test_cli_pause_alias(monkeypatch) -> None:
    calls = _patch_requests(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(cli, ["pause"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/trading-engine/cycles/stop"
    assert call["payload"] == {}


def test_cli_emergency_stop(monkeypatch) -> None:
    calls = _patch_requests(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(cli, ["emergency-stop"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    call = calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/trading-engine/positions/close-all"
    assert call["payload"] == {}

