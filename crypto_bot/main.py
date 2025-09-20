from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional

import click
import httpx

from crypto_bot.runtime_signals import cleanup_pid_file

TRADING_ENGINE_START_PATH = "/api/v1/trading/cycles/start"
TRADING_ENGINE_STOP_PATH = "/api/v1/trading/cycles/stop"
TRADING_ENGINE_STATUS_PATH = "/api/v1/trading/cycles/status"
TRADING_ENGINE_RUN_ONCE_PATH = "/api/v1/trading/cycles/run"
TRADING_ENGINE_CLOSE_ALL_PATH = "/api/v1/trading/positions/close-all"

DEFAULT_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = float(os.getenv("API_GATEWAY_TIMEOUT", "10.0"))


class TradingEngineClientError(RuntimeError):
    """Raised when the trading engine API cannot be reached or returns an error."""


@dataclass
class TradingEngineClient:
    """Thin HTTP client for interacting with the trading engine via the gateway."""

    base_url: str = DEFAULT_GATEWAY_URL
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - error branch
            detail = exc.response.text.strip()
            message = f"Gateway responded with {exc.response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise TradingEngineClientError(message) from exc
        except httpx.RequestError as exc:  # pragma: no cover - network failure
            raise TradingEngineClientError(f"Unable to reach gateway: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            try:
                return response.json()
            except ValueError:
                return response.text
        if response.text:
            return response.text
        return None

    def start_cycles(
        self,
        *,
        interval_seconds: Optional[int],
        immediate: bool,
        metadata: Optional[Dict[str, Any]],
    ) -> Any:
        payload: Dict[str, Any] = {"immediate": immediate}
        if interval_seconds is not None:
            payload["interval_seconds"] = interval_seconds
        if metadata:
            payload["metadata"] = metadata
        return self._request("POST", TRADING_ENGINE_START_PATH, payload)

    def stop_cycles(self) -> Any:
        return self._request("POST", TRADING_ENGINE_STOP_PATH, {})

    def cycle_status(self) -> Any:
        return self._request("GET", TRADING_ENGINE_STATUS_PATH, None)

    def run_once(self, *, metadata: Optional[Dict[str, Any]] = None) -> Any:
        payload: Dict[str, Any] = {}
        if metadata:
            payload["metadata"] = metadata
        return self._request("POST", TRADING_ENGINE_RUN_ONCE_PATH, payload or None)

    def emergency_liquidation(self) -> Any:
        return self._request("POST", TRADING_ENGINE_CLOSE_ALL_PATH, {})


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _format_response(data: Any) -> str:
    if data is None:
        return "null"
    if isinstance(data, (dict, list)):
        return json.dumps(data, indent=2, sort_keys=True, default=_json_default)
    return str(data)


def _parse_metadata(pairs: Iterable[str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise click.BadParameter(f"Metadata must be in KEY=VALUE format (got '{item}')")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise click.BadParameter("Metadata keys cannot be empty")
        metadata[key] = value
    return metadata


@click.group(help="Control the LegacyCoinTrader trading engine via the API gateway.")
@click.option(
    "--gateway-url",
    envvar="API_GATEWAY_URL",
    default=DEFAULT_GATEWAY_URL,
    show_default=True,
    help="Base URL for the API gateway.",
)
@click.option(
    "--timeout",
    type=float,
    envvar="API_GATEWAY_TIMEOUT",
    default=DEFAULT_TIMEOUT,
    show_default=True,
    help="HTTP request timeout in seconds.",
)
@click.pass_context
def cli(ctx: click.Context, gateway_url: str, timeout: float) -> None:
    ctx.obj = TradingEngineClient(base_url=gateway_url, timeout=timeout)


@cli.command("status", help="Fetch the current trading engine scheduler state.")
@click.pass_obj
def status(client: TradingEngineClient) -> None:
    try:
        data = client.cycle_status()
    except TradingEngineClientError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(_format_response(data))


@cli.command("start", help="Start scheduled trading cycles on the trading engine.")
@click.option("--interval", type=int, help="Override the cycle interval in seconds.")
@click.option(
    "--immediate/--no-immediate",
    default=True,
    show_default=True,
    help="Run the first cycle immediately.",
)
@click.option(
    "--metadata",
    multiple=True,
    help="Attach metadata as KEY=VALUE (may be provided multiple times).",
)
@click.pass_obj
def start(
    client: TradingEngineClient,
    interval: Optional[int],
    immediate: bool,
    metadata: Iterable[str],
) -> None:
    try:
        meta = _parse_metadata(metadata)
        data = client.start_cycles(
            interval_seconds=interval,
            immediate=immediate,
            metadata=meta,
        )
    except TradingEngineClientError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(_format_response(data))


@cli.command("stop", help="Stop scheduled trading cycles.")
@click.pass_obj
def stop(client: TradingEngineClient) -> None:
    try:
        data = client.stop_cycles()
    except TradingEngineClientError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(_format_response(data))


@cli.command("pause", help="Alias for 'stop' to pause scheduled trading cycles.")
@click.pass_context
def pause(ctx: click.Context) -> None:
    ctx.invoke(stop)


@cli.command("run-once", help="Trigger a single trading cycle immediately.")
@click.option(
    "--metadata",
    multiple=True,
    help="Attach metadata as KEY=VALUE (may be provided multiple times).",
)
@click.pass_obj
def run_once(client: TradingEngineClient, metadata: Iterable[str]) -> None:
    try:
        meta = _parse_metadata(metadata)
        data = client.run_once(metadata=meta)
    except TradingEngineClientError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(_format_response(data))


@cli.command("emergency-stop", help="Request immediate liquidation of all positions.")
@click.pass_obj
def emergency_stop(client: TradingEngineClient) -> None:
    try:
        data = client.emergency_liquidation()
    except TradingEngineClientError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(_format_response(data))


async def _main_impl(*_: Any, **__: Any) -> None:  # pragma: no cover - backwards compatibility
    raise RuntimeError(
        "The in-process trading loop has been removed. Use 'python -m crypto_bot.main start' "
        "or the dashboard to control the trading engine service."
    )


def main() -> None:
    cli()


__all__ = [
    "TradingEngineClient",
    "TradingEngineClientError",
    "cli",
    "main",
    "cleanup_pid_file",
    "_main_impl",
]


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
