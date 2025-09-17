"""HTTP client utilities for interacting with the API gateway.

This module provides both synchronous and asynchronous helpers so the Flask
frontend and the FastAPI API surface can communicate with the microservice
layer without touching local log files or in-process controllers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

try:  # Flask is optional in certain test scenarios
    from flask import has_request_context, session
except Exception:  # pragma: no cover - fallback for non-flask environments
    def has_request_context() -> bool:  # type: ignore
        return False

    class _Session(dict):
        pass

    session = _Session()  # type: ignore


DEFAULT_TIMEOUT: float = 10.0


class ApiGatewayError(RuntimeError):
    """Raised when the API gateway cannot be reached or returns an error."""


def _gateway_base_url() -> str:
    """Return the API gateway base URL from environment configuration."""

    base_url = os.getenv("API_GATEWAY_URL", "http://localhost:8000")
    return base_url.rstrip("/")


def _build_url(path: str) -> str:
    base_url = _gateway_base_url()
    return f"{base_url}/{path.lstrip('/')}"


def _session_headers() -> Dict[str, str]:
    """Return headers derived from the current Flask session."""

    if not has_request_context():
        return {}

    headers: Dict[str, str] = {}
    token = session.get("access_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_gateway_url(path: str) -> str:
    """Public helper for constructing gateway URLs."""

    return _build_url(path)


def _handle_response(response: httpx.Response) -> Any:
    response.raise_for_status()
    if not response.content:
        return None
    return response.json()


def get_gateway_json(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Perform a synchronous GET request against the API gateway."""

    url = _build_url(path)
    request_headers = {**_session_headers(), **(headers or {})}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, params=params, headers=request_headers or None)
        return _handle_response(response)
    except httpx.HTTPStatusError as exc:
        raise ApiGatewayError(
            f"Gateway request failed with status {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:  # pragma: no cover - network layer
        raise ApiGatewayError(f"Unable to reach API gateway: {exc}") from exc


def post_gateway_json(
    path: str,
    *,
    json: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Perform a synchronous POST request against the API gateway."""

    url = _build_url(path)
    request_headers = {**_session_headers(), **(headers or {})}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=json, headers=request_headers or None)
        return _handle_response(response)
    except httpx.HTTPStatusError as exc:
        raise ApiGatewayError(
            f"Gateway request failed with status {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:  # pragma: no cover - network layer
        raise ApiGatewayError(f"Unable to reach API gateway: {exc}") from exc


async def async_get_gateway_json(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Perform an asynchronous GET request against the API gateway."""

    url = _build_url(path)
    request_headers = {**_session_headers(), **(headers or {})}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params, headers=request_headers or None)
        return _handle_response(response)
    except httpx.HTTPStatusError as exc:
        raise ApiGatewayError(
            f"Gateway request failed with status {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:  # pragma: no cover - network layer
        raise ApiGatewayError(f"Unable to reach API gateway: {exc}") from exc


async def async_post_gateway_json(
    path: str,
    *,
    json: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Perform an asynchronous POST request against the API gateway."""

    url = _build_url(path)
    request_headers = {**_session_headers(), **(headers or {})}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=json, headers=request_headers or None)
        return _handle_response(response)
    except httpx.HTTPStatusError as exc:
        raise ApiGatewayError(
            f"Gateway request failed with status {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:  # pragma: no cover - network layer
        raise ApiGatewayError(f"Unable to reach API gateway: {exc}") from exc


__all__ = [
    "build_gateway_url",
    "ApiGatewayError",
    "async_get_gateway_json",
    "async_post_gateway_json",
    "get_gateway_json",
    "post_gateway_json",
]
