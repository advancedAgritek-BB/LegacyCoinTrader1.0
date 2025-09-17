from __future__ import annotations

"""Entrypoint for running the trading engine service locally."""

import uvicorn

from .app import app


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    uvicorn.run("services.trading_engine.app:app", host="0.0.0.0", port=8001, reload=False)
