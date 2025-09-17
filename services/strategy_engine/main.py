"""Entrypoint for running the strategy engine service locally."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    uvicorn.run("services.strategy_engine.app:app", host="0.0.0.0", port=8004, reload=False)
