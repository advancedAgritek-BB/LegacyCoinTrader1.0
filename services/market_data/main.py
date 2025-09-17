from __future__ import annotations

"""Entrypoint for running the market data service locally."""

import uvicorn

from .app import app


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    uvicorn.run("services.market_data.app:app", host="0.0.0.0", port=8002, reload=False)
