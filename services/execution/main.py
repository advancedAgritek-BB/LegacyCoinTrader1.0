"""Entry point for running the execution service via ``uvicorn``."""

from __future__ import annotations

import os

import uvicorn

from .app import app


def run() -> None:  # pragma: no cover - runtime helper
    host = os.getenv("EXECUTION_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("EXECUTION_SERVICE_PORT", "8006"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover - script execution
    run()

