"""Entrypoint for running the execution FastAPI service."""

from __future__ import annotations

import uvicorn

from .app import create_app
from .config import get_execution_api_settings

app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch
    settings = get_execution_api_settings()
    uvicorn.run(
        "services.execution.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
