"""Entrypoint for running the token discovery service with uvicorn."""

from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "services.token_discovery.app:app",
        host="0.0.0.0",
        port=8005,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
