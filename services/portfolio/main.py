"""Entry point for running the portfolio service."""

from __future__ import annotations

import logging

import uvicorn

from .config import PortfolioConfig
from .grpc_server import serve as serve_grpc
from .rest_api import app

logger = logging.getLogger(__name__)


def run_rest(config: PortfolioConfig) -> None:
    uvicorn.run(
        app,
        host=config.rest_host,
        port=config.rest_port,
        log_level="info",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = PortfolioConfig.from_env()
    grpc_server = serve_grpc(config)
    logger.info(
        "Portfolio gRPC server running on %s:%s", config.grpc_host, config.grpc_port
    )

    try:
        run_rest(config)
    finally:
        logger.info("Shutting down gRPC server")
        grpc_server.stop(grace=None)


if __name__ == "__main__":
    main()
