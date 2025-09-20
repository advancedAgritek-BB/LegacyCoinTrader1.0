"""Database helpers for the portfolio service."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import PortfolioConfig


Base = declarative_base()

_engine = None
_SessionLocal = None
_engine_url: str | None = None

LOGGER = logging.getLogger(__name__)

_FALLBACK_DB_URL = os.getenv(
    "PORTFOLIO_FALLBACK_DATABASE_URL", "sqlite:///portfolio_local.db"
)


def _create_engine(database_url: str):
    return create_engine(database_url, echo=False, future=True)


def get_engine(config: PortfolioConfig):
    """Return a singleton SQLAlchemy engine, with local fallback.

    In developer environments the Postgres instance defined in configuration may
    not be running. Instead of crashing the portfolio service (and every caller
    that depends on it), we attempt to connect to the configured database and
    gracefully fall back to a SQLite file if the connection fails.
    """

    global _engine, _engine_url
    if _engine is not None:
        return _engine

    primary_url = config.database_url
    try:
        engine = _create_engine(primary_url)
        # Force a connection so we fail fast if the database is offline
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        _engine = engine
        _engine_url = primary_url
        LOGGER.info("Portfolio service connected to database %s", primary_url)
        return _engine
    except SQLAlchemyError as exc:
        fallback_url = _FALLBACK_DB_URL
        LOGGER.warning(
            "Portfolio database unavailable at %s (%s); falling back to %s",
            primary_url,
            exc,
            fallback_url,
        )
        engine = _create_engine(fallback_url)
        _engine = engine
        _engine_url = fallback_url
        # Update config so subsequent calls reuse the fallback URL
        config.database_url = fallback_url
        return _engine


def get_session_factory(config: PortfolioConfig):
    """Return a singleton session factory."""

    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(config),
            expire_on_commit=False,
            class_=Session,
        )
    return _SessionLocal


@contextmanager
def get_session(config: PortfolioConfig) -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""

    session_factory = get_session_factory(config)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
