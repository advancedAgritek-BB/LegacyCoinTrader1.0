"""Database helpers for the portfolio service."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import PortfolioConfig


Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine(config: PortfolioConfig):
    """Return a singleton SQLAlchemy engine."""

    global _engine
    if _engine is None:
        _engine = create_engine(config.database_url, echo=False, future=True)
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
