"""Database helpers for the identity service."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import IdentitySettings, load_identity_settings


Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine(settings: IdentitySettings | None = None):
    """Return a singleton SQLAlchemy engine."""

    global _engine
    settings = settings or load_identity_settings()
    if _engine is None:
        _engine = create_engine(settings.database_url, echo=False, future=True)
    return _engine


def get_session_factory(settings: IdentitySettings | None = None):
    """Return a cached SQLAlchemy session factory."""

    global _SessionLocal
    settings = settings or load_identity_settings()
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(settings),
            expire_on_commit=False,
            class_=Session,
        )
    return _SessionLocal


@contextmanager
def get_session(settings: IdentitySettings | None = None) -> Iterator[Session]:
    """Provide a transactional scope for identity operations."""

    session_factory = get_session_factory(settings)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "Base",
    "get_engine",
    "get_session",
    "get_session_factory",
]
