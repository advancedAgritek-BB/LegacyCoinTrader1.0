"""Lightweight status reporting helpers for cross-service visibility."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import LOG_DIR


_STATUS_FILE = LOG_DIR / "system_status.json"
_LOCK = threading.Lock()


def _read_status() -> Dict[str, Any]:
    try:
        with _STATUS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _write_status(data: Dict[str, Any]) -> None:
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _STATUS_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    tmp_path.replace(_STATUS_FILE)


def update_status(section: str, payload: Dict[str, Any]) -> None:
    """Merge ``payload`` into the status file under ``section``."""

    if not section:
        raise ValueError("section is required")
    with _LOCK:
        data = _read_status()
        enriched = dict(payload)
        enriched["updated_at"] = time.time()
        data[section] = enriched
        _write_status(data)


def get_status(section: Optional[str] = None) -> Dict[str, Any]:
    """Return cached status for ``section`` or the entire document."""

    data = _read_status()
    if section is None:
        return data
    return data.get(section, {})

