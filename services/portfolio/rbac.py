"""Shared helpers for service-level RBAC definitions."""

from __future__ import annotations

import json
import os
from typing import Dict, List

DEFAULT_ROLE_DEFINITIONS: Dict[str, List[str]] = {
    "admin": [
        "system:admin",
        "portfolio:read",
        "portfolio:write",
        "trading:execute",
        "monitoring:read",
    ],
    "trader": [
        "portfolio:read",
        "portfolio:write",
        "trading:execute",
        "monitoring:read",
    ],
    "viewer": [
        "portfolio:read",
        "monitoring:read",
    ],
}


def load_role_definitions(raw: str | None = None) -> Dict[str, List[str]]:
    """Load role definitions from JSON or fallback to defaults."""

    value = raw if raw is not None else os.getenv("ROLE_DEFINITIONS")
    if not value:
        return {role: scopes.copy() for role, scopes in DEFAULT_ROLE_DEFINITIONS.items()}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("ROLE_DEFINITIONS must contain valid JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("ROLE_DEFINITIONS must be a JSON object mapping roles to scopes")

    role_definitions: Dict[str, List[str]] = {}
    for role, scopes in parsed.items():
        if not isinstance(scopes, (list, tuple)):
            raise ValueError("Each role must map to an array of scopes")
        role_definitions[str(role)] = [str(scope) for scope in scopes]

    if not role_definitions:
        raise ValueError("At least one role definition must be provided")

    return role_definitions


__all__ = ["DEFAULT_ROLE_DEFINITIONS", "load_role_definitions"]
