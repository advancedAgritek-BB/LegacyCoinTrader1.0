"""Public interface for the bot configuration system."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from .settings import (
    BotSettings,
    ConfigError,
    DEFAULT_CONFIG_PATH,
    load_config,
    load_settings,
    resolve_config_path,
)

__all__ = [
    "BotSettings",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "load_settings",
    "resolve_config_path",
    "save_config",
    "validate_config",
]


def save_config(
    config: Mapping[str, Any],
    config_path: Union[str, Path, None] = None,
) -> Path:
    """Persist a validated configuration to a YAML override file."""

    path = resolve_config_path(config_path)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    return Path(path)


def validate_config(config: Mapping[str, Any]) -> bool:
    """Return ``True`` when ``config`` conforms to :class:`BotSettings`."""

    try:
        BotSettings(**config)
    except ValidationError:
        return False
    return True
