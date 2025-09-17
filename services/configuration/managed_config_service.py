"""Managed configuration and secret resolution utilities."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml


class SecretNotFoundError(RuntimeError):
    """Raised when a required managed secret cannot be resolved."""


@dataclass(frozen=True)
class ManagedSecretSpec:
    """Definition describing how to inject a managed secret into config data."""

    path: str
    env: str
    required: bool = False
    description: Optional[str] = None


@dataclass(frozen=True)
class EnvironmentSecretSpec:
    """Definition for required environment variables populated by secret managers."""

    name: str
    required: bool = False
    description: Optional[str] = None


@dataclass(frozen=True)
class ManagedSecretsManifest:
    """Structured representation of ``config/managed_secrets.yaml``."""

    config_secrets: tuple[ManagedSecretSpec, ...]
    environment_secrets: tuple[EnvironmentSecretSpec, ...]

    def required_environment_variables(self) -> tuple[str, ...]:
        """Return the names of environment variables marked as required."""

        return tuple(
            spec.name for spec in self.environment_secrets if spec.required
        )

    def environment_variable_names(self) -> tuple[str, ...]:
        """Return all environment variables referenced by the manifest."""

        names = {spec.name for spec in self.environment_secrets}
        names.update(spec.env for spec in self.config_secrets)
        return tuple(sorted(names))


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "managed_secrets.yaml"
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Managed secrets manifest {path} must contain a mapping")
    return data


def load_manifest(path: Path | None = None) -> ManagedSecretsManifest:
    """Load the managed secrets manifest from ``config/managed_secrets.yaml``."""

    manifest_path = Path(path) if path else DEFAULT_MANIFEST_PATH
    data = _load_yaml(manifest_path)
    managed = data.get("managed_secrets", {})

    config_specs: list[ManagedSecretSpec] = []
    for item in managed.get("config", []) or []:
        if not isinstance(item, Mapping):
            continue
        try:
            path_value = str(item["path"])
            env_value = str(item["env"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"Missing required key {exc!s} in managed secrets config entry"
            ) from exc
        config_specs.append(
            ManagedSecretSpec(
                path=path_value,
                env=env_value,
                required=bool(item.get("required", False)),
                description=item.get("description"),
            )
        )

    env_specs: list[EnvironmentSecretSpec] = []
    for item in managed.get("environment", []) or []:
        if not isinstance(item, Mapping):
            continue
        try:
            name_value = str(item["name"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"Missing required key {exc!s} in managed secrets env entry"
            ) from exc
        env_specs.append(
            EnvironmentSecretSpec(
                name=name_value,
                required=bool(item.get("required", False)),
                description=item.get("description"),
            )
        )

    return ManagedSecretsManifest(
        config_secrets=tuple(config_specs),
        environment_secrets=tuple(env_specs),
    )


class ManagedSecretsClient:
    """Client that resolves managed secrets from environment injection."""

    def __init__(
        self,
        manifest: ManagedSecretsManifest,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._manifest = manifest
        self._env = env or os.environ

    def get(
        self,
        env_name: str,
        *,
        required: bool = False,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Return the secret bound to ``env_name`` or ``default`` when optional."""

        value = self._env.get(env_name)
        if value is None:
            if required:
                raise SecretNotFoundError(
                    f"Managed secret {env_name} is not available in the environment"
                )
            return default

        if isinstance(value, str) and not value.strip():
            if required:
                raise SecretNotFoundError(
                    f"Managed secret {env_name} is empty and marked as required"
                )
            return default

        return value

    def missing_required_environment(self) -> tuple[str, ...]:
        """Return required environment variable names that are currently unset."""

        missing = []
        for spec in self._manifest.environment_secrets:
            if not spec.required:
                continue
            value = self._env.get(spec.name)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(spec.name)
        return tuple(missing)


def _set_deep_value(target: MutableMapping[str, Any], path: Iterable[str], value: Any) -> None:
    current: MutableMapping[str, Any] = target
    keys = list(path)
    for key in keys[:-1]:
        existing = current.get(key)
        if not isinstance(existing, MutableMapping):
            current[key] = {}
        current = current[key]  # type: ignore[assignment]
    current[keys[-1]] = value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``base`` with ``override`` returning a new dictionary."""

    result: Dict[str, Any] = {}
    base_keys = set(base.keys())
    override_keys = set(override.keys())

    for key in base_keys | override_keys:
        base_value = base.get(key)
        override_value = override.get(key)

        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            result[key] = deep_merge(base_value, override_value)
        elif key in override:
            result[key] = copy.deepcopy(override_value)
        elif key in base:
            result[key] = copy.deepcopy(base_value)

    return result


class ManagedConfigService:
    """Service that merges configuration dictionaries with managed secrets."""

    def __init__(
        self,
        *,
        manifest: ManagedSecretsManifest | None = None,
        manifest_path: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if manifest is None:
            manifest = load_manifest(manifest_path)
        self._manifest = manifest
        self._client = ManagedSecretsClient(manifest, env=env)

    @property
    def manifest(self) -> ManagedSecretsManifest:
        return self._manifest

    def merge(self, base_config: Mapping[str, Any]) -> Dict[str, Any]:
        """Return ``base_config`` merged with resolved secret overrides."""

        overrides: Dict[str, Any] = {}
        for spec in self._manifest.config_secrets:
            value = self._client.get(spec.env, required=spec.required)
            if value is None:
                continue
            path_segments = tuple(segment.strip() for segment in spec.path.split(".") if segment)
            if not path_segments:
                continue
            _set_deep_value(overrides, path_segments, value)
        return deep_merge(base_config, overrides)

    def load(self, base_config_path: Path | str) -> Dict[str, Any]:
        """Load a YAML config file and merge it with managed secrets."""

        base = _load_yaml(Path(base_config_path))
        return self.merge(base)

    def missing_environment(self) -> tuple[str, ...]:
        """Return required environment variables that must be populated."""

        return self._client.missing_required_environment()


__all__ = [
    "EnvironmentSecretSpec",
    "ManagedConfigService",
    "ManagedSecretSpec",
    "ManagedSecretsClient",
    "ManagedSecretsManifest",
    "SecretNotFoundError",
    "DEFAULT_MANIFEST_PATH",
    "deep_merge",
    "load_manifest",
]
