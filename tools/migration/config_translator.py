"""Translate legacy configuration files into the multi-tenant format."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import yaml


def _read_structure(path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    raw = file_path.read_text()
    if file_path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(raw) or {}
    return json.loads(raw)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge dictionaries recursively without mutating the inputs."""

    merged: Dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class ConfigTranslator:
    """Translate a legacy single-tenant configuration into tenant-scoped files."""

    def __init__(
        self,
        base_config: Mapping[str, Any],
        tenant_overrides: Mapping[str, Mapping[str, Any]],
        *,
        feature_flags: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.base_config = deepcopy(base_config)
        self.tenant_overrides = {key: deepcopy(value) for key, value in tenant_overrides.items()}
        self.feature_flags = dict(feature_flags or {})
        self._translated: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        base_config_path: Union[str, Path],
        tenant_overrides_path: Union[str, Path],
        *,
        feature_flags_path: Optional[Union[str, Path]] = None,
    ) -> "ConfigTranslator":
        base = _read_structure(base_config_path)
        overrides = _read_structure(tenant_overrides_path)
        flags: Optional[Dict[str, Any]] = None
        if feature_flags_path:
            raw_flags = _read_structure(feature_flags_path)
            if isinstance(raw_flags, Mapping) and "feature_flags" in raw_flags:
                candidate = raw_flags.get("feature_flags")
                flags = dict(candidate) if isinstance(candidate, Mapping) else None
            elif isinstance(raw_flags, Mapping):
                flags = dict(raw_flags)
        tenants = overrides.get("tenants") if isinstance(overrides, Mapping) else overrides
        if not isinstance(tenants, Mapping):
            raise ValueError("Tenant overrides file must contain a 'tenants' mapping")
        feature_flag_payload = flags or overrides.get("feature_flags", {})
        if isinstance(feature_flag_payload, Mapping):
            feature_flag_payload = dict(feature_flag_payload)
        else:
            feature_flag_payload = {}
        return cls(base, tenants, feature_flags=feature_flag_payload)

    # ------------------------------------------------------------------
    def translate(self) -> Dict[str, Any]:
        """Create the new configuration structure."""

        tenants_config: Dict[str, Any] = {}
        for tenant, overrides in self.tenant_overrides.items():
            merged = _deep_merge(self.base_config, overrides)
            tenants_config[tenant] = merged
        self._translated = {
            "global": {
                "feature_flags": self.feature_flags,
                "legacy_config_version": self.base_config.get("version", "1.0"),
            },
            "tenants": tenants_config,
        }
        return deepcopy(self._translated)

    # ------------------------------------------------------------------
    def translated(self) -> Dict[str, Any]:
        if self._translated is None:
            return self.translate()
        return deepcopy(self._translated)

    # ------------------------------------------------------------------
    def write_outputs(
        self,
        output_dir: Union[str, Path],
        *,
        include_per_tenant: bool = True,
    ) -> Tuple[Path, Dict[str, Path]]:
        """Write the translated configuration to disk."""

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        master_path = directory / "multi_tenant_config.yaml"
        yaml.safe_dump(self.translated(), master_path.open("w"), sort_keys=False)
        per_tenant_files: Dict[str, Path] = {}
        if include_per_tenant:
            for tenant, config in self.translated()["tenants"].items():
                tenant_path = directory / f"tenant_{tenant}.yaml"
                yaml.safe_dump(config, tenant_path.open("w"), sort_keys=False)
                per_tenant_files[tenant] = tenant_path
        return master_path, per_tenant_files

    # ------------------------------------------------------------------
    def diff(self, tenant: str) -> Dict[str, Tuple[Any, Any]]:
        """Return a key-level diff between the base configuration and overrides."""

        if tenant not in self.tenant_overrides:
            raise KeyError(f"Unknown tenant '{tenant}'")
        diffs: Dict[str, Tuple[Any, Any]] = {}
        overrides = self.tenant_overrides[tenant]
        for key, value in overrides.items():
            base_value = self.base_config.get(key)
            if isinstance(value, Mapping) and isinstance(base_value, Mapping):
                nested = ConfigTranslator(base_value, {tenant: value}).diff(tenant)
                for nested_key, nested_diff in nested.items():
                    diffs[f"{key}.{nested_key}"] = nested_diff
            else:
                if base_value != value:
                    diffs[key] = (base_value, value)
        return diffs

    # ------------------------------------------------------------------
    def build_runtime_payload(self) -> Dict[str, Any]:
        """Return a payload optimised for runtime consumption."""

        translated = self.translated()
        payload = {
            "feature_flags": translated["global"].get("feature_flags", {}),
            "tenants": [],
        }
        for tenant, config in translated["tenants"].items():
            payload["tenants"].append(
                {
                    "tenant_id": tenant,
                    "config": config,
                    "overrides": self.tenant_overrides.get(tenant, {}),
                }
            )
        return payload
