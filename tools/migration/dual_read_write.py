"""Dual read/write shims to support hybrid legacy + microservice deployments."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol


class DataAdapter(Protocol):
    """Protocol describing a storage adapter."""

    def read(self, resource: str, tenant_id: str, **kwargs: Any) -> Any:  # pragma: no cover - Protocol signature
        ...

    def write(self, resource: str, tenant_id: str, payload: Any, **kwargs: Any) -> Any:  # pragma: no cover - Protocol signature
        ...

    def delete(self, resource: str, tenant_id: str, **kwargs: Any) -> Any:  # pragma: no cover - Protocol signature
        ...


class DualWriteStrategy(str, Enum):
    """Strategies for dual-write behaviour."""

    MIRROR = "mirror"  # Write to both stores
    LEGACY_PRIMARY = "legacy-primary"  # Legacy is source of truth, modern is best effort
    MODERN_PRIMARY = "modern-primary"  # Modern is source of truth, legacy receives backfill
    LEGACY_ONLY = "legacy-only"
    MODERN_ONLY = "modern-only"


class ReadStrategy(str, Enum):
    """Strategies for dual-read behaviour."""

    PREFER_LEGACY = "prefer-legacy"
    PREFER_MODERN = "prefer-modern"
    COMPARE = "compare"  # Read both and compare for drift


@dataclass
class FeatureFlagState:
    """Stateful feature flags powering the shim."""

    hybrid_mode_enabled: bool = False
    prefer_legacy_reads: bool = False
    dual_write_strategy: DualWriteStrategy = DualWriteStrategy.MIRROR
    cutover_complete: bool = False
    guardrail_enabled: bool = True
    drift_tolerance: float = 0.0

    def as_env(self) -> Dict[str, str]:
        """Return an environment-compatible representation of the flags."""

        return {
            "HYBRID_MODE_ENABLED": str(self.hybrid_mode_enabled).lower(),
            "LEGACY_READ_ENABLED": str(self.prefer_legacy_reads or self.hybrid_mode_enabled).lower(),
            "DUAL_WRITE_STRATEGY": self.dual_write_strategy.value,
            "CUTOVER_COMPLETED": str(self.cutover_complete).lower(),
            "CUTOVER_GUARDRAIL_ENABLED": str(self.guardrail_enabled).lower(),
            "MIGRATION_DRIFT_TOLERANCE": str(self.drift_tolerance),
        }


AuditCallback = Callable[[str, str, Any, Any], None]


@dataclass
class DualReadWriteShim:
    """Coordinate reads and writes during the hybrid migration phase."""

    legacy_adapter: DataAdapter
    modern_adapter: DataAdapter
    feature_flags: FeatureFlagState = field(default_factory=FeatureFlagState)
    audit_callback: Optional[AuditCallback] = None
    _callback_expects_tuple: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.audit_callback is not None:
            try:
                parameters = inspect.signature(self.audit_callback).parameters
                self._callback_expects_tuple = len(parameters) == 1
            except (TypeError, ValueError):
                self._callback_expects_tuple = False

    def read(
        self,
        resource: str,
        tenant_id: str,
        *,
        strategy: Optional[ReadStrategy] = None,
        **kwargs: Any,
    ) -> Any:
        """Read data according to the configured hybrid strategy."""

        if not self.feature_flags.hybrid_mode_enabled or self.feature_flags.cutover_complete:
            return self.modern_adapter.read(resource, tenant_id, **kwargs)

        strategy = strategy or (
            ReadStrategy.PREFER_LEGACY if self.feature_flags.prefer_legacy_reads else ReadStrategy.COMPARE
        )
        legacy_result = None
        modern_result = None

        if strategy is ReadStrategy.PREFER_LEGACY:
            legacy_result = self.legacy_adapter.read(resource, tenant_id, **kwargs)
            modern_result = self.modern_adapter.read(resource, tenant_id, **kwargs)
            self._audit(resource, tenant_id, legacy_result, modern_result)
            return legacy_result
        if strategy is ReadStrategy.PREFER_MODERN:
            legacy_result = self.legacy_adapter.read(resource, tenant_id, **kwargs)
            modern_result = self.modern_adapter.read(resource, tenant_id, **kwargs)
            self._audit(resource, tenant_id, legacy_result, modern_result)
            return modern_result

        # Compare strategy (default) reads both, audits and returns modern data
        legacy_result = self.legacy_adapter.read(resource, tenant_id, **kwargs)
        modern_result = self.modern_adapter.read(resource, tenant_id, **kwargs)
        self._audit(resource, tenant_id, legacy_result, modern_result)
        return modern_result

    def write(
        self,
        resource: str,
        tenant_id: str,
        payload: Any,
        *,
        strategy: Optional[DualWriteStrategy] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Write to both data stores according to the configured strategy."""

        strategy = strategy or self.feature_flags.dual_write_strategy
        results: Dict[str, Any] = {}

        if strategy is DualWriteStrategy.LEGACY_ONLY:
            results["legacy"] = self.legacy_adapter.write(resource, tenant_id, payload, **kwargs)
            results["modern"] = None
            return results
        if strategy is DualWriteStrategy.MODERN_ONLY:
            results["legacy"] = None
            results["modern"] = self.modern_adapter.write(resource, tenant_id, payload, **kwargs)
            return results

        # For mirror, legacy-primary and modern-primary we attempt to write to both
        results["legacy"] = self.legacy_adapter.write(resource, tenant_id, payload, **kwargs)
        results["modern"] = self.modern_adapter.write(resource, tenant_id, payload, **kwargs)

        if self.feature_flags.guardrail_enabled and results.get("legacy") != results.get("modern"):
            self._audit(resource, tenant_id, results.get("legacy"), results.get("modern"))
        return results

    def reconcile(
        self,
        resource: str,
        tenant_id: str,
        *,
        raise_on_drift: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compare legacy and modern data and optionally raise on drift."""

        legacy_result = self.legacy_adapter.read(resource, tenant_id, **kwargs)
        modern_result = self.modern_adapter.read(resource, tenant_id, **kwargs)
        drift_detected = legacy_result != modern_result
        if drift_detected and raise_on_drift and self.feature_flags.guardrail_enabled:
            raise RuntimeError(
                f"Data drift detected for resource '{resource}' tenant '{tenant_id}': "
                f"legacy={legacy_result!r} modern={modern_result!r}"
            )
        self._audit(resource, tenant_id, legacy_result, modern_result)
        return {
            "resource": resource,
            "tenant_id": tenant_id,
            "legacy": legacy_result,
            "modern": modern_result,
            "drift_detected": drift_detected,
        }

    def cutover(self) -> None:
        """Mark the cutover as completed and disable legacy paths."""

        self.feature_flags.cutover_complete = True
        self.feature_flags.hybrid_mode_enabled = False

    # ------------------------------------------------------------------
    def _audit(self, resource: str, tenant_id: str, legacy_value: Any, modern_value: Any) -> None:
        if not self.audit_callback:
            return
        if self._values_close(legacy_value, modern_value):
            return
        if self._callback_expects_tuple:
            self.audit_callback((resource, tenant_id, legacy_value, modern_value))
        else:
            self.audit_callback(resource, tenant_id, legacy_value, modern_value)

    # ------------------------------------------------------------------
    def _values_close(self, legacy_value: Any, modern_value: Any) -> bool:
        if legacy_value == modern_value:
            return True
        tolerance = self.feature_flags.drift_tolerance
        if isinstance(legacy_value, (int, float)) and isinstance(modern_value, (int, float)):
            return abs(float(legacy_value) - float(modern_value)) <= tolerance
        if isinstance(legacy_value, dict) and isinstance(modern_value, dict):
            legacy_keys = set(legacy_value.keys())
            modern_keys = set(modern_value.keys())
            if legacy_keys != modern_keys:
                return False
            return all(
                self._values_close(legacy_value[key], modern_value[key]) for key in legacy_keys
            )
        if isinstance(legacy_value, (list, tuple)) and isinstance(modern_value, (list, tuple)):
            if len(legacy_value) != len(modern_value):
                return False
            return all(
                self._values_close(left, right)
                for left, right in zip(legacy_value, modern_value)
            )
        return False
