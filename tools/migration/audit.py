"""Post-migration data integrity auditing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


@dataclass
class DriftRecord:
    tenant_id: str
    resource: str
    record_id: str
    differences: Dict[str, Tuple[Any, Any]]


@dataclass
class MigrationAuditReport:
    """Structured result of a migration audit."""

    matched_records: int
    mismatched_records: int
    missing_in_modern: List[str] = field(default_factory=list)
    missing_in_legacy: List[str] = field(default_factory=list)
    drifts: List[DriftRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        return not (self.mismatched_records or self.missing_in_modern or self.missing_in_legacy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_records": self.matched_records,
            "mismatched_records": self.mismatched_records,
            "missing_in_modern": self.missing_in_modern,
            "missing_in_legacy": self.missing_in_legacy,
            "drifts": [
                {
                    "tenant_id": drift.tenant_id,
                    "resource": drift.resource,
                    "record_id": drift.record_id,
                    "differences": {key: [legacy, modern] for key, (legacy, modern) in drift.differences.items()},
                }
                for drift in self.drifts
            ],
            "metadata": self.metadata,
        }

    def write(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return target


class DataIntegrityAuditor:
    """Compare legacy and modern data snapshots."""

    def __init__(self, *, tolerance: float = 0.0) -> None:
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    def load_snapshot(self, path: Union[str, Path]) -> Dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        raw = file_path.read_text()
        if file_path.suffix.lower() in {".yml", ".yaml"}:
            return json.loads(json.dumps(__import__("yaml").safe_load(raw)))
        return json.loads(raw)

    # ------------------------------------------------------------------
    def compare(self, legacy_snapshot: Mapping[str, Any], modern_snapshot: Mapping[str, Any]) -> MigrationAuditReport:
        legacy_index = self._build_index(legacy_snapshot)
        modern_index = self._build_index(modern_snapshot)

        matched = 0
        mismatched = 0
        missing_in_modern: List[str] = []
        missing_in_legacy: List[str] = []
        drifts: List[DriftRecord] = []

        for key, legacy_record in legacy_index.items():
            if key not in modern_index:
                missing_in_modern.append(self._format_key(key))
                continue
            modern_record = modern_index[key]
            if self._records_close(legacy_record, modern_record):
                matched += 1
            else:
                mismatched += 1
                differences = self._diff_record(legacy_record, modern_record)
                drifts.append(
                    DriftRecord(
                        tenant_id=key[0],
                        resource=key[1],
                        record_id=key[2],
                        differences=differences,
                    )
                )
        for key in modern_index:
            if key not in legacy_index:
                missing_in_legacy.append(self._format_key(key))

        return MigrationAuditReport(
            matched_records=matched,
            mismatched_records=mismatched,
            missing_in_modern=missing_in_modern,
            missing_in_legacy=missing_in_legacy,
            drifts=drifts,
            metadata={
                "legacy_records": len(legacy_index),
                "modern_records": len(modern_index),
                "tolerance": self.tolerance,
            },
        )

    # ------------------------------------------------------------------
    def _build_index(self, snapshot: Mapping[str, Any]) -> Dict[Tuple[str, str, str], Any]:
        index: Dict[Tuple[str, str, str], Any] = {}
        tenants = snapshot.get("tenants", {}) if isinstance(snapshot, Mapping) else snapshot
        for tenant_id, tenant_payload in tenants.items():
            if not isinstance(tenant_payload, Mapping):
                index[(str(tenant_id), "root", "root")] = tenant_payload
                continue
            for resource, value in tenant_payload.items():
                for record_id, payload in self._iterate_records(value):
                    index[(str(tenant_id), str(resource), record_id)] = payload
        return index

    def _iterate_records(self, value: Any) -> Iterable[Tuple[str, Any]]:
        if isinstance(value, list):
            for item in value:
                yield self._record_key(item), item
        elif isinstance(value, Mapping):
            if all(isinstance(v, Mapping) for v in value.values()):
                for key, item in value.items():
                    yield str(key), item
            else:
                yield "singleton", value
        else:
            yield "singleton", value

    def _record_key(self, payload: Any) -> str:
        if isinstance(payload, Mapping):
            for key in ("id", "uuid", "external_id", "account_id"):
                if key in payload:
                    return str(payload[key])
            digest = md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            return f"hash:{digest}"
        return str(payload)

    def _records_close(self, legacy_record: Any, modern_record: Any) -> bool:
        if legacy_record == modern_record:
            return True
        if isinstance(legacy_record, (int, float)) and isinstance(modern_record, (int, float)):
            return abs(float(legacy_record) - float(modern_record)) <= self.tolerance
        if isinstance(legacy_record, Mapping) and isinstance(modern_record, Mapping):
            if set(legacy_record.keys()) != set(modern_record.keys()):
                return False
            return all(
                self._records_close(legacy_record[key], modern_record[key])
                for key in legacy_record
            )
        if isinstance(legacy_record, list) and isinstance(modern_record, list):
            if len(legacy_record) != len(modern_record):
                return False
            return all(
                self._records_close(left, right)
                for left, right in zip(sorted(legacy_record, key=str), sorted(modern_record, key=str))
            )
        return False

    def _diff_record(self, legacy_record: Any, modern_record: Any) -> Dict[str, Tuple[Any, Any]]:
        if isinstance(legacy_record, Mapping) and isinstance(modern_record, Mapping):
            differences: Dict[str, Tuple[Any, Any]] = {}
            for key in sorted(set(legacy_record.keys()) | set(modern_record.keys())):
                legacy_value = legacy_record.get(key)
                modern_value = modern_record.get(key)
                if not self._records_close(legacy_value, modern_value):
                    differences[key] = (legacy_value, modern_value)
            return differences
        return {"value": (legacy_record, modern_record)}

    def _format_key(self, key: Tuple[str, str, str]) -> str:
        tenant, resource, record_id = key
        return f"tenant={tenant} resource={resource} record={record_id}"
