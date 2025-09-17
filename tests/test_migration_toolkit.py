import json
from pathlib import Path

import pytest

from tools.migration import (
    ConfigTranslator,
    DataIntegrityAuditor,
    DualReadWriteShim,
    DualWriteStrategy,
    FeatureFlagState,
    ReadStrategy,
    SchemaMigrator,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "migration"


class StubAdapter:
    def __init__(self) -> None:
        self.storage = {}
        self.writes = []

    def read(self, resource: str, tenant_id: str, **_: str):
        return self.storage.get((resource, tenant_id))

    def write(self, resource: str, tenant_id: str, payload, **_: str):
        self.storage[(resource, tenant_id)] = payload
        self.writes.append((resource, tenant_id, payload))
        return payload

    def delete(self, resource: str, tenant_id: str, **_: str):
        self.storage.pop((resource, tenant_id), None)


def load_fixture(name: str) -> Path:
    return FIXTURES / name


def test_schema_migrator_generates_multi_tenant_sql(tmp_path: Path):
    migrator = SchemaMigrator.from_file(load_fixture("legacy_schema.yaml"))
    sql = migrator.generate_schema_sql()
    assert 'CREATE TABLE IF NOT EXISTS "tenant_accounts"' in sql
    assert '"tenant_id" UUID NOT NULL' in sql
    assert 'PRIMARY KEY ("tenant_id", "account_id")' in sql
    plan = migrator.generate_migration_plan(
        {
            "tenant-a": {"legacy_accounts": ["111", "222"]},
            "tenant-b": {"legacy_account": "333"},
        }
    )
    assert any('INSERT INTO "tenant_accounts"' in statement for statement in plan)
    assert any("tenant-b" in statement for statement in plan)


def test_config_translator_merges_overrides(tmp_path: Path):
    translator = ConfigTranslator.from_files(
        load_fixture("base_config.yaml"),
        load_fixture("tenant_overrides.yaml"),
        feature_flags_path=load_fixture("feature_flags.yaml"),
    )
    translated = translator.translate()
    assert translated["tenants"]["tenant-a"]["risk"]["max_position"] == 10
    assert translated["tenants"]["tenant-b"]["risk"]["stop_loss_pct"] == 0.08
    assert translated["global"]["feature_flags"]["HYBRID_MODE_ENABLED"] is True
    master, per_tenant = translator.write_outputs(tmp_path)
    assert master.exists()
    assert set(per_tenant.keys()) == {"tenant-a", "tenant-b"}


def test_dual_read_write_shim_handles_hybrid_reads_and_writes():
    legacy = StubAdapter()
    modern = StubAdapter()
    audit_records = []

    flags = FeatureFlagState(hybrid_mode_enabled=True, prefer_legacy_reads=True)
    shim = DualReadWriteShim(legacy, modern, feature_flags=flags, audit_callback=audit_records.append)

    legacy.write("balances", "tenant-a", {"available": 10})
    modern.write("balances", "tenant-a", {"available": 9.5})

    result = shim.read("balances", "tenant-a", strategy=ReadStrategy.COMPARE)
    assert result == {"available": 9.5}
    assert audit_records  # mismatch recorded

    shim.write(
        "balances",
        "tenant-a",
        {"available": 11},
        strategy=DualWriteStrategy.MIRROR,
    )
    assert legacy.read("balances", "tenant-a") == {"available": 11}
    assert modern.read("balances", "tenant-a") == {"available": 11}

    reconciliation = shim.reconcile("balances", "tenant-a")
    assert reconciliation["drift_detected"] is False


def test_data_integrity_auditor_detects_drift(tmp_path: Path):
    auditor = DataIntegrityAuditor()
    legacy_snapshot = auditor.load_snapshot(load_fixture("legacy_snapshot.json"))
    modern_snapshot = auditor.load_snapshot(load_fixture("modern_snapshot.json"))
    report = auditor.compare(legacy_snapshot, modern_snapshot)
    assert report.is_successful

    # Introduce drift
    modern_snapshot["tenants"]["tenant-a"]["balances"][0]["available"] = 1.5
    drift_report = auditor.compare(legacy_snapshot, modern_snapshot)
    assert not drift_report.is_successful
    assert drift_report.mismatched_records == 1
    report_path = tmp_path / "report.json"
    drift_report.write(report_path)
    saved = json.loads(report_path.read_text())
    assert saved["mismatched_records"] == 1
