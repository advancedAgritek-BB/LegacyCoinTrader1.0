"""CLI utilities for migrating LegacyCoinTrader tenants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.migration.audit import DataIntegrityAuditor  # type: ignore
    from tools.migration.config_translator import ConfigTranslator  # type: ignore
    from tools.migration.dual_read_write import FeatureFlagState  # type: ignore
    from tools.migration.schema_migrator import SchemaMigrator  # type: ignore
else:
    from .audit import DataIntegrityAuditor
    from .config_translator import ConfigTranslator
    from .dual_read_write import FeatureFlagState
    from .schema_migrator import SchemaMigrator


def _load_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text()
    if path.suffix.lower() in {".yml", ".yaml"}:
        import yaml  # Local import to avoid mandatory dependency for consumers

        return yaml.safe_load(raw) or {}
    return json.loads(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tenant migration toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("generate-schema", help="Generate multi-tenant schema SQL")
    schema_parser.add_argument("--schema", required=True, type=Path, help="Path to legacy schema metadata (JSON/YAML)")
    schema_parser.add_argument("--output", required=True, type=Path, help="Where to write the generated SQL")
    schema_parser.add_argument("--tenant-id-type", default="UUID", help="Column type to use for tenant identifiers")

    plan_parser = subparsers.add_parser("migration-plan", help="Generate migration SQL for tenant mappings")
    plan_parser.add_argument("--schema", required=True, type=Path, help="Legacy schema metadata file")
    plan_parser.add_argument("--mapping", required=True, type=Path, help="Tenant mapping definition (JSON/YAML)")
    plan_parser.add_argument("--output", required=True, type=Path, help="Output SQL file")

    translate_parser = subparsers.add_parser("translate-config", help="Translate legacy configuration for tenants")
    translate_parser.add_argument("--base-config", required=True, type=Path, help="Legacy configuration file")
    translate_parser.add_argument("--tenant-overrides", required=True, type=Path, help="Tenant override definitions")
    translate_parser.add_argument(
        "--feature-flags",
        type=Path,
        help="Optional feature flag definitions",
    )
    translate_parser.add_argument("--output-dir", required=True, type=Path, help="Directory for translated configs")

    audit_parser = subparsers.add_parser("audit", help="Validate data integrity between legacy and modern snapshots")
    audit_parser.add_argument("--legacy-snapshot", required=True, type=Path)
    audit_parser.add_argument("--modern-snapshot", required=True, type=Path)
    audit_parser.add_argument("--report-path", required=True, type=Path)
    audit_parser.add_argument("--tolerance", type=float, default=0.0, help="Numeric tolerance for comparison")
    audit_parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 2 when drift or missing records are detected",
    )

    flags_parser = subparsers.add_parser("flags", help="Render hybrid feature flags as environment variables")
    flags_parser.add_argument("--hybrid", action="store_true", help="Enable hybrid mode")
    flags_parser.add_argument("--prefer-legacy", action="store_true", help="Prefer legacy reads")
    flags_parser.add_argument(
        "--strategy",
        choices=["mirror", "legacy-primary", "modern-primary", "legacy-only", "modern-only"],
        default="mirror",
    )
    flags_parser.add_argument("--cutover-complete", action="store_true", help="Mark cutover as complete")
    flags_parser.add_argument("--guardrail", action="store_true", help="Enable guardrail enforcement")
    flags_parser.add_argument("--tolerance", type=float, default=0.0, help="Allowed drift tolerance")

    return parser


def handle_generate_schema(args: argparse.Namespace) -> None:
    migrator = SchemaMigrator.from_file(args.schema, tenant_id_type=args.tenant_id_type)
    migrator.dump_schema(args.output)
    print(f"Generated schema written to {args.output}")


def handle_migration_plan(args: argparse.Namespace) -> None:
    migrator = SchemaMigrator.from_file(args.schema)
    mapping = _load_mapping(args.mapping)
    tenant_mapping = mapping.get("tenants", mapping)
    if not isinstance(tenant_mapping, dict):
        raise ValueError("Tenant mapping must be a dictionary")
    migrator.dump_plan(tenant_mapping, args.output)
    print(f"Migration plan written to {args.output}")


def handle_translate_config(args: argparse.Namespace) -> None:
    translator = ConfigTranslator.from_files(
        base_config_path=args.base_config,
        tenant_overrides_path=args.tenant_overrides,
        feature_flags_path=args.feature_flags,
    )
    master_path, per_tenant = translator.write_outputs(args.output_dir)
    print(f"Master config written to {master_path}")
    if per_tenant:
        print("Per-tenant configs:")
        for tenant, path in per_tenant.items():
            print(f"  {tenant}: {path}")


def handle_audit(args: argparse.Namespace) -> int:
    auditor = DataIntegrityAuditor(tolerance=args.tolerance)
    legacy = auditor.load_snapshot(args.legacy_snapshot)
    modern = auditor.load_snapshot(args.modern_snapshot)
    report = auditor.compare(legacy, modern)
    report.write(args.report_path)
    print(json.dumps(report.to_dict(), indent=2))
    if args.fail_on_drift and not report.is_successful:
        print("Drift detected during migration audit", file=sys.stderr)
        return 2
    return 0


def handle_flags(args: argparse.Namespace) -> None:
    from .dual_read_write import DualWriteStrategy

    flags = FeatureFlagState(
        hybrid_mode_enabled=args.hybrid,
        prefer_legacy_reads=args.prefer_legacy,
        dual_write_strategy=DualWriteStrategy(args.strategy),
        cutover_complete=args.cutover_complete,
        guardrail_enabled=args.guardrail or args.cutover_complete,
        drift_tolerance=args.tolerance,
    )
    for key, value in flags.as_env().items():
        print(f"{key}={value}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-schema":
        handle_generate_schema(args)
        return 0
    if args.command == "migration-plan":
        handle_migration_plan(args)
        return 0
    if args.command == "translate-config":
        handle_translate_config(args)
        return 0
    if args.command == "audit":
        return handle_audit(args)
    if args.command == "flags":
        handle_flags(args)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
