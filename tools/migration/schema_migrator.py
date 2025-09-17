"""Utilities for generating multi-tenant database schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import yaml


@dataclass
class ColumnDefinition:
    """Representation of a database column."""

    name: str
    type: str
    nullable: bool = True
    default: Optional[str] = None
    primary_key: bool = False
    unique: bool = False

    def to_sql(self) -> str:
        """Render the column as SQL."""

        parts: List[str] = [f'"{self.name}" {self.type}']
        if not self.nullable:
            parts.append("NOT NULL")
        if self.unique:
            parts.append("UNIQUE")
        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")
        return " ".join(parts)


@dataclass
class TableDefinition:
    """Representation of a legacy table that should become multi-tenant."""

    name: str
    columns: List[ColumnDefinition]
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    tenant_lookup_key: Optional[str] = None
    comment: Optional[str] = None

    def primary_key_columns(self) -> List[str]:
        return [column.name for column in self.columns if column.primary_key]

    def clone(self, *, name: Optional[str] = None) -> "TableDefinition":
        """Return a copy of the table definition."""

        return TableDefinition(
            name=name or self.name,
            columns=[ColumnDefinition(**column.__dict__) for column in self.columns],
            indexes=[dict(index) for index in self.indexes],
            tenant_lookup_key=self.tenant_lookup_key,
            comment=self.comment,
        )


class SchemaMigrator:
    """Generate SQL statements to migrate to the multi-tenant schema."""

    def __init__(
        self,
        legacy_schema: Dict[str, Any],
        *,
        tenant_id_type: str = "UUID",
        audit_columns: Optional[Sequence[ColumnDefinition]] = None,
    ) -> None:
        self.legacy_schema = legacy_schema
        self.tenant_id_type = tenant_id_type
        self.audit_columns = list(
            audit_columns
            or (
                ColumnDefinition(
                    name="migrated_at",
                    type="TIMESTAMPTZ",
                    nullable=False,
                    default="CURRENT_TIMESTAMP",
                ),
                ColumnDefinition(
                    name="source_legacy_pk",
                    type="TEXT",
                    nullable=False,
                ),
                ColumnDefinition(
                    name="source_hash",
                    type="TEXT",
                    nullable=False,
                ),
            )
        )
        self.tables = [self._parse_table(item) for item in legacy_schema.get("tables", [])]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        tenant_id_type: str = "UUID",
        audit_columns: Optional[Sequence[ColumnDefinition]] = None,
    ) -> "SchemaMigrator":
        """Load schema metadata from JSON or YAML."""

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        raw_data: Dict[str, Any]
        if file_path.suffix.lower() in {".yml", ".yaml"}:
            raw_data = yaml.safe_load(file_path.read_text())
        else:
            raw_data = json.loads(file_path.read_text())
        if not isinstance(raw_data, dict):
            raise ValueError("Schema file must contain a JSON/YAML object")
        return cls(raw_data, tenant_id_type=tenant_id_type, audit_columns=audit_columns)

    def _parse_table(self, table: Dict[str, Any]) -> TableDefinition:
        columns: List[ColumnDefinition] = []
        for column in table.get("columns", []):
            columns.append(
                ColumnDefinition(
                    name=column["name"],
                    type=column.get("type", "TEXT"),
                    nullable=column.get("nullable", True),
                    default=column.get("default"),
                    primary_key=column.get("primary_key", False),
                    unique=column.get("unique", False),
                )
            )
        indexes = [dict(index) for index in table.get("indexes", [])]
        tenant_lookup_key = table.get("tenant_lookup_key")
        return TableDefinition(
            name=table["name"],
            columns=columns,
            indexes=indexes,
            tenant_lookup_key=tenant_lookup_key,
            comment=table.get("comment"),
        )

    # ------------------------------------------------------------------
    # Schema generation
    # ------------------------------------------------------------------
    def generate_schema_sql(self) -> str:
        """Generate CREATE TABLE statements for the multi-tenant schema."""

        statements: List[str] = []
        for table in self.tables:
            statements.append(self._render_multi_tenant_table(table))
        return "\n\n".join(statements)

    def _render_multi_tenant_table(self, table: TableDefinition) -> str:
        multi_tenant_table = self._build_multi_tenant_table(table)
        column_lines = [column.to_sql() for column in multi_tenant_table.columns]
        pk_columns = multi_tenant_table.primary_key_columns()
        if "tenant_id" not in pk_columns:
            pk_columns = ["tenant_id", *pk_columns]
        constraint_lines: List[str] = []
        if pk_columns:
            cols = ", ".join(f'"{column}"' for column in pk_columns)
            constraint_lines.append(f"PRIMARY KEY ({cols})")
        create_stmt = [
            f"-- {multi_tenant_table.comment or 'Auto-generated multi-tenant table'}",
            f"CREATE TABLE IF NOT EXISTS \"{multi_tenant_table.name}\" (",
            "    " + ",\n    ".join(column_lines + constraint_lines),
            ");",
        ]
        for index in multi_tenant_table.indexes:
            index_name = index.get("name") or f"{multi_tenant_table.name}_{'_'.join(index.get('columns', []))}_idx"
            column_list = ", ".join(f'\"{col}\"' for col in index.get("columns", []))
            unique = "UNIQUE " if index.get("unique") else ""
            create_stmt.append(
                f"CREATE {unique}INDEX IF NOT EXISTS \"{index_name}\" ON \"{multi_tenant_table.name}\" ({column_list});"
            )
        return "\n".join(create_stmt)

    def _build_multi_tenant_table(self, table: TableDefinition) -> TableDefinition:
        clone = table.clone(name=f"tenant_{table.name}")
        existing_names = {column.name for column in clone.columns}
        tenant_column = ColumnDefinition(
            name="tenant_id",
            type=self.tenant_id_type,
            nullable=False,
        )
        clone.columns.insert(0, tenant_column)
        existing_names.add("tenant_id")
        for audit_column in self.audit_columns:
            if audit_column.name not in existing_names and audit_column.name != "tenant_id":
                clone.columns.append(audit_column)
                existing_names.add(audit_column.name)
        # Ensure indexes contain tenant dimension
        composite_index_columns = ["tenant_id"] + [
            column.name for column in table.columns if column.primary_key
        ]
        clone.indexes.append(
            {
                "name": f"{clone.name}_tenant_scope_idx",
                "columns": composite_index_columns,
                "unique": True,
            }
        )
        clone.comment = (
            table.comment
            or f"Multi-tenant version of {table.name} with tenant scoping enforced."
        )
        return clone

    # ------------------------------------------------------------------
    # Migration planning
    # ------------------------------------------------------------------
    def generate_migration_plan(self, tenant_mappings: Dict[str, Dict[str, Any]]) -> List[str]:
        """Return SQL statements for migrating data for the provided tenants."""

        plan: List[str] = []
        for table in self.tables:
            mt_table = f"tenant_{table.name}"
            lookup_key = table.tenant_lookup_key or "account_id"
            selected_columns = [column.name for column in table.columns]
            for tenant_id, config in tenant_mappings.items():
                raw_accounts = config.get("legacy_accounts")
                if raw_accounts is None:
                    raw_accounts = []
                elif isinstance(raw_accounts, str):
                    raw_accounts = [raw_accounts]
                legacy_account = config.get("legacy_account")
                if legacy_account:
                    raw_accounts.append(legacy_account)
                legacy_accounts = [account for account in raw_accounts if account]
                if not legacy_accounts:
                    legacy_accounts = ["*"]
                for legacy_account in legacy_accounts:
                    where_clause = ""
                    if legacy_account != "*":
                        where_clause = f"WHERE {lookup_key} = '{legacy_account}'"
                    insert_columns = ", ".join([
                        '"tenant_id"',
                        *[f'"{column}"' for column in selected_columns],
                        '"migrated_at"',
                        '"source_legacy_pk"',
                        '"source_hash"',
                    ])
                    conflict_columns = ", ".join([
                        '"tenant_id"',
                        *[f'"{column}"' for column in (selected_columns or ["source_legacy_pk"])],
                    ])
                    source_columns = ", ".join([
                        f"'{tenant_id}'",
                        *selected_columns,
                        "CURRENT_TIMESTAMP",
                        f"CAST({lookup_key} AS TEXT)",
                        self._hash_expression(selected_columns),
                    ])
                    statement = (
                        f"INSERT INTO \"{mt_table}\" ({insert_columns})\n"
                        f"SELECT {source_columns}\n"
                        f"FROM \"{table.name}\"\n"
                        f"{where_clause}\n"
                        f"ON CONFLICT ({conflict_columns}) DO UPDATE SET\n"
                        "    migrated_at = EXCLUDED.migrated_at,\n"
                        "    source_hash = EXCLUDED.source_hash;"
                    )
                    plan.append(statement)
        return plan

    def _hash_expression(self, columns: Iterable[str]) -> str:
        if not columns:
            return "md5('noop')"
        expressions = [f"COALESCE(CAST({column} AS TEXT), '')" for column in columns]
        expr = " || '|' || ".join(expressions)
        return f"md5({expr})"

    def dump_schema(self, output: Union[str, Path]) -> Path:
        """Write generated schema SQL to disk."""

        target = Path(output)
        target.write_text(self.generate_schema_sql())
        return target

    def dump_plan(self, tenant_mappings: Dict[str, Dict[str, Any]], output: Union[str, Path]) -> Path:
        """Write migration plan SQL to disk."""

        target = Path(output)
        target.write_text("\n\n".join(self.generate_migration_plan(tenant_mappings)))
        return target
