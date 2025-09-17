"""Initial identity service schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_identity"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "identity_tenants",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("slug", sa.String(length=64), nullable=False, unique=True, index=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("issuer", sa.String(length=256), nullable=False),
        sa.Column("key_id", sa.String(length=128), nullable=False),
        sa.Column("secret_provider", sa.String(length=32), nullable=False, server_default="vault"),
        sa.Column("secret_reference", sa.String(length=256), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
    )

    op.create_table(
        "identity_roles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("identity_tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "name", name="uq_identity_role_name"),
    )

    op.create_table(
        "identity_users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.Integer(), sa.ForeignKey("identity_tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=True),
        sa.Column("username", sa.String(length=128), nullable=False),
        sa.Column("email", sa.String(length=256), nullable=True),
        sa.Column("display_name", sa.String(length=256), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("is_service_account", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("mfa_enforced", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("mfa_secret", sa.String(length=256), nullable=True),
        sa.Column("mfa_recovery_codes", sa.JSON(), nullable=False),
        sa.Column("password_rotated_at", sa.DateTime(), nullable=True),
        sa.Column("password_expires_at", sa.DateTime(), nullable=True),
        sa.Column("api_key_last_rotated_at", sa.DateTime(), nullable=True),
        sa.Column("api_key_expires_at", sa.DateTime(), nullable=True),
        sa.Column("attributes", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("tenant_id", "username", name="uq_identity_user_username"),
        sa.UniqueConstraint("tenant_id", "email", name="uq_identity_user_email"),
    )

    op.create_table(
        "identity_user_roles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("identity_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role_id", sa.Integer(), sa.ForeignKey("identity_roles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("assigned_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "role_id", name="uq_identity_user_role"),
    )

    credential_type = sa.Enum("password", "api_key", "service", "refresh_token", name="credentialtype")
    credential_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "identity_credentials",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("identity_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("credential_type", credential_type, nullable=False, index=True),
        sa.Column("secret_hash", sa.String(length=512), nullable=False),
        sa.Column("secret_salt", sa.String(length=256), nullable=False),
        sa.Column("secret_iterations", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("rotation_interval_days", sa.Integer(), nullable=False),
        sa.Column("rotated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("external_secret_reference", sa.String(length=256), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "tenant_id",
            "user_id",
            "credential_type",
            "version",
            name="uq_identity_credential_version",
        ),
    )
    op.create_index(
        "ix_identity_credential_active",
        "identity_credentials",
        ["user_id", "credential_type", "is_active"],
    )

    op.create_table(
        "identity_refresh_tokens",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("identity_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_id", sa.String(length=128), nullable=False),
        sa.Column("token_hash", sa.String(length=512), nullable=False),
        sa.Column("token_salt", sa.String(length=256), nullable=False),
        sa.Column("token_iterations", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.String(length=128), nullable=True),
        sa.Column("issued_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.UniqueConstraint("tenant_id", "token_id", name="uq_identity_refresh_token"),
    )
    op.create_index(
        "ix_identity_refresh_user",
        "identity_refresh_tokens",
        ["user_id", "revoked_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_identity_refresh_user", table_name="identity_refresh_tokens")
    op.drop_table("identity_refresh_tokens")
    op.drop_index("ix_identity_credential_active", table_name="identity_credentials")
    op.drop_table("identity_credentials")
    op.drop_table("identity_user_roles")
    op.drop_table("identity_users")
    op.drop_table("identity_roles")
    op.drop_table("identity_tenants")
    op.execute("DROP TYPE IF EXISTS credentialtype")
