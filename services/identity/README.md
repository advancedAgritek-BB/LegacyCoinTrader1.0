# Identity Service

The identity service provides authentication and authorization for LegacyCoinTrader. It is
implemented as a standalone FastAPI microservice with optional gRPC bindings and manages
multi-tenant users, roles, and credentials. The service issues OAuth2/OIDC compatible
JWTs, stores refresh tokens, exposes SCIM provisioning endpoints, and integrates with
external secret managers (Hashicorp Vault or AWS Secrets Manager) for tenant specific
signing keys and API secrets.

## Features

* OAuth2/OIDC token issuance (`password` and `refresh_token` grants), refresh, and
  standards compliant token introspection.
* Multi-tenant aware SQLAlchemy models for tenants, users, roles, credentials, and
  refresh tokens. Password and API-key rotation metadata is tracked on every
  credential record.
* MFA hooks with pluggable providers (TOTP included by default).
* SCIM `Users` API for provisioning and de-provisioning identities.
* REST and gRPC surfaces for token and credential operations.
* Service-to-service authentication helpers so other microservices can validate
  identity issued tokens via JWKS or remote introspection.

## REST API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/oauth/token` | `POST` | Issue an access/refresh token pair (`password` or `refresh_token` grant). |
| `/oauth/introspect` | `POST` | RFC 7662 compatible token introspection. Requires service token. |
| `/credentials/password/rotate` | `POST` | Rotate a user password after validating the current secret. |
| `/credentials/api-key/rotate` | `POST` | Rotate an API key (optionally supply a pre-generated key). |
| `/credentials/api-key/validate` | `POST` | Validate an API key and return the owning identity metadata. |
| `/scim/v2/Users` | `GET/POST` | SCIM user listing and provisioning (requires service token). |
| `/scim/v2/Users/{id}` | `GET/PUT/DELETE` | Retrieve, replace, or delete a SCIM user. |
| `/.well-known/openid-configuration` | `GET` | Standard discovery metadata. |
| `/.well-known/jwks.json` | `GET` | JWKS containing tenant public keys for JWT verification. |

### gRPC Service

The gRPC endpoint (service name `identity.v1.Identity`) exposes the following unary
operations. All requests and responses use `google.protobuf.Struct` payloads to avoid
requiring generated client code.

* `IssueToken`
* `RefreshToken`
* `IntrospectToken`
* `RotatePassword`
* `RotateApiKey`
* `ValidateApiKey`

See `services/identity/api/grpc.py` for usage.

## Database & Migrations

Alembic migrations live in `services/identity/migrations`. Apply them with:

```bash
alembic -c services/identity/alembic.ini upgrade head
```

(or execute programmatically using `services.identity.database` helpers).
The initial migration (`0001_identity`) creates tables for tenants, roles, users,
role assignments, credentials, and refresh tokens.

## Configuration

Configuration is driven by the following environment variables (defaults shown):

| Variable | Description |
| --- | --- |
| `IDENTITY_DATABASE_URL` | SQLAlchemy database URL (`sqlite:///./identity.db`). |
| `IDENTITY_ACCESS_TOKEN_TTL_SECONDS` | Access token lifetime (seconds). |
| `IDENTITY_REFRESH_TOKEN_TTL_SECONDS` | Refresh token lifetime (seconds). |
| `IDENTITY_TOKEN_ALGORITHM` | JWT signing algorithm (default `RS256`). |
| `IDENTITY_DEFAULT_ISSUER` | Base issuer URI used for new tenants. |
| `IDENTITY_DEFAULT_TENANT` | Tenant slug to use when no header is provided (`primary`). |
| `IDENTITY_SERVICE_TOKEN_HEADER` | Header name for service tokens (`x-service-token`). |
| `IDENTITY_INTERNAL_SERVICE_TOKEN` | Optional shared secret required for privileged endpoints. |
| `IDENTITY_TENANT_HEADER_CANDIDATES` | Comma separated tenant headers to inspect (`X-Tenant-ID,X-Tenant,X-Realm`). |
| `IDENTITY_ALLOW_DEVELOPMENT_FALLBACK_KEYS` | Generate ephemeral signing keys when secrets are missing (defaults to `true`). |
| `IDENTITY_JWKS_CACHE_SECONDS` | Cache lifetime for JWKS responses (default `300`). |

### Secret Storage

Each tenant stores a `secret_reference` describing where its signing material lives. The
identity service resolves the key using `services.common.secrets.SecretManager`, which
supports environment variables, Hashicorp Vault, and AWS Secrets Manager.

* **Vault** – set `SECRETS_PROVIDER=vault` and configure `VAULT_ADDR`, `VAULT_TOKEN`, and
  `VAULT_SECRET_PATH`. Within Vault, place a JSON document at the tenant specific path,
  e.g. `tenants/<tenant>/identity` with:

  ```json
  {
    "algorithm": "RS256",
    "private_key": "-----BEGIN PRIVATE KEY-----...",
    "public_key": "-----BEGIN PUBLIC KEY-----..."
  }
  ```

* **AWS Secrets Manager** – set `SECRETS_PROVIDER=aws` with `AWS_SECRET_NAME` and
  `AWS_REGION`. The secret payload should contain the same JSON structure.

If a symmetric algorithm (e.g. `HS256`) is required, provide a `shared_secret` value in
place of the RSA keys. JWKS responses omit symmetric keys by design.

## Service-to-Service Token Validation

Consumers can validate tokens without hard-coding shared secrets by using the helper
classes in `services.identity.auth`.

```python
from services.identity.auth import IdentityTokenValidator

validator = IdentityTokenValidator(
    "http://identity:8006/.well-known/jwks.json",
    issuer="https://identity.legacycointrader.local/primary",
)
claims = validator.validate(access_token)
print(claims.subject, claims.roles)
```

For network-based validation, use the introspection client:

```python
from services.identity.auth import IdentityIntrospectionClient

with IdentityIntrospectionClient(
    "http://identity:8006",
    service_token="<gateway service token>",
    default_tenant="primary",
) as client:
    introspection = client.introspect(access_token)
    assert introspection["active"]
```

## API Gateway Integration

The API gateway loads identity settings via new environment variables:

| Variable | Description |
| --- | --- |
| `IDENTITY_SERVICE_URL` | Base URL to the identity service (default `http://identity:8006`). |
| `IDENTITY_JWKS_URL` | JWKS endpoint (defaults to `<IDENTITY_SERVICE_URL>/.well-known/jwks.json`). |
| `IDENTITY_DEFAULT_TENANT` | Tenant slug forwarded on gateway issued requests. |
| `IDENTITY_SERVICE_TOKEN` | Service token presented to the identity service. |
| `IDENTITY_TENANT_HEADER` | Header used to convey the tenant (default `X-Tenant-ID`). |
| `IDENTITY_REQUEST_TIMEOUT` | HTTP timeout for identity requests (default `10` seconds). |
| `IDENTITY_EXPECTED_ISSUER` / `IDENTITY_EXPECTED_AUDIENCE` | Optional OIDC issuer/audience checks for JWT validation. |

The gateway now delegates all authentication and credential management to the identity
service and validates JWTs using JWKS metadata from the `/well-known/jwks.json` endpoint.
