# API Gateway Security Operations

The API gateway consumes the shared identity provider implemented in
`services/portfolio/security.py`. All access tokens and API key validations originate from
those hashed credentials.

## Operational checklist

1. Configure secrets and metadata before launching the gateway:
   * `API_GATEWAY_JWT_SECRET` – signing key for issued JWTs.
   * `PORTFOLIO_API_KEY_SECRET` – HMAC key for hashing stored API keys.
   * `PORTFOLIO_PASSWORD_ROTATION_DAYS` / `PASSWORD_ROTATION_DAYS` – rotation window that
     invalidates passwords and API keys when exceeded.
   * `ROLE_DEFINITIONS` – JSON mapping of roles to service scopes; must match the frontend
     configuration to keep RBAC consistent.
2. Issue user accounts and API keys with `python -m services.portfolio.manage_users` before
   exposing routes. The gateway refuses authentication attempts when the password is expired
   or the account is disabled.

## Token-based clients

Use the OAuth 2.0 password grant endpoint to obtain a bearer token:

```bash
http POST :8000/auth/token username=alice password=secret scope=""
```

Successful responses include the role and permission scopes encoded in the token. Inject the
returned bearer token in the `Authorization: Bearer <token>` header for downstream requests.
The `/auth/verify` endpoint validates tokens and returns the resolved principal, enabling
reverse proxies or monitoring agents to perform health checks without re-issuing credentials.

## API key clients

Service-to-service traffic should rely on hashed API keys:

```bash
http POST :8000/auth/api-key api_key=<plain-api-key>
```

The gateway verifies the key against the portfolio database, enforces role membership, and
returns the principal metadata including `password_expires_at`. If the associated password
rotation window has elapsed the gateway responds with `403 password_expired`, signalling that
operators must run:

```bash
python -m services.portfolio.manage_users rotate-password <user>
```

to refresh the credentials.

## Auditing

* All JWTs embed `iss`, `aud`, `iat`, `exp`, and permission scopes to simplify downstream
  auditing.
* API key checks update the `last_login_at` column in the credential store, enabling
  operators to track service usage via `manage_users list-users`.
* Configure CORS with `API_GATEWAY_CORS_ORIGINS` to restrict web origins that can issue
  authentication requests.

Set `API_GATEWAY_ACCESS_TOKEN_MINUTES` to control token lifetime. Combine short-lived tokens
with the password rotation policy to ensure stale secrets cannot be used indefinitely.
