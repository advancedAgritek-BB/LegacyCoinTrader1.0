# API Gateway Security Operations

This guide documents the operational tasks introduced alongside the identity and
RBAC integration.

## Configuration overview

* `GATEWAY_TOKEN_TTL_SECONDS` – lifetime of issued JWTs (default `3600`).
* `GATEWAY_TOKEN_ISSUER` – issuer claim embedded in every token.
* `PORTFOLIO_PASSWORD_ROTATION_DAYS` / `PORTFOLIO_API_KEY_ROTATION_DAYS` –
  enforced credential rotation windows managed by the portfolio identity store.
* `GATEWAY_SERVICE_TOKEN_<NAME>` – service-to-service tokens remain supported and
  are granted the `internal` scope, bypassing role checks.

## User access

All user identities live in the portfolio database (`user_accounts` table). The
`PortfolioIdentityService` module exposes operations for provisioning and
maintenance:

* `create_user(username, password, roles=[...])` – initial bootstrap of hashed
  credentials and role assignments.
* `rotate_password(username, current_password, new_password)` – enforces password
  rotation while preserving audit metadata.
* `rotate_api_key(username)` – rotates a hashed API key and returns the new
  secret for distribution.
* `list_users()` – returns role and login metadata for audit exports.

When a user authenticates the gateway issues a JWT containing the assigned
roles. Each proxied service declares the role(s) required for access:

| Service           | Required role |
| ----------------- | ------------- |
| trading_engine    | `trading`     |
| market_data       | `market`      |
| portfolio         | `portfolio`   |
| strategy_engine   | `strategy`    |
| token_discovery   | `token`       |
| execution         | `execution`   |
| monitoring        | `monitoring`  |

The `admin` role implicitly grants access to every route. Service tokens are
also accepted for intra-cluster communication.

## API endpoints

The gateway now exposes three first-class identity endpoints:

| Endpoint                 | Method | Description                                  |
| ------------------------ | ------ | -------------------------------------------- |
| `/auth/token`            | POST   | Issue a JWT for username/password credentials |
| `/auth/password/rotate`  | POST   | Rotate a password after verifying the old one |
| `/auth/api-key/validate` | POST   | Validate hashed API keys                      |

All endpoints respond with `401` for invalid credentials, `403` when an account
is disabled or a rotation is due, and `503` if the identity backend is
unavailable.

Audit logs for authentication failures and token issuance are emitted under the
`api_gateway` logger.
