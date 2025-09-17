# Frontend Security Operations Guide

This document captures the day-to-day procedures required to operate the
LegacyCoinTrader frontend with the hardened identity workflow introduced in this
release.

## User provisioning

1. Use the portfolio identity store to create users with hashed credentials.
   The helper below provisions a new administrator that may access every
   service routed through the gateway:

   ```bash
   python - <<'PY'
   from services.portfolio.identity import PortfolioIdentityService

   identity = PortfolioIdentityService()
   user = identity.create_user(
       "security-admin",
       "change_me_now!",
       roles=["admin", "portfolio", "trading"],
   )
   print(f"Created user {user.username} with roles: {user.roles}")
   PY
   ```

2. The `roles` list determines which downstream services the user may access.
   The API gateway enforces the following role-to-service mapping:

   | Role        | Downstream service |
   | ----------- | ------------------ |
   | `portfolio` | Portfolio REST API |
   | `trading`   | Trading engine     |
   | `market`    | Market data        |
   | `strategy`  | Strategy engine    |
   | `token`     | Token discovery    |
   | `execution` | Order execution    |
   | `monitoring`| Monitoring stack   |
   | `admin`     | Full access        |

   Assign only the minimal roles required for the user’s duties.

3. When an API key is required (for headless integrations) rotate it through the
   same identity service. Populate `PORTFOLIO_NEW_API_KEY` (or the legacy alias
   `PORTFOLIO_ROTATION_API_KEY`) via the configured secrets manager or export it
   for a one-off rotation. The service persists the hashed representation in the
   portfolio database and returns the clear-text value so it can be stored back
   in the secure vault:

   ```bash
   export PORTFOLIO_NEW_API_KEY=$(openssl rand -hex 32)

   python - <<'PY'
   from services.portfolio.identity import PortfolioIdentityService

   service = PortfolioIdentityService()
   result = service.rotate_api_key("security-admin")
   print(
       "API key for",
       result.user.username,
       "rotated; updated secret is available via the managed store."
   )
   PY
   ```

## Secrets management

* Configure the shared secrets manager by setting `SECRETS_PROVIDER` to
  `vault`/`hashicorp` or `aws`. The frontend and trading services read
  additional connection details from:
  * `VAULT_ADDR`, `VAULT_TOKEN`, `VAULT_SECRET_PATH`, `VAULT_VERIFY`,
    `VAULT_TIMEOUT`
  * `AWS_SECRET_NAME`, `AWS_REGION`, `AWS_PROFILE`
* Provision `SESSION_SECRET_KEY` (and any aliases declared in
  `config/managed_secrets.yaml`) in the secure store. The frontend no longer
  generates a fallback secret at runtime—initialisation fails unless a value is
  supplied.
* Rotate user API keys by loading a fresh value from the secure store (as shown
  above) or by supplying the `new_api_key` argument directly when calling
  `rotate_api_key`.

## Auditing

* All user metadata lives in the `user_accounts` table inside the portfolio
  database. Retrieve an auditable snapshot by running `PortfolioIdentityService
  .list_users()`.
* Each login updates the `last_login_at` column which can be exported to your SIEM.
* API key usage is tracked in-memory by the gateway. Combine the identity store
  records with HTTP access logs emitted by `services/api_gateway` for full audit
  trails.

## Session management

* Flask sessions expire after `SECURITY__SESSION_TIMEOUT` seconds. Update the
  value in `frontend/config.py` (or the corresponding environment variable) to
  meet your organisation’s idle timeout policy.
* The API gateway issues JWTs that expire after `GATEWAY_TOKEN_TTL_SECONDS`
  seconds. When the token expires the frontend automatically clears the session
  and forces a re-authentication.
* Passwords must be rotated at least every `PORTFOLIO_PASSWORD_ROTATION_DAYS`
  days (configured in `services/portfolio/config.py`). Attempts to log in with an
  expired password will be rejected until a rotation is performed.

Follow these steps whenever onboarding or offboarding personnel to keep the
frontend aligned with service-level RBAC and credential hygiene requirements.
