# Frontend Security Operations

The frontend now authenticates exclusively against the credential records stored in the
portfolio service database. Use the commands below to provision users, rotate secrets, and
align the UI session configuration with organizational policy.

## User provisioning

1. Set the database connection parameters (for example `PORTFOLIO_DATABASE_URL`) and
   rotation controls (`PORTFOLIO_PASSWORD_ROTATION_DAYS`, `ROLE_DEFINITIONS`) in the
   environment before starting the frontend or the CLI.
2. Create users with the portfolio management CLI:

   ```bash
   python -m services.portfolio.manage_users create-user alice --role trader
   ```

   The command prompts for a password and optionally accepts `--api-key` and
   `--inactive` to stage onboarding without immediately enabling access.

3. Grant or revoke API keys as needed:

   ```bash
   python -m services.portfolio.manage_users set-api-key alice --api-key <generated-key>
   ```

   API keys are hashed with the configured `PORTFOLIO_API_KEY_SECRET` and are valid only
   while the associated password rotation policy is satisfied.

4. Use `rotate-password` whenever credentials change or on scheduled rotation events:

   ```bash
   python -m services.portfolio.manage_users rotate-password alice --password <new-secret>
   ```

## Auditing

* `python -m services.portfolio.manage_users list-users` prints all accounts together with
  their last login timestamps, password rotation metadata, and API key status. Export this
  JSON output to the logging pipeline for centralized monitoring if required.
* Authentication failures increment `failed_login_attempts` and update
  `last_failed_login_at`, providing a simple trail for intrusion detection.
* The frontend exposes `/api/auth/status` which returns the active session, associated
  permissions, and credential expiry information for UI diagnostics.

## Session management

* Configure the UI session lifetime using `SESSION_TIMEOUT` (seconds). Sessions expire
  server-side after the timeout and the client receives a `401` JSON response or is
  redirected to `/login`.
* Align the timeout with password rotation by setting `PASSWORD_ROTATION_DAYS`
  (or `PORTFOLIO_PASSWORD_ROTATION_DAYS`) so that users are forced to rotate credentials
  before the stored expiry (`password_expires_at`).
* Adjust role-to-permission mappings with the shared `ROLE_DEFINITIONS` JSON environment
  variable. Roles must be consistent across the frontend and API gateway to maintain RBAC.

For additional automation, integrate these CLI workflows into your CI/CD pipeline so new
service accounts are provisioned with hashed credentials during deployments.
