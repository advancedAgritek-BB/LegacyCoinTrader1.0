# Trading Engine Service

All REST calls to the trading engine must include an `X-Tenant-ID` header. The
service loads tenant-specific strategy, risk, and execution settings from
`config/tenants.yaml` and runs trading cycles independently for each tenant.

## Tenant-aware scheduling

* Cycle state is stored in Redis under per-tenant keys (`tenant:{id}:...`).
* `POST /cycles/start` accepts an optional `risk_allocation` field and enforces
  the tenant's `max_active_cycles` and risk budget before scheduling.
* `POST /cycles/run` applies the same budget checks for ad-hoc cycles.
* `GET /cycles/status` and `GET /health/tenant` return the current state for
  the tenant identified by the header.

The scheduler emits `tenant_id` in cycle metadata so downstream systems can
attribute work and enforce limits.
