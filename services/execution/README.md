# Execution Service

Every request to the execution API must include an `X-Tenant-ID` header. The
service derives exchange credentials, network segments, and failover routes
from `config/tenants.yaml` and keeps HTTP/WebSocket sessions isolated per
tenant.

## Tenant-specific behaviour

* `POST /exchanges` and `POST /orders` merge tenant execution overrides and
  load dedicated API credentials before opening sessions.
* Order lifecycle events (`/orders/{id}/events`) and the internal caches are
  keyed by `tenant_id:client_order_id` to prevent cross-tenant leakage.
* `GET /health/tenant` reports the number of cached exchange sessions and open
  orders for the tenant making the request.

Requests that omit the tenant header are rejected with `400`. Include the
header on every call when interacting with the execution service.
