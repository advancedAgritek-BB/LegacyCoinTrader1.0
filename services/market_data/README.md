# Market Data Service

The market data microservice now enforces tenant isolation. All client calls
must include an `X-Tenant-ID` header so the service can load the appropriate
configuration from `config/tenants.yaml`.

## Tenant-aware behaviour

* Redis caches and pub/sub channels are automatically namespaced per tenant.
  Keys follow the pattern `tenant:{id}:{namespace}:...` and channels are
  suffixed with the tenant identifier.
* The following endpoints are tenant-scoped and require the header:
  * `POST /symbols/load`
  * `POST /ohlcv/update`
  * `POST /ohlcv/multi`
  * `POST /regime/update`
  * `POST /order-book/snapshot`
* WebSocket subscriptions to `/ws/ohlcv` also require the tenant identifier
  (header or `tenant_id` query parameter).

## Health and observability

* `GET /health` retains the global status check.
* `GET /health/tenant` reports per-tenant cache counts (`ohlcv`, `regime`,
  `order-book`, and `symbols`).

Include the tenant header on every request when integrating with this service.
