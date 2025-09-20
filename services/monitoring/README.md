# Monitoring and Observability Toolkit

The `services/monitoring` package centralises Prometheus metrics, OpenSearch log
shipping and OpenTelemetry tracing for every LegacyCoinTrader microservice and
the Flask frontend. The utilities are framework-agnostic and designed to be
imported by any service that needs consistent observability features.

## Features

* **Prometheus Exporters** – `HttpMetrics` keeps per-service request counters and
  histograms. The middleware automatically exposes a `/metrics` endpoint that
  can be scraped by Prometheus or Grafana Agent.
* **Tenant & Service Role Context** – Every log line, metric and span is tagged
  with the active tenant and service role. The middleware reads the
  `X-Tenant-ID` and `X-Service-Role` headers (configurable) and falls back to the
  defaults supplied via `MonitoringSettings`.
* **Correlation IDs** – Every inbound request reuses
  `crypto_bot.utils.logger`'s correlation ID helpers. The middleware injects the
  ID into logs, responses and OpenTelemetry spans, allowing cross-service trace
  reconstruction.
* **Centralised Log Shipping** – `configure_logging` registers an
  `OpenSearchLogHandler` that forwards structured JSON logs to an OpenSearch
  cluster. The handler is resilient to connection failures and degrades to
  stdout if the cluster is unreachable.
* **Distributed Tracing** – `configure_tracing` configures an OTLP exporter for
  the service and registers a span processor with an OpenTelemetry collector.
* **Tenant SLOs & Synthetic Checks** – Rolling latency, error rate and
  throughput SLOs are calculated per tenant/service role. Synthetic guard-rail
  endpoints report RTO/RPO compliance, feeding Alertmanager and Grafana panels.
* **Compliance Metrics** – The `SECRETS_ROTATED_AT` timestamp is exported as
  Prometheus gauges (`legacycoin_compliance_secrets_rotation_*`) to satisfy
  audit retention controls.

### Tenant-aware context propagation

`MonitoringSettings` now includes `tenant_header`, `service_role_header`,
`default_tenant` and `service_role`. These values drive both the logging
formatter and the middleware. The helper functions in
`crypto_bot.utils.logger`—`set_observability_context`,
`get_tenant_id`/`get_service_role` and `clear_observability_context`—can be used
outside HTTP handlers (e.g. background jobs) to ensure consistent tagging.

### SLO and synthetic observability endpoints

Both FastAPI and Flask integrations expose a REST surface for on-call teams:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/observability/slo` | GET | Return the rolling SLO snapshots for every tenant/service role. |
| `/observability/slo/{tenant}` | GET | Return the snapshot for a specific tenant (optionally filter via `service_role`). |
| `/observability/synthetic-checks` | GET/POST | List synthetic check statuses or push new results from synthetic monitors. |
| `/observability/synthetic-checks/{name}` | GET | Fetch the latest status for a particular synthetic probe. |

All responses include SLO targets, current measurements and compliance flags so
operators can make rapid decisions during incidents. Synthetic check submissions
accept latency, recovery time (RTO) and data lag (RPO) inputs and propagate them
to Prometheus gauges.

### Compliance telemetry

`HttpMetrics` loads the `SECRETS_ROTATED_AT` environment variable (or
`MonitoringSettings.compliance.fallback_timestamp`) and publishes:

* `legacycoin_compliance_secrets_rotation_timestamp_seconds`
* `legacycoin_compliance_secrets_rotation_age_days`

Labels include the tenant/service role context so stale rotations can be tied to
the appropriate customer partition. These metrics back the
`SecretsRotationStale` alert in [`alerts.yaml`](./alerts.yaml).

### Operational runbooks (SOC2 aligned)

The observability package now documents the operational controls required for
SOC2 readiness:

1. **Initial triage** – When `TenantLatencySLOViolation` or
   `TenantErrorRateSLOViolation` fires, inspect the `/observability/slo`
   endpoint to confirm the breach window. Correlate with Jaeger using
   `legacy.tenant_id` and `legacy.service_role`.
2. **Synthetic guard-rails** – If `SyntheticCheckFailure`, `SyntheticRecoveryTimeBreach`
   or `SyntheticDataLagBreach` triggers, query
   `/observability/synthetic-checks/{name}` for detailed payloads. Compare the
   values against the documented RTO (15&nbsp;minutes) and RPO (5&nbsp;minutes) and
   begin the disaster recovery playbook if either threshold is exceeded.
3. **Secrets rotation** – On `SecretsRotationStale`, rotate credentials,
   update `SECRETS_ROTATED_AT` and attach the evidence to the compliance ticket.
4. **Escalation** – Page the incident commander after 15 minutes of sustained
   SLO violation or any synthetic check indicating a hard failure. Notify the
   compliance officer within one business day for secrets rotation breaches.

The same checklist is referenced by the new Alertmanager annotations so on-call
staff can follow a consistent SOC2-compliant response.

## Usage

```python
from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging_compat import configure_logging

settings = get_monitoring_settings().for_service("orders", environment="prod")
configure_logging(settings)
app = FastAPI()
instrument_fastapi_app(app, settings=settings)
```

For Flask applications use `instrument_flask_app`. The trading engine, portfolio
service and frontend ship with this instrumentation enabled out-of-the-box.

## Configuration

All settings can be tuned via environment variables. Nested configuration uses
`MONITORING_` as the prefix and `__` as the delimiter, for example:

| Purpose                         | Environment variable                   | Default                  |
|--------------------------------|-----------------------------------------|--------------------------|
| Service identifier             | `MONITORING_SERVICE_NAME`               | `legacycoin-service`     |
| Deployment environment         | `MONITORING_ENVIRONMENT`                | `development`            |
| Correlation header name        | `MONITORING_CORRELATION_HEADER`         | `X-Correlation-ID`       |
| Tenant header name             | `MONITORING_TENANT_HEADER`              | `X-Tenant-ID`            |
| Service role header name       | `MONITORING_SERVICE_ROLE_HEADER`        | `X-Service-Role`         |
| Default tenant                 | `MONITORING_DEFAULT_TENANT`             | `global`                 |
| Default service role           | `MONITORING_SERVICE_ROLE`               | `unspecified`            |
| Prometheus namespace           | `MONITORING_METRICS__NAMESPACE`         | `legacycoin`             |
| Prometheus exporter host/port  | `MONITORING_METRICS__EXPORTER_HOST/PORT`| `0.0.0.0` / `9000`       |
| OpenSearch host/port           | `MONITORING_OPENSEARCH__HOST/PORT`      | `localhost` / `9200`     |
| OpenSearch index               | `MONITORING_OPENSEARCH__INDEX`          | `legacycoin-logs`        |
| OTLP collector endpoint        | `MONITORING_TRACING__ENDPOINT`          | `http://localhost:4318`  |
| Tenant SLO window              | `MONITORING_SLO__WINDOW_SECONDS`        | `300`                    |
| Latency SLO target             | `MONITORING_SLO__LATENCY_TARGET_MS`     | `500`                    |
| Error rate SLO target          | `MONITORING_SLO__ERROR_RATE_TARGET`     | `0.01`                   |
| Throughput SLO target          | `MONITORING_SLO__THROUGHPUT_TARGET_RPS` | `1.0`                    |
| RTO target (seconds)           | `MONITORING_SLO__RTO_TARGET_SECONDS`    | `900`                    |
| RPO target (seconds)           | `MONITORING_SLO__RPO_TARGET_SECONDS`    | `300`                    |
| Secrets rotation fallback      | `MONITORING_COMPLIANCE__FALLBACK_TIMESTAMP` | _unset_             |
| Maximum secret age (days)      | `MONITORING_COMPLIANCE__MAX_SECRET_AGE_DAYS` | `90`                |

Refer to [`DASHBOARDS.md`](./DASHBOARDS.md) for ready-to-import Grafana
layouts and alert recommendations.

## Collectors

The repository ships with an example [OpenTelemetry collector
configuration](./otel-collector.yaml). Start it locally with:

```bash
otelcol --config services/monitoring/otel-collector.yaml
```

Prometheus can scrape every service at `http://<service-host>:<service-port>/metrics`.
A sample scrape configuration is available in [`prometheus.yaml`](./prometheus.yaml).

## CI/CD

Observability checks are executed as part of the standard pytest suite via
`tests/test_observability.py` (see below). The `Makefile` exposes an
`observability-check` target that runs these tests locally or in CI.
