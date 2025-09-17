# Monitoring and Observability Toolkit

The `services/monitoring` package centralises Prometheus metrics, OpenSearch log
shipping and OpenTelemetry tracing for every LegacyCoinTrader microservice and
the Flask frontend. The utilities are framework-agnostic and designed to be
imported by any service that needs consistent observability features.

## Features

* **Prometheus Exporters** – `HttpMetrics` keeps per-service request counters and
  histograms. The middleware automatically exposes a `/metrics` endpoint that
  can be scraped by Prometheus or Grafana Agent.
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

## Usage

```python
from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging import configure_logging

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
| Prometheus namespace           | `MONITORING_METRICS__NAMESPACE`         | `legacycoin`             |
| Prometheus exporter host/port  | `MONITORING_METRICS__EXPORTER_HOST/PORT`| `0.0.0.0` / `9000`       |
| OpenSearch host/port           | `MONITORING_OPENSEARCH__HOST/PORT`      | `localhost` / `9200`     |
| OpenSearch index               | `MONITORING_OPENSEARCH__INDEX`          | `legacycoin-logs`        |
| OTLP collector endpoint        | `MONITORING_TRACING__ENDPOINT`          | `http://localhost:4318`  |

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
