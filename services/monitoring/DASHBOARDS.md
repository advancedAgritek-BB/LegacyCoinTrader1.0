# Grafana Dashboards & Alerting

The following dashboards are designed for Grafana (or any Prometheus-compatible
visualisation layer). Each panel references the metrics emitted by the
`services/monitoring` toolkit and the `/metrics` endpoints exposed by every
microservice and the Flask frontend.

## Prometheus Scrape Targets

Every service now exposes a Prometheus exporter on the same port as its API.
The consolidated scrape configuration lives in
[`prometheus.yaml`](./prometheus.yaml).

| Service           | Endpoint                          | Notes |
|-------------------|-----------------------------------|-------|
| API Gateway       | `http://api-gateway:8000/metrics` | Includes edge auth and rate limiting metrics. |
| Trading Engine    | `http://trading-engine:8001/metrics` | Captures cycle orchestration requests. |
| Market Data       | `http://market-data:8002/metrics` | Observes symbol/ohlcv refresh calls. |
| Portfolio         | `http://portfolio:8003/metrics`   | Tracks portfolio mutation and risk checks. |
| Strategy Engine   | `http://strategy-engine:8004/metrics` | Emits strategy evaluation timings. |
| Token Discovery   | `http://token-discovery:8005/metrics` | Follows discovery scans and scoring. |
| Execution Service | `http://execution:8006/metrics`   | Measures order submission flow. |
| Frontend          | `http://frontend:5000/metrics`     | Flask dashboard instrumentation. |

## 1. Service Golden Signals

**Purpose**: track latency, traffic, errors and saturation for every service.

| Panel | Query | Notes |
|-------|-------|-------|
| Request rate | `sum by (service, method) (increase(legacycoin_http_requests_total[5m]))` | Break down traffic by HTTP method. |
| Error rate | `sum by (service) (increase(legacycoin_http_request_errors_total[5m]))` | Compare with request rate to derive error percentage. |
| P95 latency | `histogram_quantile(0.95, sum by (service, method, route, le)(rate(legacycoin_http_request_duration_seconds_bucket[5m])))` | Uses the automatically exported histogram. |
| Memory usage | `process_resident_memory_bytes{job=~"api-gateway|trading-engine|market-data|portfolio|strategy-engine|token-discovery|execution|frontend"}` | Overlay alert threshold at 1.5 GiB. |
| In-flight requests | `sum by (service) (increase(legacycoin_http_requests_total[1m]))` | Short window for saturation spikes. |

### Alerts

Alerting rules are codified in [`alerts.yaml`](./alerts.yaml) and include:

* **HighMemoryUsage** – fires when `process_resident_memory_bytes` exceeds
  1.5&nbsp;GiB for five minutes.
* **ElevatedErrorRate** – raises when the ratio of
  `legacycoin_http_request_errors_total` to
  `legacycoin_http_requests_total` is greater than 5% over ten minutes.
* **HighRequestLatency** – triggers when the 95th percentile response time is
  above two seconds for five minutes.

## 2. Trading Pipeline Drill-down

**Purpose**: follow a single trading cycle end-to-end.

Panels:

1. **Cycle duration** – Query the structured logs in OpenSearch with
   `service.name:trading-engine AND metadata.duration_seconds:*` and display the
   average duration.
2. **Phase timings** – Use the `metadata` field from the `/cycles/run` endpoint
   response or trace spans (`span.name` ~ `POST /cycles/run`) to visualise phase
   durations.
3. **Trace waterfall** – Use the OpenTelemetry collector to send traces to
   Tempo/Jaeger. Filter by the `legacy.correlation_id` attribute to correlate
   frontend requests with backend cycles.

### Alerts

* **Stalled scheduler** – Alert if no
  `legacycoin_http_requests_total{service="trading-engine", route="/cycles/run"}`
  increase occurs for 15 minutes.
* **Slow cycle** – Alert when the `metadata.duration_seconds` field in logs
  exceeds 60 seconds. This can be expressed as a LogQL/Elasticsearch alert:
  `avg_of_duration > 60` for three consecutive intervals.

## 3. API Gateway & Edge Overview

**Purpose**: confirm authentication and routing health at the ingress layer.

Panels:

1. **Token issuance success rate** –
   `sum by (status) (increase(legacycoin_http_requests_total{service="api-gateway", route="/auth/token"}[5m]))`.
2. **Rate limiter rejections** – filter `legacycoin_http_request_errors_total`
   where `route` matches the proxied prefix (for example `/api/v1/execution`)
   and `status=429`.
3. **Latency by downstream service** – use the `route` label to break out the
   histogram query per proxied prefix.

## 4. Frontend Experience

**Purpose**: ensure the Flask dashboard remains healthy.

| Panel | Query |
|-------|-------|
| Response status codes | `sum by (status) (increase(legacycoin_http_requests_total{service="frontend"}[5m]))` |
| Login errors | `sum by (status) (increase(legacycoin_http_requests_total{service="frontend", route="/login"}[5m]))` |
| Session duration logs | Query OpenSearch for `logger:"frontend.app" AND correlation_id:*` to sample session flows. |
| Trace span count | `sum(rate(spans_total{service_name="frontend"}[5m]))` (Tempo/Jaeger) |

### Alerts

* **Frontend down** – Alert if the `/health` endpoint fails more than three
  times in a five minute window.
* **Authentication failures** – Alert if more than 25 login failures occur in
  ten minutes (monitor `status=401` for `/login`).

## Implementation notes

* All dashboards expect the Prometheus scrape config in
  [`prometheus.yaml`](./prometheus.yaml); the file already lists every service
  endpoint.
* Alert rules are provided in [`alerts.yaml`](./alerts.yaml) and can be loaded
  directly into Alertmanager or Grafana's unified alerting.
* Correlation IDs are attached to logs, traces and responses via the
  `X-Correlation-ID` header, enabling pivoting between Grafana, OpenSearch and
  tracing backends.
