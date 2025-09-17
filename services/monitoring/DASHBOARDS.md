# Grafana Dashboards & Alerting

The following dashboards are designed for Grafana (or any Prometheus compatible
visualisation layer). Each panel references the metrics emitted by the
`services/monitoring` toolkit.

## 1. Service Golden Signals

**Purpose**: track latency, traffic, errors and saturation for every service.

| Panel | Query | Notes |
|-------|-------|-------|
| Request rate | `sum by (service, method) (increase(legacycoin_http_requests_total[5m]))` | Break down traffic by HTTP method. |
| Error rate | `sum by (service) (increase(legacycoin_http_request_errors_total[5m]))` | Compare with request rate to derive error percentage. |
| P95 latency | `histogram_quantile(0.95, sum by (service, method, route, le)(rate(legacycoin_http_request_duration_seconds_bucket[5m])))` | Uses the automatically exported histogram. |
| In-flight requests | `sum by (service) (increase(legacycoin_http_requests_total[1m]))` | Short window for saturation spikes. |

### Alerts

* **High error rate** – Trigger when
  `sum(increase(legacycoin_http_request_errors_total{service="trading-engine"}[5m])) /
   sum(increase(legacycoin_http_requests_total{service="trading-engine"}[5m])) > 0.05`
  for 10 minutes.
* **Elevated latency** – Trigger when the P95 latency exceeds 2s for three
  consecutive evaluation periods.

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

* **Stalled scheduler** – Alert if no `legacycoin_http_requests_total{route="/cycles/run"}` increase occurs for 15 minutes.
* **Slow cycle** – Alert when the `metadata.duration_seconds` field in logs
  exceeds 60 seconds. This can be expressed as a LogQL/Elasticsearch alert:
  `avg_of_duration > 60` for 3 consecutive intervals.

## 3. Frontend Experience

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
  [`prometheus.yaml`](./prometheus.yaml).
* Alert rules can be exported via Grafana as JSON and stored alongside this
  file; use the metric queries above as templates.
* Correlation IDs are attached to logs, traces and responses via the
  `X-Correlation-ID` header, enabling pivoting between Grafana, OpenSearch and
  tracing backends.
