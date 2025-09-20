import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const healthCheckDuration = new Trend('health_check_duration');
const serviceCallDuration = new Trend('service_call_duration');

// Test configuration
export const options = {
  stages: [
    // Ramp up to 50 users over 1 minute
    { duration: '1m', target: 50 },
    // Stay at 50 users for 3 minutes
    { duration: '3m', target: 50 },
    // Ramp up to 100 users over 2 minutes
    { duration: '2m', target: 100 },
    // Stay at 100 users for 2 minutes
    { duration: '2m', target: 100 },
    // Ramp down to 0 users over 1 minute
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    // 95% of requests should be below 500ms
    http_req_duration: ['p(95)<500'],
    // Error rate should be below 1%
    errors: ['rate<0.01'],
    // 99% of requests should succeed
    http_req_failed: ['rate<0.01'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test scenarios
export default function () {
  // Scenario 1: Health check
  const healthStart = new Date().getTime();
  const healthResponse = http.get(`${BASE_URL}/health`);
  const healthDuration = new Date().getTime() - healthStart;

  healthCheckDuration.add(healthDuration);

  check(healthResponse, {
    'health status is 200': (r) => r.status === 200,
    'health response contains status': (r) => r.json().hasOwnProperty('status'),
    'health response contains services': (r) => r.json().hasOwnProperty('services'),
  });

  errorRate.add(healthResponse.status !== 200);

  // Scenario 2: Authentication (if configured)
  const authResponse = http.post(`${BASE_URL}/auth/token`, JSON.stringify({
    username: 'test_user',
    password: 'test_password'
  }), {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  check(authResponse, {
    'auth response is valid': (r) => r.status === 200 || r.status === 401 || r.status === 422,
  });

  errorRate.add(authResponse.status === 500);

  // Scenario 3: Service token operations
  const tokenResponse = http.post(`${BASE_URL}/auth/service-token/generate?service_name=market-data`);

  check(tokenResponse, {
    'token generation response is valid': (r) => r.status === 200 || r.status === 401 || r.status === 403,
  });

  errorRate.add(tokenResponse.status === 500);

  // Scenario 4: Service communication (if services are running)
  const serviceStart = new Date().getTime();
  const serviceResponse = http.get(`${BASE_URL}/market-data/symbols`);
  const serviceDuration = new Date().getTime() - serviceStart;

  serviceCallDuration.add(serviceDuration);

  check(serviceResponse, {
    'service response is valid': (r) => r.status === 200 || r.status === 404 || r.status === 503,
  });

  errorRate.add(serviceResponse.status === 500);

  // Scenario 5: Concurrent operations simulation
  if (__ITER % 10 === 0) { // Every 10th iteration
    const concurrentRequests = [
      `${BASE_URL}/health`,
      `${BASE_URL}/auth/service-tokens`,
      `${BASE_URL}/portfolio/state`,
      `${BASE_URL}/market-data/status`,
    ];

    const concurrentResponses = http.batch(concurrentRequests.map(url => ['GET', url]));

    concurrentResponses.forEach(response => {
      check(response, {
        'concurrent request is valid': (r) => r.status < 500,
      });
      errorRate.add(response.status >= 500);
    });
  }

  // Random sleep to simulate real user behavior
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

// Setup function - runs before the test starts
export function setup() {
  console.log('🚀 Starting API Gateway load test');
  console.log(`📍 Target URL: ${BASE_URL}`);

  // Warm-up request
  const warmupResponse = http.get(`${BASE_URL}/health`);
  if (warmupResponse.status !== 200) {
    console.error(`❌ Warm-up failed: ${warmupResponse.status}`);
  } else {
    console.log('✅ Warm-up successful');
  }

  return { timestamp: new Date().toISOString() };
}

// Teardown function - runs after the test completes
export function teardown(data) {
  console.log('🏁 Load test completed');
  console.log(`🕐 Started at: ${data.timestamp}`);
  console.log(`🕐 Completed at: ${new Date().toISOString()}`);
}

// Handle summary - custom summary export
export function handleSummary(data) {
  const summary = {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'api-gateway-load-test.json': JSON.stringify(data, null, 2),
    'summary.html': htmlReport(data),
  };

  return summary;
}

function textSummary(data, options) {
  return `
📊 API Gateway Load Test Summary
==================================

Test Duration: ${data.metrics.iteration_duration.values.avg}ms avg iteration
Total Requests: ${data.metrics.http_reqs.values.count}
Failed Requests: ${data.metrics.http_req_failed.values.rate * 100}%

🚀 Performance Metrics:
• 95th percentile: ${Math.round(data.metrics.http_req_duration.values['p(95)'])}ms
• Average response time: ${Math.round(data.metrics.http_req_duration.values.avg)}ms
• Max response time: ${Math.round(data.metrics.http_req_duration.values.max)}ms

🏥 Health Check Performance:
• Average: ${Math.round(data.metrics.health_check_duration.values.avg)}ms
• 95th percentile: ${Math.round(data.metrics.health_check_duration.values['p(95)'])}ms

🔧 Service Call Performance:
• Average: ${Math.round(data.metrics.service_call_duration.values.avg)}ms
• 95th percentile: ${Math.round(data.metrics.service_call_duration.values['p(95)'])}ms

❌ Error Rate: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%

📈 Throughput: ${Math.round(data.metrics.http_reqs.values.rate)} requests/second
`;
}

function htmlReport(data) {
  return `
<!DOCTYPE html>
<html>
<head>
    <title>API Gateway Load Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>🚀 API Gateway Load Test Report</h1>
    <p><strong>Generated:</strong> ${new Date().toISOString()}</p>

    <h2>📊 Summary</h2>
    <div class="metric">
        <strong>Total Requests:</strong> ${data.metrics.http_reqs.values.count}<br>
        <strong>Failed Requests:</strong> ${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%<br>
        <strong>Average Response Time:</strong> ${Math.round(data.metrics.http_req_duration.values.avg)}ms<br>
        <strong>95th Percentile:</strong> ${Math.round(data.metrics.http_req_duration.values['p(95)'])}ms<br>
        <strong>Error Rate:</strong> ${(data.metrics.errors.values.rate * 100).toFixed(2)}%<br>
        <strong>Throughput:</strong> ${Math.round(data.metrics.http_reqs.values.rate)} req/sec
    </div>

    <h2>🔧 Detailed Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
        <tr>
            <td>95th Percentile Response Time</td>
            <td>${Math.round(data.metrics.http_req_duration.values['p(95)'])}ms</td>
            <td>&lt; 500ms</td>
            <td class="${data.metrics.http_req_duration.values['p(95)'] < 500 ? 'success' : 'error'}">
                ${data.metrics.http_req_duration.values['p(95)'] < 500 ? '✅ PASS' : '❌ FAIL'}
            </td>
        </tr>
        <tr>
            <td>Error Rate</td>
            <td>${(data.metrics.errors.values.rate * 100).toFixed(2)}%</td>
            <td>&lt; 1%</td>
            <td class="${data.metrics.errors.values.rate < 0.01 ? 'success' : 'error'}">
                ${data.metrics.errors.values.rate < 0.01 ? '✅ PASS' : '❌ FAIL'}
            </td>
        </tr>
        <tr>
            <td>Request Failure Rate</td>
            <td>${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%</td>
            <td>&lt; 1%</td>
            <td class="${data.metrics.http_req_failed.values.rate < 0.01 ? 'success' : 'error'}">
                ${data.metrics.http_req_failed.values.rate < 0.01 ? '✅ PASS' : '❌ FAIL'}
            </td>
        </tr>
    </table>

    <h2>📈 Response Time Distribution</h2>
    <ul>
        <li><strong>Min:</strong> ${Math.round(data.metrics.http_req_duration.values.min)}ms</li>
        <li><strong>Avg:</strong> ${Math.round(data.metrics.http_req_duration.values.avg)}ms</li>
        <li><strong>Median:</strong> ${Math.round(data.metrics.http_req_duration.values['p(50)'])}ms</li>
        <li><strong>95th percentile:</strong> ${Math.round(data.metrics.http_req_duration.values['p(95)'])}ms</li>
        <li><strong>99th percentile:</strong> ${Math.round(data.metrics.http_req_duration.values['p(99)'])}ms</li>
        <li><strong>Max:</strong> ${Math.round(data.metrics.http_req_duration.values.max)}ms</li>
    </ul>
</body>
</html>
`;
}
