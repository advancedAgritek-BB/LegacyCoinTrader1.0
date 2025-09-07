# End-to-End Testing Framework

This directory contains the comprehensive end-to-end testing framework for the LegacyCoinTrader microservices architecture.

## Overview

The E2E testing framework provides automated testing of the entire microservices stack, ensuring that all services work together correctly and maintain system integrity.

## Test Categories

### 1. Health Checks
- **Service Availability**: Verify all services are running and accessible
- **Health Endpoints**: Test `/health` endpoints for each service
- **Dependency Checks**: Ensure service dependencies are healthy

### 2. Integration Tests
- **API Gateway Routing**: Test request routing through the API Gateway
- **Service Discovery**: Verify service registration and discovery
- **Inter-Service Communication**: Test communication between services

### 3. Business Logic Tests
- **Trading Pipeline**: Test the complete trading data flow
- **Strategy Evaluation**: Verify strategy processing
- **Portfolio Management**: Test position and P&L calculations
- **Order Execution**: Validate order placement and execution

### 4. Performance Tests
- **Concurrent Requests**: Test handling of multiple simultaneous requests
- **Service Scaling**: Verify scaling capabilities
- **Memory Usage**: Monitor memory consumption under load

## Running Tests

### Local Development

1. **Start Services**:
   ```bash
   make dev
   ```

2. **Run E2E Tests**:
   ```bash
   make test-local
   ```

3. **View Results**:
   ```bash
   make test-results
   ```

### Docker Environment

1. **Build and Run Tests**:
   ```bash
   make test-build
   ```

2. **View Test Logs**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.test.yml logs e2e-tests
   ```

### CI/CD Pipeline

Tests run automatically in the CI/CD pipeline defined in `.github/workflows/ci-cd.yml`.

## Test Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_BASE_URL` | `http://localhost` | Base URL for services |
| `API_GATEWAY_PORT` | `8000` | API Gateway port |
| `TEST_LOG_LEVEL` | `INFO` | Test logging level |
| `SERVICE_WAIT_TIMEOUT` | `300` | Service startup timeout |
| `TEST_RESULTS_DIR` | `./test_results` | Test results directory |

### Command Line Options

```bash
python run_e2e_tests.py [options]

Options:
  --config CONFIG         Path to configuration file
  --results-dir DIR       Results directory
  --base-url URL          Base service URL
  --verbose, -v           Verbose output
```

## Test Structure

### Core Files

- **`test_e2e_framework.py`**: Main testing framework with all test logic
- **`run_e2e_tests.py`**: Test runner script with CLI interface
- **`Dockerfile`**: Docker configuration for test runner
- **`requirements.txt`**: Python dependencies

### Test Results

Test results are saved in the following formats:

- **`e2e_test_results.json`**: Complete test results in JSON format
- **`e2e_test_report.html`**: Human-readable HTML report
- **`e2e_test_report.xml`**: JUnit XML for CI/CD integration
- **`performance_report.json`**: Performance metrics and timing data

## Test Reports

### HTML Report
The HTML report provides a comprehensive view of test results with:
- Test summary and statistics
- Individual test details
- Performance metrics
- Error messages and stack traces

### JUnit XML Report
The XML report is compatible with CI/CD systems and provides:
- Test counts and status
- Individual test timing
- Failure details
- CI/CD integration support

### Performance Report
The performance report includes:
- Test execution times
- Service response times
- Memory usage statistics
- Bottleneck identification

## Extending Tests

### Adding New Test Categories

1. **Create Test Method**:
   ```python
   async def _test_new_feature(self) -> TestResult:
       """Test new feature functionality."""
       start_time = time.time()
       # Test logic here
       return TestResult(...)
   ```

2. **Add to Test Suite**:
   ```python
   async def run_business_logic_tests(self) -> List[TestResult]:
       results = []
       results.append(await self._test_new_feature())
       return results
   ```

### Adding Service Tests

1. **Update Service Ports**:
   ```python
   self.service_ports['new_service'] = 8008
   ```

2. **Add Health Check**:
   ```python
   async def _test_new_service_health(self) -> TestResult:
       return await self._test_service_health('new_service')
   ```

## Test Data Management

### Test Data Setup
- Tests use isolated test data that doesn't affect production
- Test databases are automatically created and cleaned up
- Mock data is used for external API calls

### Data Cleanup
- Automatic cleanup after each test run
- Rollback mechanisms for failed tests
- Isolation between test runs

## Performance Testing

### Load Testing
The framework includes basic load testing capabilities:

```python
# Test concurrent requests
results = await asyncio.gather(*[make_request(i) for i in range(100)])
```

### Scalability Testing
- Tests verify service scaling under load
- Monitors resource usage during tests
- Validates performance degradation points

## Troubleshooting

### Common Issues

1. **Service Not Healthy**
   ```bash
   # Check service logs
   docker-compose logs trading-engine

   # Check service health manually
   curl http://localhost:8001/health
   ```

2. **Test Timeouts**
   ```bash
   # Increase timeout in configuration
   export SERVICE_WAIT_TIMEOUT=600
   ```

3. **Network Issues**
   ```bash
   # Check network connectivity
   docker network ls
   docker network inspect legacy-coin-trader_default
   ```

### Debug Mode

Enable debug logging for detailed test execution:

```bash
export TEST_LOG_LEVEL=DEBUG
python run_e2e_tests.py --verbose
```

## Integration with CI/CD

### GitHub Actions
The framework integrates with GitHub Actions for automated testing:

```yaml
- name: Run E2E tests
  run: make test-build

- name: Upload test results
  uses: actions/upload-artifact@v3
  with:
    name: e2e-test-results
    path: test_results/
```

### Other CI/CD Systems

The test framework can be integrated with any CI/CD system using:
- JUnit XML reports
- Exit codes for pass/fail status
- Configurable timeouts and thresholds
- Environment variable configuration

## Best Practices

### Test Organization
- Keep tests focused and atomic
- Use descriptive test names
- Include clear error messages
- Document test dependencies

### Performance
- Run tests in parallel when possible
- Use appropriate timeouts
- Monitor resource usage
- Clean up after tests

### Reliability
- Handle network failures gracefully
- Implement retry logic for flaky tests
- Use stable test data
- Validate test environments

## Monitoring and Alerting

### Test Metrics
- Test execution time
- Pass/fail rates
- Performance degradation
- Service availability

### Alerts
- Test failures
- Performance regressions
- Service outages
- Resource issues

## Future Enhancements

### Planned Features
- **Distributed Testing**: Run tests across multiple environments
- **Visual Testing**: Screenshot comparisons for UI components
- **Chaos Testing**: Simulate service failures and network issues
- **Performance Benchmarking**: Historical performance tracking
- **Security Testing**: Automated security vulnerability scanning

### Integration Improvements
- **Kubernetes Integration**: Native Kubernetes testing
- **Multi-Region Testing**: Test across different geographical regions
- **Browser Testing**: End-to-end browser automation
- **Mobile Testing**: Mobile app testing integration
