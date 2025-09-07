#!/usr/bin/env python3
"""
E2E Test Runner for LegacyCoinTrader Microservices

This script orchestrates the complete end-to-end testing process including:
- Service health checks
- Integration testing
- Performance testing
- Load testing
- Report generation
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from test_e2e_framework import E2ETestFramework, TestStatus


class E2ETestRunner:
    """E2E Test Runner with comprehensive reporting and CI/CD integration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_framework = E2ETestFramework(
            base_url=config.get('base_url', 'http://localhost'),
            service_ports=config.get('service_ports', {})
        )
        self.results_dir = Path(config.get('results_dir', './test_results'))
        self.results_dir.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for test runner."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'e2e_test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def run_tests(self) -> Dict[str, Any]:
        """Run the complete E2E test suite."""
        self.logger.info("Starting E2E Test Suite")

        start_time = time.time()
        test_results = None

        try:
            # Pre-test setup
            await self._pre_test_setup()

            # Run tests
            test_results = await self.test_framework.run_all_tests()

            # Post-test cleanup
            await self._post_test_cleanup()

            # Generate reports
            await self._generate_reports(test_results)

        except Exception as e:
            self.logger.error(f"E2E test suite failed: {e}")
            test_results = {
                'summary': {'total': 0, 'passed': 0, 'failed': 1, 'skipped': 0, 'success_rate': 0},
                'results': [],
                'duration': time.time() - start_time,
                'error': str(e)
            }

        finally:
            # Save final results
            await self._save_results(test_results)

        return test_results

    async def _pre_test_setup(self):
        """Setup before running tests."""
        self.logger.info("Performing pre-test setup")

        # Wait for services to be healthy
        await self._wait_for_services()

        # Initialize test data
        await self._initialize_test_data()

        # Warm up services
        await self._warm_up_services()

    async def _wait_for_services(self):
        """Wait for all services to be healthy."""
        self.logger.info("Waiting for services to be healthy")

        max_wait = self.config.get('service_wait_timeout', 300)  # 5 minutes
        check_interval = self.config.get('service_check_interval', 10)

        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # Check all service health
                health_results = await self.test_framework.run_health_checks()
                failed_services = [
                    result.name for result in health_results
                    if result.status != TestStatus.PASSED
                ]

                if not failed_services:
                    self.logger.info("All services are healthy")
                    return

                self.logger.info(f"Waiting for services: {failed_services}")
                await asyncio.sleep(check_interval)

            except Exception as e:
                self.logger.warning(f"Health check failed: {e}")
                await asyncio.sleep(check_interval)

        raise TimeoutError(f"Services not healthy after {max_wait} seconds")

    async def _initialize_test_data(self):
        """Initialize test data."""
        self.logger.info("Initializing test data")

        # This would initialize test data in the services
        # For now, we'll just log the action
        pass

    async def _warm_up_services(self):
        """Warm up services with initial requests."""
        self.logger.info("Warming up services")

        try:
            # Make some initial requests to warm up caches
            gateway_url = self.test_framework.get_service_url('api_gateway')

            # Warm up API Gateway
            async with self.test_framework.http_client.get(f"{gateway_url}/health") as response:
                if response.status != 200:
                    self.logger.warning("API Gateway warm-up failed")

            # Warm up other services
            for service in ['trading_engine', 'market_data', 'portfolio']:
                try:
                    service_url = self.test_framework.get_service_url(service)
                    async with self.test_framework.http_client.get(f"{service_url}/health") as response:
                        if response.status != 200:
                            self.logger.warning(f"{service} warm-up failed")
                except Exception as e:
                    self.logger.warning(f"Failed to warm up {service}: {e}")

        except Exception as e:
            self.logger.warning(f"Service warm-up failed: {e}")

    async def _post_test_cleanup(self):
        """Cleanup after tests."""
        self.logger.info("Performing post-test cleanup")

        # Clean up test data
        await self._cleanup_test_data()

        # Reset service state if needed
        await self._reset_service_state()

    async def _cleanup_test_data(self):
        """Clean up test data."""
        self.logger.info("Cleaning up test data")

        # This would clean up test data from services
        # For now, we'll just log the action
        pass

    async def _reset_service_state(self):
        """Reset service state after tests."""
        self.logger.info("Resetting service state")

        # This would reset services to a clean state
        # For now, we'll just log the action
        pass

    async def _generate_reports(self, test_results: Dict[str, Any]):
        """Generate test reports."""
        self.logger.info("Generating test reports")

        # Generate HTML report
        await self._generate_html_report(test_results)

        # Generate JUnit XML report for CI/CD
        await self._generate_junit_report(test_results)

        # Generate performance report
        await self._generate_performance_report(test_results)

    async def _generate_html_report(self, test_results: Dict[str, Any]):
        """Generate HTML test report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LegacyCoinTrader E2E Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .test-result {{ margin-bottom: 10px; padding: 10px; border-radius: 5px; }}
                .passed {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
                .failed {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                .skipped {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .error {{ color: #721c24; }}
                .duration {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>LegacyCoinTrader E2E Test Report</h1>
            <div class="summary">
                <h2>Test Summary</h2>
                <p><strong>Total Tests:</strong> {test_results['summary']['total']}</p>
                <p><strong>Passed:</strong> {test_results['summary']['passed']}</p>
                <p><strong>Failed:</strong> {test_results['summary']['failed']}</p>
                <p><strong>Skipped:</strong> {test_results['summary']['skipped']}</p>
                <p><strong>Success Rate:</strong> {test_results['summary']['success_rate']:.1f}%</p>
                <p><strong>Duration:</strong> {test_results['duration']:.2f}s</p>
                <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
            </div>

            <h2>Test Results</h2>
        """

        for result in test_results['results']:
            status_class = result['status'].lower()
            html_content += f"""
            <div class="test-result {status_class}">
                <h3>{result['name']}</h3>
                <p><strong>Status:</strong> {result['status']}</p>
                <p><strong>Message:</strong> {result['message']}</p>
                <p class="duration"><strong>Duration:</strong> {result['duration']:.3f}s</p>
            """

            if result.get('error'):
                html_content += f'<p class="error"><strong>Error:</strong> {result["error"]}</p>'

            if result.get('details'):
                html_content += f'<p><strong>Details:</strong> {json.dumps(result["details"], indent=2)}</p>'

            html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        report_file = self.results_dir / 'e2e_test_report.html'
        with open(report_file, 'w') as f:
            f.write(html_content)

        self.logger.info(f"HTML report generated: {report_file}")

    async def _generate_junit_report(self, test_results: Dict[str, Any]):
        """Generate JUnit XML report for CI/CD."""
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
        <testsuites>
            <testsuite name="LegacyCoinTrader E2E Tests"
                      tests="{test_results['summary']['total']}"
                      failures="{test_results['summary']['failed']}"
                      skipped="{test_results['summary']['skipped']}"
                      time="{test_results['duration']}"
                      timestamp="{datetime.now().isoformat()}">
        """

        for result in test_results['results']:
            xml_content += f"""
                <testcase name="{result['name']}"
                         time="{result['duration']}"
                         classname="e2e_tests">
            """

            if result['status'] == 'failed':
                xml_content += f"""
                    <failure message="{result['message']}">
                        {result.get('error', '')}
                    </failure>
                """

            if result['status'] == 'skipped':
                xml_content += f"""
                    <skipped message="{result['message']}" />
                """

            xml_content += "</testcase>"

        xml_content += """
            </testsuite>
        </testsuites>
        """

        report_file = self.results_dir / 'e2e_test_report.xml'
        with open(report_file, 'w') as f:
            f.write(xml_content)

        self.logger.info(f"JUnit XML report generated: {report_file}")

    async def _generate_performance_report(self, test_results: Dict[str, Any]):
        """Generate performance metrics report."""
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': test_results['duration'],
            'test_metrics': []
        }

        for result in test_results['results']:
            performance_data['test_metrics'].append({
                'name': result['name'],
                'duration': result['duration'],
                'status': result['status']
            })

        report_file = self.results_dir / 'performance_report.json'
        with open(report_file, 'w') as f:
            json.dump(performance_data, f, indent=2)

        self.logger.info(f"Performance report generated: {report_file}")

    async def _save_results(self, test_results: Dict[str, Any]):
        """Save test results to file."""
        results_file = self.results_dir / 'e2e_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        self.logger.info(f"Test results saved: {results_file}")


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load test configuration."""
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            return json.load(f)

    # Default configuration
    return {
        'base_url': os.getenv('TEST_BASE_URL', 'http://localhost'),
        'service_ports': {
            'api_gateway': int(os.getenv('API_GATEWAY_PORT', '8000')),
            'trading_engine': int(os.getenv('TRADING_ENGINE_PORT', '8001')),
            'market_data': int(os.getenv('MARKET_DATA_PORT', '8002')),
            'portfolio': int(os.getenv('PORTFOLIO_PORT', '8003')),
            'strategy_engine': int(os.getenv('STRATEGY_ENGINE_PORT', '8004')),
            'token_discovery': int(os.getenv('TOKEN_DISCOVERY_PORT', '8005')),
            'execution': int(os.getenv('EXECUTION_PORT', '8006')),
            'monitoring': int(os.getenv('MONITORING_PORT', '8007')),
            'frontend': int(os.getenv('FRONTEND_PORT', '5000'))
        },
        'results_dir': os.getenv('TEST_RESULTS_DIR', './test_results'),
        'log_level': os.getenv('TEST_LOG_LEVEL', 'INFO'),
        'service_wait_timeout': int(os.getenv('SERVICE_WAIT_TIMEOUT', '300')),
        'service_check_interval': int(os.getenv('SERVICE_CHECK_INTERVAL', '10'))
    }


async def main():
    """Main entry point for E2E test runner."""
    parser = argparse.ArgumentParser(description='LegacyCoinTrader E2E Test Runner')
    parser.add_argument('--config', help='Path to test configuration file')
    parser.add_argument('--results-dir', help='Directory to store test results')
    parser.add_argument('--base-url', help='Base URL for services')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.results_dir:
        config['results_dir'] = args.results_dir
    if args.base_url:
        config['base_url'] = args.base_url
    if args.verbose:
        config['log_level'] = 'DEBUG'

    # Create test runner
    runner = E2ETestRunner(config)

    try:
        # Run tests
        results = await runner.run_tests()

        # Print summary
        summary = results['summary']
        print(f"\n{'='*60}")
        print("E2E TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(".1f")
        print(".2f")
        print(f"{'='*60}")

        # Exit with appropriate code
        if summary['failed'] > 0:
            print("❌ Some tests failed!")
            sys.exit(1)
        else:
            print("✅ All tests passed!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
