#!/usr/bin/env python3
"""
Quick Health Check for LegacyCoinTrader Services

This script performs a rapid health check of all microservices
and reports their status in a clean, readable format.
"""

import asyncio
import json
import sys
from typing import Dict, List
import aiohttp


class HealthChecker:
    """Simple health checker for all services."""

    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.services = {
            'API Gateway': 8000,
            'Trading Engine': 8001,
            'Market Data': 8002,
            'Portfolio': 8003,
            'Strategy Engine': 8004,
            'Token Discovery': 8005,
            'Execution': 8006,
            'Monitoring': 8007,
            'Frontend': 5000
        }
        self.http_client = None

    async def setup(self):
        """Setup HTTP client."""
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self.http_client = aiohttp.ClientSession(timeout=timeout)

    async def teardown(self):
        """Cleanup HTTP client."""
        if self.http_client:
            await self.http_client.close()

    async def check_service(self, name: str, port: int) -> Dict:
        """Check health of a single service."""
        url = f"{self.base_url}:{port}/health"

        try:
            start_time = asyncio.get_event_loop().time()
            async with self.http_client.get(url) as response:
                response_time = asyncio.get_event_loop().time() - start_time

                if response.status == 200:
                    try:
                        data = await response.json()
                        status = data.get('status', 'unknown')
                        if status in ['healthy', 'ok']:
                            return {
                                'name': name,
                                'status': '✅ HEALTHY',
                                'response_time': f"{response_time:.2f}s",
                                'details': data.get('message', '')
                            }
                        else:
                            return {
                                'name': name,
                                'status': '⚠️  WARNING',
                                'response_time': f"{response_time:.2f}s",
                                'details': f"Status: {status}"
                            }
                    except json.JSONDecodeError:
                        return {
                            'name': name,
                            'status': '✅ HEALTHY',
                            'response_time': f"{response_time:.2f}s",
                            'details': 'Response received'
                        }
                else:
                    return {
                        'name': name,
                        'status': '❌ DOWN',
                        'response_time': 'N/A',
                        'details': f"HTTP {response.status}"
                    }

        except asyncio.TimeoutError:
            return {
                'name': name,
                'status': '⏰ TIMEOUT',
                'response_time': 'N/A',
                'details': 'Request timed out'
            }
        except aiohttp.ClientError as e:
            return {
                'name': name,
                'status': '❌ ERROR',
                'response_time': 'N/A',
                'details': str(e)
            }
        except Exception as e:
            return {
                'name': name,
                'status': '❌ ERROR',
                'response_time': 'N/A',
                'details': f"Unexpected error: {str(e)}"
            }

    async def check_all_services(self) -> List[Dict]:
        """Check health of all services."""
        tasks = []
        for name, port in self.services.items():
            tasks.append(self.check_service(name, port))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in the results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_name = list(self.services.keys())[i]
                processed_results.append({
                    'name': service_name,
                    'status': '❌ ERROR',
                    'response_time': 'N/A',
                    'details': f"Task failed: {str(result)}"
                })
            else:
                processed_results.append(result)

        return processed_results

    def print_results(self, results: List[Dict]):
        """Print results in a nice format."""
        print("🔍 LegacyCoinTrader Service Health Check")
        print("=" * 60)

        healthy_count = 0
        total_count = len(results)

        for result in results:
            status_indicator = result['status']
            print("25")
            if 'HEALTHY' in result['status']:
                healthy_count += 1

        print("=" * 60)
        print(f"📊 Summary: {healthy_count}/{total_count} services healthy")

        if healthy_count == total_count:
            print("🎉 All services are running correctly!")
            return 0
        elif healthy_count >= total_count * 0.8:
            print("⚠️  Most services are healthy, but some issues detected")
            return 1
        else:
            print("❌ Multiple services are experiencing issues")
            return 2


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='LegacyCoinTrader Health Check')
    parser.add_argument('--url', default='http://localhost',
                       help='Base URL for services (default: http://localhost)')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')

    args = parser.parse_args()

    checker = HealthChecker(args.url)

    try:
        await checker.setup()
        results = await checker.check_all_services()

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            exit_code = checker.print_results(results)
            sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n🛑 Health check interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Health check failed: {e}")
        sys.exit(1)
    finally:
        await checker.teardown()


if __name__ == '__main__':
    asyncio.run(main())
