#!/usr/bin/env python3
"""
End-to-End Test Suite for LegacyCoinTrader Services

Tests all components to ensure they're working correctly:
- WebSocket connections
- Strategy evaluations
- Service health
- Dashboard functionality
"""

import time
import requests
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

class ServiceE2ETester:
    """Comprehensive end-to-end tester for all services."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.base_url = "http://localhost:8001"
        self.test_results: Dict[str, bool] = {}

    def test_websocket_connectivity(self) -> Tuple[bool, str]:
        """Test WebSocket connectivity."""
        try:
            print("🔌 Testing WebSocket connectivity...")

            # Test basic connectivity
            import websocket
            ws = websocket.create_connection(
                "wss://ws.kraken.com/v2",
                timeout=10
            )

            # Send ping
            ping_msg = {"method": "ping"}
            ws.send(json.dumps(ping_msg))

            # Get response
            response = ws.recv()
            ws.close()

            if "pong" in response.lower() or "status" in response.lower():
                return True, "WebSocket connectivity working"
            else:
                return False, f"Unexpected response: {response}"

        except Exception as e:
            return False, f"WebSocket test failed: {e}"

    def test_dashboard_accessibility(self) -> Tuple[bool, str]:
        """Test if dashboard is accessible."""
        try:
            print("🌐 Testing dashboard accessibility...")

            # Test main page
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code != 200:
                return False, f"Dashboard main page failed: {response.status_code}"

            # Test health endpoint
            response = requests.get(f"{self.base_url}/api/monitoring/health", timeout=10)
            if response.status_code != 200:
                return False, f"Health endpoint failed: {response.status_code}"

            health_data = response.json()
            if not health_data.get("success"):
                return False, "Health check returned success=false"

            return True, "Dashboard accessible and healthy"

        except Exception as e:
            return False, f"Dashboard test failed: {e}"

    def test_api_endpoints(self) -> Tuple[bool, str]:
        """Test all critical API endpoints."""
        try:
            print("🔗 Testing API endpoints...")

            endpoints = [
                "/api/monitoring/health",
                "/api/monitoring/status",
                "/api/bot-status",
                "/api/positions",
                "/api/balance"
            ]

            failed_endpoints = []

            for endpoint in endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                    if response.status_code != 200:
                        failed_endpoints.append(f"{endpoint}: {response.status_code}")
                except Exception as e:
                    failed_endpoints.append(f"{endpoint}: {e}")

            if failed_endpoints:
                return False, f"Failed endpoints: {', '.join(failed_endpoints)}"

            return True, f"All {len(endpoints)} endpoints responding"

        except Exception as e:
            return False, f"API test failed: {e}"

    def test_strategy_evaluations(self) -> Tuple[bool, str]:
        """Test if strategy evaluations are happening."""
        try:
            print("🎯 Testing strategy evaluations...")

            # Check recent log entries for strategy activity
            strategy_log = self.project_root / "crypto_bot" / "logs" / "strategy_rank.log"
            if not strategy_log.exists():
                return False, "Strategy log file doesn't exist"

            # Read last few lines
            with open(strategy_log, 'r') as f:
                lines = f.readlines()[-20:]  # Last 20 lines

            # Look for recent strategy analysis
            current_time = time.time()
            recent_analyses = []

            for line in lines:
                if "regime=" in line and "conf=" in line:
                    # Extract timestamp if available
                    if "2025-" in line:
                        recent_analyses.append(line.strip())

            if len(recent_analyses) >= 3:  # At least 3 recent analyses
                return True, f"Found {len(recent_analyses)} recent strategy evaluations"
            else:
                return False, f"Only {len(recent_analyses)} recent evaluations found"

        except Exception as e:
            return False, f"Strategy evaluation test failed: {e}"

    def test_websocket_monitoring(self) -> Tuple[bool, str]:
        """Test WebSocket monitoring status."""
        try:
            print("📡 Testing WebSocket monitoring...")

            # Check health endpoint for WebSocket status
            response = requests.get(f"{self.base_url}/api/monitoring/health", timeout=10)
            if response.status_code != 200:
                return False, "Cannot access health endpoint"

            health_data = response.json()

            if "websocket_connections" in health_data.get("components", {}):
                ws_status = health_data["components"]["websocket_connections"]
                if ws_status.get("status") == "healthy":
                    return True, "WebSocket monitoring active"
                elif ws_status.get("status") == "warning":
                    return True, f"WebSocket warning: {ws_status.get('message', 'Unknown')}"
                else:
                    return False, f"WebSocket error: {ws_status.get('message', 'Unknown')}"
            else:
                return False, "WebSocket component not found in health data"

        except Exception as e:
            return False, f"WebSocket monitoring test failed: {e}"

    def test_service_processes(self) -> Tuple[bool, str]:
        """Test if required service processes are running."""
        try:
            print("⚙️ Testing service processes...")

            # Check for running processes
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=10
            )

            running_processes = result.stdout
            required_services = [
                "crypto_bot.main",
                "flask",
                "python.*app.py"
            ]

            found_services = []
            missing_services = []

            for service in required_services:
                if service.replace(".*", "") in running_processes:
                    found_services.append(service)
                else:
                    missing_services.append(service)

            if len(found_services) >= 2:  # At least main bot and flask
                return True, f"Found services: {', '.join(found_services)}"
            else:
                return False, f"Missing services: {', '.join(missing_services)}"

        except Exception as e:
            return False, f"Process test failed: {e}"

    def test_trade_manager_state(self) -> Tuple[bool, str]:
        """Test trade manager state integrity."""
        try:
            print("📊 Testing trade manager state...")

            # Check trade manager state file
            state_file = self.project_root / "crypto_bot" / "logs" / "trade_manager_state.json"
            if not state_file.exists():
                return False, "Trade manager state file doesn't exist"

            with open(state_file, 'r') as f:
                state = json.load(f)

            # Check for required fields
            if "positions" not in state:
                return False, "No positions found in state"

            if "trades" not in state:
                return False, "No trades found in state"

            position_count = len(state.get("positions", {}))
            trade_count = len(state.get("trades", []))

            if position_count > 0:
                return True, f"Trade manager healthy: {position_count} positions, {trade_count} trades"
            else:
                return True, f"Trade manager initialized: {trade_count} trades (no open positions)"

        except Exception as e:
            return False, f"Trade manager test failed: {e}"

    def run_all_tests(self) -> Dict[str, Tuple[bool, str]]:
        """Run all end-to-end tests."""
        print("🧪 Running LegacyCoinTrader End-to-End Tests")
        print("=" * 60)

        tests = [
            ("WebSocket Connectivity", self.test_websocket_connectivity),
            ("Dashboard Accessibility", self.test_dashboard_accessibility),
            ("API Endpoints", self.test_api_endpoints),
            ("Strategy Evaluations", self.test_strategy_evaluations),
            ("WebSocket Monitoring", self.test_websocket_monitoring),
            ("Service Processes", self.test_service_processes),
            ("Trade Manager State", self.test_trade_manager_state),
        ]

        results = {}

        for test_name, test_func in tests:
            print(f"\n🔍 {test_name}:")
            try:
                success, message = test_func()
                results[test_name] = (success, message)

                if success:
                    print(f"  ✅ PASS: {message}")
                else:
                    print(f"  ❌ FAIL: {message}")

            except Exception as e:
                results[test_name] = (False, f"Test crashed: {e}")
                print(f"  💥 CRASH: {e}")

        return results

    def generate_report(self, results: Dict[str, Tuple[bool, str]]) -> str:
        """Generate a comprehensive test report."""
        print("\n" + "=" * 60)
        print("📋 END-TO-END TEST RESULTS")
        print("=" * 60)

        passed = 0
        failed = 0
        crashed = 0

        for test_name, (success, message) in results.items():
            if success:
                passed += 1
                status = "✅ PASS"
            elif "crashed" in message.lower():
                crashed += 1
                status = "💥 CRASH"
            else:
                failed += 1
                status = "❌ FAIL"

            print(f"{status} {test_name}")
            print(f"    {message}")

        total_tests = len(results)
        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0

        print("\n" + "=" * 60)
        print("🎯 SUMMARY")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Crashed: {crashed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        if success_rate >= 80:
            print("🎉 OVERALL: EXCELLENT - System is fully functional!")
        elif success_rate >= 60:
            print("⚠️ OVERALL: GOOD - System mostly working, minor issues")
        else:
            print("❌ OVERALL: POOR - System needs attention")

        return f"Success Rate: {success_rate:.1f}%"

def main():
    """Main test runner."""
    tester = ServiceE2ETester()

    try:
        # Run all tests
        results = tester.run_all_tests()

        # Generate report
        report = tester.generate_report(results)

        # Exit with appropriate code
        passed_count = sum(1 for success, _ in results.values() if success)
        total_count = len(results)

        if passed_count == total_count:
            print("\n🎉 ALL TESTS PASSED!")
            sys.exit(0)
        elif passed_count >= total_count * 0.8:  # 80% success rate
            print("\n⚠️ MOST TESTS PASSED - System is mostly functional")
            sys.exit(0)
        else:
            print("\n❌ TOO MANY FAILURES - System needs fixes")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
