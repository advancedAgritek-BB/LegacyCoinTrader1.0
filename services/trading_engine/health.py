"""
Health checking for Trading Engine service.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for the trading engine service."""

    def __init__(self, config, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.last_health_check = None
        self.consecutive_failures = 0
        self.is_healthy = True

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health_status = {
                'service': 'trading_engine',
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'checks': {}
            }

            # Check orchestrator status
            orchestrator_status = await self._check_orchestrator()
            health_status['checks']['orchestrator'] = orchestrator_status

            # Check dependencies
            dependency_status = await self._check_dependencies()
            health_status['checks']['dependencies'] = dependency_status

            # Check performance
            performance_status = await self._check_performance()
            health_status['checks']['performance'] = performance_status

            # Overall status
            all_checks_passed = all(
                check.get('status') == 'healthy'
                for check in health_status['checks'].values()
            )

            if all_checks_passed:
                health_status['status'] = 'healthy'
                self.consecutive_failures = 0
                self.is_healthy = True
            else:
                health_status['status'] = 'unhealthy'
                self.consecutive_failures += 1

                if self.consecutive_failures >= self.config.max_consecutive_failures:
                    self.is_healthy = False

            self.last_health_check = datetime.utcnow()

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'service': 'trading_engine',
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }

    async def _check_orchestrator(self) -> Dict[str, Any]:
        """Check orchestrator health."""
        try:
            if not self.orchestrator:
                return {'status': 'unhealthy', 'message': 'Orchestrator not initialized'}

            status = await self.orchestrator.get_status()

            if status.get('running'):
                return {
                    'status': 'healthy',
                    'message': 'Orchestrator is running',
                    'details': status
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': 'Orchestrator is not running',
                    'details': status
                }

        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Orchestrator check failed: {str(e)}'}

    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check service dependencies."""
        try:
            dependencies = {
                'market_data': self.orchestrator.service_urls.get('market_data'),
                'portfolio': self.orchestrator.service_urls.get('portfolio'),
                'strategy_engine': self.orchestrator.service_urls.get('strategy_engine'),
                'execution': self.orchestrator.service_urls.get('execution'),
                'monitoring': self.orchestrator.service_urls.get('monitoring')
            }

            missing_deps = [name for name, url in dependencies.items() if not url]

            if missing_deps:
                return {
                    'status': 'degraded',
                    'message': f'Missing service URLs: {missing_deps}',
                    'available': {name: url for name, url in dependencies.items() if url}
                }
            else:
                return {
                    'status': 'healthy',
                    'message': 'All service dependencies available',
                    'services': list(dependencies.keys())
                }

        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Dependency check failed: {str(e)}'}

    async def _check_performance(self) -> Dict[str, Any]:
        """Check performance metrics."""
        try:
            # Get recent cycle performance
            if self.orchestrator.cycle_count > 0:
                avg_cycle_time = self.orchestrator.average_cycle_time
                last_cycle_time = self.orchestrator.last_cycle_time

                # Check if cycles are running within expected time
                max_expected_cycle_time = self.config.cycle_interval * 1.5

                if avg_cycle_time > max_expected_cycle_time:
                    return {
                        'status': 'degraded',
                        'message': f'Average cycle time too high: {avg_cycle_time:.2f}s',
                        'metrics': {
                            'average_cycle_time': avg_cycle_time,
                            'last_cycle_time': last_cycle_time.isoformat() if last_cycle_time else None,
                            'cycle_count': self.orchestrator.cycle_count
                        }
                    }
                else:
                    return {
                        'status': 'healthy',
                        'message': f'Performance within limits: {avg_cycle_time:.2f}s avg cycle time',
                        'metrics': {
                            'average_cycle_time': avg_cycle_time,
                            'last_cycle_time': last_cycle_time.isoformat() if last_cycle_time else None,
                            'cycle_count': self.orchestrator.cycle_count
                        }
                    }
            else:
                return {
                    'status': 'unknown',
                    'message': 'No trading cycles completed yet',
                    'metrics': {'cycle_count': 0}
                }

        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Performance check failed: {str(e)}'}

    def is_service_healthy(self) -> bool:
        """Check if service is currently healthy."""
        return self.is_healthy

    def get_last_health_check(self) -> Optional[datetime]:
        """Get timestamp of last health check."""
        return self.last_health_check
