"""
Service discovery for API Gateway using Redis.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Set
import redis

logger = logging.getLogger(__name__)


class ServiceDiscovery:
    """Service discovery using Redis for service registration and lookup."""

    def __init__(self, redis_client: redis.Redis, config):
        self.redis = redis_client
        self.config = config
        self.services: Dict[str, Dict] = {}
        self.watch_tasks: Set[asyncio.Task] = set()
        self.running = False

    async def start(self):
        """Start service discovery."""
        self.running = True
        logger.info("Starting service discovery")

        # Load existing services
        await self._load_services()

        # Start watching for service changes
        await self._start_watching()

    async def stop(self):
        """Stop service discovery."""
        self.running = False

        # Cancel all watch tasks
        for task in self.watch_tasks:
            task.cancel()

        self.watch_tasks.clear()
        logger.info("Service discovery stopped")

    async def register_service(self, service_name: str, url: str,
                             metadata: Dict = None) -> bool:
        """
        Register a service.

        Args:
            service_name: Name of the service
            url: Service URL
            metadata: Additional service metadata

        Returns:
            True if registration successful
        """
        try:
            service_data = {
                'url': url,
                'registered_at': time.time(),
                'metadata': metadata or {},
                'status': 'healthy'
            }

            key = f"service:{service_name}"
            self.redis.setex(
                key,
                self.config.service_ttl,
                json.dumps(service_data)
            )

            # Update local cache
            self.services[service_name] = service_data

            logger.info(f"Registered service: {service_name} at {url}")
            return True

        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")
            return False

    async def unregister_service(self, service_name: str) -> bool:
        """
        Unregister a service.

        Args:
            service_name: Name of the service to unregister

        Returns:
            True if unregistration successful
        """
        try:
            key = f"service:{service_name}"
            self.redis.delete(key)

            # Update local cache
            self.services.pop(service_name, None)

            logger.info(f"Unregistered service: {service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister service {service_name}: {e}")
            return False

    async def get_service_url(self, service_name: str) -> Optional[str]:
        """
        Get the URL for a service.

        Args:
            service_name: Name of the service

        Returns:
            Service URL if found and healthy, None otherwise
        """
        try:
            # Check local cache first
            if service_name in self.services:
                service_data = self.services[service_name]
                if self._is_service_healthy(service_data):
                    return service_data['url']

            # Check Redis
            key = f"service:{service_name}"
            service_json = self.redis.get(key)

            if service_json:
                service_data = json.loads(service_json)

                # Update local cache
                self.services[service_name] = service_data

                if self._is_service_healthy(service_data):
                    return service_data['url']

            return None

        except Exception as e:
            logger.error(f"Failed to get service URL for {service_name}: {e}")
            return None

    async def get_all_services(self) -> Dict[str, Dict]:
        """
        Get all registered services.

        Returns:
            Dictionary of service names to service data
        """
        try:
            # Get all service keys
            keys = self.redis.keys("service:*")

            services = {}
            for key in keys:
                # keys() returns str when decode_responses=True
                service_name = str(key).replace("service:", "")
                service_json = self.redis.get(key)

                if service_json:
                    service_data = json.loads(service_json)
                    services[service_name] = service_data

            return services

        except Exception as e:
            logger.error(f"Failed to get all services: {e}")
            return {}

    async def health_check_service(self, service_name: str) -> bool:
        """
        Perform health check on a service.

        Args:
            service_name: Name of the service to check

        Returns:
            True if service is healthy
        """
        try:
            service_url = await self.get_service_url(service_name)
            if not service_url:
                return False

            # Here you would typically make an HTTP request to the service's
            # health endpoint. For now, we'll just check if the service exists
            # and hasn't expired.

            key = f"service:{service_name}"
            exists = self.redis.exists(key)

            return bool(exists)

        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return False

    def _is_service_healthy(self, service_data: Dict) -> bool:
        """Check if service data indicates a healthy service."""
        # Check if service hasn't expired
        registered_at = service_data.get('registered_at', 0)
        if time.time() - registered_at > self.config.service_ttl:
            return False

        # Check status
        status = service_data.get('status', 'unknown')
        return status == 'healthy'

    async def _load_services(self):
        """Load existing services from Redis."""
        try:
            services = await self.get_all_services()
            self.services.update(services)
            logger.info(f"Loaded {len(services)} existing services")
        except Exception as e:
            logger.error(f"Failed to load existing services: {e}")

    async def _start_watching(self):
        """Start watching for service changes."""
        # In a production system, you might use Redis pub/sub or keyspace
        # notifications to watch for service changes. For now, we'll use
        # a simple polling approach.

        async def watch_services():
            while self.running:
                try:
                    # Check for service changes every 10 seconds
                    await asyncio.sleep(10)

                    current_services = await self.get_all_services()

                    # Compute diffs against previous snapshot before updating
                    previous_keys = set(self.services.keys())
                    current_keys = set(current_services.keys())
                    new_services = current_keys - previous_keys
                    removed_services = previous_keys - current_keys

                    # Update local cache
                    self.services = current_services

                    if new_services:
                        logger.info(f"New services discovered: {list(new_services)}")
                    if removed_services:
                        logger.info(f"Services removed: {list(removed_services)}")

                except Exception as e:
                    logger.error(f"Error watching services: {e}")
                    await asyncio.sleep(5)

        task = asyncio.create_task(watch_services())
        self.watch_tasks.add(task)

        # Clean up completed tasks
        self.watch_tasks = {t for t in self.watch_tasks if not t.done()}
