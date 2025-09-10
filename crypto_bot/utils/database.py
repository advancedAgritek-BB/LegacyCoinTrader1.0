"""Database management utilities for the trading bot."""

import asyncio
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

# Optional asyncpg dependency
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

from .logger import LOG_DIR, setup_logger


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "crypto_bot"
    user: str = "postgres"
    password: str = ""
    min_size: int = 5
    max_size: int = 20
    command_timeout: int = 30
    ssl_mode: str = "prefer"
    application_name: str = "crypto_bot"


class DatabaseManager:
    """Optimized database operations with connection pooling."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "crypto_bot",
        user: str = "postgres",
        password: str = "",
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: int = 30,
        ssl_mode: str = "prefer",
        application_name: str = "crypto_bot"
    ):
        """
        Initialize the database manager.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_size: Minimum connection pool size
            max_size: Maximum connection pool size
            command_timeout: Command timeout in seconds
            ssl_mode: SSL mode for connections
            application_name: Application name for connection identification
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.ssl_mode = ssl_mode
        self.application_name = application_name
        
        # Connection pool
        self.pool: Optional[asyncpg.Pool] = None
        
        # Prepared statements cache
        self.prepared_statements: Dict[str, str] = {}
        
        # Health monitoring
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = 0
        self.connection_errors = 0
        self.max_connection_errors = 10
        
        # Statistics
        self.total_queries = 0
        self.total_errors = 0
        self.avg_query_time = 0.0
        
        self.logger = setup_logger("database_manager", LOG_DIR / "database_manager.log")
    
    def is_available(self) -> bool:
        """Check if database operations are available."""
        return ASYNCPG_AVAILABLE and self.pool is not None
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if not ASYNCPG_AVAILABLE:
            self.logger.warning("asyncpg not available, database operations will be disabled")
            return
            
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout,
                ssl=self.ssl_mode,
                server_settings={
                    "application_name": self.application_name
                }
            )
            
            self.logger.info(
                f"Database pool initialized: {self.host}:{self.port}/{self.database}"
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close(self) -> None:
        """Close all database connections."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a database connection from the pool.
        
        Yields:
            Database connection
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args, **kwargs) -> str:
        """
        Execute a query and return the result.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            **kwargs: Additional connection options
            
        Returns:
            Query result
        """
        if not self.is_available():
            raise RuntimeError("Database not available - asyncpg not installed or pool not initialized")
            
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                result = await conn.execute(query, *args, **kwargs)
                
            self.total_queries += 1
            query_time = asyncio.get_event_loop().time() - start_time
            
            # Update average query time
            self.avg_query_time = (
                (self.avg_query_time * (self.total_queries - 1) + query_time) / 
                self.total_queries
            )
            
            return result
            
        except Exception as e:
            self.total_errors += 1
            self.connection_errors += 1
            self.logger.error(f"Database query error: {e}")
            raise
    
    async def fetch(self, query: str, *args, **kwargs) -> List[Any]:
        """
        Fetch rows from a query.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            **kwargs: Additional connection options
            
        Returns:
            List of records
        """
        if not self.is_available():
            raise RuntimeError("Database not available - asyncpg not installed or pool not initialized")
            
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                result = await conn.fetch(query, *args, **kwargs)
                
            self.total_queries += 1
            query_time = asyncio.get_event_loop().time() - start_time
            
            # Update average query time
            self.avg_query_time = (
                (self.avg_query_time * (self.total_queries - 1) + query_time) / 
                self.total_queries
            )
            
            return result
            
        except Exception as e:
            self.total_errors += 1
            self.connection_errors += 1
            self.logger.error(f"Database fetch error: {e}")
            raise
    
    async def fetchrow(self, query: str, *args, **kwargs) -> Optional[Any]:
        """
        Fetch a single row from a query.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            **kwargs: Additional connection options
            
        Returns:
            Single record or None
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchrow(query, *args, **kwargs)
                
            self.total_queries += 1
            query_time = asyncio.get_event_loop().time() - start_time
            
            # Update average query time
            self.avg_query_time = (
                (self.avg_query_time * (self.total_queries - 1) + query_time) / 
                self.total_queries
            )
            
            return result
            
        except Exception as e:
            self.total_errors += 1
            self.connection_errors += 1
            self.logger.error(f"Database fetchrow error: {e}")
            raise
    
    async def execute_many(self, query: str, args_list: List[tuple]) -> None:
        """
        Execute a query with multiple parameter sets.
        
        Args:
            query: SQL query to execute
            args_list: List of parameter tuples
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                await conn.executemany(query, args_list)
                
            self.total_queries += 1
            query_time = asyncio.get_event_loop().time() - start_time
            
            # Update average query time
            self.avg_query_time = (
                (self.avg_query_time * (self.total_queries - 1) + query_time) / 
                self.total_queries
            )
            
        except Exception as e:
            self.total_errors += 1
            self.connection_errors += 1
            self.logger.error(f"Database executemany error: {e}")
            raise
    
    async def prepare(self, query: str, name: Optional[str] = None) -> str:
        """
        Prepare a query for repeated execution.
        
        Args:
            query: SQL query to prepare
            name: Optional name for the prepared statement
            
        Returns:
            Prepared statement name
        """
        if not name:
            name = f"stmt_{len(self.prepared_statements)}"
        
        try:
            async with self.get_connection() as conn:
                await conn.prepare(query, name=name)
            
            self.prepared_statements[name] = query
            self.logger.debug(f"Prepared statement: {name}")
            
            return name
            
        except Exception as e:
            self.logger.error(f"Failed to prepare statement: {e}")
            raise
    
    async def execute_prepared(self, name: str, *args, **kwargs) -> str:
        """
        Execute a prepared statement.
        
        Args:
            name: Prepared statement name
            *args: Query parameters
            **kwargs: Additional connection options
            
        Returns:
            Query result
        """
        if name not in self.prepared_statements:
            raise ValueError(f"Prepared statement '{name}' not found")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.get_connection() as conn:
                result = await conn.execute(name, *args, **kwargs)
                
            self.total_queries += 1
            query_time = asyncio.get_event_loop().time() - start_time
            
            # Update average query time
            self.avg_query_time = (
                (self.avg_query_time * (self.total_queries - 1) + query_time) / 
                self.total_queries
            )
            
            return result
            
        except Exception as e:
            self.total_errors += 1
            self.connection_errors += 1
            self.logger.error(f"Prepared statement execution error: {e}")
            raise
    
    async def check_health(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        current_time = asyncio.get_event_loop().time()
        
        # Check if we need to perform health check
        if current_time - self.last_health_check < self.health_check_interval:
            return self.connection_errors < self.max_connection_errors
        
        try:
            async with self.get_connection() as conn:
                await conn.execute("SELECT 1")
            
            # Reset error count on successful health check
            self.connection_errors = 0
            self.last_health_check = current_time
            
            return True
            
        except Exception as e:
            self.connection_errors += 1
            self.logger.warning(f"Database health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        return {
            "total_queries": self.total_queries,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_queries, 1),
            "avg_query_time": self.avg_query_time,
            "connection_errors": self.connection_errors,
            "pool_size": self.pool.get_size() if self.pool else 0,
            "free_connections": self.pool.get_free_size() if self.pool else 0,
            "prepared_statements": len(self.prepared_statements)
        }
    
    async def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metric_type VARCHAR(50) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                metadata JSONB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                symbol VARCHAR(20) NOT NULL,
                signal_type VARCHAR(20) NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                metadata JSONB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS cache_performance (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                cache_type VARCHAR(50) NOT NULL,
                hit_rate DOUBLE PRECISION NOT NULL,
                total_accesses INTEGER NOT NULL,
                total_hits INTEGER NOT NULL,
                cache_size INTEGER NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp 
            ON performance_metrics(timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timestamp 
            ON trading_signals(symbol, timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_cache_performance_timestamp 
            ON cache_performance(timestamp);
            """
        ]
        
        for table_sql in tables:
            try:
                await self.execute(table_sql)
            except Exception as e:
                self.logger.error(f"Failed to create table: {e}")
                raise


# Global database manager instance
_global_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get or create the global database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global _global_db_manager
    if _global_db_manager is None:
        _global_db_manager = DatabaseManager()
    return _global_db_manager


async def initialize_database(config: Dict[str, Any]) -> DatabaseManager:
    """
    Initialize the global database manager with configuration.
    
    Args:
        config: Database configuration dictionary
        
    Returns:
        Initialized DatabaseManager instance
    """
    global _global_db_manager
    
    if _global_db_manager is None:
        _global_db_manager = DatabaseManager(**config)
    
    await _global_db_manager.initialize()
    return _global_db_manager


async def close_database() -> None:
    """Close the global database manager."""
    global _global_db_manager
    if _global_db_manager:
        await _global_db_manager.close()
        _global_db_manager = None
