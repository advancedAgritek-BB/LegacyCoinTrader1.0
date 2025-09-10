"""
Dependency Injection Container

This module provides a comprehensive dependency injection container using
dependency-injector to manage application dependencies and ensure proper
separation of concerns.
"""

from dependency_injector import containers, providers
from dependency_injector.providers import Factory, Singleton, Configuration

from .config import AppConfig, get_settings

# Import infrastructure components
try:
    from infrastructure.database import DatabaseConnection, DatabaseRepository
    from infrastructure.cache import RedisCache, InMemoryCache
    from infrastructure.messaging import TelegramNotifier, EmailNotifier
except ImportError:
    try:
        # Try relative imports as fallback
        from ..infrastructure.database import DatabaseConnection, DatabaseRepository
        from ..infrastructure.cache import RedisCache, InMemoryCache
        from ..infrastructure.messaging import TelegramNotifier, EmailNotifier
    except ImportError:
        # Fallback for testing/development
        DatabaseConnection = None
        DatabaseRepository = None
        RedisCache = None
        InMemoryCache = None
        TelegramNotifier = None
        EmailNotifier = None

# Import service components
try:
    from services.trading import TradingService
    from services.risk_management import RiskManager
    from services.portfolio import PortfolioService
    from services.market_data import MarketDataService
    from services.strategy import StrategyService
except ImportError:
    try:
        # Try relative imports as fallback
        from ..services.trading import TradingService
        from ..services.risk_management import RiskManager
        from ..services.portfolio import PortfolioService
        from ..services.market_data import MarketDataService
        from ..services.strategy import StrategyService
    except ImportError:
        # Fallback for testing/development
        TradingService = None
        RiskManager = None
        PortfolioService = None
        MarketDataService = None
        StrategyService = None

# Import domain repositories
try:
    from domain.repositories import (
        PositionRepository,
        TradeRepository,
        SymbolRepository,
        StrategyRepository
    )
except ImportError:
    try:
        # Try relative imports as fallback
        from ..domain.repositories import (
            PositionRepository,
            TradeRepository,
            SymbolRepository,
            StrategyRepository
        )
    except ImportError:
        # Fallback for testing/development
        PositionRepository = None
        TradeRepository = None
        SymbolRepository = None
        StrategyRepository = None

# Import utilities
try:
    from utils.logger import LoggerFactory
    from utils.metrics import MetricsCollector
except ImportError:
    try:
        # Try relative imports as fallback
        from ..utils.logger import LoggerFactory
        from ..utils.metrics import MetricsCollector
    except ImportError:
        # Fallback for testing/development
        LoggerFactory = None
        MetricsCollector = None


class Container(containers.DeclarativeContainer):
    """
    Main dependency injection container.

    This container defines all application dependencies and their relationships,
    providing a centralized way to manage object creation and injection.
    """

    # Configuration provider
    config = providers.Singleton(AppConfig)

    # Logger factory
    logger_factory = providers.Singleton(LoggerFactory)

    # Metrics collector
    metrics_collector = providers.Singleton(MetricsCollector)

    # Database layer
    database_connection = providers.Singleton(
        DatabaseConnection,
        config=config.provided.database,
        logger=logger_factory
    )

    # Cache layer
    redis_cache = providers.Singleton(
        RedisCache,
        config=config.provided.redis,
        logger=logger_factory
    )

    in_memory_cache = providers.Singleton(
        InMemoryCache,
        logger=logger_factory
    )

    # Repository layer
    position_repository = providers.Singleton(
        DatabaseRepository,
        connection=database_connection,
        table_name="positions",
        model_class=providers.Object,  # Will be resolved at runtime
        cache=redis_cache,
        logger=logger_factory
    )

    trade_repository = providers.Singleton(
        DatabaseRepository,
        connection=database_connection,
        table_name="trades",
        model_class=providers.Object,  # Will be resolved at runtime
        cache=redis_cache,
        logger=logger_factory
    )

    symbol_repository = providers.Singleton(
        DatabaseRepository,
        connection=database_connection,
        table_name="symbols",
        model_class=providers.Object,  # Will be resolved at runtime
        cache=in_memory_cache,
        logger=logger_factory
    )

    strategy_repository = providers.Singleton(
        DatabaseRepository,
        connection=database_connection,
        table_name="strategies",
        model_class=providers.Object,  # Will be resolved at runtime
        cache=redis_cache,
        logger=logger_factory
    )

    # Messaging layer
    telegram_notifier = providers.Singleton(
        TelegramNotifier,
        config=config.provided.telegram,
        logger=logger_factory
    )

    email_notifier = providers.Singleton(
        EmailNotifier,
        config=config.provided.security,
        logger=logger_factory
    )

    # Service layer
    market_data_service = providers.Singleton(
        MarketDataService,
        config=config.provided,
        cache=redis_cache,
        logger=logger_factory,
        metrics=metrics_collector
    )

    risk_manager = providers.Singleton(
        RiskManager,
        config=config.provided.trading,
        position_repo=position_repository,
        logger=logger_factory,
        metrics=metrics_collector
    )

    portfolio_service = providers.Singleton(
        PortfolioService,
        config=config.provided.trading,
        position_repo=position_repository,
        risk_manager=risk_manager,
        logger=logger_factory,
        metrics=metrics_collector
    )

    strategy_service = providers.Singleton(
        StrategyService,
        config=config.provided.trading,
        strategy_repo=strategy_repository,
        market_data=market_data_service,
        logger=logger_factory,
        metrics=metrics_collector
    )

    trading_service = providers.Singleton(
        TradingService,
        config=config.provided,
        market_data=market_data_service,
        portfolio=portfolio_service,
        risk_manager=risk_manager,
        strategy=strategy_service,
        telegram=telegram_notifier,
        logger=logger_factory,
        metrics=metrics_collector
    )

    # Application layer
    trading_bot = providers.Singleton(
        providers.Object,  # Will be resolved to TradingBot class
        config=config,
        trading_service=trading_service,
        telegram=telegram_notifier,
        logger=logger_factory,
        metrics=metrics_collector
    )


# Global container instance
container = Container()

# Override configuration with runtime settings
def init_container(config: AppConfig) -> Container:
    """
    Initialize the container with application configuration.

    Args:
        config: Application configuration instance.

    Returns:
        Container: Initialized dependency injection container.
    """
    global container

    # Override configuration provider
    container.config.override(config)

    # Initialize all singleton providers
    container.init_resources()

    return container


def get_container() -> Container:
    """
    Get the global dependency injection container.

    Returns:
        Container: The global container instance.
    """
    return container


def reset_container() -> None:
    """
    Reset the container for testing or reconfiguration.

    This method should only be used in testing scenarios or when
    reconfiguring the application at runtime.
    """
    global container
    container.shutdown_resources()
    container = Container()


# Export commonly used providers
__all__ = [
    "Container",
    "container",
    "init_container",
    "get_container",
    "reset_container",
]
