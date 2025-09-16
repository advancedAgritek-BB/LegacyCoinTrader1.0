"""Global test configuration and fixtures."""
import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import Callable, Iterable, Optional

# Add crypto_bot to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their names."""
    for item in items:
        # Mark async tests
        if "async" in item.name.lower() or "await" in str(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Mark integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
            
        # Mark Solana tests
        if "solana" in item.name.lower():
            item.add_marker(pytest.mark.solana)
            
        # Mark Kraken tests
        if "kraken" in item.name.lower():
            item.add_marker(pytest.mark.kraken)

# Markers
pytest_plugins = ["pytest_asyncio"]

# Network dependencies
skip_net = pytest.mark.skip(reason="network dependencies not installed")

# Standard mocks
@pytest.fixture
def mock_exchange():
    """Standard exchange mock for all tests."""
    exchange = Mock()
    exchange.create_order = AsyncMock(return_value={'id': 'test_order_123'})
    exchange.cancel_order = AsyncMock(return_value=True)
    exchange.fetch_order = AsyncMock(return_value={'status': 'closed'})
    exchange.fetch_ticker = AsyncMock(return_value={'last': 100.0, 'bid': 99.5, 'ask': 100.5})
    exchange.fetch_balance = AsyncMock(return_value={'BTC': {'free': 1.0, 'total': 1.0}})
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.options = {'defaultType': 'spot'}
    exchange.has = {'fetchOHLCV': True, 'fetchTicker': True}
    return exchange

@pytest.fixture
def mock_solana_client():
    """Standard Solana client mock."""
    client = Mock()
    client.get_balance = AsyncMock(return_value=1000000000)  # 1 SOL in lamports
    client.get_account_info = AsyncMock(return_value={'data': b'test_data'})
    client.send_transaction = AsyncMock(return_value='test_signature')
    client.confirm_transaction = AsyncMock(return_value=True)
    return client

@pytest.fixture
def mock_telegram_bot():
    """Standard Telegram bot mock."""
    bot = Mock()
    bot.send_message = AsyncMock(return_value={'message_id': 123})
    bot.edit_message_text = AsyncMock(return_value=True)
    bot.answer_callback_query = AsyncMock(return_value=True)
    return bot

@pytest.fixture
def mock_config():
    """Standard configuration mock."""
    return {
        'trading': {
            'enabled': True,
            'max_positions': 5,
            'risk_per_trade': 0.02
        },
        'solana': {
            'enabled': True,
            'rpc_url': 'https://api.mainnet-beta.solana.com',
            'private_key': 'test_key'
        },
        'telegram': {
            'enabled': True,
            'bot_token': 'test_token',
            'chat_id': 'test_chat'
        },
        'enhanced_scanning': {
            'enabled': True,
            'scan_interval': 30
        }
    }

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    data = {
        'timestamp': dates,
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(95, 115, 100),
        'low': np.random.uniform(85, 105, 100),
        'close': np.random.uniform(90, 110, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def _build_ohlcv(
    closes: Iterable[float],
    *,
    volumes: Optional[Iterable[float]] = None,
    step_seconds: int = 60,
) -> pd.DataFrame:
    """Construct a minimal OHLCV dataframe from a series of closes.

    The helper keeps the open near the previous close and adds a small
    buffer to the high/low so strategies relying on candle ranges work
    with deterministic values.
    """

    closes = np.asarray(list(closes), dtype=float)
    if closes.size == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    if volumes is None:
        volumes = np.full(closes.shape, 1_000.0)
    else:
        volumes = np.asarray(list(volumes), dtype=float)
        if volumes.shape != closes.shape:
            raise ValueError("Volume series must match closes length")

    index = pd.date_range(
        "2024-01-01", periods=len(closes), freq=pd.Timedelta(seconds=step_seconds)
    )
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )


@pytest.fixture
def make_ohlcv() -> Callable[[Iterable[float]], pd.DataFrame]:
    """Factory fixture returning deterministic OHLCV builders."""

    def _factory(closes: Iterable[float], *, volumes: Optional[Iterable[float]] = None,
                 step_seconds: int = 60) -> pd.DataFrame:
        return _build_ohlcv(closes, volumes=volumes, step_seconds=step_seconds)

    return _factory


@pytest.fixture
def ohlcv_trending_up(make_ohlcv):
    """Upward trending OHLCV series."""

    closes = np.linspace(100, 120, 60)
    volumes = np.linspace(1_000, 2_000, 60)
    return make_ohlcv(closes, volumes=volumes)


@pytest.fixture
def ohlcv_trending_down(make_ohlcv):
    """Downward trending OHLCV series."""

    closes = np.linspace(120, 100, 60)
    volumes = np.linspace(2_000, 1_000, 60)
    return make_ohlcv(closes, volumes=volumes)


@pytest.fixture
def ohlcv_range_bound(make_ohlcv):
    """Range-bound market useful for mean-reversion scenarios."""

    base = 100 + np.sin(np.linspace(0, 3 * np.pi, 60))
    volumes = np.full_like(base, 1_500.0)
    return make_ohlcv(base, volumes=volumes)

@pytest.fixture
def sample_position():
    """Standard position object for testing."""
    return {
        'symbol': 'BTC/USDT',
        'side': 'long',
        'amount': 1.0,
        'entry_price': 100.0,
        'current_price': 105.0,
        'unrealized_pnl': 5.0,
        'realized_pnl': 0.0,
        'timestamp': pd.Timestamp.now()
    }

@pytest.fixture
def mock_wallet():
    """Standard wallet mock."""
    wallet = Mock()
    wallet.public_key = 'test_public_key'
    wallet.balance = 1000.0
    wallet.get_balance = AsyncMock(return_value=1000.0)
    wallet.send_transaction = AsyncMock(return_value='test_tx_hash')
    return wallet

@pytest.fixture
def mock_risk_manager():
    """Standard risk manager mock."""
    risk_manager = Mock()
    risk_manager.calculate_position_size = Mock(return_value=0.1)
    risk_manager.check_risk_limits = Mock(return_value=True)
    risk_manager.should_exit = Mock(return_value=False)
    return risk_manager

@pytest.fixture
def mock_strategy():
    """Standard strategy mock."""
    strategy = Mock()
    strategy.generate_signal = Mock(return_value={'action': 'buy', 'confidence': 0.8})
    strategy.calculate_score = Mock(return_value=0.85)
    strategy.is_active = Mock(return_value=True)
    return strategy

@pytest.fixture
def mock_market_data():
    """Standard market data mock."""
    market_data = Mock()
    market_data.get_ticker = AsyncMock(return_value={'last': 100.0, 'volume': 1000000})
    market_data.get_ohlcv = AsyncMock(return_value=sample_ohlcv_data())
    market_data.get_order_book = AsyncMock(return_value={'bids': [[99.5, 1.0]], 'asks': [[100.5, 1.0]]})
    return market_data

@pytest.fixture
def sample_market_data():
    """Generate sample market data for backtesting tests."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    base_price = 100.0
    returns = np.random.normal(0, 0.02, 1000)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(max(0.1, prices[-1] * (1 + ret)))
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    return df.set_index('timestamp')

# Fix for common import issues
@pytest.fixture(autouse=True)
def fix_imports():
    """Fix common import path issues."""
    # Mock problematic modules that may not exist
    with patch.dict('sys.modules', {
        'crypto_bot.volatility_filter': Mock(),
        'crypto_bot.volatility_filter.requests': Mock(),
        'crypto_bot.fund_manager': Mock(),
        'crypto_bot.portfolio_rotator': Mock(),
        'crypto_bot.regime': Mock(),
        'crypto_bot.utils.regime_pnl_tracker': Mock(),
        'crypto_bot.utils.market_analyzer': Mock(),
        'crypto_bot.strategy.grid_bot': Mock(),
        'crypto_bot.execution': Mock(),
        'crypto_bot.execution.cex_executor': Mock(),
        'crypto_bot.execution.solana_mempool': Mock(),
        'crypto_bot.execution.solana_executor': Mock(),
    }):
        # Mock specific functions that tests expect, but only if the module exists
        try:
            with patch('crypto_bot.utils.telegram.send_message', Mock()) as mock_send:
                yield
        except (ImportError, AttributeError):
            # If telegram module doesn't exist or doesn't have send_message, just continue
            yield

# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Test data factories
def create_sample_trade(symbol="BTC/USDT", side="buy", amount=1.0, price=100.0):
    """Create a sample trade object."""
    return {
        'id': 'trade_123',
        'symbol': symbol,
        'side': side,
        'amount': amount,
        'price': price,
        'timestamp': pd.Timestamp.now(),
        'fee': {'cost': 0.001, 'currency': 'USDT'}
    }

def create_sample_order(symbol="BTC/USDT", side="buy", amount=1.0, price=100.0):
    """Create a sample order object."""
    return {
        'id': 'order_123',
        'symbol': symbol,
        'side': side,
        'amount': amount,
        'price': price,
        'status': 'open',
        'timestamp': pd.Timestamp.now()
    }

# Performance test helpers
@pytest.fixture
def performance_timer():
    """Timer fixture for performance tests."""
    import time
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    print(f"Test execution time: {elapsed:.3f}s")

# Network test helpers
@pytest.fixture
def mock_requests():
    """Mock requests for network tests."""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'data': 'test_data'}
        mock_get.return_value.raise_for_status.return_value = None
        yield mock_get

@pytest.fixture
def mock_aiohttp():
    """Mock aiohttp for async network tests."""
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_get.return_value.__aenter__.return_value.status = 200
        mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value={'data': 'test_data'})
        yield mock_get
# Lightweight analyzer fixture used by pool analyzer tests
@pytest.fixture
def analyzer():
    from crypto_bot.solana.pool_analyzer import PoolAnalyzer
    return PoolAnalyzer()
