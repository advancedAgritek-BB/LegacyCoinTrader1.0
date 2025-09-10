import asyncio
import types
import pytest

from crypto_bot.solana import pool_ws_monitor


class DummyMsg:
    def __init__(self, data):
        self.data = data
        self.type = pool_ws_monitor.aiohttp.WSMsgType.TEXT

    def json(self):
        return self.data


class DummyWS:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __aiter__(self):
        async def gen():
            for m in self.messages:
                yield DummyMsg(m)
        return gen()


class DummySession:
    def __init__(self, ws):
        self.ws = ws
        self.url = None

    def ws_connect(self, url):
        self.url = url
        return self.ws

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class AiohttpMod:
    WSMsgType = types.SimpleNamespace(TEXT="text", CLOSED="closed", ERROR="error")

    def __init__(self, session):
        self._session = session

    def ClientSession(self):
        return self._session


def test_subscription_message(monkeypatch):
    ws = DummyWS([])
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)

    async def run():
        gen = pool_ws_monitor.watch_pool("KEY", "PGM")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    asyncio.run(run())
    assert session.url == "wss://atlas-mainnet.helius-rpc.com/?api-key=KEY"
    assert ws.sent and ws.sent[0]["params"][0]["accountInclude"] == ["PGM"]


def test_yields_transactions(monkeypatch):
    messages = [
        {"params": {"result": {"tx": 1}}},
        {"params": {"result": {"tx": 2}}},
    ]
    ws = DummyWS(messages)
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)

    async def run():
        gen = pool_ws_monitor.watch_pool("KEY", "PGM")
        results = [await gen.__anext__(), await gen.__anext__()]
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
        return results

    res = asyncio.run(run())
    assert res == [{"tx": 1}, {"tx": 2}]


# Tests for EnhancedPoolMonitor
def test_enhanced_monitor_initialization():
    """Test enhanced monitor initialization."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    assert monitor.api_key == "test_key"
    assert monitor.pool_program == "test_pool"
    assert monitor.connection_attempts == 0
    assert monitor.is_connected is False
    assert monitor.total_messages_received == 0
    assert monitor.total_errors == 0


def test_enhanced_monitor_custom_config():
    """Test enhanced monitor with custom configuration."""
    custom_config = {
        "reconnect": {
            "base_delay": 2.0,
            "max_delay": 120.0,
            "backoff_factor": 3.0,
            "max_attempts": 5,
        },
        "health_check": {
            "enabled": False,
            "interval": 60,
        },
        "message_validation": {
            "max_message_size": 5 * 1024 * 1024,  # 5MB
        }
    }

    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool", custom_config)

    assert monitor.reconnect_delay == 2.0
    assert monitor.max_reconnect_delay == 120.0
    assert monitor.backoff_factor == 3.0
    assert monitor.max_attempts == 5
    assert monitor.config["message_validation"]["max_message_size"] == 5 * 1024 * 1024


def test_calculate_reconnect_delay():
    """Test exponential backoff delay calculation."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    monitor.connection_attempts = 0
    assert monitor._calculate_reconnect_delay() == 1.0

    monitor.connection_attempts = 1
    assert monitor._calculate_reconnect_delay() == 2.0

    monitor.connection_attempts = 2
    assert monitor._calculate_reconnect_delay() == 4.0

    # Test max delay limit
    monitor.connection_attempts = 10
    delay = monitor._calculate_reconnect_delay()
    assert delay == monitor.max_reconnect_delay


def test_should_attempt_reconnect():
    """Test reconnection attempt logic."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    monitor.max_attempts = 5
    monitor.connection_attempts = 3
    assert monitor._should_attempt_reconnect() is True

    monitor.connection_attempts = 5
    assert monitor._should_attempt_reconnect() is False

    # Test unlimited attempts
    monitor.max_attempts = 0
    monitor.connection_attempts = 100
    assert monitor._should_attempt_reconnect() is True


def test_validate_message():
    """Test message validation."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    # Valid JSON
    valid_msg = '{"test": "data"}'
    assert monitor._validate_message(valid_msg) is True

    # Invalid JSON
    invalid_msg = '{"test": invalid}'
    assert monitor._validate_message(invalid_msg) is False

    # Oversized message
    large_msg = "x" * (monitor.config["message_validation"]["max_message_size"] + 1)
    assert monitor._validate_message(large_msg) is False


@pytest.mark.asyncio
async def test_handle_connection_success():
    """Test successful connection handling."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    await monitor._handle_connection_success()

    assert monitor.connection_attempts == 0
    assert monitor.is_connected is True
    assert monitor.last_connection_time is not None


@pytest.mark.asyncio
async def test_handle_connection_failure(monkeypatch):
    """Test connection failure handling."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    # Configure for quick failure in tests
    monitor.max_attempts = 2
    monitor.connection_attempts = 1

    error = Exception("Test connection error")

    with monkeypatch.context() as m:
        m.setattr('asyncio.sleep', lambda x: None)  # Mock sleep

        with pytest.raises(Exception, match="Test connection error"):
            await monitor._handle_connection_failure(error)

    assert monitor.is_connected is False
    assert monitor.total_errors == 1
    assert monitor.connection_attempts == 2


def test_create_subscription_message():
    """Test subscription message creation."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    import asyncio
    msg = asyncio.run(monitor._create_subscription_message())

    assert msg["jsonrpc"] == "2.0"
    assert msg["id"] == monitor.subscription_id
    assert msg["method"] == "transactionSubscribe"
    assert "params" in msg
    assert len(msg["params"]) == 2
    assert msg["params"][0]["accountInclude"] == ["test_pool"]


def test_log_connection_stats():
    """Test connection statistics logging."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    monitor.total_messages_received = 100
    monitor.total_errors = 5
    from datetime import datetime, timedelta
    monitor.start_time = datetime.now() - timedelta(hours=2)

    # Should not raise any exceptions
    monitor._log_connection_stats()


@pytest.mark.asyncio
async def test_health_check_loop_disabled():
    """Test health check loop when disabled."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    # Disable health check
    monitor.config["health_check"]["enabled"] = False

    # Should return immediately without doing anything
    await monitor._health_check_loop()


@pytest.mark.asyncio
async def test_health_check_loop_enabled(monkeypatch):
    """Test health check loop when enabled."""
    monitor = pool_ws_monitor.EnhancedPoolMonitor("test_key", "test_pool")

    # Enable health check with short interval
    monitor.config["health_check"]["enabled"] = True
    monitor.config["health_check"]["interval"] = 0.1

    monitor.is_connected = True

    # Mock sleep to control timing
    sleep_calls = []
    async def mock_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 2:  # Stop after 2 iterations
            raise asyncio.CancelledError()

    monkeypatch.setattr('asyncio.sleep', mock_sleep)

    with pytest.raises(asyncio.CancelledError):
        await monitor._health_check_loop()

    assert len(sleep_calls) >= 1


def test_enhanced_monitor_backwards_compatibility():
    """Test that the backwards compatible function still works."""
    # The watch_pool function should still exist and work
    assert hasattr(pool_ws_monitor, 'watch_pool')
    assert callable(pool_ws_monitor.watch_pool)
