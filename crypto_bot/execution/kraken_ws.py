import json
import threading
import os
from typing import Optional, Callable, Union, List, Any, Dict
from datetime import datetime, timedelta, timezone

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace()
from websocket import WebSocketApp
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from pathlib import Path
import asyncio


logger = setup_logger(__name__, LOG_DIR / "execution.log")

PUBLIC_URL = "wss://ws.kraken.com/v2"
PRIVATE_URL = "wss://ws-auth.kraken.com/v2"


def parse_ohlc_message(message: str) -> Optional[List[float]]:
    """Parse a Kraken OHLC websocket message.

    Parameters
    ----------
    message : str
        Raw JSON message from the websocket.

    Returns
    -------
    Optional[List[float]]
        ``[timestamp, open, high, low, close, volume]`` if parsable.
    """
    try:
        data: Any = json.loads(message)
    except json.JSONDecodeError:
        return None

    # object based format
    if isinstance(data, dict):
        if data.get("channel") == "ohlc" and data.get("type") in {"snapshot", "update"}:
            arr = data.get("data")
            if isinstance(arr, list) and arr:
                candle = arr[0]
                if isinstance(candle, dict):
                    ts_val = candle.get("interval_begin")
                    if isinstance(ts_val, str):
                        try:
                            ts = int(datetime.fromisoformat(ts_val.replace("Z", "+00:00")).timestamp() * 1000)
                        except Exception:
                            ts = None
                    else:
                        ts = None
                    try:
                        o = float(candle.get("open"))
                        h = float(candle.get("high"))
                        l = float(candle.get("low"))
                        c = float(candle.get("close"))
                        vol = float(candle.get("volume"))
                    except (TypeError, ValueError):
                        return None
                    if ts is not None:
                        return [ts, o, h, l, c, vol]

    # list based format
    if not isinstance(data, list) or len(data) < 3:
        return None

    chan = data[1] if len(data) > 1 else {}
    candle = data[2] if len(data) > 2 else None
    if not isinstance(chan, dict) or not isinstance(candle, list):
        return None
    if not str(chan.get("channel", "")).startswith("ohlc"):
        return None
    try:
        ts = int(float(candle[0]) * 1000)
        o, h, l, c = map(float, candle[1:5])
        vol = float(candle[6])
    except (IndexError, ValueError, TypeError):
        return None
    return [ts, o, h, l, c, vol]



def parse_instrument_message(message: str) -> Optional[dict]:
    """Parse a Kraken instrument snapshot or update message.

    Parameters
    ----------
    message : str
        Raw JSON message from the websocket.

    Returns
    -------
    Optional[dict]
        The ``data`` payload containing ``assets`` and ``pairs`` if the
        message is a valid instrument snapshot or update.
    """
    try:
        data: Any = json.loads(message)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("channel") != "instrument":
        return None

    payload = data.get("data")
    if not isinstance(payload, dict):
        return None
    return payload


def parse_book_message(message: str) -> Optional[dict]:
    """Parse a Kraken order book snapshot or update message.

    Parameters
    ----------
    message : str
        Raw JSON message from the websocket.

    Returns
    -------
    Optional[dict]
        Dictionary with ``bids`` and ``asks`` lists containing ``[price, volume]``
        floats and a ``type`` field if the message is valid.
    """

    try:
        data: Any = json.loads(message)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or data.get("channel") != "book":
        return None

    msg_type = data.get("type")
    payload = data.get("data")
    if msg_type not in ("snapshot", "update") or not isinstance(payload, dict):
        return None

    bids = payload.get("bids")
    asks = payload.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list):
        return None

    def _convert(items: List[Any]) -> List[List[float]]:
        parsed = []
        for it in items:
            try:
                price = float(it[0])
                volume = float(it[1])
            except (IndexError, ValueError, TypeError):
                continue
            parsed.append([price, volume])
        return parsed

    return {"type": msg_type, "bids": _convert(bids), "asks": _convert(asks)}


def parse_trade_message(message: str) -> Optional[List[dict]]:
    """Parse a Kraken trade snapshot or update message.

    Parameters
    ----------
    message : str
        Raw JSON message from the websocket.

    Returns
    -------
    Optional[List[dict]]
        List of trade dictionaries with fields ``symbol``,
        ``side``, ``qty``, ``price``, ``ord_type``, ``trade_id`` and
        ``timestamp_ms`` if the message is valid.
    """

    try:
        data: Any = json.loads(message)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or data.get("channel") != "trade":
        return None

    trades = data.get("data")
    if not isinstance(trades, list):
        return None

    result: List[dict] = []
    for t in trades:
        if not isinstance(t, dict):
            return None

        symbol = t.get("symbol") or data.get("symbol")
        side = t.get("side")
        qty = t.get("qty")
        price = t.get("price")
        ord_type = t.get("ord_type")
        trade_id = t.get("trade_id")
        ts = t.get("timestamp")

        if not isinstance(symbol, str):
            return None
        try:
            qty = float(qty)
            price = float(price)
        except (TypeError, ValueError):
            return None

        timestamp_ms = None
        if isinstance(ts, str):
            try:
                if ts.endswith("Z"):
                    ts_obj = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    ts_obj = datetime.fromisoformat(ts)
                timestamp_ms = int(ts_obj.timestamp() * 1000)
            except Exception:
                return None
        else:
            return None

        result.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "ord_type": ord_type,
                "trade_id": trade_id,
                "timestamp_ms": timestamp_ms,
            }
        )

    return result


def _parse_l3_orders(levels: List[Any]) -> Optional[List[dict]]:
    """Return parsed list of L3 book orders or ``None`` on error."""

    result = []
    for lvl in levels:
        if not isinstance(lvl, dict):
            return None
        try:
            price = float(lvl.get("limit_price"))
            qty = float(lvl.get("order_qty"))
            order_id = str(lvl.get("order_id"))
        except (TypeError, ValueError):
            return None
        result.append(
            {
                "order_id": order_id,
                "limit_price": price,
                "order_qty": qty,
            }
        )
    return result


def parse_level3_snapshot(msg: str) -> Optional[dict]:
    """Parse a Kraken level 3 order book snapshot message."""

    try:
        data: Any = json.loads(msg)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("channel") != "level3" or data.get("type") != "snapshot":
        return None

    payload_list = data.get("data")
    if not isinstance(payload_list, list) or not payload_list:
        return None
    payload = payload_list[0]
    if not isinstance(payload, dict):
        return None
    symbol = payload.get("symbol")
    if not isinstance(symbol, str):
        return None

    bids_raw = payload.get("bids")
    asks_raw = payload.get("asks")
    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        return None

    bids = _parse_l3_orders(bids_raw)
    asks = _parse_l3_orders(asks_raw)
    if bids is None or asks is None:
        return None

    checksum = payload.get("checksum")
    try:
        checksum = int(checksum)
    except (TypeError, ValueError):
        return None

    ts_val = payload.get("timestamp")
    timestamp = None
    if isinstance(ts_val, str):
        try:
            timestamp = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except Exception:
            timestamp = None

    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "checksum": checksum,
        "timestamp": timestamp,
    }


def _parse_l3_events(levels: List[Any]) -> Optional[List[dict]]:
    """Parse level 3 order book update events."""

    result = []
    for lvl in levels:
        if not isinstance(lvl, dict):
            return None
        event = lvl.get("event")
        order_id = lvl.get("order_id")
        try:
            price = float(lvl.get("limit_price"))
            qty = float(lvl.get("order_qty"))
        except (TypeError, ValueError):
            return None
        ts_val = lvl.get("timestamp")
        ts = None
        if isinstance(ts_val, str):
            try:
                ts = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            except Exception:
                ts = None
        result.append(
            {
                "event": str(event),
                "order_id": str(order_id),
                "limit_price": price,
                "order_qty": qty,
                "timestamp": ts,
            }
        )
    return result


def parse_level3_update(msg: str) -> Optional[dict]:
    """Parse a Kraken level 3 order book update message."""

    try:
        data: Any = json.loads(msg)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("channel") != "level3" or data.get("type") != "update":
        return None

    payload_list = data.get("data")
    if not isinstance(payload_list, list) or not payload_list:
        return None
    payload = payload_list[0]
    if not isinstance(payload, dict):
        return None
    symbol = payload.get("symbol")
    if not isinstance(symbol, str):
        return None

    bids_raw = payload.get("bids")
    asks_raw = payload.get("asks")
    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        return None

    bids = _parse_l3_events(bids_raw)
    asks = _parse_l3_events(asks_raw)
    if bids is None or asks is None:
        return None

    checksum = payload.get("checksum")
    try:
        checksum = int(checksum)
    except (TypeError, ValueError):
        return None

    ts_val = payload.get("timestamp")
    timestamp = None
    if isinstance(ts_val, str):
        try:
            timestamp = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except Exception:
            timestamp = None

    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "checksum": checksum,
        "timestamp": timestamp,
    }


class KrakenWSClient:
    """Minimal Kraken WebSocket client for public and private channels."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        ws_token: Optional[str] = None,
        api_token: Optional[str] = None,
    ):
        # First try direct parameters
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws_token = ws_token
        self.api_token = api_token

        # If not provided, try environment variables
        if not self.api_key:
            self.api_key = os.getenv("API_KEY")
        if not self.api_secret:
            self.api_secret = os.getenv("API_SECRET")
        if not self.ws_token:
            self.ws_token = os.getenv("KRAKEN_WS_TOKEN")
        if not self.api_token:
            self.api_token = os.getenv("KRAKEN_API_TOKEN")

        # If still not found, try loading from .env file directly
        if not self.api_key or not self.api_secret:
            try:
                from dotenv import dotenv_values
                from pathlib import Path
                import sys

                # Try multiple possible .env file locations
                env_paths = [
                    Path.cwd() / ".env",  # Project root
                    Path(__file__).resolve().parent.parent.parent / ".env",  # crypto_bot/.env
                    Path(sys.path[0]) / ".env" if sys.path[0] else None
                ]

                for env_path in env_paths:
                    if env_path and env_path.exists():
                        secrets = dotenv_values(str(env_path))
                        if not self.api_key and secrets.get("API_KEY"):
                            self.api_key = secrets["API_KEY"]
                        if not self.api_secret and secrets.get("API_SECRET"):
                            self.api_secret = secrets["API_SECRET"]
                        if not self.ws_token and secrets.get("KRAKEN_WS_TOKEN"):
                            self.ws_token = secrets["KRAKEN_WS_TOKEN"]
                        if not self.api_token and secrets.get("KRAKEN_API_TOKEN"):
                            self.api_token = secrets["KRAKEN_API_TOKEN"]
                        break
            except Exception:
                # Silently ignore .env loading errors
                pass

        self.exchange = None
        if self.api_key and self.api_secret:
            # Use get_exchange to get nonce improvements
            from .cex_executor import get_exchange
            config = {
                "exchange": "kraken",
                "enable_nonce_improvements": True,
                "api_retry_attempts": 3
            }
            # Temporarily set environment variables for get_exchange to use
            old_api_key = os.environ.get('API_KEY')
            old_api_secret = os.environ.get('API_SECRET')
            old_ws_token = os.environ.get('KRAKEN_WS_TOKEN')
            old_api_token = os.environ.get('KRAKEN_API_TOKEN')

            try:
                os.environ['API_KEY'] = self.api_key
                os.environ['API_SECRET'] = self.api_secret
                if self.ws_token:
                    os.environ['KRAKEN_WS_TOKEN'] = self.ws_token
                if self.api_token:
                    os.environ['KRAKEN_API_TOKEN'] = self.api_token

                self.exchange, _ = get_exchange(config)
            finally:
                # Restore original environment variables
                if old_api_key is not None:
                    os.environ['API_KEY'] = old_api_key
                elif 'API_KEY' in os.environ:
                    del os.environ['API_KEY']

                if old_api_secret is not None:
                    os.environ['API_SECRET'] = old_api_secret
                elif 'API_SECRET' in os.environ:
                    del os.environ['API_SECRET']

                if old_ws_token is not None:
                    os.environ['KRAKEN_WS_TOKEN'] = old_ws_token
                elif 'KRAKEN_WS_TOKEN' in os.environ:
                    del os.environ['KRAKEN_WS_TOKEN']

                if old_api_token is not None:
                    os.environ['KRAKEN_API_TOKEN'] = old_api_token
                elif 'KRAKEN_API_TOKEN' in os.environ:
                    del os.environ['KRAKEN_API_TOKEN']

        self.token: Optional[str] = self.ws_token
        self.token_created: Optional[datetime] = None
        if self.token:
            self.token_created = datetime.now(timezone.utc)

        # Connection tracking
        self.start_time = datetime.now(timezone.utc)
        self.public_ws: Optional[WebSocketApp] = None
        self.private_ws: Optional[WebSocketApp] = None
        self.last_public_heartbeat: Optional[datetime] = None
        self.last_private_heartbeat: Optional[datetime] = None
        self._public_subs = []
        self._private_subs = []
        
        # Real-time price data storage
        self.price_cache: Dict[str, Dict] = {}
        self.price_callbacks: Dict[str, List[Callable]] = {}

        # Health check and monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.connection_health = {
            "public": {"last_heartbeat": None, "is_alive": False, "errors": 0},
            "private": {"last_heartbeat": None, "is_alive": False, "errors": 0}
        }
        self.message_stats = {
            "total_received": 0,
            "total_sent": 0,
            "errors": 0,
            "reconnections": 0
        }

    @staticmethod
    def _to_ccxt_symbol(symbol: Optional[str]) -> Optional[str]:
        """Normalize Kraken symbols to CCXT format."""

        if not isinstance(symbol, str):
            return None

        symbol = symbol.strip()
        if not symbol:
            return None

        if "/" in symbol:
            base, quote = symbol.split("/", 1)
            if base.upper() == "XBT":
                base = "BTC"
            return f"{base}/{quote}"

        if symbol.upper() == "XBT":
            return "BTC"

        return symbol

    def add_price_callback(self, symbol: str, callback: Callable[[str, float], None]) -> None:
        """Add a callback function to be called when price updates for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., "XBT/USD")
        callback : Callable
            Function to call with (symbol, price) when price updates
        """
        if symbol not in self.price_callbacks:
            self.price_callbacks[symbol] = []
        self.price_callbacks[symbol].append(callback)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol from WebSocket cache.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol. Kraken-style bases like "XBT/USD" are
            normalized to CCXT format ("BTC/USD").
        
        Returns
        -------
        Optional[float]
            Current price if available, None otherwise
        """
        normalized_symbol = self._to_ccxt_symbol(symbol)
        if not normalized_symbol:
            return None
        return self.price_cache.get(normalized_symbol, {}).get('last')

    def _handle_message(self, ws: WebSocketApp, message: str) -> None:
        """Default ``on_message`` handler that records heartbeats and processes ticker data."""
        logger.debug("WS message: %s", message)
        self.message_stats["total_received"] += 1

        # Validate message before processing
        if not self._validate_message(message):
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON message: {e}")
            self.message_stats["errors"] += 1
            return

        def update(obj: Any) -> None:
            if isinstance(obj, dict):
                # Handle heartbeat
                if obj.get("channel") == "heartbeat":
                    now = datetime.now(timezone.utc)
                    conn_type = "private" if ws == self.private_ws else "public"
                    self.connection_health[conn_type]["last_heartbeat"] = now
                    self.connection_health[conn_type]["is_alive"] = True

                    # Update legacy heartbeat tracking for backwards compatibility
                    if ws == self.private_ws:
                        self.last_private_heartbeat = now
                    else:
                        self.last_public_heartbeat = now
                
                # Handle ticker data
                elif obj.get("channel") == "ticker" and obj.get("type") in {"snapshot", "update"}:
                    ticker_data = obj.get("data", [])
                    if isinstance(ticker_data, list) and ticker_data:
                        ticker = ticker_data[0]
                        if isinstance(ticker, dict):
                            symbol = ticker.get("symbol")
                            if symbol:
                                # Convert Kraken format back to CCXT format
                                ccxt_symbol = self._to_ccxt_symbol(symbol)
                                if not ccxt_symbol:
                                    return
                                
                                # Extract price data
                                price_data = {
                                    'last': float(ticker.get("last", 0)),
                                    'bid': float(ticker.get("bid", 0)),
                                    'ask': float(ticker.get("ask", 0)),
                                    'volume': float(ticker.get("volume", 0)),
                                    'timestamp': datetime.now(timezone.utc).timestamp()
                                }
                                
                                # Update price cache
                                self.price_cache[ccxt_symbol] = price_data
                                
                                # Call registered callbacks
                                if ccxt_symbol in self.price_callbacks:
                                    for callback in self.price_callbacks[ccxt_symbol]:
                                        try:
                                            callback(ccxt_symbol, price_data['last'])
                                        except Exception as e:
                                            logger.error(f"Error in price callback for {ccxt_symbol}: {e}")

        if isinstance(data, list):
            for item in data:
                update(item)
        else:
            update(data)

    def _regenerate_private_subs(self) -> None:
        """Update stored private subscription messages with the current token."""
        updated = []
        for sub in self._private_subs:
            try:
                msg = json.loads(sub)
                if "params" in msg:
                    msg["params"]["token"] = self.token
                updated.append(json.dumps(msg))
            except Exception:
                updated.append(sub)
        self._private_subs = updated

    def get_token(self) -> str:
        """Retrieve WebSocket authentication token via Kraken REST API."""
        if self.token:
            return self.token

        if not self.exchange:
            raise ValueError("API keys required for private websocket")

        params = {}
        if self.api_token:
            params["otp"] = self.api_token

        resp = self.exchange.privatePostGetWebSocketsToken(params)
        self.token = resp["token"]
        self.token_created = datetime.now(timezone.utc)
        return self.token

    def _start_ws(
        self,
        url: str,
        conn_type: Optional[str] = None,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        *,
        ping_interval: int = 20,
        ping_timeout: int = 10,
        **kwargs,
    ) -> WebSocketApp:
        """Start a ``WebSocketApp`` and begin the reader thread."""

        def default_on_message(ws, message):
            self._handle_message(ws, message)

        def default_on_error(ws, error):
            logger.error("WS error: %s", error)

        def default_on_close(ws, close_status_code, close_msg):
            logger.info("WS closed: %s %s", close_status_code, close_msg)

        on_message = on_message or default_on_message
        on_error = on_error or default_on_error

        def _on_close(ws, close_status_code, close_msg):
            if on_close:
                on_close(ws, close_status_code, close_msg)
            else:
                default_on_close(ws, close_status_code, close_msg)
            if conn_type:
                self.on_close(conn_type)

        ws = WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=_on_close,
            **kwargs,
        )
        thread = threading.Thread(
            target=lambda: ws.run_forever(
                ping_interval=ping_interval, ping_timeout=ping_timeout
            ),
            daemon=True,
        )
        thread.start()
        return ws

    def token_expired(self) -> bool:
        """Return True if the authentication token is older than 14 minutes."""
        if not self.token_created:
            return False
        return datetime.now(timezone.utc) - self.token_created > timedelta(minutes=14)

    def _handle_connection_error(self, conn_type: str, error: Exception):
        """Enhanced error handling with specific recovery strategies."""
        logger.error(f"WebSocket {conn_type} error: {error}")
        self.connection_health[conn_type]["errors"] += 1
        self.message_stats["errors"] += 1

        # Different strategies for different error types
        if "timeout" in str(error).lower():
            self._handle_timeout_error(conn_type)
        elif "rate limit" in str(error).lower():
            self._handle_rate_limit_error(conn_type)
        else:
            self._handle_generic_error(conn_type)

    def _handle_timeout_error(self, conn_type: str):
        """Handle timeout errors with exponential backoff."""
        logger.warning(f"Timeout error on {conn_type} connection, marking as unhealthy")
        self.connection_health[conn_type]["is_alive"] = False

    def _handle_rate_limit_error(self, conn_type: str):
        """Handle rate limiting errors."""
        logger.warning(f"Rate limit error on {conn_type} connection, implementing backoff")
        self.connection_health[conn_type]["is_alive"] = False

    def _handle_generic_error(self, conn_type: str):
        """Handle generic connection errors."""
        logger.warning(f"Generic error on {conn_type} connection")
        self.connection_health[conn_type]["is_alive"] = False

    def _validate_message(self, message: str, max_size: int = 1048576) -> bool:
        """Validate incoming WebSocket message."""
        try:
            # Check message size limits
            if len(message) > max_size:
                logger.warning(f"Message too large: {len(message)} bytes")
                return False

            # Validate JSON structure
            json.loads(message)
            return True
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received")
            self.message_stats["errors"] += 1
            return False
        except Exception as e:
            logger.warning(f"Message validation error: {e}")
            self.message_stats["errors"] += 1
            return False

    async def health_check_loop(self, interval: int = 30):
        """Periodic health check for websocket connections."""
        logger.info("Starting WebSocket health check loop")

        while True:
            try:
                await asyncio.sleep(interval)

                now = datetime.now(timezone.utc)

                # Check public connection
                if self.public_ws:
                    last_heartbeat = self.connection_health["public"]["last_heartbeat"]
                    if last_heartbeat and (now - last_heartbeat) > timedelta(seconds=60):
                        logger.warning("Public WebSocket connection appears stale, attempting reconnect")
                        self.connection_health["public"]["is_alive"] = False
                        self.connect_public()
                    elif not self.connection_health["public"]["is_alive"]:
                        logger.info("Attempting to restore public WebSocket connection")
                        self.connect_public()

                # Check private connection
                if self.private_ws:
                    last_heartbeat = self.connection_health["private"]["last_heartbeat"]
                    if last_heartbeat and (now - last_heartbeat) > timedelta(seconds=60):
                        logger.warning("Private WebSocket connection appears stale, attempting reconnect")
                        self.connection_health["private"]["is_alive"] = False
                        self.connect_private()
                    elif not self.connection_health["private"]["is_alive"]:
                        logger.info("Attempting to restore private WebSocket connection")
                        self.connect_private()

                # Log health statistics
                self._log_health_stats()

            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    def _log_health_stats(self):
        """Log connection health statistics."""
        total_errors = sum(conn["errors"] for conn in self.connection_health.values())

        logger.debug(
            f"WebSocket health - Public: {self.connection_health['public']['is_alive']}, "
            f"Private: {self.connection_health['private']['is_alive']}, "
            f"Total errors: {total_errors}, "
            f"Messages: {self.message_stats['total_received']}"
        )

    def get_connection_health(self) -> Dict[str, Any]:
        """Get detailed connection health information."""
        return {
            "connections": self.connection_health.copy(),
            "message_stats": self.message_stats.copy(),
            "uptime": str(datetime.now() - self.start_time) if hasattr(self, 'start_time') else None
        }


    def connect_public(self) -> None:
        if not self.public_ws:
            self.public_ws = self._start_ws(PUBLIC_URL, conn_type="public")

    def connect_private(self) -> None:
        prev_token = self.token
        if self.token_expired():
            self.token = None
        if not self.token:
            self.get_token()
        token_changed = self.token != prev_token
        if not self.private_ws:
            self.private_ws = self._start_ws(PRIVATE_URL, conn_type="private")
            token_changed = True
        if token_changed:
            self._regenerate_private_subs()
            for sub in self._private_subs:
                self.private_ws.send(sub)

    def subscribe_ticker(
        self,
        symbol: Union[str, List[str]],
        *,
        event_trigger: Optional[str] = None,
        snapshot: Optional[bool] = None,
        req_id: Optional[int] = None,
    ) -> None:
        """Subscribe to ticker updates for one or more symbols."""

        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        params = {"channel": "ticker", "symbol": symbol}
        if event_trigger is not None:
            params["eventTrigger"] = event_trigger
        if snapshot is not None:
            params["snapshot"] = snapshot
        if req_id is not None:
            params["req_id"] = req_id

        msg = {"method": "subscribe", "params": params}
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def unsubscribe_ticker(
        self,
        symbol: Union[str, List[str]],
        *,
        event_trigger: Optional[str] = None,
        req_id: Optional[int] = None,
    ) -> None:
        """Unsubscribe from ticker updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "unsubscribe",
            "params": {"channel": "ticker", "symbol": symbol},
        }
        if event_trigger is not None:
            msg["params"]["eventTrigger"] = event_trigger
        if req_id is not None:
            msg["req_id"] = req_id
        data = json.dumps(msg)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "ticker"
                and sorted(params.get("symbol", [])) == sorted(symbol)
                and (
                    event_trigger is None
                    or params.get("event_trigger") == event_trigger
                )
                and (req_id is None or params.get("req_id") == req_id)
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]
        self.public_ws.send(data)

    def subscribe_trades(
        self, symbol: Union[str, List[str]], *, snapshot: bool = True
    ) -> None:
        """Subscribe to trade updates for one or more symbols."""

        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        params = {"channel": "trade", "symbol": symbol}
        if snapshot is not True:
            params["snapshot"] = snapshot
        msg = {"method": "subscribe", "params": params}
        data = json.dumps(msg)
        self._public_subs.append(data)
        if hasattr(self.public_ws, "sent"):
            try:
                self.public_ws.sent.clear()
            except Exception:
                pass
        self.public_ws.send(data)

    def unsubscribe_trades(self, symbol: Union[str, List[str]]) -> None:
        """Unsubscribe from trade updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "unsubscribe",
            "params": {"channel": "trade", "symbol": symbol},
        }
        data = json.dumps(msg)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "trade"
                and sorted(params.get("symbol", [])) == sorted(symbol)
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]
        self.public_ws.send(data)

    def subscribe_ohlc(
        self,
        symbol: Union[str, List[str]],
        interval: int,
        *,
        snapshot: bool = True,
        req_id: Optional[int] = None,
    ) -> None:
        """Subscribe to OHLC updates for one or more symbols."""

        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]

        params = {
            "channel": "ohlc",
            "symbol": symbol,
            "interval": interval,
            "snapshot": snapshot,
        }
        if req_id is not None:
            params["req_id"] = req_id

        msg = {"method": "subscribe", "params": params}
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def subscribe_book(
        self,
        symbol: Union[str, List[str]],
        *,
        depth: int = 10,
        snapshot: bool = True,
    ) -> None:
        """Subscribe to order book updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": symbol,
                "depth": depth,
                "snapshot": snapshot,
            },
        }
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def subscribe_instruments(self, snapshot: bool = True) -> None:
        """Subscribe to the instrument reference data channel."""
        self.connect_public()
        if hasattr(self.public_ws, "sent"):
            try:
                self.public_ws.sent.clear()
            except Exception:
                pass
        msg = {
            "method": "subscribe",
            "params": {"channel": "instrument", "snapshot": snapshot},
        }
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def unsubscribe_instruments(self) -> None:
        """Unsubscribe from the instrument reference data channel."""
        self.connect_public()
        msg = {"method": "unsubscribe", "params": {"channel": "instrument"}}
        data = json.dumps(msg)
        self.public_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "instrument"
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]

    def unsubscribe_book(self, symbol: Union[str, List[str]], depth: int = 10) -> None:
        """Unsubscribe from order book updates for the given symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        params = {"channel": "book", "symbol": symbol}
        if depth != 10:
            params["depth"] = depth
        msg = {"method": "unsubscribe", "params": params}
        data = json.dumps(msg)
        self.public_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "book"
                and params.get("depth", depth) == depth
                and sorted(params.get("symbol", [])) == sorted(symbol)
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]

    def unsubscribe_ohlc(
        self,
        symbol: Union[str, List[str]],
        interval: int,
        *,
        req_id: Optional[int] = None,
    ) -> None:
        """Unsubscribe from OHLC updates for the given symbols."""

        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]

        params = {
            "channel": "ohlc",
            "symbol": symbol,
            "interval": interval,
        }
        if req_id is not None:
            params["req_id"] = req_id

        msg = {"method": "unsubscribe", "params": params}
        data = json.dumps(msg)
        self.public_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "ohlc"
                and params.get("interval") == interval
                and sorted(params.get("symbol", [])) == sorted(symbol)
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]

    def subscribe_orders(self, symbol: Optional[str] = None) -> None:
        """Subscribe to private open order updates.

        If ``symbol`` is provided the channel name uses ``openOrders`` and the
        symbol list, matching Kraken's subscription format used in the tests.
        Otherwise the older ``open_orders`` channel is used.
        """
        self.connect_private()
        channel = "openOrders" if symbol is not None else "open_orders"
        msg = {
            "method": "subscribe",
            "params": {"channel": channel, "token": self.token},
        }
        data = json.dumps(msg)
        self._private_subs.append(data)
        self.private_ws.send(data)

    def unsubscribe_orders(self, symbol: Optional[str] = None) -> None:
        """Unsubscribe from private open order updates."""

        self.connect_private()
        channel = "openOrders" if symbol is not None else "open_orders"
        msg = {
            "method": "unsubscribe",
            "params": {"channel": channel, "token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == channel
            )

        self._private_subs = [s for s in self._private_subs if not _matches(s)]

    def subscribe_executions(self) -> None:
        """Subscribe to private execution (trade fill) updates."""

        self.connect_private()
        msg = {
            "method": "subscribe",
            "params": {"channel": "executions", "token": self.token},
        }
        data = json.dumps(msg)
        self._private_subs.append(data)
        self.private_ws.send(data)

    def unsubscribe_executions(self) -> None:
        """Unsubscribe from private execution updates."""

        self.connect_private()
        msg = {
            "method": "unsubscribe",
            "params": {"channel": "executions", "token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "executions"
            )

        self._private_subs = [s for s in self._private_subs if not _matches(s)]

    def subscribe_level3(
        self,
        symbol: Union[str, List[str]],
        *,
        depth: int = 10,
        snapshot: bool = True,
    ) -> None:
        """Subscribe to authenticated level3 order book updates."""

        self.connect_private()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "subscribe",
            "params": {
                "channel": "level3",
                "symbol": symbol,
                "depth": depth,
                "snapshot": snapshot,
                "token": self.token,
            },
        }
        data = json.dumps(msg)
        self._private_subs.append(data)
        self.private_ws.send(data)

    def unsubscribe_level3(self, symbol: Union[str, List[str]], depth: int = 10) -> None:
        """Unsubscribe from authenticated level3 order book updates."""

        self.connect_private()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "unsubscribe",
            "params": {
                "channel": "level3",
                "symbol": symbol,
                "depth": depth,
                "token": self.token,
            },
        }
        data = json.dumps(msg)
        self.private_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "level3"
                and params.get("depth", depth) == depth
                and sorted(params.get("symbol", [])) == sorted(symbol)
            )

        self._private_subs = [s for s in self._private_subs if not _matches(s)]

    def add_order(
        self,
        symbol: Union[str, List[str]],
        side: str,
        order_qty: float,
        order_type: str = "market",
        *,
        limit_price: Optional[float] = None,
        limit_price_type: Optional[str] = None,
        triggers: Optional[Dict[str, Any]] = None,
        time_in_force: Optional[str] = None,
        margin: Optional[bool] = None,
        post_only: Optional[bool] = None,
        reduce_only: Optional[bool] = None,
        effective_time: Optional[str] = None,
        expire_time: Optional[str] = None,
        deadline: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
        order_userref: Optional[int] = None,
        conditional: Optional[Dict[str, Any]] = None,
        display_qty: Optional[float] = None,
        fee_preference: Optional[str] = None,
        no_mpp: Optional[bool] = None,
        stp_type: Optional[str] = None,
        cash_order_qty: Optional[float] = None,
        validate: Optional[bool] = None,
        sender_sub_id: Optional[str] = None,
        req_id: Optional[int] = None,
    ) -> dict:
        """Send an add_order request via the private websocket.

        Parameters
        ----------
        symbol : Union[str, List[str]]
            Trading pair symbol (e.g., "XBT/USD")
        side : str
            Order side ("buy" or "sell")
        order_qty : float
            Order quantity in base asset
        order_type : str, default "market"
            Order type: "market", "limit", "stop-loss", "take-profit", etc.
        limit_price : Optional[float]
            Limit price for limit orders
        limit_price_type : Optional[str]
            Units for limit price ("static", "pct", "quote")
        triggers : Optional[Dict[str, Any]]
            Trigger parameters for stop-loss/take-profit orders
        time_in_force : Optional[str]
            Time-in-force ("gtc", "gtd", "ioc")
        margin : Optional[bool]
            Enable margin funding
        post_only : Optional[bool]
            Only post if it adds liquidity
        reduce_only : Optional[bool]
            Reduce existing position only
        effective_time : Optional[str]
            Scheduled start time (RFC3339)
        expire_time : Optional[str]
            Expiration time for GTD orders (RFC3339)
        deadline : Optional[str]
            Max lifetime before matching (RFC3339)
        cl_ord_id : Optional[str]
            Client order ID
        order_userref : Optional[int]
            User reference number
        conditional : Optional[Dict[str, Any]]
            Parameters for OTO orders
        display_qty : Optional[float]
            Display quantity for iceberg orders
        fee_preference : Optional[str]
            Fee preference ("base" or "quote")
        no_mpp : Optional[bool]
            Disable Market Price Protection
        stp_type : Optional[str]
            Self-trade prevention type
        cash_order_qty : Optional[float]
            Quote currency volume for buy market orders
        validate : Optional[bool]
            Validate only without trading
        sender_sub_id : Optional[str]
            Sub-account identifier
        req_id : Optional[int]
            Request ID for tracking

        Returns
        -------
        dict
            The WebSocket message sent
        """
        self.connect_private()
        if isinstance(symbol, list):
            symbol = symbol[0] if symbol else ""

        params = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "order_qty": order_qty,  # Keep as float, not string
            "token": self.token,
        }

        # Add optional parameters if provided
        optional_params = {
            "limit_price": limit_price,
            "limit_price_type": limit_price_type,
            "triggers": triggers,
            "time_in_force": time_in_force,
            "margin": margin,
            "post_only": post_only,
            "reduce_only": reduce_only,
            "effective_time": effective_time,
            "expire_time": expire_time,
            "deadline": deadline,
            "cl_ord_id": cl_ord_id,
            "order_userref": order_userref,
            "conditional": conditional,
            "display_qty": display_qty,
            "fee_preference": fee_preference,
            "no_mpp": no_mpp,
            "stp_type": stp_type,
            "cash_order_qty": cash_order_qty,
            "validate": validate,
            "sender_sub_id": sender_sub_id,
        }

        # Only include non-None values
        params.update({k: v for k, v in optional_params.items() if v is not None})

        msg = {"method": "add_order", "params": params}
        if req_id is not None:
            msg["req_id"] = req_id

        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def edit_order(
        self,
        order_id: str,
        symbol: str,
        order_qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        deadline: Optional[str] = None,
    ) -> dict:
        """Send an edit_order request via the private websocket."""
        self.connect_private()
        params = {"order_id": order_id, "symbol": symbol, "token": self.token}
        if order_qty is not None:
            params["order_qty"] = order_qty
        if limit_price is not None:
            params["limit_price"] = limit_price
        if deadline is not None:
            params["deadline"] = deadline
        msg = {"method": "edit_order", "params": params}
        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def cancel_order(self, txid: str) -> dict:
        self.connect_private()
        msg = {
            "method": "cancel_order",
            "params": {"txid": txid, "token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def cancel_all_orders(self) -> dict:
        self.connect_private()
        msg = {
            "method": "cancel_all_orders",
            "params": {"token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def open_orders(self) -> dict:
        self.connect_private()
        msg = {
            "method": "open_orders",
            "params": {"token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def amend_order(
        self,
        *,
        order_id: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
        order_qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        display_qty: Optional[float] = None,
        limit_price_type: Optional[str] = None,
        post_only: Optional[bool] = None,
        trigger_price: Optional[float] = None,
        trigger_price_type: Optional[str] = None,
        deadline: Optional[str] = None,
        req_id: Optional[int] = None,
    ) -> dict:
        """Send an ``amend_order`` request via the private websocket."""

        self.connect_private()

        params: Dict[str, Any] = {"token": self.token}
        if order_id is not None:
            params["order_id"] = order_id
        if cl_ord_id is not None:
            params["cl_ord_id"] = cl_ord_id
        if order_qty is not None:
            params["order_qty"] = order_qty
        if limit_price is not None:
            params["limit_price"] = limit_price
        if display_qty is not None:
            params["display_qty"] = display_qty
        if limit_price_type is not None:
            params["limit_price_type"] = limit_price_type
        if post_only is not None:
            params["post_only"] = post_only
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if trigger_price_type is not None:
            params["trigger_price_type"] = trigger_price_type
        if deadline is not None:
            params["deadline"] = deadline

        msg: Dict[str, Any] = {"method": "amend_order", "params": params}
        if req_id is not None:
            msg["req_id"] = req_id

        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def subscribe_ticker(self, symbol: str) -> dict:
        """Subscribe to ticker data for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., "XBT/USD")
        
        Returns
        -------
        dict
            Subscription message sent to WebSocket
        """
        # Convert symbol format from CCXT to Kraken format
        kraken_symbol = symbol.replace("/", "")
        
        msg = {
            "method": "subscribe",
            "params": {
                "channel": "ticker",
                "symbol": [kraken_symbol]
            }
        }
        
        data = json.dumps(msg)
        
        # Ensure public WebSocket is connected
        if not self.public_ws:
            self.connect_public()
            # Wait a moment for connection to establish
            import time
            time.sleep(0.5)
        
        # Check if connection is ready
        if self.public_ws and self.public_ws.sock and self.public_ws.sock.connected:
            self.public_ws.send(data)
            self._public_subs.append(data)
            logger.info(f"Subscribed to ticker for {symbol}")
        else:
            logger.warning(f"WebSocket not ready for {symbol}, subscription queued")
            self._public_subs.append(data)
        
        return msg

    def subscribe_ohlcv(self, symbol: str, interval: int = 1) -> dict:
        """Subscribe to OHLCV data for a symbol.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., "XBT/USD")
        interval : int
            Interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        
        Returns
        -------
        dict
            Subscription message sent to WebSocket
        """
        # Convert symbol format from CCXT to Kraken format
        kraken_symbol = symbol.replace("/", "")
        
        # Map interval to Kraken format
        interval_map = {
            1: 1,      # 1m
            5: 5,      # 5m  
            15: 15,    # 15m
            30: 30,    # 30m
            60: 60,    # 1h
            240: 240,  # 4h
            1440: 1440, # 1d
            10080: 10080, # 1w
            21600: 21600  # 1M
        }
        
        kraken_interval = interval_map.get(interval, 1)
        
        msg = {
            "method": "subscribe",
            "params": {
                "channel": "ohlc",
                "symbol": [kraken_symbol],
                "interval": kraken_interval
            }
        }
        
        data = json.dumps(msg)
        
        # Ensure public WebSocket is connected
        if not self.public_ws:
            self.connect_public()
            # Wait a moment for connection to establish
            import time
            time.sleep(0.5)
        
        # Check if connection is ready
        if self.public_ws and self.public_ws.sock and self.public_ws.sock.connected:
            self.public_ws.send(data)
            self._public_subs.append(data)
            logger.info(f"Subscribed to OHLCV for {symbol} with interval {interval}")
        else:
            logger.warning(f"WebSocket not ready for {symbol}, OHLCV subscription queued")
            self._public_subs.append(data)
        
        return msg

    def ping(self, req_id: Optional[int] = None) -> dict:
        """Send a ping message to keep the websocket connection alive."""
        msg = {"method": "ping", "req_id": req_id}
        data = json.dumps(msg)
        ws = self.private_ws or self.public_ws
        if not ws:
            raise RuntimeError("WebSocket not connected")
        ws.send(data)
        return msg

    def is_alive(self, conn_type: str) -> bool:
        """Return ``True`` if the connection received a heartbeat recently."""
        now = datetime.now(timezone.utc)
        if conn_type == "private":
            last = self.last_private_heartbeat
        else:
            last = self.last_public_heartbeat
        return bool(last and (now - last) <= timedelta(seconds=10))

    def on_close(self, conn_type: str) -> None:
        """Handle WebSocket closure by reconnecting and resubscribing."""

        if conn_type == "public":
            self.public_ws = self._start_ws(PUBLIC_URL, conn_type="public")
            for sub in self._public_subs:
                self.public_ws.send(sub)
        else:
            if self.token_expired():
                self.token = None
            if not self.token:
                self.get_token()
            self._regenerate_private_subs()
            self.private_ws = self._start_ws(PRIVATE_URL, conn_type="private")
            for sub in self._private_subs:
                self.private_ws.send(sub)

    def start_health_monitoring(self, interval: int = 30):
        """Start the health check monitoring loop."""
        if self.health_check_task and not self.health_check_task.done():
            logger.warning("Health check already running")
            return

        self.health_check_task = asyncio.create_task(self.health_check_loop(interval))
        logger.info("Started WebSocket health monitoring")

    def stop_health_monitoring(self):
        """Stop the health check monitoring loop."""
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
            logger.info("Stopped WebSocket health monitoring")

    def close(self) -> None:
        """Close active WebSocket connections and clear subscriptions."""

        # Stop health monitoring
        self.stop_health_monitoring()

        if self.public_ws:
            try:
                self.public_ws.on_close = None
                self.public_ws.close()
            except Exception as exc:
                logger.error("Error closing public websocket: %s", exc)
            self.public_ws = None

        if self.private_ws:
            try:
                self.private_ws.on_close = None
                self.private_ws.close()
            except Exception as exc:
                logger.error("Error closing private websocket: %s", exc)
            self.private_ws = None

        self._public_subs = []
        self._private_subs = []

    async def close_async(self) -> None:
        """Async version of close that also closes the ccxt exchange instance."""
        # Close WebSocket connections
        self.close()
        
        # Close the ccxt exchange instance if it exists
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                if asyncio.iscoroutinefunction(getattr(self.exchange, 'close')):
                    await self.exchange.close()
                else:
                    # If it's not async, run it in a thread
                    await asyncio.to_thread(self.exchange.close)
                logger.info("Kraken exchange instance closed successfully")
            except Exception as exc:
                logger.error("Error closing Kraken exchange instance: %s", exc)
            finally:
                self.exchange = None
