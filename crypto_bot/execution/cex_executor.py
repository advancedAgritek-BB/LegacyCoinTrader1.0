from __future__ import annotations

import os
import time
try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace()
import asyncio
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# Remove CCXTpro import - user doesn't have license
ccxtpro = None

from crypto_bot.utils.telegram import TelegramNotifier, send_message
from crypto_bot.utils.notifier import Notifier
from crypto_bot.execution.kraken_ws import KrakenWSClient
from crypto_bot.utils.trade_logger import log_trade
from crypto_bot import tax_logger
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "execution.log")


# Thread-safe nonce management for Kraken
import threading
nonce_lock = threading.Lock()
last_nonce = 0


def custom_nonce():
    """Thread-safe custom nonce function for Kraken to prevent nonce errors."""
    global last_nonce
    with nonce_lock:
        current_time = int(time.time() * 1000)
        # Add a small buffer to ensure we're ahead of server time
        current_time += 100  # 100ms buffer
        # Ensure nonce is always increasing
        if current_time <= last_nonce:
            current_time = last_nonce + 1
        last_nonce = current_time
        return current_time


def setup_kraken_nonce_improvements(exchange, config):
    """Apply Kraken-specific nonce improvements to the exchange instance."""
    if exchange.id.lower() != 'kraken':
        return

    kraken_settings = config.get("kraken_settings", {})
    enable_improvements = kraken_settings.get("enable_nonce_improvements", True)

    if enable_improvements:
        # Set custom nonce function
        exchange.nonce = custom_nonce

        # Set time synchronization
        time_sync_threshold = kraken_settings.get("time_sync_threshold", 1000)
        exchange.options['timeSyncThreshold'] = time_sync_threshold
        
        # Enable automatic time synchronization
        exchange.options['adjustForTimeDifference'] = True
        
        # Try to synchronize time with Kraken server
        try:
            server_time = exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time)
            if time_diff > 1000:  # If difference is more than 1 second
                logger.warning(f"Time difference with Kraken server: {time_diff}ms")
                # Adjust the nonce buffer based on time difference
                global last_nonce
                with nonce_lock:
                    last_nonce = max(last_nonce, server_time + 100)
        except Exception as e:
            logger.warning(f"Could not synchronize time with Kraken server: {e}")

        logger.info("Applied Kraken nonce improvements: custom nonce function, time sync, and auto-adjustment")


def fetch_balance_with_retry(exchange, max_retries=3):
    """Fetch balance with retry mechanism for nonce errors."""
    for attempt in range(max_retries):
        try:
            return exchange.fetch_balance()
        except Exception as e:
            error_str = str(e)
            if "Invalid nonce" in error_str and attempt < max_retries - 1:
                logger.warning(f"Nonce error on attempt {attempt + 1}, retrying...")
                # Reset nonce for next attempt
                if hasattr(exchange, 'nonce') and callable(exchange.nonce):
                    exchange.nonce()  # Call to reset internal state
                time.sleep(2.0)  # Longer delay for nonce errors
                continue
            elif "Rate limit" in error_str and attempt < max_retries - 1:
                logger.warning(f"Rate limit on attempt {attempt + 1}, retrying...")
                time.sleep(3.0)  # Longer delay for rate limits
                continue
            else:
                # Re-raise the exception if it's not a nonce error or we've exhausted retries
                raise e
    return None


def get_exchange(config) -> Tuple[ccxt.Exchange, Optional[KrakenWSClient]]:
    """Instantiate and return a ccxt exchange and optional websocket client.

    Uses standard ``ccxt`` exchange with enhanced Kraken nonce improvements
    for better reliability. ``KrakenWSClient`` is retained for backward
    compatibility when WebSocket trading is desired.
    """

    supported_exchanges = ("coinbase", "kraken")

    # Get exchange from config, with fallback to environment variable if config doesn't specify
    exchange_cfg = config.get("exchange")
    if exchange_cfg is None:
        # If not in config, check environment variable
        env_exchange = os.getenv("EXCHANGE", "").lower()
        if env_exchange in supported_exchanges:
            exchange_cfg = env_exchange
        else:
            # Last resort default to kraken (safer than coinbase for this bot)
            exchange_cfg = "kraken"
    if isinstance(exchange_cfg, dict):
        exchange_name = exchange_cfg.get("name") or exchange_cfg.get("id") or ""
    else:
        exchange_name = str(exchange_cfg)
    exchange_name = exchange_name.lower()

    use_ws = config.get("use_websocket", False)

    ws_client: Optional[KrakenWSClient] = None
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    ws_token = os.getenv("KRAKEN_WS_TOKEN")
    api_token = os.getenv("KRAKEN_API_TOKEN")

    # Always use regular CCXT - no CCXTpro
    ccxt_mod = ccxt

    if exchange_name == "coinbase":
        exchange = ccxt_mod.coinbase(
            {
                "apiKey": os.getenv("API_KEY"),
                "secret": os.getenv("API_SECRET"),
                "password": os.getenv("API_PASSPHRASE"),
                "enableRateLimit": True,
            }
        )
        
        # Add retry method to exchange instance
        exchange.fetch_balance_with_retry = lambda: fetch_balance_with_retry(exchange)
    elif exchange_name == "kraken":
        if use_ws:
            ws_client = KrakenWSClient(api_key, api_secret, ws_token, api_token)

        exchange = ccxt_mod.kraken(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

        # Apply Kraken nonce improvements
        setup_kraken_nonce_improvements(exchange, config)
        
        # Add retry method to exchange instance
        exchange.fetch_balance_with_retry = lambda: fetch_balance_with_retry(exchange)
    else:
        raise ValueError(
            "Unsupported exchange: {}. Supported exchanges: {}".format(
                exchange_name or "unknown", ", ".join(supported_exchanges)
            )
        )

    exchange.options["ws_scan"] = config.get("use_websocket", False)

    return exchange, ws_client


def execute_trade(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    amount: float,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    notifier: Optional[TelegramNotifier] = None,
    dry_run: bool = True,
    use_websocket: bool = False,
    config: Optional[Dict] = None,
    score: float = 0.0,
) -> Dict:
    if notifier is None:
        if isinstance(token, TelegramNotifier):
            notifier = token
        else:
            if token is None or chat_id is None:
                raise ValueError("token/chat_id or notifier must be provided")
            notifier = TelegramNotifier(token, chat_id)
    if use_websocket and ws_client is None and not dry_run:
        raise ValueError("WebSocket trading enabled but ws_client is missing")
    config = config or {}

    def has_liquidity(order_size: float) -> bool:
        try:
            depth = config.get("liquidity_depth", 10)
            ob = exchange.fetch_order_book(symbol, limit=depth)
            book = ob["asks" if side == "buy" else "bids"]
            vol = 0.0
            for _, qty in book:
                vol += qty
                if vol >= order_size:
                    return True
            return False
        except Exception as err:
            err_msg = notifier.notify(f"Order book error: {err}")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            return False

    def place(size: float) -> Dict:
        if dry_run:
            return {"symbol": symbol, "side": side, "amount": size, "dry_run": True}
        try:
            if score > 0.8 and hasattr(exchange, "create_limit_order"):
                price = None
                try:
                    t = exchange.fetch_ticker(symbol)
                    bid = t.get("bid")
                    ask = t.get("ask")
                    if bid and ask:
                        price = (bid + ask) / 2
                except Exception as err:
                    logger.warning("Limit price fetch failed: %s", err)
                if price:
                    params = {"postOnly": True}
                    if config.get("hidden_limit"):
                        params["hidden"] = True
                    return exchange.create_limit_order(symbol, side, size, price, params)

            if ws_client is not None:
                return ws_client.add_order(symbol, side, size, order_type="market")
            return exchange.create_market_order(symbol, side, size)
        except Exception as exc:
            err_msg = notifier.notify(f"Order failed: {exc}")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            return {}

    err = notifier.notify(f"Placing {side} order for {amount} {symbol}")
    if err:
        logger.error("Failed to send message: %s", err)

    try:
        # Check if exchange has fetch_ticker method before calling it
        if hasattr(exchange, "fetch_ticker") and callable(getattr(exchange, "fetch_ticker", None)):
            ticker = exchange.fetch_ticker(symbol)
            bid = ticker.get("bid")
            ask = ticker.get("ask")
            if bid and ask:
                slippage = (ask - bid) / ((ask + bid) / 2)
                if slippage > config.get("max_slippage_pct", 1.0):
                    logger.warning("Trade skipped due to slippage.")
                    err_msg = notifier.notify("Trade skipped due to slippage.")
                    if err_msg:
                        logger.error("Failed to send message: %s", err_msg)
                    return {}
        else:
            logger.debug("Exchange does not support fetch_ticker, skipping slippage check")
    except Exception as err:  # pragma: no cover - network
        logger.warning("Slippage check failed: %s", err)

    if (
        config.get("liquidity_check", True)
        and hasattr(exchange, "fetch_order_book")
        and not has_liquidity(amount)
    ):
        notifier.notify("Insufficient liquidity for order size")
        return {}

    orders: List[Dict] = []
    if config.get("twap_enabled", False) and config.get("twap_slices", 1) > 1:
        slices = config.get("twap_slices", 1)
        delay = config.get("twap_interval_seconds", 1)
        slice_amount = amount / slices
        for i in range(slices):
            if (
                config.get("liquidity_check", True)
                and hasattr(exchange, "fetch_order_book")
                and not has_liquidity(slice_amount)
            ):
                err_liq = notifier.notify(
                    "Insufficient liquidity during TWAP execution"
                )
                if err_liq:
                    logger.error("Failed to send message: %s", err_liq)
                break
            order = place(slice_amount)
            if order:
                if dry_run:
                    try:
                        if hasattr(exchange, "fetch_ticker") and callable(getattr(exchange, "fetch_ticker", None)):
                            t = exchange.fetch_ticker(symbol)
                            order["price"] = t.get("last") or t.get("bid") or t.get("ask") or 0.0
                        else:
                            order["price"] = 0.0
                    except Exception:
                        order["price"] = 0.0
                log_trade(order)
                if (config or {}).get("tax_tracking", {}).get("enabled"):
                    try:
                        if order.get("side") == "buy":
                            tax_logger.record_entry(order)
                        else:
                            tax_logger.record_exit(order)
                    except Exception:
                        pass
                orders.append(order)
                err_slice = notifier.notify(
                    f"TWAP slice {i+1}/{slices} executed: {order}"
                )
                if err_slice:
                    logger.error("Failed to send message: %s", err_slice)
                oid = (
                    order.get("id")
                    or order.get("order_id")
                    or order.get("tx_hash")
                    or order.get("txid")
                )
                logger.info(
                    "TWAP slice %s/%s %s %s %.8f executed (id/tx: %s)",
                    i + 1,
                    slices,
                    side,
                    symbol,
                    slice_amount,
                    oid,
                )
            if i < slices - 1:
                time.sleep(delay)
    else:
        order = place(amount)
        if order:
            if dry_run:
                try:
                    if hasattr(exchange, "fetch_ticker") and callable(getattr(exchange, "fetch_ticker", None)):
                        t = exchange.fetch_ticker(symbol)
                        order["price"] = t.get("last") or t.get("bid") or t.get("ask") or 0.0
                    else:
                        order["price"] = 0.0
                except Exception:
                    order["price"] = 0.0
            log_trade(order)
            if (config or {}).get("tax_tracking", {}).get("enabled"):
                try:
                    if order.get("side") == "buy":
                        tax_logger.record_entry(order)
                    else:
                        tax_logger.record_exit(order)
                except Exception:
                    pass
            orders.append(order)
            err_exec = notifier.notify(f"Order executed: {order}")
            if err_exec:
                logger.error("Failed to send message: %s", err_exec)
            oid = (
                order.get("id")
                or order.get("order_id")
                or order.get("tx_hash")
                or order.get("txid")
            )
            logger.info(
                "Order executed %s %s %.8f (id/tx: %s)",
                side,
                symbol,
                amount,
                oid,
            )

    if len(orders) == 1:
        return orders[0]
    return {"orders": orders}


async def execute_trade_async(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    amount: float,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    notifier: Optional[TelegramNotifier] = None,
    dry_run: bool = True,
    use_websocket: bool = False,
    config: Optional[Dict] = None,
    score: float = 0.0,
) -> Dict:
    """Asynchronous version of :func:`execute_trade`. It supports both
    ``ccxt.pro`` exchanges and the threaded ``KrakenWSClient`` fallback."""

    if notifier is None:
        if isinstance(token, TelegramNotifier):
            notifier = token
        else:
            if token is None or chat_id is None:
                raise ValueError("token/chat_id or notifier must be provided")
            notifier = TelegramNotifier(token, chat_id)

    msg = f"Placing {side} order for {amount} {symbol}"
    err = notifier.notify(msg)
    
    if err:
        logger.error("Failed to send message: %s", err)
    if dry_run:
        order = {"symbol": symbol, "side": side, "amount": amount, "dry_run": True}
    else:
        try:
            if score > 0.8 and hasattr(exchange, "create_limit_order"):
                price = None
                try:
                    if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                        t = await exchange.fetch_ticker(symbol)
                    else:
                        t = await asyncio.to_thread(exchange.fetch_ticker, symbol)
                    bid = t.get("bid")
                    ask = t.get("ask")
                    if bid and ask:
                        price = (bid + ask) / 2
                except Exception as err:
                    logger.warning("Limit price fetch failed: %s", err)
                if price:
                    params = {"postOnly": True}
                    if config.get("hidden_limit"):
                        params["hidden"] = True
                    if asyncio.iscoroutinefunction(getattr(exchange, "create_limit_order", None)):
                        order = await exchange.create_limit_order(symbol, side, amount, price, params)
                    else:
                        order = await asyncio.to_thread(
                            exchange.create_limit_order, symbol, side, amount, price, params
                        )
                else:
                    if use_websocket and ws_client is not None and not ccxtpro:
                        order = ws_client.add_order(symbol, side, amount, order_type="market")
                    elif asyncio.iscoroutinefunction(
                        getattr(exchange, "create_market_order", None)
                    ):
                        order = await exchange.create_market_order(symbol, side, amount)
                    else:
                        order = await asyncio.to_thread(
                            exchange.create_market_order, symbol, side, amount
                        )
            else:
                if use_websocket and ws_client is not None and not ccxtpro:
                    order = ws_client.add_order(symbol, side, amount, order_type="market")
                elif asyncio.iscoroutinefunction(
                    getattr(exchange, "create_market_order", None)
                ):
                    order = await exchange.create_market_order(symbol, side, amount)
                else:
                    order = await asyncio.to_thread(
                        exchange.create_market_order, symbol, side, amount
                    )
        except Exception as e:  # pragma: no cover - network
            err_msg = notifier.notify(f"\u26a0\ufe0f Error: Order failed: {e}")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            return {}
    err = notifier.notify(f"Order executed: {order}")
    if err:
        logger.error("Failed to send message: %s", err)
    oid = (
        order.get("id")
        or order.get("order_id")
        or order.get("tx_hash")
        or order.get("txid")
    )
    logger.info(
        "Order executed %s %s %.8f (id/tx: %s)",
        side,
        symbol,
        amount,
        oid,
    )
    if dry_run:
        try:
            if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                t = await exchange.fetch_ticker(symbol)
            else:
                t = await asyncio.to_thread(exchange.fetch_ticker, symbol)
            order["price"] = t.get("last") or t.get("bid") or t.get("ask") or 0.0
        except Exception:
            order["price"] = 0.0
    log_trade(order)
    logger.info(
        "Order executed - id=%s side=%s amount=%s price=%s dry_run=%s",
        order.get("id"),
        order.get("side"),
        order.get("amount"),
        order.get("price") or order.get("average"),
        dry_run,
    )
    if (config or {}).get("tax_tracking", {}).get("enabled"):
        try:
            if order.get("side") == "buy":
                tax_logger.record_entry(order)
            else:
                tax_logger.record_exit(order)
        except Exception:
            pass
    return order


def place_stop_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    stop_price: float,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    notifier: Optional[TelegramNotifier] = None,
    dry_run: bool = True,
) -> Dict:
    """Submit a stop-loss order on the exchange."""
    if notifier is None:
        if token is None or chat_id is None:
            raise ValueError("token/chat_id or notifier must be provided")
        notifier = TelegramNotifier(token, chat_id)

    msg = f"Placing stop {side} order for {amount} {symbol} at {stop_price:.2f}"
    err = notifier.notify(msg)
    if err:
        logger.error("Failed to send message: %s", err)
    if dry_run:
        order = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "stop": stop_price,
            "dry_run": True,
        }
    else:
        try:
            order = exchange.create_order(
                symbol,
                "stop_market",
                side,
                amount,
                params={"stopPrice": stop_price},
            )
        except Exception as e:
            err_msg = notifier.notify(f"Stop order failed: {e}")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            return {}
    err = notifier.notify(f"Stop order submitted: {order}")
    if err:
        logger.error("Failed to send message: %s", err)
    log_trade(order, is_stop=True)
    return order
