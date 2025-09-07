import logging
import time
from typing import Optional, Dict, Any
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Cache for failed symbols to avoid repeated attempts
_FAILED_SYMBOL_CACHE = {}
_CACHE_TIMEOUT = 300  # 5 minutes

# Fallback price sources configuration
FALLBACK_SOURCES = {
    "coingecko": {
        "url": "https://api.coingecko.com/api/v3/simple/price",
        "params": lambda symbol: {
            "ids": symbol.split("/")[0].lower(),
            "vs_currencies": "usd"
        }
    },
    "binance": {
        "url": "https://api.binance.com/api/v3/ticker/price",
        "params": lambda symbol: {"symbol": _format_binance_symbol(symbol)}
    },
    "kraken": {
        "url": "https://api.kraken.com/0/public/Ticker",
        "params": lambda symbol: {"pair": _format_kraken_symbol(symbol)}
    }
}


def _format_binance_symbol(symbol: str) -> str:
    """Format symbol for Binance API."""
    base, quote = symbol.split("/")
    # Binance uses specific quote currencies
    if quote.upper() == "USD":
        quote = "USDT"  # Binance uses USDT instead of USD
    elif quote.upper() == "EUR":
        quote = "EUR"  # Keep EUR as is
    return f"{base.upper()}{quote.upper()}"


def _format_kraken_symbol(symbol: str) -> str:
    """Format symbol for Kraken API."""
    base, quote = symbol.split("/")
    # Kraken has specific symbol formats
    if quote.upper() == "USD":
        quote = "USD"
    elif quote.upper() == "EUR":
        quote = "EUR"
    return f"{base.upper()}{quote.upper()}"


def get_pyth_price(symbol: str, max_retries: int = 2) -> Optional[float]:
    """Return latest Pyth price for ``symbol`` with improved error handling.

    ``symbol`` should be formatted like ``"BTC/USD"``.
    Returns ``None`` on failure.
    """
    # Check cache for recently failed symbols
    current_time = time.time()
    if symbol in _FAILED_SYMBOL_CACHE:
        cache_time, _ = _FAILED_SYMBOL_CACHE[symbol]
        if current_time - cache_time < _CACHE_TIMEOUT:
            logger.debug(f"Skipping recently failed symbol: {symbol}")
            return None

    parts = symbol.split("/")
    if len(parts) != 2:
        logger.warning(f"Invalid symbol format: {symbol}")
        return None

    base, quote = parts

    # Try different symbol formats for better coverage
    symbol_formats = [
        f"Crypto.{base}/{quote}",
        f"Crypto.{base.upper()}/{quote.upper()}",
        f"Crypto.{base.lower()}/{quote.lower()}"
    ]

    # Multiple Pyth endpoints for redundancy
    endpoints = [
        "https://hermes.pyth.network/v2/price_feeds",
        "https://xc-mainnet.pyth.network/api/latest_price_feeds"  # Alternative endpoint
    ]

    for attempt in range(max_retries):
        for query in symbol_formats:
            for endpoint in endpoints:
                try:
                    # First request: get feed ID
                    if "hermes" in endpoint:
                        resp = requests.get(
                            endpoint,
                            params={"query": query},
                            timeout=8,  # Increased timeout
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        resp.raise_for_status()
                        data = resp.json()

                        if not data:
                            continue

                        feed_id = data[0].get("id")
                        if not feed_id:
                            continue

                        # Second request: get actual price
                        price_resp = requests.get(
                            "https://hermes.pyth.network/api/latest_price_feeds",
                            params={"ids[]": feed_id},
                            timeout=8,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        price_resp.raise_for_status()
                        price_data = price_resp.json()

                        if not price_data or not isinstance(price_data, list) or len(price_data) == 0:
                            continue

                        price_info = price_data[0].get("price")
                        if not price_info:
                            continue

                        price_value = price_info.get("price")
                        exponent = price_info.get("expo")

                        if price_value is None or exponent is None:
                            continue

                        price = float(price_value) * (10 ** int(exponent))

                        # Validate price is reasonable (not zero, negative, or extremely high)
                        if price > 0 and price < 10000000:  # Reasonable upper bound
                            logger.info(f"Successfully fetched Pyth price for {symbol}: ${price:.6f}")
                            return price
                        else:
                            logger.warning(f"Invalid price value for {symbol}: ${price}")

                    else:
                        # Alternative endpoint logic
                        continue

                except requests.exceptions.Timeout:
                    logger.debug(f"Timeout fetching Pyth price for {symbol} (attempt {attempt + 1})")
                    time.sleep(0.5)  # Brief pause before retry
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Network error fetching Pyth price for {symbol}: {e}")
                    time.sleep(0.5)
                except (ValueError, KeyError, IndexError) as e:
                    logger.debug(f"Data parsing error for {symbol}: {e}")
                except Exception as exc:
                    logger.error(f"Unexpected error fetching Pyth price for {symbol}: {exc}")

    # Try fallback sources if Pyth fails
    logger.debug(f"Pyth price fetch failed for {symbol}, trying fallback sources...")
    fallback_price = _get_fallback_price(symbol)
    if fallback_price is not None:
        logger.info(f"Successfully fetched fallback price for {symbol}: ${fallback_price:.6f}")
        return fallback_price

    # Log appropriate level based on symbol type
    mainstream_cryptos = {"BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "UNI", "AAVE", "MATIC", "AVAX", "XRP", "LTC", "BCH"}
    if base.upper() in mainstream_cryptos:
        logger.warning(f"All price fetch attempts failed for {symbol} (Pyth + fallbacks)")
    else:
        logger.debug(f"No price feed available for {symbol} (this is normal for non-mainstream tokens)")

    # For EUR pairs, try converting to USD first
    if quote.upper() == "EUR" and base.upper() not in ["BTC", "ETH"]:
        logger.debug(f"Trying USD conversion for EUR pair: {symbol}")
        usd_symbol = f"{base}/USD"
        usd_price = get_pyth_price(usd_symbol, max_retries=1)
        if usd_price is not None:
            # Simple EUR conversion (approximate)
            eur_price = usd_price * 0.85  # Rough EUR/USD rate
            logger.info(f"Converted {symbol} from USD: ${eur_price:.6f}")
            return eur_price

    # Cache failed symbol to avoid repeated attempts
    _FAILED_SYMBOL_CACHE[symbol] = (current_time, "all_attempts_failed")
    return None


def _get_fallback_price(symbol: str) -> Optional[float]:
    """Get price from fallback sources if Pyth fails."""
    base, quote = symbol.split("/")

    # Try each fallback source
    for source_name, source_config in FALLBACK_SOURCES.items():
        try:
            url = source_config["url"]
            params = source_config["params"](symbol)

            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            price = _parse_fallback_response(source_name, data, symbol)
            if price is not None:
                logger.debug(f"Got price from {source_name} for {symbol}: ${price:.6f}")
                return price

        except Exception as e:
            logger.debug(f"Fallback source {source_name} failed for {symbol}: {e}")
            continue

    return None


def _parse_fallback_response(source: str, data: Dict[str, Any], symbol: str) -> Optional[float]:
    """Parse price response from fallback sources."""
    try:
        if source == "coingecko":
            # Handle CoinGecko response format
            base_currency = symbol.split("/")[0].lower()
            if base_currency in data and "usd" in data[base_currency]:
                return float(data[base_currency]["usd"])

        elif source == "binance":
            # Handle Binance response format
            binance_symbol = _format_binance_symbol(symbol)
            if isinstance(data, list):
                for item in data:
                    if item.get("symbol") == binance_symbol:
                        return float(item.get("price", 0))
            elif isinstance(data, dict) and data.get("symbol") == binance_symbol:
                return float(data.get("price", 0))

        elif source == "kraken":
            # Handle Kraken response format
            kraken_symbol = _format_kraken_symbol(symbol)
            if "result" in data and kraken_symbol in data["result"]:
                pair_data = data["result"][kraken_symbol]
                if "c" in pair_data and len(pair_data["c"]) > 0:
                    return float(pair_data["c"][0])

    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"Failed to parse {source} response for {symbol}: {e}")

    return None


async def get_price_async(symbol: str, max_retries: int = 2) -> Optional[float]:
    """Async version of price fetching with concurrent fallback attempts."""
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Start Pyth fetch in background
        pyth_future = loop.run_in_executor(executor, get_pyth_price, symbol, max_retries)

        # Start fallback fetches in parallel
        fallback_futures = []
        for source_name in FALLBACK_SOURCES.keys():
            future = loop.run_in_executor(executor, _get_single_fallback_price, source_name, symbol)
            fallback_futures.append(future)

        # Wait for any result
        all_futures = [pyth_future] + fallback_futures
        done, pending = await asyncio.wait(all_futures, return_when=asyncio.FIRST_COMPLETED)

        # Cancel pending requests
        for future in pending:
            future.cancel()

        # Return first successful result
        for future in done:
            try:
                result = future.result()
                if result is not None:
                    return result
            except Exception:
                continue

    return None


def _get_single_fallback_price(source_name: str, symbol: str) -> Optional[float]:
    """Get price from a single fallback source."""
    try:
        source_config = FALLBACK_SOURCES[source_name]
        url = source_config["url"]
        params = source_config["params"](symbol)

        resp = requests.get(url, params=params, timeout=3)
        resp.raise_for_status()
        data = resp.json()

        return _parse_fallback_response(source_name, data, symbol)

    except Exception:
        return None
