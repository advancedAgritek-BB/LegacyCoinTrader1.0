import logging
import time
from typing import Optional
import requests

logger = logging.getLogger(__name__)


def get_pyth_price(symbol: str, max_retries: int = 2) -> Optional[float]:
    """Return latest Pyth price for ``symbol`` with improved error handling.

    ``symbol`` should be formatted like ``"BTC/USD"``.
    Returns ``None`` on failure.
    """
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
                    logger.warning(f"Timeout fetching Pyth price for {symbol} (attempt {attempt + 1})")
                    time.sleep(0.5)  # Brief pause before retry
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Network error fetching Pyth price for {symbol}: {e}")
                    time.sleep(0.5)
                except (ValueError, KeyError, IndexError) as e:
                    logger.warning(f"Data parsing error for {symbol}: {e}")
                except Exception as exc:
                    logger.error(f"Unexpected error fetching Pyth price for {symbol}: {exc}")

    logger.warning(f"All Pyth price fetch attempts failed for {symbol}")
    return None
