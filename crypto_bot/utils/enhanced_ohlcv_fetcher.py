"""Enhanced OHLCV fetcher with intelligent symbol routing and async concurrency."""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from collections import defaultdict

from .market_loader import (
    fetch_ohlcv_async,
    fetch_geckoterminal_ohlcv,
    fetch_helius_ohlcv,
    fetch_dex_ohlcv,
)
from .token_validator import _is_valid_base_token

logger = logging.getLogger(__name__)

class EnhancedOHLCVFetcher:
    """Enhanced OHLCV fetcher with intelligent symbol routing."""
    
    def __init__(self, exchange, config: Dict[str, Any]):
        self.exchange = exchange
        self.config = config
        self.max_concurrent_cex = config.get("max_concurrent_ohlcv", 3)
        self.max_concurrent_dex = config.get("max_concurrent_dex_ohlcv", 10)
        self.min_volume_usd = float(config.get("min_volume_usd", 0) or 0)

        # Circuit breakers for different APIs
        try:
            self.cex_semaphore = asyncio.Semaphore(self.max_concurrent_cex)
            self.dex_semaphore = asyncio.Semaphore(self.max_concurrent_dex)
        except RuntimeError as e:
            # Handle event loop conflicts
            logger.warning(f"Enhanced OHLCV Fetcher: Event loop conflict in semaphore creation: {e}")
            # Create semaphores in a safe way
            import threading
            self._semaphore_lock = threading.Lock()
            self._cex_semaphore_value = self.max_concurrent_cex
            self._dex_semaphore_value = self.max_concurrent_dex

        # Initialize supported timeframes
        self.supported_timeframes = self._get_supported_timeframes()

    def _get_semaphore(self, is_cex: bool = True):
        """Safely get semaphore, handling event loop conflicts."""
        if hasattr(self, 'cex_semaphore'):
            return self.cex_semaphore if is_cex else self.dex_semaphore

        # Fallback for event loop conflicts
        import threading
        if not hasattr(self, '_semaphore_lock'):
            self._semaphore_lock = threading.Lock()
            self._cex_semaphore_value = self.max_concurrent_cex
            self._dex_semaphore_value = self.max_concurrent_dex

        class ThreadSafeSemaphore:
            def __init__(self, value, lock):
                self.value = value
                self.lock = lock

            async def __aenter__(self):
                # For event loop conflicts, just yield control briefly
                await asyncio.sleep(0.001)
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        return ThreadSafeSemaphore(
            self._cex_semaphore_value if is_cex else self._dex_semaphore_value,
            self._semaphore_lock
        )

    def _get_supported_timeframes(self) -> List[str]:
        """Get supported timeframes from exchange or use defaults."""
        if hasattr(self.exchange, 'timeframes') and self.exchange.timeframes:
            # Handle both dict and set/list formats
            if isinstance(self.exchange.timeframes, dict):
                return list(self.exchange.timeframes.keys())
            else:
                # Assume it's a set or list
                return list(self.exchange.timeframes)

        # Default timeframes if exchange doesn't provide them
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

    def validate_timeframe_request(self, timeframe: str) -> Tuple[bool, str]:
        """Validate if a timeframe is supported and return validation result."""
        if timeframe in self.supported_timeframes:
            return True, f"Timeframe '{timeframe}' is supported"

        # Try to find closest match
        closest = self._find_closest_timeframe(timeframe)
        if closest:
            return False, f"Timeframe '{timeframe}' not supported. Closest match: '{closest}'"
        else:
            return False, f"Timeframe '{timeframe}' not supported and no suitable alternative found"

    def get_supported_timeframe(self, timeframe: str) -> str:
        """Get the best supported timeframe for the requested timeframe."""
        if timeframe in self.supported_timeframes:
            return timeframe

        # Return closest match
        closest = self._find_closest_timeframe(timeframe)
        return closest if closest else '1h'  # Default fallback

    def _find_closest_timeframe(self, timeframe: str) -> Optional[str]:
        """Find the closest supported timeframe to the requested one."""
        if timeframe in self.supported_timeframes:
            return timeframe

        # Parse timeframe to minutes
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        if timeframe_minutes is None:
            return None

        # Find closest supported timeframe
        closest_tf = None
        min_diff = float('inf')

        for supported_tf in self.supported_timeframes:
            supported_minutes = self._timeframe_to_minutes(supported_tf)
            if supported_minutes is None:
                continue

            diff = abs(timeframe_minutes - supported_minutes)
            if diff < min_diff:
                min_diff = diff
                closest_tf = supported_tf

        return closest_tf

    def _timeframe_to_minutes(self, timeframe: str) -> Optional[int]:
        """Convert timeframe string to minutes."""
        if not timeframe:
            return None

        timeframe = timeframe.lower()

        # Handle different formats
        try:
            if timeframe.endswith('m'):
                return int(timeframe[:-1])
            elif timeframe.endswith('h'):
                return int(timeframe[:-1]) * 60
            elif timeframe.endswith('d'):
                return int(timeframe[:-1]) * 24 * 60
            elif timeframe.endswith('w'):
                return int(timeframe[:-1]) * 7 * 24 * 60
            else:
                return None
        except ValueError:
            return None

    def _classify_symbols(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        """Classify symbols into CEX and DEX categories."""
        cex_symbols = []
        dex_symbols = []

        for symbol in symbols:
            if symbol is None or not isinstance(symbol, str):
                continue

            # Check if it's a raw Solana Base58 address (no slash separator)
            if "/" not in symbol and len(symbol) >= 32 and len(symbol) <= 44:
                # Check if it matches Solana Base58 pattern
                import re
                if re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", symbol):
                    dex_symbols.append(symbol)
                    continue

            base, _, quote = symbol.partition("/")
            if quote.upper() == "USDC" and _is_valid_base_token(base):
                dex_symbols.append(symbol)
            else:
                cex_symbols.append(symbol)

        return cex_symbols, dex_symbols
    
    async def _fetch_cex_ohlcv_batch(
        self, 
        symbols: List[str], 
        timeframe: str, 
        limit: int,
        since_map: Optional[Dict[str, int]] = None
    ) -> Dict[str, list]:
        """Fetch OHLCV data for CEX symbols with proper concurrency."""
        if not symbols:
            return {}
            
        since_map = since_map or {}
        
        async def fetch_single(symbol: str):
            async with self._get_semaphore(is_cex=True):
                try:
                    # Ensure markets are loaded before checking support list
                    try:
                        if hasattr(self.exchange, 'markets') and (not getattr(self.exchange, 'markets')):
                            # Load markets synchronously (ccxt is sync by default)
                            self.exchange.load_markets()
                    except Exception as load_err:
                        logger.debug(f"Enhanced fetcher: load_markets() failed or not needed: {load_err}")

                    # Check if symbol is supported by the exchange before attempting fetch
                    if hasattr(self.exchange, 'markets') and self.exchange.markets:
                        if symbol not in self.exchange.markets:
                            logger.warning(f"Skipping unsupported symbol {symbol} on {getattr(self.exchange, 'name', 'exchange')} - not in market list")
                            return symbol, None

                    logger.debug(f"Enhanced fetcher: About to call fetch_ohlcv_async for {symbol}")
                    data = await fetch_ohlcv_async(
                        self.exchange,
                        symbol,
                        timeframe=timeframe,
                        limit=limit,
                        since=since_map.get(symbol),
                        use_websocket=False,  # Use REST for reliability
                        force_websocket_history=False
                    )

                    if data:
                        logger.debug(f"Enhanced fetcher: Got {len(data)} candles for {symbol}")
                    else:
                        logger.warning(f"Enhanced fetcher: No data returned for {symbol}")
                    logger.debug(f"Enhanced fetcher: fetch_ohlcv_async returned: {type(data)}, length: {len(data) if data else 'None'}")

                    # Rate limiting
                    rate_limit = getattr(self.exchange, "rateLimit", 100)
                    await asyncio.sleep(rate_limit / 1000)

                    if data:
                        logger.info(f"Successfully fetched CEX OHLCV for {symbol}: {len(data)} candles")
                    else:
                        logger.warning(f"No data returned for CEX symbol {symbol}")

                    return symbol, data
                except Exception as e:
                    error_msg = str(e).lower()
                    if "not supported" in error_msg or "invalid symbol" in error_msg or "market not found" in error_msg:
                        logger.warning(f"Skipping unsupported symbol {symbol} on {getattr(self.exchange, 'name', 'exchange')}: {e}")
                    else:
                        logger.error(f"Failed to fetch CEX OHLCV for {symbol}: {e}")
                    return symbol, None
        
        tasks = [fetch_single(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_map = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"CEX OHLCV task failed: {result}")
                continue

            symbol_result, data = result
            if data is not None:
                logger.debug(f"Enhanced fetcher: Processing result for {symbol_result}: {type(data)}, length: {len(data) if data else 'None'}")
                data_map[symbol_result] = data
            else:
                logger.warning(f"No data returned for CEX symbol {symbol_result}")
                
        return data_map
    
    async def _fetch_dex_ohlcv_batch(
        self, 
        symbols: List[str], 
        timeframe: str, 
        limit: int
    ) -> Dict[str, list]:
        """Fetch OHLCV data for DEX symbols with proper concurrency."""
        if not symbols:
            return {}
            
        async def fetch_single(symbol: str):
            async with self._get_semaphore(is_cex=False):
                try:
                    # Try GeckoTerminal first
                    data = None
                    try:
                        res = await fetch_geckoterminal_ohlcv(
                            symbol,
                            timeframe=timeframe,
                            limit=limit,
                            min_24h_volume=self.min_volume_usd,
                        )
                        if res and isinstance(res, tuple):
                            data, vol, *_ = res
                        elif res:
                            data = res
                    except Exception as e:
                        logger.debug(f"GeckoTerminal failed for {symbol}: {e}")
                    
                    # Fallback to Helius if GeckoTerminal fails
                    if not data:
                        try:
                            data = await fetch_helius_ohlcv(
                                symbol,
                                timeframe=timeframe,
                                limit=limit
                            )
                        except Exception as e:
                            logger.debug(f"Helius failed for {symbol}: {e}")
                    
                    # Final fallback to DEX OHLCV
                    if not data:
                        data = await fetch_dex_ohlcv(
                            self.exchange,
                            symbol,
                            timeframe=timeframe,
                            limit=limit,
                            min_volume_usd=self.min_volume_usd,
                            use_gecko=False
                        )
                    
                    return symbol, data
                except Exception as e:
                    logger.warning(f"Failed to fetch DEX OHLCV for {symbol}: {e}")
                    return symbol, None
        
        tasks = [fetch_single(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_map = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"DEX OHLCV task failed: {result}")
                continue
            if result and isinstance(result, (list, tuple)) and len(result) == 2 and result[1] is not None:
                data_map[result[0]] = result[1]
                
        return data_map
    
    async def fetch_ohlcv_batch(
        self,
        symbols: List[str],
        timeframe: str,
        limit: int,
        since_map: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, list], Dict[str, list]]:
        """Fetch OHLCV data for all symbols with intelligent routing."""
        logger.info(f"Enhanced OHLCV Fetcher: fetch_ohlcv_batch called with {len(symbols)} symbols")

        if not symbols:
            return {}, {}

        # Validate timeframe
        is_valid_tf, tf_message = self.validate_timeframe_request(timeframe)
        if not is_valid_tf:
            logger.warning(f"Enhanced OHLCV Fetcher: {tf_message}")
            # Use closest supported timeframe
            supported_tf = self.get_supported_timeframe(timeframe)
            logger.info(f"Enhanced OHLCV Fetcher: Using supported timeframe '{supported_tf}' instead of '{timeframe}'")
            timeframe = supported_tf

        # Classify symbols
        cex_symbols, dex_symbols = self._classify_symbols(symbols)

        logger.info(f"Enhanced OHLCV Fetcher: {len(cex_symbols)} CEX symbols, {len(dex_symbols)} DEX symbols")
        if cex_symbols:
            logger.debug(f"Enhanced OHLCV Fetcher: CEX symbols: {cex_symbols[:5]}{'...' if len(cex_symbols) > 5 else ''}")
        if dex_symbols:
            logger.debug(f"Enhanced OHLCV Fetcher: DEX symbols: {dex_symbols[:5]}{'...' if len(dex_symbols) > 5 else ''}")

        # Set timeout for the entire batch operation
        timeout_seconds = self.config.get("ohlcv_fetcher_timeout", 120)  # 2 minutes default

        try:
            # Create timeout-aware tasks
            async def fetch_with_timeout(fetch_func, *args, **kwargs):
                try:
                    return await asyncio.wait_for(fetch_func(*args, **kwargs), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    logger.error(f"Enhanced OHLCV Fetcher: Timeout ({timeout_seconds}s) reached for {fetch_func.__name__}")
                    return {}
                except Exception as e:
                    logger.error(f"Enhanced OHLCV Fetcher: Error in {fetch_func.__name__}: {e}")
                    return {}

            # Fetch both batches concurrently with timeout protection
            cex_task = fetch_with_timeout(self._fetch_cex_ohlcv_batch, cex_symbols, timeframe, limit, since_map)
            dex_task = fetch_with_timeout(self._fetch_dex_ohlcv_batch, dex_symbols, timeframe, limit)

            logger.info(f"Enhanced OHLCV Fetcher: Starting concurrent fetch for CEX and DEX with {timeout_seconds}s timeout")

            # Use asyncio.gather with exception handling
            results = await asyncio.gather(cex_task, dex_task, return_exceptions=True)

            # Handle results and exceptions
            cex_data = results[0] if not isinstance(results[0], Exception) else {}
            dex_data = results[1] if not isinstance(results[1], Exception) else {}

            if isinstance(results[0], Exception):
                logger.error(f"Enhanced OHLCV Fetcher: CEX fetch failed: {results[0]}")
            if isinstance(results[1], Exception):
                logger.error(f"Enhanced OHLCV Fetcher: DEX fetch failed: {results[1]}")

            logger.info(f"Enhanced OHLCV Fetcher: CEX fetched {len(cex_data)} symbols, DEX fetched {len(dex_data)} symbols")

            # Keep CEX and DEX data separate - DO NOT COMBINE
            logger.info(f"Enhanced OHLCV Fetcher: Returning {len(cex_data)} CEX symbols and {len(dex_data)} DEX symbols separately")

            return cex_data, dex_data

        except Exception as e:
            logger.error(f"Enhanced OHLCV Fetcher: Error in fetch_ohlcv_batch: {e}")
            import traceback
            logger.error(f"Enhanced OHLCV Fetcher: Traceback: {traceback.format_exc()}")
            return {}, {}

    async def update_cache(
        self,
        cache: Dict[str, pd.DataFrame],
        symbols: List[str],
        timeframe: str,
        limit: int,
        since_map: Optional[Dict[str, int]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Update cache with new OHLCV data."""
        logger.info(f"Enhanced OHLCV Fetcher: update_cache called with {len(symbols)} symbols for {timeframe}")

        if not symbols:
            return cache

        try:
            # Fetch new data for all symbols - now returns separate CEX and DEX data
            cex_data, dex_data = await self.fetch_ohlcv_batch(symbols, timeframe, limit, since_map)

            logger.info(f"Enhanced OHLCV Fetcher: Fetched {len(cex_data)} CEX and {len(dex_data)} DEX symbols")

            # Update cache with CEX data
            for symbol, ohlcv_data in cex_data.items():
                if not ohlcv_data:
                    continue

                try:
                    # Debug logging with more detail
                    logger.info(f"Enhanced OHLCV Fetcher: Processing data for {symbol}: {type(ohlcv_data)}")
                    if hasattr(ohlcv_data, '__len__'):
                        logger.info(f"Enhanced OHLCV Fetcher: Data length: {len(ohlcv_data)}")
                        if len(ohlcv_data) > 0:
                            logger.info(f"Enhanced OHLCV Fetcher: First item: {ohlcv_data[0]}, type: {type(ohlcv_data[0])}")

                    # Validate data format before creating DataFrame
                    if isinstance(ohlcv_data, (list, tuple)):
                        # Handle case where single tuple is returned instead of list
                        if isinstance(ohlcv_data, tuple) and len(ohlcv_data) == 6:
                            # Convert single tuple to list with one element
                            logger.info(f"Enhanced OHLCV Fetcher: Converting single tuple to list for {symbol}")
                            ohlcv_data = [ohlcv_data]
                        elif isinstance(ohlcv_data, tuple):
                            # Handle other tuple formats - log as debug instead of warning
                            logger.debug(f"Enhanced OHLCV Fetcher: Unexpected tuple format for {symbol}: length={len(ohlcv_data)}, data={ohlcv_data}")
                            continue
                        elif not isinstance(ohlcv_data, list):
                            logger.error(f"Enhanced OHLCV Fetcher: Invalid data type for {symbol}: {type(ohlcv_data)}")
                            continue
                    else:
                        logger.error(f"Enhanced OHLCV Fetcher: Invalid data type for {symbol}: {type(ohlcv_data)}")
                        continue

                    if len(ohlcv_data) == 0:
                        logger.warning(f"Enhanced OHLCV Fetcher: Empty data for {symbol}")
                        continue

                    # Ensure each row is a list/tuple with 6 elements
                    validated_data = []
                    for i, row in enumerate(ohlcv_data):
                        if not isinstance(row, (list, tuple)):
                            logger.error(f"Enhanced OHLCV Fetcher: Row {i} is not a list/tuple for {symbol}: {type(row)}")
                            continue
                        if len(row) != 6:
                            logger.error(f"Enhanced OHLCV Fetcher: Row {i} doesn't have 6 elements for {symbol}: {len(row)}")
                            continue
                        validated_data.append(row)

                    if not validated_data:
                        logger.error(f"Enhanced OHLCV Fetcher: No valid data rows for {symbol}")
                        continue

                    # Create DataFrame with validation
                    df_new = pd.DataFrame(
                        validated_data,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )

                    # Validate DataFrame
                    if df_new.empty:
                        logger.warning(f"Enhanced OHLCV Fetcher: Empty DataFrame for {symbol}")
                        continue

                    # Check for minimum required candles
                    min_candles_required = int(limit * 0.5)
                    if len(df_new) < min_candles_required:
                        logger.info(
                            f"Enhanced OHLCV Fetcher: Incomplete data for {symbol}: "
                            f"{len(df_new)}/{limit} candles"
                        )
                        # Still cache partial data

                    # Update cache
                    if symbol in cache and not cache[symbol].empty:
                        # Merge with existing data
                        last_ts = cache[symbol]["timestamp"].iloc[-1]
                        df_new = df_new[df_new["timestamp"] > last_ts]
                        if not df_new.empty:
                            cache[symbol] = pd.concat([cache[symbol], df_new], ignore_index=True)
                            logger.debug(f"Enhanced OHLCV Fetcher: Updated cache for {symbol} with {len(df_new)} new candles")
                    else:
                        # New symbol
                        cache[symbol] = df_new
                        logger.debug(f"Enhanced OHLCV Fetcher: Created new cache entry for {symbol} with {len(df_new)} candles")

                    # Add return column if not present
                    if "return" not in cache[symbol].columns:
                        cache[symbol]["return"] = cache[symbol]["close"].pct_change()

                except Exception as e:
                    logger.error(f"Enhanced OHLCV Fetcher: Failed to process CEX data for {symbol}: {e}")
                    continue

            # Update cache with DEX data
            for symbol, ohlcv_data in dex_data.items():
                if not ohlcv_data:
                    continue

                try:
                    # Debug logging with more detail
                    logger.info(f"Enhanced OHLCV Fetcher: Processing DEX data for {symbol}: {type(ohlcv_data)}")
                    if hasattr(ohlcv_data, '__len__'):
                        logger.info(f"Enhanced OHLCV Fetcher: DEX data length: {len(ohlcv_data)}")
                        if len(ohlcv_data) > 0:
                            logger.info(f"Enhanced OHLCV Fetcher: DEX first item: {ohlcv_data[0]}, type: {type(ohlcv_data[0])}")

                    # Validate data format before creating DataFrame
                    if isinstance(ohlcv_data, (list, tuple)):
                        # Handle case where single tuple is returned instead of list
                        if isinstance(ohlcv_data, tuple) and len(ohlcv_data) == 6:
                            # Convert single tuple to list with one element
                            logger.info(f"Enhanced OHLCV Fetcher: Converting single DEX tuple to list for {symbol}")
                            ohlcv_data = [ohlcv_data]
                        elif isinstance(ohlcv_data, tuple):
                            # Handle other tuple formats - log as debug instead of warning
                            logger.debug(f"Enhanced OHLCV Fetcher: Unexpected DEX tuple format for {symbol}: length={len(ohlcv_data)}, data={ohlcv_data}")
                            continue
                        elif not isinstance(ohlcv_data, list):
                            logger.error(f"Enhanced OHLCV Fetcher: Invalid DEX data type for {symbol}: {type(ohlcv_data)}")
                            continue
                    else:
                        logger.error(f"Enhanced OHLCV Fetcher: Invalid DEX data type for {symbol}: {type(ohlcv_data)}")
                        continue

                    if len(ohlcv_data) == 0:
                        logger.warning(f"Enhanced OHLCV Fetcher: Empty DEX data for {symbol}")
                        continue

                    # Ensure each row is a list/tuple with 6 elements
                    validated_data = []
                    for i, row in enumerate(ohlcv_data):
                        if not isinstance(row, (list, tuple)):
                            logger.error(f"Enhanced OHLCV Fetcher: DEX row {i} is not a list/tuple for {symbol}: {type(row)}")
                            continue
                        if len(row) != 6:
                            logger.error(f"Enhanced OHLCV Fetcher: DEX row {i} doesn't have 6 elements for {symbol}: {len(row)}")
                            continue
                        validated_data.append(row)

                    if not validated_data:
                        logger.error(f"Enhanced OHLCV Fetcher: No valid DEX data rows for {symbol}")
                        continue

                    # Create DataFrame with validation
                    df_new = pd.DataFrame(
                        validated_data,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )

                    # Validate DataFrame
                    if df_new.empty:
                        logger.warning(f"Enhanced OHLCV Fetcher: Empty DEX DataFrame for {symbol}")
                        continue

                    # Check for minimum required candles
                    min_candles_required = int(limit * 0.5)
                    if len(df_new) < min_candles_required:
                        logger.info(
                            f"Enhanced OHLCV Fetcher: Incomplete DEX data for {symbol}: "
                            f"{len(df_new)}/{limit} candles"
                        )
                        # Still cache partial data

                    # Update cache
                    if symbol in cache and not cache[symbol].empty:
                        # Merge with existing data
                        last_ts = cache[symbol]["timestamp"].iloc[-1]
                        df_new = df_new[df_new["timestamp"] > last_ts]
                        if not df_new.empty:
                            cache[symbol] = pd.concat([cache[symbol], df_new], ignore_index=True)
                            logger.debug(f"Enhanced OHLCV Fetcher: Updated DEX cache for {symbol} with {len(df_new)} new candles")
                    else:
                        # New symbol
                        cache[symbol] = df_new
                        logger.debug(f"Enhanced OHLCV Fetcher: Created new DEX cache entry for {symbol} with {len(df_new)} candles")

                    # Add return column if not present
                    if "return" not in cache[symbol].columns:
                        cache[symbol]["return"] = cache[symbol]["close"].pct_change()

                except Exception as e:
                    logger.error(f"Enhanced OHLCV Fetcher: Failed to process DEX data for {symbol}: {e}")
                    continue

            logger.info(f"Enhanced OHLCV Fetcher: Successfully updated cache with {len(cex_data)} CEX and {len(dex_data)} DEX symbols")
            return cache

        except Exception as e:
            logger.error(f"Enhanced OHLCV Fetcher: Error in update_cache: {e}")
            import traceback
            logger.error(f"Enhanced OHLCV Fetcher: Traceback: {traceback.format_exc()}")
            return cache
