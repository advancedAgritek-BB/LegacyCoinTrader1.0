"""
Enhanced OHLCV fetcher with graceful handling of unsupported timeframes.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Union, Tuple
import ccxt
import pandas as pd

from .config_validator import get_exchange_supported_timeframes
from .market_loader import fetch_ohlcv_async

logger = logging.getLogger(__name__)

class EnhancedOHLCVFetcher:
    """
    Enhanced OHLCV fetcher that handles unsupported timeframes gracefully.
    """
    
    def __init__(self, exchange: ccxt.Exchange, config: Dict):
        self.exchange = exchange
        self.config = config
        self.exchange_name = getattr(exchange, 'id', 'unknown').lower()
        self.supported_timeframes = get_exchange_supported_timeframes(self.exchange_name)
        
        # Timeframe fallback mapping
        self.timeframe_fallbacks = {
            '10m': '15m',  # 10m -> 15m
            '30m': '15m',  # 30m -> 15m
            '6h': '4h',    # 6h -> 4h
            '2w': '1w',    # 2w -> 1w
            '3w': '1w',    # 3w -> 1w
        }
    
    def get_supported_timeframe(self, requested_tf: str) -> str:
        """
        Get a supported timeframe, using fallback if necessary.
        
        Args:
            requested_tf: Requested timeframe
            
        Returns:
            Supported timeframe (may be different from requested)
        """
        if requested_tf in self.supported_timeframes:
            return requested_tf
        
        # Try fallback
        if requested_tf in self.timeframe_fallbacks:
            fallback = self.timeframe_fallbacks[requested_tf]
            if fallback in self.supported_timeframes:
                logger.warning(
                    f"Timeframe '{requested_tf}' not supported on {self.exchange_name}, "
                    f"using fallback '{fallback}'"
                )
                return fallback
        
        # If no fallback available, return the closest supported timeframe
        logger.warning(
            f"Timeframe '{requested_tf}' not supported on {self.exchange_name}, "
            f"using closest supported timeframe"
        )
        return self._find_closest_timeframe(requested_tf)
    
    def _find_closest_timeframe(self, requested_tf: str) -> str:
        """Find the closest supported timeframe to the requested one."""
        # Parse requested timeframe to minutes
        requested_minutes = self._timeframe_to_minutes(requested_tf)
        if requested_minutes is None:
            return '1h'  # Default fallback
        
        # Find closest supported timeframe
        closest_tf = '1h'
        min_diff = float('inf')
        
        for tf in self.supported_timeframes:
            tf_minutes = self._timeframe_to_minutes(tf)
            if tf_minutes is not None:
                diff = abs(tf_minutes - requested_minutes)
                if diff < min_diff:
                    min_diff = diff
                    closest_tf = tf
        
        logger.info(f"Using closest supported timeframe '{closest_tf}' for '{requested_tf}'")
        return closest_tf
    
    def _timeframe_to_minutes(self, timeframe: str) -> Optional[int]:
        """Convert timeframe string to minutes."""
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            
            if unit == 'm':
                return value
            elif unit == 'h':
                return value * 60
            elif unit == 'd':
                return value * 1440
            elif unit == 'w':
                return value * 10080
            elif unit == 'M':
                return value * 43200
            else:
                return None
        except (ValueError, IndexError):
            return None
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = "1h", 
        limit: int = 100,
        since: Optional[int] = None,
        use_websocket: bool = False
    ) -> Union[List, Exception]:
        """
        Fetch OHLCV data with automatic timeframe fallback.
        
        Args:
            symbol: Trading symbol
            timeframe: Requested timeframe
            limit: Number of candles to fetch
            since: Start timestamp
            use_websocket: Whether to use WebSocket
            
        Returns:
            OHLCV data or Exception
        """
        # Get supported timeframe
        supported_tf = self.get_supported_timeframe(timeframe)
        
        if supported_tf != timeframe:
            logger.info(
                f"Fetching {symbol} OHLCV with timeframe '{supported_tf}' "
                f"(requested: '{timeframe}')"
            )
        
        try:
            # Fetch data with supported timeframe
            data = await fetch_ohlcv_async(
                self.exchange,
                symbol,
                timeframe=supported_tf,
                limit=limit,
                since=since,
                use_websocket=use_websocket
            )
            
            if isinstance(data, Exception):
                logger.error(f"Failed to fetch OHLCV for {symbol} on {supported_tf}: {data}")
                return data
            
            if not data:
                logger.warning(f"No OHLCV data returned for {symbol} on {supported_tf}")
                return []
            
            # If we used a fallback timeframe, we might need to resample
            if supported_tf != timeframe:
                data = self._resample_data(data, supported_tf, timeframe)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} on {supported_tf}: {e}")
            return e
    
    def _resample_data(self, data: List, from_tf: str, to_tf: str) -> List:
        """
        Resample data from one timeframe to another.
        
        Args:
            data: OHLCV data in [timestamp, open, high, low, close, volume] format
            from_tf: Source timeframe
            to_tf: Target timeframe
            
        Returns:
            Resampled OHLCV data
        """
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(
                data, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert timeframes to pandas offset strings
            from_offset = self._timeframe_to_pandas_offset(from_tf)
            to_offset = self._timeframe_to_pandas_offset(to_tf)
            
            if from_offset and to_offset:
                # Resample
                resampled = df.resample(to_offset).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # Convert back to list format
                result = []
                for timestamp, row in resampled.iterrows():
                    result.append([
                        int(timestamp.timestamp() * 1000),  # Convert to milliseconds
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ])
                
                logger.info(f"Resampled {len(data)} candles from {from_tf} to {to_tf}: {len(result)} candles")
                return result
            
        except Exception as e:
            logger.error(f"Error resampling data from {from_tf} to {to_tf}: {e}")
        
        # Return original data if resampling fails
        return data
    
    def _timeframe_to_pandas_offset(self, timeframe: str) -> Optional[str]:
        """Convert timeframe string to pandas offset string."""
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            
            if unit == 'm':
                return f'{value}T'  # pandas uses 'T' for minutes
            elif unit == 'h':
                return f'{value}H'
            elif unit == 'd':
                return f'{value}D'
            elif unit == 'w':
                return f'{value}W'
            elif unit == 'M':
                return f'{value}M'
            else:
                return None
        except (ValueError, IndexError):
            return None
    
    async def fetch_multiple_timeframes(
        self, 
        symbol: str, 
        timeframes: List[str], 
        limit: int = 100
    ) -> Dict[str, Union[List, Exception]]:
        """
        Fetch OHLCV data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of requested timeframes
            limit: Number of candles to fetch per timeframe
            
        Returns:
            Dictionary mapping timeframes to OHLCV data
        """
        results = {}
        
        for tf in timeframes:
            data = await self.fetch_ohlcv(symbol, tf, limit)
            results[tf] = data
        
        return results
    
    def get_supported_timeframes_for_symbol(self, symbol: str) -> List[str]:
        """
        Get list of timeframes that are supported for a specific symbol.
        This can be useful for determining which timeframes to use for analysis.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of supported timeframes
        """
        # For now, return all supported timeframes
        # In the future, this could check symbol-specific limitations
        return sorted(list(self.supported_timeframes))
    
    def validate_timeframe_request(self, timeframe: str) -> Tuple[bool, str]:
        """
        Validate if a timeframe request is supported.
        
        Args:
            timeframe: Requested timeframe
            
        Returns:
            Tuple of (is_supported, message)
        """
        if timeframe in self.supported_timeframes:
            return True, f"Timeframe '{timeframe}' is supported on {self.exchange_name}"
        
        # Check if fallback is available
        if timeframe in self.timeframe_fallbacks:
            fallback = self.timeframe_fallbacks[timeframe]
            if fallback in self.supported_timeframes:
                return False, f"Timeframe '{timeframe}' not supported, will use fallback '{fallback}'"
        
        return False, f"Timeframe '{timeframe}' not supported on {self.exchange_name}, no fallback available"
