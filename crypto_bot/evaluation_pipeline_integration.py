"""
Evaluation Pipeline Integration Module

This module provides a robust, bulletproof integration between the enhanced scanning
system and the main trading evaluation pipeline. It ensures tokens flow seamlessly
from scanning → evaluation → trading signals with comprehensive error handling,
monitoring, and fallback mechanisms.

Key Features:
- Robust token flow from scanner to evaluator
- Multiple fallback mechanisms
- Comprehensive error handling and recovery
- Real-time monitoring and health checks
- Graceful degradation when components fail
- Performance optimization and caching
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .enhanced_scan_integration import get_enhanced_scan_integration
from .utils.logger import setup_logger, LOG_DIR

logger = setup_logger(__name__, LOG_DIR / "evaluation_pipeline_integration.log")


class PipelineStatus(Enum):
    """Enumeration of pipeline status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class TokenSource(Enum):
    """Enumeration of token sources."""
    ENHANCED_SCANNER = "enhanced_scanner"
    SOLANA_SCANNER = "solana_scanner"
    STATIC_CONFIG = "static_config"
    FALLBACK = "fallback"


@dataclass
class PipelineMetrics:
    """Metrics for monitoring pipeline performance."""
    tokens_received: int = 0
    tokens_processed: int = 0
    tokens_failed: int = 0
    avg_processing_time: float = 0.0
    error_rate: float = 0.0
    last_successful_run: Optional[float] = None
    consecutive_failures: int = 0
    total_runtime: float = 0.0


@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline integration."""
    enabled: bool = True
    max_batch_size: int = 20
    processing_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0
    max_consecutive_failures: int = 5
    enable_fallback_sources: bool = True
    cache_ttl: float = 300.0  # 5 minutes


@dataclass
class TokenBatch:
    """Represents a batch of tokens for processing."""
    tokens: List[str] = field(default_factory=list)
    source: TokenSource = TokenSource.FALLBACK
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority = processed first


class EvaluationPipelineIntegration:
    """
    Robust integration layer between enhanced scanner and main trading evaluation.

    This class ensures tokens flow from scanning → evaluation → signals with:
    - Multiple fallback mechanisms
    - Comprehensive error handling
    - Real-time health monitoring
    - Performance optimization
    - Graceful degradation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_config = self._load_pipeline_config()
        self.enhanced_scanner = None
        self.metrics = PipelineMetrics()
        self.is_running = False
        self.last_health_check = 0.0
        self.token_cache: Dict[str, TokenBatch] = {}
        self.processing_lock = asyncio.Lock()
        
        # Exchange context for dynamic symbol discovery
        self.exchange = None
        self.exchange_symbols_cache = []
        self.symbols_cache_time = 0

        # Health monitoring
        self.status = PipelineStatus.OFFLINE
        self.health_monitor_task: Optional[asyncio.Task] = None

        logger.info("Evaluation Pipeline Integration initialized")

    def _load_pipeline_config(self) -> PipelineConfig:
        """Load pipeline configuration from main config."""
        pipeline_cfg = self.config.get("evaluation_pipeline", {})

        return PipelineConfig(
            enabled=pipeline_cfg.get("enabled", True),
            max_batch_size=pipeline_cfg.get("max_batch_size", 20),
            processing_timeout=pipeline_cfg.get("processing_timeout", 30.0),
            retry_attempts=pipeline_cfg.get("retry_attempts", 3),
            retry_delay=pipeline_cfg.get("retry_delay", 1.0),
            health_check_interval=pipeline_cfg.get("health_check_interval", 60.0),
            max_consecutive_failures=pipeline_cfg.get("max_consecutive_failures", 5),
            enable_fallback_sources=pipeline_cfg.get("enable_fallback_sources", True),
            cache_ttl=pipeline_cfg.get("cache_ttl", 300.0)
        )

    async def initialize(self) -> bool:
        """Initialize the pipeline integration."""
        try:
            logger.info("Initializing evaluation pipeline integration...")

            # Initialize enhanced scanner integration
            self.enhanced_scanner = get_enhanced_scan_integration(self.config)

            # Start the enhanced scanner integration if enabled
            try:
                if self.config.get("enhanced_scanning", {}).get("enabled", True):
                    await self.enhanced_scanner.start()
                    logger.info("Enhanced scanner integration started")
            except Exception as e:
                logger.warning(f"Failed to start enhanced scanner: {e}")

            # Validate scanner is working
            if not await self._validate_scanner():
                logger.warning("Enhanced scanner validation failed, will use fallback sources")
                if not self.pipeline_config.enable_fallback_sources:
                    logger.error("Fallback sources disabled and scanner failed - pipeline offline")
                    self.status = PipelineStatus.OFFLINE
                    return False

            self.status = PipelineStatus.HEALTHY
            logger.info("Evaluation pipeline integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize pipeline integration: {e}")
            self.status = PipelineStatus.CRITICAL
            return False

    async def _validate_scanner(self) -> bool:
        """Validate that the enhanced scanner is working."""
        try:
            if not self.enhanced_scanner:
                return False

            # Try to get scanner stats
            stats = self.enhanced_scanner.get_integration_stats()
            if not stats.get("running", False):
                logger.warning("Enhanced scanner is not running")
                return False

            logger.info("Enhanced scanner validation successful")
            return True

        except Exception as e:
            logger.error(f"Scanner validation failed: {e}")
            return False

    async def get_tokens_for_evaluation(self, max_tokens: int = 20) -> List[str]:
        """
        Get tokens for evaluation from all available sources with robust fallback chain.

        Priority order:
        1. Enhanced scanner (highest quality)
        2. Solana scanner (DEX tokens)
        3. Static config (user-defined)
        4. Hardcoded fallback (emergency)

        Returns tokens in priority order with comprehensive fallbacks.
        """
        async with self.processing_lock:
            try:
                start_time = time.time()
                all_tokens = []
                sources_used = []
                errors_encountered = []

                # Phase 1: Enhanced Scanner (Highest Priority)
                try:
                    scanner_tokens = await self._get_scanner_tokens(max_tokens)
                    if scanner_tokens:
                        all_tokens.extend(scanner_tokens)
                        sources_used.append(TokenSource.ENHANCED_SCANNER)
                        logger.info(f"✅ Got {len(scanner_tokens)} tokens from enhanced scanner")
                        self.metrics.last_successful_run = time.time()
                    else:
                        logger.warning("⚠️ Enhanced scanner returned no tokens")
                except Exception as e:
                    logger.error(f"❌ Enhanced scanner failed: {e}")
                    errors_encountered.append(f"scanner: {e}")

                # Phase 2: Solana Scanner (Secondary Priority)
                if len(all_tokens) < max_tokens:
                    try:
                        needed = max_tokens - len(all_tokens)
                        solana_tokens = await self._get_solana_tokens(needed)
                        if solana_tokens:
                            all_tokens.extend(solana_tokens)
                            sources_used.append(TokenSource.SOLANA_SCANNER)
                            logger.info(f"✅ Got {len(solana_tokens)} tokens from Solana scanner")
                        else:
                            logger.debug("ℹ️ Solana scanner returned no tokens")
                    except Exception as e:
                        logger.error(f"❌ Solana scanner failed: {e}")
                        errors_encountered.append(f"solana: {e}")

                # Phase 3: Static Config (Reliable Fallback)
                if len(all_tokens) < max_tokens:
                    try:
                        needed = max_tokens - len(all_tokens)
                        config_tokens = await self._get_config_tokens(needed)
                        if config_tokens:
                            all_tokens.extend(config_tokens)
                            sources_used.append(TokenSource.STATIC_CONFIG)
                            logger.info(f"✅ Got {len(config_tokens)} tokens from static config")
                        else:
                            logger.warning("⚠️ Static config returned no tokens")
                    except Exception as e:
                        logger.error(f"❌ Static config failed: {e}")
                        errors_encountered.append(f"config: {e}")

                # Phase 4: Hardcoded Fallback (Emergency Only)
                if len(all_tokens) == 0:
                    try:
                        logger.warning("🚨 All sources failed, using hardcoded fallback")
                        fallback_tokens = await self._get_fallback_tokens(max_tokens)
                        if fallback_tokens:
                            all_tokens.extend(fallback_tokens)
                            sources_used.append(TokenSource.FALLBACK)
                            logger.info(f"✅ Using {len(fallback_tokens)} fallback tokens")
                        else:
                            logger.error("❌ Even fallback tokens failed!")
                    except Exception as e:
                        logger.error(f"❌ Fallback tokens failed: {e}")
                        errors_encountered.append(f"fallback: {e}")

                # Validate and deduplicate tokens
                validated_tokens = await self._validate_and_deduplicate_tokens(all_tokens)

                # Update metrics
                processing_time = time.time() - start_time
                self.metrics.tokens_received += len(validated_tokens)
                self.metrics.tokens_processed += len(validated_tokens)
                self.metrics.avg_processing_time = (
                    (self.metrics.avg_processing_time + processing_time) / 2
                )

                # Update error metrics
                if errors_encountered:
                    self.metrics.error_rate = len(errors_encountered) / 4  # 4 phases
                    self.metrics.consecutive_failures += len(errors_encountered)
                else:
                    self.metrics.consecutive_failures = 0
                    self.metrics.error_rate = 0.0

                self._update_pipeline_status()

                # Log comprehensive results
                logger.info(
                    f"🎯 Pipeline completed in {processing_time:.2f}s: "
                    f"{len(validated_tokens)} tokens from sources: {[s.value for s in sources_used]}"
                )

                if errors_encountered:
                    logger.warning(f"⚠️ Pipeline encountered {len(errors_encountered)} errors: {errors_encountered}")

                # Cache successful results
                await self._cache_pipeline_results(validated_tokens, sources_used)

                return validated_tokens[:max_tokens]

            except Exception as e:
                logger.error(f"💥 Critical pipeline failure: {e}")
                self.metrics.consecutive_failures += 1
                self._update_pipeline_status()

                # Emergency fallback
                emergency_tokens = ["BTC/USD", "ETH/USD"]
                logger.error(f"🚨 Emergency fallback: returning {len(emergency_tokens)} tokens")
                return emergency_tokens

    async def _validate_and_deduplicate_tokens(self, tokens: List[str]) -> List[str]:
        """Validate token formats and remove duplicates while preserving priority order."""
        try:
            validated = []
            seen = set()

            for token in tokens:
                if not token or not isinstance(token, str):
                    continue

                # Basic format validation (should be SYMBOL/QUOTE)
                if '/' not in token:
                    logger.debug(f"Skipping invalid token format: {token}")
                    continue

                # Normalize token format
                normalized = token.strip().upper()

                # Skip duplicates
                if normalized in seen:
                    continue

                seen.add(normalized)
                validated.append(normalized)

            logger.debug(f"Validated {len(validated)} tokens from {len(tokens)} raw tokens")
            return validated

        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return tokens  # Return original on error

    async def _cache_pipeline_results(self, tokens: List[str], sources: List[TokenSource]):
        """Cache successful pipeline results for future reference."""
        try:
            cache_key = f"pipeline_{int(time.time())}"
            cache_entry = TokenBatch(
                tokens=tokens[:],
                source=sources[0] if sources else TokenSource.FALLBACK,
                timestamp=time.time(),
                metadata={
                    "sources": [s.value for s in sources],
                    "token_count": len(tokens)
                }
            )

            self.token_cache[cache_key] = cache_entry

            # Clean old cache entries
            current_time = time.time()
            to_remove = []
            for key, entry in self.token_cache.items():
                if (current_time - entry.timestamp) > self.pipeline_config.cache_ttl:
                    to_remove.append(key)

            for key in to_remove:
                del self.token_cache[key]

            if to_remove:
                logger.debug(f"Cleaned {len(to_remove)} expired cache entries")

        except Exception as e:
            logger.debug(f"Cache update failed: {e}")

    async def get_cached_results(self, max_age_seconds: float = 300.0) -> Optional[TokenBatch]:
        """Get most recent cached pipeline results."""
        try:
            current_time = time.time()
            valid_entries = [
                entry for entry in self.token_cache.values()
                if (current_time - entry.timestamp) <= max_age_seconds
            ]

            if not valid_entries:
                return None

            # Return most recent entry
            return max(valid_entries, key=lambda x: x.timestamp)

        except Exception as e:
            logger.debug(f"Cache retrieval failed: {e}")
            return None

    async def _get_scanner_tokens(self, max_tokens: int) -> List[str]:
        """Get CEX tokens from enhanced scanner (NEVER Solana tokens)."""
        try:
            if not self.enhanced_scanner:
                return []

            # Only get tokens from Kraken exchange - this is for CEX processing only
            kraken_tokens = await self._get_kraken_usd_symbols(max_tokens)
            if kraken_tokens:
                logger.info(f"Retrieved {len(kraken_tokens)} CEX USD symbols from Kraken")
                return kraken_tokens[:max_tokens]

            # DO NOT fallback to Solana scanner - that would mix Solana tokens into CEX processing
            logger.warning("Failed to get CEX tokens from Kraken - no Solana fallback allowed")
            return []

        except Exception as e:
            logger.error(f"Failed to get CEX scanner tokens: {e}")
            return []

    def set_exchange(self, exchange) -> None:
        """Set the exchange instance for dynamic symbol discovery."""
        self.exchange = exchange
        logger.info(f"Exchange instance set: {getattr(exchange, 'id', 'unknown')}")

    async def _get_kraken_usd_symbols(self, max_tokens: int) -> List[str]:
        """Get USD trading pairs from Kraken exchange."""
        try:
            # Try to get symbols from live exchange first
            if self.exchange and hasattr(self.exchange, 'symbols'):
                # Use cached symbols if available and fresh (within 1 hour)
                current_time = time.time()
                if (self.exchange_symbols_cache and 
                    (current_time - self.symbols_cache_time) < 3600):
                    logger.debug("Using cached exchange symbols")
                    return self.exchange_symbols_cache[:max_tokens]
                
                # Get fresh symbols from exchange
                try:
                    if not self.exchange.symbols and hasattr(self.exchange, 'load_markets'):
                        logger.info("Loading markets from exchange...")
                        if asyncio.iscoroutinefunction(self.exchange.load_markets):
                            await self.exchange.load_markets()
                        else:
                            await asyncio.to_thread(self.exchange.load_markets)
                    
                    if self.exchange.symbols:
                        usd_symbols = [s for s in self.exchange.symbols if s.endswith('/USD')]
                        logger.info(f"Found {len(usd_symbols)} USD pairs from live exchange")
                        
                        # Cache the results
                        self.exchange_symbols_cache = usd_symbols
                        self.symbols_cache_time = current_time
                        
                        return usd_symbols[:max_tokens]
                        
                except Exception as e:
                    logger.warning(f"Failed to get symbols from live exchange: {e}")
            
            # Fallback to comprehensive hardcoded list of known Kraken USD pairs
            kraken_usd_symbols = [
                # Major cryptocurrencies
                "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
                "LINK/USD", "UNI/USD", "AAVE/USD", "AVAX/USD", "MATIC/USD",
                "ATOM/USD", "NEAR/USD", "ALGO/USD", "XTZ/USD", "MANA/USD",
                
                # Additional popular tokens on Kraken
                "XRP/USD", "LTC/USD", "BCH/USD", "ETC/USD", "ZEC/USD",
                "DASH/USD", "XMR/USD", "EOS/USD", "TRX/USD", "XLM/USD",
                "DOGE/USD", "SHIB/USD", "FIL/USD", "GRT/USD", "CRV/USD",
                
                # DeFi tokens
                "COMP/USD", "MKR/USD", "SNX/USD", "YFI/USD", "SUSHI/USD",
                "1INCH/USD", "BAL/USD", "KNC/USD", "REN/USD", "LRC/USD",
                
                # Layer 1 & 2 tokens  
                "FTM/USD", "LUNA/USD", "WAVES/USD", "ICX/USD", "ONT/USD",
                "QTUM/USD", "ZIL/USD", "BAT/USD", "ENJ/USD", "CHZ/USD",
                
                # Additional tokens
                "SAND/USD", "AXS/USD", "GALA/USD", "IMX/USD", "LPT/USD",
                "STORJ/USD", "ANKR/USD", "BAND/USD", "NU/USD", "OGN/USD",
                "NMR/USD", "KEEP/USD", "T/USD", "TBTC/USD", "BADGER/USD",
                
                # More Kraken pairs (80+ symbols total)
                "REP/USD", "KSM/USD", "FLOW/USD", "SC/USD", "KAVA/USD",
                "LSK/USD", "NANO/USD", "OMG/USD", "OXT/USD", "PAXG/USD",
                "PERP/USD", "PHA/USD", "POND/USD", "RAD/USD", "RARI/USD",
                "REPV2/USD", "ROOK/USD", "SAMO/USD", "SDN/USD", "SGB/USD",
                "SKL/USD", "SRM/USD", "STEP/USD", "STG/USD", "SUI/USD",
                "SUPER/USD", "TEER/USD", "TIA/USD", "TRU/USD", "UMA/USD", 
                "UNFI/USD", "WBTC/USD", "WETH/USD", "WSTETH/USD", "XCN/USD"
            ]
            
            # Remove duplicates and filter out invalid symbols
            valid_symbols = []
            seen = set()
            
            for symbol in kraken_usd_symbols:
                if symbol not in seen and '/' in symbol and symbol.endswith('/USD'):
                    valid_symbols.append(symbol)
                    seen.add(symbol)
            
            logger.info(f"Using {len(valid_symbols)} known Kraken USD symbols (fallback)")
            return valid_symbols[:max_tokens]
            
        except Exception as e:
            logger.error(f"Failed to get Kraken USD symbols: {e}")
            return []

    async def _get_solana_tokens(self, max_tokens: int) -> List[str]:
        """Get tokens from Solana scanner."""
        try:
            sol_cfg = self.config.get("solana_scanner", {})
            if not sol_cfg.get("enabled", False):
                return []

            # Import here to avoid circular imports
            from .utils.solana_scanner import get_solana_new_tokens

            solana_tokens = await get_solana_new_tokens(sol_cfg)
            return list(solana_tokens)[:max_tokens]

        except Exception as e:
            logger.error(f"Failed to get Solana tokens: {e}")
            return []

    async def _get_config_tokens(self, max_tokens: int) -> List[str]:
        """Get tokens from static configuration."""
        try:
            symbols = self.config.get("symbols", [])
            if not symbols:
                return []

            # Return symbols that aren't already cached recently
            current_time = time.time()
            filtered_symbols = []

            for symbol in symbols:
                cache_key = f"config_{symbol}"
                if cache_key not in self.token_cache:
                    filtered_symbols.append(symbol)
                elif (current_time - self.token_cache[cache_key].timestamp) > self.pipeline_config.cache_ttl:
                    filtered_symbols.append(symbol)

            return filtered_symbols[:max_tokens]

        except Exception as e:
            logger.error(f"Failed to get config tokens: {e}")
            return []

    async def _get_fallback_tokens(self, max_tokens: int) -> List[str]:
        """Get hardcoded fallback tokens."""
        fallback_tokens = [
            "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
            "LINK/USD", "UNI/USD", "AVAX/USD", "MATIC/USD", "ATOM/USD"
        ]
        return fallback_tokens[:max_tokens]

    def _update_pipeline_status(self):
        """Update pipeline status based on metrics."""
        if self.metrics.consecutive_failures >= self.pipeline_config.max_consecutive_failures:
            self.status = PipelineStatus.CRITICAL
        elif self.metrics.error_rate > 0.5:
            self.status = PipelineStatus.DEGRADED
        elif self.metrics.last_successful_run and (time.time() - self.metrics.last_successful_run) < 300:
            self.status = PipelineStatus.HEALTHY
        else:
            self.status = PipelineStatus.DEGRADED

    async def start_health_monitoring(self):
        """Start the health monitoring loop."""
        if self.health_monitor_task and not self.health_monitor_task.done():
            return

        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Health monitoring started")

    async def stop_health_monitoring(self):
        """Stop the health monitoring loop."""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Health monitoring stopped")

    async def _health_monitor_loop(self):
        """Health monitoring loop."""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.pipeline_config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(30)  # Retry sooner on error

    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            self.last_health_check = time.time()

            # Check scanner health
            scanner_healthy = await self._validate_scanner()

            # Check token sources
            test_tokens = await self.get_tokens_for_evaluation(5)

            # Update status
            if scanner_healthy and len(test_tokens) > 0:
                self.status = PipelineStatus.HEALTHY
                self.metrics.last_successful_run = time.time()
                self.metrics.consecutive_failures = 0
            elif len(test_tokens) > 0:
                self.status = PipelineStatus.DEGRADED
            else:
                self.status = PipelineStatus.CRITICAL
                self.metrics.consecutive_failures += 1

            logger.debug(f"Health check completed. Status: {self.status.value}, Tokens: {len(test_tokens)}")

        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.status = PipelineStatus.CRITICAL

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            "status": self.status.value,
            "metrics": {
                "tokens_received": self.metrics.tokens_received,
                "tokens_processed": self.metrics.tokens_processed,
                "tokens_failed": self.metrics.tokens_failed,
                "avg_processing_time": round(self.metrics.avg_processing_time, 3),
                "error_rate": round(self.metrics.error_rate, 3),
                "consecutive_failures": self.metrics.consecutive_failures,
                "last_successful_run": self.metrics.last_successful_run,
                "total_runtime": round(self.metrics.total_runtime, 1)
            },
            "config": {
                "enabled": self.pipeline_config.enabled,
                "max_batch_size": self.pipeline_config.max_batch_size,
                "processing_timeout": self.pipeline_config.processing_timeout,
                "enable_fallback_sources": self.pipeline_config.enable_fallback_sources
            },
            "scanner": {
                "available": self.enhanced_scanner is not None,
                "healthy": self.status in [PipelineStatus.HEALTHY, PipelineStatus.DEGRADED]
            },
            "last_health_check": self.last_health_check
        }

    async def force_refresh_cache(self):
        """Force refresh of all token caches."""
        try:
            logger.info("Forcing cache refresh...")

            # Clear local cache
            self.token_cache.clear()

            # Force scanner refresh if available
            if self.enhanced_scanner and hasattr(self.enhanced_scanner, 'force_scan'):
                await self.enhanced_scanner.force_scan()

            logger.info("Cache refresh completed")

        except Exception as e:
            logger.error(f"Cache refresh failed: {e}")

    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_health_monitoring()
        
        # Stop enhanced scanner integration
        if self.enhanced_scanner and hasattr(self.enhanced_scanner, 'stop'):
            try:
                await self.enhanced_scanner.stop()
                logger.info("Enhanced scanner integration stopped")
            except Exception as e:
                logger.warning(f"Error stopping enhanced scanner: {e}")
        
        self.token_cache.clear()
        logger.info("Pipeline integration cleanup completed")


# Global instance
_pipeline_integration: Optional[EvaluationPipelineIntegration] = None


def get_evaluation_pipeline_integration(config: Dict[str, Any]) -> EvaluationPipelineIntegration:
    """Get or create the global pipeline integration instance."""
    global _pipeline_integration

    if _pipeline_integration is None:
        _pipeline_integration = EvaluationPipelineIntegration(config)

    return _pipeline_integration


async def initialize_evaluation_pipeline(config: Dict[str, Any]) -> bool:
    """Initialize the evaluation pipeline integration."""
    integration = get_evaluation_pipeline_integration(config)
    return await integration.initialize()


async def get_tokens_for_evaluation(config: Dict[str, Any], max_tokens: int = 20, exchange=None) -> List[str]:
    """Get tokens for evaluation with full pipeline integration."""
    integration = get_evaluation_pipeline_integration(config)
    
    # Set exchange instance if provided for dynamic symbol discovery
    if exchange:
        integration.set_exchange(exchange)
    
    return await integration.get_tokens_for_evaluation(max_tokens)


def get_pipeline_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get pipeline status."""
    integration = get_evaluation_pipeline_integration(config)
    return integration.get_pipeline_status()
