"""
Enhanced Scan Integration Module

This module integrates the enhanced scanning system with the existing main bot
infrastructure, providing seamless integration of scan caching, strategy fit
analysis, and execution opportunity detection.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import yaml

from .utils.scan_cache_manager import get_scan_cache_manager, ScanResult
from .utils.logger import setup_logger
from .solana.enhanced_scanner import get_enhanced_scanner, start_enhanced_scanner, stop_enhanced_scanner
from .utils.telegram import TelegramNotifier

logger = setup_logger(__name__)


class EnhancedScanIntegration:
    """
    Integrates enhanced scanning with the main bot infrastructure.
    
    Features:
    - Automatic scan result caching
    - Integration with existing strategy analysis
    - Execution opportunity detection
    - Performance monitoring and reporting
    """
    
    def __init__(self, config: Dict[str, Any], notifier: Optional[TelegramNotifier] = None):
        self.config = config
        self.notifier = notifier
        
        # Load enhanced scanning config
        self.enhanced_config = self._load_enhanced_config()
        
        # Initialize components
        self.cache_manager = get_scan_cache_manager(self.enhanced_config)
        self.enhanced_scanner = get_enhanced_scanner(self.enhanced_config)
        
        # Integration settings
        self.integration_enabled = self.enhanced_config.get("integration", {}).get("enable_bot_integration", True)
        self.strategy_integration = self.enhanced_config.get("integration", {}).get("enable_strategy_router_integration", True)
        self.risk_integration = self.enhanced_config.get("integration", {}).get("enable_risk_manager_integration", True)
        
        # Performance tracking
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "strategy_analyses": 0,
            "execution_opportunities": 0,
            "integration_errors": 0
        }
        
        # Background tasks
        self.integration_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("Enhanced scan integration initialized")
    
    def _load_enhanced_config(self) -> Dict[str, Any]:
        """Load enhanced scanning configuration."""
        try:
            config_path = Path(__file__).resolve().parent.parent / "config" / "enhanced_scanning.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info("Loaded enhanced scanning configuration")
                return config
            else:
                logger.warning("Enhanced scanning config not found, using defaults")
                return self._get_default_config()
        except Exception as exc:
            logger.error(f"Failed to load enhanced scanning config: {exc}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found."""
        return {
            "scan_cache": {
                "max_cache_size": 1000,
                "review_interval_minutes": 15,
                "max_age_hours": 24,
                "min_score_threshold": 0.3
            },
            "solana_scanner": {
                "enabled": True,
                "scan_interval_minutes": 30,
                "max_tokens_per_scan": 100,
                "min_score_threshold": 0.3
            },
            "integration": {
                "enable_bot_integration": True,
                "enable_strategy_router_integration": True,
                "enable_risk_manager_integration": True
            }
        }
    
    async def start(self):
        """Start the enhanced scan integration."""
        if self.running:
            return
        
        try:
            # Start enhanced scanner
            await start_enhanced_scanner(self.enhanced_config)
            
            # Start background tasks
            self.running = True
            self.integration_task = asyncio.create_task(self._integration_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Enhanced scan integration started")
            
            if self.notifier:
                self.notifier.notify("🚀 Enhanced scan integration started")
                
        except Exception as exc:
            logger.error(f"Failed to start enhanced scan integration: {exc}")
            raise
    
    async def stop(self):
        """Stop the enhanced scan integration."""
        if not self.running:
            return
        
        try:
            # Stop background tasks
            self.running = False
            
            if self.integration_task:
                self.integration_task.cancel()
                try:
                    await self.integration_task
                except asyncio.CancelledError:
                    pass
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Stop enhanced scanner
            await stop_enhanced_scanner()
            
            logger.info("Enhanced scan integration stopped")
            
            if self.notifier:
                self.notifier.notify("🛑 Enhanced scan integration stopped")
                
        except Exception as exc:
            logger.error(f"Error stopping enhanced scan integration: {exc}")
    
    async def _integration_loop(self):
        """Main integration loop for processing cached scan results."""
        while self.running:
            try:
                await self._process_cached_results()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in integration loop: {exc}")
                self.performance_stats["integration_errors"] += 1
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _monitoring_loop(self):
        """Monitoring loop for performance tracking and reporting."""
        while self.running:
            try:
                await self._generate_performance_report()
                await asyncio.sleep(300)  # Report every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in monitoring loop: {exc}")
                await asyncio.sleep(60)
    
    async def _process_cached_results(self):
        """Process cached scan results for strategy analysis and execution."""
        try:
            # Get execution opportunities
            opportunities = self.cache_manager.get_execution_opportunities(
                min_confidence=0.7
            )
            
            if not opportunities:
                return
            
            logger.info(f"Processing {len(opportunities)} execution opportunities")
            
            for opportunity in opportunities:
                try:
                    await self._process_execution_opportunity(opportunity)
                except Exception as exc:
                    logger.error(f"Failed to process opportunity {opportunity.symbol}: {exc}")
            
            self.performance_stats["execution_opportunities"] += len(opportunities)
            
        except Exception as exc:
            logger.error(f"Failed to process cached results: {exc}")
    
    async def _process_execution_opportunity(self, opportunity):
        """Process a single execution opportunity."""
        try:
            # Check if opportunity is still valid
            if not self._is_opportunity_valid(opportunity):
                return
            
            # Get current market data
            current_data = await self._get_current_market_data(opportunity.symbol)
            if not current_data:
                return
            
            # Validate opportunity against current conditions
            if not self._validate_opportunity(opportunity, current_data):
                return
            
            # Check risk management
            if not self._check_risk_management(opportunity):
                return
            
            # Execute or queue for execution
            await self._handle_execution(opportunity)
            
        except Exception as exc:
            logger.error(f"Failed to process opportunity {opportunity.symbol}: {exc}")
    
    def _is_opportunity_valid(self, opportunity) -> bool:
        """Check if an execution opportunity is still valid."""
        # Check age
        max_age_hours = 2  # Opportunities expire after 2 hours
        age_hours = (time.time() - opportunity.timestamp) / 3600
        
        if age_hours > max_age_hours:
            return False
        
        # Check status
        if opportunity.status != "pending":
            return False
        
        return True
    
    async def _get_current_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol."""
        try:
            # Try to get from cache first
            scan_result = self.cache_manager.get_scan_result(symbol)
            if scan_result:
                self.performance_stats["cache_hits"] += 1
                return scan_result.data
            
            self.performance_stats["cache_misses"] += 1
            
            # Fallback to real-time data (would integrate with your data sources)
            # For now, return None to indicate no data available
            return None
            
        except Exception as exc:
            logger.debug(f"Failed to get market data for {symbol}: {exc}")
            return None
    
    def _validate_opportunity(self, opportunity, current_data: Dict[str, Any]) -> bool:
        """Validate opportunity against current market conditions."""
        try:
            # Check price deviation
            current_price = current_data.get("price", 0)
            if not current_price:
                return False
            
            price_deviation = abs(current_price - opportunity.entry_price) / opportunity.entry_price
            if price_deviation > 0.05:  # 5% price deviation threshold
                return False
            
            # Check volume conditions
            current_volume = current_data.get("volume", 0)
            if current_volume < 10000:  # Minimum volume threshold
                return False
            
            # Check spread conditions
            current_spread = current_data.get("spread_pct", 0)
            if current_spread > 1.0:  # Maximum spread threshold
                return False
            
            return True
            
        except Exception as exc:
            logger.debug(f"Failed to validate opportunity: {exc}")
            return False
    
    def _check_risk_management(self, opportunity) -> bool:
        """Check risk management constraints."""
        try:
            # Check risk/reward ratio
            if opportunity.risk_reward_ratio < 2.0:
                return False
            
            # Check position size
            max_position_size = 0.1  # 10% of account
            if opportunity.position_size > max_position_size:
                return False
            
            return True
            
        except Exception as exc:
            logger.debug(f"Failed to check risk management: {exc}")
            return False
    
    async def _handle_execution(self, opportunity):
        """Handle execution of a validated opportunity."""
        try:
            # Mark as attempted
            self.cache_manager.mark_execution_attempted(opportunity.symbol)
            
            # Log opportunity details
            logger.info(
                f"Execution opportunity: {opportunity.symbol} | "
                f"Strategy: {opportunity.strategy} | "
                f"Direction: {opportunity.direction} | "
                f"Confidence: {opportunity.confidence:.3f} | "
                f"R/R: {opportunity.risk_reward_ratio:.2f}"
            )
            
            # Send notification
            if self.notifier:
                self.notifier.notify(
                    f"🎯 Execution Opportunity: {opportunity.symbol}\n"
                    f"Strategy: {opportunity.strategy}\n"
                    f"Direction: {opportunity.direction.upper()}\n"
                    f"Confidence: {opportunity.confidence:.1%}\n"
                    f"Risk/Reward: {opportunity.risk_reward_ratio:.2f}"
                )
            
            # Here you would integrate with your execution engine
            # For now, we'll just mark it as processed
            # await self._execute_trade(opportunity)
            
        except Exception as exc:
            logger.error(f"Failed to handle execution for {opportunity.symbol}: {exc}")
    
    async def _generate_performance_report(self):
        """Generate and log performance report."""
        try:
            # Get cache statistics
            cache_stats = self.cache_manager.get_cache_stats()
            
            # Get scanner statistics
            scanner_stats = self.enhanced_scanner.get_scan_stats()
            
            # Calculate cache hit rate
            total_cache_access = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
            cache_hit_rate = (self.performance_stats["cache_hits"] / total_cache_access * 100) if total_cache_access > 0 else 0
            
            # Generate report
            report = {
                "timestamp": time.time(),
                "cache_stats": cache_stats,
                "scanner_stats": scanner_stats,
                "performance_stats": self.performance_stats.copy(),
                "cache_hit_rate": cache_hit_rate
            }
            
            # Log report
            logger.info(
                f"Performance Report: "
                f"Cache: {cache_stats['scan_results']} results, "
                f"Scanner: {scanner_stats['total_scans']} scans, "
                f"Opportunities: {self.performance_stats['execution_opportunities']}, "
                f"Cache Hit Rate: {cache_hit_rate:.1f}%"
            )
            
            # Send periodic notification
            if self.notifier and self.performance_stats["execution_opportunities"] > 0:
                self.notifier.notify(
                    f"📊 Scan Performance Update:\n"
                    f"Cache: {cache_stats['scan_results']} results\n"
                    f"Scans: {scanner_stats['total_scans']}\n"
                    f"Opportunities: {self.performance_stats['execution_opportunities']}\n"
                    f"Cache Hit Rate: {cache_hit_rate:.1f}%"
                )
            
            # Reset counters
            self.performance_stats["execution_opportunities"] = 0
            
        except Exception as exc:
            logger.error(f"Failed to generate performance report: {exc}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "running": self.running,
            "performance_stats": self.performance_stats.copy(),
            "cache_stats": self.cache_manager.get_cache_stats(),
            "scanner_stats": self.enhanced_scanner.get_scan_stats()
        }
    
    def get_top_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top execution opportunities."""
        try:
            return self.cache_manager.get_execution_opportunities(min_confidence=0.7)[:limit]
        except Exception as exc:
            logger.error(f"Failed to get top opportunities: {exc}")
            return []
    
    async def force_scan(self):
        """Force an immediate scan cycle."""
        try:
            if self.enhanced_scanner:
                await self.enhanced_scanner._perform_scan()
                logger.info("Forced scan cycle completed")
                
                if self.notifier:
                    self.notifier.notify("🔍 Forced scan cycle completed")
                    
        except Exception as exc:
            logger.error(f"Forced scan failed: {exc}")
    
    async def clear_cache(self):
        """Clear all caches."""
        try:
            self.cache_manager.clear_cache()
            logger.info("All caches cleared")
            
            if self.notifier:
                self.notifier.notify("🗑️ All scan caches cleared")
                
        except Exception as exc:
            logger.error(f"Failed to clear cache: {exc}")


# Global integration instance
_enhanced_integration: Optional[EnhancedScanIntegration] = None


def get_enhanced_scan_integration(config: Dict[str, Any], notifier: Optional[TelegramNotifier] = None) -> EnhancedScanIntegration:
    """Get or create the global enhanced scan integration instance."""
    global _enhanced_integration
    
    if _enhanced_integration is None:
        _enhanced_integration = EnhancedScanIntegration(config, notifier)
    
    return _enhanced_integration


async def start_enhanced_scan_integration(config: Dict[str, Any], notifier: Optional[TelegramNotifier] = None):
    """Start the enhanced scan integration."""
    integration = get_enhanced_scan_integration(config, notifier)
    await integration.start()


async def stop_enhanced_scan_integration():
    """Stop the enhanced scan integration."""
    if _enhanced_integration:
        await _enhanced_integration.stop()


def get_integration_stats() -> Dict[str, Any]:
    """Get integration statistics if available."""
    if _enhanced_integration:
        return _enhanced_integration.get_integration_stats()
    return {"error": "Enhanced scan integration not initialized"}
