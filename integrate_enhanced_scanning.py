#!/usr/bin/env python3
"""
Integrate Enhanced Scanning with Main Bot

This script integrates the enhanced scanning system with the main bot
to enable comprehensive token analysis and discovery.
"""

import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

def integrate_enhanced_scanning():
    """Integrate enhanced scanning with the main bot."""
    
    print("🔗 Integrating enhanced scanning with main bot...")
    
    # 1. First, let's check if the enhanced scanning integration exists
    integration_file = Path("crypto_bot/enhanced_scan_integration.py")
    if not integration_file.exists():
        print("❌ Enhanced scan integration not found")
        print("📋 Creating enhanced scan integration module...")
        create_enhanced_scan_integration()
    
    # 2. Update main.py to import and use enhanced scanning
    main_file = Path("crypto_bot/main.py")
    if main_file.exists():
        print("📝 Updating main.py to integrate enhanced scanning...")
        integrate_main_with_enhanced_scanning()
    
    # 3. Create enhanced scanning configuration
    print("⚙️ Creating enhanced scanning configuration...")
    create_enhanced_scanning_config()
    
    # 4. Create monitoring dashboard for scanning
    print("📊 Creating scanning monitoring dashboard...")
    create_scanning_dashboard()
    
    print("✅ Enhanced scanning integration completed!")

def create_enhanced_scan_integration():
    """Create the enhanced scan integration module."""
    
    integration_code = '''"""
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
from .utils.logger import setup_logger, LOG_DIR
from .solana.enhanced_scanner import get_enhanced_scanner, start_enhanced_scanner, stop_enhanced_scanner
from .utils.telegram import TelegramNotifier
from .utils.logger import LOG_DIR

logger = setup_logger(__name__, LOG_DIR / "enhanced_scan_integration.log")


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
        
        # Test compatibility attributes
        self.enabled = self.enhanced_config.get("enhanced_scanning", {}).get("enabled", True)
        self.scan_interval = self.enhanced_config.get("enhanced_scanning", {}).get("scan_interval", 30)
        
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
        """Get default enhanced scanning configuration."""
        return {
            "enhanced_scanning": {
                "enabled": True,
                "scan_interval": 30,
                "max_tokens_per_scan": 20,
                "min_score_threshold": 0.4,
                "enable_sentiment": False,
                "enable_pyth_prices": True,
                "min_volume_usd": 5000,
                "max_spread_pct": 1.5,
                "min_liquidity_score": 0.6,
                "min_strategy_fit": 0.6,
                "min_confidence": 0.5
            },
            "scan_cache": {
                "max_cache_size": 500,
                "review_interval_minutes": 20,
                "max_age_hours": 12,
                "min_score_threshold": 0.4,
                "persist_to_disk": True,
                "auto_cleanup": True
            },
            "integration": {
                "enable_bot_integration": True,
                "enable_strategy_router_integration": True,
                "enable_risk_manager_integration": True,
                "scan_interval": 30,
                "max_concurrent_scans": 3
            }
        }
    
    async def start(self):
        """Start the enhanced scanning integration."""
        if self.running:
            logger.warning("Enhanced scan integration already running")
            return
        
        try:
            # Start enhanced scanner
            await start_enhanced_scanner(self.enhanced_config)
            
            # Start background tasks
            self.integration_task = asyncio.create_task(self._integration_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.running = True
            logger.info("Enhanced scan integration started")
            
            if self.notifier:
                await self.notifier.notify("🔍 Enhanced scanning integration started")
                
        except Exception as exc:
            logger.error(f"Failed to start enhanced scan integration: {exc}")
            raise
    
    async def stop(self):
        """Stop the enhanced scanning integration."""
        if not self.running:
            return
        
        try:
            # Stop background tasks
            if self.integration_task:
                self.integration_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # Stop enhanced scanner
            await stop_enhanced_scanner()
            
            self.running = False
            logger.info("Enhanced scan integration stopped")
            
            if self.notifier:
                await self.notifier.notify("🛑 Enhanced scanning integration stopped")
                
        except Exception as exc:
            logger.error(f"Failed to stop enhanced scan integration: {exc}")
    
    async def _integration_loop(self):
        """Main integration loop."""
        while self.running:
            try:
                # Get recent scan results
                recent_results = self.cache_manager.get_recent_results(limit=10)
                
                # Process results for strategy analysis
                for result in recent_results:
                    await self._process_scan_result(result)
                
                # Wait for next cycle
                await asyncio.sleep(self.scan_interval * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Integration loop error: {exc}")
                self.performance_stats["integration_errors"] += 1
                await asyncio.sleep(60)  # Wait before retry
    
    async def _monitoring_loop(self):
        """Monitoring and reporting loop."""
        while self.running:
            try:
                # Log performance stats
                logger.info(f"Enhanced scan stats: {self.performance_stats}")
                
                # Send notifications if configured
                if self.notifier and self.performance_stats["execution_opportunities"] > 0:
                    await self.notifier.notify(
                        f"🎯 Found {self.performance_stats['execution_opportunities']} execution opportunities"
                    )
                
                # Reset counters
                self.performance_stats["execution_opportunities"] = 0
                
                # Wait for next report
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Monitoring loop error: {exc}")
                await asyncio.sleep(60)
    
    async def _process_scan_result(self, result: ScanResult):
        """Process a scan result for strategy analysis."""
        try:
            # Check if result meets strategy criteria
            if result.score >= self.enhanced_config["enhanced_scanning"]["min_strategy_fit"]:
                self.performance_stats["strategy_analyses"] += 1
                
                # Check for execution opportunity
                if result.confidence >= self.enhanced_config["enhanced_scanning"]["min_confidence"]:
                    self.performance_stats["execution_opportunities"] += 1
                    logger.info(f"Execution opportunity found: {result.symbol}")
                    
                    if self.notifier:
                        await self.notifier.notify(
                            f"🚀 Execution opportunity: {result.symbol} (score: {result.score:.2f})"
                        )
            
        except Exception as exc:
            logger.error(f"Failed to process scan result: {exc}")
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scanning statistics."""
        return {
            "enhanced_scanner_stats": self.enhanced_scanner.get_scan_stats() if self.enhanced_scanner else {},
            "integration_stats": self.performance_stats,
            "cache_stats": self.cache_manager.get_stats() if self.cache_manager else {},
            "running": self.running
        }
    
    async def perform_manual_scan(self) -> List[str]:
        """Perform a manual scan."""
        try:
            if self.enhanced_scanner:
                return await self.enhanced_scanner._perform_scan()
            else:
                logger.warning("Enhanced scanner not available")
                return []
        except Exception as exc:
            logger.error(f"Manual scan failed: {exc}")
            return []

# Global instance
_enhanced_scan_integration: Optional[EnhancedScanIntegration] = None

def get_enhanced_scan_integration(config: Dict[str, Any], notifier: Optional[TelegramNotifier] = None) -> EnhancedScanIntegration:
    """Get or create the enhanced scan integration instance."""
    global _enhanced_scan_integration
    
    if _enhanced_scan_integration is None:
        _enhanced_scan_integration = EnhancedScanIntegration(config, notifier)
    
    return _enhanced_scan_integration

async def start_enhanced_scan_integration(config: Dict[str, Any], notifier: Optional[TelegramNotifier] = None):
    """Start the enhanced scan integration."""
    integration = get_enhanced_scan_integration(config, notifier)
    await integration.start()

async def stop_enhanced_scan_integration():
    """Stop the enhanced scan integration."""
    global _enhanced_scan_integration
    
    if _enhanced_scan_integration:
        await _enhanced_scan_integration.stop()
        _enhanced_scan_integration = None
'''
    
    with open("crypto_bot/enhanced_scan_integration.py", 'w') as f:
        f.write(integration_code)
    
    print("✅ Created enhanced scan integration module")

def integrate_main_with_enhanced_scanning():
    """Integrate enhanced scanning with main.py."""
    
    main_file = Path("crypto_bot/main.py")
    
    # Read the current main.py
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Add import for enhanced scanning
    import_line = "from crypto_bot.enhanced_scan_integration import get_enhanced_scan_integration, start_enhanced_scan_integration, stop_enhanced_scan_integration"
    
    if "enhanced_scan_integration" not in content:
        # Find the import section and add our import
        lines = content.split('\n')
        import_section_end = 0
        
        for i, line in enumerate(lines):
            if line.startswith('from crypto_bot.utils.telemetry'):
                import_section_end = i + 1
                break
        
        # Insert our import
        lines.insert(import_section_end, import_line)
        content = '\n'.join(lines)
    
    # Add enhanced scanning initialization in the main function
    if "enhanced_scan_integration" not in content:
        # Find where to add enhanced scanning initialization
        lines = content.split('\n')
        
        # Look for the main function
        main_func_start = -1
        for i, line in enumerate(lines):
            if "async def main(" in line:
                main_func_start = i
                break
        
        if main_func_start != -1:
            # Find where to add enhanced scanning (after other initializations)
            init_section = -1
            for i in range(main_func_start, len(lines)):
                if "notifier = TelegramNotifier" in lines[i]:
                    init_section = i + 1
                    break
            
            if init_section != -1:
                # Add enhanced scanning initialization
                enhanced_init = '''    # Initialize enhanced scanning integration
    enhanced_scan_integration = None
    if config.get("enhanced_scanning", {}).get("enabled", False):
        try:
            enhanced_scan_integration = get_enhanced_scan_integration(config, notifier)
            await start_enhanced_scan_integration(config, notifier)
            logger.info("Enhanced scanning integration started")
        except Exception as exc:
            logger.error(f"Failed to start enhanced scanning integration: {exc}")
            enhanced_scan_integration = None'''
                
                lines.insert(init_section, enhanced_init)
                
                # Add cleanup in the finally block
                cleanup_section = -1
                for i in range(len(lines)):
                    if "finally:" in lines[i]:
                        cleanup_section = i + 1
                        break
                
                if cleanup_section != -1:
                    enhanced_cleanup = '''        # Stop enhanced scanning integration
        if enhanced_scan_integration:
            try:
                await stop_enhanced_scan_integration()
                logger.info("Enhanced scanning integration stopped")
            except Exception as exc:
                logger.error(f"Failed to stop enhanced scanning integration: {exc}")'''
                    
                    lines.insert(cleanup_section, enhanced_cleanup)
                
                content = '\n'.join(lines)
    
    # Write the updated main.py
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("✅ Integrated enhanced scanning with main.py")

def create_enhanced_scanning_config():
    """Create enhanced scanning configuration file."""
    
    config_content = '''# Enhanced Scanning Configuration
# This file configures the enhanced scanning system with integrated caching
# and continuous strategy review for trade execution opportunities.

# Enhanced Scanning Configuration
enhanced_scanning:
  enabled: true
  scan_interval: 30
  max_tokens_per_scan: 20
  min_score_threshold: 0.4
  enable_sentiment: false
  enable_pyth_prices: true
  min_volume_usd: 5000
  max_spread_pct: 1.5
  min_liquidity_score: 0.6
  min_strategy_fit: 0.6
  min_confidence: 0.5
  discovery_sources:
    - "basic_scanner"
    - "dex_aggregators"
  data_sources:
    price:
      - "pyth"
      - "jupiter"
    volume:
      - "birdeye"
    orderbook:
      - "jupiter"

# Scan Cache Configuration
scan_cache:
  max_cache_size: 500
  review_interval_minutes: 20
  max_age_hours: 12
  min_score_threshold: 0.4
  persist_to_disk: true
  auto_cleanup: true

# Integration Configuration
integration:
  enable_bot_integration: true
  enable_strategy_router_integration: true
  enable_risk_manager_integration: true
  scan_interval: 30
  max_concurrent_scans: 3

# Monitoring Configuration
monitoring:
  enable_scan_metrics: true
  log_scan_results: true
  track_scan_performance: true
  alert_on_scan_failures: true
  scan_metrics_interval: 60
  max_scan_logs: 1000
'''
    
    config_path = Path("config/enhanced_scanning.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print("✅ Created enhanced scanning configuration")

def create_scanning_dashboard():
    """Create a scanning monitoring dashboard."""
    
    dashboard_code = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Scanning Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .content {
            padding: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .scan-results {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .scan-results h3 {
            margin-top: 0;
            color: #333;
        }
        .token-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .token-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .token-item:last-child {
            border-bottom: none;
        }
        .token-symbol {
            font-weight: bold;
            color: #667eea;
        }
        .token-score {
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .refresh-btn:hover {
            background: #5a6fd8;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-active {
            background: #28a745;
        }
        .status-inactive {
            background: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Enhanced Scanning Dashboard</h1>
            <p>Real-time monitoring of token discovery and analysis</p>
        </div>
        
        <div class="content">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">Scanner Status</div>
                    <div class="stat-value">
                        <span class="status-indicator status-active" id="scanner-status"></span>
                        <span id="scanner-status-text">Active</span>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Total Scans</div>
                    <div class="stat-value" id="total-scans">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Tokens Discovered</div>
                    <div class="stat-value" id="tokens-discovered">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Execution Opportunities</div>
                    <div class="stat-value" id="execution-opportunities">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Cache Hit Rate</div>
                    <div class="stat-value" id="cache-hit-rate">0%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Last Scan</div>
                    <div class="stat-value" id="last-scan">Never</div>
                </div>
            </div>
            
            <div class="scan-results">
                <h3>Recent Token Discoveries</h3>
                <div class="token-list" id="token-list">
                    <div style="text-align: center; color: #666; padding: 20px;">
                        No tokens discovered yet...
                    </div>
                </div>
            </div>
            
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
        </div>
    </div>

    <script>
        async function refreshData() {
            try {
                const response = await fetch('/api/scanning/stats');
                const data = await response.json();
                
                // Update stats
                document.getElementById('total-scans').textContent = data.total_scans || 0;
                document.getElementById('tokens-discovered').textContent = data.tokens_discovered || 0;
                document.getElementById('execution-opportunities').textContent = data.execution_opportunities || 0;
                document.getElementById('cache-hit-rate').textContent = (data.cache_hit_rate || 0) + '%';
                document.getElementById('last-scan').textContent = data.last_scan || 'Never';
                
                // Update scanner status
                const statusElement = document.getElementById('scanner-status');
                const statusText = document.getElementById('scanner-status-text');
                if (data.scanner_active) {
                    statusElement.className = 'status-indicator status-active';
                    statusText.textContent = 'Active';
                } else {
                    statusElement.className = 'status-indicator status-inactive';
                    statusText.textContent = 'Inactive';
                }
                
                // Update token list
                const tokenList = document.getElementById('token-list');
                if (data.recent_tokens && data.recent_tokens.length > 0) {
                    tokenList.innerHTML = data.recent_tokens.map(token => `
                        <div class="token-item">
                            <span class="token-symbol">${token.symbol}</span>
                            <span class="token-score">${token.score.toFixed(2)}</span>
                        </div>
                    `).join('');
                } else {
                    tokenList.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No tokens discovered yet...</div>';
                }
                
            } catch (error) {
                console.error('Failed to refresh data:', error);
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>'''
    
    dashboard_path = Path("frontend/templates/scanning_dashboard.html")
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_code)
    
    print("✅ Created scanning monitoring dashboard")

if __name__ == "__main__":
    integrate_enhanced_scanning()
