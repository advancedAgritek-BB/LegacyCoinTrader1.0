#!/usr/bin/env python3
"""
WebSocket health monitoring script for the crypto bot.
Monitors WebSocket connections and identifies common issues.
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import websocket
import threading

from crypto_bot.config import load_config as load_bot_config, resolve_config_path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration overrides and defaults."""
    try:
        return load_bot_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

class WebSocketHealthChecker:
    """Monitor WebSocket connections and health."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.websocket_config = {
            'use_websocket': config.get('use_websocket', True),
            'ws_ohlcv_timeout': config.get('ws_ohlcv_timeout', 20),
            'ws_ping_interval': config.get('ws_ping_interval', 8),
            'ws_failures_before_disable': config.get('ws_failures_before_disable', 5),
            'ws_reconnect_delay': config.get('ws_reconnect_delay', 10),
            'ws_max_retries': config.get('ws_max_retries', 3)
        }
        
    def check_configuration(self) -> List[str]:
        """Check WebSocket configuration for issues."""
        issues = []
        
        if not self.websocket_config['use_websocket']:
            print("ℹ️  WebSocket is disabled in configuration")
            return issues
        
        # Check timeout values
        timeout = self.websocket_config['ws_ohlcv_timeout']
        if timeout < 5:
            issues.append(f"ws_ohlcv_timeout ({timeout}) is too low, should be >= 5")
        elif timeout > 30:
            issues.append(f"ws_ohlcv_timeout ({timeout}) is too high, should be <= 30")
        
        # Check ping interval
        ping = self.websocket_config['ws_ping_interval']
        if ping < 3:
            issues.append(f"ws_ping_interval ({ping}) is too low, should be >= 3")
        elif ping > 15:
            issues.append(f"ws_ping_interval ({ping}) is too high, should be <= 15")
        
        # Check failure threshold
        failures = self.websocket_config['ws_failures_before_disable']
        if failures < 2:
            issues.append(f"ws_failures_before_disable ({failures}) is too low, should be >= 2")
        elif failures > 10:
            issues.append(f"ws_failures_before_disable ({failures}) is too high, should be <= 10")
        
        return issues
    
    def test_kraken_websocket(self) -> bool:
        """Test basic Kraken WebSocket connectivity."""
        print("🔗 Testing Kraken WebSocket connectivity...")

        try:
            # Test public WebSocket
            ws = websocket.create_connection(
                "wss://ws.kraken.com/v2",
                timeout=10
            )

            # Send a simple ping
            ping_msg = {"method": "ping"}
            ws.send(json.dumps(ping_msg))

            # Wait for response
            response = ws.recv()
            ws.close()

            # Check for pong or valid status response
            response_lower = response.lower()
            if "pong" in response_lower or ("status" in response_lower and "online" in response_lower):
                print("✅ Kraken public WebSocket is accessible")
                return True
            else:
                print(f"⚠️  Unexpected response from Kraken: {response}")
                return False

        except Exception as e:
            print(f"❌ Kraken WebSocket test failed: {e}")
            return False

    def test_enhanced_websocket_features(self) -> Dict[str, bool]:
        """Test enhanced WebSocket features like health checks and validation."""
        print("🔧 Testing Enhanced WebSocket Features...")
        results = {}

        try:
            # Test if enhanced Kraken client can be imported
            import sys
            import os
            # Add the project root directory to the path
            project_root = os.path.dirname(os.path.dirname(__file__))
            sys.path.insert(0, project_root)

            from crypto_bot.execution.kraken_ws import KrakenWSClient
            client = KrakenWSClient()
            results["kraken_enhancements"] = True
            print("✅ Enhanced Kraken WebSocket client available")

            # Test health monitoring methods
            if hasattr(client, 'start_health_monitoring'):
                results["health_monitoring"] = True
                print("✅ Health monitoring available")
            else:
                results["health_monitoring"] = False
                print("❌ Health monitoring not available")

            # Test message validation
            if hasattr(client, '_validate_message'):
                test_msg = '{"test": "message"}'
                is_valid = client._validate_message(test_msg)
                results["message_validation"] = is_valid
                if is_valid:
                    print("✅ Message validation working")
                else:
                    print("⚠️  Message validation not working properly")
            else:
                results["message_validation"] = False
                print("❌ Message validation not available")

        except (ImportError, Exception) as e:
            results["kraken_enhancements"] = False
            print(f"❌ Enhanced Kraken WebSocket client not available: {e}")

        try:
            # Test if enhanced Solana monitor can be imported
            from crypto_bot.solana.pool_ws_monitor import EnhancedPoolMonitor
            monitor = EnhancedPoolMonitor("test_key", "test_pool")
            results["solana_enhancements"] = True
            print("✅ Enhanced Solana pool monitor available")

            # Test exponential backoff
            if hasattr(monitor, '_calculate_reconnect_delay'):
                delay = monitor._calculate_reconnect_delay()
                results["exponential_backoff"] = isinstance(delay, float)
                if results["exponential_backoff"]:
                    print("✅ Exponential backoff working")
                else:
                    print("⚠️  Exponential backoff not working properly")
            else:
                results["exponential_backoff"] = False
                print("❌ Exponential backoff not available")

        except (ImportError, Exception) as e:
            results["solana_enhancements"] = False
            print(f"❌ Enhanced Solana pool monitor not available: {e}")

        try:
            # Test WebSocket pool
            from crypto_bot.utils.websocket_pool import WebSocketPool
            pool = WebSocketPool(max_connections=2)
            results["websocket_pool"] = True
            print("✅ WebSocket connection pool available")

            # Test pool stats
            stats = pool.get_pool_stats()
            results["pool_stats"] = isinstance(stats, dict)
            if results["pool_stats"]:
                print("✅ Connection pool statistics working")
            else:
                print("⚠️  Connection pool statistics not working properly")

        except (ImportError, Exception) as e:
            results["websocket_pool"] = False
            print(f"❌ WebSocket connection pool not available: {e}")

        return results
    
    def check_ohlcv_quality(self) -> List[str]:
        """Check OHLCV quality configuration."""
        issues = []
        ohlcv = self.config.get('ohlcv_quality', {})
        
        min_ratio = ohlcv.get('min_data_ratio', 0.6)
        if min_ratio < 0.3:
            issues.append(f"min_data_ratio ({min_ratio}) is too low, may cause too many symbols to be skipped")
        elif min_ratio > 0.9:
            issues.append(f"min_data_ratio ({min_ratio}) is too high, may be too strict")
        
        min_candles = ohlcv.get('min_required_candles', 30)
        if min_candles < 10:
            issues.append(f"min_required_candles ({min_candles}) is too low, may cause analysis issues")
        elif min_candles > 100:
            issues.append(f"min_required_candles ({min_candles}) is too high, may be too strict")
        
        return issues
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current configuration."""
        recommendations = []
        
        if self.websocket_config['use_websocket']:
            recommendations.append("✅ WebSocket is enabled - good for real-time data")
            
            if self.websocket_config['ws_ohlcv_timeout'] > 20:
                recommendations.append("💡 Consider reducing ws_ohlcv_timeout to 15-20 seconds for faster fallback")
            
            if self.websocket_config['ws_ping_interval'] > 10:
                recommendations.append("💡 Consider reducing ws_ping_interval to 5-8 seconds for better connection stability")
        else:
            recommendations.append("ℹ️  WebSocket is disabled - using REST API only")
        
        ohlcv = self.config.get('ohlcv_quality', {})
        if ohlcv.get('fallback_to_rest', True):
            recommendations.append("✅ REST fallback is enabled - good for reliability")
        
        return recommendations

def main():
    """Main health check function."""
    config_path = resolve_config_path()
    if not Path(config_path).exists():
        print(f"ℹ️ Override configuration not found at {config_path}; using defaults.")

    print("🔌 WebSocket Health Check")
    print("=" * 40)

    config = load_config(config_path)
    if not config:
        print("❌ Failed to load configuration")
        return
    
    checker = WebSocketHealthChecker(config)
    
    # Check configuration
    print("\n📋 Configuration Check:")
    config_issues = checker.check_configuration()
    if config_issues:
        print("❌ Found configuration issues:")
        for issue in config_issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration looks good")
    
    # Check OHLCV quality settings
    print("\n📊 OHLCV Quality Check:")
    ohlcv_issues = checker.check_ohlcv_quality()
    if ohlcv_issues:
        print("⚠️  OHLCV quality issues:")
        for issue in ohlcv_issues:
            print(f"  - {issue}")
    else:
        print("✅ OHLCV quality settings look good")
    
    # Test WebSocket connectivity
    print("\n🌐 Connectivity Test:")
    if config.get('use_websocket', True):
        kraken_ok = checker.test_kraken_websocket()
        if not kraken_ok:
            print("❌ Kraken WebSocket connectivity issues detected")
    else:
        print("ℹ️  Skipping WebSocket test (disabled in config)")

    # Test enhanced WebSocket features
    print("\n🔧 Enhanced Features Test:")
    enhanced_results = checker.test_enhanced_websocket_features()

    # Generate recommendations
    print("\n💡 Recommendations:")
    recommendations = checker.generate_recommendations()

    # Add recommendations based on enhanced features test
    if enhanced_results.get("kraken_enhancements", False):
        recommendations.append("✅ Enhanced Kraken WebSocket client is available")
    else:
        recommendations.append("⚠️  Consider upgrading to enhanced Kraken WebSocket client")

    if enhanced_results.get("solana_enhancements", False):
        recommendations.append("✅ Enhanced Solana pool monitor is available")
    else:
        recommendations.append("⚠️  Consider upgrading to enhanced Solana pool monitor")

    if enhanced_results.get("websocket_pool", False):
        recommendations.append("✅ WebSocket connection pooling is available")
    else:
        recommendations.append("💡 Consider implementing WebSocket connection pooling")

    for rec in recommendations:
        print(f"  {rec}")
    
    # Summary
    total_issues = len(config_issues) + len(ohlcv_issues)
    if total_issues == 0:
        print("\n🎉 All checks passed! Your WebSocket configuration is healthy.")
    else:
        print(f"\n⚠️  Found {total_issues} issue(s) that should be addressed.")
        print("   Consider running the validation script for detailed fixes.")

if __name__ == "__main__":
    main()
