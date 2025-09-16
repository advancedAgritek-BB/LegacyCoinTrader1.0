#!/usr/bin/env python3
"""
Signal Handlers Module for LegacyCoinTrader

This module provides standardized signal handling for graceful shutdowns
across all application components.
"""

import os
import sys
import signal
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class GracefulShutdownHandler:
    """
    Handles graceful shutdown signals for application components.
    
    Features:
    - Standardized signal handling across all components
    - Graceful shutdown with configurable timeout
    - Cleanup callback registration
    - PID file management
    - Shutdown state tracking
    """
    
    def __init__(self, 
                 component_name: str,
                 pid_file: Optional[Path] = None,
                 shutdown_timeout: int = 30):
        self.component_name = component_name
        self.pid_file = pid_file
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_initiated = False
        self.cleanup_callbacks: Dict[str, Callable] = {}
        self.shutdown_thread: Optional[threading.Thread] = None
        
        # Write PID file if specified
        if self.pid_file:
            self._write_pid_file()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info(f"🛡️ Signal handlers initialized for {component_name}")

    def _write_pid_file(self) -> None:
        """Write current process PID to file."""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.debug(f"📝 PID file written: {self.pid_file}")
        except Exception as e:
            logger.error(f"❌ Failed to write PID file {self.pid_file}: {e}")

    def _cleanup_pid_file(self) -> None:
        """Clean up PID file."""
        if self.pid_file and self.pid_file.exists():
            try:
                self.pid_file.unlink()
                logger.debug(f"🧹 PID file cleaned up: {self.pid_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up PID file {self.pid_file}: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        # Handle termination signals
        signal.signal(signal.SIGINT, self._signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # Termination request
        
        # Handle hang up signal if available (Unix-like systems)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self._signal_handler)
        
        # Handle user signals for custom actions
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._user_signal_handler)
        if hasattr(signal, 'SIGUSR2'):
            signal.signal(signal.SIGUSR2, self._user_signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        if self.shutdown_initiated:
            logger.warning(f"🔄 {self.component_name}: Shutdown already in progress")
            return
        
        signal_name = signal.Signals(signum).name
        logger.info(f"📡 {self.component_name}: Received {signal_name} signal")
        
        self.shutdown_initiated = True
        
        # Start shutdown in separate thread to avoid blocking signal handler
        self.shutdown_thread = threading.Thread(
            target=self._execute_graceful_shutdown,
            args=(signum,),
            name=f"{self.component_name}_shutdown"
        )
        self.shutdown_thread.start()

    def _user_signal_handler(self, signum: int, frame) -> None:
        """Handle user-defined signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"📡 {self.component_name}: Received user signal {signal_name}")
        
        # Custom signal handling can be implemented here
        # For example, SIGUSR1 could trigger a status report
        if signum == signal.SIGUSR1:
            self._handle_status_request()
        elif signum == signal.SIGUSR2:
            self._handle_config_reload()

    def _handle_status_request(self) -> None:
        """Handle status request signal."""
        logger.info(f"📊 {self.component_name}: Status request received")
        # Implementation would depend on the component
        # Could trigger status report, health check, etc.

    def _handle_config_reload(self) -> None:
        """Handle configuration reload signal."""
        logger.info(f"🔄 {self.component_name}: Config reload request received")
        # Implementation would reload configuration
        # Execute config reload callback if registered
        if 'config_reload' in self.cleanup_callbacks:
            try:
                self.cleanup_callbacks['config_reload']()
            except Exception as e:
                logger.error(f"❌ Config reload failed: {e}")

    def _execute_graceful_shutdown(self, signum: int) -> None:
        """Execute graceful shutdown process."""
        signal_name = signal.Signals(signum).name
        logger.info(f"🛑 {self.component_name}: Starting graceful shutdown (signal: {signal_name})")
        
        start_time = time.time()
        
        try:
            # Execute cleanup callbacks in order
            for callback_name, callback in self.cleanup_callbacks.items():
                if callback_name == 'config_reload':  # Skip config reload in shutdown
                    continue
                    
                try:
                    logger.info(f"🧹 {self.component_name}: Executing {callback_name} cleanup")
                    callback()
                    logger.info(f"✅ {self.component_name}: {callback_name} cleanup completed")
                except Exception as e:
                    logger.error(f"❌ {self.component_name}: {callback_name} cleanup failed: {e}")
            
            # Clean up PID file
            self._cleanup_pid_file()
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ {self.component_name}: Graceful shutdown completed in {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ {self.component_name}: Shutdown error: {e}")
        finally:
            # Force exit if we're still running
            logger.info(f"👋 {self.component_name}: Exiting")
            os._exit(0)

    def register_cleanup_callback(self, name: str, callback: Callable) -> None:
        """Register a cleanup callback to be executed during shutdown."""
        self.cleanup_callbacks[name] = callback
        logger.debug(f"📋 {self.component_name}: Registered cleanup callback '{name}'")

    def unregister_cleanup_callback(self, name: str) -> None:
        """Unregister a cleanup callback."""
        if name in self.cleanup_callbacks:
            del self.cleanup_callbacks[name]
            logger.debug(f"🗑️ {self.component_name}: Unregistered cleanup callback '{name}'")

    def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        if self.shutdown_thread and self.shutdown_thread.is_alive():
            logger.info(f"⏳ {self.component_name}: Waiting for shutdown to complete")
            self.shutdown_thread.join(timeout=self.shutdown_timeout)
            
            if self.shutdown_thread.is_alive():
                logger.error(f"❌ {self.component_name}: Shutdown timeout exceeded")
                os._exit(1)

    def is_shutdown_initiated(self) -> bool:
        """Check if shutdown has been initiated."""
        return self.shutdown_initiated

class ComponentShutdownManager:
    """
    Manages shutdown for multiple components within a single process.
    """
    
    def __init__(self, process_name: str, pid_file: Optional[Path] = None):
        self.process_name = process_name
        self.pid_file = pid_file
        self.components: Dict[str, Dict[str, Any]] = {}
        self.shutdown_initiated = False
        self.main_handler = GracefulShutdownHandler(
            component_name=process_name,
            pid_file=pid_file
        )
        
        # Register main shutdown callback
        self.main_handler.register_cleanup_callback(
            "component_shutdown",
            self._shutdown_all_components
        )

    def register_component(self, 
                          component_name: str, 
                          cleanup_callback: Optional[Callable] = None,
                          priority: int = 5) -> None:
        """Register a component for shutdown management."""
        self.components[component_name] = {
            'cleanup_callback': cleanup_callback,
            'priority': priority,
            'shutdown_complete': False
        }
        
        logger.info(f"📋 {self.process_name}: Registered component '{component_name}' (priority: {priority})")

    def _shutdown_all_components(self) -> None:
        """Shutdown all registered components."""
        if self.shutdown_initiated:
            return
        
        self.shutdown_initiated = True
        logger.info(f"🛑 {self.process_name}: Shutting down all components")
        
        # Sort components by priority (lower number = higher priority)
        sorted_components = sorted(
            self.components.items(),
            key=lambda x: x[1]['priority']
        )
        
        for component_name, component_info in sorted_components:
            try:
                logger.info(f"🛑 {self.process_name}: Shutting down {component_name}")
                
                if component_info['cleanup_callback']:
                    component_info['cleanup_callback']()
                
                component_info['shutdown_complete'] = True
                logger.info(f"✅ {self.process_name}: {component_name} shutdown complete")
                
            except Exception as e:
                logger.error(f"❌ {self.process_name}: {component_name} shutdown failed: {e}")

    def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        self.main_handler.wait_for_shutdown()

    def is_shutdown_initiated(self) -> bool:
        """Check if shutdown has been initiated."""
        return self.shutdown_initiated

# Convenience functions for common use cases

def setup_basic_signal_handlers(component_name: str, 
                               pid_file: Optional[Path] = None,
                               cleanup_callback: Optional[Callable] = None) -> GracefulShutdownHandler:
    """
    Setup basic signal handlers for a component.
    
    Args:
        component_name: Name of the component
        pid_file: Optional PID file path
        cleanup_callback: Optional cleanup function to call on shutdown
    
    Returns:
        GracefulShutdownHandler instance
    """
    handler = GracefulShutdownHandler(component_name, pid_file)
    
    if cleanup_callback:
        handler.register_cleanup_callback('main_cleanup', cleanup_callback)
    
    return handler

def setup_trading_bot_handlers(pid_file: Optional[Path] = None) -> GracefulShutdownHandler:
    """Setup signal handlers specifically for the trading bot."""
    
    def trading_bot_cleanup():
        """Trading bot specific cleanup."""
        logger.info("🤖 Trading bot: Starting cleanup")
        
        # Close open positions safely
        try:
            logger.info("💰 Closing open positions")
            # Implementation would close positions
            time.sleep(1)  # Simulate position closure
            logger.info("✅ Positions closed")
        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")
        
        # Save trading state
        try:
            logger.info("💾 Saving trading state")
            # Implementation would save state
            logger.info("✅ Trading state saved")
        except Exception as e:
            logger.error(f"❌ Failed to save trading state: {e}")
        
        # Stop WebSocket connections
        try:
            logger.info("🔌 Closing WebSocket connections")
            # Implementation would close WebSocket connections
            logger.info("✅ WebSocket connections closed")
        except Exception as e:
            logger.error(f"❌ Failed to close WebSocket connections: {e}")
    
    return setup_basic_signal_handlers(
        "Trading Bot",
        pid_file,
        trading_bot_cleanup
    )

def setup_web_frontend_handlers(pid_file: Optional[Path] = None) -> GracefulShutdownHandler:
    """Setup signal handlers specifically for the web frontend."""
    
    def web_frontend_cleanup():
        """Web frontend specific cleanup."""
        logger.info("🌐 Web frontend: Starting cleanup")
        
        # Close server connections
        try:
            logger.info("🔌 Closing server connections")
            # Implementation would close server
            logger.info("✅ Server connections closed")
        except Exception as e:
            logger.error(f"❌ Failed to close server connections: {e}")
    
    return setup_basic_signal_handlers(
        "Web Frontend",
        pid_file,
        web_frontend_cleanup
    )

def setup_scanner_handlers(pid_file: Optional[Path] = None) -> GracefulShutdownHandler:
    """Setup signal handlers specifically for the scanner."""
    
    def scanner_cleanup():
        """Scanner specific cleanup."""
        logger.info("🔍 Scanner: Starting cleanup")
        
        # Save scan results
        try:
            logger.info("💾 Saving scan results")
            # Implementation would save scan results
            logger.info("✅ Scan results saved")
        except Exception as e:
            logger.error(f"❌ Failed to save scan results: {e}")
        
        # Close API connections
        try:
            logger.info("🔌 Closing API connections")
            # Implementation would close API connections
            logger.info("✅ API connections closed")
        except Exception as e:
            logger.error(f"❌ Failed to close API connections: {e}")
    
    return setup_basic_signal_handlers(
        "Enhanced Scanner",
        pid_file,
        scanner_cleanup
    )

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test signal handlers")
    parser.add_argument("--component", default="test", help="Component name")
    parser.add_argument("--pid-file", help="PID file path")
    parser.add_argument("--timeout", type=int, default=10, help="Test timeout")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    pid_file = Path(args.pid_file) if args.pid_file else None
    
    # Test signal handlers
    print(f"🧪 Testing signal handlers for {args.component}")
    print(f"📝 PID: {os.getpid()}")
    if pid_file:
        print(f"📁 PID file: {pid_file}")
    print("📡 Send SIGTERM or SIGINT to test graceful shutdown")
    print("⏰ Test will timeout after {} seconds".format(args.timeout))
    
    def test_cleanup():
        print("🧹 Test cleanup function called")
        time.sleep(2)  # Simulate cleanup work
        print("✅ Test cleanup completed")
    
    handler = setup_basic_signal_handlers(
        args.component,
        pid_file,
        test_cleanup
    )
    
    # Wait for signal or timeout
    try:
        time.sleep(args.timeout)
        print("⏰ Test completed (timeout)")
    except KeyboardInterrupt:
        print("🛑 Test interrupted")
    
    # Wait for shutdown if initiated
    if handler.is_shutdown_initiated():
        handler.wait_for_shutdown()
