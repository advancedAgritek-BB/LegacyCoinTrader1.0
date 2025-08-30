"""
Trade Pipeline Monitor and Debugger

This module provides comprehensive monitoring and debugging capabilities for the entire trade pipeline,
including real-time performance metrics, error tracking, and debugging tools for both live and paper trading.
"""

import asyncio
import time
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import queue
import logging
from pathlib import Path

from .enhanced_logger import get_enhanced_logging_manager, log_trade_pipeline_event
from .logger import LOG_DIR, setup_logger

# Setup logger for this module
logger = setup_logger(__name__, LOG_DIR / "trade_pipeline_monitor.log")


@dataclass
class PipelineEvent:
    """Represents a single event in the trade pipeline."""
    timestamp: datetime
    component: str
    event_type: str
    symbol: str
    strategy: str
    execution_mode: str
    data: Dict[str, Any]
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    success: bool = True


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the trade pipeline."""
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    total_trades: int = 0
    paper_trades: int = 0
    live_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_execution_time: float = 0.0
    error_rate: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComponentMetrics:
    """Metrics for a specific pipeline component."""
    name: str
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    avg_duration_ms: float = 0.0
    last_event: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 0.0


class TradePipelineMonitor:
    """Real-time monitor for the trade pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logging_manager = get_enhanced_logging_manager(config)
        
        # Event tracking
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.event_lock = threading.Lock()
        
        # Metrics tracking
        self.pipeline_metrics = PipelineMetrics()
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.metrics_lock = threading.Lock()
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.performance_lock = threading.Lock()
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.error_lock = threading.Lock()
        
        # Real-time monitoring
        self.monitoring_enabled = config.get('monitoring', {}).get('real_time', {}).get('enabled', True)
        self.update_interval = config.get('monitoring', {}).get('real_time', {}).get('update_interval', 5)
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Debug mode
        self.debug_mode = config.get('debug_mode', {}).get('enabled', False)
        
        # Initialize component metrics
        self._initialize_component_metrics()
        
        # Start monitoring if enabled
        if self.monitoring_enabled:
            self._start_monitoring()
    
    def _initialize_component_metrics(self):
        """Initialize metrics for all pipeline components."""
        components = [
            'signal_generation',
            'risk_management',
            'position_sizing',
            'trade_execution',
            'position_management',
            'balance_tracking',
            'performance_monitoring'
        ]
        
        for component in components:
            self.component_metrics[component] = ComponentMetrics(name=component)
    
    def _start_monitoring(self):
        """Start the real-time monitoring task."""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Trade pipeline monitoring started")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self._update_metrics()
                await self._log_performance_summary()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_metrics(self):
        """Update pipeline metrics."""
        with self.metrics_lock:
            # Update pipeline metrics
            total_events = len(self.events)
            successful_events = sum(1 for event in self.events if event.success)
            failed_events = total_events - successful_events
            
            self.pipeline_metrics.total_events = total_events
            self.pipeline_metrics.successful_events = successful_events
            self.pipeline_metrics.failed_events = failed_events
            
            if total_events > 0:
                self.pipeline_metrics.error_rate = failed_events / total_events
            
            # Update component metrics
            for component_name, component_metrics in self.component_metrics.items():
                component_events = [e for e in self.events if e.component == component_name]
                
                if component_events:
                    component_metrics.total_events = len(component_events)
                    component_metrics.successful_events = sum(1 for e in component_events if e.success)
                    component_metrics.failed_events = component_metrics.total_events - component_metrics.successful_events
                    
                    if component_metrics.total_events > 0:
                        component_metrics.success_rate = component_metrics.successful_events / component_metrics.total_events
                    
                    # Calculate average duration
                    durations = [e.duration_ms for e in component_events if e.duration_ms is not None]
                    if durations:
                        component_metrics.avg_duration_ms = sum(durations) / len(durations)
                    
                    # Update last event timestamp
                    if component_events:
                        component_metrics.last_event = max(e.timestamp for e in component_events)
            
            self.pipeline_metrics.last_update = datetime.utcnow()
    
    async def _log_performance_summary(self):
        """Log performance summary periodically."""
        if self.debug_mode:
            logger.debug(f"Pipeline Performance Summary: {self.pipeline_metrics}")
            
            for component_name, metrics in self.component_metrics.items():
                if metrics.total_events > 0:
                    logger.debug(f"Component {component_name}: {metrics}")
    
    def record_event(self, component: str, event_type: str, symbol: str, strategy: str, 
                    execution_mode: str, data: Dict[str, Any], duration_ms: Optional[float] = None,
                    error: Optional[str] = None, success: bool = True):
        """Record a pipeline event."""
        event = PipelineEvent(
            timestamp=datetime.utcnow(),
            component=component,
            event_type=event_type,
            symbol=symbol,
            strategy=strategy,
            execution_mode=execution_mode,
            data=data,
            duration_ms=duration_ms,
            error=error,
            success=success
        )
        
        with self.event_lock:
            self.events.append(event)
        
        # Update component metrics immediately
        with self.metrics_lock:
            if component in self.component_metrics:
                comp_metrics = self.component_metrics[component]
                comp_metrics.total_events += 1
                
                if success:
                    comp_metrics.successful_events += 1
                else:
                    comp_metrics.failed_events += 1
                    comp_metrics.error_count += 1
                
                if duration_ms is not None:
                    # Update running average
                    if comp_metrics.total_events == 1:
                        comp_metrics.avg_duration_ms = duration_ms
                    else:
                        comp_metrics.avg_duration_ms = (
                            (comp_metrics.avg_duration_ms * (comp_metrics.total_events - 1) + duration_ms) /
                            comp_metrics.total_events
                        )
                
                comp_metrics.last_event = event.timestamp
        
        # Log to enhanced logger
        try:
            if success:
                self.logging_manager.log_trade_pipeline_event(
                    component=component,
                    event_type=event_type,
                    symbol=symbol,
                    strategy=strategy,
                    execution_mode=execution_mode,
                    **data
                )
            else:
                self.logging_manager.log_trade_pipeline_event(
                    component=component,
                    event_type='error',
                    error=Exception(error) if error else Exception("Unknown error"),
                    context=component,
                    symbol=symbol,
                    strategy=strategy,
                    **data
                )
        except Exception as e:
            logger.error(f"Failed to log pipeline event: {e}")
        
        # Record performance data
        if duration_ms is not None:
            with self.performance_lock:
                self.performance_history.append({
                    'timestamp': event.timestamp,
                    'component': component,
                    'duration_ms': duration_ms,
                    'success': success
                })
        
        # Record errors
        if not success:
            with self.error_lock:
                self.error_history.append({
                    'timestamp': event.timestamp,
                    'component': component,
                    'error': error,
                    'symbol': symbol,
                    'strategy': strategy,
                    'data': data
                })
    
    def get_pipeline_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        with self.metrics_lock:
            return PipelineMetrics(
                total_events=self.pipeline_metrics.total_events,
                successful_events=self.pipeline_metrics.successful_events,
                failed_events=self.pipeline_metrics.failed_events,
                total_trades=self.pipeline_metrics.total_trades,
                paper_trades=self.pipeline_metrics.paper_trades,
                live_trades=self.pipeline_metrics.live_trades,
                total_pnl=self.pipeline_metrics.total_pnl,
                win_rate=self.pipeline_metrics.win_rate,
                avg_execution_time=self.pipeline_metrics.avg_execution_time,
                error_rate=self.pipeline_metrics.error_rate,
                last_update=self.pipeline_metrics.last_update
            )
    
    def get_component_metrics(self, component: str) -> Optional[ComponentMetrics]:
        """Get metrics for a specific component."""
        with self.metrics_lock:
            if component in self.component_metrics:
                metrics = self.component_metrics[component]
                return ComponentMetrics(
                    name=metrics.name,
                    total_events=metrics.total_events,
                    successful_events=metrics.successful_events,
                    failed_events=metrics.failed_events,
                    avg_duration_ms=metrics.avg_duration_ms,
                    last_event=metrics.last_event,
                    error_count=metrics.error_count,
                    success_rate=metrics.success_rate
                )
        return None
    
    def get_all_component_metrics(self) -> Dict[str, ComponentMetrics]:
        """Get metrics for all components."""
        with self.metrics_lock:
            return {
                name: ComponentMetrics(
                    name=metrics.name,
                    total_events=metrics.total_events,
                    successful_events=metrics.successful_events,
                    failed_events=metrics.failed_events,
                    avg_duration_ms=metrics.avg_duration_ms,
                    last_event=metrics.last_event,
                    error_count=metrics.error_count,
                    success_rate=metrics.success_rate
                )
                for name, metrics in self.component_metrics.items()
            }
    
    def get_recent_events(self, limit: int = 100) -> List[PipelineEvent]:
        """Get recent pipeline events."""
        with self.event_lock:
            return list(self.events)[-limit:]
    
    def get_events_by_component(self, component: str, limit: int = 100) -> List[PipelineEvent]:
        """Get events for a specific component."""
        with self.event_lock:
            component_events = [e for e in self.events if e.component == component]
            return component_events[-limit:]
    
    def get_events_by_symbol(self, symbol: str, limit: int = 100) -> List[PipelineEvent]:
        """Get events for a specific symbol."""
        with self.event_lock:
            symbol_events = [e for e in self.events if e.symbol == symbol]
            return symbol_events[-limit:]
    
    def get_performance_history(self, component: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance history."""
        with self.performance_lock:
            if component:
                history = [p for p in self.performance_history if p['component'] == component]
            else:
                history = list(self.performance_history)
            return history[-limit:]
    
    def get_error_history(self, component: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get error history."""
        with self.error_lock:
            if component:
                errors = [e for e in self.error_history if e['component'] == component]
            else:
                errors = list(self.error_history)
            return errors[-limit:]
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of all trades."""
        with self.event_lock:
            trade_events = [e for e in self.events if e.event_type in ['trade_execution', 'paper_trading_simulation']]
            
            summary = {
                'total_trades': len(trade_events),
                'paper_trades': len([e for e in trade_events if e.execution_mode == 'dry_run']),
                'live_trades': len([e for e in trade_events if e.execution_mode == 'live']),
                'by_strategy': defaultdict(int),
                'by_symbol': defaultdict(int),
                'recent_trades': []
            }
            
            for event in trade_events:
                summary['by_strategy'][event.strategy] += 1
                summary['by_symbol'][event.symbol] += 1
                
                # Add to recent trades
                summary['recent_trades'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'symbol': event.symbol,
                    'strategy': event.strategy,
                    'execution_mode': event.execution_mode,
                    'success': event.success
                })
            
            # Keep only last 50 trades
            summary['recent_trades'] = summary['recent_trades'][-50:]
            
            return summary
    
    def export_metrics(self, filepath: Optional[str] = None) -> str:
        """Export metrics to JSON file."""
        if filepath is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = LOG_DIR / f"pipeline_metrics_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'pipeline_metrics': self.get_pipeline_metrics().__dict__,
            'component_metrics': {
                name: metrics.__dict__ for name, metrics in self.get_all_component_metrics().items()
            },
            'trade_summary': self.get_trade_summary(),
            'recent_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'component': event.component,
                    'event_type': event.event_type,
                    'symbol': event.symbol,
                    'strategy': event.strategy,
                    'execution_mode': event.execution_mode,
                    'success': event.success,
                    'duration_ms': event.duration_ms,
                    'error': event.error
                }
                for event in self.get_recent_events(1000)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Pipeline metrics exported to {filepath}")
        return str(filepath)
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self.event_lock:
            self.events.clear()
        
        with self.metrics_lock:
            self.pipeline_metrics = PipelineMetrics()
            for component_metrics in self.component_metrics.values():
                component_metrics.total_events = 0
                component_metrics.successful_events = 0
                component_metrics.failed_events = 0
                component_metrics.avg_duration_ms = 0.0
                component_metrics.last_event = None
                component_metrics.error_count = 0
                component_metrics.success_rate = 0.0
        
        with self.performance_lock:
            self.performance_history.clear()
        
        with self.error_lock:
            self.error_history.clear()
        
        logger.info("Pipeline metrics reset")
    
    def stop_monitoring(self):
        """Stop the monitoring task."""
        self.monitoring_enabled = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        logger.info("Trade pipeline monitoring stopped")


class TradePipelineDebugger:
    """Debugging tools for the trade pipeline."""
    
    def __init__(self, monitor: TradePipelineMonitor, config: Dict[str, Any]):
        self.monitor = monitor
        self.config = config
        self.debug_mode = config.get('debug_mode', {}).get('enabled', False)
        self.logger = setup_logger(__name__, LOG_DIR / "trade_pipeline_debugger.log")
        
        # Debug callbacks
        self.debug_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_execution_time_ms': 1000.0,  # 1 second
            'max_error_rate': 0.1,  # 10%
            'max_memory_usage_mb': 512.0  # 512 MB
        }
    
    def add_debug_callback(self, event_type: str, callback: Callable):
        """Add a debug callback for specific event types."""
        self.debug_callbacks[event_type].append(callback)
        self.logger.debug(f"Added debug callback for {event_type}")
    
    def remove_debug_callback(self, event_type: str, callback: Callable):
        """Remove a debug callback."""
        if event_type in self.debug_callbacks:
            try:
                self.debug_callbacks[event_type].remove(callback)
                self.logger.debug(f"Removed debug callback for {event_type}")
            except ValueError:
                pass
    
    def debug_event(self, event: PipelineEvent):
        """Process an event for debugging."""
        if not self.debug_mode:
            return
        
        # Check performance thresholds
        if event.duration_ms and event.duration_ms > self.performance_thresholds['max_execution_time_ms']:
            self.logger.warning(
                f"Slow execution detected: {event.component}.{event.event_type} "
                f"took {event.duration_ms:.2f}ms for {event.symbol}"
            )
        
        # Check error conditions
        if not event.success:
            self.logger.error(
                f"Pipeline error in {event.component}.{event.event_type}: "
                f"{event.error} for {event.symbol}"
            )
        
        # Execute debug callbacks
        if event.event_type in self.debug_callbacks:
            for callback in self.debug_callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Debug callback error: {e}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze pipeline performance and identify bottlenecks."""
        if not self.debug_mode:
            return {}
        
        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'performance_issues': [],
            'recommendations': [],
            'component_analysis': {}
        }
        
        # Analyze component performance
        component_metrics = self.monitor.get_all_component_metrics()
        
        for component_name, metrics in component_metrics.items():
            if metrics.total_events == 0:
                continue
            
            component_analysis = {
                'name': component_name,
                'total_events': metrics.total_events,
                'success_rate': metrics.success_rate,
                'avg_duration_ms': metrics.avg_duration_ms,
                'issues': []
            }
            
            # Check for performance issues
            if metrics.avg_duration_ms > self.performance_thresholds['max_execution_time_ms']:
                component_analysis['issues'].append({
                    'type': 'slow_execution',
                    'severity': 'warning',
                    'message': f"Average execution time {metrics.avg_duration_ms:.2f}ms exceeds threshold"
                })
                analysis['performance_issues'].append({
                    'component': component_name,
                    'issue': 'slow_execution',
                    'severity': 'warning'
                })
            
            if metrics.success_rate < (1.0 - self.performance_thresholds['max_error_rate']):
                component_analysis['issues'].append({
                    'type': 'high_error_rate',
                    'severity': 'error',
                    'message': f"Success rate {metrics.success_rate:.2%} below threshold"
                })
                analysis['performance_issues'].append({
                    'component': component_name,
                    'issue': 'high_error_rate',
                    'severity': 'error'
                })
            
            analysis['component_analysis'][component_name] = component_analysis
        
        # Generate recommendations
        if analysis['performance_issues']:
            analysis['recommendations'].append(
                "Consider enabling debug mode for detailed component analysis"
            )
            
            slow_components = [p['component'] for p in analysis['performance_issues'] if p['issue'] == 'slow_execution']
            if slow_components:
                analysis['recommendations'].append(
                    f"Optimize execution in components: {', '.join(slow_components)}"
                )
            
            error_components = [p['component'] for p in analysis['performance_issues'] if p['issue'] == 'high_error_rate']
            if error_components:
                analysis['recommendations'].append(
                    f"Investigate errors in components: {', '.join(error_components)}"
                )
        
        return analysis
    
    def generate_debug_report(self) -> str:
        """Generate a comprehensive debug report."""
        if not self.debug_mode:
            return "Debug mode is disabled"
        
        report_lines = [
            "=== TRADE PIPELINE DEBUG REPORT ===",
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
            "--- PIPELINE METRICS ---",
        ]
        
        # Add pipeline metrics
        pipeline_metrics = self.monitor.get_pipeline_metrics()
        report_lines.extend([
            f"Total Events: {pipeline_metrics.total_events}",
            f"Success Rate: {pipeline_metrics.successful_events / max(pipeline_metrics.total_events, 1):.2%}",
            f"Error Rate: {pipeline_metrics.error_rate:.2%}",
            f"Total Trades: {pipeline_metrics.total_trades}",
            f"Paper Trades: {pipeline_metrics.paper_trades}",
            f"Live Trades: {pipeline_metrics.live_trades}",
            ""
        ])
        
        # Add component analysis
        report_lines.append("--- COMPONENT ANALYSIS ---")
        component_metrics = self.monitor.get_all_component_metrics()
        
        for component_name, metrics in component_metrics.items():
            if metrics.total_events == 0:
                continue
                
            report_lines.extend([
                f"Component: {component_name}",
                f"  Events: {metrics.total_events}",
                f"  Success Rate: {metrics.success_rate:.2%}",
                f"  Avg Duration: {metrics.avg_duration_ms:.2f}ms",
                f"  Last Event: {metrics.last_event.isoformat() if metrics.last_event else 'Never'}",
                ""
            ])
        
        # Add performance analysis
        report_lines.append("--- PERFORMANCE ANALYSIS ---")
        performance_analysis = self.analyze_performance()
        
        if performance_analysis.get('performance_issues'):
            report_lines.append("Performance Issues Found:")
            for issue in performance_analysis['performance_issues']:
                report_lines.append(f"  {issue['severity'].upper()}: {issue['component']} - {issue['issue']}")
        else:
            report_lines.append("No performance issues detected")
        
        if performance_analysis.get('recommendations'):
            report_lines.append("")
            report_lines.append("Recommendations:")
            for rec in performance_analysis['recommendations']:
                report_lines.append(f"  - {rec}")
        
        report_lines.append("")
        report_lines.append("=== END REPORT ===")
        
        report = "\n".join(report_lines)
        
        # Save report to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = LOG_DIR / f"debug_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Debug report generated: {report_file}")
        return report


# Global monitor instance
_pipeline_monitor = None
_pipeline_debugger = None

def get_pipeline_monitor(config: Dict[str, Any]) -> TradePipelineMonitor:
    """Get or create the global pipeline monitor."""
    global _pipeline_monitor
    if _pipeline_monitor is None:
        _pipeline_monitor = TradePipelineMonitor(config)
    return _pipeline_monitor

def get_pipeline_debugger(config: Dict[str, Any]) -> TradePipelineDebugger:
    """Get or create the global pipeline debugger."""
    global _pipeline_debugger
    if _pipeline_debugger is None:
        monitor = get_pipeline_monitor(config)
        _pipeline_debugger = TradePipelineDebugger(monitor, config)
    return _pipeline_debugger

def record_pipeline_event(component: str, event_type: str, symbol: str, strategy: str,
                         execution_mode: str, data: Dict[str, Any], duration_ms: Optional[float] = None,
                         error: Optional[str] = None, success: bool = True):
    """Convenience function to record pipeline events."""
    try:
        monitor = get_pipeline_monitor({})  # Use empty config as fallback
        monitor.record_event(
            component=component,
            event_type=event_type,
            symbol=symbol,
            strategy=strategy,
            execution_mode=execution_mode,
            data=data,
            duration_ms=duration_ms,
            error=error,
            success=success
        )
    except Exception as e:
        logger.error(f"Failed to record pipeline event: {e}")
