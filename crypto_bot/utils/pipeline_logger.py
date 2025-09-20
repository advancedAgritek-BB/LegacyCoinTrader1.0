"""
Pipeline Logger - Comprehensive logging for the trading pipeline

This module provides specialized logging utilities for tracking the complete
journey through the scanning, evaluation, and trading pipeline with
human-readable logs and structured data.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .logger import (
    setup_logger,
    LOG_DIR,
    get_pipeline_context,
    set_pipeline_context,
    update_pipeline_context,
    clear_pipeline_context,
    correlation_id_context,
)


@dataclass
class PipelineMetrics:
    """Metrics collected throughout a pipeline run."""
    start_time: float = 0.0
    phase_start_times: Dict[str, float] = field(default_factory=dict)
    phase_durations: Dict[str, float] = field(default_factory=dict)
    symbols_discovered: int = 0
    symbols_evaluated: int = 0
    signals_generated: int = 0
    trades_attempted: int = 0
    trades_executed: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    errors: List[str] = field(default_factory=list)


class PipelineLogger:
    """Specialized logger for tracking trading pipeline execution."""

    def __init__(self, name: str = "pipeline", log_file: Optional[str] = None):
        # Use default pipeline log file if none specified
        if log_file is None:
            log_file = LOG_DIR / "pipeline.log"

        self.logger = setup_logger(name, log_file)
        self.metrics = PipelineMetrics()
        self.current_phase: Optional[str] = None

    def start_pipeline(self, cycle_id: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a new pipeline run."""
        correlation_id = f"cycle_{cycle_id or int(time.time())}"

        with correlation_id_context(correlation_id):
            self.metrics = PipelineMetrics()
            self.metrics.start_time = time.time()

            pipeline_ctx = {
                "cycle_id": cycle_id,
                "phase": "initialization",
                "correlation_id": correlation_id,
                "start_time": datetime.now(timezone.utc).isoformat(),
            }

            if metadata:
                pipeline_ctx.update(metadata)

            set_pipeline_context(pipeline_ctx)

            self.logger.info(
                "🚀 Starting trading pipeline",
                extra={
                    "cycle_id": cycle_id,
                    "pipeline_type": "full_cycle",
                    "metadata": metadata or {},
                }
            )

            return correlation_id

    def start_phase(self, phase_name: str, description: str = "") -> None:
        """Mark the start of a pipeline phase."""
        if self.current_phase:
            self.end_phase(self.current_phase)

        self.current_phase = phase_name
        self.metrics.phase_start_times[phase_name] = time.time()

        update_pipeline_context({"phase": phase_name})

        phase_descriptions = {
            "discovery": "🔍 Scanning markets and discovering tokens",
            "market_data": "📊 Loading market data and indicators",
            "evaluation": "🧠 Evaluating strategies and generating signals",
            "risk_check": "⚖️ Performing risk management checks",
            "execution": "💰 Executing trades",
            "finalization": "📋 Finalizing cycle and recording results",
        }

        readable_desc = description or phase_descriptions.get(phase_name, f"Running {phase_name} phase")

        self.logger.info(
            readable_desc,
            extra={
                "phase": phase_name,
                "phase_start": True,
                "pipeline_progress": self._get_progress_indicator(phase_name),
            }
        )

    def end_phase(self, phase_name: str) -> None:
        """Mark the end of a pipeline phase and record timing."""
        if phase_name not in self.metrics.phase_start_times:
            return

        start_time = self.metrics.phase_start_times[phase_name]
        duration = time.time() - start_time
        self.metrics.phase_durations[phase_name] = duration

        phase_summaries = {
            "discovery": f"Discovered {self.metrics.symbols_discovered} symbols",
            "market_data": f"Loaded data for {self.metrics.symbols_evaluated} symbols",
            "evaluation": f"Generated {self.metrics.signals_generated} trading signals",
            "execution": f"Executed {self.metrics.trades_executed}/{self.metrics.trades_attempted} trades",
        }

        summary = phase_summaries.get(phase_name, f"Completed {phase_name}")

        self.logger.info(
            f"✅ {summary}",
            extra={
                "phase": phase_name,
                "phase_complete": True,
                "duration": duration,
                "phase_metrics": self._get_phase_metrics(phase_name),
            }
        )

    def log_discovery(self, symbol: str, source: str, confidence: float = 0.0, **kwargs) -> None:
        """Log market discovery events."""
        self.metrics.symbols_discovered += 1

        update_pipeline_context({"symbol": symbol, "discovery_source": source})

        confidence_desc = ""
        if confidence > 0.8:
            confidence_desc = " (high confidence)"
        elif confidence > 0.5:
            confidence_desc = " (medium confidence)"
        elif confidence > 0:
            confidence_desc = " (low confidence)"

        self.logger.info(
            f"📈 Discovered {symbol} from {source}{confidence_desc}",
            extra={
                "symbol": symbol,
                "discovery_source": source,
                "confidence": confidence,
                "discovery_count": self.metrics.symbols_discovered,
                **kwargs
            }
        )

    def log_evaluation(self, symbol: str, strategy: str, score: float, direction: str, **kwargs) -> None:
        """Log strategy evaluation results."""
        self.metrics.symbols_evaluated += 1
        self.metrics.signals_generated += 1

        update_pipeline_context({"symbol": symbol, "strategy": strategy})

        # Format score and direction readably
        score_desc = ".2f"
        direction_icon = "📈" if direction.lower() == "long" else "📉"

        confidence_level = "low"
        if score > 0.8:
            confidence_level = "high"
        elif score > 0.6:
            confidence_level = "medium"

        self.logger.info(
            f"{direction_icon} {symbol} - {strategy}: {direction.upper()} signal ({confidence_level} confidence)",
            extra={
                "symbol": symbol,
                "strategy": strategy,
                "score": score,
                "direction": direction,
                "confidence_level": confidence_level,
                "evaluation_count": self.metrics.symbols_evaluated,
                **kwargs
            }
        )

    def log_trade_attempt(self, symbol: str, side: str, amount: float, price: float, reason: str = "", **kwargs) -> None:
        """Log trade execution attempts."""
        self.metrics.trades_attempted += 1

        update_pipeline_context({"symbol": symbol, "trade_side": side})

        usd_value = amount * price
        self.metrics.total_volume += usd_value

        reason_desc = f" - {reason}" if reason else ""

        self.logger.info(
            f"🎯 Attempting {side.upper()} trade: {symbol} {amount:.4f} units @ ${price:.4f} (${usd_value:.2f}){reason_desc}",
            extra={
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "usd_value": usd_value,
                "trade_count": self.metrics.trades_attempted,
                "reason": reason,
                **kwargs
            }
        )

    def log_trade_execution(self, symbol: str, side: str, amount: float, price: float,
                          order_id: Optional[str] = None, pnl: float = 0.0, **kwargs) -> None:
        """Log successful trade executions."""
        self.metrics.trades_executed += 1
        self.metrics.total_pnl += pnl

        usd_value = amount * price

        pnl_desc = ""
        if pnl != 0:
            pnl_icon = "💰" if pnl > 0 else "📉"
            pnl_desc = f" {pnl_icon} P&L: ${pnl:.2f}"

        order_desc = f" (Order: {order_id})" if order_id else ""

        self.logger.info(
            f"✅ Executed {side.upper()}: {symbol} {amount:.4f} @ ${price:.4f} (${usd_value:.2f}){pnl_desc}{order_desc}",
            extra={
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "usd_value": usd_value,
                "order_id": order_id,
                "pnl": pnl,
                "execution_count": self.metrics.trades_executed,
                **kwargs
            }
        )

    def log_error(self, error_msg: str, symbol: Optional[str] = None, phase: Optional[str] = None, **kwargs) -> None:
        """Log pipeline errors."""
        self.metrics.errors.append(error_msg)

        context_desc = ""
        if symbol:
            context_desc += f" {symbol}"
        if phase:
            context_desc += f" in {phase}"

        update_pipeline_context({"last_error": error_msg, "error_symbol": symbol})

        self.logger.error(
            f"❌ Error{context_desc}: {error_msg}",
            extra={
                "error_message": error_msg,
                "symbol": symbol,
                "phase": phase,
                "error_count": len(self.metrics.errors),
                **kwargs
            }
        )

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics."""
        self.logger.info(
            f"⚡ {operation}",
            extra={
                "operation": operation,
                "duration": duration,
                "performance_metric": True,
                **kwargs
            }
        )

    def complete_pipeline(self, success: bool = True, final_pnl: float = 0.0) -> None:
        """Mark pipeline completion and log final summary."""
        total_duration = time.time() - self.metrics.start_time

        if self.current_phase:
            self.end_phase(self.current_phase)

        # Calculate success rate
        success_rate = 0.0
        if self.metrics.trades_attempted > 0:
            success_rate = (self.metrics.trades_executed / self.metrics.trades_attempted) * 100

        # Build summary
        summary_parts = []
        if self.metrics.symbols_discovered > 0:
            summary_parts.append(f"{self.metrics.symbols_discovered} symbols discovered")
        if self.metrics.signals_generated > 0:
            summary_parts.append(f"{self.metrics.signals_generated} signals generated")
        if self.metrics.trades_executed > 0:
            summary_parts.append(f"{self.metrics.trades_executed} trades executed")
        if self.metrics.total_volume > 0:
            summary_parts.append(f"${self.metrics.total_volume:.2f} volume")
        if final_pnl != 0:
            pnl_icon = "📈" if final_pnl > 0 else "📉"
            summary_parts.append(f"{pnl_icon} P&L: ${final_pnl:.2f}")

        summary = ", ".join(summary_parts) if summary_parts else "No activity"

        status_icon = "🎉" if success else "⚠️"
        status_desc = "completed successfully" if success else "completed with issues"

        self.logger.info(
            f"{status_icon} Pipeline {status_desc}: {summary}",
            extra={
                "pipeline_complete": True,
                "success": success,
                "total_duration": total_duration,
                "success_rate": success_rate,
                "final_pnl": final_pnl,
                "metrics": {
                    "symbols_discovered": self.metrics.symbols_discovered,
                    "signals_generated": self.metrics.signals_generated,
                    "trades_executed": self.metrics.trades_executed,
                    "total_volume": self.metrics.total_volume,
                    "error_count": len(self.metrics.errors),
                }
            }
        )

        # Clear pipeline context
        clear_pipeline_context()

    def _get_progress_indicator(self, phase: str) -> str:
        """Get progress indicator for current phase."""
        phases = ["discovery", "market_data", "evaluation", "risk_check", "execution", "finalization"]
        try:
            current_idx = phases.index(phase)
            progress = (current_idx + 1) / len(phases) * 100
            return ".1f"
        except ValueError:
            return "0.0%"

    def _get_phase_metrics(self, phase: str) -> Dict[str, Any]:
        """Get metrics specific to a phase."""
        metrics = {"duration": self.metrics.phase_durations.get(phase, 0.0)}

        if phase == "discovery":
            metrics["symbols_discovered"] = self.metrics.symbols_discovered
        elif phase == "evaluation":
            metrics["signals_generated"] = self.metrics.signals_generated
        elif phase == "execution":
            metrics["trades_executed"] = self.metrics.trades_executed
            metrics["success_rate"] = (
                (self.metrics.trades_executed / max(self.metrics.trades_attempted, 1)) * 100
                if self.metrics.trades_attempted > 0 else 0
            )

        return metrics


class PipelineContextManager:
    """Async context manager for pipeline execution."""

    def __init__(self, cycle_id: Optional[int] = None, logger_name: str = "pipeline"):
        self.cycle_id = cycle_id
        self.logger_name = logger_name
        self.pipeline_logger: Optional[PipelineLogger] = None

    async def __aenter__(self):
        self.pipeline_logger = PipelineLogger(self.logger_name)
        self.pipeline_logger.start_pipeline(cycle_id=self.cycle_id)
        return self.pipeline_logger

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.pipeline_logger.log_error(f"Pipeline failed: {exc_val}", phase="unknown")
            self.pipeline_logger.complete_pipeline(success=False)
        else:
            self.pipeline_logger.complete_pipeline(success=True)


def pipeline_context(cycle_id: Optional[int] = None, logger_name: str = "pipeline"):
    """Async context manager for pipeline execution with automatic logging."""
    return PipelineContextManager(cycle_id, logger_name)


# Global pipeline logger instance
_pipeline_logger: Optional[PipelineLogger] = None


def get_pipeline_logger(name: str = "trading_pipeline") -> PipelineLogger:
    """Get or create the global pipeline logger instance."""
    global _pipeline_logger
    if _pipeline_logger is None:
        _pipeline_logger = PipelineLogger(name)
    return _pipeline_logger


def log_pipeline_event(event_type: str, **kwargs) -> None:
    """Log a pipeline event using the global logger."""
    logger = get_pipeline_logger()
    if hasattr(logger, f"log_{event_type}"):
        getattr(logger, f"log_{event_type}")(**kwargs)
    else:
        logger.logger.info(f"Pipeline event: {event_type}", extra={"event_type": event_type, **kwargs})


__all__ = [
    "PipelineLogger",
    "PipelineMetrics",
    "pipeline_context",
    "get_pipeline_logger",
    "log_pipeline_event",
]
