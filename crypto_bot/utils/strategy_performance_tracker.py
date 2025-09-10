"""Strategy Performance Tracking System

This module tracks strategy performance and usage to ensure balanced utilization
and identify the most effective strategies for different market conditions.
"""

import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from crypto_bot.utils.logger import LOG_DIR

logger = logging.getLogger(__name__)

class StrategyPerformanceTracker:
    """Track strategy performance and usage metrics."""
    
    def __init__(self, data_file: Optional[str] = None):
        self.data_file = Path(data_file) if data_file else LOG_DIR / "strategy_performance.json"
        self.performance_data = self._load_data()
        self.usage_history = defaultdict(lambda: deque(maxlen=1000))
        self.regime_history = defaultdict(lambda: deque(maxlen=1000))
        
    def _load_data(self) -> Dict[str, Any]:
        """Load existing performance data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load strategy performance data: {e}")
        return {
            "strategies": {},
            "regimes": {},
            "last_updated": datetime.now().isoformat(),
            "total_signals": 0
        }
    
    def _save_data(self):
        """Save performance data to file."""
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save strategy performance data: {e}")
    
    def record_strategy_usage(self, strategy_name: str, regime: str, signal_strength: float, direction: str):
        """Record strategy usage and signal."""
        now = datetime.now()
        
        # Update usage history
        self.usage_history[strategy_name].append(now)
        self.regime_history[regime].append(now)
        
        # Initialize strategy data if not exists
        if strategy_name not in self.performance_data["strategies"]:
            self.performance_data["strategies"][strategy_name] = {
                "total_usage": 0,
                "regime_usage": defaultdict(int),
                "signal_strengths": [],
                "directions": defaultdict(int),
                "last_used": None,
                "avg_signal_strength": 0.0,
                "success_rate": 0.0,
                "total_pnl": 0.0
            }
        
        # Update strategy metrics
        strategy_data = self.performance_data["strategies"][strategy_name]
        strategy_data["total_usage"] += 1
        strategy_data["regime_usage"][regime] += 1
        strategy_data["signal_strengths"].append(signal_strength)
        strategy_data["directions"][direction] += 1
        strategy_data["last_used"] = now.isoformat()
        
        # Calculate average signal strength
        if strategy_data["signal_strengths"]:
            strategy_data["avg_signal_strength"] = sum(strategy_data["signal_strengths"]) / len(strategy_data["signal_strengths"])
        
        # Update regime data
        if regime not in self.performance_data["regimes"]:
            self.performance_data["regimes"][regime] = {
                "total_occurrences": 0,
                "strategy_usage": defaultdict(int),
                "last_occurrence": None
            }
        
        regime_data = self.performance_data["regimes"][regime]
        regime_data["total_occurrences"] += 1
        regime_data["strategy_usage"][strategy_name] += 1
        regime_data["last_occurrence"] = now.isoformat()
        
        # Update total signals
        self.performance_data["total_signals"] += 1
        self.performance_data["last_updated"] = now.isoformat()
        
        # Save data periodically
        if self.performance_data["total_signals"] % 100 == 0:
            self._save_data()
    
    def record_trade_result(self, strategy_name: str, pnl: float, success: bool):
        """Record trade result for strategy performance calculation."""
        if strategy_name not in self.performance_data["strategies"]:
            return
        
        strategy_data = self.performance_data["strategies"][strategy_name]
        strategy_data["total_pnl"] += pnl
        
        # Calculate success rate
        if "trades" not in strategy_data:
            strategy_data["trades"] = {"successful": 0, "total": 0}
        
        strategy_data["trades"]["total"] += 1
        if success:
            strategy_data["trades"]["successful"] += 1
        
        strategy_data["success_rate"] = strategy_data["trades"]["successful"] / strategy_data["trades"]["total"]
    
    def get_strategy_usage_stats(self, strategy_name: str) -> Dict[str, Any]:
        """Get usage statistics for a specific strategy."""
        if strategy_name not in self.performance_data["strategies"]:
            return {}
        
        strategy_data = self.performance_data["strategies"][strategy_name]
        usage_count = len(self.usage_history[strategy_name])
        
        return {
            "total_usage": strategy_data["total_usage"],
            "recent_usage": usage_count,
            "regime_usage": dict(strategy_data["regime_usage"]),
            "avg_signal_strength": strategy_data["avg_signal_strength"],
            "success_rate": strategy_data["success_rate"],
            "total_pnl": strategy_data["total_pnl"],
            "last_used": strategy_data["last_used"],
            "directions": dict(strategy_data["directions"])
        }
    
    def get_regime_stats(self, regime: str) -> Dict[str, Any]:
        """Get statistics for a specific regime."""
        if regime not in self.performance_data["regimes"]:
            return {}
        
        regime_data = self.performance_data["regimes"][regime]
        occurrence_count = len(self.regime_history[regime])
        
        return {
            "total_occurrences": regime_data["total_occurrences"],
            "recent_occurrences": occurrence_count,
            "strategy_usage": dict(regime_data["strategy_usage"]),
            "last_occurrence": regime_data["last_occurrence"]
        }
    
    def get_underutilized_strategies(self, min_usage_threshold: int = 10) -> List[str]:
        """Get list of underutilized strategies."""
        underutilized = []
        for strategy_name, strategy_data in self.performance_data["strategies"].items():
            if strategy_data["total_usage"] < min_usage_threshold:
                underutilized.append(strategy_name)
        return underutilized
    
    def get_overutilized_strategies(self, max_usage_percentage: float = 0.4) -> List[str]:
        """Get list of overutilized strategies."""
        total_signals = self.performance_data["total_signals"]
        if total_signals == 0:
            return []
        
        overutilized = []
        for strategy_name, strategy_data in self.performance_data["strategies"].items():
            usage_percentage = strategy_data["total_usage"] / total_signals
            if usage_percentage > max_usage_percentage:
                overutilized.append(strategy_name)
        return overutilized
    
    def get_strategy_recommendations(self) -> Dict[str, List[str]]:
        """Get recommendations for strategy usage optimization."""
        recommendations = {
            "increase_usage": [],
            "decrease_usage": [],
            "monitor_performance": []
        }
        
        # Check for underutilized strategies
        underutilized = self.get_underutilized_strategies()
        recommendations["increase_usage"] = underutilized
        
        # Check for overutilized strategies
        overutilized = self.get_overutilized_strategies()
        recommendations["decrease_usage"] = overutilized
        
        # Check for strategies with poor performance
        for strategy_name, strategy_data in self.performance_data["strategies"].items():
            if strategy_data["total_usage"] >= 10:  # Only check strategies with sufficient data
                if strategy_data["success_rate"] < 0.3:  # Less than 30% success rate
                    recommendations["monitor_performance"].append(strategy_name)
        
        return recommendations
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("=== Strategy Performance Report ===")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Signals: {self.performance_data['total_signals']}")
        report.append("")
        
        # Strategy usage summary
        report.append("Strategy Usage Summary:")
        for strategy_name, strategy_data in self.performance_data["strategies"].items():
            usage_pct = (strategy_data["total_usage"] / self.performance_data["total_signals"] * 100) if self.performance_data["total_signals"] > 0 else 0
            report.append(f"  {strategy_name}: {strategy_data['total_usage']} uses ({usage_pct:.1f}%)")
        report.append("")
        
        # Regime distribution
        report.append("Regime Distribution:")
        for regime_name, regime_data in self.performance_data["regimes"].items():
            report.append(f"  {regime_name}: {regime_data['total_occurrences']} occurrences")
        report.append("")
        
        # Recommendations
        recommendations = self.get_strategy_recommendations()
        report.append("Recommendations:")
        if recommendations["increase_usage"]:
            report.append(f"  Increase usage: {', '.join(recommendations['increase_usage'])}")
        if recommendations["decrease_usage"]:
            report.append(f"  Decrease usage: {', '.join(recommendations['decrease_usage'])}")
        if recommendations["monitor_performance"]:
            report.append(f"  Monitor performance: {', '.join(recommendations['monitor_performance'])}")
        
        return "\n".join(report)
    
    def export_to_csv(self, output_file: Optional[str] = None):
        """Export performance data to CSV format."""
        if output_file is None:
            output_file = LOG_DIR / "strategy_performance.csv"
        
        output_path = Path(output_file)
        
        # Create DataFrame from performance data
        rows = []
        for strategy_name, strategy_data in self.performance_data["strategies"].items():
            row = {
                "strategy": strategy_name,
                "total_usage": strategy_data["total_usage"],
                "avg_signal_strength": strategy_data["avg_signal_strength"],
                "success_rate": strategy_data["success_rate"],
                "total_pnl": strategy_data["total_pnl"],
                "last_used": strategy_data["last_used"]
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Strategy performance data exported to {output_path}")

# Global instance
performance_tracker = StrategyPerformanceTracker()

def record_strategy_usage(strategy_name: str, regime: str, signal_strength: float, direction: str):
    """Convenience function to record strategy usage."""
    performance_tracker.record_strategy_usage(strategy_name, regime, signal_strength, direction)

def record_trade_result(strategy_name: str, pnl: float, success: bool):
    """Convenience function to record trade result."""
    performance_tracker.record_trade_result(strategy_name, pnl, success)

def get_strategy_recommendations() -> Dict[str, List[str]]:
    """Convenience function to get strategy recommendations."""
    return performance_tracker.get_strategy_recommendations()
