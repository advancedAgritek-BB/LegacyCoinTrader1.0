"""
Pump Sniper Integration Module

This module provides integration between the advanced pump sniper system
and the main trading bot infrastructure.
"""

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

from .pump_sniper_orchestrator import PumpSniperOrchestrator, SniperDecision
from .wallet_context import load_wallet_context, WalletContext
from ..utils.telemetry import telemetry

logger = logging.getLogger(__name__)


class PumpSniperIntegration:
    """
    Integration layer between pump sniper and main bot.
    
    Handles:
    - Configuration loading and management
    - Integration with main bot lifecycle
    - Shared resource management
    - Notification routing
    """
    
    def __init__(self, main_config: Dict, dry_run: bool = True, paper_wallet=None):
        self.main_config = main_config
        self.dry_run = dry_run
        self.paper_wallet = paper_wallet
        self.pump_sniper_config = self._load_pump_sniper_config()
        self.wallet_context: WalletContext = load_wallet_context(
            self.pump_sniper_config.get("pump_wallet", {})
        )
        self.pump_sniper_config.setdefault("wallet_context", self.wallet_context)
        self.orchestrator: Optional[PumpSniperOrchestrator] = None
        self.integration_enabled = False
        
    def _load_pump_sniper_config(self) -> Dict:
        """Load pump sniper configuration."""
        config_file = Path("config/pump_sniper_config.yaml")
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Merge with main config
                merged_config = self.main_config.copy()
                merged_config.update(config)
                
                logger.info("Pump sniper configuration loaded successfully")
                return merged_config
            else:
                logger.warning("Pump sniper config file not found, using defaults")
                return self.main_config
                
        except Exception as exc:
            logger.error(f"Failed to load pump sniper config: {exc}")
            return self.main_config
            
    async def start(self) -> bool:
        """Start the pump sniper system."""
        try:
            # Check if enabled
            orchestrator_config = self.pump_sniper_config.get("pump_sniper_orchestrator", {})
            if not orchestrator_config.get("enabled", False):
                logger.info("Pump sniper system disabled in configuration")
                return False

            if not self.wallet_context.is_configured:
                if self.dry_run:
                    logger.warning(
                        "Pump sniper wallet not fully configured; continuing in paper trading mode"
                    )
                else:
                    logger.error(
                        "Pump sniper wallet configuration is incomplete; running live mode is not permitted"
                    )
                    return False

            # Initialize orchestrator with paper trading parameters
            self.orchestrator = PumpSniperOrchestrator(
                self.pump_sniper_config,
                dry_run=self.dry_run,
                paper_wallet=self.paper_wallet,
                wallet_context=self.wallet_context,
            )
            
            # Set up notifications
            self.orchestrator.add_notification_callback(self._handle_sniper_notification)
            
            # Start the orchestrator
            await self.orchestrator.start()
            
            self.integration_enabled = True
            
            telemetry.inc("pump_sniper.system_started")
            logger.info("Pump sniper system started successfully")
            return True
            
        except Exception as exc:
            logger.error(f"Failed to start pump sniper system: {exc}")
            return False
            
    async def stop(self):
        """Stop the pump sniper system."""
        if self.orchestrator:
            try:
                await self.orchestrator.stop()
                logger.info("Pump sniper system stopped")
            except Exception as exc:
                logger.error(f"Error stopping pump sniper system: {exc}")
                
        self.integration_enabled = False
        
    async def _handle_sniper_notification(self, decision: SniperDecision):
        """Handle notifications from the pump sniper."""
        try:
            # Log the decision
            logger.info(
                f"Pump Sniper Decision: {decision.decision} for {decision.token_symbol} "
                f"(confidence: {decision.confidence:.2f})"
            )
            
            # Send telegram notification if configured
            if (self.pump_sniper_config.get("monitoring", {}).get("telegram_notifications", False) and
                hasattr(self, 'telegram_notifier')):
                
                message = self._format_telegram_message(decision)
                await self.telegram_notifier.notify(message)
                
            # Record telemetry
            telemetry.inc(f"pump_sniper.decisions.{decision.decision}")
            telemetry.gauge("pump_sniper.decision_confidence", decision.confidence)
            
        except Exception as exc:
            logger.error(f"Error handling sniper notification: {exc}")
            
    def _format_telegram_message(self, decision: SniperDecision) -> str:
        """Format a telegram message for sniper decisions."""
        symbol = decision.token_symbol
        action = decision.decision.upper()
        confidence = decision.confidence
        
        if decision.decision == "snipe":
            message = f"🎯 PUMP SNIPE: {symbol}\n"
            message += f"💪 Confidence: {confidence:.1%}\n"
            message += f"💰 Size: {decision.position_size_sol:.3f} SOL\n"
            message += f"🛑 Stop Loss: {decision.stop_loss_pct:.1%}\n"
            message += f"🎯 Take Profit: {decision.take_profit_pct:.1%}\n"
            
            if decision.reasoning:
                message += f"📝 Reasoning:\n"
                for reason in decision.reasoning[:3]:  # Top 3 reasons
                    message += f"  • {reason}\n"
                    
        elif decision.decision == "monitor":
            message = f"👀 MONITORING: {symbol}\n"
            message += f"📊 Confidence: {confidence:.1%}\n"
            message += "🔄 Watching for entry opportunity\n"
            
        else:  # ignore
            message = f"❌ IGNORED: {symbol}\n"
            message += f"📉 Confidence: {confidence:.1%}\n"
            
        return message
        
    def get_status(self) -> Dict:
        """Get pump sniper system status."""
        if not self.integration_enabled or not self.orchestrator:
            return {"enabled": False, "status": "stopped"}
            
        try:
            stats = self.orchestrator.get_statistics()
            positions = self.orchestrator.get_active_positions()
            recent_decisions = self.orchestrator.get_recent_decisions(5)
            
            return {
                "enabled": True,
                "status": "running",
                "statistics": stats,
                "active_positions": len(positions),
                "recent_decisions": len(recent_decisions),
                "components_status": {
                    "pump_detector": "running",
                    "pool_analyzer": "running", 
                    "rapid_executor": "running",
                    "risk_manager": "running",
                    "sentiment_analyzer": "running",
                    "momentum_detector": "running"
                }
            }
            
        except Exception as exc:
            logger.error(f"Error getting pump sniper status: {exc}")
            return {"enabled": True, "status": "error", "error": str(exc)}
            
    async def manual_evaluate_token(self, token_mint: str, token_symbol: str) -> Optional[Dict]:
        """Manually trigger evaluation of a token."""
        if not self.integration_enabled or not self.orchestrator:
            return None
            
        try:
            decision = await self.orchestrator.manual_evaluate_token(token_mint, token_symbol)
            
            if decision:
                return {
                    "token_mint": decision.token_mint,
                    "token_symbol": decision.token_symbol,
                    "decision": decision.decision,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "timestamp": decision.timestamp
                }
                
        except Exception as exc:
            logger.error(f"Manual token evaluation failed: {exc}")
            
        return None
        
    def emergency_stop(self):
        """Emergency stop all pump sniper operations."""
        if self.orchestrator:
            self.orchestrator.emergency_stop_all()
            logger.critical("PUMP SNIPER EMERGENCY STOP ACTIVATED")
            
    def resume_operations(self):
        """Resume pump sniper operations."""
        if self.orchestrator:
            self.orchestrator.resume_operations()
            logger.info("Pump sniper operations resumed")
            
    def set_telegram_notifier(self, notifier):
        """Set telegram notifier for integration."""
        self.telegram_notifier = notifier


# Global pump sniper integration instance
_pump_sniper_integration: Optional[PumpSniperIntegration] = None


def get_pump_sniper_integration(main_config: Dict) -> PumpSniperIntegration:
    """Get or create the pump sniper integration instance."""
    global _pump_sniper_integration
    
    if _pump_sniper_integration is None:
        _pump_sniper_integration = PumpSniperIntegration(main_config)
        
    return _pump_sniper_integration


async def start_pump_sniper_system(main_config: Dict, dry_run: bool = True, paper_wallet=None) -> bool:
    """Start the pump sniper system."""
    global _pump_sniper_integration

    # Create new integration instance with paper trading parameters
    _pump_sniper_integration = PumpSniperIntegration(main_config, dry_run, paper_wallet)
    return await _pump_sniper_integration.start()


async def stop_pump_sniper_system():
    """Stop the pump sniper system."""
    if _pump_sniper_integration:
        await _pump_sniper_integration.stop()


def get_pump_sniper_status() -> Dict:
    """Get pump sniper system status."""
    if _pump_sniper_integration:
        return _pump_sniper_integration.get_status()
    return {"enabled": False, "status": "not_initialized"}


async def manual_evaluate_token(token_mint: str, token_symbol: str) -> Optional[Dict]:
    """Manually evaluate a token for sniping."""
    if _pump_sniper_integration:
        return await _pump_sniper_integration.manual_evaluate_token(token_mint, token_symbol)
    return None


def emergency_stop_pump_sniper():
    """Emergency stop pump sniper."""
    if _pump_sniper_integration:
        _pump_sniper_integration.emergency_stop()


def resume_pump_sniper():
    """Resume pump sniper operations."""
    if _pump_sniper_integration:
        _pump_sniper_integration.resume_operations()
