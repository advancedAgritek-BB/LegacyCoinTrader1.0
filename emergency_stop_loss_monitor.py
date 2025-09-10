#!/usr/bin/env python3
"""
EMERGENCY STOP LOSS MONITOR

This script provides emergency monitoring for active positions
when the main bot's stop loss system is not working properly.
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List

class EmergencyStopLossMonitor:
    def __init__(self):
        self.positions_log = Path("crypto_bot/logs/positions.log")
        self.config_path = Path("crypto_bot/config.yaml")
        self.active_positions = {}
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return {"exit_strategy": {"stop_loss_pct": 0.01, "trailing_stop_pct": 0.008}}
    
    def get_active_positions(self):
        """Get active positions from log."""
        if not self.positions_log.exists():
            return {}
        
        try:
            with open(self.positions_log, 'r') as f:
                lines = f.readlines()
            
            positions = {}
            for line in lines[-50:]:  # Check last 50 lines
                if "Active" in line and "entry" in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        symbol = parts[1]
                        side = parts[2]
                        size = float(parts[3])
                        entry_price = float(parts[5])
                        
                        positions[symbol] = {
                            "side": side,
                            "size": size,
                            "entry_price": entry_price,
                            "current_price": entry_price,
                            "highest_price": entry_price,
                            "lowest_price": entry_price,
                            "trailing_stop": entry_price * (1 - 0.01) if side == "buy" else entry_price * (1 + 0.01),
                            "last_update": time.time()
                        }
            
            return positions
            
        except Exception as e:
            print(f"Failed to get active positions: {e}")
            return {}
    
    def calculate_stop_loss(self, position: Dict, current_price: float) -> float:
        """Calculate stop loss for a position."""
        exit_cfg = self.config.get("exit_strategy", {})
        stop_loss_pct = exit_cfg.get("stop_loss_pct", 0.01)
        
        if position["side"] == "buy":
            return position["entry_price"] * (1 - stop_loss_pct)
        else:
            return position["entry_price"] * (1 + stop_loss_pct)
    
    def calculate_trailing_stop(self, position: Dict, current_price: float) -> float:
        """Calculate trailing stop for a position."""
        exit_cfg = self.config.get("exit_strategy", {})
        trailing_stop_pct = exit_cfg.get("trailing_stop_pct", 0.008)
        min_gain_to_trail = exit_cfg.get("min_gain_to_trail", 0.005)
        
        # Calculate current PnL
        entry_price = position["entry_price"]
        pnl_pct = ((current_price - entry_price) / entry_price) * (1 if position["side"] == "buy" else -1)
        
        # Only trail if we're in profit beyond minimum threshold
        if pnl_pct >= min_gain_to_trail:
            if position["side"] == "buy":
                # Update highest price
                if current_price > position["highest_price"]:
                    position["highest_price"] = current_price
                return position["highest_price"] * (1 - trailing_stop_pct)
            else:
                # Update lowest price
                if current_price < position["lowest_price"]:
                    position["lowest_price"] = current_price
                return position["lowest_price"] * (1 + trailing_stop_pct)
        
        # Return original stop loss if not trailing
        return self.calculate_stop_loss(position, current_price)
    
    def should_exit(self, symbol: str, position: Dict, current_price: float) -> tuple[bool, str]:
        """Determine if position should be exited."""
        # Check stop loss
        stop_loss = self.calculate_stop_loss(position, current_price)
        if position["side"] == "buy" and current_price <= stop_loss:
            return True, "stop_loss"
        elif position["side"] == "sell" and current_price >= stop_loss:
            return True, "stop_loss"
        
        # Check trailing stop
        trailing_stop = self.calculate_trailing_stop(position, current_price)
        if position["side"] == "buy" and current_price <= trailing_stop:
            return True, "trailing_stop"
        elif position["side"] == "sell" and current_price >= trailing_stop:
            return True, "trailing_stop"
        
        # Check take profit
        exit_cfg = self.config.get("exit_strategy", {})
        take_profit_pct = exit_cfg.get("take_profit_pct", 0.04)
        if position["side"] == "buy":
            take_profit_price = position["entry_price"] * (1 + take_profit_pct)
            if current_price >= take_profit_price:
                return True, "take_profit"
        else:
            take_profit_price = position["entry_price"] * (1 - take_profit_pct)
            if current_price <= take_profit_price:
                return True, "take_profit"
        
        return False, ""
    
    async def monitor_positions(self):
        """Monitor active positions for exit conditions."""
        print("🚨 EMERGENCY STOP LOSS MONITOR STARTED")
        print("=" * 50)
        
        while True:
            try:
                # Get active positions
                self.active_positions = self.get_active_positions()
                
                if not self.active_positions:
                    print("No active positions found")
                    await asyncio.sleep(30)
                    continue
                
                print(f"\n📊 Monitoring {len(self.active_positions)} active positions")
                print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                for symbol, position in self.active_positions.items():
                    # Simulate current price (in real implementation, this would fetch from exchange)
                    # For now, we'll use a simple simulation
                    current_price = position["current_price"]
                    
                    # Update trailing stop
                    new_trailing_stop = self.calculate_trailing_stop(position, current_price)
                    if new_trailing_stop != position["trailing_stop"]:
                        position["trailing_stop"] = new_trailing_stop
                        print(f"📈 {symbol}: Trailing stop updated to ${new_trailing_stop:.6f}")
                    
                    # Check exit conditions
                    should_exit, exit_reason = self.should_exit(symbol, position, current_price)
                    
                    if should_exit:
                        print(f"🚨 EXIT SIGNAL: {symbol} - {exit_reason.upper()}")
                        print(f"   Side: {position['side']}")
                        print(f"   Size: {position['size']}")
                        print(f"   Entry: ${position['entry_price']:.6f}")
                        print(f"   Current: ${current_price:.6f}")
                        print(f"   Stop: ${position['trailing_stop']:.6f}")
                        
                        # In emergency mode, we just log the exit signal
                        # The main bot should handle the actual execution
                        print(f"   ⚠️  EMERGENCY EXIT SIGNAL LOGGED")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"❌ Error in emergency monitor: {e}")
                await asyncio.sleep(30)

async def main():
    """Main function for emergency monitoring."""
    monitor = EmergencyStopLossMonitor()
    await monitor.monitor_positions()

if __name__ == "__main__":
    asyncio.run(main())
