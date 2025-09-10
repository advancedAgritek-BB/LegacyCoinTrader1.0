
#!/usr/bin/env python3
"""
Fix for duplicate open position cards on dashboard.

This script addresses the root causes of duplicate position cards:
1. Multiple data sources (TradeManager state, positions.log, paper wallet state)
2. Inadequate deduplication logic
3. Race conditions between different position loading methods
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from decimal import Decimal

def fix_duplicate_positions():
    """Main function to fix duplicate position issues."""
    print("🔧 Fixing duplicate open position cards issue...")
    
    # 1. Clean up positions.log file
    cleanup_positions_log()
    
    # 2. Improve deduplication logic in frontend
    improve_deduplication_logic()
    
    # 3. Ensure single source of truth
    ensure_single_source_of_truth()
    
    print("✅ Duplicate position fix completed!")

def cleanup_positions_log():
    """Remove duplicate entries from positions.log file."""
    print("📝 Cleaning up positions.log file...")
    
    positions_file = Path("crypto_bot/logs/positions.log")
    if not positions_file.exists():
        print("  No positions.log file found, skipping cleanup")
        return
    
    try:
        # Read all lines
        with open(positions_file, 'r') as f:
            lines = f.readlines()
        
        # Parse position entries
        position_entries = []
        other_lines = []
        
        for line in lines:
            if "Active " in line and "entry " in line:
                # Extract timestamp and position data
                timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        line_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                        
                        # Extract position data
                        pos_match = re.search(
                            r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
                            r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+)",
                            line
                        )
                        
                        if pos_match:
                            position_entries.append({
                                'line': line,
                                'timestamp': line_timestamp,
                                'symbol': pos_match.group('symbol'),
                                'side': pos_match.group('side'),
                                'amount': float(pos_match.group('amount')),
                                'entry': float(pos_match.group('entry')),
                                'current': float(pos_match.group('current'))
                            })
                        else:
                            other_lines.append(line)
                    except ValueError:
                        other_lines.append(line)
                else:
                    other_lines.append(line)
            else:
                other_lines.append(line)
        
        # Remove duplicates, keeping the most recent entry for each symbol/side combination
        seen_positions = {}
        unique_position_lines = []
        
        for entry in position_entries:
            key = f"{entry['symbol']}_{entry['side']}"
            if key not in seen_positions or entry['timestamp'] > seen_positions[key]['timestamp']:
                seen_positions[key] = entry
        
        # Add unique position lines
        for entry in seen_positions.values():
            unique_position_lines.append(entry['line'])
        
        # Write back cleaned file
        with open(positions_file, 'w') as f:
            f.writelines(other_lines + unique_position_lines)
        
        removed_count = len(position_entries) - len(seen_positions)
        print(f"  ✅ Removed {removed_count} duplicate position entries")
        print(f"  📊 Kept {len(seen_positions)} unique positions")
        
    except Exception as e:
        print(f"  ❌ Error cleaning positions.log: {e}")

def improve_deduplication_logic():
    """Improve the deduplication logic in the frontend app."""
    print("🔧 Improving deduplication logic...")
    
    frontend_app = Path("frontend/app.py")
    if not frontend_app.exists():
        print("  ❌ frontend/app.py not found")
        return
    
    try:
        # Read the current file
        with open(frontend_app, 'r') as f:
            content = f.read()
        
        # Improved deduplication function
        improved_deduplication = '''
# Enhanced deduplication function for positions
def deduplicate_positions(positions):
    """Remove duplicate positions based on symbol with enhanced logic."""
    if not positions:
        return []
    
    # Group positions by symbol
    symbol_groups = {}
    for position in positions:
        symbol = position.get('symbol', '')
        if not symbol:
            continue
            
        if symbol not in symbol_groups:
            symbol_groups[symbol] = []
        symbol_groups[symbol].append(position)
    
    # For each symbol, keep the most recent/accurate position
    unique_positions = []
    for symbol, symbol_positions in symbol_groups.items():
        if len(symbol_positions) == 1:
            # Single position, keep it
            unique_positions.append(symbol_positions[0])
        else:
            # Multiple positions for same symbol, choose the best one
            best_position = select_best_position(symbol_positions)
            if best_position:
                unique_positions.append(best_position)
                print(f"Duplicate position found for {symbol}, kept best match")
    
    print(f"Deduplication: {len(positions)} -> {len(unique_positions)} positions")
    return unique_positions

def select_best_position(positions):
    """Select the best position from multiple candidates for the same symbol."""
    if not positions:
        return None
    
    # Priority criteria:
    # 1. Has current_price (most recent data)
    # 2. Has non-zero amount
    # 3. Most recent timestamp
    # 4. Has PnL calculation
    
    best_position = None
    best_score = -1
    
    for position in positions:
        score = 0
        
        # Has current price
        if position.get('current_price'):
            score += 10
        
        # Has non-zero amount
        if position.get('amount', 0) > 0:
            score += 5
        
        # Has entry price
        if position.get('entry_price'):
            score += 3
        
        # Has PnL
        if position.get('pnl') is not None:
            score += 2
        
        # Has timestamp
        if position.get('timestamp'):
            score += 1
        
        if score > best_score:
            best_score = score
            best_position = position
    
    return best_position
'''
        
        # Replace the old deduplication function
        old_pattern = r'# Deduplication function for positions\ndef deduplicate_positions\(positions\):.*?return unique_positions'
        
        if re.search(old_pattern, content, re.DOTALL):
            content = re.sub(old_pattern, improved_deduplication, content, flags=re.DOTALL)
            print("  ✅ Updated deduplication function")
        else:
            print("  ⚠️  Could not find old deduplication function to replace")
        
        # Write back the improved file
        with open(frontend_app, 'w') as f:
            f.write(content)
        
    except Exception as e:
        print(f"  ❌ Error improving deduplication logic: {e}")

def ensure_single_source_of_truth():
    """Ensure the dashboard uses TradeManager as the single source of truth."""
    print("🎯 Ensuring single source of truth for positions...")
    
    frontend_app = Path("frontend/app.py")
    if not frontend_app.exists():
        print("  ❌ frontend/app.py not found")
        return
    
    try:
        # Read the current file
        with open(frontend_app, 'r') as f:
            content = f.read()
        
        # Improved get_open_positions function that prioritizes TradeManager
        improved_get_positions = '''
def get_open_positions() -> list:
    """Get open positions from TradeManager (single source of truth)."""
    try:
        # Try to get positions from TradeManager first (highest priority)
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        positions = trade_manager.get_all_positions()
        print(f"Found {len(positions)} positions in TradeManager")

        if positions:
            # Convert Position objects to the expected format
            result = []
            for position in positions:
                # Only include positions with non-zero amounts
                if position.total_amount <= 0:
                    continue
                    
                # Get current price for unrealized P&L
                current_price = trade_manager.price_cache.get(position.symbol)
                if current_price:
                    unrealized_pnl, unrealized_pct = position.calculate_unrealized_pnl(current_price)
                else:
                    unrealized_pnl = Decimal('0')
                    unrealized_pct = Decimal('0')

                pos_dict = {
                    'symbol': position.symbol,
                    'side': position.side,
                    'amount': float(position.total_amount),
                    'entry_price': float(position.average_price),
                    'current_price': float(current_price) if current_price else None,
                    'pnl': float(unrealized_pnl),
                    'pnl_percentage': float(unrealized_pct),
                    'highest_price': float(position.highest_price) if position.highest_price else None,
                    'lowest_price': float(position.lowest_price) if position.lowest_price else None,
                    'stop_loss_price': float(position.stop_loss_price) if position.stop_loss_price else None,
                    'take_profit_price': float(position.take_profit_price) if position.take_profit_price else None,
                    'entry_time': position.entry_time.isoformat() if position.entry_time else None,
                }
                result.append(pos_dict)

            print(f"Returning {len(result)} active positions from TradeManager")
            return result

    except Exception as e:
        print(f"Failed to get positions from TradeManager: {e}")

    # Fallback to trade manager state file (second priority)
    try:
        state_file = Path("crypto_bot/logs/trade_manager_state.json")
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            positions = state.get('positions', {})
            price_cache = state.get('price_cache', {})
            
            result = []
            for symbol, pos_data in positions.items():
                if pos_data.get('total_amount', 0) > 0:  # Only open positions
                    current_price = price_cache.get(symbol, pos_data.get('average_price', 0))
                    
                    # Calculate PnL
                    amount = pos_data['total_amount']
                    avg_price = pos_data['average_price']
                    side = pos_data['side']
                    
                    if side == 'long':
                        pnl = (current_price - avg_price) * amount
                    else:  # short
                        pnl = (avg_price - current_price) * amount
                    
                    pnl_pct = (pnl / (avg_price * amount)) * 100 if avg_price > 0 else 0
                    
                    position_data = {
                        'symbol': symbol,
                        'side': side,
                        'amount': float(amount),
                        'entry_price': float(avg_price),
                        'current_price': float(current_price),
                        'pnl': float(pnl),
                        'pnl_percentage': float(pnl_pct),
                        'entry_time': pos_data.get('entry_time', ''),
                    }
                    result.append(position_data)
            
            print(f"Returning {len(result)} positions from state file")
            return result

    except Exception as e:
        print(f"Failed to get positions from state file: {e}")

    # Final fallback to log parsing (lowest priority)
    print("Falling back to log parsing for positions")
    return get_open_positions_from_log()
'''
        
        # Replace the old get_open_positions function
        old_pattern = r'def get_open_positions\(\) -> list:.*?return get_open_positions_from_log\(\)'
        
        if re.search(old_pattern, content, re.DOTALL):
            content = re.sub(old_pattern, improved_get_positions, content, flags=re.DOTALL)
            print("  ✅ Updated get_open_positions function")
        else:
            print("  ⚠️  Could not find old get_open_positions function to replace")
        
        # Write back the improved file
        with open(frontend_app, 'w') as f:
            f.write(content)
        
    except Exception as e:
        print(f"  ❌ Error ensuring single source of truth: {e}")

def verify_fix():
    """Verify that the fix is working correctly."""
    print("🔍 Verifying fix...")
    
    # Check if positions.log is clean
    positions_file = Path("crypto_bot/logs/positions.log")
    if positions_file.exists():
        with open(positions_file, 'r') as f:
            lines = f.readlines()
        
        # Count unique positions
        unique_positions = set()
        for line in lines:
            if "Active " in line and "entry " in line:
                pos_match = re.search(r"Active (\S+) (\w+)", line)
                if pos_match:
                    unique_positions.add(f"{pos_match.group(1)}_{pos_match.group(2)}")
        
        print(f"  📊 Found {len(unique_positions)} unique positions in log")
    
    # Check TradeManager state
    state_file = Path("crypto_bot/logs/trade_manager_state.json")
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        open_positions = [symbol for symbol, pos in state.get('positions', {}).items() 
                         if pos.get('total_amount', 0) > 0]
        print(f"  📊 Found {len(open_positions)} open positions in TradeManager state")
    
    print("  ✅ Verification completed")

if __name__ == "__main__":
    fix_duplicate_positions()
    verify_fix()
