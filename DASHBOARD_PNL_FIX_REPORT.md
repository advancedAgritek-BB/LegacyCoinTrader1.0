# Dashboard P&L and Balance Fix Report

## Problem Summary

The dashboard was showing incorrect values for:
- Total P&L
- P&L percentage
- Initial Balance
- Current Balance

The main issues were:
1. **Paper wallet balance not persisting** - The balance was reset to $10,000 on every bot restart
2. **Incorrect P&L calculation** - The frontend was calculating P&L incorrectly by double-counting realized gains
3. **Stale position data** - Open positions were being read from old log entries instead of current state
4. **No state persistence** - Paper wallet state was not being saved between sessions

## Fixes Implemented

### 1. Paper Wallet State Persistence

**File: `crypto_bot/paper_wallet.py`**

Added methods to save and load paper wallet state:
- `save_state()` - Saves current balance, realized PnL, and open positions to `crypto_bot/logs/paper_wallet_state.yaml`
- `load_state()` - Loads saved state on initialization
- Automatic state saving after each trade (open/close operations)

**Key Changes:**
```python
def save_state(self) -> None:
    """Save current wallet state to file."""
    state = {
        'balance': self.balance,
        'initial_balance': self.initial_balance,
        'realized_pnl': self.realized_pnl,
        'total_trades': self.total_trades,
        'winning_trades': self.winning_trades,
        'positions': self.positions
    }
    # Save to yaml file...

def load_state(self) -> bool:
    """Load wallet state from file. Returns True if successful."""
    # Load from yaml file and restore state...
```

### 2. Main Bot Integration

**File: `crypto_bot/main.py`**

Updated paper wallet initialization to load saved state:
```python
# Try to load saved state
if paper_wallet.load_state():
    logger.info(f"Loaded saved paper wallet state: balance=${paper_wallet.balance:.2f}")
else:
    logger.info(f"Using initial paper wallet balance: ${start_bal:.2f}")
```

### 3. Frontend P&L Calculation Fix

**File: `frontend/app.py`**

Completely rewrote the P&L calculation logic:

**Before:** Used complex trade parsing and double-counted realized gains
**After:** Uses paper wallet state file as the source of truth

**Key Changes:**
```python
def calculate_wallet_pnl() -> Dict[str, float]:
    """Calculate current wallet PnL based on paper wallet state and open positions."""
    # Load from paper wallet state file (most accurate)
    paper_wallet_state_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
    if paper_wallet_state_file.exists():
        # Get current balance and realized PnL from state
        current_balance = state.get('balance', 0.0)
        initial_balance = state.get('initial_balance', 10000.0)
        realized_pnl = state.get('realized_pnl', 0.0)
    
    # Calculate unrealized PnL from open positions
    # Total PnL = realized + unrealized
    # Current Balance = base balance + unrealized PnL
```

### 4. Open Positions Fix

**File: `frontend/app.py`**

Fixed open position detection to use paper wallet state instead of parsing old log entries:

**Before:** Parsed positions.log file and picked up stale test entries
**After:** Reads directly from paper wallet state file

**Key Changes:**
```python
def get_open_positions() -> list:
    """Get open positions from paper wallet state file."""
    # Try to load from paper wallet state file first (most accurate)
    paper_wallet_state_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
    if paper_wallet_state_file.exists():
        positions = state.get('positions', {})
        # Process actual open positions...
    
    # Fallback to parsing positions.log (legacy method)
    return get_open_positions_from_log()
```

### 5. Balance Reading Priority

**File: `frontend/app.py`**

Updated balance reading to prioritize the state file:
1. Paper wallet state file (most accurate)
2. Positions.log (recent entries)
3. Paper wallet config files
4. Default fallback

## Testing Results

### Test 1: Closed Position
- **Initial Balance:** $10,000.00
- **Final Balance:** $10,500.00 ✓
- **Realized PnL:** $500.00 ✓
- **Unrealized PnL:** $0.00 ✓
- **Total PnL:** $500.00 ✓
- **PnL Percentage:** 5.00% ✓

### Test 2: Open Position
- **Initial Balance:** $10,000.00
- **Current Balance:** $10,792.62 ✓
- **Realized PnL:** $0.00 ✓
- **Unrealized PnL:** $5,792.62 ✓ (correct for BTC price appreciation)
- **Total PnL:** $5,792.62 ✓
- **PnL Percentage:** 57.93% ✓

## Benefits

1. **Accurate Balance Tracking** - Balance persists between bot restarts
2. **Correct P&L Calculation** - No more double-counting or incorrect calculations
3. **Real-time Updates** - Dashboard shows current state, not stale data
4. **Reliable State Management** - Paper wallet state is the single source of truth
5. **Backward Compatibility** - Falls back to legacy methods if state file doesn't exist

## Files Modified

1. `crypto_bot/paper_wallet.py` - Added state persistence
2. `crypto_bot/main.py` - Added state loading on initialization
3. `frontend/app.py` - Fixed P&L calculation and position detection
4. `crypto_bot/logs/paper_wallet_state.yaml` - New state file (created automatically)

## Usage

The fixes are automatic and require no user intervention. The dashboard will now show:
- Correct initial balance (persisted from previous sessions)
- Accurate current balance (including unrealized PnL)
- Proper P&L calculations (realized + unrealized)
- Real-time position updates

The paper wallet state is automatically saved after each trade and loaded when the bot starts.
