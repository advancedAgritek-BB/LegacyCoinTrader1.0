# Single Source of Truth Architecture

## 🎯 Overview

This document describes the new **Single Source of Truth** architecture implemented for trade data management. The core principle is that **all trade and position data flows through one central system**, eliminating synchronization issues and ensuring consistency across the entire application.

## 🏗️ Architecture Components

### 1. SingleSourceTradeManager
**Location**: `crypto_bot/utils/single_source_trade_manager.py`

The enhanced TradeManager that serves as the single source of truth for:
- ✅ All trade records
- ✅ All position calculations
- ✅ Price cache management
- ✅ Risk management
- ✅ Event-driven notifications

### 2. Event-Driven System
**Components**:
- `TradeEventBus`: Asynchronous event processing system
- `TradeEvent`: Structured event data format
- Component-specific subscribers for real-time updates

### 3. Component Integration
**Frontend**: Subscribes to trade events for real-time UI updates
**Portfolio Service**: Receives trade notifications for external systems
**Notifications**: Sends alerts for important trading events
**CSV Audit**: Maintains audit trail (not primary storage)

## 🔄 Data Flow

```
Trade Execution → SingleSourceTradeManager → Event Bus → Subscribers
     ↓                    ↓                     ↓            ↓
  CEX Executor      Primary Storage      Frontend     Portfolio Service
     ↓                    ↓                     ↓            ↓
  CSV Audit        JSON Persistence     UI Updates   External Sync
```

## 📋 Key Principles

### ✅ Single Source of Truth
- **TradeManager owns all trade data**
- **No duplicate storage** - CSV is audit-only
- **Consistent calculations** across all components
- **Event-driven updates** prevent manual sync

### ✅ Event-Driven Architecture
- **Real-time notifications** when trades execute
- **Component decoupling** through events
- **Asynchronous processing** for performance
- **Reliable delivery** with error handling

### ✅ Backward Compatibility
- **Legacy functions** still work
- **Gradual migration** path
- **No breaking changes** for existing code

## 🚀 Usage Guide

### Basic Usage

```python
from crypto_bot.utils.single_source_trade_manager import get_single_source_trade_manager

# Get the single instance
trade_manager = get_single_source_trade_manager()

# All operations go through this single instance
trades = trade_manager.get_all_positions()
status = trade_manager.get_system_status()
```

### Frontend Integration

```python
from crypto_bot.utils.single_source_trade_manager import (
    get_single_source_trade_manager,
    create_frontend_subscriber
)

trade_manager = get_single_source_trade_manager()

def handle_trade_update(event):
    """Handle real-time trade updates in frontend."""
    if event.event_type == 'trade_executed':
        # Update UI with new trade
        update_frontend_positions(event.trade)
    elif event.event_type == 'position_updated':
        # Update position display
        refresh_position_display(event.position)

# Subscribe to events
subscriber = create_frontend_subscriber(handle_trade_update)
trade_manager.add_frontend_subscriber(subscriber)
```

### Portfolio Service Integration

```python
from crypto_bot.utils.single_source_trade_manager import (
    get_single_source_trade_manager,
    create_portfolio_subscriber
)

trade_manager = get_single_source_trade_manager()

def sync_to_portfolio_service(event):
    """Sync trade data to external portfolio service."""
    if event.event_type == 'trade_executed':
        portfolio_api.record_trade(event.trade)

# Subscribe to events
subscriber = create_portfolio_subscriber(sync_to_portfolio_service)
trade_manager.add_portfolio_service_subscriber(subscriber)
```

### Notification Integration

```python
from crypto_bot.utils.single_source_trade_manager import (
    get_single_source_trade_manager,
    create_notification_subscriber
)
from libs.notifications import TelegramNotifier

notifier = TelegramNotifier(token="your_token", chat_id="your_chat")

trade_manager = get_single_source_trade_manager()

# Subscribe to notifications
subscriber = create_notification_subscriber(notifier, {})
trade_manager.add_notification_subscriber(subscriber)
```

## 🔧 Migration Guide

### From Legacy TradeManager

```python
# Old way (multiple sources)
from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot.utils.trade_logger import log_trade

trade_manager = get_trade_manager()
# ... do operations ...

# Manual CSV logging
log_trade(order)

# New way (single source)
from crypto_bot.utils.single_source_trade_manager import get_single_source_trade_manager

trade_manager = get_single_source_trade_manager()
# All operations go through single instance
# CSV logging happens automatically as audit trail
```

### From Manual Synchronization

```python
# Old way (error-prone manual sync)
# CSV and TradeManager get out of sync
# Need manual reconciliation scripts

# New way (automatic sync)
# Everything flows through single source
# No manual sync needed
trade_manager = get_single_source_trade_manager()
# Events automatically notify all subscribers
```

## 🧪 Testing

Run the validation tests:

```bash
python3 test_single_source_truth.py
```

### Test Results Summary
- ✅ **Single Instance**: Same TradeManager instance returned
- ✅ **Trade Recording**: Trades recorded successfully
- ✅ **Event System**: Events published and received correctly
- ✅ **CSV Audit**: Audit logging works
- ✅ **System Status**: Status reporting functional

## 📊 Benefits

### 🔒 Data Consistency
- **No more sync issues** between CSV and TradeManager
- **Consistent calculations** across all components
- **Atomic operations** prevent race conditions

### 🚀 Performance
- **Event-driven updates** reduce polling overhead
- **Single instance** reduces memory usage
- **Asynchronous processing** improves responsiveness

### 🛠️ Maintainability
- **Clear data flow** makes debugging easier
- **Component isolation** through events
- **Centralized logic** for trade calculations

### 📈 Scalability
- **Event subscribers** can be added/removed dynamically
- **Multiple consumers** can subscribe to the same events
- **Non-blocking** event processing

## 🐛 Troubleshooting

### Common Issues

**1. Events not being received**
```python
# Check subscriber registration
print(f"Frontend subscribers: {len(trade_manager._frontend_subscribers)}")
print(f"Event queue size: {trade_manager.event_bus._event_queue.qsize()}")
```

**2. TradeManager state not persisting**
```python
# Check file permissions and path
import os
state_file = "crypto_bot/logs/trade_manager_state.json"
print(f"State file exists: {os.path.exists(state_file)}")
print(f"State file writable: {os.access(state_file, os.W_OK)}")
```

**3. Memory usage concerns**
```python
# Monitor instance count
trade_manager = get_single_source_trade_manager()
print(f"Total trades in memory: {len(trade_manager.trades)}")
print(f"Total positions in memory: {len(trade_manager.positions)}")
```

## 🔮 Future Enhancements

### Planned Features
- **Database persistence** for high-volume trading
- **Trade compression** for long-term storage
- **Advanced event filtering** for performance
- **Metrics and monitoring** dashboard
- **Configuration-driven** event routing

### Integration Opportunities
- **WebSocket streaming** for real-time frontend updates
- **External API sync** for third-party integrations
- **Audit logging** to external systems
- **Trade analytics** and reporting

## 📚 API Reference

### SingleSourceTradeManager Methods

| Method | Description |
|--------|-------------|
| `record_trade(trade)` | Record a new trade and notify subscribers |
| `get_all_positions()` | Get all open positions |
| `get_position(symbol)` | Get position for specific symbol |
| `update_price(symbol, price)` | Update price and trigger position updates |
| `get_system_status()` | Get comprehensive system status |
| `add_frontend_subscriber(callback)` | Add frontend event subscriber |
| `add_portfolio_service_subscriber(callback)` | Add portfolio service subscriber |
| `add_notification_subscriber(callback)` | Add notification subscriber |

### Event Types

| Event Type | Description | Data |
|------------|-------------|------|
| `trade_executed` | New trade recorded | `trade`, `trade_id` |
| `position_updated` | Position changed | `position`, `trade` |
| `price_updated` | Price updated | `symbol`, `old_price`, `new_price` |

## 🎯 Conclusion

The Single Source of Truth architecture provides:
- **Reliable data consistency** across all components
- **Real-time event-driven updates** for responsive UIs
- **Maintainable and scalable** system design
- **Future-proof foundation** for advanced features

This architecture eliminates the synchronization issues that plagued the previous multi-source system and provides a solid foundation for enterprise-grade trading applications.
