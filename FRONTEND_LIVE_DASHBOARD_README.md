# Frontend Live Dashboard & Wallet Balance Configuration

## Overview

The LegacyCoinTrader frontend now includes a **live dashboard** with real-time updates and **wallet balance configuration** for dry run mode. This provides users with an interactive, up-to-date view of their trading bot's performance and allows them to configure paper trading parameters.

## 🚀 New Features

### 1. Live Dashboard Updates

The dashboard now updates automatically every few seconds to provide real-time information:

- **Performance Metrics**: P&L, total trades, win rate updates every 5 seconds
- **Bot Status**: Running status, uptime, execution mode updates every 3 seconds
- **Live Signals**: Asset scores and market data updates every 10 seconds
- **Real-time Charts**: Portfolio performance and trading volume charts update automatically

### 2. Wallet Balance Configuration

When selecting **Dry Run (Paper Trading)** mode, users can now configure their paper wallet:

- **Initial Balance**: Set starting capital for paper trading
- **Real-time Updates**: Balance updates automatically as trades are executed
- **P&L Tracking**: Monitor paper trading performance in real-time
- **Configuration Persistence**: Settings are saved and restored between sessions

## 🛠️ Technical Implementation

### API Endpoints

#### `/api/paper-wallet-balance`
- **GET**: Retrieve current paper wallet balance
- **POST**: Update paper wallet balance
- **Response**: JSON with success status and balance amount

#### `/api/live-updates`
- **GET**: Retrieve comprehensive live dashboard data
- **Response**: JSON with bot status, performance metrics, and asset scores
- **Update Frequency**: Every 5 seconds

#### `/api/dashboard-metrics`
- **GET**: Retrieve detailed dashboard metrics
- **Response**: JSON with performance data, allocation, and recent trades

### Configuration Files

#### `crypto_bot/paper_wallet_config.yaml`
```yaml
# Paper Wallet Configuration for Dry Run Mode
initial_balance: 10000.0
max_open_trades: 10
allow_short: true
```

#### `crypto_bot/config.yaml` (Updated)
```yaml
# Paper Wallet Configuration for Dry Run Mode
paper_wallet:
  enabled: true
  initial_balance: 10000.0
  max_open_trades: 10
  allow_short: true
  config_file: "crypto_bot/paper_wallet_config.yaml"
```

## 📱 User Interface

### Bot Control Modal

The bot control modal now includes:

1. **Execution Mode Selection**: Choose between Dry Run, Live Trading, and Backtest
2. **Wallet Balance Configuration**: Appears only when Dry Run is selected
3. **Real-time Validation**: Ensures proper configuration before starting

### Dashboard Cards

#### Paper Wallet Status Card (Dry Run Only)
- **Initial Balance**: Starting capital amount
- **Current Balance**: Real-time balance including P&L
- **P&L**: Current profit/loss from paper trading
- **Update Button**: Quick access to modify balance

#### Live Metrics
- **Total P&L**: Updates every 5 seconds
- **Total Trades**: Real-time trade count
- **Win Rate**: Live win percentage
- **Bot Uptime**: Current running time

## 🔄 Real-time Update System

### Update Intervals

| Component | Update Frequency | Description |
|-----------|------------------|-------------|
| Bot Status | 3 seconds | Running state, mode, uptime |
| Performance | 5 seconds | P&L, trades, win rate |
| Live Signals | 10 seconds | Asset scores, market data |
| Charts | 5 seconds | Portfolio performance, volume |

### Data Flow

```
Frontend JavaScript → API Endpoints → Backend Data Sources → Real-time Updates
```

### Error Handling

- **Connection Failures**: Graceful fallback to last known data
- **API Errors**: Console logging with user-friendly error messages
- **Data Validation**: Ensures data integrity before updating UI

## 🧪 Testing

### Test Script

Run the test suite to verify functionality:

```bash
python test_frontend_live.py
```

### Test Coverage

1. **API Endpoints**: Verify all new endpoints work correctly
2. **Configuration Files**: Check file creation and updates
3. **Real-time Updates**: Validate timestamp updates and data flow
4. **Wallet Balance**: Test balance setting and retrieval
5. **Error Handling**: Verify graceful error handling

## 🚀 Getting Started

### 1. Start the Frontend

```bash
cd frontend
python app.py
```

### 2. Access the Dashboard

Navigate to `http://localhost:8000` in your browser.

### 3. Configure Dry Run Mode

1. Click **Start Bot**
2. Select **Dry Run (Paper Trading)**
3. Enter your desired **Paper Wallet Balance**
4. Click **Confirm**

### 4. Monitor Live Updates

The dashboard will automatically update with real-time data every few seconds.

## 🔧 Configuration Options

### Paper Wallet Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `initial_balance` | 10000.0 | Starting capital in USD |
| `max_open_trades` | 10 | Maximum concurrent positions |
| `allow_short` | true | Enable short selling |

### Update Frequencies

| Setting | Value | Description |
|---------|-------|-------------|
| Bot Status | 3s | Bot running state updates |
| Performance | 5s | Trading metrics updates |
| Live Signals | 10s | Market data updates |

## 🐛 Troubleshooting

### Common Issues

1. **Dashboard Not Updating**
   - Check browser console for JavaScript errors
   - Verify API endpoints are accessible
   - Check network connectivity

2. **Wallet Balance Not Saving**
   - Ensure proper file permissions
   - Check YAML syntax in config files
   - Verify API response format

3. **Real-time Updates Slow**
   - Check server performance
   - Reduce update frequency if needed
   - Monitor API response times

### Debug Mode

Enable debug logging in the frontend:

```python
app.run(host='0.0.0.0', port=8000, debug=True)
```

## 🔮 Future Enhancements

### Planned Features

- **WebSocket Support**: Real-time bidirectional communication
- **Custom Update Intervals**: User-configurable update frequencies
- **Advanced Notifications**: Push notifications for important events
- **Mobile Optimization**: Responsive design for mobile devices
- **Data Export**: Export dashboard data to CSV/JSON

### Performance Improvements

- **Data Caching**: Implement client-side caching for better performance
- **Lazy Loading**: Load data only when needed
- **Compression**: Compress API responses for faster updates

## 📚 Additional Resources

- [Frontend App Structure](frontend/app.py)
- [API Endpoints](frontend/api.py)
- [Paper Wallet Implementation](crypto_bot/paper_wallet.py)
- [Configuration Management](crypto_bot/config.yaml)

## 🤝 Contributing

To contribute to the live dashboard:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Test thoroughly before submitting

## 📄 License

This project is part of LegacyCoinTrader and follows the same licensing terms.
