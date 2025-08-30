# Frontend Configuration Pages

This document describes the new configuration pages added to the LegacyCoinTrader frontend.

## Overview

Two new pages have been added to the frontend to allow users to configure their API credentials and trading bot settings through a web interface:

1. **API Configuration Page** (`/api_config`) - For setting exchange API keys and credentials
2. **Config Settings Page** (`/config_settings`) - For configuring trading parameters and strategy settings

## API Configuration Page

### Access
Navigate to **API Config** in the sidebar navigation or visit `/api_config`

### Features
- **Exchange Selection**: Choose between Kraken, Coinbase, and Binance
- **Trading Mode**: Select between CEX (Centralized Exchange) and DEX (Decentralized Exchange)
- **API Credentials**: 
  - Kraken API Key and Secret
  - Coinbase API Key, Secret, and Passphrase
- **Telegram Configuration**: Bot token and chat ID
- **Solana Configuration**: Wallet address for DEX trading

### Security
- API keys and secrets are stored as password fields
- Values are saved to `crypto_bot/user_config.yaml`
- Form validation and connection testing

### API Endpoints
- `GET /api_config` - Display the configuration page
- `POST /api/save_api_config` - Save API configuration

## Config Settings Page

### Access
Navigate to **Settings** in the sidebar navigation or visit `/config_settings`

### Features
- **Trading Settings**: Timeframe, stop loss, take profit, symbol refresh interval
- **Risk Management**: Max positions, position size, drawdown limits, daily loss limits
- **Strategy Allocation**: Percentage allocation for each trading strategy
- **Symbol Filter**: Volume thresholds, spread limits, correlation settings
- **Enhanced Backtesting**: Enable/disable, lookback periods, performance thresholds
- **Telegram Settings**: Enable/disable various notification types

### Validation
- Strategy allocation must total 100%
- Stop loss must be less than take profit
- Real-time validation feedback

### API Endpoints
- `GET /config_settings` - Display the settings page
- `POST /api/save_config_settings` - Save configuration settings
- `POST /api/refresh_config` - Refresh configuration from files

## Navigation

Both pages are accessible from the main sidebar navigation:
- **API Config** - Key icon (`fas fa-key`)
- **Settings** - Cogs icon (`fas fa-cogs`)

## File Structure

### Templates
- `frontend/templates/api_config.html` - API configuration page template
- `frontend/templates/config_settings.html` - Config settings page template

### Backend Routes
- `frontend/app.py` - Contains all the new routes and API endpoints

### Configuration Files
- `crypto_bot/user_config.yaml` - User API credentials and settings
- `crypto_bot/config.yaml` - Main bot configuration

## Usage

### Setting Up API Credentials
1. Navigate to **API Config** page
2. Select your primary exchange
3. Enter your API credentials
4. Configure Telegram bot if desired
5. Click **Save Configuration**

### Configuring Trading Settings
1. Navigate to **Settings** page
2. Adjust trading parameters as needed
3. Set strategy allocation percentages (must total 100%)
4. Configure risk management settings
5. Click **Save Configuration**

### Testing Configuration
- Use the **Test Connection** button on the API Config page
- Use the **Validate** button on the Settings page
- Use the **Refresh Config** button to reload settings

## Security Considerations

- API credentials are stored in `crypto_bot/user_config.yaml`
- Passwords are masked in the UI
- Configuration files should be kept secure and not committed to version control
- Consider using environment variables for sensitive data in production

## Troubleshooting

### Common Issues
1. **Configuration not saving**: Check file permissions on config files
2. **Validation errors**: Ensure strategy allocation totals 100%
3. **Connection test failures**: Verify API credentials are correct

### Testing
Run the test script to verify all endpoints are working:
```bash
python3 test_new_routes.py
```

## Future Enhancements

- Add support for more exchanges
- Implement configuration import/export
- Add configuration versioning
- Add audit logging for configuration changes
- Implement configuration templates
- Add real-time configuration validation
