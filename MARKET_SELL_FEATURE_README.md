# Market Sell Feature for Frontend Dashboard

## Overview

This feature adds a "Market Sell" button to each open position card on the frontend dashboard, allowing users to immediately sell positions with a single click.

## Features

- **Market Sell Button**: Each open position displays a red "Market Sell" button
- **Confirmation Dialog**: Users must confirm the sell action before execution
- **Real-time Feedback**: Button shows loading state during execution
- **Success/Error Notifications**: Toast notifications inform users of the result
- **Automatic Refresh**: Positions list refreshes after successful sells
- **Paper Trading Support**: Works in both live and paper trading modes

## Implementation Details

### Frontend Changes

1. **Template Updates** (`frontend/templates/index.html`):
   - Added market sell button to each position card
   - Added JavaScript function `marketSellPosition()` to handle button clicks
   - Updated dynamic position updates to include sell buttons
   - Added CSS styling for position actions

2. **API Endpoint** (`frontend/app.py`):
   - New `/api/sell-position` endpoint accepts POST requests
   - Validates symbol and amount parameters
   - Writes sell requests to `sell_requests.json` file

### Backend Changes

1. **Main Bot Updates** (`crypto_bot/main.py`):
   - Added processing of sell requests from `sell_requests.json`
   - Handles both full and partial position sells
   - Updates paper wallet in dry run mode
   - Sends Telegram notifications for executed sells
   - Cleans up old requests automatically

## Usage

1. **Start the Frontend**: Run `python3 -m frontend.app`
2. **Navigate to Dashboard**: Open the web interface
3. **View Open Positions**: Look for the "Open Positions" card
4. **Click Market Sell**: Click the red "Market Sell" button on any position
5. **Confirm Action**: Click "OK" in the confirmation dialog
6. **Monitor Execution**: Watch for success/error notifications

## File Structure

```
frontend/
├── app.py                    # API endpoint for sell requests
└── templates/
    └── index.html            # Updated with market sell buttons

crypto_bot/
├── main.py                   # Processes sell requests
└── logs/
    └── sell_requests.json    # Sell request queue (auto-generated)
```

## API Reference

### POST /api/sell-position

**Request Body:**
```json
{
  "symbol": "BTC/USD",
  "amount": 0.001
}
```

**Response:**
```json
{
  "success": true,
  "message": "Market sell order submitted for 0.001 BTC/USD",
  "symbol": "BTC/USD",
  "amount": 0.001
}
```

## Error Handling

- **Missing Parameters**: Returns error if symbol or amount is missing
- **Invalid Position**: Gracefully handles requests for non-existent positions
- **Network Errors**: Shows appropriate error messages
- **File I/O Errors**: Logs errors but doesn't crash the bot

## Security Considerations

- **Confirmation Required**: Users must confirm before selling
- **Position Validation**: Only sells existing positions
- **Amount Validation**: Cannot sell more than position size
- **Request Timeout**: Old requests are automatically cleaned up

## Testing

Run the test script to verify functionality:

```bash
python3 test_market_sell.py
```

## Notes

- Works in both live trading and paper trading modes
- Integrates with existing Telegram notification system
- Maintains compatibility with all existing trading strategies
- No changes required to existing configuration files
