# .env File Template for LegacyCoinTrader
# Copy this content to crypto_bot/.env

# Complete environment configuration for the crypto bot
# Environment variables override values saved in user_config.yaml and config.yaml
# Uncomment and fill in the real values as needed

# Exchange Configuration
EXCHANGE=kraken
KRAKEN_API_KEY=pTDIfRbVaP3Y7y8Ph8CHiAC5O/wlBkCX0so9N91iobqhnysguyX7o18/
KRAKEN_API_SECRET=9mmck1Rpzs8skQhtgGg+6NpzkRP1HiXf+q0HfMQ9d447av7LkaFhBh/gCNdWaOaYZFUA0AWS30FDP1NVT+BeTQ==
KRAKEN_WS_TOKEN=your_kraken_ws_token_here

# Alternative Exchange (Coinbase)
# COINBASE_API_KEY=your_coinbase_api_key
# COINBASE_API_SECRET=your_coinbase_secret
# COINBASE_PASSPHRASE=your_coinbase_passphrase

# Telegram Configuration
TELEGRAM_TOKEN=8126215032:AAEhQZLiXpssauKf0ktQsq1XqXl94QriCdE
TELEGRAM_CHAT_ID=827777274
TELE_CHAT_ADMINS=827777274

# Solana Configuration
HELIUS_KEY=43f311d0-a726-44fc-a5dc-36420cd07bb4
WALLET_ADDRESS=EoiVpzLA6b6JBKXTB5WRFor3mPkseM6UisLHt8qK9g1c

# LunarCrush Sentiment Analysis (Optional)
# LUNARCRUSH_API_KEY=your_lunarcrush_api_key

# Google Cloud (Optional - for advanced logging)
# GOOGLE_CRED_JSON=path_to_google_credentials.json

# Trading Mode
MODE=cex

# Additional Configuration
# TESTING_MODE=true
# EXECUTION_MODE=dry_run

## Instructions:
# 1. Copy this entire content
# 2. Replace the content of crypto_bot/.env with this
# 3. Uncomment and fill in any missing values (like KRAKEN_WS_TOKEN)
# 4. Save the file
# 5. Restart your application

## What This Fixes:
# - Exchange API authentication
# - Telegram bot functionality  
# - Solana integration
# - Proper configuration loading
# - Missing environment variable prompts
