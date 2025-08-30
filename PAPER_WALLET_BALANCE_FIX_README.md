# Paper Wallet Balance Fix

## Problem
The bot was showing a paper wallet balance of $1,000 instead of the configured $10,000, even though the configuration files were set correctly.

## Root Cause
The issue was in `crypto_bot/main.py` where the paper wallet initialization was asking for user input and falling back to a hardcoded value of $1,000 when running in non-interactive mode:

```python
try:
    start_bal = float(input("Enter paper trading balance in USDT: "))
except Exception:
    start_bal = 1000.0  # Hardcoded fallback!
```

## Solution
Modified the paper wallet initialization logic to read from configuration files in the following priority order:

1. **Environment Variable** - `PAPER_WALLET_BALANCE` (highest priority)
2. **Main Config File** - `crypto_bot/config.yaml` paper_wallet section
3. **Dedicated Config File** - `crypto_bot/paper_wallet_config.yaml`
4. **User Input** - Only if no config files exist
5. **Default Fallback** - $10,000 (instead of $1,000)

## Code Changes
The following changes were made to `crypto_bot/main.py`:

```python
# Check environment variable first
env_balance = os.getenv("PAPER_WALLET_BALANCE")
if env_balance:
    try:
        start_bal = float(env_balance)
        logger.info(f"Loaded paper wallet balance from environment variable: ${start_bal:.2f}")
        config_loaded = True
    except ValueError:
        logger.warning(f"Invalid PAPER_WALLET_BALANCE environment variable: {env_balance}")

# If no environment variable, try main config file first
if not config_loaded and config.get("paper_wallet", {}).get("initial_balance"):
    try:
        start_bal = float(config["paper_wallet"]["initial_balance"])
        logger.info(f"Loaded paper wallet balance from main config: ${start_bal:.2f}")
        config_loaded = True
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid paper wallet balance in main config: {e}")

# If still no config loaded, try dedicated paper wallet config files
if not config_loaded:
    possible_paths = [
        Path("crypto_bot/paper_wallet_config.yaml"),  # Relative to current directory
        Path(__file__).parent / "paper_wallet_config.yaml",  # Relative to main.py
        Path.cwd() / "crypto_bot" / "paper_wallet_config.yaml",  # Relative to working directory
    ]
    
    for paper_wallet_config_path in possible_paths:
        if paper_wallet_config_path.exists():
            try:
                with open(paper_wallet_config_path, 'r') as f:
                    paper_config = yaml.safe_load(f) or {}
                    start_bal = paper_config.get('initial_balance', 10000.0)
                    logger.info(f"Loaded paper wallet balance from config {paper_wallet_config_path}: ${start_bal:.2f}")
                    config_loaded = True
                    break
            except Exception as e:
                logger.warning(f"Failed to read paper wallet config {paper_wallet_config_path}: {e}")
                continue

# Only ask for user input if no config was loaded
if not config_loaded:
    try:
        start_bal = float(input("Enter paper trading balance in USDT: "))
        logger.info(f"User set paper trading balance: ${start_bal:.2f}")
    except Exception:
        logger.info(f"Using default paper trading balance: ${start_bal:.2f}")
```

## Configuration Files
The bot now reads from these configuration sources:

### 1. Environment Variable
```bash
export PAPER_WALLET_BALANCE=15000.0
```

### 2. Main Config File (`crypto_bot/config.yaml`)
```yaml
paper_wallet:
  initial_balance: 10000.0
  allow_short: true
  max_open_trades: 10
```

### 3. Dedicated Config File (`crypto_bot/paper_wallet_config.yaml`)
```yaml
initial_balance: 10000.0
max_open_trades: 10
allow_short: true
```

## Testing
The fix was tested with a script that verified:
- Environment variable loading (priority 1)
- Main config file loading (priority 2)
- Dedicated config file loading (priority 3)
- Proper fallback to default values

## Result
The bot now correctly reads the $10,000 balance from the configuration files instead of defaulting to $1,000, and the log message will show:

```
2025-08-30 09:16:34,035 - INFO - Paper wallet status: Balance=$10000.0, PnL=$0.0, Win Rate=0.0
```

## Usage
To set a custom paper wallet balance, you can:

1. **Set an environment variable:**
   ```bash
   export PAPER_WALLET_BALANCE=20000.0
   ```

2. **Edit the main config file:**
   ```yaml
   # crypto_bot/config.yaml
   paper_wallet:
     initial_balance: 20000.0
   ```

3. **Edit the dedicated config file:**
   ```yaml
   # crypto_bot/paper_wallet_config.yaml
   initial_balance: 20000.0
   ```

The environment variable takes precedence over all configuration files, making it ideal for temporary overrides or automated deployments.
