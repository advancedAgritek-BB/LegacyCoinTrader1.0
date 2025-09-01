#!/usr/bin/env python3
"""
Script to enable pump.fun API for wallet evaluation and rug pull filtering.
"""

import os
import re

def enable_pump_fun_api():
    """Enable the pump.fun API key in .env file."""
    env_file = ".env"

    if not os.path.exists(env_file):
        print("❌ .env file not found!")
        return False

    # Read current .env file
    with open(env_file, 'r') as f:
        content = f.read()

    # Check if pump.fun API key is already enabled
    if "PUMPFUN_API_KEY=uAAVZaLTQxzm1pIAL6P5XMJapapVHU91JX94xJlqaRw4CTeOyrBwPuB9" in content:
        print("✅ pump.fun API key is already enabled!")
        return True

    # Uncomment the pump.fun API key
    old_pattern = r"# PUMPFUN_API_KEY=uAAVZaLTQxzm1pIAL6P5XMJapapVHU91JX94xJlqaRw4CTeOyrBwPuB9"
    new_replacement = "PUMPFUN_API_KEY=uAAVZaLTQxzm1pIAL6P5XMJapapVHU91JX94xJlqaRw4CTeOyrBwPuB9"

    if old_pattern in content:
        content = content.replace(old_pattern, new_replacement)

        # Write back to file
        with open(env_file, 'w') as f:
            f.write(content)

        print("✅ Successfully enabled pump.fun API key!")
        print("🔄 Please restart your bot for the changes to take effect.")
        return True
    else:
        print("❌ Could not find pump.fun API key line in .env file")
        return False

if __name__ == "__main__":
    print("🚀 Enabling pump.fun API for wallet evaluation...")
    success = enable_pump_fun_api()

    if success:
        print("\n📋 Next steps:")
        print("1. Restart your bot")
        print("2. The scanner will now use pump.fun API with wallet evaluation")
        print("3. Only tokens from credible wallets (score >= 60) will be traded")
        print("4. Check logs for 'pump.fun API returned X credible tokens' messages")
    else:
        print("\n❌ Failed to enable pump.fun API. Please check your .env file.")
