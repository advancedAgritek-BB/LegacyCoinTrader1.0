#!/usr/bin/env python3
"""
Environment Variable Fix for Evaluation Pipeline
This script fixes the environment variable loading issues that are causing
circuit breakers to open due to missing API credentials.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.logger import setup_logger, LOG_DIR

# Setup logging
logger = setup_logger("env_fix", LOG_DIR / "env_fix.log")

class EnvironmentFixer:
    """Fix environment variable loading issues."""
    
    def __init__(self):
        self.env_path = project_root / ".env"
        self.secrets = {}
        self.issues_found = []
        self.fixes_applied = []
        
    def load_current_env(self):
        """Load current environment variables from .env file."""
        logger.info("📋 Loading current environment variables...")
        
        if self.env_path.exists():
            self.secrets = dotenv_values(str(self.env_path))
            logger.info(f"✅ Loaded {len(self.secrets)} variables from .env")
        else:
            logger.error("❌ .env file not found")
            self.issues_found.append(".env file not found")
            return False
        
        return True
    
    def check_api_keys(self):
        """Check for API key issues."""
        logger.info("🔑 Checking API keys...")
        
        # Check for Kraken API keys
        kraken_key = self.secrets.get('KRAKEN_API_KEY')
        kraken_secret = self.secrets.get('KRAKEN_API_SECRET')
        
        if not kraken_key or kraken_key == 'your_kraken_api_key_here':
            self.issues_found.append("Missing or template Kraken API key")
            logger.error("❌ Missing or template Kraken API key")
        else:
            logger.info("✅ Kraken API key found")
        
        if not kraken_secret or kraken_secret == 'your_kraken_api_secret_here':
            self.issues_found.append("Missing or template Kraken API secret")
            logger.error("❌ Missing or template Kraken API secret")
        else:
            logger.info("✅ Kraken API secret found")
        
        # Check for generic API keys (used by the bot)
        api_key = self.secrets.get('API_KEY')
        api_secret = self.secrets.get('API_SECRET')
        
        if not api_key:
            self.issues_found.append("Missing generic API_KEY")
            logger.warning("⚠️ Missing API_KEY - will use KRAKEN_API_KEY")
        else:
            logger.info("✅ Generic API_KEY found")
        
        if not api_secret:
            self.issues_found.append("Missing generic API_SECRET")
            logger.warning("⚠️ Missing API_SECRET - will use KRAKEN_API_SECRET")
        else:
            logger.info("✅ Generic API_SECRET found")
    
    def fix_variable_mapping(self):
        """Fix the mapping between Kraken-specific and generic API variables."""
        logger.info("🔄 Fixing variable mapping...")
        
        # Create mapping from Kraken-specific to generic variables
        mappings = {
            'KRAKEN_API_KEY': 'API_KEY',
            'KRAKEN_API_SECRET': 'API_SECRET',
        }
        
        for kraken_var, generic_var in mappings.items():
            kraken_value = self.secrets.get(kraken_var)
            generic_value = self.secrets.get(generic_var)
            
            if kraken_value and not generic_value:
                # Add generic variable if it doesn't exist
                self.secrets[generic_var] = kraken_value
                logger.info(f"✅ Added {generic_var} = {kraken_var}")
                self.fixes_applied.append(f"Added {generic_var} mapping")
            elif kraken_value and generic_value and kraken_value != generic_value:
                # Update generic variable to match Kraken variable
                self.secrets[generic_var] = kraken_value
                logger.info(f"✅ Updated {generic_var} to match {kraken_var}")
                self.fixes_applied.append(f"Updated {generic_var} mapping")
            elif not kraken_value and generic_value:
                # Add Kraken variable if it doesn't exist
                self.secrets[kraken_var] = generic_value
                logger.info(f"✅ Added {kraken_var} = {generic_var}")
                self.fixes_applied.append(f"Added {kraken_var} mapping")
    
    def fix_exchange_configuration(self):
        """Fix exchange configuration."""
        logger.info("🏪 Fixing exchange configuration...")
        
        # Ensure EXCHANGE is set to kraken
        if 'EXCHANGE' not in self.secrets:
            self.secrets['EXCHANGE'] = 'kraken'
            logger.info("✅ Added EXCHANGE=kraken")
            self.fixes_applied.append("Added EXCHANGE configuration")
        elif self.secrets['EXCHANGE'] != 'kraken':
            self.secrets['EXCHANGE'] = 'kraken'
            logger.info("✅ Updated EXCHANGE to kraken")
            self.fixes_applied.append("Updated EXCHANGE configuration")
        else:
            logger.info("✅ EXCHANGE already set to kraken")
    
    def fix_template_values(self):
        """Replace any remaining template values."""
        logger.info("📝 Fixing template values...")
        
        template_replacements = {
            'your_kraken_api_key_here': '',
            'your_kraken_api_secret_here': '',
            'your_telegram_token_here': '',
            'your_chat_id_here': '',
            'your_helius_key_here': '',
            'your_wallet_address_here': '',
        }
        
        for template, replacement in template_replacements.items():
            for key, value in self.secrets.items():
                if value == template:
                    self.secrets[key] = replacement
                    logger.warning(f"⚠️ Replaced template value for {key}")
                    self.fixes_applied.append(f"Replaced template value for {key}")
    
    def validate_api_keys(self):
        """Validate that API keys look correct."""
        logger.info("🔍 Validating API keys...")
        
        api_key = self.secrets.get('API_KEY') or self.secrets.get('KRAKEN_API_KEY')
        api_secret = self.secrets.get('API_SECRET') or self.secrets.get('KRAKEN_API_SECRET')
        
        if not api_key:
            self.issues_found.append("No API key found")
            logger.error("❌ No API key found")
            return False
        
        if not api_secret:
            self.issues_found.append("No API secret found")
            logger.error("❌ No API secret found")
            return False
        
        # Basic validation - Kraken API keys have specific formats
        if len(api_key) < 10:
            self.issues_found.append("API key seems too short")
            logger.warning("⚠️ API key seems too short")
        
        if len(api_secret) < 10:
            self.issues_found.append("API secret seems too short")
            logger.warning("⚠️ API secret seems too short")
        
        logger.info("✅ API keys validated")
        return True
    
    def save_updated_env(self):
        """Save the updated environment variables."""
        logger.info("💾 Saving updated .env file...")
        
        try:
            # Create backup
            backup_path = self.env_path.with_suffix('.env.backup.before_fix')
            if self.env_path.exists():
                import shutil
                shutil.copy2(self.env_path, backup_path)
                logger.info(f"📋 Created backup: {backup_path}")
            
            # Write updated .env file
            with open(self.env_path, 'w') as f:
                f.write("# LegacyCoinTrader Environment Configuration\n")
                f.write("# Updated by environment fixer\n\n")
                
                for key, value in self.secrets.items():
                    f.write(f"{key}={value}\n")
            
            logger.info("✅ Updated .env file saved")
            self.fixes_applied.append("Saved updated .env file")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save .env file: {e}")
            self.issues_found.append(f"Failed to save .env file: {e}")
            return False
    
    def test_environment_loading(self):
        """Test that environment variables can be loaded properly."""
        logger.info("🧪 Testing environment loading...")
        
        try:
            # Load the updated .env file
            load_dotenv(self.env_path)
            
            # Check if variables are available
            api_key = os.getenv('API_KEY') or os.getenv('KRAKEN_API_KEY')
            api_secret = os.getenv('API_SECRET') or os.getenv('KRAKEN_API_SECRET')
            exchange = os.getenv('EXCHANGE')
            
            if api_key and api_secret and exchange == 'kraken':
                logger.info("✅ Environment loading test passed")
                return True
            else:
                logger.error("❌ Environment loading test failed")
                self.issues_found.append("Environment loading test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Environment loading test error: {e}")
            self.issues_found.append(f"Environment loading test error: {e}")
            return False
    
    def generate_report(self):
        """Generate a comprehensive fix report."""
        print("\n" + "="*60)
        print("🔧 ENVIRONMENT VARIABLE FIX REPORT")
        print("="*60)
        
        print(f"\n📊 Issues Found: {len(self.issues_found)}")
        for i, issue in enumerate(self.issues_found, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n🔧 Fixes Applied: {len(self.fixes_applied)}")
        for i, fix in enumerate(self.fixes_applied, 1):
            print(f"   {i}. {fix}")
        
        print("\n" + "="*60)
        print("📋 CURRENT STATUS")
        print("="*60)
        
        # Show current API key status
        api_key = self.secrets.get('API_KEY') or self.secrets.get('KRAKEN_API_KEY')
        api_secret = self.secrets.get('API_SECRET') or self.secrets.get('KRAKEN_API_SECRET')
        exchange = self.secrets.get('EXCHANGE')
        
        print(f"API_KEY: {'✅ SET' if api_key else '❌ NOT SET'}")
        print(f"API_SECRET: {'✅ SET' if api_secret else '❌ NOT SET'}")
        print(f"EXCHANGE: {exchange or '❌ NOT SET'}")
        
        if self.issues_found:
            print("\n❌ CRITICAL ISSUES TO ADDRESS:")
            print("   1. Add your actual Kraken API keys to .env file")
            print("   2. Ensure API keys have proper permissions")
            print("   3. Test API connectivity manually")
        else:
            print("\n✅ All environment issues resolved!")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Verify your API keys are correct")
        print("   2. Restart the trading bot")
        print("   3. Monitor logs for API connectivity")
        
        print("\n" + "="*60)
    
    def apply_all_fixes(self):
        """Apply all environment fixes."""
        logger.info("🔧 Applying environment fixes...")
        
        try:
            # Load current environment
            if not self.load_current_env():
                return False
            
            # Apply fixes
            self.check_api_keys()
            self.fix_variable_mapping()
            self.fix_exchange_configuration()
            self.fix_template_values()
            
            # Validate and save
            validation_ok = self.validate_api_keys()
            save_ok = self.save_updated_env()
            
            # Test loading
            test_ok = self.test_environment_loading()
            
            # Generate report
            self.generate_report()
            
            if validation_ok and save_ok and test_ok:
                logger.info("🎉 Environment fixes applied successfully!")
                return True
            else:
                logger.warning("⚠️ Some environment issues may require manual attention")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error applying environment fixes: {e}")
            return False

def main():
    """Main function to run the environment fixer."""
    fixer = EnvironmentFixer()
    success = fixer.apply_all_fixes()
    
    if success:
        print("✅ Environment fixes applied successfully!")
        print("🔄 Please restart your trading bot to apply the changes.")
    else:
        print("❌ Some issues require manual attention.")
        print("📋 Review the report above for specific actions needed.")

if __name__ == "__main__":
    main()
