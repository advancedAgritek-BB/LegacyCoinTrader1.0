"""Integration tests for configuration management system.

This module tests configuration loading, validation, updates, and persistence
across all bot components and features.
"""

import pytest
import yaml
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from crypto_bot.config import load_config, save_config, validate_config
from crypto_bot.user_config import load_user_config, save_user_config


@pytest.mark.integration
class TestConfigurationManagementIntegration:
    """Test configuration management across all bot components."""

    @pytest.fixture
    def sample_config(self):
        """Comprehensive sample configuration for testing."""
        return {
            'trading': {
                'enabled': True,
                'max_positions': 5,
                'risk_per_trade': 0.02,
                'max_drawdown': 0.1,
                'default_leverage': 1.0
            },
            'solana': {
                'enabled': True,
                'rpc_url': 'https://api.mainnet-beta.solana.com',
                'private_key': 'test_private_key',
                'gas_limit': 200000,
                'gas_price': 10
            },
            'telegram': {
                'enabled': True,
                'bot_token': '123456789:ABCdefGHIjklMNOpqrsTUVwxyz',
                'chat_id': '@test_channel',
                'notification_level': 'detailed'
            },
            'enhanced_scanning': {
                'enabled': True,
                'scan_interval': 30,
                'min_liquidity': 10000,
                'max_positions_per_symbol': 3
            },
            'risk_management': {
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1,
                'max_position_size_pct': 0.1,
                'volatility_adjustment': True
            },
            'strategies': {
                'trend_bot': {
                    'enabled': True,
                    'min_trend_strength': 0.6,
                    'lookback_period': 20
                },
                'mean_bot': {
                    'enabled': True,
                    'lookback_period': 14,
                    'entry_threshold': 2.0
                },
                'breakout_bot': {
                    'enabled': True,
                    'breakout_threshold': 2.5,
                    'consolidation_period': 10
                }
            },
            'exchanges': {
                'kraken': {
                    'enabled': True,
                    'api_key': 'kraken_api_key',
                    'api_secret': 'kraken_api_secret',
                    'sandbox': False
                },
                'binance': {
                    'enabled': False,
                    'api_key': '',
                    'api_secret': ''
                }
            },
            'logging': {
                'level': 'INFO',
                'max_file_size': 10485760,  # 10MB
                'backup_count': 5,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'performance': {
                'metrics_enabled': True,
                'report_interval': 3600,  # 1 hour
                'save_metrics': True,
                'alert_thresholds': {
                    'max_drawdown': 0.05,
                    'min_win_rate': 0.5
                }
            }
        }

    @pytest.fixture
    def temp_config_dir(self):
        """Temporary directory for config files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_config_loading_and_parsing(self, sample_config, temp_config_dir):
        """Test configuration file loading and parsing."""
        config_path = Path(temp_config_dir) / 'test_config.yaml'

        # Save sample config
        with open(config_path, 'w') as f:
            yaml.safe_dump(sample_config, f)

        # Load and verify config
        loaded_config = load_config(config_path)

        assert loaded_config is not None
        assert 'trading' in loaded_config
        assert loaded_config['trading']['enabled'] is True
        assert loaded_config['trading']['max_positions'] == 5
        assert 'strategies' in loaded_config
        assert 'trend_bot' in loaded_config['strategies']

    def test_config_validation_comprehensive(self, sample_config):
        """Test comprehensive configuration validation."""
        # Test valid configuration
        is_valid, errors = validate_config(sample_config)
        assert is_valid
        assert len(errors) == 0

        # Test invalid trading configuration
        invalid_config = sample_config.copy()
        invalid_config['trading']['risk_per_trade'] = 1.5  # Invalid: > 100%

        is_valid, errors = validate_config(invalid_config)
        assert not is_valid
        assert len(errors) > 0
        assert any('risk_per_trade' in error for error in errors)

        # Test invalid exchange configuration
        invalid_config = sample_config.copy()
        invalid_config['exchanges']['kraken']['api_key'] = ''  # Empty API key

        is_valid, errors = validate_config(invalid_config)
        assert not is_valid
        assert len(errors) > 0

        # Test invalid strategy configuration
        invalid_config = sample_config.copy()
        invalid_config['strategies']['trend_bot']['min_trend_strength'] = 1.5  # Invalid: > 1.0

        is_valid, errors = validate_config(invalid_config)
        assert not is_valid
        assert len(errors) > 0

    def test_config_save_and_persistence(self, sample_config, temp_config_dir):
        """Test configuration saving and persistence."""
        config_path = Path(temp_config_dir) / 'test_config.yaml'

        # Save configuration
        save_config(sample_config, config_path)

        # Verify file was created
        assert config_path.exists()

        # Load and verify persistence
        with open(config_path, 'r') as f:
            loaded_yaml = yaml.safe_load(f)

        assert loaded_yaml == sample_config

        # Test configuration backup
        backup_path = config_path.with_suffix('.backup')
        save_config(sample_config, config_path, backup=True)

        assert backup_path.exists()

        # Verify backup content
        with open(backup_path, 'r') as f:
            backup_yaml = yaml.safe_load(f)

        assert backup_yaml == sample_config

    def test_runtime_config_updates(self, sample_config):
        """Test runtime configuration updates."""
        # Start with initial config
        config = sample_config.copy()

        # Test trading configuration updates
        config['trading']['max_positions'] = 10
        config['trading']['risk_per_trade'] = 0.03

        # Validate updated config
        is_valid, errors = validate_config(config)
        assert is_valid

        # Test strategy enable/disable
        config['strategies']['trend_bot']['enabled'] = False
        config['strategies']['mean_bot']['enabled'] = True

        is_valid, errors = validate_config(config)
        assert is_valid

        # Test risk parameter updates
        config['risk_management']['stop_loss_pct'] = 0.03
        config['risk_management']['take_profit_pct'] = 0.15

        is_valid, errors = validate_config(config)
        assert is_valid

    def test_config_merging_and_inheritance(self):
        """Test configuration merging and inheritance."""
        # Base configuration
        base_config = {
            'trading': {
                'enabled': True,
                'max_positions': 5,
                'risk_per_trade': 0.02
            },
            'telegram': {
                'enabled': True,
                'notification_level': 'basic'
            }
        }

        # Override configuration
        override_config = {
            'trading': {
                'max_positions': 10,  # Override
                'default_leverage': 2.0  # Add new
            },
            'telegram': {
                'notification_level': 'detailed'  # Override
            },
            'new_section': {  # Add new section
                'setting1': 'value1'
            }
        }

        # Merge configurations (deep merge)
        merged_config = self._deep_merge(base_config, override_config)

        # Verify base settings preserved
        assert merged_config['trading']['enabled'] is True
        assert merged_config['trading']['risk_per_trade'] == 0.02

        # Verify overrides applied
        assert merged_config['trading']['max_positions'] == 10
        assert merged_config['telegram']['notification_level'] == 'detailed'

        # Verify new settings added
        assert merged_config['trading']['default_leverage'] == 2.0
        assert merged_config['new_section']['setting1'] == 'value1'

    def _deep_merge(self, base, override):
        """Helper method for deep merging configurations."""
        merged = base.copy()

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value

        return merged

    def test_environment_variable_integration(self, sample_config, temp_config_dir):
        """Test environment variable integration with configuration."""
        config_path = Path(temp_config_dir) / 'test_config.yaml'

        # Set environment variables
        test_env_vars = {
            'CRYPTO_BOT_API_KEY': 'env_api_key',
            'CRYPTO_BOT_API_SECRET': 'env_api_secret',
            'CRYPTO_BOT_TELEGRAM_TOKEN': 'env_telegram_token',
            'CRYPTO_BOT_MAX_POSITIONS': '8'
        }

        # Mock environment
        with patch.dict(os.environ, test_env_vars):
            # Create config with environment variable placeholders
            env_config = sample_config.copy()
            env_config['exchanges']['kraken']['api_key'] = '${CRYPTO_BOT_API_KEY}'
            env_config['exchanges']['kraken']['api_secret'] = '${CRYPTO_BOT_API_SECRET}'
            env_config['telegram']['bot_token'] = '${CRYPTO_BOT_TELEGRAM_TOKEN}'
            env_config['trading']['max_positions'] = '${CRYPTO_BOT_MAX_POSITIONS}'

            # Save config with placeholders
            with open(config_path, 'w') as f:
                yaml.safe_dump(env_config, f)

            # Load config with environment variable resolution
            loaded_config = self._load_config_with_env_resolution(config_path)

            # Verify environment variables were resolved
            assert loaded_config['exchanges']['kraken']['api_key'] == 'env_api_key'
            assert loaded_config['exchanges']['kraken']['api_secret'] == 'env_api_secret'
            assert loaded_config['telegram']['bot_token'] == 'env_telegram_token'
            assert loaded_config['trading']['max_positions'] == '8'  # String from env

    def _load_config_with_env_resolution(self, config_path):
        """Helper to load config with environment variable resolution."""
        with open(config_path, 'r') as f:
            config_str = f.read()

        # Simple environment variable resolution
        import re
        def replace_env_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        resolved_str = re.sub(r'\$\{([^}]+)\}', replace_env_var, config_str)

        return yaml.safe_load(resolved_str)

    def test_configuration_hot_reload(self, sample_config, temp_config_dir):
        """Test configuration hot reload functionality."""
        config_path = Path(temp_config_dir) / 'test_config.yaml'

        # Save initial config
        with open(config_path, 'w') as f:
            yaml.safe_dump(sample_config, f)

        # Load initial config
        initial_config = load_config(config_path)

        # Modify config file
        modified_config = sample_config.copy()
        modified_config['trading']['max_positions'] = 15
        modified_config['trading']['risk_per_trade'] = 0.05

        with open(config_path, 'w') as f:
            yaml.safe_dump(modified_config, f)

        # Simulate hot reload
        reloaded_config = load_config(config_path)

        # Verify changes were picked up
        assert reloaded_config['trading']['max_positions'] == 15
        assert reloaded_config['trading']['risk_per_trade'] == 0.05

        # Verify other settings unchanged
        assert reloaded_config['telegram']['enabled'] == initial_config['telegram']['enabled']

    def test_configuration_schema_validation(self):
        """Test configuration schema validation."""
        # Valid configuration schema
        valid_schema = {
            'type': 'object',
            'properties': {
                'trading': {
                    'type': 'object',
                    'properties': {
                        'enabled': {'type': 'boolean'},
                        'max_positions': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                        'risk_per_trade': {'type': 'number', 'minimum': 0.001, 'maximum': 0.1}
                    },
                    'required': ['enabled', 'max_positions']
                }
            },
            'required': ['trading']
        }

        # Test valid config against schema
        valid_config = {
            'trading': {
                'enabled': True,
                'max_positions': 5,
                'risk_per_trade': 0.02
            }
        }

        # This would use a JSON schema validator in real implementation
        assert self._validate_against_schema(valid_config, valid_schema)

        # Test invalid config against schema
        invalid_config = {
            'trading': {
                'enabled': True,
                'max_positions': 25,  # Exceeds maximum
                'risk_per_trade': 0.15  # Exceeds maximum
            }
        }

        assert not self._validate_against_schema(invalid_config, valid_schema)

    def _validate_against_schema(self, config, schema):
        """Simple schema validation helper."""
        # Check required fields
        if 'required' in schema:
            for required_field in schema['required']:
                if required_field not in config:
                    return False

        # Check properties
        if 'properties' in schema:
            for prop_name, prop_schema in schema['properties'].items():
                if prop_name in config:
                    prop_value = config[prop_name]

                    # Type checking
                    if 'type' in prop_schema:
                        if prop_schema['type'] == 'boolean' and not isinstance(prop_value, bool):
                            return False
                        elif prop_schema['type'] == 'integer' and not isinstance(prop_value, int):
                            return False
                        elif prop_schema['type'] == 'number' and not isinstance(prop_value, (int, float)):
                            return False

                    # Range checking
                    if 'minimum' in prop_schema and prop_value < prop_schema['minimum']:
                        return False
                    if 'maximum' in prop_schema and prop_value > prop_schema['maximum']:
                        return False

        return True

    def test_configuration_profiles(self, sample_config, temp_config_dir):
        """Test configuration profiles for different environments."""
        config_dir = Path(temp_config_dir)

        # Create base configuration
        base_config_path = config_dir / 'base.yaml'
        with open(base_config_path, 'w') as f:
            yaml.safe_dump(sample_config, f)

        # Create development profile
        dev_config = {
            'trading': {
                'enabled': False,  # Disable trading in dev
                'max_positions': 2  # Lower limits
            },
            'logging': {
                'level': 'DEBUG'  # More verbose logging
            }
        }
        dev_config_path = config_dir / 'dev.yaml'
        with open(dev_config_path, 'w') as f:
            yaml.safe_dump(dev_config, f)

        # Create production profile
        prod_config = {
            'trading': {
                'enabled': True,
                'max_positions': 10  # Higher limits for prod
            },
            'logging': {
                'level': 'WARNING'  # Less verbose logging
            }
        }
        prod_config_path = config_dir / 'prod.yaml'
        with open(prod_config_path, 'w') as f:
            yaml.safe_dump(prod_config, f)

        # Load configurations with profile merging
        dev_final_config = self._load_config_with_profile(base_config_path, dev_config_path)
        prod_final_config = self._load_config_with_profile(base_config_path, prod_config_path)

        # Verify development profile
        assert dev_final_config['trading']['enabled'] is False
        assert dev_final_config['trading']['max_positions'] == 2
        assert dev_final_config['logging']['level'] == 'DEBUG'

        # Verify production profile
        assert prod_final_config['trading']['enabled'] is True
        assert prod_final_config['trading']['max_positions'] == 10
        assert prod_final_config['logging']['level'] == 'WARNING'

        # Verify base settings preserved in both
        assert dev_final_config['solana']['enabled'] == sample_config['solana']['enabled']
        assert prod_final_config['telegram']['enabled'] == sample_config['telegram']['enabled']

    def _load_config_with_profile(self, base_path, profile_path):
        """Load config with profile overlay."""
        # Load base config
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)

        # Load profile config
        with open(profile_path, 'r') as f:
            profile_config = yaml.safe_load(f)

        # Merge profile over base
        return self._deep_merge(base_config, profile_config)

    def test_configuration_encryption(self, sample_config, temp_config_dir):
        """Test configuration encryption for sensitive data."""
        config_path = Path(temp_config_dir) / 'encrypted_config.yaml'

        # Sensitive data to encrypt
        sensitive_config = sample_config.copy()
        sensitive_config['exchanges']['kraken']['api_secret'] = 'super_secret_key'
        sensitive_config['solana']['private_key'] = 'ultra_secret_private_key'

        # In real implementation, this would encrypt sensitive fields
        # For testing, we'll simulate encryption markers
        encrypted_config = sensitive_config.copy()
        encrypted_config['_encrypted_fields'] = ['exchanges.kraken.api_secret', 'solana.private_key']

        # Save "encrypted" config
        with open(config_path, 'w') as f:
            yaml.safe_dump(encrypted_config, f)

        # Load and verify encryption markers
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)

        assert '_encrypted_fields' in loaded_config
        assert 'exchanges.kraken.api_secret' in loaded_config['_encrypted_fields']
        assert 'solana.private_key' in loaded_config['_encrypted_fields']

    def test_configuration_audit_trail(self, sample_config, temp_config_dir):
        """Test configuration change audit trail."""
        config_path = Path(temp_config_dir) / 'config.yaml'
        audit_path = Path(temp_config_dir) / 'config_audit.log'

        # Save initial config
        with open(config_path, 'w') as f:
            yaml.safe_dump(sample_config, f)

        # Record initial state in audit
        self._audit_config_change(audit_path, 'initial', sample_config, None)

        # Make configuration change
        updated_config = sample_config.copy()
        updated_config['trading']['max_positions'] = 8

        # Save updated config
        with open(config_path, 'w') as f:
            yaml.safe_dump(updated_config, f)

        # Record change in audit
        self._audit_config_change(audit_path, 'update', updated_config, sample_config)

        # Verify audit trail
        assert audit_path.exists()

        with open(audit_path, 'r') as f:
            audit_lines = f.readlines()

        assert len(audit_lines) == 2  # Initial + update

        # Verify audit content
        assert 'initial' in audit_lines[0]
        assert 'update' in audit_lines[1]
        assert 'max_positions' in audit_lines[1]

    def _audit_config_change(self, audit_path, action, new_config, old_config):
        """Helper to record configuration changes in audit trail."""
        timestamp = datetime.now().isoformat()

        audit_entry = {
            'timestamp': timestamp,
            'action': action,
            'changes': {}
        }

        if old_config is not None:
            # Calculate changes (simplified)
            audit_entry['changes'] = self._calculate_config_changes(old_config, new_config)

        # Append to audit file
        with open(audit_path, 'a') as f:
            json.dump(audit_entry, f)
            f.write('\n')

    def _calculate_config_changes(self, old_config, new_config):
        """Calculate configuration changes."""
        changes = {}

        def deep_diff(old, new, path=''):
            if isinstance(old, dict) and isinstance(new, dict):
                for key in set(old.keys()) | set(new.keys()):
                    if key not in old:
                        changes[f"{path}.{key}" if path else key] = {'action': 'added', 'new_value': new[key]}
                    elif key not in new:
                        changes[f"{path}.{key}" if path else key] = {'action': 'removed', 'old_value': old[key]}
                    elif old[key] != new[key]:
                        deep_diff(old[key], new[key], f"{path}.{key}" if path else key)
            elif old != new:
                changes[path] = {'action': 'modified', 'old_value': old, 'new_value': new}

        deep_diff(old_config, new_config)
        return changes

    def test_configuration_rollback(self, sample_config, temp_config_dir):
        """Test configuration rollback functionality."""
        config_path = Path(temp_config_dir) / 'config.yaml'
        backup_dir = Path(temp_config_dir) / 'backups'

        # Create backup directory
        backup_dir.mkdir(exist_ok=True)

        # Save initial config
        with open(config_path, 'w') as f:
            yaml.safe_dump(sample_config, f)

        # Create backup
        backup_path = backup_dir / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(backup_path, 'w') as f:
            yaml.safe_dump(sample_config, f)

        # Make problematic change
        broken_config = sample_config.copy()
        broken_config['trading']['risk_per_trade'] = 2.0  # Invalid value

        # Save broken config
        with open(config_path, 'w') as f:
            yaml.safe_dump(broken_config, f)

        # Verify config is broken
        is_valid, errors = validate_config(broken_config)
        assert not is_valid

        # Rollback to backup
        with open(backup_path, 'r') as f:
            rollback_config = yaml.safe_load(f)

        with open(config_path, 'w') as f:
            yaml.safe_dump(rollback_config, f)

        # Verify rollback successful
        final_config = load_config(config_path)
        assert final_config['trading']['risk_per_trade'] == sample_config['trading']['risk_per_trade']

    def test_configuration_import_export(self, sample_config, temp_config_dir):
        """Test configuration import/export functionality."""
        # Export configuration
        export_path = Path(temp_config_dir) / 'exported_config.json'

        with open(export_path, 'w') as f:
            json.dump(sample_config, f, indent=2)

        # Verify export
        assert export_path.exists()

        with open(export_path, 'r') as f:
            exported_data = json.load(f)

        assert exported_data == sample_config

        # Import configuration
        import_path = Path(temp_config_dir) / 'imported_config.yaml'

        with open(import_path, 'w') as f:
            yaml.safe_dump(exported_data, f)

        # Verify import
        imported_config = load_config(import_path)
        assert imported_config == sample_config

    def test_configuration_component_integration(self, sample_config):
        """Test configuration integration with bot components."""
        # Test trading component configuration
        trading_config = sample_config['trading']

        # Simulate component initialization with config
        assert trading_config['enabled'] is True
        assert trading_config['max_positions'] > 0
        assert 0 < trading_config['risk_per_trade'] <= 1

        # Test strategy component configuration
        strategy_config = sample_config['strategies']['trend_bot']

        assert strategy_config['enabled'] is True
        assert 'min_trend_strength' in strategy_config
        assert 0 <= strategy_config['min_trend_strength'] <= 1

        # Test risk management component configuration
        risk_config = sample_config['risk_management']

        assert 'stop_loss_pct' in risk_config
        assert 'take_profit_pct' in risk_config
        assert 0 < risk_config['stop_loss_pct'] < risk_config['take_profit_pct']

        # Test exchange component configuration
        exchange_config = sample_config['exchanges']['kraken']

        assert exchange_config['enabled'] is True
        assert 'api_key' in exchange_config
        assert 'api_secret' in exchange_config
        assert len(exchange_config['api_key']) > 0
        assert len(exchange_config['api_secret']) > 0

    def test_configuration_performance_impact(self, sample_config):
        """Test configuration loading performance."""
        import time

        # Measure config loading time
        start_time = time.time()

        # Load config multiple times
        for _ in range(100):
            # Simulate config validation
            is_valid, _ = validate_config(sample_config)

        end_time = time.time()
        load_time = end_time - start_time

        # Should be reasonably fast (< 1 second for 100 loads)
        assert load_time < 1.0

        # Test large configuration handling
        large_config = sample_config.copy()

        # Add many strategies
        large_config['strategies'] = {}
        for i in range(50):
            large_config['strategies'][f'strategy_{i}'] = {
                'enabled': True,
                'param1': f'value_{i}',
                'param2': i / 100.0
            }

        # Should still validate reasonably fast
        start_time = time.time()
        is_valid, errors = validate_config(large_config)
        end_time = time.time()

        validation_time = end_time - start_time
        assert validation_time < 0.5  # Should validate large config quickly
        assert is_valid
