"""Tests for wallet manager functionality."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from crypto_bot.wallet_manager import (
    WalletManager,
    get_wallet,
    validate_private_key,
    create_wallet_from_seed
)


class TestWalletManager:
    """Test suite for Wallet Manager."""

    @pytest.fixture
    def wallet_manager(self):
        return WalletManager()

    @pytest.fixture
    def mock_keypair(self):
        """Mock Solana Keypair for testing."""
        keypair = Mock()
        keypair.public_key = Mock()
        keypair.public_key.to_base58.return_value = "test_public_key_123"
        keypair.secret_key = b"test_secret_key"
        return keypair

    def test_wallet_manager_init(self, wallet_manager):
        """Test wallet manager initialization."""
        assert wallet_manager is not None
        assert hasattr(wallet_manager, 'wallets')

    @patch('crypto_bot.wallet_manager.Keypair')
    def test_get_wallet_missing(self, mock_keypair):
        """Test getting wallet when environment variable is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="SOLANA_PRIVATE_KEY environment variable not set"):
                get_wallet()

    @patch('crypto_bot.wallet_manager.Keypair')
    def test_get_wallet_invalid(self, mock_keypair):
        """Test getting wallet with invalid private key."""
        with patch.dict(os.environ, {'SOLANA_PRIVATE_KEY': 'not-json'}):
            with pytest.raises(ValueError, match="Invalid private key format"):
                get_wallet()

    @patch('crypto_bot.wallet_manager.Keypair')
    def test_get_wallet_valid(self, mock_keypair):
        """Test getting wallet with valid private key."""
        mock_keypair.from_secret_key.return_value = Mock()
        
        with patch.dict(os.environ, {'SOLANA_PRIVATE_KEY': '[1,2,3,4,5]'}):
            wallet = get_wallet()
            assert wallet is not None

    def test_validate_private_key_valid(self):
        """Test validation of valid private key."""
        valid_key = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        
        result = validate_private_key(valid_key)
        assert result == True

    def test_validate_private_key_invalid_length(self):
        """Test validation of private key with invalid length."""
        invalid_key = [1, 2, 3, 4, 5]  # Too short
        
        with pytest.raises(ValueError, match="Private key must be exactly 32 bytes"):
            validate_private_key(invalid_key)

    def test_validate_private_key_invalid_type(self):
        """Test validation of private key with invalid type."""
        invalid_key = "not_a_list"
        
        with pytest.raises(ValueError, match="Private key must be a list of integers"):
            validate_private_key(invalid_key)

    def test_validate_private_key_invalid_values(self):
        """Test validation of private key with invalid values."""
        invalid_key = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, "invalid"]
        
        with pytest.raises(ValueError, match="All private key values must be integers"):
            validate_private_key(invalid_key)

    @patch('crypto_bot.wallet_manager.Keypair')
    def test_create_wallet_from_seed(self, mock_keypair, mock_keypair_instance):
        """Test creating wallet from seed phrase."""
        mock_keypair.from_seed.return_value = mock_keypair_instance
        
        seed_phrase = "test seed phrase for wallet creation"
        wallet = create_wallet_from_seed(seed_phrase)
        
        assert wallet is not None
        mock_keypair.from_seed.assert_called_once()

    def test_wallet_manager_add_wallet(self, wallet_manager, mock_keypair):
        """Test adding wallet to manager."""
        wallet_id = "test_wallet_1"
        
        wallet_manager.add_wallet(wallet_id, mock_keypair)
        
        assert wallet_id in wallet_manager.wallets
        assert wallet_manager.wallets[wallet_id] == mock_keypair

    def test_wallet_manager_get_wallet(self, wallet_manager, mock_keypair):
        """Test getting wallet from manager."""
        wallet_id = "test_wallet_1"
        wallet_manager.add_wallet(wallet_id, mock_keypair)
        
        retrieved_wallet = wallet_manager.get_wallet(wallet_id)
        assert retrieved_wallet == mock_keypair

    def test_wallet_manager_get_nonexistent_wallet(self, wallet_manager):
        """Test getting non-existent wallet from manager."""
        with pytest.raises(KeyError):
            wallet_manager.get_wallet("nonexistent_wallet")

    def test_wallet_manager_remove_wallet(self, wallet_manager, mock_keypair):
        """Test removing wallet from manager."""
        wallet_id = "test_wallet_1"
        wallet_manager.add_wallet(wallet_id, mock_keypair)
        
        wallet_manager.remove_wallet(wallet_id)
        
        assert wallet_id not in wallet_manager.wallets

    def test_wallet_manager_list_wallets(self, wallet_manager, mock_keypair):
        """Test listing all wallets in manager."""
        wallet_manager.add_wallet("wallet_1", mock_keypair)
        wallet_manager.add_wallet("wallet_2", mock_keypair)
        
        wallet_list = wallet_manager.list_wallets()
        
        assert "wallet_1" in wallet_list
        assert "wallet_2" in wallet_list
        assert len(wallet_list) == 2

    def test_wallet_manager_clear_wallets(self, wallet_manager, mock_keypair):
        """Test clearing all wallets from manager."""
        wallet_manager.add_wallet("wallet_1", mock_keypair)
        wallet_manager.add_wallet("wallet_2", mock_keypair)
        
        wallet_manager.clear_wallets()
        
        assert len(wallet_manager.wallets) == 0

    def test_wallet_manager_wallet_count(self, wallet_manager, mock_keypair):
        """Test getting wallet count."""
        assert wallet_manager.wallet_count() == 0
        
        wallet_manager.add_wallet("wallet_1", mock_keypair)
        assert wallet_manager.wallet_count() == 1
        
        wallet_manager.add_wallet("wallet_2", mock_keypair)
        assert wallet_manager.wallet_count() == 2

    def test_wallet_manager_has_wallet(self, wallet_manager, mock_keypair):
        """Test checking if wallet exists."""
        assert wallet_manager.has_wallet("wallet_1") == False
        
        wallet_manager.add_wallet("wallet_1", mock_keypair)
        assert wallet_manager.has_wallet("wallet_1") == True

    @patch('crypto_bot.wallet_manager.Keypair')
    def test_wallet_import_export(self, mock_keypair, mock_keypair_instance):
        """Test wallet import and export functionality."""
        # Mock the keypair methods
        mock_keypair_instance.secret_key = b"test_secret_key_32_bytes_long_12345"
        mock_keypair_instance.public_key.to_base58.return_value = "test_public_key"
        
        # Test export
        exported_data = {
            'public_key': mock_keypair_instance.public_key.to_base58(),
            'secret_key': list(mock_keypair_instance.secret_key)
        }
        
        assert exported_data['public_key'] == "test_public_key"
        assert len(exported_data['secret_key']) == 32

    def test_wallet_manager_duplicate_wallet_id(self, wallet_manager, mock_keypair):
        """Test adding wallet with duplicate ID."""
        wallet_id = "duplicate_id"
        
        wallet_manager.add_wallet(wallet_id, mock_keypair)
        
        # Adding with same ID should overwrite
        new_keypair = Mock()
        wallet_manager.add_wallet(wallet_id, new_keypair)
        
        assert wallet_manager.wallets[wallet_id] == new_keypair
        assert len(wallet_manager.wallets) == 1

    def test_wallet_manager_invalid_wallet_id(self, wallet_manager, mock_keypair):
        """Test adding wallet with invalid ID."""
        invalid_ids = ["", None, 123, [], {}]
        
        for invalid_id in invalid_ids:
            with pytest.raises(ValueError, match="Wallet ID must be a non-empty string"):
                wallet_manager.add_wallet(invalid_id, mock_keypair)

    def test_wallet_manager_invalid_wallet_object(self, wallet_manager):
        """Test adding invalid wallet object."""
        invalid_wallets = [None, "not_a_wallet", 123, [], {}]
        
        for invalid_wallet in invalid_wallets:
            with pytest.raises(ValueError, match="Wallet must be a valid Keypair object"):
                wallet_manager.add_wallet("test_id", invalid_wallet)


@pytest.mark.integration
class TestWalletManagerIntegration:
    """Integration tests for wallet manager."""

    @patch('crypto_bot.wallet_manager.Keypair')
    def test_full_wallet_lifecycle(self, mock_keypair):
        """Test complete wallet lifecycle."""
        mock_keypair.from_secret_key.return_value = Mock()
        
        # Create wallet manager
        manager = WalletManager()
        
        # Add multiple wallets
        wallet1 = Mock()
        wallet2 = Mock()
        
        manager.add_wallet("wallet_1", wallet1)
        manager.add_wallet("wallet_2", wallet2)
        
        # Verify wallets exist
        assert manager.wallet_count() == 2
        assert manager.has_wallet("wallet_1")
        assert manager.has_wallet("wallet_2")
        
        # Get wallets
        retrieved1 = manager.get_wallet("wallet_1")
        retrieved2 = manager.get_wallet("wallet_2")
        
        assert retrieved1 == wallet1
        assert retrieved2 == wallet2
        
        # Remove wallet
        manager.remove_wallet("wallet_1")
        assert manager.wallet_count() == 1
        assert not manager.has_wallet("wallet_1")
        
        # Clear all
        manager.clear_wallets()
        assert manager.wallet_count() == 0

    @patch('crypto_bot.wallet_manager.Keypair')
    def test_environment_wallet_creation(self, mock_keypair):
        """Test wallet creation from environment variables."""
        mock_keypair.from_secret_key.return_value = Mock()
        
        # Test with valid environment variable
        with patch.dict(os.environ, {'SOLANA_PRIVATE_KEY': '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]'}):
            wallet = get_wallet()
            assert wallet is not None

    def test_private_key_validation_edge_cases(self):
        """Test private key validation edge cases."""
        # Test with exactly 32 bytes
        valid_key = list(range(32))
        assert validate_private_key(valid_key) == True
        
        # Test with 31 bytes (too short)
        short_key = list(range(31))
        with pytest.raises(ValueError):
            validate_private_key(short_key)
        
        # Test with 33 bytes (too long)
        long_key = list(range(33))
        with pytest.raises(ValueError):
            validate_private_key(long_key)
        
        # Test with mixed valid/invalid values
        mixed_key = list(range(31)) + ["invalid"]
        with pytest.raises(ValueError):
            validate_private_key(mixed_key)


@pytest.mark.solana
def test_solana_specific_wallet_features():
    """Test Solana-specific wallet functionality."""
    # Test that wallet manager can handle Solana-specific features
    manager = WalletManager()
    
    # Should be able to create wallet manager
    assert manager is not None
    assert hasattr(manager, 'wallets')
    assert isinstance(manager.wallets, dict)

