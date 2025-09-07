"""
Token validation utilities to break circular imports.
"""

import re
from typing import Set

# Common base tokens that are typically valid
VALID_BASE_TOKENS: Set[str] = {
    "BTC", "ETH", "SOL", "ADA", "DOT", "LINK", "UNI", "MATIC", "AVAX", "ATOM",
    "NEAR", "FTM", "ALGO", "VET", "ICP", "FIL", "XLM", "TRX", "LTC", "BCH",
    "XRP", "DOGE", "SHIB", "LINK", "UNI", "AAVE", "COMP", "MKR", "SNX", "CRV",
    "YFI", "SUSHI", "1INCH", "BAL", "REN", "ZRX", "BAND", "KNC", "STORJ", "OMG",
    "NMR", "REP", "BAT", "ZEC", "DASH", "XMR", "ETC", "XTZ", "EOS", "IOTA",
    "NEO", "ONT", "QTUM", "VET", "ICX", "WAVES", "ZIL", "HBAR", "HOT", "ENJ",
    "MANA", "SAND", "AXS", "CHZ", "ANKR", "COTI", "DYDX", "IMX", "OP", "ARB",
    "SUI", "APT", "SEI", "TIA", "JUP", "PYTH", "BONK", "WIF", "POPCAT", "BOOK",
    "MYRO", "FLOKI", "PEPE", "SHIB", "DOGE", "BABYDOGE", "SAFEMOON", "MOON",
    "ROCKET", "ELON", "MUSK", "SPACE", "GALAXY", "COSMOS", "UNIVERSE", "STAR",
    "PLANET", "MOONSHOT", "LAMBO", "YACHT", "DREAM", "FUTURE", "FORTUNE",
    "WEALTH", "RICH", "MILLION", "BILLION", "TRILLION", "QUADRILLION"
}

# Patterns for invalid tokens
INVALID_TOKEN_PATTERNS = [
    r"^.{50,}$",  # Extremely long tokens (likely contract addresses)
    r".*[^A-Z0-9].*",  # Tokens with invalid characters
    r"^0x[a-fA-F0-9]{40}$",  # Ethereum contract addresses
    # Removed Base58 pattern as we now accept Solana addresses
    r".*[^A-Z0-9/-].*",  # Any non-alphanumeric characters except dash
]

# Solana Base58 address pattern (32-44 characters, Base58 alphabet)
SOLANA_ADDRESS_PATTERN = r"^[1-9A-HJ-NP-Za-km-z]{32,44}$"

def _is_valid_base_token(token: str) -> bool:
    """
    Check if a token is a valid base token.
    
    Args:
        token: The token to validate
        
    Returns:
        True if the token is valid, False otherwise
    """
    if not isinstance(token, str):
        return False
        
    # Convert to uppercase for comparison
    token_upper = token.upper()
    
    # Check if it's in our known valid tokens
    if token_upper in VALID_BASE_TOKENS:
        return True

    # Check if it's a Solana Base58 address (should be treated as valid for DEX)
    if re.match(SOLANA_ADDRESS_PATTERN, token):
        return True

    # Check against invalid patterns
    for pattern in INVALID_TOKEN_PATTERNS:
        if re.match(pattern, token_upper):
            return False
    
    # Additional validation rules
    if len(token) < 2 or len(token) > 20:
        return False
    
    # Must be alphanumeric
    if not token_upper.isalnum():
        return False
    
    # Must not be all numbers
    if token_upper.isdigit():
        return False
    
    # Must not be all the same character
    if len(set(token_upper)) == 1:
        return False
    
    return True
