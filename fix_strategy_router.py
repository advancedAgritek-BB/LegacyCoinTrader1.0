#!/usr/bin/env python3
"""Fix malformed type annotations in strategy_router.py"""

with open('crypto_bot/strategy_router.py', 'r') as f:
    content = f.read()

# Fix the malformed types
content = content.replace('Union[RouterConfig, Mapping][str, Any] | None', 'Optional[Union[RouterConfig, Mapping[str, Any]]]')
content = content.replace('Union[RouterConfig, dict] | None', 'Optional[Union[RouterConfig, dict]]')

with open('crypto_bot/strategy_router.py', 'w') as f:
    f.write(content)

print("Fixed malformed types in strategy_router.py")
