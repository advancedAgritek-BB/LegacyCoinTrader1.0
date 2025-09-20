import sys
import types

try:  # use real PyYAML if available
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    if 'yaml' not in sys.modules:
        sys.modules['yaml'] = types.SimpleNamespace(
            safe_load=lambda *a, **k: {},
            safe_dump=lambda *a, **k: '',
            dump=lambda *a, **k: ''
        )

# Ensure numpy is available; the project requires the real dependency.
try:  # prefer real numpy
    import numpy  # type: ignore  # noqa: F401 - imported for side effects
except Exception as exc:  # pragma: no cover - dependency is mandatory
    raise ImportError(
        "numpy is required for LegacyCoinTrader. Please install it via requirements.txt "
        "or pyproject.toml."
    ) from exc

# Ensure critical subpackages register with sys.modules before pytest applies import shims.
try:  # pragma: no cover - best effort guard
    import crypto_bot.regime  # noqa: F401
except Exception:
    pass

# Load environment variables from project-level .env files early so every entry point sees them.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - tolerate missing dependency
    load_dotenv = None

if load_dotenv is not None:
    from pathlib import Path

    project_root = Path(__file__).resolve().parent
    # Standard behaviour: load working-directory .env first, then project fallbacks without overriding.
    load_dotenv(override=False)
    for candidate in (project_root / ".env", project_root / "crypto_bot" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
