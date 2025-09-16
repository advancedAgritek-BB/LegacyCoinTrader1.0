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

# Remove pandas stub import since pandas is available and the stub causes DataFrame conversion issues
# try:  # prefer real pandas
#     import pandas  # type: ignore
# except Exception:  # pragma: no cover - fallback stub
#     try:
#         from pandas_stub import *  # type: ignore
#         import pandas_stub as _pd_stub  # type: ignore
#         sys.modules.setdefault('pandas', _pd_stub)
#     except Exception:
#         pass
