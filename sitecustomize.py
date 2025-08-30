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

# Provide lightweight fallbacks for heavy deps if not installed
try:  # prefer real numpy
    import numpy  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    try:
        from numpy_stub import *  # type: ignore
        import numpy_stub as _np_stub  # type: ignore
        sys.modules.setdefault('numpy', _np_stub)
    except Exception:
        pass

try:  # prefer real pandas
    import pandas  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    try:
        from pandas_stub import *  # type: ignore
        import pandas_stub as _pd_stub  # type: ignore
        sys.modules.setdefault('pandas', _pd_stub)
    except Exception:
        pass
