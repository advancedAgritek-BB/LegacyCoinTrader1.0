try:
    from .app import app  # re-export Flask app for tests
except Exception:  # pragma: no cover - allow tests to import package without Flask
    app = None


