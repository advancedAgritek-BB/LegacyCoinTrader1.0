"""Frontend package initialization.

Intentionally empty to avoid shadowing the `frontend.app` module with a
variable named `app`, which breaks test patching like
`frontend.app.subprocess.Popen`.
"""
