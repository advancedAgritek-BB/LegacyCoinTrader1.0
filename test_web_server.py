#!/usr/bin/env python3
"""
Simple web server test
"""

import os
import sys
from pathlib import Path
import pytest
from werkzeug.serving import make_server

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.mark.integration
def test_flask_app_health_endpoint() -> None:
    from frontend.app import app
    with app.test_client() as client:
        res = client.get('/health')
        assert res.status_code == 200
        assert res.is_json
        data = res.get_json()
        assert data.get('status') == 'ok'


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv('FLASK_RUN_E2E', '0') != '1',
    reason='Set FLASK_RUN_E2E=1 to run live server binding test'
)
def test_flask_app_can_bind_ephemeral_port() -> None:
    from frontend.app import app
    # Bind to port 0 to request an ephemeral port from the OS
    server = make_server('127.0.0.1', 0, app)
    try:
        port = server.server_port
        assert port > 0
    finally:
        server.server_close()
