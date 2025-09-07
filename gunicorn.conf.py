# Gunicorn configuration for LegacyCoinTrader
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"
worker_connections = 1000

# Restart workers after this many requests, with randomness
max_requests = 1000
max_requests_jitter = 50

# Timeout for handling requests
timeout = 120
keepalive = 10

# Logging
loglevel = "info"
accesslog = "/Users/brandonburnette/Downloads/LegacyCoinTrader1.0/logs/gunicorn_access.log"
errorlog = "/Users/brandonburnette/Downloads/LegacyCoinTrader1.0/logs/gunicorn_error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "legacycointrader"

# Server mechanics
preload_app = True
pidfile = "/Users/brandonburnette/Downloads/LegacyCoinTrader1.0/gunicorn.pid"
user = os.getenv("USER", "nobody")
group = os.getenv("USER", "nobody")
tmp_upload_dir = None

# SSL (if needed in the future)
# keyfile = "/path/to/ssl/private.key"
# certfile = "/path/to/ssl/certificate.crt"
