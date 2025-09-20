FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install runtime dependencies required by health checks and tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared dependency manifests and install requirements
COPY requirements.txt /tmp/base-requirements.txt
COPY frontend/requirements.txt /tmp/frontend-requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/base-requirements.txt \
    && pip install --no-cache-dir -r /tmp/frontend-requirements.txt

# Copy application and shared libraries
COPY frontend /app/frontend
COPY crypto_bot /app/crypto_bot
COPY libs /app/libs
COPY services /app/services

WORKDIR /app/frontend
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Run the ASGI application from the frontend package
CMD ["uvicorn", "frontend.app:asgi_app", "--host", "0.0.0.0", "--port", "5000"]
