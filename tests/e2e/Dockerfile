FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY tests/e2e/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy test files
COPY tests/e2e/ /app/tests/

# Create test results directory
RUN mkdir -p /app/test_results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "/app/tests/test_e2e_framework.py"]
