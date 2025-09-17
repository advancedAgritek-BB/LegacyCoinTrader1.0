FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/token_discovery/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services /app/services
COPY crypto_bot /app/crypto_bot

WORKDIR /app/services/token_discovery
ENV PYTHONPATH=/app

EXPOSE 8005

CMD ["uvicorn", "services.token_discovery.app:app", "--host", "0.0.0.0", "--port", "8005"]
