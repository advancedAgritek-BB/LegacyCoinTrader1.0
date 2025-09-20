FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY services /app/services
COPY crypto_bot /app/crypto_bot
COPY libs /app/libs

WORKDIR /app/services/market_data
ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["uvicorn", "services.market_data.app:app", "--host", "0.0.0.0", "--port", "8002"]
