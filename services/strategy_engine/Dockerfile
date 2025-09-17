FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY libs /app/libs
RUN pip install --no-cache-dir /app/libs

COPY services/strategy_engine/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services /app/services
COPY crypto_bot /app/crypto_bot

WORKDIR /app/services/strategy_engine
ENV PYTHONPATH=/app

EXPOSE 8004

CMD ["uvicorn", "services.strategy_engine.app:app", "--host", "0.0.0.0", "--port", "8004"]
