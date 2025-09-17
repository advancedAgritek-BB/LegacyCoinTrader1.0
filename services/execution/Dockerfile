FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY libs /app/libs
RUN pip install --no-cache-dir /app/libs

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY services /app/services
COPY crypto_bot /app/crypto_bot

WORKDIR /app/services/execution
ENV PYTHONPATH=/app

EXPOSE 8006

CMD ["uvicorn", "services.execution.main:app", "--host", "0.0.0.0", "--port", "8006"]
