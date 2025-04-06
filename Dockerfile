FROM python:3.12-alpine AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.12-alpine

WORKDIR /app

RUN apk add --no-cache curl

COPY --from=builder /app/wheels /wheels

RUN pip install --no-cache --break-system-packages /wheels/*

COPY weight_sync.py excel_interface.py .

LABEL org.opencontainers.image.source=https://github.com/watsona4/weight_sync

CMD ["gunicorn", "--access-logfile", "-", "--error-logfile", "-", "--bind=0.0.0.0:8000", "--log-level=debug", "weight_sync:app"]
