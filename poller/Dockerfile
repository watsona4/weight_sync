FROM python:3.12-alpine AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.12-alpine

WORKDIR /app

COPY --from=builder /app/wheels /wheels

RUN pip install --no-cache --break-system-packages /wheels/*

COPY poller.py healthcheck.py ./

HEALTHCHECK CMD python healthcheck.py

LABEL org.opencontainers.image.source=https://github.com/watsona4/weight_sync

CMD ["python3", "poller.py"]
