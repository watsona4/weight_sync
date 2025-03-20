FROM python:3.12-alpine AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.12-alpine

WORKDIR /app

COPY --from=builder /app/wheels /wheels

RUN pip install --no-cache --break-system-packages /wheels/*

ENV TZ="America/New_York"
RUN cp /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY weight_sync.py .
COPY excel_interface.py .

LABEL org.opencontainers.image.source=https://github.com/watsona4/weight_sync

CMD ["gunicorn", "--bind=192.168.1.5:8568", "--access-logfile", "-", "--error-logfile", "-", "weight_sync:app"]
