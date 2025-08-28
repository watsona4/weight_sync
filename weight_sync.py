import datetime
import logging
import os
import threading

import cachelib  # type: ignore
import pandas as pd
from flask import Flask, request
from sortedcontainers import SortedDict
from werkzeug.middleware.proxy_fix import ProxyFix

from excel_interface import LOG as EXCEL_LOG
from excel_interface import ExcelInterface

TZ: str = os.environ.get("TZ", "UTC")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig()
LOG: logging.Logger = logging.getLogger("weight_sync")

LOG.setLevel(getattr(logging, LOG_LEVEL.upper(), "INFO"))
EXCEL_LOG.setLevel(getattr(logging, LOG_LEVEL.upper(), "INFO"))

app: Flask = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_prefix=1)  # type: ignore

SERVER: str = "https://home.battenkillwoodworks.com/sync"

CACHE = cachelib.RedisCache(host="redis", port=6379, password=os.environ.get("REDIS_PASSWORD"))

LOCK = threading.Lock()


def parse_weight(weight: str) -> float:
    val = weight.split()[0]
    return float(val) / 453.6


def parse_percentage(percentage: str) -> float:
    val = percentage.strip("%")
    return float(val)


def parse_timestamp(timestamp: str) -> pd.Period:
    pd_ts = pd.Timestamp(timestamp, tz="UTC")
    return pd_ts.astimezone(TZ).to_period("s")  # type: ignore


def parse_date(timestamp: str | None = None, period: pd.Period | None = None) -> datetime.date:
    if timestamp is not None:
        pd_ts = pd.Timestamp(timestamp, tz=TZ)
    if period is not None:
        pd_ts = period.to_timestamp().tz_localize(TZ)
    return pd_ts.date()


@app.post("/")
def sync():

    LOG.debug(f"sync(): {request=}")

    if CACHE.has("data"):
        data = CACHE.get("data")
    else:
        data = SortedDict({})

    raw_data = request.get_json()
    LOG.debug(f"sync(): {len(raw_data)=}")
    LOG.debug(f"sync(): {raw_data=}")

    for key, val in raw_data.items():
        date = parse_timestamp(key)
        if "grams" in val:
            data.setdefault(date, [None, None])[0] = parse_weight(val)
        else:
            data.setdefault(date, [None, None])[1] = parse_percentage(val)

    LOG.debug(f"sync(): {len(data)=}")
    LOG.debug(f"sync(): {data=}")

    with LOCK:
        ExcelInterface.sync(data)

    CACHE.add("data", data)

    return "OK", 200


@app.get("/health")
def health():
    LOG.debug(f"health(): {request=}")
    return "OK", 200
