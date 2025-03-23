import logging
import os
import os.path
import shelve

import pandas as pd
from flask import Flask, request
from sortedcontainers import SortedDict
from werkzeug.middleware.proxy_fix import ProxyFix

from excel_interface import LOG as EXCEL_LOG
from excel_interface import auth, sync

TZ: str = str(os.environ.get("TZ", "UTC"))
LOG_LEVEL: str = str(os.environ.get("LOG_LEVEL", "INFO"))

logging.basicConfig()
LOG: logging.Logger = logging.getLogger("weight_sync")

LOG.setLevel(getattr(logging, LOG_LEVEL.upper(), None))  # type: ignore
EXCEL_LOG.setLevel(getattr(logging, LOG_LEVEL.upper(), None))  # type: ignore

app: Flask = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=2, x_host=2)  # type: ignore


def parse_weight(weight: str) -> float:
    val = weight.split()[0]
    return float(val) / 453.6


def parse_percentage(percentage: str) -> float:
    val = percentage.strip("%")
    return float(val)


def parse_timestamp(timestamp: str) -> pd.Period:
    pd_ts = pd.Timestamp(timestamp, tz="UTC")
    return pd_ts.astimezone(TZ).to_period("s")  # type: ignore


DB_FILENAME: str = "/data/weight_data.db"


@app.post("/auth")
def post_auth() -> str:

    try:
        auth()
    except Exception as e:
        LOG.exception("Error in auth")
        return str(e)

    return "200 ok"


@app.post("/")
def post_sync() -> str:

    if os.path.exists(DB_FILENAME):
        with shelve.open(DB_FILENAME) as db:
            data = db["data"]
    else:
        data = SortedDict({})

    raw_data = request.get_json()
    LOG.debug(f"post(): {len(raw_data)=}")
    LOG.debug(f"post(): {raw_data=}")

    weight = False
    for key, val in raw_data.items():
        if "grams" in val:
            weight = True
            data.setdefault(parse_timestamp(key), [None, None])[0] = parse_weight(val)
        else:
            data.setdefault(parse_timestamp(key), [None, None])[1] = parse_percentage(val)

    LOG.debug(f"post(): {weight=}")
    LOG.debug(f"post(): {len(data)=}")
    LOG.debug(f"post(): {data=}")

    try:
        sync(data)
    except Exception as e:
        LOG.exception("Error in sync")
        return str(e)

    with shelve.open(DB_FILENAME) as db:
        db["data"] = data

    return "200 OK"
