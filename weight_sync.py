import logging
import os
import os.path
import shelve

import pandas as pd
from flask import Flask, request
from sortedcontainers import SortedDict
from werkzeug.middleware.proxy_fix import ProxyFix

from excel_interface import sync

TZ: str = str(os.environ.get("TZ", "UTC"))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)


def parse_weight(weight):
    val = weight.split()[0]
    return float(val) / 453.6


def parse_percentage(percentage):
    val = percentage.strip("%")
    return float(val)


def parse_timestamp(timestamp):
    pd_ts = pd.Timestamp(timestamp, tz="UTC")
    return pd_ts.astimezone(TZ).to_period("s")


DB_FILENAME = "/data/weight_data.db"


@app.post("/")
def post():

    if os.path.exists(DB_FILENAME):
        with shelve.open(DB_FILENAME) as db:
            data = db["data"]
    else:
        data = SortedDict({})

    raw_data = request.json
    logger.debug(f"post(): {len(raw_data)=}")
    logger.debug(f"post(): {raw_data=}")

    weight = False
    for key, val in raw_data.items():
        if "grams" in val:
            weight = True
            data.setdefault(parse_timestamp(key), [None, None])[0] = parse_weight(val)
        else:
            data.setdefault(parse_timestamp(key), [None, None])[1] = parse_percentage(val)

    logger.debug(f"post(): {weight=}")
    logger.debug(f"post(): {len(data)=}")
    logger.debug(f"post(): {data=}")

    try:
        sync(data)
    except Exception as e:
        logger.error(str(e), exc_info=True)
        return str(e)

    with shelve.open(DB_FILENAME) as db:
        db["data"] = data

    return "200 OK"
