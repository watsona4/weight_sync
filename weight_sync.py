import datetime
import logging
import os
import os.path

import cachelib  # type: ignore
import pandas as pd
from flask import Flask, redirect, request
from google.auth.transport.requests import Request  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
from sortedcontainers import SortedDict
from werkzeug.middleware.proxy_fix import ProxyFix

from excel_interface import CREDENTIALS
from excel_interface import LOG as EXCEL_LOG
from excel_interface import SCOPES, TOKEN, ExcelInterface

TZ: str = os.environ.get("TZ", "UTC")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig()
LOG: logging.Logger = logging.getLogger("weight_sync")

LOG.setLevel(getattr(logging, LOG_LEVEL.upper(), "INFO"))
EXCEL_LOG.setLevel(getattr(logging, LOG_LEVEL.upper(), "INFO"))

app: Flask = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_prefix=1)  # type: ignore

SERVER: str = "https://home.battenkillwoodworks.com/sync"

FLOW = InstalledAppFlow.from_client_secrets_file(CREDENTIALS, SCOPES)

CACHE = cachelib.RedisCache(
    host="redis", port=6379, db=0, password=os.environ.get("REDIS_PASSWORD")
)


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


DB_FILENAME: str = "/data/weight_data.db"
CUTOFF_DATE: datetime.date = parse_date(timestamp=os.environ.get("CUTOFF_DATE", "01/01/1900"))


@app.route("/auth", methods=["GET", "POST"])
def auth():

    LOG.debug(f"auth(): {request=}")
    if os.path.exists(TOKEN):
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
        if not creds.valid:
            creds.refresh(Request())
        CACHE.add("creds", creds)
        return "OK", 200

    FLOW.redirect_uri = f"{SERVER}/callback"

    LOG.debug(f"auth(): {FLOW.redirect_uri=}")

    authorization_url, state = FLOW.authorization_url(
        access_type="offline", include_granted_scopes="true"
    )
    LOG.debug(f"auth(): {authorization_url=}")
    LOG.debug(f"auth(): {state=}")
    CACHE.add("state", state)
    LOG.debug(f"auth(): {CACHE.get("state")=}")

    return redirect(authorization_url)


@app.get("/callback")
def callback():

    LOG.debug(f"callback(): {request=}")
    state = CACHE.get("state")
    LOG.debug(f"callback(): {state=}")
    if state != request.args["state"]:
        return "Invalid state parameter", 401

    FLOW.fetch_token(authorization_response=request.url.replace("http", "https"))

    creds = FLOW.credentials
    CACHE.add("creds", creds)
    LOG.debug(f"callback(): {CACHE.get("creds")=}")

    with open(TOKEN, "w") as token:
        token.write(creds.to_json())

    return "OK", 200


@app.post("/")
def sync():

    LOG.info(f"sync(): {request=}")
    LOG.debug(f"sync(): {CACHE.get("creds")=}")

    if not CACHE.has("creds"):
        return redirect(f"{SERVER}/auth")

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

    # data = {key: val for key, val in data.items() if parse_date(period=key) >= CUTOFF_DATE}
    LOG.debug(f"sync(): {len(data)=}")
    LOG.debug(f"sync(): {data=}")

    ExcelInterface.sync(data)

    CACHE.add("data", data)

    return "OK", 200


@app.get("/health")
def health():
    LOG.debug(f"health(): {request=}")
    return "OK", 200
