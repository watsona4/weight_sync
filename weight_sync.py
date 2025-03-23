import logging
import os
import os.path
import shelve

import pandas as pd
from cachelib import FileSystemCache  # type: ignore
from flask import Flask, redirect, request, session, url_for
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

LOG.setLevel(getattr(logging, LOG_LEVEL.upper(), None))  # type: ignore
EXCEL_LOG.setLevel(getattr(logging, LOG_LEVEL.upper(), None))  # type: ignore

app: Flask = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_default_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)  # type: ignore
app.proto = "https"
app.server_name = "home.battenkillwoodworks.com"
app.application_base = "/sync"

FLOW = InstalledAppFlow.from_client_secrets_file(CREDENTIALS, SCOPES)

CACHE = FileSystemCache(cache_dir="/data")


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


@app.route("/auth", methods=["GET", "POST"])
def auth():

    LOG.debug(f"auth(): {request=}")
    if os.path.exists(TOKEN):
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
        if not creds.valid:
            creds.refresh(Request())
        return "OK", 200

    FLOW.redirect_uri = (  # url_for("callback", _external=True)
        "https://home.battenkillwoodworks.com/sync/callback"
    )
    LOG.debug(f"auth(): {FLOW.redirect_uri=}")

    authorization_url, state = FLOW.authorization_url(
        access_type="offline", include_granted_scopes="true"
    )
    LOG.debug(f"auth(): {authorization_url=}")
    LOG.debug(f"auth(): {state=}")
    CACHE.add("state", state)

    return redirect(authorization_url)


@app.get("/callback")
def callback():

    LOG.debug(f"callback(): {request=}")
    state = CACHE.get("state")
    LOG.debug(f"callback(): {state=}")
    # if state != request.args["state"]:
    #     return "Invalid state parameter", 401

    FLOW.fetch_token(authorization_response=request.url.replace("http", "https"))

    creds = FLOW.credentials
    CACHE.add("creds", creds)

    with open(TOKEN, "w") as token:
        token.write(creds.to_json())

    return "OK", 200


@app.post("/")
def sync():

    if not CACHE.has("creds"):
        return redirect("https://home.battenkillwoodworks.com/sync/auth")  # url_for("auth"))

    if os.path.exists(DB_FILENAME):
        with shelve.open(DB_FILENAME) as db:
            data = db["data"]
    else:
        data = SortedDict({})

    raw_data = request.get_json()
    LOG.debug(f"sync(): {len(raw_data)=}")
    LOG.debug(f"sync(): {raw_data=}")

    weight = False
    for key, val in raw_data.items():
        if "grams" in val:
            weight = True
            data.setdefault(parse_timestamp(key), [None, None])[0] = parse_weight(val)
        else:
            data.setdefault(parse_timestamp(key), [None, None])[1] = parse_percentage(val)

    LOG.debug(f"sync(): {weight=}")
    LOG.debug(f"sync(): {len(data)=}")
    LOG.debug(f"sync(): {data=}")

    ExcelInterface.sync(data)

    with shelve.open(DB_FILENAME) as db:
        db["data"] = data

    return "OK", 200
