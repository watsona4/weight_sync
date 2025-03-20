import logging
import os.path
from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1JNA2ZuNQuX2TNNFF_vuLC4pIwzz_upBPfuAk1qrCqx4"

BOUNDS = (
    (-np.inf, -np.inf, -np.inf, -np.inf, 0, -np.inf, 0),
    (np.inf, np.inf, np.inf, np.inf, 7, np.inf, 1),
)


class DataType(Enum):
    WEIGHT = 1
    BODYFAT = 2


def f(t, w0, Δ0, c4, Δp1, τ1, Δp2, τ2, winf):
    return (
        winf
        + (
            w0
            - winf
            - Δp1 / 2 * np.cos(2 * np.pi / 7 * τ1)
            - Δp2 / 2 * np.cos(2 * np.pi * τ2)
            - c4
        )
        * 2 ** (-5 / 7 * t)
        + c4 * np.exp(Δ0 / c4 * t)
        + Δp1 / 2 * np.cos(2 * np.pi / 7 * (t - τ1))
        + Δp2 / 2 * np.cos(2 * np.pi * (t - τ2))
    )


def e(x, y, sol, winf=None):
    fpart = f(x, *sol, winf) if winf is not None else f(x, *sol)
    return 1 - np.sum((fpart - y) ** 2) / np.sum(np.array(y) ** 2)


def convert_excel(period):
    datetime_obj = period.to_timestamp()
    excel_epoch = pd.Timestamp("1900-01-01")
    delta = datetime_obj - excel_epoch
    excel_serial_number = delta.days + delta.seconds / (24 * 60 * 60) + 2
    return excel_serial_number


TOKEN = "/data/token.json"
CREDENTIALS = "/data/credentials.json"


class ExcelInterface:

    def __init__(self):
        self.sheet = None

    def sync(self, data):
        logger.debug(f"sync(): {len(data)=}")
        logger.debug(f"sync(): {data=}")
        self.auth()
        self.write_data(data)
        xw, yw, xf, yf = self.read_data()
        wwinf, wsol, _ = self.compute_parameters(DataType.WEIGHT, xw, yw)
        fwinf, fsol, _ = self.compute_parameters(DataType.BODYFAT, xf, yf)
        self.write_parameters(wwinf, wsol, fwinf, fsol)

    def auth(self):
        logger.debug("auth():")
        creds = None
        if os.path.exists(TOKEN):
            creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(TOKEN, "w") as token:
                token.write(creds.to_json())
        service = build("sheets", "v4", credentials=creds)
        self.sheet = service.spreadsheets()

    def read_data(self):
        logger.debug("read_data():")
        result = (
            self.sheet.values()
            .get(spreadsheetId=SPREADSHEET_ID, range="Data!M2:P")
            .execute()
        )
        array = np.array(result.get("values", []))
        wvalues = np.array([[x, y] for x, y, _, _ in array], dtype=float)
        fvalues = np.array(
            [[x, y] for x, _, y, _ in array if y != ""], dtype=float
        )
        return wvalues[:, 0], wvalues[:, 1], fvalues[:, 0], fvalues[:, 1]

    def write_data(self, data):
        logger.debug(f"write_data(): {data=}")
        values = [
            [convert_excel(key), val[0], val[1]] for key, val in data.items()
        ]
        logger.debug(f"write_data(): {values=}")
        self.sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range="Data!A2",
            valueInputOption="RAW",
            body={"values": values},
        ).execute()

    def compute_parameters(self, data_type, x, y):
        logger.debug(f"compute_parameters(): {data_type=}")
        logger.debug(f"compute_parameters(): {x=}")
        logger.debug(f"compute_parameters(): {y=}")
        if data_type == DataType.WEIGHT:
            winf = (
                (1800 - 88.362 + 5.677 * 43 - 4.799 * 76 * 2.54) / 13.397 * 2.2
            )
        elif data_type == DataType.BODYFAT:
            winf = 0
        logger.debug(f"compute_parameters(): {winf=}")
        sol, _ = curve_fit(
            partial(f, winf=winf), x, y, bounds=BOUNDS, max_nfev=10000
        )
        logger.debug(f"compute_parameters(): {sol=}")
        err = e(x, y, sol, winf=winf)
        logger.debug(f"compute_parameters(): {err=}")
        return winf, sol, err

    def write_parameters(self, wwinf, wsol, fwinf, fsol):
        logger.debug(f"write_parameters(): {wwinf=}")
        logger.debug(f"write_parameters(): {wsol=}")
        logger.debug(f"write_parameters(): {fwinf=}")
        logger.debug(f"write_parameters(): {fsol=}")
        wvalues = [wwinf] + list(wsol)
        fvalues = [fwinf] + list(fsol)
        values = np.array([wvalues, fvalues]).T.tolist()
        logger.debug(f"write_parameters(): {values=}")
        self.sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range="Data!S2",
            valueInputOption="RAW",
            body={"values": values},
        ).execute()


def sync(data):
    iface = ExcelInterface()
    iface.sync(data)
