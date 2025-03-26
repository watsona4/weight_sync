import logging
from enum import IntEnum
from functools import partial

import numpy as np
import pandas as pd
from google.auth.transport.requests import Request  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore
from googleapiclient.discovery import build  # type: ignore
from numpy.typing import NDArray
from scipy.optimize import curve_fit  # type: ignore

logging.basicConfig(level=logging.INFO)
LOG: logging.Logger = logging.getLogger("excel_interface")

SCOPES: list[str] = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID: str = "1JNA2ZuNQuX2TNNFF_vuLC4pIwzz_upBPfuAk1qrCqx4"

BOUNDS: tuple[tuple[float, ...], tuple[float, ...]] = (
    (-np.inf, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0),
    (np.inf, np.inf, np.inf, np.inf, 7.0, np.inf, 1.0),
)


class DataType(IntEnum):
    WEIGHT = 1
    BODYFAT = 2


def f(
    t: NDArray[np.float64],
    w0: float,
    Δ0: float,
    c4: float,
    Δp1: float,
    τ1: float,
    Δp2: float,
    τ2: float,
    winf: float,
) -> NDArray[np.float64]:
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


def e(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sol: NDArray[np.float64],
    winf: float | None = None,
) -> float:
    fpart = f(x, *sol, winf) if winf is not None else f(x, *sol)  # type: ignore
    return 1 - np.sum((fpart - y) ** 2) / np.sum(np.array(y) ** 2)


def convert_excel(period: pd.Period) -> float:
    datetime_obj = period.to_timestamp()
    excel_epoch = pd.Timestamp("1900-01-01")
    delta = datetime_obj - excel_epoch
    excel_serial_number = delta.days + delta.seconds / (24 * 60 * 60) + 2
    return excel_serial_number


TOKEN: str = "/data/token.json"
CREDENTIALS: str = "/data/credentials.json"


class ExcelInterface:

    def __init__(self):
        self.sheet = None

    @classmethod
    def sync(cls, data: dict[pd.Period, list[float | None]]):
        LOG.debug(f"sync(): {len(data)=}")
        LOG.debug(f"sync(): {data=}")
        iface = cls()
        iface.auth()
        iface.write_data(data)
        xw, yw, xf, yf = iface.read_data()
        wwinf, wsol, _ = iface.compute_parameters(DataType.WEIGHT, xw, yw)
        fwinf, fsol, _ = iface.compute_parameters(DataType.BODYFAT, xf, yf)
        iface.write_parameters(wwinf, wsol, fwinf, fsol)

    def auth(self):
        LOG.debug("auth():")
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)
        if not creds.valid:
            creds.refresh(Request())
        service = build("sheets", "v4", credentials=creds)
        self.sheet = service.spreadsheets()

    def read_data(self) -> tuple[NDArray[np.float64], ...]:
        LOG.debug("read_data():")
        result = self.sheet.values().get(spreadsheetId=SPREADSHEET_ID, range="Data!M2:P").execute()
        array = np.array(result.get("values", []))
        array[array == ""] = 0.0
        wvalues = np.array([[x, y] for x, y, _, _ in array], dtype=float)
        fvalues = np.array([[x, y] for x, _, y, _ in array], dtype=float)
        return wvalues[:, 0], wvalues[:, 1], fvalues[:, 0], fvalues[:, 1]

    def write_data(self, data: dict[pd.Period, list[float | None]]):
        LOG.debug(f"write_data(): {data=}")
        values = [[convert_excel(key), val[0], val[1]] for key, val in data.items()]
        LOG.debug(f"write_data(): {values=}")
        self.sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range="Data!A2",
            valueInputOption="RAW",
            body={"values": values},
        ).execute()

    def compute_parameters(
        self, data_type: DataType, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64], float]:
        LOG.debug(f"compute_parameters(): {data_type=}")
        LOG.debug(f"compute_parameters(): {x=}")
        LOG.debug(f"compute_parameters(): {y=}")
        if data_type == DataType.WEIGHT:
            winf = (1800 - 88.362 + 5.677 * 43 - 4.799 * 76 * 2.54) / 13.397 * 2.2
        elif data_type == DataType.BODYFAT:
            winf = 0
        LOG.debug(f"compute_parameters(): {winf=}")
        try:
            sol, _ = curve_fit(partial(f, winf=winf), x, y, bounds=BOUNDS, max_nfev=10000)
            LOG.debug(f"compute_parameters(): {sol=}")
        except Exception:
            LOG.exception("Error in compute_parameters:")
            return winf, np.zeros(7), 0
        err = e(x, y, sol, winf=winf)
        LOG.debug(f"compute_parameters(): {err=}")
        return winf, sol, err

    def write_parameters(
        self, wwinf: float, wsol: NDArray[np.float64], fwinf: float, fsol: NDArray[np.float64]
    ):
        LOG.debug(f"write_parameters(): {wwinf=}")
        LOG.debug(f"write_parameters(): {wsol=}")
        LOG.debug(f"write_parameters(): {fwinf=}")
        LOG.debug(f"write_parameters(): {fsol=}")
        wvalues = [wwinf] + list(wsol)
        fvalues = [fwinf] + list(fsol)
        values = np.array([wvalues, fvalues]).T.tolist()
        LOG.debug(f"write_parameters(): {values=}")
        self.sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range="Data!S2",
            valueInputOption="RAW",
            body={"values": values},
        ).execute()
