import logging
import os
import sys
import time

import requests

URL: str = "https://home.battenkillwoodworks.com/sync/auth"
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig()
LOG: logging.Logger = logging.getLogger("poller")
LOG.setLevel(getattr(logging, LOG_LEVEL.upper()))


def poll():
    LOG.info("Polling...")
    response = requests.get(URL)
    if not response.ok:
        LOG.error("Error: %s", response.reason)


def main() -> int:
    LOG.info("Starting...")
    time.sleep(10)
    while True:
        try:
            poll()
        except Exception:
            LOG.exception("Exception in poll():")
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
