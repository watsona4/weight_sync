import logging
import os
import socket
import sys
import time

import requests

URL: str = "https://home.battenkillwoodworks.com/sync/auth"
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig()
LOG: logging.Logger = logging.getLogger("poller")
LOG.setLevel(getattr(logging, LOG_LEVEL.upper()))

SOCKET_PATH = "/tmp/unix_socket_example.sock"

if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)


def server():
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(SOCKET_PATH)
    server_socket.listen(1)
    LOG.info("Server listening on %s", SOCKET_PATH)

    while True:
        conn, addr = server_socket.accept()
        with conn:
            LOG.info("Connected by %s", addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                LOG.info("Received: %s", data.decode())
                conn.sendall(data) # Echo back to client


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
    import threading
    server_thread = threading.Thread(target=server)
    server_thread.daemon = True
    server_thread.start()    
    sys.exit(main())
