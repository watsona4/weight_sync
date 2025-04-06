import socket

SOCKET_PATH = "/tmp/unix_socket_example.sock"


def client():
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client_socket.connect(SOCKET_PATH)
        print(f"Connected to {SOCKET_PATH}")

        message = "Hello, Unix sockets!"
        print(f"Sending: {message}")
        client_socket.sendall(message.encode())

        data = client_socket.recv(1024)
        print(f"Received: {data.decode()}")
    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        client_socket.close()


if __name__ == "__main__":
    client()