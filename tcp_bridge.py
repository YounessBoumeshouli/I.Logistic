import socket
import asyncio
import websockets
import json
import threading

# Configuration
TCP_HOST = '0.0.0.0'
TCP_PORT = 9999
WS_URI = "ws://fastapi-producer:8000/ws/orders"


def start_tcp_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((TCP_HOST, TCP_PORT))
    server_socket.listen(1)
    print(f"Serveur TCP en écoute sur {TCP_HOST}:{TCP_PORT}...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connexion TCP acceptée de {addr}")
        # Lancer un thread pour gérer le flux WebSocket vers ce client TCP
        threading.Thread(target=run_ws_to_tcp, args=(client_socket,)).start()


def run_ws_to_tcp(client_socket):
    asyncio.run(forward_data(client_socket))


async def forward_data(client_socket):
    try:
        async with websockets.connect(WS_URI) as websocket:
            print("Connecté au WebSocket FastAPI.")
            while True:
                data = await websocket.recv()
                # Spark Streaming attend du texte avec un saut de ligne (\n)
                message = data + "\n"
                try:
                    client_socket.sendall(message.encode('utf-8'))
                except BrokenPipeError:
                    print("Client TCP déconnecté.")
                    break
    except Exception as e:
        print(f"Erreur de pont: {e}")
    finally:
        client_socket.close()


if __name__ == "__main__":
    # Attendre un peu que FastAPI démarre
    import time

    time.sleep(5)
    start_tcp_server()