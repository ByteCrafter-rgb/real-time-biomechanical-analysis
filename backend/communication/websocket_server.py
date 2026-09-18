import asyncio
import json

import websockets


class WebSocketServer:
    """
    Handles communication between the Python backend
    and the Electron frontend.
    """

    def __init__(
        self,
        host="localhost",
        port=8765,
    ):
        self.host = host
        self.port = port

        self.connected_client = None
        self.event_loop = None
        self.server = None

    async def start(self):
        """
        Start the WebSocket server.
        """

        self.event_loop = asyncio.get_running_loop()

        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )

        print(
            f"WebSocket server running on "
            f"ws://{self.host}:{self.port}",
            flush=True,
        )

    async def _handle_client(self, websocket):
        """
        Handle an Electron client connection.
        """

        self.connected_client = websocket

        print(
            "Electron connected.",
            flush=True,
        )

        try:

            await websocket.send(
                "Python backend connected"
            )

            async for message in websocket:

                print(
                    f"Electron: {message}",
                    flush=True,
                )

        except websockets.exceptions.ConnectionClosed:

            print(
                "Electron disconnected.",
                flush=True,
            )

        except Exception as e:

            print(
                f"WebSocket error: {e}",
                flush=True,
            )

        finally:

            if self.connected_client == websocket:
                self.connected_client = None

    def send_frame(self, frame_bytes):
        """
        Send a processed JPEG frame to Electron.
        """

        if self.connected_client is None:
            return

        if self.event_loop is None:
            return

        try:

            asyncio.run_coroutine_threadsafe(
                self.connected_client.send(
                    frame_bytes
                ),
                self.event_loop,
            )

        except Exception as e:

            print(
                f"Frame send error: {e}",
                flush=True,
            )

    def send_angles(self, angles):
        """
        Send biomechanical measurements to Electron.

        `angles` should be a dictionary.
        """

        if self.connected_client is None:
            return

        if self.event_loop is None:
            return

        data = {
            "type": "angles",
            **angles,
        }

        try:

            asyncio.run_coroutine_threadsafe(
                self.connected_client.send(
                    json.dumps(data)
                ),
                self.event_loop,
            )

        except Exception as e:

            print(
                f"Angle send error: {e}",
                flush=True,
            )

    async def wait_closed(self):
        """
        Keep the WebSocket server alive.
        """

        if self.server is not None:
            await self.server.wait_closed()