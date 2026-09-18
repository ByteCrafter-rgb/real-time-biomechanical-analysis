import asyncio
import json

import websockets


class WebSocketServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port

        self.connected_client = None
        self.event_loop = None
        self.server = None

        # Latest-frame buffer.
        #
        # We intentionally keep only the newest frame.
        # For real-time video, an old frame is less useful
        # than dropping it and displaying the latest frame.
        self.latest_frame = None
        self.frame_sender_running = False

    async def start(self):
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

            self.latest_frame = None
            self.frame_sender_running = False

    def send_frame(self, frame_bytes):
        """
        Queue only the newest frame.

        This method is called from the synchronous
        processing loop, potentially outside the
        asyncio event-loop thread.
        """

        if self.connected_client is None:
            return

        if self.event_loop is None:
            return

        # Schedule the frame update on the WebSocket
        # event-loop thread.
        self.event_loop.call_soon_threadsafe(
            self._queue_latest_frame,
            frame_bytes,
        )

    def _queue_latest_frame(self, frame_bytes):
        """
        Runs inside the asyncio event loop.

        Replace any previously queued frame.
        """

        self.latest_frame = frame_bytes

        if not self.frame_sender_running:
            self.frame_sender_running = True

            asyncio.create_task(
                self._send_latest_frames()
            )

    async def _send_latest_frames(self):
        """
        Continuously sends the newest available frame.

        If several frames arrive while a frame is being
        transmitted, older frames are discarded.
        """

        try:

            while (
                self.connected_client is not None
            ):

                # Nothing waiting to send.
                if self.latest_frame is None:
                    break

                frame = self.latest_frame

                # Mark this frame as consumed.
                self.latest_frame = None

                try:
                    await self.connected_client.send(
                        frame
                    )

                except websockets.exceptions.ConnectionClosed:
                    break

                except Exception as e:
                    print(
                        f"Frame send error: {e}",
                        flush=True,
                    )
                    break

        finally:
            self.frame_sender_running = False

            # A new frame could have arrived immediately
            # after the loop checked latest_frame.
            if (
                self.latest_frame is not None
                and self.connected_client is not None
                and not self.frame_sender_running
            ):
                self.frame_sender_running = True

                asyncio.create_task(
                    self._send_latest_frames()
                )

    def send_angles(self, angles):
        """
        Send the latest joint-angle measurements.

        Angle messages are small JSON messages, so they
        do not need the same frame-dropping mechanism.
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
        if self.server is not None:
            await self.server.wait_closed()