import cv2
import mediapipe as mp
import threading


class PoseDetector:
    """
    Handles:
    - Continuous webcam capture
    - MediaPipe Pose Landmarker
    - Pose inference

    The webcam runs independently from pose inference.
    Only the latest captured frame is kept, so stale frames
    are dropped instead of building up a queue.

    This class does NOT:
    - Calculate joint angles
    - Draw skeletons
    - Communicate with Electron
    """

    def __init__(
        self,
        model_path,
        camera_index=0,
        width=1280,
        height=720,
    ):
        self.model_path = model_path

        # ----------------------------------------------------
        # Webcam
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam."
            )

        actual_width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        print(
            f"Camera resolution: "
            f"{actual_width} x {actual_height}",
            flush=True,
        )

        # ----------------------------------------------------
        # Latest-frame buffer
        # ----------------------------------------------------

        self.latest_frame = None

        self.frame_lock = threading.Lock()

        self.running = True

        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )

        self.capture_thread.start()

        # ----------------------------------------------------
        # MediaPipe Pose Landmarker
        # ----------------------------------------------------

        BaseOptions = mp.tasks.BaseOptions

        PoseLandmarker = (
            mp.tasks.vision.PoseLandmarker
        )

        PoseLandmarkerOptions = (
            mp.tasks.vision.PoseLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
        )

        self.landmarker = (
            PoseLandmarker.create_from_options(
                self.options
            )
        )

        self.frame_timestamp_ms = 0

        print(
            "Pose detector initialized.",
            flush=True,
        )

    # ========================================================
    # CAMERA THREAD
    # ========================================================

    def _capture_loop(self):
        """
        Continuously capture frames from the webcam.

        Only the newest frame is retained.
        """

        while self.running:

            success, frame = self.cap.read()

            if not success:
                continue

            with self.frame_lock:
                self.latest_frame = frame

    # ========================================================
    # READ + INFERENCE
    # ========================================================

    def read(self):
        """
        Get the latest available webcam frame and
        run MediaPipe pose detection.

        Returns:
            frame, result

        If no frame is currently available, returns:
            None, None
        """

        # ----------------------------------------------------
        # Get latest frame
        # ----------------------------------------------------

        with self.frame_lock:

            if self.latest_frame is None:
                return None, None

            frame = self.latest_frame.copy()

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # Create MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # ----------------------------------------------------
        # Pose inference
        # ----------------------------------------------------

        result = self.landmarker.detect_for_video(
            mp_image,
            self.frame_timestamp_ms,
        )

        self.frame_timestamp_ms += 33

        return frame, result

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self):
        """
        Stop camera capture and release resources.
        """

        self.running = False

        if self.capture_thread.is_alive():
            self.capture_thread.join(
                timeout=1.0
            )

        self.cap.release()

        self.landmarker.close()

        print(
            "Pose detector stopped.",
            flush=True,
        )