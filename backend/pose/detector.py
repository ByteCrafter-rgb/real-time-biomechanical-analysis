import cv2
import mediapipe as mp


class PoseDetector:
    """
    Handles:
    - Webcam capture
    - MediaPipe Pose Landmarker
    - Pose inference

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

        # We intentionally do not force the camera
        # resolution here because doing so caused a
        # significant startup delay on this system.

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

    def read(self):
        """
        Read one frame from the webcam and
        run MediaPipe pose detection.

        Returns:
            frame, result

        frame:
            OpenCV BGR image.

        result:
            MediaPipe pose detection result.
        """

        success, frame = self.cap.read()

        if not success:
            return None, None

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

        # Approximately 30 FPS timestamp.
        self.frame_timestamp_ms += 33

        return frame, result

    def close(self):
        """
        Release webcam and MediaPipe resources.
        """

        self.cap.release()
        self.landmarker.close()

        print(
            "Pose detector stopped.",
            flush=True,
        )