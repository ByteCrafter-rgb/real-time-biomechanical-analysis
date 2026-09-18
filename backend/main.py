import asyncio
import cv2
import time

from pose.detector import PoseDetector
from pose.skeleton import draw_skeleton

from biomechanics.elbow import calculate_elbow_flexion
from biomechanics.knee import calculate_knee_flexion
from biomechanics.shoulder import (
    calculate_shoulder_flexion,
    calculate_shoulder_abduction,
)
from biomechanics.hip import calculate_hip_flexion
from biomechanics.ankle import calculate_ankle_flexion
from biomechanics.filter import ExponentialMovingAverage
from communication.websocket_server import WebSocketServer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "../models/pose_landmarker_full.task"


# ============================================================
# FRAME ENCODING
# ============================================================

def encode_frame(frame):
    """
    Convert an OpenCV frame into JPEG bytes
    for sending to Electron.
    """

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            70,
        ],
    )

    if not success:
        return None

    return encoded.tobytes()


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_analysis(detector, websocket):
    """
    Main real-time biomechanics pipeline.

    Webcam
        ↓
    Pose detection
        ↓
    3D landmarks
        ↓
    Joint calculations
        ↓
    Electron
    """

    print(
        "Starting biomechanics analysis...",
        flush=True,
    )

    left_elbow_filter = ExponentialMovingAverage(alpha=0.3)
    right_elbow_filter = ExponentialMovingAverage(alpha=0.3)

    left_knee_filter = ExponentialMovingAverage(alpha=0.3)
    right_knee_filter = ExponentialMovingAverage(alpha=0.3)

    left_shoulder_flexion_filter = ExponentialMovingAverage(alpha=0.3)
    right_shoulder_flexion_filter = ExponentialMovingAverage(alpha=0.3)

    left_shoulder_abduction_filter = ExponentialMovingAverage(alpha=0.3)
    right_shoulder_abduction_filter = ExponentialMovingAverage(alpha=0.3)

    left_hip_filter = ExponentialMovingAverage(alpha=0.3)
    right_hip_filter = ExponentialMovingAverage(alpha=0.3)

    left_ankle_filter = ExponentialMovingAverage(alpha=0.3)
    right_ankle_filter = ExponentialMovingAverage(alpha=0.3)

    while True:

        # ----------------------------------------------------
        # GET FRAME + POSE
        # ----------------------------------------------------

        frame, result = detector.read()

        if frame is None:
            print(
                "Could not read webcam frame.",
                flush=True,
            )
            break

        # ----------------------------------------------------
        # PROCESS POSE
        # ----------------------------------------------------

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            # Draw skeleton
            draw_skeleton(
                frame,
                landmarks,
            )

            # ------------------------------------------------
            # 3D WORLD LANDMARKS
            # ------------------------------------------------

            if result.pose_world_landmarks:

                world = (
                    result.pose_world_landmarks[0]
                )

                # ============================================
                # ELBOW LANDMARKS
                # ============================================

                left_shoulder = world[11]
                left_elbow = world[13]
                left_wrist = world[15]

                right_shoulder = world[12]
                right_elbow = world[14]
                right_wrist = world[16]

                # ============================================
                # KNEE LANDMARKS
                # ============================================

                left_hip = world[23]
                left_knee = world[25]
                left_ankle = world[27]

                right_hip = world[24]
                right_knee = world[26]
                right_ankle = world[28]

                # ============================================
                # ELBOW ANGLES
                # ============================================

                left_elbow_angle = calculate_elbow_flexion(
                    left_shoulder,
                    left_elbow,
                    left_wrist,
                )

                right_elbow_angle = calculate_elbow_flexion(
                    right_shoulder,
                    right_elbow,
                    right_wrist,
                )

                if left_elbow_angle is not None:
                    print(
                        f"Elbow | Left: {left_elbow_angle:.1f}°",
                        flush=True,
                    )

                if right_elbow_angle is not None:
                    print(
                        f"Elbow | Right: {right_elbow_angle:.1f}°",
                        flush=True,
                    )

                # ============================================
                # KNEE ANGLES
                # ============================================

                left_knee_angle = calculate_knee_flexion(
                    left_hip,
                    left_knee,
                    left_ankle,
                )

                right_knee_angle = calculate_knee_flexion(
                    right_hip,
                    right_knee,
                    right_ankle,
                )

                left_shoulder_flexion = calculate_shoulder_flexion(
                    left_hip,
                    left_shoulder,
                    left_elbow,
                )

                right_shoulder_flexion = calculate_shoulder_flexion(
                    right_hip,
                    right_shoulder,
                    right_elbow,
                )

                left_shoulder_abduction = calculate_shoulder_abduction(
                    left_shoulder,
                    left_elbow,
                    left_hip,
                )

                right_shoulder_abduction = calculate_shoulder_abduction(
                    right_shoulder,
                    right_elbow,
                    right_hip,
                )

                left_hip_flexion = calculate_hip_flexion(
                    left_shoulder,
                    left_hip,
                    left_knee,
                )

                right_hip_flexion = calculate_hip_flexion(
                    right_shoulder,
                    right_hip,
                    right_knee,
                )

                left_ankle_flexion = calculate_ankle_flexion(
                    left_knee,
                    left_ankle,
                    world[31],
                )

                right_ankle_flexion = calculate_ankle_flexion(
                    right_knee,
                    right_ankle,
                    world[32],
                )

                # ============================================
                # Apply Filters
                # ============================================
                
                left_elbow_angle = left_elbow_filter.update(
                    left_elbow_angle
                )

                right_elbow_angle = right_elbow_filter.update(
                    right_elbow_angle
                )

                left_knee_angle = left_knee_filter.update(
                    left_knee_angle
                )

                right_knee_angle = right_knee_filter.update(
                    right_knee_angle
                )

                left_shoulder_flexion = (
                    left_shoulder_flexion_filter.update(
                        left_shoulder_flexion
                    )
                )

                right_shoulder_flexion = (
                    right_shoulder_flexion_filter.update(
                        right_shoulder_flexion
                    )
                )

                left_shoulder_abduction = (
                    left_shoulder_abduction_filter.update(
                        left_shoulder_abduction
                    )
                )

                right_shoulder_abduction = (
                    right_shoulder_abduction_filter.update(
                        right_shoulder_abduction
                    )
                )

                left_hip_flexion = left_hip_filter.update(
                    left_hip_flexion
                )

                right_hip_flexion = right_hip_filter.update(
                    right_hip_flexion
                )

                left_ankle_flexion = left_ankle_filter.update(
                    left_ankle_flexion
                )

                right_ankle_flexion = right_ankle_filter.update(
                    right_ankle_flexion
                )

                # ============================================
                # SEND MEASUREMENTS
                # ============================================

                websocket.send_angles(
                    {
                        "left_elbow": left_elbow_angle,
                        "right_elbow": right_elbow_angle,

                        "left_knee": left_knee_angle,
                        "right_knee": right_knee_angle,

                        "left_shoulder_flexion": left_shoulder_flexion,
                        "right_shoulder_flexion": right_shoulder_flexion,

                        "left_shoulder_abduction": left_shoulder_abduction,
                        "right_shoulder_abduction": right_shoulder_abduction,

                        "left_hip_flexion": left_hip_flexion,
                        "right_hip_flexion": right_hip_flexion,

                        "left_ankle_flexion": left_ankle_flexion,
                        "right_ankle_flexion": right_ankle_flexion,
                    }
                )

                # ============================================
                # TERMINAL OUTPUT
                # ============================================

                if (
                    left_elbow_angle is not None
                    and right_elbow_angle is not None
                ):

                    print(
                        f"Elbow | "
                        f"Left: {left_elbow_angle:.1f}° | "
                        f"Right: {right_elbow_angle:.1f}°",
                        flush=True,
                    )

        # ----------------------------------------------------
        # SEND PROCESSED FRAME
        # ----------------------------------------------------

        frame_bytes = encode_frame(frame)

        if frame_bytes is not None:

            websocket.send_frame(
                frame_bytes
            )


# ============================================================
# APPLICATION
# ============================================================

async def main():
    start_time = time.perf_counter()

    print("Starting backend...", flush=True)

    websocket = WebSocketServer(
        host="localhost",
        port=8765,
    )

    print(
        f"WebSocket object created: "
        f"{time.perf_counter() - start_time:.2f}s",
        flush=True,
    )

    detector_start = time.perf_counter()

    detector = PoseDetector(
        MODEL_PATH
    )

    print(
        f"PoseDetector initialized in "
        f"{time.perf_counter() - detector_start:.2f}s",
        flush=True,
    )

    websocket_start = time.perf_counter()

    await websocket.start()

    print(
        f"WebSocket started in "
        f"{time.perf_counter() - websocket_start:.2f}s",
        flush=True,
    )

    print(
        f"Total startup time: "
        f"{time.perf_counter() - start_time:.2f}s",
        flush=True,
    )

    try:
        await asyncio.to_thread(
            run_analysis,
            detector,
            websocket,
        )
    finally:
        detector.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "Backend stopped.",
            flush=True,
        )