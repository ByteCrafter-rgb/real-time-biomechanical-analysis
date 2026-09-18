import os
import statistics
import sys
import time

import cv2
import mediapipe as mp


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)

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


MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "pose_landmarker_full.task",
    )
)


def average(values):
    if not values:
        return 0.0

    return statistics.mean(values)


def percentile(values, p):
    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
        / 100
    )

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = index - lower

    return (
        values[lower]
        + fraction
        * (values[upper] - values[lower])
    )


def print_stage(name, values):

    avg = average(values)
    p95 = percentile(values, 95)

    print(
        f"{name:<25}"
        f"avg: {avg:>7.2f} ms    "
        f"p95: {p95:>7.2f} ms"
    )


def main():

    detector = PoseDetector(
        MODEL_PATH
    )

    filters = {
        name: ExponentialMovingAverage(alpha=0.3)
        for name in [
            "left_elbow",
            "right_elbow",
            "left_knee",
            "right_knee",
            "left_shoulder_flexion",
            "right_shoulder_flexion",
            "left_shoulder_abduction",
            "right_shoulder_abduction",
            "left_hip",
            "right_hip",
            "left_ankle",
            "right_ankle",
        ]
    }

    capture_times = []
    inference_times = []
    calculation_times = []
    drawing_times = []
    encoding_times = []
    total_times = []

    warmup = 30
    measurements = 300

    print()
    print(
        "Profiling live pipeline..."
    )
    print(
        f"Warmup: {warmup} frames"
    )
    print(
        f"Measurements: {measurements} frames"
    )
    print()

    try:

        for frame_index in range(
            warmup + measurements
        ):

            total_start = time.perf_counter()

            # ==========================================
            # CAMERA CAPTURE
            # ==========================================

            capture_start = time.perf_counter()

            success, frame = (
                detector.cap.read()
            )

            capture_time = (
                time.perf_counter()
                - capture_start
            ) * 1000

            if not success:
                break

            # ==========================================
            # POSE INFERENCE
            # ==========================================

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = (
                frame_index * 33
            )

            inference_start = (
                time.perf_counter()
            )

            result = (
                detector.landmarker
                .detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            inference_time = (
                time.perf_counter()
                - inference_start
            ) * 1000

            # ==========================================
            # ANGLE CALCULATIONS + EMA
            # ==========================================

            calculation_start = (
                time.perf_counter()
            )

            if result.pose_world_landmarks:

                world = (
                    result.pose_world_landmarks[0]
                )

                left_shoulder = world[11]
                right_shoulder = world[12]

                left_elbow = world[13]
                right_elbow = world[14]

                left_wrist = world[15]
                right_wrist = world[16]

                left_hip = world[23]
                right_hip = world[24]

                left_knee = world[25]
                right_knee = world[26]

                left_ankle = world[27]
                right_ankle = world[28]

                left_foot = world[31]
                right_foot = world[32]

                angles = {
                    "left_elbow":
                        calculate_elbow_flexion(
                            left_shoulder,
                            left_elbow,
                            left_wrist,
                        ),

                    "right_elbow":
                        calculate_elbow_flexion(
                            right_shoulder,
                            right_elbow,
                            right_wrist,
                        ),

                    "left_knee":
                        calculate_knee_flexion(
                            left_hip,
                            left_knee,
                            left_ankle,
                        ),

                    "right_knee":
                        calculate_knee_flexion(
                            right_hip,
                            right_knee,
                            right_ankle,
                        ),

                    "left_shoulder_flexion":
                        calculate_shoulder_flexion(
                            left_hip,
                            left_shoulder,
                            left_elbow,
                        ),

                    "right_shoulder_flexion":
                        calculate_shoulder_flexion(
                            right_hip,
                            right_shoulder,
                            right_elbow,
                        ),

                    "left_shoulder_abduction":
                        calculate_shoulder_abduction(
                            left_shoulder,
                            left_elbow,
                            left_hip,
                        ),

                    "right_shoulder_abduction":
                        calculate_shoulder_abduction(
                            right_shoulder,
                            right_elbow,
                            right_hip,
                        ),

                    "left_hip":
                        calculate_hip_flexion(
                            left_shoulder,
                            left_hip,
                            left_knee,
                        ),

                    "right_hip":
                        calculate_hip_flexion(
                            right_shoulder,
                            right_hip,
                            right_knee,
                        ),

                    "left_ankle":
                        calculate_ankle_flexion(
                            left_knee,
                            left_ankle,
                            left_foot,
                        ),

                    "right_ankle":
                        calculate_ankle_flexion(
                            right_knee,
                            right_ankle,
                            right_foot,
                        ),
                }

                for name, value in angles.items():
                    filters[name].update(value)

            calculation_time = (
                time.perf_counter()
                - calculation_start
            ) * 1000

            # ==========================================
            # SKELETON DRAWING
            # ==========================================

            drawing_start = (
                time.perf_counter()
            )

            if result.pose_landmarks:

                draw_skeleton(
                    frame,
                    result.pose_landmarks[0],
                )

            drawing_time = (
                time.perf_counter()
                - drawing_start
            ) * 1000

            # ==========================================
            # JPEG ENCODING
            # ==========================================

            encoding_start = (
                time.perf_counter()
            )

            cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80,
                ],
            )

            encoding_time = (
                time.perf_counter()
                - encoding_start
            ) * 1000

            total_time = (
                time.perf_counter()
                - total_start
            ) * 1000

            if frame_index >= warmup:

                capture_times.append(
                    capture_time
                )

                inference_times.append(
                    inference_time
                )

                calculation_times.append(
                    calculation_time
                )

                drawing_times.append(
                    drawing_time
                )

                encoding_times.append(
                    encoding_time
                )

                total_times.append(
                    total_time
                )

            if frame_index % 50 == 0:
                print(
                    f"Processed "
                    f"{frame_index}/"
                    f"{warmup + measurements}"
                )

    finally:

        detector.close()

    print()
    print("==============================")
    print("PIPELINE PROFILE")
    print("==============================")

    print_stage(
        "Camera capture",
        capture_times,
    )

    print_stage(
        "Pose inference",
        inference_times,
    )

    print_stage(
        "Angles + EMA",
        calculation_times,
    )

    print_stage(
        "Skeleton drawing",
        drawing_times,
    )

    print_stage(
        "JPEG encoding",
        encoding_times,
    )

    print_stage(
        "TOTAL",
        total_times,
    )

    print("==============================")

    total_average = average(
        total_times
    )

    print(
        f"\nMeasured FPS: "
        f"{1000 / total_average:.2f}"
    )


if __name__ == "__main__":
    main()