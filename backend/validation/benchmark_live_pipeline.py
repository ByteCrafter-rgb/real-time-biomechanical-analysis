import os
import statistics
import sys
import time

import cv2


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


def percentile(values, percentile):
    values = sorted(values)

    index = (
        (len(values) - 1)
        * percentile
        / 100
    )

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower])
        * weight
    )


def main():

    detector = PoseDetector(
        MODEL_PATH
    )

    # Same filters used by the application.
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

    frame_times = []

    warmup_frames = 30
    measured_frames = 300

    print()
    print("Starting live pipeline benchmark.")
    print(
        f"Warmup frames: {warmup_frames}"
    )
    print(
        f"Measured frames: {measured_frames}"
    )
    print()

    try:

        total_frames = (
            warmup_frames
            + measured_frames
        )

        for frame_index in range(total_frames):

            start = time.perf_counter()

            frame, result = detector.read()

            if frame is None:
                break

            if result.pose_world_landmarks:

                world = result.pose_world_landmarks[0]

                # ------------------------------------------
                # Required landmarks
                # ------------------------------------------

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

                # ------------------------------------------
                # Calculations
                # ------------------------------------------

                angles = {
                    "left_elbow": calculate_elbow_flexion(
                        left_shoulder,
                        left_elbow,
                        left_wrist,
                    ),

                    "right_elbow": calculate_elbow_flexion(
                        right_shoulder,
                        right_elbow,
                        right_wrist,
                    ),

                    "left_knee": calculate_knee_flexion(
                        left_hip,
                        left_knee,
                        left_ankle,
                    ),

                    "right_knee": calculate_knee_flexion(
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

                # ------------------------------------------
                # EMA filtering
                # ------------------------------------------

                for name, value in angles.items():
                    filters[name].update(value)

                # ------------------------------------------
                # Skeleton drawing
                # ------------------------------------------

                if result.pose_landmarks:
                    draw_skeleton(
                        frame,
                        result.pose_landmarks[0],
                    )

            # ----------------------------------------------
            # JPEG encoding
            # ----------------------------------------------

            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    80,
                ],
            )

            if not success:
                continue

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            # Ignore warmup.
            if frame_index >= warmup_frames:
                frame_times.append(
                    elapsed_ms
                )

            if (
                frame_index % 50 == 0
            ):
                print(
                    f"Processed frame "
                    f"{frame_index}/{total_frames}"
                )

    finally:
        detector.close()

    if not frame_times:
        raise RuntimeError(
            "No benchmark frames collected."
        )

    average = statistics.mean(
        frame_times
    )

    median = statistics.median(
        frame_times
    )

    minimum = min(
        frame_times
    )

    maximum = max(
        frame_times
    )

    p95 = percentile(
        frame_times,
        95,
    )

    print()
    print("==============================")
    print("LIVE PIPELINE BENCHMARK")
    print("==============================")
    print(
        f"Frames measured : {len(frame_times)}"
    )
    print(
        f"Average         : {average:.2f} ms"
    )
    print(
        f"Median          : {median:.2f} ms"
    )
    print(
        f"Minimum         : {minimum:.2f} ms"
    )
    print(
        f"Maximum         : {maximum:.2f} ms"
    )
    print(
        f"P95             : {p95:.2f} ms"
    )
    print(
        f"Measured FPS    : {1000 / average:.2f}"
    )
    print("==============================")


if __name__ == "__main__":
    main()