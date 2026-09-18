import csv
import os
import sys
import time

import cv2
import mediapipe as mp

# Allow imports from backend/
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from biomechanics.knee import calculate_knee_flexion


MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "pose_landmarker_lite.task",
    )
)

VIDEO_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "validation",
        "input",
        "squat.mp4",
    )
)

OUTPUT_CSV = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "validation",
        "output",
        "squat_knee_angles.csv",
    )
)


class Landmark:
    def __init__(self, landmark):
        self.x = landmark.x
        self.y = landmark.y
        self.z = landmark.z
        self.visibility = landmark.visibility


def main():
    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True,
    )

    print("Opening squat video...")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"Video FPS: {fps}")
    print(f"Video frames: {frame_count}")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    print("Loading pose model...")

    landmarker = PoseLandmarker.create_from_options(
        options
    )

    results = []

    frame_index = 0
    total_inference_time = 0.0

    print("Processing video...")

    while True:
        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int(
            frame_index * 1000 / fps
        )

        start_time = time.perf_counter()

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        inference_time = (
            time.perf_counter() - start_time
        )

        total_inference_time += inference_time

        left_knee_angle = None
        right_knee_angle = None

        if result.pose_world_landmarks:
            landmarks = result.pose_world_landmarks[0]

            left_hip = Landmark(landmarks[23])
            left_knee = Landmark(landmarks[25])
            left_ankle = Landmark(landmarks[27])

            right_hip = Landmark(landmarks[24])
            right_knee = Landmark(landmarks[26])
            right_ankle = Landmark(landmarks[28])

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

        results.append(
            {
                "frame": frame_index,
                "time_seconds": frame_index / fps,
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle,
                "inference_ms": inference_time * 1000,
            }
        )

        frame_index += 1

        if frame_index % 30 == 0:
            print(
                f"Processed {frame_index}/{frame_count} frames"
            )

    cap.release()
    landmarker.close()

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "time_seconds",
                "left_knee",
                "right_knee",
                "inference_ms",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    processed_frames = len(results)

    average_inference_ms = (
        total_inference_time
        / processed_frames
        * 1000
        if processed_frames
        else 0
    )

    print()
    print("Validation complete.")
    print(f"Frames processed: {processed_frames}")
    print(
        f"Average inference: "
        f"{average_inference_ms:.2f} ms"
    )
    print(f"CSV saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()