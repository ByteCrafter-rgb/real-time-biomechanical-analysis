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

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "pose_landmarker_lite.task",
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
        + weight
        * (values[upper] - values[lower])
    )


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Video: {width} x {height} @ {fps:.2f} FPS"
    )
    print(f"Frames: {frame_count}")

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

    print("Loading model...")

    landmarker = PoseLandmarker.create_from_options(
        options
    )

    inference_times = []

    frame_index = 0

    print("Benchmarking inference...")

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

        start = time.perf_counter()

        landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        elapsed = (
            time.perf_counter() - start
        )

        inference_times.append(
            elapsed * 1000
        )

        frame_index += 1

    cap.release()
    landmarker.close()

    if not inference_times:
        raise RuntimeError(
            "No frames were processed."
        )

    average = statistics.mean(
        inference_times
    )

    median = statistics.median(
        inference_times
    )

    minimum = min(
        inference_times
    )

    maximum = max(
        inference_times
    )

    p95 = percentile(
        inference_times,
        95,
    )

    print()
    print("==============================")
    print("POSE INFERENCE BENCHMARK")
    print("==============================")
    print(
        f"Frames measured : {len(inference_times)}"
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
        f"Theoretical FPS : {1000 / average:.2f}"
    )
    print("==============================")


if __name__ == "__main__":
    main()