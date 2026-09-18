import os
import sys

import cv2
import mediapipe as mp


# Allow imports from backend/
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)

from biomechanics.knee import calculate_knee_flexion
from pose.skeleton import draw_skeleton


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

OUTPUT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "validation",
        "output",
        "squat_validation.mp4",
    )
)

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "pose_landmarker_full.task",
    )
)


def main():
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

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

    print(f"Input video: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Frames: {frame_count}")

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

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

    frame_index = 0

    print("Creating validation video...")

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

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        left_knee_angle = None
        right_knee_angle = None

        if result.pose_world_landmarks:
            world = result.pose_world_landmarks[0]

            # Draw skeleton using image landmarks.
            if result.pose_landmarks:
                draw_skeleton(
                    frame,
                    result.pose_landmarks[0],
                )

            class Landmark:
                def __init__(self, landmark):
                    self.x = landmark.x
                    self.y = landmark.y
                    self.z = landmark.z
                    self.visibility = landmark.visibility

            # Left leg
            left_hip = Landmark(world[23])
            left_knee = Landmark(world[25])
            left_ankle = Landmark(world[27])

            # Right leg
            right_hip = Landmark(world[24])
            right_knee = Landmark(world[26])
            right_ankle = Landmark(world[28])

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

        # --------------------------------------------------
        # Draw angle information
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (20, 20),
            (470, 145),
            (0, 0, 0),
            -1,
        )

        left_text = (
            f"Left Knee: "
            f"{left_knee_angle:.1f} deg"
            if left_knee_angle is not None
            else "Left Knee: N/A"
        )

        right_text = (
            f"Right Knee: "
            f"{right_knee_angle:.1f} deg"
            if right_knee_angle is not None
            else "Right Knee: N/A"
        )

        cv2.putText(
            frame,
            "KNEE FLEXION",
            (40, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            left_text,
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            right_text,
            (40, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        frame_index += 1

        if frame_index % 30 == 0:
            print(
                f"Processed "
                f"{frame_index}/{frame_count}"
            )

    cap.release()
    writer.release()
    landmarker.close()

    print()
    print("Validation video created.")
    print(f"Saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()