import cv2
import mediapipe as mp
import os


# ============================================================
# SETTINGS
# ============================================================

USE_WEBCAM = True

INPUT_VIDEO = "validation/input/squat.mp4"
OUTPUT_VIDEO = "validation/output/squat_skeleton.mp4"
MODEL_PATH = "models/pose_landmarker_full.task"


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def create_landmarker():
    base_options = BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    return PoseLandmarker.create_from_options(options)


# ============================================================
# DRAW SKELETON
# ============================================================

def draw_skeleton(frame, landmarks):
    height, width = frame.shape[:2]

    # MediaPipe Pose connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),

        (9, 10),

        (11, 12),

        (11, 13), (13, 15),
        (15, 17), (15, 19), (15, 21),

        (12, 14), (14, 16),
        (16, 18), (16, 20), (16, 22),

        (11, 23),
        (12, 24),
        (23, 24),

        (23, 25), (25, 27),
        (27, 29), (29, 31),

        (24, 26), (26, 28),
        (28, 30), (30, 32),
    ]

    # Draw bones
    for start_idx, end_idx in connections:

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        start_point = (
            int(start.x * width),
            int(start.y * height)
        )

        end_point = (
            int(end.x * width),
            int(end.y * height)
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    # Draw joints
    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )


# ============================================================
# WEBCAM MODE
# ============================================================

def run_webcam():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started.")
    print("Press Q to quit.")

    with create_landmarker() as landmarker:

        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Webcam timestamps must increase
            timestamp_ms = int(
                frame_number * 1000 / 30
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                draw_skeleton(
                    frame,
                    landmarks
                )

            cv2.imshow(
                "Biomechanical Analysis",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# VIDEO MODE
# ============================================================

def run_video():

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print(f"Could not open video: {INPUT_VIDEO}")
        return

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video size: {width} x {height}")
    print(f"Video FPS: {fps}")

    os.makedirs(
        "validation/output",
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    with create_landmarker() as landmarker:

        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                frame_number * 1000 / fps
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                draw_skeleton(
                    frame,
                    landmarks
                )

            writer.write(frame)

            frame_number += 1

    cap.release()
    writer.release()

    print("Finished!")
    print(f"Output: {OUTPUT_VIDEO}")


# ============================================================
# MAIN
# ============================================================

def main():

    if USE_WEBCAM:
        run_webcam()
    else:
        run_video()


if __name__ == "__main__":
    main()