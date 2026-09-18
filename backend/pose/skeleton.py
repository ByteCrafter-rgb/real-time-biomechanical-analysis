import cv2


# MediaPipe Pose landmark connections
POSE_CONNECTIONS = [
    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Upper body
    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    # Left arm / hand
    (15, 17),
    (15, 19),
    (15, 21),

    # Right arm / hand
    (16, 18),
    (16, 20),
    (16, 22),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28),

    # Feet
    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32),
]


def draw_skeleton(frame, landmarks):
    """
    Draw the detected pose skeleton
    directly onto the OpenCV frame.
    """

    height, width, _ = frame.shape

    # --------------------------------------------------------
    # Draw connections
    # --------------------------------------------------------

    for start_idx, end_idx in POSE_CONNECTIONS:

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * width)
        y1 = int(start.y * height)

        x2 = int(end.x * width)
        y2 = int(end.y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    # --------------------------------------------------------
    # Draw landmarks
    # --------------------------------------------------------

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