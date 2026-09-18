from .geometry import calculate_flexion_angle


VISIBILITY_THRESHOLD = 0.6


def calculate_elbow_flexion(
    shoulder,
    elbow,
    wrist,
):
    """
    Calculate elbow flexion from 2D image landmarks.

    Points:
        shoulder -> elbow -> wrist

    Convention:
        0°  = fully extended
        90° = approximately right angle

    2D image coordinates are used because
    monocular depth estimation introduced significant
    error for the straight-arm position.
    """

    if (
        shoulder.visibility < VISIBILITY_THRESHOLD
        or elbow.visibility < VISIBILITY_THRESHOLD
        or wrist.visibility < VISIBILITY_THRESHOLD
    ):
        return None

    shoulder_point = (
        shoulder.x,
        shoulder.y,
        0.0,
    )

    elbow_point = (
        elbow.x,
        elbow.y,
        0.0,
    )

    wrist_point = (
        wrist.x,
        wrist.y,
        0.0,
    )

    return calculate_flexion_angle(
        shoulder_point,
        elbow_point,
        wrist_point,
    )