from .geometry import calculate_angle


VISIBILITY_THRESHOLD = 0.6


def _point(landmark):
    return (
        landmark.x,
        landmark.y,
        landmark.z,
    )


def calculate_shoulder_flexion(
    hip,
    shoulder,
    elbow,
):
    """
    Calculate shoulder flexion/extension.

    Points:
        hip -> shoulder -> elbow

    Convention:
        0°   = arm alongside body
        90°  = arm approximately horizontal
        180° = arm overhead

    The calculation uses the image-plane (x, y)
    geometry.

    This assumes the subject is positioned
    approximately sideways to the camera.
    """

    if (
        hip.visibility < VISIBILITY_THRESHOLD
        or shoulder.visibility < VISIBILITY_THRESHOLD
        or elbow.visibility < VISIBILITY_THRESHOLD
    ):
        return None

    hip_point = (
        hip.x,
        hip.y,
        0.0,
    )

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

    geometric_angle = calculate_angle(
        hip_point,
        shoulder_point,
        elbow_point,
    )

    if geometric_angle is None:
        return None

    return 180.0 - geometric_angle


def calculate_shoulder_abduction(
    shoulder,
    elbow,
    hip,
):
    """
    Calculate shoulder abduction/adduction.

    Points:
        shoulder -> elbow
        shoulder -> hip

    Convention:
        0°   = arm alongside body
        90°  = arm approximately horizontal

    Uses the image-plane (x, y) geometry.

    This measurement assumes the subject is facing
    approximately toward the camera.
    """

    if (
        shoulder.visibility < VISIBILITY_THRESHOLD
        or elbow.visibility < VISIBILITY_THRESHOLD
        or hip.visibility < VISIBILITY_THRESHOLD
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

    hip_point = (
        hip.x,
        hip.y,
        0.0,
    )

    geometric_angle = calculate_angle(
        elbow_point,
        shoulder_point,
        hip_point,
    )

    if geometric_angle is None:
        return None

    return 180.0 - geometric_angle