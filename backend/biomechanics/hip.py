from .geometry import calculate_angle


VISIBILITY_THRESHOLD = 0.6


def calculate_hip_flexion(
    shoulder,
    hip,
    knee,
):
    """
    Calculate hip flexion/extension.

    Points:
        shoulder -> hip -> knee

    Convention:
        0°   = upright / neutral hip
        90°  = approximately 90° hip flexion
        180° = maximum extension in the opposite direction

    Uses 3D world coordinates.
    """

    if (
        shoulder.visibility < VISIBILITY_THRESHOLD
        or hip.visibility < VISIBILITY_THRESHOLD
        or knee.visibility < VISIBILITY_THRESHOLD
    ):
        return None

    shoulder_point = (
        shoulder.x,
        shoulder.y,
        shoulder.z,
    )

    hip_point = (
        hip.x,
        hip.y,
        hip.z,
    )

    knee_point = (
        knee.x,
        knee.y,
        knee.z,
    )

    geometric_angle = calculate_angle(
        shoulder_point,
        hip_point,
        knee_point,
    )

    if geometric_angle is None:
        return None

    return 180.0 - geometric_angle