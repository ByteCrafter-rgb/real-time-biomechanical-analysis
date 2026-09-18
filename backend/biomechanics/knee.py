from .geometry import calculate_flexion_angle


VISIBILITY_THRESHOLD = 0.6


def calculate_knee_flexion(
    hip,
    knee,
    ankle,
):
    """
    Calculate knee flexion angle.

    Points:
        hip -> knee -> ankle

    Convention:
        0°  = fully extended
        90° = approximately right-angle bend

    Uses 3D world coordinates.

    Returns None when any required landmark
    is below the visibility threshold.
    """

    if (
        hip.visibility < VISIBILITY_THRESHOLD
        or knee.visibility < VISIBILITY_THRESHOLD
        or ankle.visibility < VISIBILITY_THRESHOLD
    ):
        return None

    return calculate_flexion_angle(
        (
            hip.x,
            hip.y,
            hip.z,
        ),
        (
            knee.x,
            knee.y,
            knee.z,
        ),
        (
            ankle.x,
            ankle.y,
            ankle.z,
        ),
    )