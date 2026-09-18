from .geometry import calculate_angle


VISIBILITY_THRESHOLD = 0.6


def calculate_ankle_flexion(
    knee,
    ankle,
    foot_index,
):
    """
    Calculate ankle plantarflexion/dorsiflexion.

    Points:
        knee -> ankle -> foot_index

    The ankle is the vertex.

    Convention:
        0°  = neutral ankle position
        Positive values represent dorsiflexion
        Negative values represent plantarflexion.

    Uses 3D world coordinates.
    """

    if (
        knee.visibility < VISIBILITY_THRESHOLD
        or ankle.visibility < VISIBILITY_THRESHOLD
        or foot_index.visibility < VISIBILITY_THRESHOLD
    ):
        return None

    knee_point = (
        knee.x,
        knee.y,
        knee.z,
    )

    ankle_point = (
        ankle.x,
        ankle.y,
        ankle.z,
    )

    foot_point = (
        foot_index.x,
        foot_index.y,
        foot_index.z,
    )

    geometric_angle = calculate_angle(
        knee_point,
        ankle_point,
        foot_point,
    )

    if geometric_angle is None:
        return None

    return 180.0 - geometric_angle