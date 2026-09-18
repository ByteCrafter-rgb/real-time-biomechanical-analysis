import math

VISIBILITY_THRESHOLD = 0.6


def calculate_ankle_flexion(knee, ankle, foot_index):
    """
    Calculate ankle dorsiflexion / plantarflexion.

    Goniometry convention:
        0°   = neutral ankle position
        +°   = dorsiflexion
        -°   = plantarflexion

    Reference:
        Axis: lateral malleolus
        Stationary arm: lateral aspect of leg
        Moving arm: along the foot

    Required landmarks:
        knee -> ankle -> foot_index

    Uses MediaPipe 3D world coordinates.

    Neutral ankle position is approximately 90° between
    the lower-leg axis and foot axis.

    Assumption:
        Subject is positioned approximately sideways to
        the camera.

    Note:
        This is a pose-estimated biomechanical measurement,
        not a clinically calibrated goniometer measurement.
        Monocular pose estimation and landmark placement can
        introduce residual measurement error.
    """

    if (
        knee.visibility < VISIBILITY_THRESHOLD
        or ankle.visibility < VISIBILITY_THRESHOLD
        or foot_index.visibility < VISIBILITY_THRESHOLD
    ):
        return None

    # Vector from ankle toward knee.
    knee_vector = (
        knee.x - ankle.x,
        knee.y - ankle.y,
        knee.z - ankle.z,
    )

    # Vector from ankle toward foot.
    foot_vector = (
        foot_index.x - ankle.x,
        foot_index.y - ankle.y,
        foot_index.z - ankle.z,
    )

    knee_length = math.sqrt(
        knee_vector[0] ** 2
        + knee_vector[1] ** 2
        + knee_vector[2] ** 2
    )

    foot_length = math.sqrt(
        foot_vector[0] ** 2
        + foot_vector[1] ** 2
        + foot_vector[2] ** 2
    )

    if knee_length == 0 or foot_length == 0:
        return None

    # Geometric angle between the lower-leg and foot vectors.
    dot_product = (
        knee_vector[0] * foot_vector[0]
        + knee_vector[1] * foot_vector[1]
        + knee_vector[2] * foot_vector[2]
    )

    cosine = dot_product / (knee_length * foot_length)
    cosine = max(-1.0, min(1.0, cosine))

    geometric_angle = math.degrees(
        math.acos(cosine)
    )

    # Convert the geometric angle to the anatomical
    # dorsiflexion / plantarflexion convention.
    #
    # Neutral ankle ≈ 90° geometric angle.
    #
    #   geometric < 90° -> dorsiflexion (+)
    #   geometric > 90° -> plantarflexion (-)
    ankle_angle = 90.0 - geometric_angle

    return ankle_angle