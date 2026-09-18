import math


def calculate_angle(a, b, c):
    """
    Calculate geometric angle ABC in 3D.

    Returns an angle between 0 and 180 degrees.
    """

    ba = (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    )

    bc = (
        c[0] - b[0],
        c[1] - b[1],
        c[2] - b[2],
    )

    dot_product = (
        ba[0] * bc[0]
        + ba[1] * bc[1]
        + ba[2] * bc[2]
    )

    length_ba = math.sqrt(
        ba[0] ** 2
        + ba[1] ** 2
        + ba[2] ** 2
    )

    length_bc = math.sqrt(
        bc[0] ** 2
        + bc[1] ** 2
        + bc[2] ** 2
    )

    if length_ba == 0 or length_bc == 0:
        return None

    cosine = dot_product / (
        length_ba * length_bc
    )

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(
        math.acos(cosine)
    )


def calculate_flexion_angle(a, b, c):
    """
    Convert geometric joint angle to
    anatomical flexion angle.

    Straight = 0 degrees.
    """

    geometric_angle = calculate_angle(
        a,
        b,
        c,
    )

    if geometric_angle is None:
        return None

    return 180.0 - geometric_angle