import math


def calculate_angle(a, b, c):
    """
    Calculate the geometric angle ABC in 3D.

    a, b, c are points represented as:
        (x, y, z)

    Returns:
        Geometric angle in degrees, between 0 and 180.
    """

    # Vector BA
    ba = (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    )

    # Vector BC
    bc = (
        c[0] - b[0],
        c[1] - b[1],
        c[2] - b[2],
    )

    # Dot product
    dot_product = (
        ba[0] * bc[0]
        + ba[1] * bc[1]
        + ba[2] * bc[2]
    )

    # Lengths
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

    # Avoid division by zero
    if length_ba == 0 or length_bc == 0:
        return None

    # Cosine rule
    cosine = dot_product / (
        length_ba * length_bc
    )

    # Protect against floating-point errors
    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    angle = math.degrees(
        math.acos(cosine)
    )

    return angle


def calculate_flexion_angle(a, b, c):
    """
    Convert geometric joint angle into
    anatomical flexion angle.

    Straight joint = 0 degrees.

    Example:
        geometric = 180°
        flexion   = 0°

        geometric = 90°
        flexion   = 90°
    """

    geometric_angle = calculate_angle(
        a,
        b,
        c
    )

    if geometric_angle is None:
        return None

    return 180.0 - geometric_angle


# ============================================================
# ELBOW
# ============================================================

def calculate_elbow_flexion(
    shoulder,
    elbow,
    wrist,
):
    """
    Calculate elbow flexion angle.

    Anatomical convention:
        0°   = fully extended
        90°  = approximately right angle
        180° = theoretical maximum

    The three points are:

        Shoulder
            |
            |
          Elbow
             \
              \
              Wrist
    """

    return calculate_flexion_angle(
        shoulder,
        elbow,
        wrist,
    )


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    print("\n--- Generic flexion tests ---")

    tests = [
        (
            "Straight",
            (0, 1, 0),
            (0, 0, 0),
            (0, -1, 0),
            0,
        ),
        (
            "90 degree bend",
            (0, 1, 0),
            (0, 0, 0),
            (1, 0, 0),
            90,
        ),
        (
            "45 degree bend",
            (0, 1, 0),
            (0, 0, 0),
            (1, -1, 0),
            45,
        ),
    ]

    for name, a, b, c, expected in tests:

        actual = calculate_flexion_angle(
            a,
            b,
            c,
        )

        print(
            f"{name}: "
            f"expected={expected}°, "
            f"actual={actual:.1f}°"
        )

    print("\n--- Elbow tests ---")

    elbow_tests = [
        (
            "Straight elbow",
            (0, 1, 0),     # shoulder
            (0, 0, 0),     # elbow
            (0, -1, 0),    # wrist
            0,
        ),
        (
            "90 degree elbow",
            (0, 1, 0),     # shoulder
            (0, 0, 0),     # elbow
            (1, 0, 0),     # wrist
            90,
        ),
        (
            "45 degree elbow",
            (0, 1, 0),     # shoulder
            (0, 0, 0),     # elbow
            (1, -1, 0),    # wrist
            45,
        ),
    ]

    for name, shoulder, elbow, wrist, expected in elbow_tests:

        actual = calculate_elbow_flexion(
            shoulder,
            elbow,
            wrist,
        )

        print(
            f"{name}: "
            f"expected={expected}°, "
            f"actual={actual:.1f}°"
        )