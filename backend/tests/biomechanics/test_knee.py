import pytest

from biomechanics.knee import calculate_knee_flexion


class Landmark:
    def __init__(self, x, y, z, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_fully_extended_knee_is_zero_degrees():
    """
    Hip -> Knee -> Ankle are perfectly straight.

    Expected anatomical knee flexion = 0°.
    """

    hip = Landmark(-1.0, 0.0, 0.0)
    knee = Landmark(0.0, 0.0, 0.0)
    ankle = Landmark(1.0, 0.0, 0.0)

    angle = calculate_knee_flexion(
        hip,
        knee,
        ankle,
    )

    assert angle == pytest.approx(0.0)


def test_right_angle_knee_is_90_degrees():
    """
    Hip -> Knee -> Ankle form a 90° geometric angle.

    Expected anatomical knee flexion = 90°.
    """

    hip = Landmark(1.0, 0.0, 0.0)
    knee = Landmark(0.0, 0.0, 0.0)
    ankle = Landmark(0.0, 1.0, 0.0)

    angle = calculate_knee_flexion(
        hip,
        knee,
        ankle,
    )

    assert angle == pytest.approx(90.0)


def test_45_degree_knee_flexion():
    """
    Hip -> Knee -> Ankle form a 135° geometric angle.

    Expected anatomical knee flexion = 45°.
    """

    hip = Landmark(-1.0, 0.0, 0.0)
    knee = Landmark(0.0, 0.0, 0.0)
    ankle = Landmark(1.0, 1.0, 0.0)

    angle = calculate_knee_flexion(
        hip,
        knee,
        ankle,
    )

    assert angle == pytest.approx(45.0)


def test_knee_returns_none_when_hip_is_not_visible():
    """
    The calculation should be unavailable when
    a required landmark has insufficient visibility.
    """

    hip = Landmark(
        -1.0,
        0.0,
        0.0,
        visibility=0.4,
    )

    knee = Landmark(0.0, 0.0, 0.0)
    ankle = Landmark(1.0, 0.0, 0.0)

    angle = calculate_knee_flexion(
        hip,
        knee,
        ankle,
    )

    assert angle is None


def test_knee_returns_none_when_knee_is_not_visible():
    """
    The calculation should be unavailable when
    the knee landmark is below the visibility threshold.
    """

    hip = Landmark(-1.0, 0.0, 0.0)
    knee = Landmark(
        0.0,
        0.0,
        0.0,
        visibility=0.4,
    )
    ankle = Landmark(1.0, 0.0, 0.0)

    angle = calculate_knee_flexion(
        hip,
        knee,
        ankle,
    )

    assert angle is None


def test_knee_returns_none_when_ankle_is_not_visible():
    """
    The calculation should be unavailable when
    the ankle landmark is below the visibility threshold.
    """

    hip = Landmark(-1.0, 0.0, 0.0)
    knee = Landmark(0.0, 0.0, 0.0)
    ankle = Landmark(
        1.0,
        0.0,
        0.0,
        visibility=0.4,
    )

    angle = calculate_knee_flexion(
        hip,
        knee,
        ankle,
    )

    assert angle is None