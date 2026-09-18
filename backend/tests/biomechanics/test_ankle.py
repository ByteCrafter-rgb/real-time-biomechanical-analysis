import pytest

from biomechanics.ankle import calculate_ankle_flexion


class Landmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_neutral_ankle_is_zero_degrees():
    """
    Knee -> Ankle -> Foot are perfectly straight.

    Expected ankle flexion magnitude = 0°.
    """

    knee = Landmark(-1.0, 0.0, 0.0)
    ankle = Landmark(0.0, 0.0, 0.0)
    foot = Landmark(1.0, 0.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle == pytest.approx(0.0)


def test_90_degree_ankle_angle():
    """
    Knee -> Ankle -> Foot form a 90° geometric angle.

    Expected value from the current implementation = 90°.
    """

    knee = Landmark(1.0, 0.0, 0.0)
    ankle = Landmark(0.0, 0.0, 0.0)
    foot = Landmark(0.0, 1.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle == pytest.approx(90.0)


def test_45_degree_ankle_angle():
    """
    Knee -> Ankle -> Foot form a 135° geometric angle.

    Expected value from the current implementation = 45°.
    """

    knee = Landmark(-1.0, 0.0, 0.0)
    ankle = Landmark(0.0, 0.0, 0.0)
    foot = Landmark(1.0, 1.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle == pytest.approx(45.0)


def test_ankle_returns_none_when_landmark_not_visible():
    """
    The calculation should be unavailable when
    a required landmark has insufficient visibility.
    """

    knee = Landmark(
        -1.0,
        0.0,
        0.0,
        visibility=0.4,
    )

    ankle = Landmark(0.0, 0.0, 0.0)
    foot = Landmark(1.0, 0.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle is None