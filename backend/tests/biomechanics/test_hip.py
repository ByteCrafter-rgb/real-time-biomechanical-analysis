import pytest

from biomechanics.hip import calculate_hip_flexion


class Landmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_hip_neutral_is_zero_degrees():
    shoulder = Landmark(0.0, -1.0, 0.0)
    hip = Landmark(0.0, 0.0, 0.0)
    knee = Landmark(0.0, 1.0, 0.0)

    angle = calculate_hip_flexion(shoulder, hip, knee)

    assert angle == pytest.approx(0.0)


def test_hip_flexion_90_degrees():
    shoulder = Landmark(0.0, -1.0, 0.0)
    hip = Landmark(0.0, 0.0, 0.0)
    knee = Landmark(1.0, 0.0, 0.0)

    angle = calculate_hip_flexion(shoulder, hip, knee)

    assert angle == pytest.approx(90.0)


def test_hip_returns_none_when_landmark_not_visible():
    shoulder = Landmark(0.0, -1.0, visibility=0.4)
    hip = Landmark(0.0, 0.0)
    knee = Landmark(0.0, 1.0)

    angle = calculate_hip_flexion(shoulder, hip, knee)

    assert angle is None