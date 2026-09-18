import pytest

from biomechanics.shoulder import (
    calculate_shoulder_flexion,
    calculate_shoulder_abduction,
)


class Landmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_shoulder_flexion_neutral_is_zero():
    hip = Landmark(0.0, 1.0)
    shoulder = Landmark(0.0, 0.0)
    elbow = Landmark(0.0, -1.0)

    angle = calculate_shoulder_flexion(hip, shoulder, elbow)

    assert angle == pytest.approx(0.0)


def test_shoulder_flexion_90_degrees():
    hip = Landmark(0.0, 1.0)
    shoulder = Landmark(0.0, 0.0)
    elbow = Landmark(1.0, 0.0)

    angle = calculate_shoulder_flexion(hip, shoulder, elbow)

    assert angle == pytest.approx(90.0)


def test_shoulder_abduction_neutral_is_zero():
    shoulder = Landmark(0.0, 0.0)
    elbow = Landmark(0.0, -1.0)
    hip = Landmark(0.0, 1.0)

    angle = calculate_shoulder_abduction(shoulder, elbow, hip)

    assert angle == pytest.approx(0.0)


def test_shoulder_abduction_90_degrees():
    shoulder = Landmark(0.0, 0.0)
    elbow = Landmark(1.0, 0.0)
    hip = Landmark(0.0, 1.0)

    angle = calculate_shoulder_abduction(shoulder, elbow, hip)

    assert angle == pytest.approx(90.0)