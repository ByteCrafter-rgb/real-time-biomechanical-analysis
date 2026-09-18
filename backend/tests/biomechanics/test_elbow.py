import pytest

from biomechanics.elbow import calculate_elbow_flexion


class Landmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_straight_elbow_is_zero_degrees():
    shoulder = Landmark(-1.0, 0.0)
    elbow = Landmark(0.0, 0.0)
    wrist = Landmark(1.0, 0.0)

    angle = calculate_elbow_flexion(
        shoulder, elbow, wrist
    )

    assert angle == pytest.approx(0.0)


def test_right_angle_elbow_is_90_degrees():
    shoulder = Landmark(1.0, 0.0)
    elbow = Landmark(0.0, 0.0)
    wrist = Landmark(0.0, 1.0)

    angle = calculate_elbow_flexion(
        shoulder, elbow, wrist
    )

    assert angle == pytest.approx(90.0)


def test_45_degree_elbow_flexion():
    shoulder = Landmark(-1.0, 0.0)
    elbow = Landmark(0.0, 0.0)
    wrist = Landmark(1.0, 1.0)

    angle = calculate_elbow_flexion(
        shoulder, elbow, wrist
    )

    assert angle == pytest.approx(45.0)


def test_elbow_returns_none_when_landmark_not_visible():
    shoulder = Landmark(-1.0, 0.0, visibility=0.4)
    elbow = Landmark(0.0, 0.0)
    wrist = Landmark(1.0, 0.0)

    angle = calculate_elbow_flexion(
        shoulder, elbow, wrist
    )

    assert angle is None