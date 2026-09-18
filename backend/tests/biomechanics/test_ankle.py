import pytest
from types import SimpleNamespace

from biomechanics.ankle import calculate_ankle_flexion


def landmark(x, y, z=0.0, visibility=1.0):
    return SimpleNamespace(
        x=x,
        y=y,
        z=z,
        visibility=visibility,
    )


def test_neutral_ankle_is_zero_degrees():
    """
    Neutral ankle position.

    Lower leg and foot form approximately 90°.
    Expected ankle angle = 0°.
    """

    knee = landmark(0.0, -1.0, 0.0)
    ankle = landmark(0.0, 0.0, 0.0)
    foot = landmark(1.0, 0.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle == pytest.approx(0.0)


def test_dorsiflexion_is_positive():
    """
    Foot moves upward toward the shin.

    Expected result: positive ankle angle.
    """

    knee = landmark(0.0, -1.0, 0.0)
    ankle = landmark(0.0, 0.0, 0.0)
    foot = landmark(1.0, -1.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle > 0


def test_plantarflexion_is_negative():
    """
    Foot moves downward away from the shin.

    Expected result: negative ankle angle.
    """

    knee = landmark(0.0, -1.0, 0.0)
    ankle = landmark(0.0, 0.0, 0.0)
    foot = landmark(1.0, 1.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle < 0


def test_20_degree_dorsiflexion():
    """
    Approximately 20° dorsiflexion.
    """

    import math

    knee = landmark(0.0, -1.0, 0.0)
    ankle = landmark(0.0, 0.0, 0.0)

    foot = landmark(
        math.cos(math.radians(20)),
        -math.sin(math.radians(20)),
        0.0,
    )

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle == pytest.approx(20.0, abs=0.5)


def test_50_degree_plantarflexion():
    """
    Approximately 50° plantarflexion.
    """

    import math

    knee = landmark(0.0, -1.0, 0.0)
    ankle = landmark(0.0, 0.0, 0.0)

    foot = landmark(
        math.cos(math.radians(50)),
        math.sin(math.radians(50)),
        0.0,
    )

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle == pytest.approx(-50.0, abs=0.5)


def test_ankle_returns_none_when_landmark_not_visible():
    """
    Measurement should be unavailable when a required
    landmark has low visibility.
    """

    knee = landmark(
        0.0,
        -1.0,
        0.0,
        visibility=0.5,
    )

    ankle = landmark(0.0, 0.0, 0.0)
    foot = landmark(1.0, 0.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle is None


def test_zero_length_knee_vector_returns_none():
    """
    Measurement should be unavailable if knee and ankle
    occupy the same position.
    """

    knee = landmark(0.0, 0.0, 0.0)
    ankle = landmark(0.0, 0.0, 0.0)
    foot = landmark(1.0, 0.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle is None


def test_zero_length_foot_vector_returns_none():
    """
    Measurement should be unavailable if foot and ankle
    occupy the same position.
    """

    knee = landmark(0.0, -1.0, 0.0)
    ankle = landmark(0.0, 0.0, 0.0)
    foot = landmark(0.0, 0.0, 0.0)

    angle = calculate_ankle_flexion(
        knee,
        ankle,
        foot,
    )

    assert angle is None