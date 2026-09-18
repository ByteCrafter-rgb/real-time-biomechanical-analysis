import pytest

from biomechanics.geometry import (
    calculate_angle,
    calculate_flexion_angle,
)


def test_straight_joint_is_180_geometric():
    """
    A -> B -> C are perfectly straight.

    Geometric angle = 180°
    """

    a = (-1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (1.0, 0.0, 0.0)

    angle = calculate_angle(a, b, c)

    assert angle == pytest.approx(180.0)


def test_right_angle_is_90_geometric():
    """
    A -> B -> C form a 90° angle.
    """

    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (0.0, 1.0, 0.0)

    angle = calculate_angle(a, b, c)

    assert angle == pytest.approx(90.0)


def test_45_degree_angle():
    """
    A -> B -> C form a 45° angle.
    """

    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (1.0, 1.0, 0.0)

    angle = calculate_angle(a, b, c)

    assert angle == pytest.approx(45.0)


def test_straight_joint_has_zero_flexion():
    """
    Goniometry convention:

    Straight = 0° flexion

    Internally the geometric angle is 180°,
    so anatomical flexion should be 0°.
    """

    a = (-1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (1.0, 0.0, 0.0)

    angle = calculate_flexion_angle(a, b, c)

    assert angle == pytest.approx(0.0)


def test_right_angle_has_90_degree_flexion():
    """
    A 90° geometric joint angle corresponds
    to 90° anatomical flexion.
    """

    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (0.0, 1.0, 0.0)

    angle = calculate_flexion_angle(a, b, c)

    assert angle == pytest.approx(90.0)


def test_zero_length_segment_returns_none():
    """
    If two landmarks occupy the same position,
    the angle cannot be calculated.
    """

    a = (0.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (1.0, 0.0, 0.0)

    angle = calculate_angle(a, b, c)

    assert angle is None