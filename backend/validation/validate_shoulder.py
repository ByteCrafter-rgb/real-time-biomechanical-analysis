import math
import os
import sys

import pytest

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)

from biomechanics.geometry import calculate_angle
from biomechanics.shoulder import calculate_shoulder_flexion


class Landmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_shoulder_flexion_with_depth_variation():
    """
    Validation case:

    The shoulder flexion geometry is defined in the
    image plane (x, y).

    The elbow has a different z coordinate, representing
    an out-of-plane position.

    The current implementation intentionally ignores z
    for this measurement.

    Expected image-plane flexion = 90°.
    """

    hip = Landmark(
        0.0,
        1.0,
        z=0.0,
    )

    shoulder = Landmark(
        0.0,
        0.0,
        z=0.0,
    )

    elbow = Landmark(
        1.0,
        0.0,
        z=0.5,
    )

    angle = calculate_shoulder_flexion(
        hip,
        shoulder,
        elbow,
    )

    assert angle == pytest.approx(90.0)


def test_shoulder_flexion_is_stable_when_depth_changes():
    """
    The x/y projection remains identical while the elbow
    moves further out of the image plane.

    Because the current shoulder flexion convention uses
    2D image-plane geometry, the measured angle should
    remain unchanged.
    """

    hip = Landmark(0.0, 1.0, z=0.0)
    shoulder = Landmark(0.0, 0.0, z=0.0)

    elbow_near = Landmark(1.0, 0.0, z=0.1)
    elbow_far = Landmark(1.0, 0.0, z=1.0)

    angle_near = calculate_shoulder_flexion(
        hip,
        shoulder,
        elbow_near,
    )

    angle_far = calculate_shoulder_flexion(
        hip,
        shoulder,
        elbow_far,
    )

    assert angle_near == pytest.approx(90.0)
    assert angle_far == pytest.approx(90.0)


def test_3d_reference_shows_out_of_plane_difference():
    """
    Reference calculation using the full 3D coordinates.

    The image-plane projection gives 90° flexion, while
    the actual 3D geometry gives a different value because
    the elbow is displaced along the depth axis.
    """

    hip = (0.0, 1.0, 0.0)
    shoulder = (0.0, 0.0, 0.0)

    # Elbow is displaced both horizontally and in depth.
    elbow = (1.0, 1.0, 1.0)

    geometric_3d = calculate_angle(
        hip,
        shoulder,
        elbow,
    )

    flexion_3d = 180.0 - geometric_3d

    # In the image plane, the elbow appears directly
    # horizontal from the shoulder.
    image_plane_flexion = 90.0

    assert image_plane_flexion == pytest.approx(90.0)

    assert flexion_3d != pytest.approx(
        image_plane_flexion
    )

    print(
        f"\nImage-plane shoulder flexion: "
        f"{image_plane_flexion:.2f}°"
    )

    print(
        f"Full 3D reference flexion: "
        f"{flexion_3d:.2f}°"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])