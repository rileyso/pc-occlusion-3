"""Run: python -m dairv2x.test_visibility_frame"""

import numpy as np

from dairv2x.tools.visibility_frame import solid_angle_visibility


def test_solid_angle_visibility():
    origin = np.zeros(3)
    rear = [10, 0, 0, 2, 2, 2, 0]
    front = [5, 0, 0, 2, 2, 2, 0]
    half_front = [5, 1, 0, 2, 2, 2, 0]
    np.testing.assert_allclose(solid_angle_visibility([rear], origin), [1])
    np.testing.assert_allclose(solid_angle_visibility([front, rear], origin), [1, 0])
    np.testing.assert_allclose(solid_angle_visibility([rear, front], origin), [0, 1])
    boxes = np.array([half_front, rear, half_front], dtype=float)
    # Two identical foreground silhouettes must occlude half, not all, of rear.
    np.testing.assert_allclose(solid_angle_visibility(boxes, origin), [1, 0.5, 1])
    np.testing.assert_allclose(
        solid_angle_visibility([front, [10, 8, 0, 2, 2, 2, 0]], origin), [1, 1])
    # Depth is per ray: a long truck whose centre is farther than the target
    # still hides it, and the nearer-centred target must not hide the truck.
    target, truck = [9, 3.2, 0, 1, 1, 1, 0], [10, 1.8, 0, 14, 1, 1, 0]
    np.testing.assert_allclose(solid_angle_visibility([target, truck], origin), [0, 1])
    # The paper's literal centre ordering gets this case wrong.
    assert solid_angle_visibility([target, truck], origin, ordering="centre")[0] == 1
    # An occluder above the target is not a BEV occluder in this 3D metric.
    np.testing.assert_allclose(
        solid_angle_visibility([[5, 0, 5, 2, 2, 2, 0], rear], origin), [1, 1])
    # Rotating/translating the scene must not change solid angle, including
    # across the +/-pi azimuth seam.
    angle = 3.1
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    shift = np.array([12, -7, 3])
    transformed = boxes.copy()
    transformed[:, :3] = boxes[:, :3] @ rotation.T + shift
    transformed[:, 6] += angle
    np.testing.assert_allclose(solid_angle_visibility(transformed, shift), [1, 0.5, 1])
    scores = solid_angle_visibility([[0, 0, 0, 2, 2, 2, 0], rear], origin)
    assert np.isnan(scores[0]) and scores[1] == 1
    assert solid_angle_visibility(np.empty((0, 7)), origin).size == 0
    try:
        solid_angle_visibility([[1, 0, 0, -1, 2, 2, 0]], origin)
    except ValueError:
        pass
    else:
        raise AssertionError("negative dimensions accepted")


if __name__ == "__main__":
    test_solid_angle_visibility()
    print("Solid-angle visibility checks passed")
