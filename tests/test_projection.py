"""
Unit tests for src/geometry/camera_model.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from geometry.camera_model import (
    make_projection_matrix,
    project_points,
    camera_center_world,
    points_in_front_of_camera,
)

K = np.array([
    [570.0,   0.0, 319.5],
    [  0.0, 570.0, 239.5],
    [  0.0,   0.0,   1.0],
])


def test_point_in_front_of_camera_projects_near_image_center():
    """A world point directly in front of Camera A (on the optical axis)
    must project near the image center (cx, cy)."""
    R = np.eye(3)
    t = np.zeros(3)

    point = np.array([[0.0, 0.0, 1.0]])
    pixel = project_points(point, K, R, t)

    assert np.allclose(pixel[0], [319.5, 239.5], atol=1e-6)


def test_projection_matrix_shape():
    R = np.eye(3)
    t = np.zeros(3)
    P = make_projection_matrix(K, R, t)
    assert P.shape == (3, 4)


def test_camera_center_world_at_origin_for_identity_pose():
    R = np.eye(3)
    t = np.zeros(3)
    C = camera_center_world(R, t)
    assert np.allclose(C, [0.0, 0.0, 0.0])


def test_camera_center_world_for_translated_camera():
    """If a camera sits at world position (0.05, 0, 0) with identity
    orientation, then t_cw = -R @ C, and camera_center_world should recover
    C = (0.05, 0, 0)."""
    R = np.eye(3)
    C_true = np.array([0.05, 0.0, 0.0])
    t = -R @ C_true

    C_recovered = camera_center_world(R, t)
    assert np.allclose(C_recovered, C_true)


def test_points_in_front_of_camera():
    R = np.eye(3)
    t = np.zeros(3)

    points = np.array([
        [0.0, 0.0, 1.0],   # in front
        [0.0, 0.0, -1.0],  # behind
    ])
    mask = points_in_front_of_camera(points, R, t)
    assert mask.tolist() == [True, False]


def test_off_axis_point_projects_off_center():
    """Sanity check: a point that is not on the optical axis should not
    project exactly to the image center."""
    R = np.eye(3)
    t = np.zeros(3)

    point = np.array([[0.2, 0.0, 1.0]])
    pixel = project_points(point, K, R, t)

    assert not np.allclose(pixel[0], [319.5, 239.5], atol=1e-3)
    assert pixel[0, 0] > 319.5  # positive X should shift the pixel to the right


if __name__ == "__main__":
    test_point_in_front_of_camera_projects_near_image_center()
    test_projection_matrix_shape()
    test_camera_center_world_at_origin_for_identity_pose()
    test_camera_center_world_for_translated_camera()
    test_points_in_front_of_camera()
    test_off_axis_point_projects_off_center()
    print("All test_projection.py tests passed.")
