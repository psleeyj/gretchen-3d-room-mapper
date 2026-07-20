"""
Unit tests for src/geometry/confidence.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from geometry.camera_model import make_projection_matrix, project_points, camera_center_world
from geometry.triangulation import triangulate_points
from geometry.confidence import (
    calculate_reprojection_error,
    calculate_parallax_angles,
    confidence_mask,
    cheirality_mask,
)

K = np.array([
    [570.0,   0.0, 319.5],
    [  0.0, 570.0, 239.5],
    [  0.0,   0.0,   1.0],
])


def test_reprojection_error_is_near_zero_for_perfect_observations():
    R_A = np.eye(3)
    t_A = np.zeros(3)
    R_B = np.eye(3)
    t_B = np.array([0.05, 0.0, 0.0])

    point_world = np.array([[0.1, 0.05, 1.0]])

    P1 = make_projection_matrix(K, R_A, t_A)
    P2 = make_projection_matrix(K, R_B, t_B)

    pixel1 = project_points(point_world, K, R_A, t_A)
    pixel2 = project_points(point_world, K, R_B, t_B)

    reconstructed = triangulate_points(P1, P2, pixel1, pixel2)

    err1 = calculate_reprojection_error(reconstructed, pixel1, P1)
    err2 = calculate_reprojection_error(reconstructed, pixel2, P2)

    assert err1[0] < 1e-6
    assert err2[0] < 1e-6


def test_parallax_angle_zero_baseline_is_zero():
    """If both cameras are at the same position, the parallax angle should
    be (numerically) zero."""
    R = np.eye(3)
    t = np.zeros(3)

    point = np.array([[0.1, 0.0, 1.0]])
    C1 = camera_center_world(R, t)
    C2 = camera_center_world(R, t)

    angles = calculate_parallax_angles(point, C1, C2)
    assert np.allclose(angles, 0.0, atol=1e-6)


def test_parallax_angle_increases_with_baseline():
    R = np.eye(3)
    t_A = np.zeros(3)

    point = np.array([[0.0, 0.0, 1.0]])
    C_A = camera_center_world(R, t_A)

    t_small = -R @ np.array([0.01, 0.0, 0.0])
    t_large = -R @ np.array([0.10, 0.0, 0.0])

    C_small = camera_center_world(R, t_small)
    C_large = camera_center_world(R, t_large)

    angle_small = calculate_parallax_angles(point, C_A, C_small)[0]
    angle_large = calculate_parallax_angles(point, C_A, C_large)[0]

    assert angle_large > angle_small


def test_confidence_mask_rejects_low_parallax():
    points_3d = np.array([[0.0, 0.0, 1.0]])
    reprojection_errors = np.array([0.1])
    parallax_angles = np.array([0.1])  # below the 1.0 deg default threshold

    mask = confidence_mask(points_3d, reprojection_errors, parallax_angles)
    assert mask[0] == False


def test_confidence_mask_accepts_good_point():
    points_3d = np.array([[0.0, 0.0, 1.0]])
    reprojection_errors = np.array([0.1])
    parallax_angles = np.array([5.0])

    mask = confidence_mask(points_3d, reprojection_errors, parallax_angles)
    assert mask[0] == True


def test_confidence_mask_rejects_high_reprojection_error():
    points_3d = np.array([[0.0, 0.0, 1.0]])
    reprojection_errors = np.array([10.0])  # above the 2.0 px default threshold
    parallax_angles = np.array([5.0])

    mask = confidence_mask(points_3d, reprojection_errors, parallax_angles)
    assert mask[0] == False


def test_confidence_mask_rejects_out_of_range_depth():
    points_3d = np.array([[0.0, 0.0, 10.0]])  # beyond the 5.0 m default max depth
    reprojection_errors = np.array([0.1])
    parallax_angles = np.array([5.0])

    mask = confidence_mask(points_3d, reprojection_errors, parallax_angles)
    assert mask[0] == False


def test_cheirality_mask_rejects_point_behind_camera():
    R = np.eye(3)
    t = np.zeros(3)

    points = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    mask = cheirality_mask(points, R, t)
    assert mask.tolist() == [True, False]


if __name__ == "__main__":
    test_reprojection_error_is_near_zero_for_perfect_observations()
    test_parallax_angle_zero_baseline_is_zero()
    test_parallax_angle_increases_with_baseline()
    test_confidence_mask_rejects_low_parallax()
    test_confidence_mask_accepts_good_point()
    test_confidence_mask_rejects_high_reprojection_error()
    test_cheirality_mask_rejects_point_behind_camera()
    print("All test_confidence.py tests passed.")
