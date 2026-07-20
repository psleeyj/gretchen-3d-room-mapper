"""
Unit tests for src/geometry/triangulation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from geometry.camera_model import make_projection_matrix, project_points
from geometry.triangulation import triangulate_points

K = np.array([
    [570.0,   0.0, 319.5],
    [  0.0, 570.0, 239.5],
    [  0.0,   0.0,   1.0],
])


def test_triangulation_exact_with_no_noise():
    """With zero pixel noise, triangulation must recover the ground-truth
    3D point to numerical precision."""
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

    assert np.allclose(reconstructed, point_world, atol=1e-9)


def test_triangulation_multiple_points():
    R_A = np.eye(3)
    t_A = np.zeros(3)
    R_B = np.eye(3)
    t_B = np.array([0.05, 0.0, 0.0])

    points_world = np.array([
        [0.0, 0.0, 0.5],
        [0.1, -0.1, 1.0],
        [-0.2, 0.15, 2.0],
    ])

    P1 = make_projection_matrix(K, R_A, t_A)
    P2 = make_projection_matrix(K, R_B, t_B)

    pixels1 = project_points(points_world, K, R_A, t_A)
    pixels2 = project_points(points_world, K, R_B, t_B)

    reconstructed = triangulate_points(P1, P2, pixels1, pixels2)

    assert reconstructed.shape == points_world.shape
    assert np.allclose(reconstructed, points_world, atol=1e-6)


def test_larger_baseline_reduces_depth_error_under_noise():
    """With fixed pixel noise, a larger baseline should generally produce
    a smaller depth reconstruction error (checked via a single noisy trial
    with a fixed random seed for reproducibility)."""
    rng = np.random.default_rng(0)

    R_A = np.eye(3)
    t_A = np.zeros(3)
    R_B = np.eye(3)

    point_world = np.array([[0.05, 0.05, 1.0]])

    def depth_error_for_baseline(baseline_m, noise_std=1.0):
        t_B = np.array([baseline_m, 0.0, 0.0])
        P1 = make_projection_matrix(K, R_A, t_A)
        P2 = make_projection_matrix(K, R_B, t_B)

        pixel1 = project_points(point_world, K, R_A, t_A)
        pixel2 = project_points(point_world, K, R_B, t_B)

        noisy1 = pixel1 + rng.normal(0, noise_std, size=pixel1.shape)
        noisy2 = pixel2 + rng.normal(0, noise_std, size=pixel2.shape)

        reconstructed = triangulate_points(P1, P2, noisy1, noisy2)
        return abs(reconstructed[0, 2] - point_world[0, 2])

    err_small_baseline = depth_error_for_baseline(0.01)
    err_large_baseline = depth_error_for_baseline(0.10)

    assert err_large_baseline < err_small_baseline


if __name__ == "__main__":
    test_triangulation_exact_with_no_noise()
    test_triangulation_multiple_points()
    test_larger_baseline_reduces_depth_error_under_noise()
    print("All test_triangulation.py tests passed.")
