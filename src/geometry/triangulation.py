"""
Triangulation utilities: recover 3D points from corresponding 2D
observations in two calibrated cameras.
"""

import numpy as np
import cv2


def triangulate_points(P1, P2, points1, points2):
    """Triangulate corresponding Nx2 observations into Nx3 points.

    Args:
        P1: (3, 4) projection matrix of camera 1.
        P2: (3, 4) projection matrix of camera 2.
        points1: (N, 2) pixel coordinates observed in camera 1.
        points2: (N, 2) pixel coordinates observed in camera 2.

    Returns:
        (N, 3) array of triangulated world points.
    """
    points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)

    # cv2.triangulatePoints expects points as (2, N) arrays
    points1_t = points1.T  # (2, N)
    points2_t = points2.T  # (2, N)

    points_4d = cv2.triangulatePoints(P1, P2, points1_t, points2_t)  # (4, N)

    # Convert from homogeneous to Euclidean coordinates
    points_3d = (points_4d[:3] / points_4d[3]).T  # (N, 3)

    return points_3d
