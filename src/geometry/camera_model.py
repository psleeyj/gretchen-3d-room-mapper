"""
Camera model utilities.

Coordinate convention (must be followed consistently across the project):
- World point:                P_w  (3,)  or  (N, 3)
- World-to-camera transform:  T_cw = [R_cw | t_cw]
- Projection matrix:          P = K [R_cw | t_cw]
- Camera center in world:     C_w = -R_cw.T @ t_cw
"""

import numpy as np
import cv2


def make_projection_matrix(K, R_world_to_camera, t_world_to_camera):
    """Return the 3x4 projection matrix K [R | t].

    Args:
        K: (3, 3) camera intrinsic matrix.
        R_world_to_camera: (3, 3) rotation matrix, world-to-camera.
        t_world_to_camera: (3,) or (3, 1) translation vector, world-to-camera.

    Returns:
        (3, 4) projection matrix P.
    """
    R = np.asarray(R_world_to_camera, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_to_camera, dtype=np.float64).reshape(3, 1)
    Rt = np.hstack([R, t])  # (3, 4)
    P = K @ Rt
    return P


def camera_center_world(R_world_to_camera, t_world_to_camera):
    """Return the camera center C_w in world coordinates.

    C_w = -R_cw.T @ t_cw
    """
    R = np.asarray(R_world_to_camera, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_to_camera, dtype=np.float64).reshape(3, 1)
    C = -R.T @ t
    return C.flatten()


def project_points(points_world, K, R_world_to_camera, t_world_to_camera):
    """Project Nx3 world points to Nx2 image coordinates.

    Uses cv2.projectPoints under the hood (rotation matrix is converted
    to a Rodrigues vector internally).

    Args:
        points_world: (N, 3) array of world points.
        K: (3, 3) camera intrinsic matrix.
        R_world_to_camera: (3, 3) rotation matrix, world-to-camera.
        t_world_to_camera: (3,) translation vector, world-to-camera.

    Returns:
        (N, 2) array of pixel coordinates.
    """
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_world_to_camera, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_to_camera, dtype=np.float64).reshape(3, 1)

    rvec, _ = cv2.Rodrigues(R)
    dist_coeffs = np.zeros((4, 1))  # no lens distortion in the synthetic model

    image_points, _ = cv2.projectPoints(points_world, rvec, t, K, dist_coeffs)
    return image_points.reshape(-1, 2)


def points_in_front_of_camera(points_world, R_world_to_camera, t_world_to_camera):
    """Return a boolean mask: True where the point has positive depth (Z_cam > 0)."""
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_world_to_camera, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_to_camera, dtype=np.float64).reshape(3, 1)

    points_cam = (R @ points_world.T + t).T  # (N, 3), camera-frame coordinates
    return points_cam[:, 2] > 0