"""
Confidence metrics for triangulated points: reprojection error,
parallax angle, and a combined acceptance gate.
"""

import numpy as np


def calculate_reprojection_error(points_3d, points_2d, projection_matrix):
    """Return one reprojection error (in pixels) per point.

    Args:
        points_3d: (N, 3) triangulated world points.
        points_2d: (N, 2) originally observed pixel coordinates.
        projection_matrix: (3, 4) projection matrix K [R | t].

    Returns:
        (N,) array of per-point reprojection errors (Euclidean, pixels).
    """
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)

    N = points_3d.shape[0]
    points_hom = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)

    projected_hom = (projection_matrix @ points_hom.T).T  # (N, 3)
    projected_2d = projected_hom[:, :2] / projected_hom[:, 2:3]

    errors = np.linalg.norm(projected_2d - points_2d, axis=1)
    return errors


def calculate_parallax_angles(points_3d, camera_center_1, camera_center_2):
    """Return one parallax angle in degrees per point.

    theta = arccos( (r1 . r2) / (|r1| |r2|) )
    where r1, r2 are the viewing rays from each camera center to the point.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    C1 = np.asarray(camera_center_1, dtype=np.float64).reshape(1, 3)
    C2 = np.asarray(camera_center_2, dtype=np.float64).reshape(1, 3)

    r1 = points_3d - C1  # (N, 3)
    r2 = points_3d - C2  # (N, 3)

    r1_norm = np.linalg.norm(r1, axis=1)
    r2_norm = np.linalg.norm(r2, axis=1)

    cos_theta = np.sum(r1 * r2, axis=1) / (r1_norm * r2_norm)
    # numerical safety: clip to a valid arccos domain
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def confidence_mask(
    points_3d,
    reprojection_errors,
    parallax_angles,
    minimum_parallax_deg=1.0,
    maximum_reprojection_error_px=2.0,
    minimum_depth_m=0.1,
    maximum_depth_m=5.0,
):
    """Return a Boolean acceptance mask.

    A point is accepted if:
      - its depth (Z) lies within [minimum_depth_m, maximum_depth_m]
      - parallax angle >= minimum_parallax_deg
      - mean reprojection error <= maximum_reprojection_error_px
    """
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    reprojection_errors = np.asarray(reprojection_errors, dtype=np.float64).reshape(-1)
    parallax_angles = np.asarray(parallax_angles, dtype=np.float64).reshape(-1)

    depth = points_3d[:, 2]

    depth_ok = (depth >= minimum_depth_m) & (depth <= maximum_depth_m)
    parallax_ok = parallax_angles >= minimum_parallax_deg
    reprojection_ok = reprojection_errors <= maximum_reprojection_error_px

    mask = depth_ok & parallax_ok & reprojection_ok
    return mask


def cheirality_mask(points_3d, R_world_to_camera, t_world_to_camera):
    """Return a Boolean mask: True where the point is in front of the camera."""
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_world_to_camera, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_to_camera, dtype=np.float64).reshape(3, 1)

    points_cam = (R @ points_3d.T + t).T
    return points_cam[:, 2] > 0