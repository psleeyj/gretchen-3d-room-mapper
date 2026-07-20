"""
Visualize one synthetic triangulation experiment with Open3D.

Shows:
  - Ground-truth points in green
  - Accepted reconstructed points in blue
  - Rejected reconstructed points in red
  - Camera A and Camera B coordinate frames
  - Thin lines connecting each reconstructed point to its ground truth
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d

from geometry.camera_model import camera_center_world
from synthetic_triangulation_test import simulate_experiment, rotation_from_yaw_deg, K


def make_camera_frame(R_world_to_camera, t_world_to_camera, size=0.05):
    """Return an Open3D coordinate frame mesh placed at the camera center,
    oriented like the camera (camera-to-world rotation)."""
    C = camera_center_world(R_world_to_camera, t_world_to_camera)
    R_cw = np.asarray(R_world_to_camera, dtype=np.float64).reshape(3, 3)
    R_wc = R_cw.T  # camera-to-world orientation

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.rotate(R_wc, center=(0, 0, 0))
    frame.translate(C)
    return frame


def build_geometries(baseline_m=0.05, noise_std_px=0.5, yaw_deg=0.0):
    summary, per_point = simulate_experiment(baseline_m, noise_std_px, yaw_deg=yaw_deg)
    if per_point is None:
        raise RuntimeError("No visible points for this configuration.")

    gt_points = per_point[["gt_x", "gt_y", "gt_z"]].to_numpy()
    recon_points = per_point[["recon_x", "recon_y", "recon_z"]].to_numpy()
    accepted = per_point["accepted"].to_numpy()

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    gt_pcd.paint_uniform_color([0.0, 0.8, 0.0])  # green

    recon_accepted = o3d.geometry.PointCloud()
    recon_accepted.points = o3d.utility.Vector3dVector(recon_points[accepted])
    recon_accepted.paint_uniform_color([0.0, 0.2, 1.0])  # blue

    recon_rejected = o3d.geometry.PointCloud()
    recon_rejected.points = o3d.utility.Vector3dVector(recon_points[~accepted])
    recon_rejected.paint_uniform_color([1.0, 0.0, 0.0])  # red

    n = len(gt_points)
    line_points = np.vstack([gt_points, recon_points])
    lines = [[i, i + n] for i in range(n)]
    line_colors = [[0.6, 0.6, 0.6] for _ in range(n)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    R_A = np.eye(3)
    t_A = np.zeros(3)
    R_B = rotation_from_yaw_deg(yaw_deg)
    C_B = np.array([baseline_m, 0.0, 0.0])
    t_B = -R_B @ C_B

    frame_A = make_camera_frame(R_A, t_A)
    frame_B = make_camera_frame(R_B, t_B)

    geometries = [gt_pcd, recon_accepted, recon_rejected, line_set, frame_A, frame_B]
    return geometries, summary


def main():
    geometries, summary = build_geometries(baseline_m=0.05, noise_std_px=0.5, yaw_deg=0.0)
    print("Experiment summary:", summary)
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
