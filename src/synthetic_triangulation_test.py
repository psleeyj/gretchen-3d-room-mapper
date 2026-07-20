"""
Synthetic triangulation experiment.

Creates synthetic 3D points, observes them from two known virtual camera
poses, adds optional pixel noise, triangulates using OpenCV, applies
confidence gating, and saves an evaluation table to CSV.

Run:
    python3 synthetic_triangulation_test.py
(from inside src/, or adjust the sys.path insert below)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from geometry.camera_model import (
    make_projection_matrix,
    project_points,
    camera_center_world,
    points_in_front_of_camera,
)
from geometry.triangulation import triangulate_points
from geometry.confidence import (
    calculate_reprojection_error,
    calculate_parallax_angles,
    confidence_mask,
)


# ----------------------------------------------------------------------
# Simulation setup
# ----------------------------------------------------------------------

K = np.array([
    [570.0,   0.0, 319.5],
    [  0.0, 570.0, 239.5],
    [  0.0,   0.0,   1.0],
])

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

DEPTHS_M = [0.5, 0.75, 1.0, 1.5, 2.0]
BASELINES_M = [0.01, 0.02, 0.05, 0.10]
PIXEL_NOISE_STDS = [0.0, 0.25, 0.5, 1.0, 2.0]

RANDOM_SEED = 42


def rotation_from_yaw_deg(yaw_deg):
    """Return a 3x3 rotation matrix for a yaw rotation (about the Y axis), in degrees."""
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [ c, 0.0,   s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,   c],
    ])
    return R


def make_ground_truth_points():
    """Create a small grid of 3D points at several depths, all inside the
    field of view of a camera looking straight down +Z from the origin."""
    points = []
    offsets = [-0.15, -0.05, 0.05, 0.15]
    for Z in DEPTHS_M:
        for dx in offsets:
            for dy in offsets:
                points.append([dx, dy, Z])
    return np.array(points, dtype=np.float64)


def simulate_experiment(baseline_m, noise_std_px, yaw_deg=0.0, rng=None):
    """Run one (baseline, noise) experiment and return a results dict plus
    per-point diagnostics."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    points_world = make_ground_truth_points()

    R_A = np.eye(3)
    t_A = np.zeros(3)

    R_B = rotation_from_yaw_deg(yaw_deg)
    C_B = np.array([baseline_m, 0.0, 0.0])
    t_B = -R_B @ C_B

    P1 = make_projection_matrix(K, R_A, t_A)
    P2 = make_projection_matrix(K, R_B, t_B)

    pixels1 = project_points(points_world, K, R_A, t_A)
    pixels2 = project_points(points_world, K, R_B, t_B)

    in_front_A = points_in_front_of_camera(points_world, R_A, t_A)
    in_front_B = points_in_front_of_camera(points_world, R_B, t_B)

    in_image_A = (
        (pixels1[:, 0] >= 0) & (pixels1[:, 0] < IMAGE_WIDTH) &
        (pixels1[:, 1] >= 0) & (pixels1[:, 1] < IMAGE_HEIGHT)
    )
    in_image_B = (
        (pixels2[:, 0] >= 0) & (pixels2[:, 0] < IMAGE_WIDTH) &
        (pixels2[:, 1] >= 0) & (pixels2[:, 1] < IMAGE_HEIGHT)
    )

    visible = in_front_A & in_front_B & in_image_A & in_image_B

    points_world_v = points_world[visible]
    pixels1_v = pixels1[visible]
    pixels2_v = pixels2[visible]

    if points_world_v.shape[0] == 0:
        return None, None

    noisy1 = pixels1_v + rng.normal(0.0, noise_std_px, size=pixels1_v.shape)
    noisy2 = pixels2_v + rng.normal(0.0, noise_std_px, size=pixels2_v.shape)

    reconstructed = triangulate_points(P1, P2, noisy1, noisy2)

    err1 = calculate_reprojection_error(reconstructed, noisy1, P1)
    err2 = calculate_reprojection_error(reconstructed, noisy2, P2)
    mean_reproj_err = (err1 + err2) / 2.0

    C1 = camera_center_world(R_A, t_A)
    C2 = camera_center_world(R_B, t_B)
    parallax = calculate_parallax_angles(reconstructed, C1, C2)

    mask = confidence_mask(reconstructed, mean_reproj_err, parallax)

    error_3d = np.linalg.norm(reconstructed - points_world_v, axis=1)
    error_depth = np.abs(reconstructed[:, 2] - points_world_v[:, 2])

    per_point = pd.DataFrame({
        "gt_x": points_world_v[:, 0],
        "gt_y": points_world_v[:, 1],
        "gt_z": points_world_v[:, 2],
        "recon_x": reconstructed[:, 0],
        "recon_y": reconstructed[:, 1],
        "recon_z": reconstructed[:, 2],
        "reproj_err_A": err1,
        "reproj_err_B": err2,
        "mean_reproj_err": mean_reproj_err,
        "parallax_deg": parallax,
        "error_3d": error_3d,
        "error_depth": error_depth,
        "accepted": mask,
    })

    n_total = len(mask)
    n_accepted = int(mask.sum())

    summary = {
        "baseline_m": baseline_m,
        "yaw_deg": yaw_deg,
        "pixel_noise_std": noise_std_px,
        "mean_parallax_deg": float(np.mean(parallax)),
        "mean_3d_error_m": float(np.mean(error_3d)),
        "mean_depth_error_m": float(np.mean(error_depth)),
        "mean_3d_error_m_gated": float(np.mean(error_3d[mask])) if n_accepted > 0 else np.nan,
        "mean_depth_error_m_gated": float(np.mean(error_depth[mask])) if n_accepted > 0 else np.nan,
        "retained_points": n_accepted,
        "total_points": n_total,
        "drop_rate": 1.0 - (n_accepted / n_total) if n_total > 0 else np.nan,
    }

    return summary, per_point


def run_all_experiments():
    results = []
    rng = np.random.default_rng(RANDOM_SEED)

    for baseline in BASELINES_M:
        for noise in PIXEL_NOISE_STDS:
            summary, _ = simulate_experiment(baseline, noise, yaw_deg=0.0, rng=rng)
            if summary is not None:
                results.append(summary)

    for noise in PIXEL_NOISE_STDS:
        summary, _ = simulate_experiment(0.05, noise, yaw_deg=8.0, rng=rng)
        if summary is not None:
            results.append(summary)

    df = pd.DataFrame(results)
    return df


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = run_all_experiments()
    out_path = os.path.join(out_dir, "synthetic_triangulation_results.csv")
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved results to: {out_path}")

    zero_noise = df[(df["pixel_noise_std"] == 0.0) & (df["yaw_deg"] == 0.0)]
    print("\n--- Zero-noise check (should have ~0 error) ---")
    print(zero_noise[["baseline_m", "mean_3d_error_m", "mean_depth_error_m"]])

    print("\n--- Effect of baseline (noise fixed at 1.0 px) ---")
    fixed_noise = df[(df["pixel_noise_std"] == 1.0) & (df["yaw_deg"] == 0.0)]
    print(fixed_noise[["baseline_m", "mean_depth_error_m"]].sort_values("baseline_m"))

    print("\n--- Effect of pixel noise (baseline fixed at 0.05 m) ---")
    fixed_baseline = df[(df["baseline_m"] == 0.05) & (df["yaw_deg"] == 0.0)]
    print(fixed_baseline[["pixel_noise_std", "mean_3d_error_m"]].sort_values("pixel_noise_std"))


if __name__ == "__main__":
    main()