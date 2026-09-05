import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/synthetic_triangulation_results.csv")

# Use one baseline for a clear comparison
subset = df[
    (df["baseline_m"] == 0.02) &
    (df["yaw_deg"] == 0.0)
].sort_values("pixel_noise_std")

plt.figure(figsize=(8, 5))

plt.plot(
    subset["pixel_noise_std"],
    subset["mean_3d_error_m"],
    marker="o",
    label="Before confidence gating"
)

plt.plot(
    subset["pixel_noise_std"],
    subset["mean_3d_error_m_gated"],
    marker="o",
    label="After confidence gating"
)

plt.xlabel("Pixel Noise Standard Deviation")
plt.ylabel("Mean 3D Reconstruction Error (m)")
plt.title("Synthetic Triangulation Robustness")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/synthetic_triangulation_error.png",
    dpi=200
)

print("Saved: outputs/synthetic_triangulation_error.png")
