"""
Plot GPS Trajectory from KITTI oxts data.

Usage:
    python plot_gps_trajectory.py

Reads all .txt files from the KITTI oxts/data/ directory,
extracts lat/lon (columns 0 & 1), converts to local metric
coordinates, and plots the vehicle trajectory.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

circle = "~/Project_1/Data/Kitti_circle/data"
straight = "~/Project_1/Data/Kitti_straight/data"
urban = "~/Project_1/Data/Kitti_urban/data"
winding = "~/Project_1/Data/Kitti_winding/data"
# ── Configuration ──────────────────────────────────────────────
DATA_DIR = os.path.expanduser(circle)

EARTH_RADIUS = 6_378_137.0  # meters (WGS-84 semi-major axis)

# ── Load oxts files ────────────────────────────────────────────
txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
if not txt_files:
    raise FileNotFoundError(
        f"No .txt files found in {DATA_DIR}\n"
        "Update DATA_DIR to point to your oxts/data/ folder."
    )

lat = np.zeros(len(txt_files))
lon = np.zeros(len(txt_files))

for i, fpath in enumerate(txt_files):
    with open(fpath, "r") as f:
        vals = f.readline().strip().split()
    lat[i] = float(vals[0])  # column 0: latitude  (deg)
    lon[i] = float(vals[1])  # column 1: longitude (deg)

print(f"Loaded {len(txt_files)} frames from {DATA_DIR}")

# ── Convert lat/lon → local East-North (meters) ───────────────
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

# Use first point as the origin
lat0 = lat_rad[0]
lon0 = lon_rad[0]

# Flat-earth approximation (accurate for small areas)
x = EARTH_RADIUS * (lon_rad - lon0) * np.cos(lat0)   # East  (m)
y = EARTH_RADIUS * (lat_rad - lat0)                    # North (m)

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Main trajectory
ax.plot(x, y, color="#2563EB", linewidth=2, label="GPS Trajectory")

# Start / end markers
ax.plot(x[0],  y[0],  "o", color="#16A34A", markersize=12,
        zorder=5, label="Start")
ax.plot(x[-1], y[-1], "s", color="#DC2626", markersize=12,
        zorder=5, label="End")

# Direction arrows every N frames
N = max(1, len(x) // 15)
for i in range(0, len(x) - 1, N):
    dx = x[i + 1] - x[i]
    dy = y[i + 1] - y[i]
    ax.annotate(
        "",
        xy=(x[i] + dx, y[i] + dy),
        xytext=(x[i], y[i]),
        arrowprops=dict(arrowstyle="->", color="#1E40AF", lw=1.5),
    )

ax.set_xlabel("East (m)", fontsize=13)
ax.set_ylabel("North (m)", fontsize=13)
ax.set_title(
    "KITTI GPS Trajectory — Circle",
    fontsize=15, fontweight="bold",
)
ax.legend(fontsize=11, loc="best")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()

# Save and show
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gps_trajectory_circle.png")
fig.savefig(out_path, dpi=150)
print(f"Saved plot → {out_path}")
plt.show()
