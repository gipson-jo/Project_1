import os
import glob
import numpy as np
import matplotlib.pyplot as plt

circle = "~/Project_1/Data/Kitti_circle/data"
straight = "~/Project_1/Data/Kitti_straight/data"
urban = "~/Project_1/Data/Kitti_urban/data"
winding = "~/Project_1/Data/Kitti_winding/data"

DATA_DIR = os.path.expanduser(winding)

# ── Load oxts files ────────────────────────────────────────────
txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
if not txt_files:
    raise FileNotFoundError(
        f"No .txt files found in {DATA_DIR}\n"
        "Make sure the path points to the oxts/data/ folder."
    )

vf = np.zeros(len(txt_files))  # forward velocity  (m/s)
vl = np.zeros(len(txt_files))  # leftward velocity  (m/s)
vu = np.zeros(len(txt_files))  # upward velocity    (m/s)

for i, fpath in enumerate(txt_files):
    with open(fpath, "r") as f:
        vals = f.readline().strip().split()
    vf[i] = float(vals[8])   # column 8:  vf
    vl[i] = float(vals[9])   # column 9:  vl
    vu[i] = float(vals[10])  # column 10: vu

print(f"Loaded {len(txt_files)} frames from {DATA_DIR}")

# ── Compute speed ──────────────────────────────────────────────
speed_ms = np.sqrt(vf**2 + vl**2 + vu**2)   # m/s
speed_kmh = speed_ms * 3.6                    # km/h
speed_mph = speed_ms * 2.23694                # mph

# ── Time axis ──────────────────────────────────────────────────
# Check for a timestamps.txt file one or two levels up
ts_path = None
for parent in [os.path.dirname(DATA_DIR),
               os.path.dirname(os.path.dirname(DATA_DIR))]:
    candidate = os.path.join(parent, "timestamps.txt")
    if os.path.isfile(candidate):
        ts_path = candidate
        break

if ts_path:
    from datetime import datetime
    timestamps = []
    with open(ts_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Trim nanosecond precision to microseconds (26 chars)
            line = line[:26]
            dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S.%f")
            timestamps.append(dt)
    timestamps = timestamps[:len(txt_files)]
    t0 = timestamps[0]
    time_s = np.array([(t - t0).total_seconds() for t in timestamps])
    time_label = "Time (s)"
    print(f"Using timestamps from {ts_path}")
    print(f"Duration: {time_s[-1]:.1f} seconds")
else:
    # Fallback: assume ~10 Hz
    time_s = np.arange(len(txt_files)) * 0.1
    time_label = "Time (s)  [estimated @ 10 Hz]"
    print("No timestamps.txt found — assuming 10 Hz sample rate")

# ── Stats ──────────────────────────────────────────────────────
print(f"Speed — min: {speed_kmh.min():.1f} km/h, "
      f"max: {speed_kmh.max():.1f} km/h, "
      f"mean: {speed_kmh.mean():.1f} km/h")

# ── Derive dataset name ───────────────────────────────────────
dataset_name = os.path.basename(os.path.dirname(DATA_DIR))
if dataset_name == "data":
    dataset_name = os.path.basename(
        os.path.dirname(os.path.dirname(DATA_DIR))
    )
if not dataset_name:
    dataset_name = "KITTI Dataset"

# ── Plot ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# ---- Top plot: Speed over time ----
ax1.plot(time_s, speed_kmh, color="#2563EB", linewidth=1.5, label="Speed (km/h)")
ax1.fill_between(time_s, speed_kmh, alpha=0.15, color="#2563EB")
ax1.axhline(speed_kmh.mean(), color="#DC2626", linestyle="--",
            linewidth=1, alpha=0.7, label=f"Mean: {speed_kmh.mean():.1f} km/h")
ax1.set_ylabel("Speed (km/h)", fontsize=13)
ax1.set_title(f"Vehicle Speed — {dataset_name}", fontsize=15, fontweight="bold")
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(True, alpha=0.3)

# ---- Bottom plot: Velocity components ----
ax2.plot(time_s, vf, color="#2563EB", linewidth=1.2, label="Forward (vf)")
ax2.plot(time_s, vl, color="#16A34A", linewidth=1.2, label="Leftward (vl)")
ax2.plot(time_s, vu, color="#DC2626", linewidth=1.2, label="Upward (vu)")
ax2.set_xlabel(time_label, fontsize=13)
ax2.set_ylabel("Velocity (m/s)", fontsize=13)
ax2.set_title("Velocity Components", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(True, alpha=0.3)

fig.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"speed_{dataset_name}.png")
fig.savefig(out_path, dpi=150)
print(f"Saved plot → {out_path}")
plt.show()
