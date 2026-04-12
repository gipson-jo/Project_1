"""
data_loader.py
Loads KITTI oxts data from the txt files into clean NumPy arrays.

Usage:
    from data_loader import load_kitti_oxts
    data, timestamps = load_kitti_oxts('/path/to/oxts')
"""

import numpy as np
import os
import glob
from datetime import datetime


def parse_timestamp(ts_str):
    """Convert KITTI timestamp string to seconds since first frame."""
    ts_str = ts_str.strip()
    # KITTI timestamps have nanoseconds — trim to microseconds
    ts_str = ts_str[:26]
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


def load_kitti_oxts(oxts_dir):
    """
    Load all oxts txt files and timestamps.

    Args:
        oxts_dir: path to the oxts/ folder
                  e.g. ~/Project_1/Data/kitti_2011_09_30_drive_0020/oxts

    Returns:
        data:       np.array of shape (N, 30) — all 30 columns per timestep
        timestamps: np.array of shape (N,)   — time in seconds since start
    """

    # ── Load all txt files in sorted order ──────────────────────────────────
    data_dir = os.path.join(oxts_dir, 'data')
    files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))

    if len(files) == 0:
        raise FileNotFoundError(f'No txt files found in {data_dir}')

    print(f'Found {len(files)} oxts files')

    # ── Parse each file into a row of floats ────────────────────────────────
    rows = []
    for f in files:
        with open(f, 'r') as fh:
            values = [float(x) for x in fh.read().strip().split()]
            rows.append(values)

    data = np.array(rows)  # shape: (N, 30)

    # ── Load timestamps ──────────────────────────────────────────────────────
    ts_file = os.path.join(oxts_dir, 'timestamps.txt')
    with open(ts_file, 'r') as fh:
        lines = fh.readlines()

    raw_times = [parse_timestamp(l) for l in lines if l.strip()]
    timestamps = np.array(raw_times)

    # Normalize so t=0 at first frame
    timestamps -= timestamps[0]

    print(f'Sequence duration: {timestamps[-1]:.2f} seconds')
    print(f'Average dt: {np.mean(np.diff(timestamps))*1000:.2f} ms')

    return data, timestamps


def extract_imu(data):
    """
    Pull out just the raw IMU columns from the full data array.

    Returns:
        accel: (N, 3) — ax, ay, az in m/s^2 (body frame, includes gravity)
        gyro:  (N, 3) — wx, wy, wz in rad/s
    """
    accel = data[:, 11:14]   # columns 11, 12, 13
    gyro  = data[:, 14:17]   # columns 14, 15, 16
    return accel, gyro


def extract_ground_truth(data):
    """
    Pull out GPS/INS ground truth and convert lat/lon to local metres.

    Returns:
        gt_xy:  (N, 2) — x (east), y (north) in metres relative to start
        gt_alt: (N,)   — altitude in metres
        gt_rpy: (N, 3) — roll, pitch, yaw in radians
    """
    lat = data[:, 0]   # degrees
    lon = data[:, 1]   # degrees
    alt = data[:, 2]   # metres

    roll  = data[:, 3]
    pitch = data[:, 4]
    yaw   = data[:, 5]

    # ── Convert lat/lon to local metres ─────────────────────────────────────
    # Treat first frame as origin (0, 0)
    lat0 = np.deg2rad(lat[0])
    lon0 = np.deg2rad(lon[0])

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Standard flat-earth approximation (accurate for short sequences <10km)
    R = 6378137.0  # Earth radius in metres
    x = R * (lon_rad - lon0) * np.cos(lat0)   # east
    y = R * (lat_rad - lat0)                   # north

    gt_xy  = np.column_stack([x, y])
    gt_rpy = np.column_stack([roll, pitch, yaw])

    return gt_xy, alt, gt_rpy


if __name__ == '__main__':
    # Quick test — run this file directly to check your data loaded correctly
    import sys

    oxts_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser('~/Project_1/Data/kitti_2011_09_30_drive_0020/oxts')

    data, timestamps = load_kitti_oxts(oxts_path)
    accel, gyro = extract_imu(data)
    gt_xy, gt_alt, gt_rpy = extract_ground_truth(data)

    print(f'\nData shape:      {data.shape}')
    print(f'Accel shape:     {accel.shape}')
    print(f'Gyro shape:      {gyro.shape}')
    print(f'Ground truth xy: {gt_xy.shape}')
    print(f'\nFirst accel reading: ax={accel[0,0]:.4f} ay={accel[0,1]:.4f} az={accel[0,2]:.4f}')
    print(f'First gyro  reading: wx={gyro[0,0]:.4f}  wy={gyro[0,1]:.4f}  wz={gyro[0,2]:.4f}')
    print(f'First GT position:   x={gt_xy[0,0]:.2f}m  y={gt_xy[0,1]:.2f}m')
    print(f'Total distance traveled (GT): {np.sum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)):.1f}m')