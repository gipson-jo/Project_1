"""
data_loader.py
Loads KITTI oxts data from txt files into clean NumPy arrays.
Handles both folder structures:
  - Flat:   Kitti_circle/data/*.txt  + Kitti_circle/timestamps.txt
  - Nested: oxts/data/*.txt          + oxts/timestamps.txt

Usage:
    from data_loader import load_kitti_oxts, extract_imu, extract_ground_truth
    data, timestamps = load_kitti_oxts('/path/to/Kitti_circle')
"""

import numpy as np
import os
import glob
from datetime import datetime


def parse_timestamp(ts_str):
    """Convert KITTI timestamp string to seconds."""
    ts_str = ts_str.strip()[:26]  # trim nanoseconds
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


def find_data_paths(root_dir):
    """
    Automatically find data/ and timestamps.txt regardless of folder structure.
    Handles both flat and nested (oxts/) layouts.
    """
    root_dir = os.path.expanduser(root_dir)

    # Try flat structure first: root_dir/data/*.txt
    flat_data = os.path.join(root_dir, 'data')
    flat_ts   = os.path.join(root_dir, 'timestamps.txt')
    if os.path.isdir(flat_data) and os.path.isfile(flat_ts):
        return flat_data, flat_ts

    # Try nested oxts structure: root_dir/oxts/data/*.txt
    nested_data = os.path.join(root_dir, 'oxts', 'data')
    nested_ts   = os.path.join(root_dir, 'oxts', 'timestamps.txt')
    if os.path.isdir(nested_data) and os.path.isfile(nested_ts):
        return nested_data, nested_ts

    raise FileNotFoundError(
        f'Could not find data/ and timestamps.txt in {root_dir}\n'
        f'Expected either:\n'
        f'  {root_dir}/data/*.txt\n'
        f'  {root_dir}/oxts/data/*.txt'
    )


def load_kitti_oxts(root_dir):
    """
    Load all oxts txt files and timestamps.

    Args:
        root_dir: path to dataset folder
                  e.g. ~/Project_1/Data/Kitti_circle
                  or   ~/Project_1/Data/kitti_2011_09_30_drive_0020/oxts

    Returns:
        data:       np.array of shape (N, 30)
        timestamps: np.array of shape (N,) in seconds since start
    """
    data_dir, ts_file = find_data_paths(root_dir)

    # ── Load all txt files in sorted order ──────────────────────────────────
    files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
    if len(files) == 0:
        raise FileNotFoundError(f'No txt files found in {data_dir}')

    print(f'Found {len(files)} oxts files in {os.path.basename(root_dir)}')

    rows = []
    for f in files:
        with open(f, 'r') as fh:
            values = [float(x) for x in fh.read().strip().split()]
            rows.append(values)

    data = np.array(rows)

    # ── Load timestamps ──────────────────────────────────────────────────────
    with open(ts_file, 'r') as fh:
        lines = fh.readlines()

    raw_times  = [parse_timestamp(l) for l in lines if l.strip()]
    timestamps = np.array(raw_times)
    timestamps -= timestamps[0]   # normalize to start at 0

    print(f'Duration: {timestamps[-1]:.1f}s  |  '
          f'Avg dt: {np.mean(np.diff(timestamps))*1000:.1f}ms')

    return data, timestamps


def extract_imu(data):
    """Pull out raw IMU columns."""
    accel = data[:, 11:14]   # ax, ay, az  (m/s^2)
    gyro  = data[:, 14:17]   # wx, wy, wz  (rad/s)
    return accel, gyro


def extract_ground_truth(data):
    """
    Pull out GPS/INS ground truth and convert lat/lon to local metres.
    Returns positions in ENU frame (east, north, up) relative to first frame.
    """
    lat = data[:, 0]
    lon = data[:, 1]
    alt = data[:, 2]

    roll  = data[:, 3]
    pitch = data[:, 4]
    yaw   = data[:, 5]

    # Flat-earth conversion to metres
    R    = 6378137.0
    lat0 = np.deg2rad(lat[0])
    lon0 = np.deg2rad(lon[0])

    x = R * (np.deg2rad(lon) - lon0) * np.cos(lat0)   # east
    y = R * (np.deg2rad(lat) - lat0)                   # north
    z = alt - alt[0]                                   # up

    gt_pos = np.column_stack([x, y, z])
    gt_rpy = np.column_stack([roll, pitch, yaw])

    return gt_pos, gt_rpy
