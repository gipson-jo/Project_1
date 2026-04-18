"""
dead_reckoning.py
Implements IMU-only dead reckoning using raw KITTI oxts data.

Usage:
    python3 dead_reckoning.py /path/to/Kitti_circle
"""

import numpy as np
from scipy.spatial.transform import Rotation
from data_loader import load_kitti_oxts, extract_imu, extract_ground_truth
import os


def run_dead_reckoning(accel, gyro, timestamps):
    N = len(timestamps)

    position    = np.zeros(3)
    velocity    = np.zeros(3)
    orientation = Rotation.identity()
    gravity     = np.array([0.0, 0.0, -9.81])

    positions    = np.zeros((N, 3))
    orientations = np.zeros((N, 3))

    for i in range(N - 1):   # N-1 avoids out of bounds on timestamps[i+1]

        positions[i]    = position
        orientations[i] = orientation.as_euler('xyz')

        dt = timestamps[i + 1] - timestamps[i]
        if dt <= 0 or dt > 1.0:
            continue

        # Step 1: Update orientation from gyroscope
        w     = gyro[i]
        angle = np.linalg.norm(w) * dt
        if angle > 1e-10:
            axis        = w / np.linalg.norm(w)
            delta_rot   = Rotation.from_rotvec(axis * angle)
            orientation = orientation * delta_rot

        # Step 2: Rotate accel to world frame
        accel_world  = orientation.apply(accel[i])

        # Step 3: Remove gravity
        accel_world -= gravity

        # Step 4: Integrate accel → velocity
        velocity += accel_world * dt

        # Step 5: Integrate velocity → position
        position += velocity * dt

    # Store final state
    positions[-1]    = position
    orientations[-1] = orientation.as_euler('xyz')

    print(f'Dead reckoning complete — {N} steps integrated')
    return positions, orientations


if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser('~/Project_1/Data/Kitti_circle')

    data, timestamps = load_kitti_oxts(path)
    accel, gyro      = extract_imu(data)
    gt_pos, gt_rpy   = extract_ground_truth(data)
    gt_xy            = gt_pos[:, :2]

    positions, orientations = run_dead_reckoning(accel, gyro, timestamps)

    est_xy = positions[:, :2]
    error  = np.linalg.norm(est_xy - gt_xy, axis=1)

    print(f'Final error: {error[-1]:.2f}m  |  Max: {error.max():.2f}m')
