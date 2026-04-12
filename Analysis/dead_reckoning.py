"""
dead_reckoning.py
Implements IMU-only dead reckoning using raw KITTI oxts data.

This is the BASELINE model — integrates gyroscope and accelerometer
readings step by step to estimate position and orientation.
It will drift over time — that's expected and is the whole point.

Usage:
    python3 dead_reckoning.py
"""

import numpy as np
from scipy.spatial.transform import Rotation
from data_loader import load_kitti_oxts, extract_imu, extract_ground_truth
import os


def run_dead_reckoning(accel, gyro, timestamps):
    """
    Core dead reckoning loop.

    Args:
        accel:      (N, 3) raw accelerometer readings in body frame (m/s^2)
        gyro:       (N, 3) raw gyroscope readings in body frame (rad/s)
        timestamps: (N,)   time in seconds since start

    Returns:
        positions:    (N, 3) estimated [x, y, z] at each timestep (metres)
        orientations: (N, 3) estimated [roll, pitch, yaw] at each timestep
    """

    N = len(timestamps)

    # ── State variables — all start at zero ─────────────────────────────────
    position    = np.zeros(3)        # [x, y, z] in metres
    velocity    = np.zeros(3)        # [vx, vy, vz] in m/s
    orientation = Rotation.identity() # current orientation as quaternion

    # Gravity in world frame — KITTI uses ENU (East North Up)
    # In ENU, gravity points DOWN = negative Z
    gravity = np.array([0.0, 0.0, -9.81])

    # ── Output arrays ────────────────────────────────────────────────────────
    positions    = np.zeros((N, 3))
    orientations = np.zeros((N, 3))  # stored as roll, pitch, yaw for easy plotting

    # ── Main integration loop ────────────────────────────────────────────────
    for i in range(N):

        # Store current state BEFORE updating
        positions[i]    = position
        orientations[i] = orientation.as_euler('xyz')  # roll, pitch, yaw

        # Skip update on last frame
        if i == N - 1:
            break

        # ── Compute dt ───────────────────────────────────────────────────────
        dt = timestamps[i + 1] - timestamps[i]

        # Safety check — skip if dt is weird
        if dt <= 0 or dt > 1.0:
            continue

        # ── Step 1: Update orientation from gyroscope ─────────────────────
        # gyro gives angular velocity in rad/s in body frame
        # multiply by dt to get the angle rotated this timestep
        w = gyro[i]                          # [wx, wy, wz]
        angle = np.linalg.norm(w) * dt       # total angle rotated (radians)

        if angle > 1e-10:                    # avoid division by zero
            axis = w / np.linalg.norm(w)    # unit vector of rotation axis
            delta_rot = Rotation.from_rotvec(axis * angle)
            orientation = orientation * delta_rot

        # ── Step 2: Rotate accelerometer to world frame ───────────────────
        # The accelerometer measures in the sensor's body frame.
        # We need to rotate those readings into the world (ENU) frame
        # using our current orientation estimate.
        accel_world = orientation.apply(accel[i])

        # ── Step 3: Remove gravity ────────────────────────────────────────
        # The accelerometer always measures gravity even when stationary.
        # After rotating to world frame, gravity is [0, 0, -9.81] in ENU.
        # Subtracting it leaves only motion-induced acceleration.
        accel_world -= gravity

        # ── Step 4: Integrate acceleration → velocity ─────────────────────
        # v = v + a * dt
        velocity += accel_world * dt

        # ── Step 5: Integrate velocity → position ─────────────────────────
        # p = p + v * dt
        position += velocity * dt

    print(f'Dead reckoning complete — {N} steps integrated')
    return positions, orientations


if __name__ == '__main__':
    import sys

    oxts_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser('~/Project_1/Data/kitti_2011_09_30_drive_0020/oxts')

    # ── Load data ────────────────────────────────────────────────────────────
    print('Loading KITTI oxts data...')
    data, timestamps = load_kitti_oxts(oxts_path)
    accel, gyro      = extract_imu(data)
    gt_xy, gt_alt, gt_rpy = extract_ground_truth(data)

    # ── Run dead reckoning ───────────────────────────────────────────────────
    print('Running dead reckoning...')
    positions, orientations = run_dead_reckoning(accel, gyro, timestamps)

    # ── Quick error summary ──────────────────────────────────────────────────
    # Compare estimated XY against ground truth XY
    est_xy = positions[:, :2]   # take just x, y from estimated positions

    # Align ground truth to same starting frame
    # (GT starts at 0,0 already from our loader)
    error = np.linalg.norm(est_xy - gt_xy, axis=1)

    print(f'\n--- Results ---')
    print(f'Final position error: {error[-1]:.2f} m')
    print(f'Max position error:   {error.max():.2f} m')
    print(f'Error at 10s:         {error[np.argmin(np.abs(timestamps-10))  ]:.2f} m')
    print(f'Error at 30s:         {error[np.argmin(np.abs(timestamps-30))  ]:.2f} m')
    print(f'Error at 60s:         {error[np.argmin(np.abs(timestamps-60))  ]:.2f} m')

    # Save results for plotting
    np.save('dead_reckoning_positions.npy', positions)
    np.save('ground_truth_xy.npy', gt_xy)
    np.save('timestamps.npy', timestamps)
    print('\nSaved results to .npy files — run plot_results.py to visualize')