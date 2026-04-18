"""
plot_trajectory_comparison.py
Trajectory comparison — Dead Reckoning vs EKF vs Ground Truth.

Left plot:  Full scale — shows how far DR drifts
Right plot: Scaled DR — shows directional error without magnitude blowing axis

Usage:
    python3 plot_trajectory_comparison.py --data Kitti_circle
    python3 plot_trajectory_comparison.py --data Kitti_urban
    python3 plot_trajectory_comparison.py --data Kitti_winding
    python3 plot_trajectory_comparison.py --data Kitti_straight
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# ── Make sure Analysis/ is on the path ──────────────────────────────────────
ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ANALYSIS_DIR)

from data_loader import load_kitti_oxts, extract_imu, extract_ground_truth
from dead_reckoning import run_dead_reckoning
from ekf import EKF


def run_ekf_full(accel, gyro, timestamps, gt_pos):
    """Run EKF with full GPS — no outage."""
    ekf = EKF()
    N   = len(timestamps)
    for i in range(N):
        dt = timestamps[i+1] - timestamps[i] if i < N-1 else 0.1
        if dt <= 0 or dt > 1.0:
            dt = 0.1
        ekf.predict(accel[i], gyro[i], dt)
        ekf.update(gt_pos[i])
    return ekf.get_position_history()


def compute_rmse(estimated, ground_truth):
    n     = min(len(estimated), len(ground_truth))
    error = np.linalg.norm(
        estimated[:n, :2] - ground_truth[:n, :2], axis=1
    )
    return np.sqrt(np.mean(error**2)), error


def scale_trajectory(est_xy, gt_xy):
    """
    Scale estimated trajectory so total path length matches ground truth.
    Removes magnitude error — shows only directional/shape error.
    """
    est_total = np.sum(np.linalg.norm(np.diff(est_xy, axis=0), axis=1))
    gt_total  = np.sum(np.linalg.norm(np.diff(gt_xy,  axis=0), axis=1))
    if est_total < 1e-6:
        return est_xy
    return est_xy * (gt_total / est_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True,
                        help='Dataset name e.g. Kitti_circle')
    args = parser.parse_args()

    data_path = os.path.expanduser(f'~/Project_1/Data/{args.data}')
    if not os.path.exists(data_path):
        print(f'ERROR: Could not find dataset at {data_path}')
        sys.exit(1)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        script_dir, f'trajectory_comparison_{args.data}.png'
    )

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f'Loading {args.data}...')
    data, timestamps = load_kitti_oxts(data_path)
    accel, gyro      = extract_imu(data)
    gt_pos, _        = extract_ground_truth(data)

    # ── Run models ───────────────────────────────────────────────────────────
    print('Running dead reckoning...')
    dr_pos, _  = run_dead_reckoning(accel, gyro, timestamps)

    print('Running EKF...')
    ekf_pos    = run_ekf_full(accel, gyro, timestamps, gt_pos)

    # ── Errors ───────────────────────────────────────────────────────────────
    dr_rmse,  _ = compute_rmse(dr_pos,  gt_pos)
    ekf_rmse, _ = compute_rmse(ekf_pos, gt_pos)

    n          = min(len(gt_pos), len(dr_pos), len(ekf_pos))
    gt_xy      = gt_pos[:n, :2]
    dr_xy      = dr_pos[:n, :2]
    ekf_xy     = ekf_pos[:n, :2]
    dr_scaled  = scale_trajectory(dr_xy, gt_xy)

    scale_factor = (
        np.sum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)) /
        np.sum(np.linalg.norm(np.diff(dr_xy, axis=0), axis=1))
    )

    distance_traveled = np.sum(
        np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)
    )

    # ── Figure — two subplots ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: full scale ─────────────────────────────────────────────────────
    ax1.plot(gt_xy[:, 0],  gt_xy[:, 1],
             color='green', linewidth=2.5, label='Ground truth', zorder=5)
    ax1.plot(dr_xy[:, 0],  dr_xy[:, 1],
             color='red',   linewidth=1.5, linestyle='--',
             label=f'Dead reckoning  (RMSE: {dr_rmse:.1f}m)', alpha=0.85)
    ax1.plot(ekf_xy[:, 0], ekf_xy[:, 1],
             color='blue',  linewidth=1.5, linestyle='--',
             label=f'EKF  (RMSE: {ekf_rmse:.1f}m)', alpha=0.85)
    ax1.scatter([0], [0], color='black', s=120, zorder=10,
                marker='*', label='Start')
    ax1.set_title('Full scale\n(shows magnitude of DR drift)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.legend(fontsize=9)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # ── Right: scaled DR ─────────────────────────────────────────────────────
    ax2.plot(gt_xy[:, 0],     gt_xy[:, 1],
             color='green', linewidth=2.5, label='Ground truth', zorder=5)
    ax2.plot(dr_scaled[:, 0], dr_scaled[:, 1],
             color='red',   linewidth=1.5, linestyle='--',
             label=f'Dead reckoning scaled  (scale: {scale_factor:.3f})',
             alpha=0.85)
    ax2.plot(ekf_xy[:, 0],    ekf_xy[:, 1],
             color='blue',  linewidth=1.5, linestyle='--',
             label=f'EKF  (RMSE: {ekf_rmse:.1f}m)', alpha=0.85)
    ax2.scatter([0], [0], color='black', s=120, zorder=10,
                marker='*', label='Start')
    ax2.set_title('DR scaled to match GT path length\n'
                  '(shows directional error only)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax2.legend(fontsize=9)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'Trajectory Comparison — {args.data}\n'
        f'{len(timestamps)} frames  ·  {timestamps[-1]:.0f}s  ·  '
        f'~{distance_traveled:.0f}m traveled',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output_path}')
    plt.show()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f'\n--- {args.data} Results ---')
    print(f'Duration:          {timestamps[-1]:.1f}s')
    print(f'Distance traveled: {distance_traveled:.0f}m')
    print(f'DR  RMSE:          {dr_rmse:.2f}m')
    print(f'EKF RMSE:          {ekf_rmse:.2f}m')
    print(f'EKF improvement:   {((dr_rmse - ekf_rmse) / dr_rmse * 100):.1f}%')
    print(f'DR scale factor:   {scale_factor:.4f}')


if __name__ == '__main__':
    main()
