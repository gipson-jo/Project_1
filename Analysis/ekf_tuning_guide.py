"""
ekf_tuning_guide.py
Shows different Q and R tuning configurations and their effect.
Run this to find the best tuning for your datasets.

Usage:
    python3 ekf_tuning_guide.py --data Kitti_circle
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ANALYSIS_DIR)

from data_loader import load_kitti_oxts, extract_imu, extract_ground_truth
from ekf import EKF

TUNINGS = {
    'Current (loose)': {
        'Q': np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]),
        'R': np.diag([1.0, 1.0, 1.0]),
        'color': 'red'
    },
    'Tight GPS trust': {
        'Q': np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]),
        'R': np.diag([0.01, 0.01, 0.01]),   # trust GPS much more
        'color': 'blue'
    },
    'Low process noise': {
        'Q': np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]),
        'R': np.diag([1.0, 1.0, 1.0]),
        'color': 'green'
    },
    'Both tightened': {
        'Q': np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]),
        'R': np.diag([0.01, 0.01, 0.01]),   # best of both
        'color': 'purple'
    },
}


def run_ekf_tuned(accel, gyro, timestamps, gt_pos, Q, R):
    ekf   = EKF()
    ekf.Q = Q
    ekf.R = R
    N     = len(timestamps)
    for i in range(N):
        dt = timestamps[i+1] - timestamps[i] if i < N-1 else 0.1
        if dt <= 0 or dt > 1.0:
            dt = 0.1
        ekf.predict(accel[i], gyro[i], dt)
        ekf.update(gt_pos[i])
    return ekf.get_position_history()


def compute_error(estimated, ground_truth):
    n = min(len(estimated), len(ground_truth))
    return np.linalg.norm(
        estimated[:n, :2] - ground_truth[:n, :2], axis=1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()

    data_path = os.path.expanduser(f'~/Project_1/Data/{args.data}')
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'tuning_{args.data}.png')

    print(f'Loading {args.data}...')
    data, timestamps = load_kitti_oxts(data_path)
    accel, gyro      = extract_imu(data)
    gt_pos, _        = extract_ground_truth(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    print(f'\n{"Tuning":>20} | {"RMSE":>10} | {"Final error":>12} | {"Time >5m":>10}')
    print('-' * 60)

    mean_dt = np.mean(np.diff(timestamps))

    for name, config in TUNINGS.items():
        print(f'Running {name}...')
        ekf_pos = run_ekf_tuned(
            accel, gyro, timestamps, gt_pos,
            config['Q'], config['R']
        )
        error   = compute_error(ekf_pos, gt_pos)
        rmse    = np.sqrt(np.mean(error**2))
        final   = error[-1]
        time_above = np.sum(error > 5) * mean_dt
        n       = min(len(timestamps), len(error))

        ax1.plot(gt_pos[:n, 0],  gt_pos[:n, 1],
                 color='green', linewidth=2, label='Ground truth'
                 if name == list(TUNINGS.keys())[0] else '')
        ax1.plot(ekf_pos[:n, 0], ekf_pos[:n, 1],
                 color=config['color'], linewidth=1.5,
                 linestyle='--', label=name, alpha=0.8)

        ax2.plot(timestamps[:n], error[:n],
                 color=config['color'], linewidth=1.5, label=name)

        print(f'{name:>20} | {rmse:>8.2f}m | {final:>10.2f}m | {time_above:>8.1f}s')

    ax2.axhline(5, color='orange', linestyle='--',
                linewidth=1.5, label='5m limit')

    ax1.scatter([0], [0], color='black', s=100, zorder=10, marker='*')
    ax1.set_title('Trajectory — tuning comparison')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.legend(fontsize=8)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Error over time — tuning comparison')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position error (m)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'EKF Tuning Comparison — {args.data}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output_path}')
    plt.show()


if __name__ == '__main__':
    main()
