"""
run.py
Master script — runs all experiments and generates plots.

Usage:
    python3 run.py --mode dead_reckoning
    python3 run.py --mode ekf
    python3 run.py --mode compare
    python3 run.py --mode dropout --duration 30
    python3 run.py --mode all
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os
from scipy.spatial.transform import Rotation

from data_loader import load_kitti_oxts, extract_imu, extract_ground_truth
from dead_reckoning import run_dead_reckoning
from ekf import EKF


# ── Default data path ────────────────────────────────────────────────────────
OXTS_PATH = os.path.expanduser(
    '~/Project_1/Data/kitti_2011_09_30_drive_0020/oxts'
)


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(oxts_path=OXTS_PATH):
    print('Loading KITTI oxts data...')
    data, timestamps = load_kitti_oxts(oxts_path)
    accel, gyro      = extract_imu(data)
    gt_xy, gt_alt, gt_rpy = extract_ground_truth(data)

    # Build 3D ground truth position (x=east, y=north, z=alt relative to start)
    gt_z   = gt_alt - gt_alt[0]
    gt_pos = np.column_stack([gt_xy, gt_z])   # (N, 3)

    return accel, gyro, timestamps, gt_pos, gt_rpy


# ════════════════════════════════════════════════════════════════════════════
# EKF RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_ekf(accel, gyro, timestamps, gt_pos,
            outage_start=None, outage_end=None):
    """
    Run EKF over the full sequence.
    Optionally block GPS updates between outage_start and outage_end seconds.

    Args:
        outage_start: float or None — time (s) to start blocking GPS
        outage_end:   float or None — time (s) to resume GPS
    """

    ekf = EKF()
    N   = len(timestamps)

    gps_blocked_count = 0

    for i in range(N):
        t  = timestamps[i]
        dt = timestamps[i+1] - timestamps[i] if i < N-1 else timestamps[-1] - timestamps[-2]

        if dt <= 0 or dt > 1.0:
            dt = 0.1

        # ── Predict step (always runs) ────────────────────────────────────────
        ekf.predict(accel[i], gyro[i], dt)

        # ── Update step (blocked during outage) ───────────────────────────────
        in_outage = (outage_start is not None and
                     outage_end   is not None and
                     outage_start < t < outage_end)

        if not in_outage:
            ekf.update(gt_pos[i])   # use GT as GPS proxy
        else:
            gps_blocked_count += 1

    if outage_start is not None:
        print(f'GPS blocked for {gps_blocked_count} frames '
              f'({outage_start}s → {outage_end}s)')

    return ekf.get_position_history(), ekf.get_covariance_history()


# ════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def compute_error(estimated, ground_truth):
    """Euclidean distance between estimated and ground truth XY at each step."""
    n = min(len(estimated), len(ground_truth))
    return np.linalg.norm(estimated[:n, :2] - ground_truth[:n, :2], axis=1)


def shade_outage(ax, outage_start, outage_end, timestamps):
    """Add a red shaded region to a plot showing the GPS outage window."""
    if outage_start is not None:
        ax.axvspan(outage_start, outage_end, alpha=0.15,
                   color='red', label='GPS outage')


# ════════════════════════════════════════════════════════════════════════════
# MODE: DEAD RECKONING
# ════════════════════════════════════════════════════════════════════════════

def mode_dead_reckoning(accel, gyro, timestamps, gt_pos):
    print('\n--- Running dead reckoning ---')
    dr_pos, _ = run_dead_reckoning(accel, gyro, timestamps)
    error      = compute_error(dr_pos, gt_pos)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Trajectory
    axes[0].plot(gt_pos[:, 0], gt_pos[:, 1], 'g-',  lw=2,   label='Ground truth')
    axes[0].plot(dr_pos[:, 0], dr_pos[:, 1], 'r--', lw=1.5, label='Dead reckoning')
    axes[0].scatter([0], [0], color='black', s=80, zorder=5, label='Start')
    axes[0].set_title('Trajectory — dead reckoning vs ground truth')
    axes[0].set_xlabel('East (m)')
    axes[0].set_ylabel('North (m)')
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)

    # Error over time
    axes[1].plot(timestamps, error, 'r-', lw=1.5)
    axes[1].set_title('Position error over time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error (m)')
    axes[1].grid(True, alpha=0.3)

    print(f'Final error: {error[-1]:.1f}m  |  Max: {error.max():.1f}m')

    plt.suptitle('Dead Reckoning Baseline', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results_dead_reckoning.png', dpi=150)
    plt.show()
    return dr_pos, error


# ════════════════════════════════════════════════════════════════════════════
# MODE: EKF (no outage)
# ════════════════════════════════════════════════════════════════════════════

def mode_ekf(accel, gyro, timestamps, gt_pos):
    print('\n--- Running EKF (no outage) ---')
    ekf_pos, ekf_cov = run_ekf(accel, gyro, timestamps, gt_pos)
    error             = compute_error(ekf_pos, gt_pos)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Trajectory
    axes[0].plot(gt_pos[:, 0],  gt_pos[:, 1],  'g-',  lw=2,   label='Ground truth')
    axes[0].plot(ekf_pos[:, 0], ekf_pos[:, 1], 'b--', lw=1.5, label='EKF')
    axes[0].scatter([0], [0], color='black', s=80, zorder=5)
    axes[0].set_title('EKF trajectory vs ground truth')
    axes[0].set_xlabel('East (m)')
    axes[0].set_ylabel('North (m)')
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)

    # Error over time
    axes[1].plot(timestamps[:len(error)], error, 'b-', lw=1.5)
    axes[1].set_title('EKF position error over time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error (m)')
    axes[1].grid(True, alpha=0.3)

    # Covariance (uncertainty) over time
    axes[2].plot(timestamps[:len(ekf_cov)], ekf_cov, 'purple', lw=1.5)
    axes[2].set_title('EKF uncertainty (trace of P) over time')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Uncertainty')
    axes[2].grid(True, alpha=0.3)

    print(f'Final error: {error[-1]:.2f}m  |  Max: {error.max():.2f}m')

    plt.suptitle('EKF Results (no GPS outage)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results_ekf.png', dpi=150)
    plt.show()
    return ekf_pos, error


# ════════════════════════════════════════════════════════════════════════════
# MODE: COMPARE dead reckoning vs EKF
# ════════════════════════════════════════════════════════════════════════════

def mode_compare(accel, gyro, timestamps, gt_pos):
    print('\n--- Comparing dead reckoning vs EKF ---')
    dr_pos,  dr_error  = mode_dead_reckoning(accel, gyro, timestamps, gt_pos)
    ekf_pos, ekf_error = mode_ekf(accel, gyro, timestamps, gt_pos)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Trajectory comparison
    axes[0].plot(gt_pos[:, 0],  gt_pos[:, 1],  'g-',  lw=2.5, label='Ground truth')
    axes[0].plot(dr_pos[:, 0],  dr_pos[:, 1],  'r--', lw=1.5, label='Dead reckoning')
    axes[0].plot(ekf_pos[:, 0], ekf_pos[:, 1], 'b--', lw=1.5, label='EKF')
    axes[0].scatter([0], [0], color='black', s=80, zorder=5, label='Start')
    axes[0].set_title('Trajectory comparison')
    axes[0].set_xlabel('East (m)')
    axes[0].set_ylabel('North (m)')
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)

    # Error comparison
    n = min(len(dr_error), len(ekf_error))
    axes[1].plot(timestamps[:n], dr_error[:n],  'r-', lw=1.5, label='Dead reckoning')
    axes[1].plot(timestamps[:n], ekf_error[:n], 'b-', lw=1.5, label='EKF')
    axes[1].set_title('Position error comparison')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Dead Reckoning vs EKF', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results_compare.png', dpi=150)
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# MODE: GPS DROPOUT EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

def mode_dropout(accel, gyro, timestamps, gt_pos):
    print('\n--- Running GPS dropout experiments ---')

    outage_start = 30.0
    durations    = [5, 10, 30, 60]

    # Run baseline EKF (no outage) once — reused in every subplot
    print('  Running baseline EKF (no outage)...')
    ekf_base, _  = run_ekf(accel, gyro, timestamps, gt_pos)
    base_error   = compute_error(ekf_base, gt_pos)

    # Run dead reckoning once — reused in every subplot
    dr_pos, _ = run_dead_reckoning(accel, gyro, timestamps)
    dr_error  = compute_error(dr_pos, gt_pos)

    # One figure with 4 subplots — one per outage duration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()   # makes indexing easier

    summary = {}

    for idx, duration in enumerate(durations):
        ax = axes[idx]

        outage_end = min(outage_start + duration, timestamps[-1])

        print(f'  Running {duration}s outage ({outage_start}s → {outage_end:.0f}s)...')
        ekf_pos, ekf_cov = run_ekf(accel, gyro, timestamps, gt_pos,
                                    outage_start, outage_end)
        error = compute_error(ekf_pos, gt_pos)
        summary[duration] = error

        n = min(len(timestamps), len(error), len(base_error), len(dr_error))
        t = timestamps[:n]

        # Dead reckoning error
        #ax.plot(t, dr_error[:n],   'r-',  lw=1.5, alpha=0.7, label='Dead reckoning')

        # Baseline EKF error
        ax.plot(t, base_error[:n], 'k--', lw=1.5, alpha=0.6, label='EKF (no outage)')

        # This outage's EKF error
        ax.plot(t, error[:n],      'b-',  lw=2,               label=f'EKF ({duration}s outage)')

        # Shade the outage window
        ax.axvspan(outage_start, outage_end, alpha=0.15,
                   color='red', label='GPS blocked')

        # Mark outage start and end
        ax.axvline(outage_start, color='red',    linestyle='--', alpha=0.6, lw=1)
        ax.axvline(outage_end,   color='green',  linestyle='--', alpha=0.6, lw=1)

        ax.set_title(f'{duration}s GPS outage  ({outage_start:.0f}s → {outage_end:.0f}s)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position error (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, timestamps[-1]])

    # Print summary table
    print(f'\n{"Outage":>10} | {"Max error during outage":>24} | {"Error at outage end":>20}')
    print('-' * 62)
    for duration in durations:
        error      = summary[duration]
        outage_end = min(outage_start + duration, timestamps[-1])
        mask       = (timestamps[:len(error)] >= outage_start) & \
                     (timestamps[:len(error)] <= outage_end)
        if mask.any():
            max_during = error[mask].max()
            end_idx    = np.argmin(np.abs(timestamps - outage_end))
            end_error  = error[min(end_idx, len(error)-1)]
            print(f'{duration:>8}s | {max_during:>22.1f}m | {end_error:>18.1f}m')

    plt.suptitle('EKF GPS Dropout Experiments — KITTI 2011_09_30_drive_0020\n'
                 'Each plot shows dead reckoning (red), EKF baseline (black), '
                 'EKF with outage (blue)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results_dropout.png', dpi=150)
    plt.show()
    print('\nSaved to results_dropout.png')


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='EECE5554 Localization Project')
    parser.add_argument('--mode', choices=['dead_reckoning', 'ekf', 'compare',
                                           'dropout', 'all'],
                        default='compare',
                        help='Which experiment to run')
    parser.add_argument('--data', default=OXTS_PATH,
                        help='Path to oxts folder')
    args = parser.parse_args()

    # Load data once, share across all modes
    accel, gyro, timestamps, gt_pos, gt_rpy = load_data(args.data)

    if args.mode == 'dead_reckoning':
        mode_dead_reckoning(accel, gyro, timestamps, gt_pos)

    elif args.mode == 'ekf':
        mode_ekf(accel, gyro, timestamps, gt_pos)

    elif args.mode == 'compare':
        mode_compare(accel, gyro, timestamps, gt_pos)

    elif args.mode == 'dropout':
        mode_dropout(accel, gyro, timestamps, gt_pos)

    elif args.mode == 'all':
        mode_dead_reckoning(accel, gyro, timestamps, gt_pos)
        mode_ekf(accel, gyro, timestamps, gt_pos)
        mode_compare(accel, gyro, timestamps, gt_pos)
        mode_dropout(accel, gyro, timestamps, gt_pos)


if __name__ == '__main__':
    main()