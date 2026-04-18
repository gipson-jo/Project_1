"""
plot_dropout.py
Plot 3 — GPS dropout experiments for a single route.

Produces one figure with 4 subplots (5s, 10s, 30s, 60s outages).
Each subplot shows:
  - Black dashed: EKF baseline (no outage)
  - Blue:         EKF with outage
  - Red shaded:   GPS blocked window
  - Orange dashed: 5m acceptable limit

Metrics reported per outage:
  - Max error during outage
  - Error at outage end
  - Final error
  - Recovery time (seconds to get back under 5m)
  - Time above 5m acceptable limit
  - RMSE during outage window only

Usage:
    python3 plot_dropout.py --data Kitti_circle
    python3 plot_dropout.py --data Kitti_urban
    python3 plot_dropout.py --data Kitti_winding
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ANALYSIS_DIR)

from data_loader import load_kitti_oxts, extract_imu, extract_ground_truth
from dead_reckoning import run_dead_reckoning
from ekf import EKF

ACCEPTABLE_LIMIT = 5.0   # metres


def run_ekf(accel, gyro, timestamps, gt_pos,
            outage_start=None, outage_end=None):
    ekf = EKF()
    N   = len(timestamps)
    for i in range(N):
        dt = timestamps[i+1] - timestamps[i] if i < N-1 else 0.1
        if dt <= 0 or dt > 1.0:
            dt = 0.1
        ekf.predict(accel[i], gyro[i], dt)
        in_outage = (outage_start is not None and
                     outage_end   is not None and
                     outage_start < timestamps[i] < outage_end)
        if not in_outage:
            ekf.update(gt_pos[i])
    return ekf.get_position_history()


def compute_error(estimated, ground_truth):
    n = min(len(estimated), len(ground_truth))
    return np.linalg.norm(
        estimated[:n, :2] - ground_truth[:n, :2], axis=1
    )


def compute_metrics(error, timestamps, outage_start, outage_end,
                    acceptable_limit=ACCEPTABLE_LIMIT):
    """Compute all performance metrics for one outage experiment."""
    n = len(error)
    t = timestamps[:n]

    # Outage window mask
    outage_mask = (t >= outage_start) & (t <= outage_end)

    # Max error during outage
    max_during  = error[outage_mask].max() if outage_mask.any() else 0.0

    # Error at outage end
    end_idx     = np.argmin(np.abs(t - outage_end))
    end_error   = error[min(end_idx, n-1)]

    # Final error
    final_error = error[-1]

    # RMSE during outage only
    outage_rmse = np.sqrt(np.mean(error[outage_mask]**2)) \
                  if outage_mask.any() else 0.0

    # Time above acceptable limit (seconds)
    mean_dt      = np.mean(np.diff(timestamps))
    time_above   = np.sum(error > acceptable_limit) * mean_dt

    # Recovery time — seconds after outage end to get back under limit
    post_mask    = t > outage_end
    post_t       = t[post_mask]
    post_error   = error[post_mask]
    below_limit  = np.where(post_error <= acceptable_limit)[0]
    recovery_time = post_t[below_limit[0]] - outage_end \
                    if len(below_limit) > 0 else float('inf')

    return {
        'max_during':    max_during,
        'end_error':     end_error,
        'final_error':   final_error,
        'outage_rmse':   outage_rmse,
        'time_above':    time_above,
        'recovery_time': recovery_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True,
                        help='Dataset name e.g. Kitti_circle')
    parser.add_argument('--outage_start', type=float, default=30.0,
                        help='When GPS outage begins in seconds (default: 30)')
    args = parser.parse_args()

    data_path = os.path.expanduser(f'~/Project_1/Data/{args.data}')
    if not os.path.exists(data_path):
        print(f'ERROR: Could not find dataset at {data_path}')
        sys.exit(1)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'dropout_{args.data}.png')

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f'Loading {args.data}...')
    data, timestamps = load_kitti_oxts(data_path)
    accel, gyro      = extract_imu(data)
    gt_pos, _        = extract_ground_truth(data)

    outage_start = args.outage_start
    durations    = [5, 10, 30, 60]

    # ── Baseline EKF ─────────────────────────────────────────────────────────
    print('Running EKF baseline (no outage)...')
    ekf_base   = run_ekf(accel, gyro, timestamps, gt_pos)
    base_error = compute_error(ekf_base, gt_pos)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes      = axes.flatten()

    all_metrics = {}

    for idx, duration in enumerate(durations):
        ax         = axes[idx]
        outage_end = min(outage_start + duration, timestamps[-1])

        print(f'Running EKF with {duration}s outage '
              f'({outage_start:.0f}s → {outage_end:.0f}s)...')

        ekf_pos   = run_ekf(accel, gyro, timestamps, gt_pos,
                            outage_start, outage_end)
        ekf_error = compute_error(ekf_pos, gt_pos)

        metrics   = compute_metrics(ekf_error, timestamps,
                                    outage_start, outage_end)
        all_metrics[duration] = metrics

        n = min(len(timestamps), len(base_error), len(ekf_error))
        t = timestamps[:n]

        # ── Plot ─────────────────────────────────────────────────────────────
        ax.plot(t, base_error[:n], color='black', linewidth=1.5,
                linestyle='--', alpha=0.6, label='EKF (no outage)')
        ax.plot(t, ekf_error[:n],  color='blue',  linewidth=2,
                label=f'EKF ({duration}s outage)')

        # Acceptable limit line
        ax.axhline(ACCEPTABLE_LIMIT, color='orange', linestyle='--',
                   linewidth=1.5, label=f'Acceptable limit ({ACCEPTABLE_LIMIT}m)')

        # Outage window
        ax.axvspan(outage_start, outage_end,
                   alpha=0.12, color='red', label='GPS blocked')
        ax.axvline(outage_start, color='red',   linestyle='--',
                   alpha=0.5, linewidth=1)
        ax.axvline(outage_end,   color='green', linestyle='--',
                   alpha=0.5, linewidth=1)

        # Recovery time annotation
        rec = metrics['recovery_time']
        rec_str = f'{rec:.1f}s' if rec != float('inf') else 'No recovery'
        ax.set_title(
            f'{duration}s GPS outage  ({outage_start:.0f}s → {outage_end:.0f}s)\n'
            f'Max: {metrics["max_during"]:.1f}m  '
            f'Recovery: {rec_str}  '
            f'Time >5m: {metrics["time_above"]:.1f}s',
            fontsize=9, fontweight='bold'
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position error (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, timestamps[-1]])

    # ── Print full metrics table ──────────────────────────────────────────────
    print(f'\n{"="*80}')
    print(f'{args.data} — Full Dropout Metrics')
    print(f'{"="*80}')
    print(f'{"Outage":>8} | {"Max error":>10} | {"End error":>10} | '
          f'{"Final":>8} | {"Outage RMSE":>12} | '
          f'{"Time >5m":>10} | {"Recovery":>10}')
    print('-' * 80)

    for duration in durations:
        m   = all_metrics[duration]
        rec = f'{m["recovery_time"]:.1f}s' \
              if m['recovery_time'] != float('inf') else 'None'
        print(f'{duration:>6}s | {m["max_during"]:>8.1f}m | '
              f'{m["end_error"]:>8.1f}m | {m["final_error"]:>6.1f}m | '
              f'{m["outage_rmse"]:>10.1f}m | '
              f'{m["time_above"]:>8.1f}s | {rec:>10}')

    plt.suptitle(
        f'GPS Dropout Experiments — {args.data}\n'
        f'Black dashed = EKF baseline   Blue = EKF with outage   '
        f'Orange dashed = 5m limit   Shaded = GPS blocked',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved to {output_path}')
    plt.show()


if __name__ == '__main__':
    main()
