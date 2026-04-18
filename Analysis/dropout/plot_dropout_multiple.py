"""
plot_dropout_multiple.py
Multiple short GPS outages experiment.

Fixed parameters across all routes:
  Buffer:          20s  (EKF initialization time)
  Outage duration: 5s
  Gap between:     10s
  Max outages:     1, 2, 3, 5

Outage windows (same for all routes):
  1 outage:  20s → 25s
  2 outages: 20s → 25s  |  35s → 40s
  3 outages: 20s → 25s  |  35s → 40s  |  50s → 55s
  5 outages: 20s → 25s  |  35s → 40s  |  50s → 55s  |  65s → 70s  |  80s → 85s

Produces two figures:
  1. 4 subplots — error over time for each outage count
  2. Bar chart  — total time above 5m acceptable limit

Usage:
    python3 plot_dropout_multiple.py --data Kitti_circle
    python3 plot_dropout_multiple.py --data Kitti_straight
    python3 plot_dropout_multiple.py --data Kitti_winding
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

ACCEPTABLE_LIMIT = 5.0
BUFFER           = 20.0
OUTAGE_DURATION  = 5.0
GAP              = 10.0
N_OUTAGES_LIST   = [1, 2, 3, 5]


def get_windows(n_outages):
    windows = []
    for i in range(n_outages):
        start = BUFFER + i * (OUTAGE_DURATION + GAP)
        end   = start + OUTAGE_DURATION
        windows.append((start, end))
    return windows


def run_ekf_multiple_outages(accel, gyro, timestamps, gt_pos, outage_windows):
    ekf = EKF()
    N   = len(timestamps)
    for i in range(N):
        dt = timestamps[i+1] - timestamps[i] if i < N-1 else 0.1
        if dt <= 0 or dt > 1.0:
            dt = 0.1
        ekf.predict(accel[i], gyro[i], dt)
        t         = timestamps[i]
        in_outage = any(start < t < end for start, end in outage_windows)
        if not in_outage:
            ekf.update(gt_pos[i])
    return ekf.get_position_history()


def compute_error(estimated, ground_truth):
    n = min(len(estimated), len(ground_truth))
    return np.linalg.norm(
        estimated[:n, :2] - ground_truth[:n, :2], axis=1
    )


def compute_metrics(error, timestamps, outage_windows):
    n       = len(error)
    t       = timestamps[:n]
    mean_dt = np.mean(np.diff(timestamps))

    max_error    = error.max()
    final_error  = error[-1]
    overall_rmse = np.sqrt(np.mean(error**2))
    time_above   = np.sum(error > ACCEPTABLE_LIMIT) * mean_dt

    per_window       = []
    recovery_times   = []
    cumulative_drift = []

    for start, end in outage_windows:
        end_idx   = np.argmin(np.abs(t - end))
        end_error = error[min(end_idx, n-1)]
        cumulative_drift.append(end_error)

        mask       = (t >= start) & (t <= end)
        max_during = error[mask].max() if mask.any() else 0.0

        post_mask  = t > end
        post_t     = t[post_mask]
        post_error = error[post_mask]
        below      = np.where(post_error <= ACCEPTABLE_LIMIT)[0]
        rec_time   = post_t[below[0]] - end \
                     if len(below) > 0 else float('inf')
        recovery_times.append(rec_time)

        per_window.append({
            'start':      start,
            'end':        end,
            'max_during': max_during,
            'end_error':  end_error,
            'recovery':   rec_time,
        })

    finite_rec   = [r for r in recovery_times if r != float('inf')]
    avg_recovery = np.mean(finite_rec) if finite_rec else float('inf')

    drift_increasing = all(
        cumulative_drift[i] <= cumulative_drift[i+1]
        for i in range(len(cumulative_drift)-1)
    ) if len(cumulative_drift) > 1 else None

    return {
        'max_error':        max_error,
        'final_error':      final_error,
        'overall_rmse':     overall_rmse,
        'time_above':       time_above,
        'avg_recovery':     avg_recovery,
        'per_window':       per_window,
        'cumulative_drift': cumulative_drift,
        'drift_increasing': drift_increasing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True,
                        help='Dataset name e.g. Kitti_circle')
    args = parser.parse_args()

    data_path = os.path.expanduser(f'~/Project_1/Data/{args.data}')
    if not os.path.exists(data_path):
        print(f'ERROR: Could not find dataset at {data_path}')
        sys.exit(1)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    output_path  = os.path.join(script_dir,
                                f'dropout_multiple_{args.data}.png')
    bar_path     = os.path.join(script_dir,
                                f'dropout_multiple_{args.data}_time_above_5m.png')

    print(f'Loading {args.data}...')
    data, timestamps = load_kitti_oxts(data_path)
    accel, gyro      = extract_imu(data)
    gt_pos, _        = extract_ground_truth(data)

    total_duration = timestamps[-1]

    # Baseline EKF
    print('Running EKF baseline (no outage)...')
    ekf_base   = run_ekf_multiple_outages(accel, gyro, timestamps, gt_pos, [])
    base_error = compute_error(ekf_base, gt_pos)

    # ── Figure 1: 4 subplots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes      = axes.flatten()

    all_metrics = {}

    for idx, n_outages in enumerate(N_OUTAGES_LIST):
        ax      = axes[idx]
        windows = get_windows(n_outages)
        total_blocked = n_outages * OUTAGE_DURATION

        print(f'\nRunning {n_outages}x {OUTAGE_DURATION:.0f}s outages...')
        for w in windows:
            print(f'  {w[0]:.0f}s → {w[1]:.0f}s')

        ekf_multi   = run_ekf_multiple_outages(
            accel, gyro, timestamps, gt_pos, windows
        )
        multi_error = compute_error(ekf_multi, gt_pos)
        metrics     = compute_metrics(multi_error, timestamps, windows)
        all_metrics[n_outages] = metrics

        single_start = total_duration / 2 - total_blocked / 2
        single_end   = single_start + total_blocked
        ekf_single   = run_ekf_multiple_outages(
            accel, gyro, timestamps, gt_pos,
            [(single_start, single_end)]
        )
        single_error = compute_error(ekf_single, gt_pos)

        n = min(len(timestamps), len(multi_error),
                len(base_error), len(single_error))
        t = timestamps[:n]

        ax.plot(t, base_error[:n],   color='black',  linewidth=1.5,
                linestyle='--', alpha=0.6, label='EKF (no outage)')
        ax.plot(t, single_error[:n], color='orange', linewidth=1.5,
                linestyle='--', alpha=0.8,
                label=f'Single {total_blocked:.0f}s outage')
        ax.plot(t, multi_error[:n],  color='blue',   linewidth=2,
                label=f'{n_outages}× {OUTAGE_DURATION:.0f}s outages')

        ax.axhline(ACCEPTABLE_LIMIT, color='red', linestyle='--',
                   linewidth=1.5, label=f'{ACCEPTABLE_LIMIT}m limit')

        first = True
        for start, end in windows:
            ax.axvspan(start, end, alpha=0.15, color='blue',
                       label='GPS blocked' if first else '')
            first = False

        ax.axvspan(single_start, single_end, alpha=0.06, color='orange')

        rec = metrics['avg_recovery']
        rec_str = f'{rec:.1f}s' if rec != float('inf') else 'No recovery'
        trend = ''
        if metrics['drift_increasing'] is True:
            trend = '↑ compounding'
        elif metrics['drift_increasing'] is False:
            trend = '↓ recovering'

        ax.set_title(
            f'{n_outages}× {OUTAGE_DURATION:.0f}s outages  '
            f'({total_blocked:.0f}s total blocked)\n'
            f'Max: {metrics["max_error"]:.1f}m  '
            f'Recovery: {rec_str}  '
            f'Time >{ACCEPTABLE_LIMIT}m: {metrics["time_above"]:.1f}s  '
            f'{trend}',
            fontsize=9, fontweight='bold'
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position error (m)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, total_duration])

    plt.suptitle(
        f'Multiple Short GPS Outages vs Single Equivalent — {args.data}\n'
        f'Blue = multiple {OUTAGE_DURATION:.0f}s outages   '
        f'Orange = single equivalent   '
        f'Red dashed = {ACCEPTABLE_LIMIT}m limit',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved subplot figure to {output_path}')
    plt.show()

    # ── Figure 2: Bar chart — time above 5m ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    times_above  = [all_metrics[n]['time_above'] for n in N_OUTAGES_LIST]
    x            = np.arange(len(N_OUTAGES_LIST))
    bar_colors   = ['#185FA5', '#1D9E75', '#D85A30', '#7F77DD']
    bars         = ax2.bar(x, times_above, width=0.5,
                           color=bar_colors, alpha=0.85)

    # Value labels on bars
    for bar, val in zip(bars, times_above):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.15,
                 f'{val:.1f}s',
                 ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    # Average increase per outage
    avg_increase = (times_above[-1] - times_above[0]) / (len(N_OUTAGES_LIST) - 1)

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f'{n}× {OUTAGE_DURATION:.0f}s\noutages' for n in N_OUTAGES_LIST],
        fontsize=11
    )
    ax2.set_xlabel('Number of outages', fontsize=12)
    ax2.set_ylabel('Total time above 5m acceptable limit (s)', fontsize=12)
    ax2.set_title(
        f'Time Above {ACCEPTABLE_LIMIT}m Acceptable Limit — {args.data}\n'
        f'Each additional outage adds ~{avg_increase:.1f}s of unacceptable localization',
        fontsize=12, fontweight='bold'
    )
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(times_above) * 1.25)

    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f'Saved bar chart to {bar_path}')
    plt.show()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'\n{"="*85}')
    print(f'{args.data} — Multiple Outage Metrics  '
          f'(buffer={BUFFER}s  duration={OUTAGE_DURATION}s  gap={GAP}s)')
    print(f'{"="*85}')
    print(f'{"N":>4} | {"Windows":>35} | {"Max":>8} | '
          f'{"RMSE":>8} | {"Time>5m":>8} | {"Recovery":>10} | {"Trend":>12}')
    print('-' * 85)

    for n_outages in N_OUTAGES_LIST:
        m       = all_metrics[n_outages]
        windows = get_windows(n_outages)
        win_str = '  '.join([f'{s:.0f}-{e:.0f}s' for s, e in windows])
        rec     = f'{m["avg_recovery"]:.1f}s' \
                  if m['avg_recovery'] != float('inf') else 'None'
        trend   = 'Compounding' if m['drift_increasing'] else 'Recovering'
        print(f'{n_outages:>4} | {win_str:>35} | '
              f'{m["max_error"]:>6.1f}m | {m["overall_rmse"]:>6.1f}m | '
              f'{m["time_above"]:>6.1f}s | {rec:>10} | {trend:>12}')

    print(f'\n--- Cumulative drift per window ---')
    for n_outages in N_OUTAGES_LIST:
        m      = all_metrics[n_outages]
        drifts = ' → '.join([f'{d:.1f}m' for d in m['cumulative_drift']])
        print(f'{n_outages}x: {drifts}')


if __name__ == '__main__':
    main()
