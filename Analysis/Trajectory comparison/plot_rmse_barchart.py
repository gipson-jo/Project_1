"""
plot_rmse_barchart.py
Plot 2 — RMSE summary bar chart across all four routes.

Hardcoded from trajectory comparison results.
Saves to the same folder this script lives in.

Usage:
    python3 plot_rmse_barchart.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Data from trajectory comparison runs ────────────────────────────────────
routes = ['Straight', 'Winding', 'Circle', 'Urban']

dr_rmse  = [5070.04, 468562.01, 7984.88, 837.56]
ekf_rmse = [4.35,    8.18,      3.79,    4.0]

improvement = [99.9, 100.0, 100.0, 99.5]   # %
distance    = [1297, 4130,  1231,  332]     # metres
duration    = [96.6, 537.8, 114.5, 46.2]   # seconds

# ── Output path ──────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'rmse_barchart.png')

# ── Figure — two subplots ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

x     = np.arange(len(routes))
width = 0.35

# ── Left: DR RMSE only (log scale) ──────────────────────────────────────────
bars1 = ax1.bar(x - width/2, dr_rmse,  width,
                color='red',  alpha=0.8, label='Dead reckoning')
bars2 = ax1.bar(x + width/2, ekf_rmse, width,
                color='blue', alpha=0.8, label='EKF')

ax1.set_yscale('log')
ax1.set_xlabel('Route', fontsize=12)
ax1.set_ylabel('RMSE (m) — log scale', fontsize=12)
ax1.set_title('RMSE Comparison — Log Scale\n'
              '(shows true magnitude difference)',
              fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(routes, fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h * 1.1,
             f'{h:,.0f}m', ha='center', va='bottom', fontsize=8, color='red')
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h * 1.1,
             f'{h:.1f}m', ha='center', va='bottom', fontsize=8, color='blue')

# ── Right: EKF RMSE only (linear scale — readable) ──────────────────────────
bars3 = ax2.bar(routes, ekf_rmse, color='blue', alpha=0.8, width=0.5)

ax2.set_xlabel('Route', fontsize=12)
ax2.set_ylabel('EKF RMSE (m)', fontsize=12)
ax2.set_title('EKF RMSE by Route\n'
              '(linear scale — shows differences between routes)',
              fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels and route info
for bar, route, dist, dur in zip(bars3, routes, distance, duration):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.05,
             f'{h:.2f}m', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='blue')
    ax2.text(bar.get_x() + bar.get_width()/2, -0.6,
             f'{dist}m  ·  {dur:.0f}s',
             ha='center', va='top', fontsize=8, color='gray')

ax2.set_ylim(0, max(ekf_rmse) * 1.3)

plt.suptitle(
    'Plot 2 — RMSE Summary Across Route Types\n'
    'Dead Reckoning vs EKF',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved to {output_path}')
plt.show()

# ── Print summary ────────────────────────────────────────────────────────────
print(f'\n{"Route":>10} | {"DR RMSE":>12} | {"EKF RMSE":>10} | {"Improvement":>12}')
print('-' * 52)
for r, dr, ekf, imp in zip(routes, dr_rmse, ekf_rmse, improvement):
    print(f'{r:>10} | {dr:>10,.1f}m | {ekf:>8.2f}m | {imp:>10.1f}%')
