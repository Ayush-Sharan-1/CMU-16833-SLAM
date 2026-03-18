"""
Parameter sweep for EKF SLAM.
Varies each noise parameter (sig_x, sig_y, sig_alpha, sig_beta, sig_r)
10x larger and 10x smaller than the baseline, fixing the others.
Saves final-state figures to ../figures/.

Figures are identical to what the original ekf_slam.py produces — all
trajectory segments, prediction ellipses, and landmark ellipses are drawn
exactly as in the original, just without button-press pauses.
"""

import matplotlib
matplotlib.use('Agg')  # must be set before importing pyplot

import matplotlib.pyplot as plt
plt.waitforbuttonpress = lambda *_: None   # suppress all pauses
plt.ion = lambda: None                           # suppress interactive mode

import numpy as np
import re
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ekf_slam import (
    init_landmarks, predict, update,
    draw_traj_and_map, draw_traj_and_pred
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_FILE   = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.txt')
L_TRUE      = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)

BASELINE = dict(sig_x=0.25, sig_y=0.1, sig_alpha=0.1, sig_beta=0.01, sig_r=0.08)


def run_and_save(sig_x, sig_y, sig_alpha, sig_beta, sig_r, title, filename):
    control_cov = np.diag([sig_x**2, sig_y**2, sig_alpha**2])
    measure_cov = np.diag([sig_beta**2, sig_r**2])
    pose_cov    = np.diag([0.02**2, 0.02**2, 0.1**2])
    pose        = np.zeros((3, 1))

    data_file = open(DATA_FILE)
    line      = data_file.readline()
    fields    = re.split(r'[\t ]', line)[:-1]
    measure   = np.expand_dims(np.array([float(f) for f in fields]), axis=1)

    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose, pose_cov)

    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov,           np.zeros((3, 2*k))],
                  [np.zeros((2*k, 3)), landmark_cov      ]])

    # --- one figure for the entire run, exactly like the original ---
    fig = plt.figure(figsize=(8, 8))

    last_X = X
    draw_traj_and_map(X, last_X, P, 0)   # t=0: red landmark ellipses

    X_pre, P_pre = X.copy(), P.copy()
    t = 1

    for line in data_file:
        fields = re.split(r'[\t ]', line)[:-1]
        arr    = np.array([float(f) for f in fields])

        if arr.shape[0] == 2:                          # control
            control      = np.array([[arr[0]], [arr[1]]])
            X_pre, P_pre = predict(X, P, control, control_cov, k)
            draw_traj_and_pred(X_pre, P_pre)           # magenta ellipse
        else:                                          # measurement
            measure      = np.expand_dims(arr, axis=1)
            X, P         = update(X_pre, P_pre, measure, measure_cov, k)
            draw_traj_and_map(X, last_X, P, t)         # blue traj + green ellipses
            last_X = X
            t += 1

    data_file.close()

    # ground truth scatter (same as evaluate())
    plt.scatter(L_TRUE[0::2], L_TRUE[1::2], c='k', marker='+', s=120, zorder=5)

    plt.title(title, fontsize=10)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.grid(True, linewidth=0.4)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def sweep():
    # baseline first
    run_and_save(**BASELINE, title='Baseline parameters', filename='baseline.png')

    for param in BASELINE:
        for scale, label in [(0.1, '10x_smaller'), (10.0, '10x_larger')]:
            cfg        = BASELINE.copy()
            cfg[param] = BASELINE[param] * scale
            val_str    = f'{cfg[param]:.4g}'
            title      = f'{param} = {val_str}  ({label} than baseline)'
            filename   = f'{param}_{val_str}.png'

            print(f'Running: {title}')
            run_and_save(**cfg, title=title, filename=filename)


if __name__ == '__main__':
    sweep()
