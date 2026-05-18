"""
Experiment runner for MIA Project 1 - Group 23
===============================================
Research question:
    What is the effect of a varying learning rate in the gradient ascent algorithm
    instead of a fixed one, on the performance of intermodal intensity-based
    registration, using mutual information as a similarity measure?
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from registration_project import varying_learning_rate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR    = '../data/image_data'
RESULTS_DIR = 'results'
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')

NUM_ITER    = 150           # gradient ascent iterations
NUM_BINS    = 16            # joint histogram bins
FIXED_LR    = 1e-3          # constant learning rate
MIN_LR      = 1e-5          # floor for adaptive LR
PLATEAU_LEN = 10            # early stopping: look-back window
PLATEAU_TOL = 0.001         # early stopping: max change to call plateau

# 3 patients x 3 slices
PATIENTS = [1, 2, 3]
SLICES   = [1, 2, 3]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pair(patient: int, slice_idx: int):
    """Load a T1 (fixed) and T2 (moving) image pair."""
    t1_path = os.path.join(DATA_DIR, f'{patient}_{slice_idx}_t1.tif')
    t2_path = os.path.join(DATA_DIR, f'{patient}_{slice_idx}_t2.tif')
    I  = plt.imread(t1_path).astype(float)
    Im = plt.imread(t2_path).astype(float)
    return I, Im


def affine_mi_scalar(I, Im, x):
    """Wrapper: returns only the MI scalar (for ngradient)."""
    return reg.affine_mi(I, Im, x, return_transform=False)


def run_registration(I, Im, use_varying_lr: bool):
    
    x = np.array([0., 1., 1., 0., 0., 0., 0.])
    fun = lambda x: affine_mi_scalar(I, Im, x)

    similarity  = np.full(NUM_ITER, np.nan)
    lr_history  = []
    mu          = FIXED_LR
    mu_memory   = {}
    converged_at = None

    for k in range(NUM_ITER):
        g  = reg.ngradient(fun, x)
        x += g * mu

        S, _, _ = reg.affine_mi(I, Im, x, return_transform=True)
        similarity[k] = S
        lr_history.append(mu)

        # Adaptive LR update
        if use_varying_lr:
            if k >= 2:
                sim_valid = similarity[:k+1][~np.isnan(similarity[:k+1])]
                sim_slice = [[v] for v in sim_valid]
                mu, mu_memory = varying_learning_rate(mu, sim_slice, mu_memory, k)
        # else: mu stays FIXED_LR

        # Early stopping
        if k >= PLATEAU_LEN:
            recent = similarity[k - PLATEAU_LEN + 1: k + 1]
            changes = np.abs(np.diff(recent))
            if np.max(changes) < PLATEAU_TOL:
                converged_at = k
                print(f'    Early stop at iteration {k}  (MI={S:.4f})')
                break

    return similarity, lr_history, converged_at, x


def save_learning_curve(similarity_fixed, similarity_varying,
                        patient, slice_idx, converged_fixed, converged_varying):
    """Save a side-by-side learning curve figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    iters = np.arange(1, NUM_ITER + 1)

    for ax, sim, label, conv in zip(
        axes,
        [similarity_fixed, similarity_varying],
        ['Fixed LR', 'Varying LR'],
        [converged_fixed, converged_varying]
    ):
        ax.plot(iters, sim, lw=2, label=label)
        if conv is not None:
            ax.axvline(conv, color='red', linestyle='--', alpha=0.7,
                       label=f'Early stop @ {conv}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mutual Information')
        ax.set_title(f'{label}  —  Patient {patient}, Slice {slice_idx}')
        ax.set_xlim(0, NUM_ITER)
        ax.set_ylim(0, 2)
        ax.grid(True, alpha=0.4)
        ax.legend()

    fig.tight_layout()
    fname = os.path.join(PLOTS_DIR, f'p{patient}_s{slice_idx}_learning_curves.png')
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # CSV writers
    detail_path  = os.path.join(RESULTS_DIR, 'experiment_results.csv')
    summary_path = os.path.join(RESULTS_DIR, 'summary.csv')

    detail_rows  = []   # one row per (pair, condition, iteration)
    summary_rows = []   # one row per (pair, condition)

    pairs_done = 0

    for patient in PATIENTS:
        for slice_idx in SLICES:
            pair_label = f'P{patient}_S{slice_idx}'
            print(f'\n[{pair_label}] Loading images...')

            try:
                I, Im = load_pair(patient, slice_idx)
            except FileNotFoundError as e:
                print(f'  WARNING: {e} — skipping pair.')
                continue

            for condition, use_varying in [('fixed', False), ('varying', True)]:
                print(f'  Running {condition} LR...')

                sim, lrs, conv, x_final = run_registration(I, Im, use_varying)

                final_mi = float(np.nanmax(sim))
                final_iter = int(np.nanargmax(sim)) + 1  # 1-indexed

                # Store detail rows
                for k, (s, lr) in enumerate(zip(sim, lrs)):
                    if not np.isnan(s):
                        detail_rows.append({
                            'pair':      pair_label,
                            'patient':   patient,
                            'slice':     slice_idx,
                            'condition': condition,
                            'iteration': k + 1,
                            'MI':        round(float(s), 6),
                            'lr':        round(float(lr), 8),
                        })

                # Store summary row
                summary_rows.append({
                    'pair':            pair_label,
                    'patient':         patient,
                    'slice':           slice_idx,
                    'condition':       condition,
                    'final_MI':        round(final_mi, 6),
                    'best_iteration':  final_iter,
                    'converged_at':    conv if conv is not None else 'N/A',
                    'theta_rad':       round(float(x_final[0]), 6),
                    'sx':              round(float(x_final[1]), 6),
                    'sy':              round(float(x_final[2]), 6),
                    'shx':             round(float(x_final[3]), 6),
                    'shy':             round(float(x_final[4]), 6),
                    'tx':              round(float(x_final[5]), 6),
                    'ty':              round(float(x_final[6]), 6),
                })

            # Learning curves (both conditions on same figure)
            sim_fixed   = np.array([r['MI'] if r['condition'] == 'fixed'
                                    else np.nan
                                    for r in detail_rows
                                    if r['pair'] == pair_label] +
                                   [np.nan] * NUM_ITER)[:NUM_ITER]
            sim_varying = np.array([r['MI'] if r['condition'] == 'varying'
                                    else np.nan
                                    for r in detail_rows
                                    if r['pair'] == pair_label] +
                                   [np.nan] * NUM_ITER)[:NUM_ITER]

            # Rebuild correctly per condition for the plot
            def get_sim(cond):
                out = np.full(NUM_ITER, np.nan)
                for r in detail_rows:
                    if r['pair'] == pair_label and r['condition'] == cond:
                        out[r['iteration'] - 1] = r['MI']
                return out

            conv_f = next((r['converged_at'] for r in summary_rows
                           if r['pair'] == pair_label and r['condition'] == 'fixed'), None)
            conv_v = next((r['converged_at'] for r in summary_rows
                           if r['pair'] == pair_label and r['condition'] == 'varying'), None)

            save_learning_curve(get_sim('fixed'), get_sim('varying'),
                                patient, slice_idx, conv_f, conv_v)

            pairs_done += 1
            print(f'  Done. Plot saved.')

    # Write CSVs
    if detail_rows:
        with open(detail_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f'\nDetailed results saved to: {detail_path}')

    if summary_rows:
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'Summary saved to:          {summary_path}')

    # Print summary table to console
    print('\n' + '=' * 72)
    print(f'{"Pair":<10} {"Condition":<10} {"Final MI":>10} {"Best iter":>10} {"Conv. at":>10}')
    print('-' * 72)
    for r in summary_rows:
        print(f'{r["pair"]:<10} {r["condition"]:<10} {r["final_MI"]:>10.4f} '
              f'{r["best_iteration"]:>10} {str(r["converged_at"]):>10}')
    print('=' * 72)
    print(f'\nExperiment complete. {pairs_done} pairs processed.')


if __name__ == '__main__':
    main()