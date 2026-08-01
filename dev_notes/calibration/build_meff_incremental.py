"""Checkpointed rebuild of the m_eff-axis calibration table, one sigma row
at a time.

Why this exists: the plain `build_emiles_calibration.py --only meff` run
(2026-07-28) only writes to disk once, at the very end, via
`save_calibration_table`. A background run that gets killed by a timeout
(this happened twice tonight, once at 3600s/24 cells and once at
7200s/56-of-64 cells -- observed per-cell cost scales heavily with sigma,
e.g. sigma=300 cells took ~35 min EACH vs. tens of seconds at sigma=20, so
a fixed timeout sized for the average cell badly underestimates the tail)
loses ALL progress, not just the incomplete tail. This script checkpoints
after every sigma value (8 checkpoints instead of 1 final write), so a
timeout or interruption loses at most one sigma row's worth of compute
(worst case ~35 min at sigma=400, not the full ~2+ hours).

Reuses every piece of the existing, already-correct machinery
(`_load_templates_and_weights`, `run_parallel`, `_header`,
`save_calibration_table`) from `build_emiles_calibration.py` unchanged --
this only restructures the outer loop to checkpoint, it does not
reimplement the sweep itself.

Resumable: if `results/meff_checkpoint_sigma{S}.csv` already exists for a
given sigma, that sigma is skipped and its rows are loaded from disk
instead of recomputed.
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from build_emiles_calibration import (  # noqa: E402
    AGES, METALS, WEIGHT_SEED, SIGMA_GRID, SNR_GRID, N_REPS,
    LAM_GRID_FOR_MEFF, CALIBRATION_DATA_DIR, _load_templates_and_weights,
    _header, run_parallel,
)
from kinextract import save_calibration_table  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(__file__), "meff_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

ROW_FIELDS = ["sigma", "snr", "n", "v_bias_mean", "v_bias_std", "sig_bias_mean",
              "sig_bias_std", "h4_mean", "h4_std", "lam", "m_eff_mean", "m_eff_std"]


def _ckpt_path(sigma):
    return os.path.join(CKPT_DIR, f"sigma_{sigma}.csv")


def _save_ckpt(sigma, rows):
    path = _ckpt_path(sigma)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ROW_FIELDS})
    print(f"  checkpointed sigma={sigma}: {len(rows)} rows -> {path}", flush=True)


def _load_ckpt(sigma):
    path = _ckpt_path(sigma)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ROW_FIELDS:
            r[k] = float(r[k]) if k not in ("n",) else int(float(r[k]))
    return rows


def main():
    templates, _, _ = _load_templates_and_weights()
    n_templates = len(templates)
    print(f"loaded {n_templates} E-MILES templates; sigma_grid={SIGMA_GRID}, "
          f"snr_grid={SNR_GRID}, lam_grid={LAM_GRID_FOR_MEFF}, n_reps={N_REPS}", flush=True)

    all_rows = []
    for sigma in SIGMA_GRID:
        if os.path.exists(_ckpt_path(sigma)):
            rows = _load_ckpt(sigma)
            print(f"[sigma={sigma}] resumed from checkpoint: {len(rows)} rows", flush=True)
        else:
            t0 = time.time()
            print(f"[sigma={sigma}] running {len(SNR_GRID)} S/N cells x "
                  f"{len(LAM_GRID_FOR_MEFF)} lam x {N_REPS} reps...", flush=True)
            rows = run_parallel([sigma], SNR_GRID, lam_grid=LAM_GRID_FOR_MEFF,
                                h4_true=0.0, n_reps=N_REPS, label=f"meff-sigma{sigma}")
            print(f"[sigma={sigma}] done in {time.time()-t0:.0f}s, {len(rows)} rows", flush=True)
            _save_ckpt(sigma, rows)
        all_rows.extend(rows)

    header = _header("meff", n_templates, lam_grid=LAM_GRID_FOR_MEFF)
    out_path = os.path.join(CALIBRATION_DATA_DIR, "regularization_bias_emiles_multitemplate_meff.txt")
    save_calibration_table(all_rows, out_path, header_comment=header)
    print(f"\nwrote {len(all_rows)} total rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
