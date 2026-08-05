"""Checkpointed build of an h3-wing calibration table, one sigma row at a
time -- mirrors `build_meff_incremental.py`'s own checkpointing pattern
(see that script's docstring for why: a background run that gets killed
partway through loses at most one sigma row's worth of compute, not the
whole run, since per-cell cost scales heavily with sigma).

`calibrate_bias.run_calibration_sweep` has supported `h3_true` since
before this script existed (see
`dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md`'s 2026-08-01
entry) but nothing had ever actually swept it -- the h4-wing table
(`regularization_bias_emiles_h4wing.txt`) characterizes bias for a
genuinely non-Gaussian (heavy/light-tailed) truth, but there was no
equivalent for a genuinely *asymmetric* (skewed) truth. Reuses
`build_emiles_calibration.py`'s own machinery
(`_load_templates_and_weights`, `run_parallel`, `_header`,
`H3_SIGMA_GRID`/`H3_SNR_GRID`/`H3_TRUE_SKEW`) unchanged -- this only
restructures the outer loop to checkpoint per sigma, matching
`build_meff_incremental.py` exactly.

Resumable: if `h3_checkpoints/sigma_{S}.csv` already exists for a given
sigma, that sigma is skipped and its rows are loaded from disk instead of
recomputed.

Once this finishes, wiring the resulting table into `bias_correction.py`
(matching how `h4wing` is already wired in) is a separate, not-yet-done
step -- see `dev_notes/V1_HARDENING_ROADMAP.md` Phase 4.
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from build_emiles_calibration import (  # noqa: E402
    H3_SIGMA_GRID, H3_SNR_GRID, H3_TRUE_SKEW, N_REPS,
    CALIBRATION_DATA_DIR, _load_templates_and_weights, _header, run_parallel,
)
from kinextract import save_calibration_table  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(__file__), "h3_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

ROW_FIELDS = ["sigma", "snr", "n", "v_bias_mean", "v_bias_std", "sig_bias_mean",
              "sig_bias_std", "h4_mean", "h4_std", "h3_mean", "h3_std", "lam",
              "m_eff_mean", "m_eff_std"]


def _ckpt_path(sigma):
    return os.path.join(CKPT_DIR, f"sigma_{sigma}.csv")


def _save_ckpt(sigma, rows):
    path = _ckpt_path(sigma)
    fields = [f for f in ROW_FIELDS if f in rows[0]] if rows else ROW_FIELDS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"  checkpointed sigma={sigma}: {len(rows)} rows -> {path}", flush=True)


def _load_ckpt(sigma):
    path = _ckpt_path(sigma)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            r[k] = int(float(v)) if k == "n" else float(v)
    return rows


def main():
    templates, _, _ = _load_templates_and_weights()
    n_templates = len(templates)
    print(f"loaded {n_templates} E-MILES templates; sigma_grid={H3_SIGMA_GRID}, "
          f"snr_grid={H3_SNR_GRID}, h3_true={H3_TRUE_SKEW}, n_reps={N_REPS}", flush=True)

    all_rows = []
    for sigma in H3_SIGMA_GRID:
        if os.path.exists(_ckpt_path(sigma)):
            rows = _load_ckpt(sigma)
            print(f"[sigma={sigma}] resumed from checkpoint: {len(rows)} rows", flush=True)
        else:
            t0 = time.time()
            print(f"[sigma={sigma}] running {len(H3_SNR_GRID)} S/N cells x "
                  f"{N_REPS} reps...", flush=True)
            rows = run_parallel([sigma], H3_SNR_GRID, lam_grid=None, h4_true=0.0,
                                h3_true=H3_TRUE_SKEW, n_reps=N_REPS, label=f"h3-sigma{sigma}")
            print(f"[sigma={sigma}] done in {time.time()-t0:.0f}s, {len(rows)} rows", flush=True)
            _save_ckpt(sigma, rows)
        all_rows.extend(rows)

    header = _header("h3-wing (genuine asymmetric-LOSVD truth) table", n_templates,
                      h3_true=H3_TRUE_SKEW)
    out_path = os.path.join(CALIBRATION_DATA_DIR, "regularization_bias_emiles_h3wing.txt")
    save_calibration_table(all_rows, out_path, header_comment=header)
    print(f"\nwrote {len(all_rows)} total rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
