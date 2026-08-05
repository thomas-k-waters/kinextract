"""Builds kinextract's regularization-bias calibration tables against
the shipped E-MILES SSP grid, replacing the previous default (a fixed
mixture of 33 REAL MUSE stellar templates, `regularization_bias_muse_
multitemplate.txt`).

**Why this replaces the MUSE-star default.** The real-star calibration was
tied to one specific, external, non-portable template library at a
hardcoded path -- correct for real MUSE fits *specifically*, but silently
wrong to apply as the *package-wide* default for anyone using a different
instrument/template library (e.g. HST/STIS): nothing in `bias_correction`
warned that the correction's own training data was MUSE-star-shaped, so a
STIS user would get a confidently-wrong correction with no signal
anything was off (found directly: this same correction, applied to this
package's own single-template synthetic-mock notebook, made sigma
recovery ~10x *worse* in 24/25 seeds -- a template-shape mismatch, not a
code bug, confirmed by a real-MUSE-star positive control that DID show the
expected improvement).

E-MILES is a synthetic, parametrized (age x metallicity), portable,
LSF-matchable SSP grid (`kinextract.emiles`) -- a reproducible basis
any instrument's LSF can be matched to, not tied to one specific
real-star observation. This script is also the reference example for
building your OWN calibration against your OWN template library (STIS,
a different MUSE template set, anything `build_template_stack` accepts):
swap `_load_templates_and_weights` for your own loader and rerun. The
underlying mechanism (`calibrate_bias.run_calibration_sweep`) was already
fully template-library-agnostic before this script existed -- what
changes here is which templates the *shipped default* is built from, not
the mechanism itself.

Builds three tables, all sharing the same E-MILES template subset and
fixed Dirichlet mixture (`WEIGHT_SEED`) for direct comparability:

1. Main (sigma, S/N) table, h4_true=0.0, lam=2.0 fixed --
   `regularization_bias_emiles_multitemplate.txt`, replaces
   `regularization_bias_muse_multitemplate.txt` as the shipped default.
2. h4-wing table, h4_true=0.15 (genuine non-Gaussian wing truth) --
   `regularization_bias_emiles_h4wing.txt`, replaces
   `regularization_bias_h4wing_muse.txt`.
3. m_eff-axis table, `lam_grid` swept (same grid `select_lambda_aic_local`
   itself scans) -- `regularization_bias_emiles_multitemplate_meff.txt`,
   replaces `regularization_bias_muse_multitemplate_meff.txt`. Needed
   before the correction can be validly applied to an adaptively-
   regularized fit.

Parallelized across (sigma, S/N) cells via `ProcessPoolExecutor` -- a
module-level worker function is required for this to pickle correctly for
spawned worker processes (macOS `spawn`; a notebook/script-local closure
cannot be pickled), matching the pattern already validated in
`calibrate_bias_muse_multitemplate_meff.py` and `bias_sweep_fixed_vs_
adaptive.py`.

Usage: python build_emiles_calibration.py [--only main|h4wing|meff] [--n-reps N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from kinextract import run_calibration_sweep, save_calibration_table, load_emiles_templates

EMILES_GRID = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "kinextract",
    "template_data", "emiles_muse_grid.npz",
)
CALIBRATION_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "kinextract", "calibration_data",
)

# A broad, representative age/metallicity subset -- comparable in scale
# (40 templates) to the real-star default's 33, spanning young-to-old and
# metal-poor-to-metal-rich rather than the full 636-template native grid
# (which would make the *mock's own generative mixture* implausibly
# diffuse/degenerate for calibration purposes -- neighboring E-MILES
# templates are strongly correlated in the CaII fit window, see
# 02_realistic_mock_fit.ipynb's own condition-number diagnostic).
AGES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 14.0]
METALS = [-1.49, -0.66, 0.06, 0.40]
WEIGHT_SEED = 30000  # new seed for the EMILES-basis calibration (not reused from the MUSE-star tables)

SIGMA_GRID = [20, 30, 50, 80, 140, 220, 300, 400]
SNR_GRID = [10, 15, 25, 40, 60, 100, 150, 200]
N_REPS = 30  # tighter than the MUSE-star default's 20 -- overnight run, cost is not the constraint here

H4WING_SIGMA_GRID = [30, 50, 80, 140]
H4WING_SNR_GRID = [25, 40, 60, 100, 150]
H4_TRUE_WING = 0.15

H3_SIGMA_GRID = [30, 50, 80, 140]
H3_SNR_GRID = [25, 40, 60, 100, 150]
H3_TRUE_SKEW = 0.15  # matches H4_TRUE_WING's magnitude, for the same reason: a genuine,
# moderate, physically-plausible asymmetric-LOSVD truth, not an extreme stress value

LAM_GRID_FOR_MEFF = [1.0, 8.0, 64.0, 512.0, 4096.0, 32768.0]
# Matches FitConfig()'s current order-3 default lam grid (select_lambda_aic_local's own
# candidate set) exactly. Stale as of 2026-07-28: the previous values [0.0, 0.5, 1.0, 2.0,
# 5.0, 10.0, 20.0, 50.0] matched the OLD order-2 default grid's scale (see
# FitConfig.resolved_lam_grid's own order-2/order-3 branches) -- built before penalty_order
# flipped to 3 as the package default (VALIDATION_CAMPAIGN.md Stage 22). A calibration table
# built at order-2-scale lam values is not usable for characterizing bias at the lam values
# adaptive selection actually considers under the current order-3 default; this must match
# whatever FitConfig()'s own default grid is, not be chosen independently.


def _load_templates_and_weights():
    """Reloads per worker call (cheap relative to the fitting cost, and
    avoids pickling large arrays through the process pool for every task)
    -- swap this function alone to calibrate against a different template
    library."""
    select = [(a, z) for a in AGES for z in METALS]
    templates, wavelength, names, ages, metals = load_emiles_templates(EMILES_GRID, select=select)
    rng = np.random.default_rng(WEIGHT_SEED)
    weights = rng.dirichlet(np.ones(len(templates)))
    return templates, wavelength, weights


def _run_one_cell(sigma, snr, lam_grid, h4_true, n_reps, h3_true=0.0):
    templates, wavelength, weights = _load_templates_and_weights()
    return run_calibration_sweep(
        templates, wavelength, weights, sigma_grid=[sigma], snr_grid=[snr],
        n_reps=n_reps, lam_grid=lam_grid, h4_true=h4_true, h3_true=h3_true, verbose=False,
    )


def run_parallel(sigma_grid, snr_grid, lam_grid=None, h4_true=0.0, n_reps=N_REPS,
                  max_workers=None, verbose=True, label="", h3_true=0.0):
    max_workers = max_workers or os.cpu_count()
    cells = [(s, n) for s in sigma_grid for n in snr_grid]
    all_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one_cell, s, n, lam_grid, h4_true, n_reps, h3_true): (s, n) for s, n in cells}
        n_done = 0
        for fut in as_completed(futures):
            s, n = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            n_done += 1
            if verbose:
                print(f"[{label}] cell (sigma={s}, snr={n}) done -- "
                      f"{n_done}/{len(cells)} cells, t={time.time()-t0:.0f}s", flush=True)
    print(f"[{label}] total wall time: {time.time()-t0:.0f}s ({max_workers} workers), "
          f"{len(all_rows)} total rows", flush=True)
    return all_rows


def _header(kind: str, n_templates: int, **extra) -> str:
    base = (f"E-MILES ({n_templates} templates, ages {AGES}, [M/H] {METALS}), "
            f"fixed Dirichlet mixture (seed={WEIGHT_SEED}), V=0, n_reps={N_REPS}")
    parts = [f"{k}={v}" for k, v in extra.items()]
    return base + (", " + ", ".join(parts) if parts else "") + (
        f" -- {kind}. Replaces the real-MUSE-star default; see "
        f"build_emiles_calibration.py's own module docstring."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["main", "h4wing", "meff"], default=None)
    ap.add_argument("--n-reps", type=int, default=N_REPS)
    args = ap.parse_args()
    N_REPS = args.n_reps

    templates, _, _ = _load_templates_and_weights()
    n_templates = len(templates)
    print(f"loaded {n_templates} E-MILES templates for calibration "
          f"(ages={AGES}, metals={METALS})", flush=True)

    todo = [args.only] if args.only else ["main", "h4wing", "meff"]

    if "main" in todo:
        print(f"\n=== main table: {len(SIGMA_GRID)}x{len(SNR_GRID)} grid, n_reps={N_REPS}, "
              f"{len(SIGMA_GRID)*len(SNR_GRID)*N_REPS} total fits ===", flush=True)
        rows = run_parallel(SIGMA_GRID, SNR_GRID, lam_grid=None, h4_true=0.0, n_reps=N_REPS, label="main")
        out = os.path.join(CALIBRATION_DATA_DIR, "regularization_bias_emiles_multitemplate.txt")
        save_calibration_table(rows, out, header_comment=_header("h4_true=0.0 main table", n_templates))
        print(f"wrote {len(rows)} rows to {out}", flush=True)

    if "h4wing" in todo:
        print(f"\n=== h4-wing table: {len(H4WING_SIGMA_GRID)}x{len(H4WING_SNR_GRID)} grid, "
              f"n_reps={N_REPS}, h4_true={H4_TRUE_WING}, "
              f"{len(H4WING_SIGMA_GRID)*len(H4WING_SNR_GRID)*N_REPS} total fits ===", flush=True)
        rows = run_parallel(H4WING_SIGMA_GRID, H4WING_SNR_GRID, lam_grid=None,
                             h4_true=H4_TRUE_WING, n_reps=N_REPS, label="h4wing")
        out = os.path.join(CALIBRATION_DATA_DIR, "regularization_bias_emiles_h4wing.txt")
        save_calibration_table(rows, out, header_comment=_header(
            "genuine-wing table (NOT the shipped h4=0 default)", n_templates, h4_true=H4_TRUE_WING))
        print(f"wrote {len(rows)} rows to {out}", flush=True)

    if "meff" in todo:
        total = len(SIGMA_GRID) * len(SNR_GRID) * len(LAM_GRID_FOR_MEFF) * N_REPS
        print(f"\n=== m_eff-axis table: {len(SIGMA_GRID)}x{len(SNR_GRID)} grid, "
              f"{len(LAM_GRID_FOR_MEFF)} lam values, n_reps={N_REPS}, {total} total fits ===", flush=True)
        rows = run_parallel(SIGMA_GRID, SNR_GRID, lam_grid=LAM_GRID_FOR_MEFF, h4_true=0.0,
                             n_reps=N_REPS, label="meff")
        out = os.path.join(CALIBRATION_DATA_DIR, "regularization_bias_emiles_multitemplate_meff.txt")
        save_calibration_table(rows, out, header_comment=_header(
            "m_eff axis for adaptive-regularization use (select_lambda_aic_local)",
            n_templates, lam_grid=LAM_GRID_FOR_MEFF))
        print(f"wrote {len(rows)} rows to {out}", flush=True)

    print("\nDone.", flush=True)
