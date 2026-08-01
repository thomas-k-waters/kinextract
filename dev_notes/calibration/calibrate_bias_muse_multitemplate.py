"""Builds kinextract's shipped regularization-bias calibration
against the REAL MUSE template library, with ONE fixed representative
Dirichlet population mixture (`WEIGHT_SEED`) reused across every
replicate.

A fresh-random-mixture-per-*calibration*-replicate variant was tried and
directly rejected: checked against known synthetic truth
(`validate_bias_correction_against_truth.py`), it improved only 88% of
cells (median corrected error 4.45 km/s) vs. this fixed-mixture approach's
98% (median 4.79 km/s), and made several sigma=80-100 cells measurably
worse. Resampling the mixture every calibration replicate adds a real
noise source to the calibration's own mean-bias estimate at a fixed
`n_reps` without a compensating increase in effective sample size -- the
validation side already tests generalization across mixtures (a fresh
draw at *application* time, regardless of which calibration is used), so
varying the calibration's own training mixture bought variance, not
robustness. See `kinextract.bias_correction`'s own module docstring
and `VALIDATION_CAMPAIGN.md` Stage 9 for the full history (this also
superseded an even earlier idealized single-template calibration, which
failed on real NGC 5102 data entirely).

Uses the same 32/33-template MUSE library
(sco_framework_updated/.../muse_templates/old) as fit_all_bins.py and this
session's other real/synthetic comparisons. `fit_losvd` (called inside
`run_calibration_sweep`) always freely estimates weights via NNLS
regardless of what generated the mock -- only the *mock's own* true
mixture is fixed here, not the fitting process.

Usage: python calibrate_bias_muse_multitemplate.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from kinextract import run_calibration_sweep, save_calibration_table

TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WEIGHT_SEED = 10000  # fixed Dirichlet mixture, reused every replicate -- see module docstring for why

SIGMA_GRID = [20, 30, 50, 80, 140, 220, 300, 400]
SNR_GRID = [10, 15, 25, 40, 60, 100, 150, 200]
N_REPS = 20

if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else "regularization_bias_muse_multitemplate.txt"

    with open(os.path.join(TEMPLATE_DIR, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    wavelength = 4750.0 + np.arange(3681) * 1.25
    templates = []
    for name in names:
        d = np.loadtxt(os.path.join(TEMPLATE_DIR, name))
        templates.append(np.interp(wavelength, d[:, 0], d[:, 1]))
    n_templates = len(templates)
    print(f"loaded {n_templates} real MUSE templates from {TEMPLATE_DIR}")

    rng = np.random.default_rng(WEIGHT_SEED)
    weights = rng.dirichlet(np.ones(n_templates))
    print(f"fixed Dirichlet mixture weights (seed={WEIGHT_SEED}): {np.round(weights, 3)}")

    t0 = time.time()
    rows = run_calibration_sweep(
        templates, wavelength, weights,
        sigma_grid=SIGMA_GRID, snr_grid=SNR_GRID, n_reps=N_REPS, verbose=True,
    )
    save_calibration_table(
        rows, outfile,
        header_comment=(
            f"Real MUSE templates ({n_templates}), fixed Dirichlet mixture "
            f"(seed={WEIGHT_SEED}), V=0, n_reps={N_REPS} -- see calibrate_bias.py's "
            f"module docstring and VALIDATION_CAMPAIGN.md Stage 9."
        ),
    )
    print(f"\nDone in {time.time()-t0:.0f}s. Saved {len(rows)} cells to {outfile}")
