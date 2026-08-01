"""Builds a regularization-bias calibration table at a GENUINE non-Gaussian
truth (h4_true=0.15), to directly test the hypothesis (found during last
night's code audit) that the shipped default calibration
(`regularization_bias_muse_multitemplate.txt`, built at h4_true=0.0
exclusively -- see `bias_correction.py`'s own docstring) is why the
coverage-calibration test showed catastrophic (0.0%/3.3%) h4 coverage at
sigma=50 under genuine wings: the correction being applied there was for
the wrong scenario entirely, not a weaker/approximate version of the
right one.

Same 33-template MUSE library and the SAME fixed mixture seed (20000) used
in `coverage_calibration_test.py`/`fix_low_sigma_wing_coverage.py`, so this
isolates the h4-domain-match question alone -- no mixture-choice confound.

Targeted grid (not the full shipped-calibration range) since this is a
direct test of one specific hypothesis, not a new shipped default:
sigma x S/N chosen to bracket the failing cell (sigma=50, S/N=40/100) for
clean 2D interpolation.

Usage: python calibrate_bias_h4wing.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from kinextract import run_calibration_sweep, save_calibration_table

TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WEIGHT_SEED = 20000  # matches coverage_calibration_test.py's own MIXTURE_SEED

SIGMA_GRID = [30, 50, 80, 140]
SNR_GRID = [25, 40, 60, 100, 150]
N_REPS = 20
H4_TRUE = 0.15

if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else "regularization_bias_h4wing_muse.txt"

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
    print(f"h4_true={H4_TRUE} (genuine wing, not h4=0)")

    t0 = time.time()
    rows = run_calibration_sweep(
        templates, wavelength, weights,
        sigma_grid=SIGMA_GRID, snr_grid=SNR_GRID, n_reps=N_REPS, h4_true=H4_TRUE,
    )
    save_calibration_table(
        rows, outfile,
        header_comment=(
            f"Real MUSE templates ({n_templates}), fixed Dirichlet mixture "
            f"(seed={WEIGHT_SEED}), h4_true={H4_TRUE} (genuine wing, NOT the shipped h4=0 "
            f"calibration) -- built to test whether an h4-domain-matched correction fixes "
            f"the 0%/3.3% h4 coverage catastrophe found at sigma=50 in coverage_calibration_test.py. "
            f"Total time {time.time()-t0:.0f}s"
        ),
    )
    print(f"wrote {len(rows)} rows to {outfile}, total time {time.time()-t0:.0f}s")
