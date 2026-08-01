"""The ablation ladder: quantifies how much bias each individual
kinextract fitting-pipeline mechanism contributes, in isolation and
cumulatively, against real MUSE templates with a fresh random population
mixture per replicate (same realism convention as
`validate_bias_correction_against_truth.py`) and KNOWN true sigma.

Each config in the ladder adds exactly one thing to the previous:

    0 bare_unregularized  -- lam=0 (no curvature penalty), single start,
                             plain trapz moment, no h3/h4, no shape-aware fix
    1 plus_regularization -- + lam=2.0 (current shipped default penalty)
    2 plus_multistart     -- + fit_log_spline_map's own default multi-start
    3 plus_taper_moment   -- + compute_direct_moments's smooth logistic taper
                             (replacing the plain trapz moment)
    4 full_stack_default  -- + h3/h4 estimation (fit_h3_h4) -- THIS IS
                             kinextract's actual shipped default
                             configuration end to end
    5 plus_shape_aware    -- + shape_aware_correction (branch off config 4;
                             opt-in, off by default in the real package)

See `kinextract.ablation`'s own module docstring for the framework
this drives (`AblationConfig`, `run_ablation_sweep`) -- extend the ladder
below (not a bespoke script) when a new feature needs the same treatment.

Usage: python ablation_sweep.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from kinextract import AblationConfig, run_ablation_sweep, save_ablation_table

TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WEIGHT_SEED_BASE = 70000  # disjoint from calibrate_bias's 10000 and validate's 90000

SIGMA_GRID = [20, 30, 50, 80, 140, 220, 300, 400]
SNR_GRID = [10, 15, 25, 40, 60, 100, 150, 200]
N_REPS = 20

LADDER = [
    AblationConfig(name="0_bare_unregularized", lam=0.0, losvd_single_start=True,
                    moment_method="trapz", fit_h3h4=False, shape_aware_correction=False),
    AblationConfig(name="1_plus_regularization", lam=2.0, losvd_single_start=True,
                    moment_method="trapz", fit_h3h4=False, shape_aware_correction=False),
    AblationConfig(name="2_plus_multistart", lam=2.0, losvd_single_start=False,
                    moment_method="trapz", fit_h3h4=False, shape_aware_correction=False),
    AblationConfig(name="3_plus_taper_moment", lam=2.0, losvd_single_start=False,
                    moment_method="taper", fit_h3h4=False, shape_aware_correction=False),
    AblationConfig(name="4_full_stack_default", lam=2.0, losvd_single_start=False,
                    moment_method="taper", fit_h3h4=True, shape_aware_correction=False),
    AblationConfig(name="5_plus_shape_aware", lam=2.0, losvd_single_start=False,
                    moment_method="taper", fit_h3h4=True, shape_aware_correction=True),
]

if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else "ablation_sweep_results.txt"

    with open(os.path.join(TEMPLATE_DIR, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    wavelength = 4750.0 + np.arange(3681) * 1.25
    templates = []
    for name in names:
        d = np.loadtxt(os.path.join(TEMPLATE_DIR, name))
        templates.append(np.interp(wavelength, d[:, 0], d[:, 1]))
    n_templates = len(templates)
    print(f"loaded {n_templates} real MUSE templates from {TEMPLATE_DIR}")
    print(f"ladder: {[c.name for c in LADDER]}")
    print(f"grid: sigma={SIGMA_GRID} x snr={SNR_GRID}, n_reps={N_REPS} "
          f"({len(SIGMA_GRID)*len(SNR_GRID)*len(LADDER)} rows expected)")

    _rng = np.random.default_rng(WEIGHT_SEED_BASE)

    def weight_sampler():
        return _rng.dirichlet(np.ones(n_templates))

    t0 = time.time()
    rows = run_ablation_sweep(
        LADDER, templates, wavelength, weight_sampler,
        sigma_grid=SIGMA_GRID, snr_grid=SNR_GRID, n_reps=N_REPS, verbose=True,
    )
    save_ablation_table(
        rows, outfile,
        header_comment=(
            f"Ablation ladder against real MUSE templates ({n_templates}), fresh Dirichlet "
            f"mixture per replicate (base seed={WEIGHT_SEED_BASE}), V=0, n_reps={N_REPS}. "
            f"Ladder: {[c.name for c in LADDER]}"
        ),
    )
    print(f"\nDone in {time.time()-t0:.0f}s. Saved {len(rows)} rows to {outfile}")
