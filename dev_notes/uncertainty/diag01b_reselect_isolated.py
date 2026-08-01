"""Rerun of diag01's ensemble B with the single-start confound fixed
(force_single_start=True on the reselect variant, matching
bootstrap_losvd's own replicate_single_start=True default) -- isolates
ONLY the lam-pin-vs-reselect difference from ensemble A.

Previously recorded (diag01 output, unaffected by this fix -- A used the
real bootstrap_losvd as shipped; C never used single-start, matching what
a real point-estimate fit actually looks like):
  sigma=50, S/N=60:  A (pinned)   sherm_std=4.119  |  C (true) sherm_std=3.178
  sigma=90, S/N=60:  A (pinned)   sherm_std=4.756  |  C (true) sherm_std=3.492
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from kinextract import FitConfig, build_template_stack, gauss_hermite_model, load_emiles_templates, whiten
from diag01_lam_lottery_vs_bootstrap import build, reselect_bootstrap, N_ENSEMBLE

PREV_A = {50.0: 4.119, 90.0: 4.756}
PREV_C = {50.0: 3.178, 90.0: 3.492}

for sigma, snr in [(50., 60.), (90., 60.)]:
    cfg, vg, A, g0, wave = build(sigma, snr)
    nl = np.median(g0) / snr
    rng0 = np.random.default_rng(0)
    gal0 = g0 + rng0.normal(0, nl, len(wave))
    A_w, y_w, _ = whiten(A, gal0, np.full(len(wave), nl), rho=0.0)

    print(f"sigma={sigma:g}, S/N={snr:g}: running B (~7s/rep x {N_ENSEMBLE} reps)...", flush=True)
    sherm_B, h4_B, lams_B = reselect_bootstrap(A_w, y_w, cfg, N_ENSEMBLE, seed=2, force_single_start=True)
    print(f"  A (pinned, single-start, shipped)  : sherm_std={PREV_A[sigma]:.3f}", flush=True)
    print(f"  B (RESELECT, single-start, isolated): sherm_std={sherm_B.std(ddof=1):.3f}  "
          f"h4_std={h4_B.std(ddof=1):.4f}  lam unique={sorted(set(lams_B.tolist()))}", flush=True)
    print(f"  C (true sampling dist, multi-start) : sherm_std={PREV_C[sigma]:.3f}", flush=True)
    print(flush=True)
