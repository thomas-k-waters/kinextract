"""Diagnostic: does starting the optimizer near the TRUE answer let it
converge there and stay, with comparable-or-better chi2 than the wrong
answer it normally finds? Distinguishes "bad starting point, multi-start
would fix it" from "genuinely lower chi2 at the wrong answer, a harder
identifiability problem multi-start alone won't fix."
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from harness_emiles import build_emiles_mock, write_spec_file  # noqa: E402

from kinextract import FitConfig, run_spectral_fit, set_verbose  # noqa: E402
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite  # noqa: E402
from kinextract.joint import (  # noqa: E402
    build_initial_guess, normalize_template_matrix, difference_penalty_matrix,
    build_jax_objective_value_and_grad, objective_joint, DEFAULT_XLAM_CONT,
)

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"

TRUE_V, TRUE_SIGMA, SEED = 80.0, 350.0, 42
VRANGE, SIGL = 1225.0, 350.0

tmpdir = Path(tempfile.mkdtemp(prefix="kinextract_multistart_"))
gal, errs = build_emiles_mock(TRUE_V, TRUE_SIGMA, SEED)
spec_path = tmpdir / "mock.spec"
write_spec_file(gal, errs, spec_path)

cfg = FitConfig(
    template_list_file=EMILES_TLIST, template_dir=EMILES_DIR,
    wavemin_full=4750.0, step=1.25, wavefitmin=8415.0, wavefitmax=8750.0,
    zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
    losvd_vmin=-VRANGE, losvd_vmax=VRANGE, sigl=SIGL, clean=False,
    map_maxiter=10000, print_every=999999,
)
fit = run_spectral_fit(cfg, gal_file=str(spec_path))
st = fit["state"]
gh_normal = fit_losvd_gauss_hermite(st.xl, fit["outputs"]["b"], fit_h3h4=True)
print(f"Normal fit (uniform start): V={gh_normal['vherm']:+.2f} sigma={gh_normal['sherm']:.2f} "
      f"chi2_red={fit['outputs']['chi2_red']:.4f}")

# ---- Now rebuild the SAME optimization but with a custom, "cheat" initial guess ----
n_interior_knots, degree = cfg.joint_n_interior_knots, cfg.joint_degree
fit_continuum = cfg.fit_continuum
xlam_cont = cfg.joint_xlam_cont
cont_diff_order = cfg.joint_cont_diff_order
w_bounds = (1e-5, 1.0)

x0, bounds, xscale, design = build_initial_guess(
    st, n_interior_knots, degree, fit_continuum=fit_continuum, w_bounds=w_bounds)

nl, nt = st.nl, st.nt
b_true = np.exp(-0.5 * ((st.xl - TRUE_V) / TRUE_SIGMA) ** 2)
b_true = np.clip(b_true, 1e-6, None)
b_true = b_true / b_true.sum()  # match the uniform b0's own sum convention

x0_cheat = x0.copy()
x0_cheat[:nl] = b_true

t_norm, _ = normalize_template_matrix(st)
u0_cheat = x0_cheat / xscale
u_bounds = [(lo / s if np.isfinite(lo) else lo, hi / s if np.isfinite(hi) else hi)
            for (lo, hi), s in zip(bounds, xscale)]

n_coef = design.shape[1]
if n_coef > 0:
    D = difference_penalty_matrix(n_coef, cont_diff_order)
    cont0 = x0[-n_coef:]
    coeff_scale = float(np.maximum(np.median(np.abs(cont0)), 1e-12))
else:
    D = None
    coeff_scale = 1.0

value_and_grad = build_jax_objective_value_and_grad(st, design, t_norm, D, coeff_scale, xlam_cont)

def _obj_jac(u):
    a = u * xscale
    val, grad = value_and_grad(a, st.g, st.gerr)
    return val, grad * xscale

result_cheat = minimize(
    _obj_jac, u0_cheat, method="L-BFGS-B", bounds=u_bounds, jac=True,
    options={"maxiter": 10000, "maxfun": 200000, "ftol": 1e-12},
)
a_cheat = result_cheat.x * xscale
b_cheat = a_cheat[:nl]
gh_cheat = fit_losvd_gauss_hermite(st.xl, b_cheat, fit_h3h4=True)

print(f"Cheat-start fit (init at TRUE Gaussian): V={gh_cheat['vherm']:+.2f} "
      f"sigma={gh_cheat['sherm']:.2f}  final_objective={result_cheat.fun:.4f}")

# Compare against the normal fit's own final objective value for an apples-to-apples
# chi2-like comparison (both share the SAME st/design/objective function).
u0_normal = x0 / xscale
val_normal, _ = value_and_grad(u0_normal * xscale, st.g, st.gerr)
a_normal_full = np.concatenate([fit["outputs"]["b"], fit["outputs"]["w"]])
# Reconstruct the *actual* normal-fit optimum's objective value for comparison:
a_full = np.zeros_like(x0)
a_full[:nl] = fit["outputs"]["b"]
a_full[nl:nl+nt] = fit["outputs"]["w"]
if n_coef > 0:
    a_full[nl+nt:] = fit["outputs"].get("continuum_coeffs", x0[nl+nt:])
val_at_normal_answer, _ = value_and_grad(a_full, st.g, st.gerr)

print(f"\nObjective value at the NORMAL (uniform-start) answer: {val_at_normal_answer:.4f}")
print(f"Objective value at the CHEAT-start answer:             {result_cheat.fun:.4f}")
print("(lower is better; if cheat-start's value is similar or lower, the true answer")
print(" is a real, comparably-good local optimum the optimizer just isn't finding from")
print(" a uniform start -- multi-start would fix it. If cheat-start's value is much")
print(" HIGHER, the wrong answer genuinely fits the data better -- a harder problem.)")
