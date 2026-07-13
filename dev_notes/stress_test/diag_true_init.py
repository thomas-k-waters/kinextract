"""Does starting the optimizer at the EXACT true (b, w, continuum) let it
converge there and stay (near-truth, comparable-or-lower objective), or does
it drift back to the same V-biased answer the normal uniform-start fit
finds? Distinguishes "bad starting point" from "the biased answer is a
genuine (near-)optimum of the current objective" -- the latter means the
fix has to be in the objective/regularization, not initialization.
"""
import sys, tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles import build_emiles_mock, write_spec_file, GEN_WEIGHTS
from harness import WAVEMIN, STEP

from kinextract import FitConfig, run_spectral_fit, set_verbose
set_verbose(False)
from kinextract.losvd import fit_losvd_gauss_hermite
import kinextract.joint as jt

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"

TRUE_V, TRUE_SIGMA, SEED = 80.0, 300.0, 42
VRANGE, SIGL = max(300.0, 3.5 * TRUE_SIGMA), TRUE_SIGMA
W_TRUE_IDX = [9, 19]  # column positions of the 2 true generating SSPs in the 20-template fit grid

tmpdir = Path(tempfile.mkdtemp(prefix="kinextract_trueinit_"))
gal, errs = build_emiles_mock(TRUE_V, TRUE_SIGMA, SEED)
spec_path = tmpdir / "mock.spec"
write_spec_file(gal, errs, spec_path)

cfg = FitConfig(
    template_list_file=EMILES_TLIST, template_dir=EMILES_DIR,
    wavemin_full=WAVEMIN, step=STEP, wavefitmin=8000.0, wavefitmax=9000.0,
    zgal=0.0, fit_continuum=True, use_spectrum_errors=True, xlam_auto=True,
    losvd_vmin=-VRANGE, losvd_vmax=VRANGE, sigl=SIGL, clean=False,
    map_maxiter=10000, print_every=999999,
)

# capture the FINAL (converged xlam/sigl0/v_center) FitState + design/t_norm/etc.
captured = {}
orig_fit_joint = jt.fit_joint
def hook(st, n_interior_knots=10, degree=3, xlam_cont=jt.DEFAULT_XLAM_CONT, cont_diff_order=2,
         maxiter=50000, ftol=1e-12, maxfun=500000, use_jax=True, auto_recenter_v=False,
         fit_continuum=True, w_bounds=(1e-5, 1.0)):
    if auto_recenter_v:
        import dataclasses
        st = dataclasses.replace(st, v_center=jt.estimate_velocity_xcorr(st))
    x0, bounds, xscale, design = jt.build_initial_guess(st, n_interior_knots, degree, fit_continuum=fit_continuum, w_bounds=w_bounds)
    t_norm, _ = jt.normalize_template_matrix(st)
    n_coef = design.shape[1]
    if n_coef > 0:
        D = jt.difference_penalty_matrix(n_coef, cont_diff_order)
        cont0 = x0[-n_coef:]
        coeff_scale = float(np.maximum(np.median(np.abs(cont0)), 1e-12))
    else:
        D = None; coeff_scale = 1.0
    captured["snap"] = dict(st=st, design=design, t_norm=t_norm, D=D, coeff_scale=coeff_scale,
                             xlam_cont=xlam_cont, x0=x0, bounds=bounds, xscale=xscale)
    return orig_fit_joint(st, n_interior_knots, degree, xlam_cont, cont_diff_order, maxiter, ftol,
                           maxfun, use_jax, False, fit_continuum, w_bounds)
jt.fit_joint = hook

fit = run_spectral_fit(cfg, gal_file=str(spec_path))
gh_normal = fit_losvd_gauss_hermite(fit["state"].xl, fit["outputs"]["b"], fit_h3h4=True)
print(f"Normal fit (uniform start): V={gh_normal['vherm']:+.2f} sigma={gh_normal['sherm']:.2f} "
      f"chi2_red={fit['outputs']['chi2_red']:.4f}  (truth: V={TRUE_V} sigma={TRUE_SIGMA})")

snap = captured["snap"]
st, design, t_norm, D, coeff_scale, xlam_cont = snap["st"], snap["design"], snap["t_norm"], snap["D"], snap["coeff_scale"], snap["xlam_cont"]
x0, bounds, xscale = snap["x0"], snap["bounds"], snap["xscale"]
nl, nt = st.nl, st.nt

# ---- build the TRUE x0: exact Gaussian b_true, exact sparse w_true, same continuum init ----
b_true = np.exp(-0.5 * ((st.xl - TRUE_V) / TRUE_SIGMA) ** 2)
b_true = np.clip(b_true, 1e-6, None)
b_true = b_true / b_true.sum()

w_true = np.full(nt, 1e-5)
w_true[W_TRUE_IDX[0]] = GEN_WEIGHTS[0]
w_true[W_TRUE_IDX[1]] = GEN_WEIGHTS[1]

x0_cheat = x0.copy()
x0_cheat[:nl] = b_true
x0_cheat[nl:nl + nt] = w_true

u0_cheat = x0_cheat / xscale
u_bounds = [(lo / s if np.isfinite(lo) else lo, hi / s if np.isfinite(hi) else hi)
            for (lo, hi), s in zip(bounds, xscale)]

value_and_grad = jt._get_or_build_jax_vg_joint(st, design, t_norm, D, xlam_cont)

def _obj_jac(u):
    a = u * xscale
    val, grad = value_and_grad(a, st.g, st.gerr, coeff_scale)
    return val, grad * xscale

# sanity check: what's the objective value AT the true parameters themselves?
val_at_truth, _ = _obj_jac(u0_cheat)
print(f"Objective value AT the exact true parameters: {val_at_truth:.4f}")
print(f"Objective value at the NORMAL fit's converged answer: {fit['result'].fun if 'result' in fit else 'N/A'}")

result_cheat = minimize(
    _obj_jac, u0_cheat, method="L-BFGS-B", bounds=u_bounds, jac=True,
    options={"maxiter": 10000, "maxfun": 200000, "ftol": 1e-12},
)
a_cheat = result_cheat.x * xscale
b_cheat = a_cheat[:nl]
gh_cheat = fit_losvd_gauss_hermite(st.xl, b_cheat, fit_h3h4=True)

print(f"Cheat-start fit (init at TRUE b/w): V={gh_cheat['vherm']:+.2f} sigma={gh_cheat['sherm']:.2f} "
      f"final_objective={result_cheat.fun:.4f}")
print(f"\n(if cheat-start V/sigma stays near {TRUE_V}/{TRUE_SIGMA} with objective <= normal fit's, "
      f"it's a bad-starting-point issue fixable by better init/multi-start;")
print(f" if it drifts back to ~{gh_normal['vherm']:.1f}/{gh_normal['sherm']:.1f} with objective <= the true-params value, "
      f"the biased answer is a genuine optimum -- fix has to be in the objective/regularization itself)")
