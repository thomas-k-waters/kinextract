"""Tests whether variable projection (VARPRO -- Golub & Pereyra 1973;
Kaufman 1975's simplified-gradient variant) beats this package's current
alternating-minimization scheme (`multi_template._alternate_once`) for
the joint template-weight/LOSVD fit.

**The idea, and why it's worth testing.** The model is linear in the
template weights `w` for any fixed LOSVD `b(c) = exp(basis @ c)`, and
nonlinear (via the exp-link) in `c` for fixed `w`. The current scheme
handles this by *alternating*: solve `w` by NNLS holding `c` fixed, solve
`c` by L-BFGS holding `w` fixed, repeat until neither moves (`patience`
iterations of no improvement). VARPRO instead profiles `w` out of the
problem entirely -- for *every* trial `c` the outer optimizer proposes,
solve `w(c) = NNLS(M(c), y_w)` exactly (`M(c)`'s columns are each raw
template convolved with the trial LOSVD), and hand the outer optimizer
the *profiled* objective `F(c) = ||y_w - M(c) w(c)||^2 + lam*penalty(c)`
directly -- a single nonlinear optimization over `c` alone, not two
alternating blocks. The gradient of `F` w.r.t. `c` can be computed
*holding `w(c)` fixed at its current NNLS-optimal value* (the envelope
theorem / Danskin's theorem: since `w(c)` is already optimal for that
`c`, its own first-order sensitivity contributes nothing to `dF/dc` in
directions where the active NNLS constraint set doesn't change --
"Kaufman's approximation" to full VARPRO, cheaper than the exact
Golub-Pereyra correction term and the standard practical simplification
in the literature). Concretely this means: reuse this package's existing
`fit_log_spline_map` objective/gradient machinery unchanged, just with
its `A_w` argument refreshed via a fresh NNLS solve at *every* L-BFGS
function/gradient evaluation, instead of once per outer alternation pass.

**Why this might matter here specifically.** This package has an open,
undiagnosed non-convergence issue (continuum-cofitting interacting badly
with the LOSVD sub-fit -- see `continuum.py`'s docstring), and the
E-MILES mismatch tests just run this session showed a *much* higher
failure rate than previously documented (up to ~40-47% of fits flagged
`success=False` at S/N=60 with continuum on). Naive alternating
minimization is a known-fragile scheme for exactly this class of
separable nonlinear least squares problem (it can zigzag or stall in a
way that has nothing to do with the true profiled landscape `F(c)`);
VARPRO is the textbook fix. If it measurably improves convergence rate
and/or bias here, that's a concrete, well-motivated candidate fix for
the open non-convergence problem -- not a replacement architecture, just
a better optimizer for the same objective.

**Scope of this test**: single-template-group only (no continuum), to
isolate the optimizer-scheme question cleanly before combining it with
anything else -- consistent with this project's own "verify each piece
in isolation before combining" rule. Compares, at matched seeds:
  - `alternating`: `kinextract.fit_losvd` (current default)
  - `varpro`: the profiled-objective scheme implemented below

on success rate, V/sigma/h4 bias, and wall-clock time, across a small
sigma/S-N grid using the real 33-star MUSE library (deliberately
collinear -- the regime most likely to expose optimizer fragility).

Usage: python varpro_test.py [outfile]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import minimize as scipy_minimize, nnls

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model, fit_losvd, compute_losvd_moments,
)
from kinextract.losvd import build_bspline_basis, difference_matrix, _default_multi_starts

MUSE_TEMPLATE_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/sco_framework_updated/sco_workdir_example/analysis/muse_templates/old"
WAVELENGTH = 4750.0 + np.arange(3681) * 1.25
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0

SIGMA_GRID = [30, 50, 80, 150, 300]
SNR_GRID = [20.0, 60.0]
N_REPS = 15
LAM = 2.0
V_TRUE = 0.0
MIXTURE_SEED = 40000


def _load_muse_templates():
    with open(os.path.join(MUSE_TEMPLATE_DIR, "Tlist")) as f:
        names = [line.strip() for line in f if line.strip()]
    templates = []
    for name in names:
        d = np.loadtxt(os.path.join(MUSE_TEMPLATE_DIR, name))
        templates.append(np.interp(WAVELENGTH, d[:, 0], d[:, 1]))
    return templates


def _neg_log_post(coeffs, A_eff, basis_j, D2_j, y_w_j, lam):
    b = jnp.exp(basis_j @ coeffs)
    resid = A_eff @ b - y_w_j
    chi2 = 0.5 * jnp.sum(resid ** 2)
    rough = 0.5 * lam * jnp.sum((D2_j @ coeffs) ** 2)
    return chi2 + rough


_val_fn = jax.jit(_neg_log_post, static_argnums=())
_grad_fn = jax.jit(jax.grad(_neg_log_post, argnums=0), static_argnums=())


def fit_losvd_varpro(A_stack, y_w, basis, D2, lam, coeffs0_pool):
    """Profiled (VARPRO/Kaufman) fit: single L-BFGS-B optimization over
    LOSVD coefficients `c`, with template weights `w` re-solved via NNLS
    at every function/gradient evaluation (envelope-theorem gradient --
    see module docstring). Multi-start over the same coefficient starts
    `fit_log_spline_map` would use, for a fair comparison.
    """
    n_templates, n_pix, n_bins = A_stack.shape
    basis_j, D2_j, y_w_j = jnp.array(basis), jnp.array(D2), jnp.array(y_w)

    def objective_and_grad(coeffs_np):
        # L-BFGS's line search can propose a step large enough that exp()
        # overflows to inf before the objective's own curvature penalty has
        # a chance to correct it (unlike the alternating scheme, where each
        # LOSVD sub-fit is re-started from a fresh, bounded multi-start pool
        # every outer iteration) -- clip defensively (exp(50) is already
        # far beyond any physically meaningful LOSVD amplitude) so a wild
        # trial point returns a large-but-finite value/gradient instead of
        # crashing the NNLS call on a non-finite design matrix.
        coeffs_np = np.clip(np.asarray(coeffs_np, dtype=float), -50.0, 50.0)
        b = np.exp(basis @ coeffs_np)
        M = np.stack([A_stack[i] @ b for i in range(n_templates)], axis=1)  # (n_pix, n_templates)
        w, _ = nnls(M, y_w)
        A_eff = np.tensordot(w, A_stack, axes=(0, 0))  # (n_pix, n_bins)
        A_eff_j = jnp.array(A_eff)
        coeffs_j = jnp.array(coeffs_np)
        val = float(_val_fn(coeffs_j, A_eff_j, basis_j, D2_j, y_w_j, lam))
        grad = np.array(_grad_fn(coeffs_j, A_eff_j, basis_j, D2_j, y_w_j, lam), dtype=float)
        return val, grad

    t0 = time.perf_counter()
    fits = []
    for c0 in coeffs0_pool:
        res = scipy_minimize(objective_and_grad, c0, jac=True, method='L-BFGS-B')
        fits.append(res)
    runtime_s = time.perf_counter() - t0

    objectives = [float(r.fun) for r in fits]
    best_idx = int(np.argmin(objectives))
    res = fits[best_idx]
    coeffs = res.x
    b = np.exp(basis @ coeffs)
    M = np.stack([A_stack[i] @ b for i in range(n_templates)], axis=1)
    w, _ = nnls(M, y_w)
    return dict(b=b, coeffs=coeffs, w=w, success=bool(res.success), message=str(res.message),
                runtime_s=runtime_s, objective=objectives[best_idx])


def run(sigma_grid=SIGMA_GRID, snr_grid=SNR_GRID):
    muse_templates = _load_muse_templates()
    print(f"loaded {len(muse_templates)} MUSE templates")
    mask = (WAVELENGTH >= WAVEFITMIN) & (WAVELENGTH <= WAVEFITMAX)
    wave_fit = WAVELENGTH[mask]

    rng_mix = np.random.default_rng(MIXTURE_SEED)
    true_weights = rng_mix.dirichlet(np.ones(len(muse_templates)))

    rows = []
    t0 = time.time()
    for snr in snr_grid:
        for sig in sigma_grid:
            cfg = FitConfig(vmin=V_TRUE - 7 * sig, vmax=V_TRUE + 7 * sig, lam=LAM)
            v_grid = cfg.build_velocity_grid()
            n_knots = cfg.resolved_n_interior_knots()
            basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
            dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
            D2 = difference_matrix(n_coef, order=2, dv=dv, dv_ref=cfg.knot_spacing_kms)

            b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, V_TRUE, sig, 0.0, 0.0), 0.0)
            b_truth /= b_truth.sum()
            A_stack = build_template_stack(wave_fit, WAVELENGTH, muse_templates, v_grid)

            vv = np.linspace(V_TRUE - 15 * sig, V_TRUE + 15 * sig, 200_001)
            bb = np.maximum(gauss_hermite_model(vv, 1.0, V_TRUE, sig, 0.0, 0.0), 0.0)
            tot = np.trapezoid(bb, vv)
            true_v = float(np.trapezoid(bb * vv, vv) / tot)
            true_s = float(np.sqrt(np.trapezoid(bb * (vv - true_v) ** 2, vv) / tot))

            errs = {"alternating": dict(v=[], s=[], h4=[], t=[], nfail=0),
                    "varpro": dict(v=[], s=[], h4=[], t=[], nfail=0)}
            for rep in range(N_REPS):
                seed = 3_000_000 + rep + int(sig * 10) + int(snr)
                galaxy_noiseless = sum(true_weights[k] * (A_stack[k] @ b_truth) for k in range(len(muse_templates)))
                noise_level = np.median(galaxy_noiseless) / snr
                rng = np.random.default_rng(seed)
                galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
                errors = np.full(len(wave_fit), noise_level)
                y_w = galaxy / errors
                A_stack_w = A_stack / errors[None, :, None]

                ta0 = time.perf_counter()
                result_a = fit_losvd(A_stack_w, y_w, cfg)
                ta = time.perf_counter() - ta0
                if not result_a.get("success", True):
                    errs["alternating"]["nfail"] += 1
                if np.all(np.isfinite(result_a["b"])):
                    m = compute_losvd_moments(v_grid, result_a["b"])
                    if np.isfinite(m["vherm"]) and np.isfinite(m["sherm"]):
                        errs["alternating"]["v"].append(m["vherm"] - true_v)
                        errs["alternating"]["s"].append(m["sherm"] - true_s)
                        errs["alternating"]["h4"].append(m["h4"])
                        errs["alternating"]["t"].append(ta)

                starts = _default_multi_starts(A_stack_w[0], y_w, basis, ridge_lam=1.0)
                tv0 = time.perf_counter()
                result_v = fit_losvd_varpro(A_stack_w, y_w, basis, D2, LAM, starts)
                tv = time.perf_counter() - tv0
                if not result_v.get("success", True):
                    errs["varpro"]["nfail"] += 1
                if np.all(np.isfinite(result_v["b"])):
                    m = compute_losvd_moments(v_grid, result_v["b"])
                    if np.isfinite(m["vherm"]) and np.isfinite(m["sherm"]):
                        errs["varpro"]["v"].append(m["vherm"] - true_v)
                        errs["varpro"]["s"].append(m["sherm"] - true_s)
                        errs["varpro"]["h4"].append(m["h4"])
                        errs["varpro"]["t"].append(tv)

            for label in ("alternating", "varpro"):
                d = errs[label]
                n = len(d["s"])
                if n < 3:
                    print(f"[snr={snr:5.0f}][{label:>12s}] sigma={sig:5.0f}  n={n} too few valid fits "
                          f"(nfail={d['nfail']})  t={time.time()-t0:.0f}s", flush=True)
                    continue
                row = dict(
                    label=label, snr=snr, sigma=float(sig), n=n, nfail=d["nfail"],
                    v_bias_mean=float(np.mean(d["v"])), v_bias_std=float(np.std(d["v"], ddof=1)),
                    sig_bias_mean=float(np.mean(d["s"])), sig_bias_std=float(np.std(d["s"], ddof=1)),
                    h4_mean=float(np.mean(d["h4"])), h4_std=float(np.std(d["h4"], ddof=1)),
                    mean_runtime=float(np.mean(d["t"])),
                )
                rows.append(row)
                print(f"[snr={snr:5.0f}][{label:>12s}] sigma={sig:5.0f} n={n:3d} nfail={d['nfail']:2d}  "
                      f"V_bias={row['v_bias_mean']:+7.2f}  "
                      f"sig_bias={row['sig_bias_mean']:+7.2f} ({100*row['sig_bias_mean']/sig:+6.1f}%)  "
                      f"h4={row['h4_mean']:+.4f}  runtime={row['mean_runtime']*1000:.0f}ms  "
                      f"t={time.time()-t0:.0f}s", flush=True)
    return rows


def save_rows(rows, path):
    with open(path, "w") as f:
        f.write(f"{'label':>12} {'snr':>6} {'sigma':>6} {'n':>4} {'nfail':>6}   "
                f"{'V_bias_mean':>12} {'V_bias_std':>11}   "
                f"{'sig_bias_mean':>14} {'sig_bias_std':>13} {'sig_bias_pct':>13}   "
                f"{'h4_mean':>9} {'h4_std':>8}   {'runtime_ms':>10}\n")
        for r in rows:
            pct = 100 * r["sig_bias_mean"] / r["sigma"]
            f.write(f"{r['label']:>12} {r['snr']:6.0f} {r['sigma']:6.0f} {r['n']:4d} {r['nfail']:6d}   "
                    f"{r['v_bias_mean']:+12.2f} {r['v_bias_std']:11.2f}   "
                    f"{r['sig_bias_mean']:+14.2f} {r['sig_bias_std']:13.2f} {pct:+12.1f}%   "
                    f"{r['h4_mean']:+9.4f} {r['h4_std']:8.4f}   {r['mean_runtime']*1000:10.1f}\n")


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "varpro_test_results.txt")
    rows = run()
    save_rows(rows, outfile)
    print(f"wrote {len(rows)} rows to {outfile}")
