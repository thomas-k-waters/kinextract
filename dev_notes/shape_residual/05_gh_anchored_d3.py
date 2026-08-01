"""GH-anchored D3 penalty test.

Method under test (semi-parametric):
  1. Pilot: fit a 5-parameter Gauss-Hermite model (amp, V, sigma, h3, h4)
     directly to the whitened spectrum through the same design matrix
     (classic parametric fit, two starts to dodge local minima).
  2. Anchor: project log(max(b_GH, floor)) onto the spline basis ->
     c_anchor, t = D3 @ c_anchor.
  3. Fit the log-spline LOSVD with penalty 0.5*lam*||D3 c - t||^2
     (multi-start: anchor itself + the data-driven ridge start).

At lam -> inf this recovers the parametric GH fit (bias = parametric
bias); at lam -> 0 the unregularized spline. The hope: Gaussian-truth
performance matches plain D3, while real h3/h4 survives because the
anchor carries it.

Also records the pure parametric GH fit's own shape residual/moments
per case (the lam=inf limit), for comparison.

Truth shapes: Gaussian, (0,+0.10), (+0.10,0), (0.06,-0.06); sigma=50,
V=20; S/N {30,60,100}; seeds 0..14 (same draws as 01/02/03).
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMILES_GRID = os.path.join(HERE, "..", "..", "src", "kinextract", "template_data", "emiles_muse_grid.npz")
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7.0
TRUE_V, SIGMA = 20.0, 50.0

SHAPES = [(0.0, 0.0), (0.0, 0.10), (0.10, 0.0), (0.06, -0.06)]
LAM_GRID = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0]
SNR_LIST = [30.0, 60.0, 100.0]
SEEDS = list(range(15))

CSV_PATH = os.path.join(HERE, "05_gh_anchored_d3.csv")
FIELDS = ["h3_true", "h4_true", "snr", "seed", "lam", "max_resid_pct",
          "frac_over5_pct", "wing_mass_ratio", "m_eff", "chi2",
          "V", "sigma_mom", "h3_fit", "h4_fit", "h3_truthcurve", "h4_truthcurve",
          "sherm_truthcurve",
          # anchor (parametric GH pilot) quality per case, repeated on each row
          "anchor_max_resid_pct", "anchor_h3", "anchor_h4", "anchor_V", "anchor_sig"]


def parametric_gh_fit(A0_w, y_w, v_grid):
    """5-param GH fit to the whitened spectrum; two starts, keep best."""
    from scipy.optimize import least_squares
    from kinextract import gauss_hermite_model

    def model_flux(p):
        amp, V, logsig, h3, h4 = p
        b = np.maximum(gauss_hermite_model(v_grid, 1.0, V, np.exp(logsig), h3, h4), 0.0)
        s = b.sum()
        if s <= 0:
            return np.zeros(A0_w.shape[0])
        return amp * (A0_w @ (b / s))

    def resid(p):
        return model_flux(p) - y_w

    # amplitude scale init: project a flat-ish guess
    amp0 = float(np.dot(A0_w.mean(axis=1), y_w) / max(np.sum(A0_w.mean(axis=1) ** 2), 1e-12))
    best = None
    for V0, sig0 in [(0.0, 80.0), (0.0, 200.0)]:
        p0 = [max(amp0, 1e-3), V0, np.log(sig0), 0.0, 0.0]
        r = least_squares(resid, p0,
                          bounds=([0.0, -500.0, np.log(10.0), -0.3, -0.3],
                                  [np.inf, 500.0, np.log(500.0), 0.3, 0.3]),
                          max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    amp, V, logsig, h3, h4 = best.x
    return dict(V=V, sig=float(np.exp(logsig)), h3=h3, h4=h4, amp=amp)


def run_case(h3t, h4t, snr, seed):
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize as scipy_minimize
    from kinextract import (FitConfig, build_template_stack, gauss_hermite_model,
                                 load_emiles_templates, compute_losvd_moments)
    from kinextract.losvd import build_bspline_basis, difference_matrix, default_initial_coeffs

    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    cfg = FitConfig(vmin=TRUE_V - NSIG * SIGMA, vmax=TRUE_V + NSIG * SIGMA)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, TRUE_V, SIGMA, h3t, h4t), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth

    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A0_w = A_stack[0] / errors[:, None]
    y_w = galaxy / errors

    b_truth_n = b_truth / np.trapezoid(b_truth, v_grid)
    peak = b_truth_n.max()
    core = b_truth_n > 0.01 * peak
    dist = np.abs(v_grid - TRUE_V) / SIGMA
    wing_band = (dist >= 3.0) & (dist <= 6.0)
    truth_wing_mass = np.trapezoid(np.where(wing_band, b_truth_n, 0.0), v_grid)
    mom_t = compute_losvd_moments(v_grid, b_truth_n, fit_h3h4=True)

    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D3 = difference_matrix(n_coef, order=3, dv=dv, dv_ref=cfg.knot_spacing_kms)
    DtD = D3.T @ D3

    # --- pilot parametric GH fit + anchor ---
    gh = parametric_gh_fit(A0_w, y_w, v_grid)
    b_anchor = np.maximum(gauss_hermite_model(v_grid, 1.0, gh["V"], gh["sig"], gh["h3"], gh["h4"]), 0.0)
    b_anchor_n = b_anchor / max(np.trapezoid(b_anchor, v_grid), 1e-30)
    anchor_resid = np.abs(b_anchor_n - b_truth_n) / peak * 100
    # scale anchor to data amplitude before taking log (absolute scale is
    # what the spline actually fits; anchor amp from the parametric fit)
    b_anchor_scaled = np.clip(gh["amp"] * b_anchor / max(b_anchor.sum(), 1e-30), 1e-30, None)
    floor = 1e-6 * b_anchor_scaled.max()
    c_anchor, *_ = np.linalg.lstsq(basis, np.log(np.clip(b_anchor_scaled, floor, None)), rcond=None)
    t_anchor = D3 @ c_anchor

    basis_j, D3_j, A_j, y_j, t_j = map(jnp.array, (basis, D3, A0_w, y_w, t_anchor))

    def neg_log_post(c, lam):
        b = jnp.exp(basis_j @ c)
        r = A_j @ b - y_j
        p = D3_j @ c - t_j
        return 0.5 * jnp.sum(r ** 2) + 0.5 * lam * jnp.sum(p ** 2)

    vg = jax.jit(jax.value_and_grad(neg_log_post))

    ridge_start = default_initial_coeffs(A0_w, y_w, basis)
    rows = []
    for lam in LAM_GRID:
        lam_j = jnp.asarray(float(lam))

        def obj(c):
            v_, g_ = vg(jnp.array(c), lam_j)
            return float(v_), np.array(g_)

        fits = [scipy_minimize(obj, c0, jac=True, method="L-BFGS-B")
                for c0 in (c_anchor, ridge_start)]
        objs = [f.fun for f in fits]
        res = fits[int(np.argmin(objs))]
        b = np.exp(basis @ res.x)

        chi2 = float(np.sum((y_w - A0_w @ b) ** 2))
        J_b = b[:, None] * basis
        X = A0_w @ J_b
        XtX = X.T @ X
        m_eff = float(np.trace(np.linalg.lstsq(XtX + lam * DtD, XtX, rcond=None)[0]))

        b_fit_n = b / np.trapezoid(np.maximum(b, 0.0), v_grid)
        resid = np.abs(b_fit_n - b_truth_n) / peak * 100
        mom = compute_losvd_moments(v_grid, b_fit_n, fit_h3h4=True)
        fit_wing_mass = np.trapezoid(np.where(wing_band, b_fit_n, 0.0), v_grid)
        rows.append(dict(
            h3_true=h3t, h4_true=h4t, snr=snr, seed=seed, lam=lam,
            max_resid_pct=round(float(resid[core].max()), 3),
            frac_over5_pct=round(float(100 * np.mean(resid[core] > 5)), 2),
            wing_mass_ratio=round(float(fit_wing_mass / truth_wing_mass), 3),
            m_eff=round(m_eff, 3), chi2=round(chi2, 3),
            V=round(float(mom["vherm"]), 3), sigma_mom=round(float(mom["sherm"]), 3),
            h3_fit=round(float(mom["h3"]), 4), h4_fit=round(float(mom["h4"]), 4),
            h3_truthcurve=round(float(mom_t["h3"]), 4), h4_truthcurve=round(float(mom_t["h4"]), 4),
            sherm_truthcurve=round(float(mom_t["sherm"]), 3),
            anchor_max_resid_pct=round(float(anchor_resid[core].max()), 3),
            anchor_h3=round(float(gh["h3"]), 4), anchor_h4=round(float(gh["h4"]), 4),
            anchor_V=round(float(gh["V"]), 3), anchor_sig=round(float(gh["sig"]), 3),
        ))
    return rows


def main():
    t0 = time.perf_counter()
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            futs = {ex.submit(run_case, h3, h4, snr, seed): (h3, h4, snr, seed)
                    for (h3, h4) in SHAPES for snr in SNR_LIST for seed in SEEDS}
            done = 0
            for fut in as_completed(futs):
                for row in fut.result():
                    writer.writerow(row)
                fh.flush()
                done += 1
                print(f"[{time.perf_counter()-t0:7.1f}s] {done}/{len(futs)} {futs[fut]}", flush=True)
    print(f"finished in {time.perf_counter()-t0:.1f}s -> {CSV_PATH}")


if __name__ == "__main__":
    main()
