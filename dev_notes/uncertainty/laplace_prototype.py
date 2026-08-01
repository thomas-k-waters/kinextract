"""Prototype: a faster alternative to `bootstrap_losvd`'s uncertainty band.

Two ingredients, both reusing existing package machinery rather than adding
new fitting infrastructure:

1. **Laplace (curvature) covariance on the LOSVD spline coefficients**, at a
   fixed `lam`: the P-spline roughness penalty is already the Bayesian-
   equivalent Gaussian prior on `coeffs` (Eilers & Marx 1996's own framing),
   so the negative-log-posterior's Hessian at the converged solution
   (`losvd._neg_log_post`, the exact objective `fit_log_spline_map`
   minimizes, evaluated with the template weights held at their own
   converged value -- the same linearization `aic_regularization.
   estimate_m_eff_local` already uses for a related but different quantity,
   effective degrees of freedom) gives an approximate posterior covariance
   in one linear solve, no resampling. Propagated to `(vherm, sherm, h3,
   h4)` via a finite-difference Jacobian of `compute_losvd_moments` at the
   converged `coeffs` (that function isn't JAX-differentiable itself, so
   this one step stays numerical rather than autodiff).

2. **AIC-weighted model averaging over `cfg.resolved_lam_grid()`**, using the
   law of total variance: `Var_total = E_lam[Var(theta|lam)] + Var_lam[E(theta|lam)]`,
   with per-lam weights from the same `chi2 + w_m*m_eff` criterion
   `select_lambda_aic_local` already computes (`estimate_m_eff_local` is
   reused directly, so this costs about one AIC-selection pass, not one
   extra bootstrap-sized cost).

**Honest scope, read before trusting this over bootstrap.** `uncertainty.py`'s
own module docstring explains why full bootstrap was chosen over a Laplace
approximation: multi-start lets a replicate land in a genuinely different
local optimum, which shows up as spread in the bootstrap ensemble but would
be invisible to a naive single-point curvature estimate. This prototype does
NOT solve that -- it computes the Laplace covariance around the single
best-found optimum only, at each lam. It reports `fit_losvd`'s own
`objective_gap` (the gap between the best and second-best multi-start
candidate) as a coarse diagnostic instead: a small gap means another basin
was competitive, which is exactly when this method's unimodal assumption is
least trustworthy. Treat this as a fast *screening* estimator, not a
drop-in replacement -- see the benchmark script's own printed caveats.

Usage: run standalone (`python laplace_prototype.py`) for a quick multi-case
benchmark against real `bootstrap_losvd`, or import `laplace_aic_uncertainty`
directly.
"""
from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np

from kinextract import FitConfig, compute_losvd_moments
from kinextract.multi_template import fit_losvd
from kinextract.losvd import (
    _neg_log_post, build_bspline_basis, difference_matrix,
)
from kinextract.aic_regularization import estimate_m_eff_local

_hessian_fn = jax.jit(jax.hessian(_neg_log_post))


def _moments_jacobian(v_grid, basis, coeffs, eps=1e-4):
    """Finite-difference Jacobian of (vherm, sherm, h3, h4) w.r.t. `coeffs`,
    at the converged solution. `compute_losvd_moments` is not JAX-traceable
    (it calls `scipy.optimize.least_squares` internally for h3/h4), so this
    one step is numerical rather than autodiff, unlike the Hessian above.
    """
    n_coef = len(coeffs)
    keys = ("vherm", "sherm", "h3", "h4")
    J = np.zeros((4, n_coef))
    for k in range(n_coef):
        dc = np.zeros(n_coef)
        dc[k] = eps
        b_plus = np.exp(basis @ (coeffs + dc))
        b_minus = np.exp(basis @ (coeffs - dc))
        m_plus = compute_losvd_moments(v_grid, b_plus, fit_h3h4=True)
        m_minus = compute_losvd_moments(v_grid, b_minus, fit_h3h4=True)
        for i, key in enumerate(keys):
            J[i, k] = (m_plus[key] - m_minus[key]) / (2 * eps)
    return J, keys


def laplace_covariance_at_lam(A_stack_w, y_w, cfg: FitConfig, lam: float):
    """Fit at a fixed `lam` (reusing `estimate_m_eff_local`'s own fit --
    it already computes `chi2`/`m_eff`/the full `fit_losvd` result at this
    lam, so this is not an extra fit), then return
    `(result, moments, cov_moments, chi2, m_eff)`.

    `cov_moments` is the Laplace-approximated covariance of
    `(vherm, sherm, h3, h4)` -- see module docstring for the two
    simplifications this makes (weights held fixed, single optimum only).
    """
    m_eff, chi2, result = estimate_m_eff_local(A_stack_w, y_w, cfg, lam)

    v_grid = cfg.build_velocity_grid()
    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D = difference_matrix(n_coef, order=cfg.penalty_order, dv=dv, dv_ref=cfg.knot_spacing_kms)
    lam_vec = np.full(D.shape[0], lam)  # cfg.lam_wing == 0.0 (default): no taper, uniform lam

    A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
    coeffs = np.asarray(result["coeffs"])

    H = np.asarray(_hessian_fn(jnp.array(coeffs), jnp.array(basis), jnp.array(D),
                               jnp.array(A_eff_w), jnp.array(y_w), jnp.array(lam_vec)))
    try:
        cov_coeffs = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_coeffs = np.linalg.pinv(H)

    J, keys = _moments_jacobian(v_grid, basis, coeffs)
    cov_moments = J @ cov_coeffs @ J.T

    moments = compute_losvd_moments(v_grid, result["b"], fit_h3h4=True)
    return result, moments, cov_moments, float(chi2), float(m_eff)


def laplace_aic_uncertainty(A_stack_w, y_w, cfg: FitConfig, w_m: float = 2.0):
    """AIC-weighted model averaging of `laplace_covariance_at_lam` over
    `cfg.resolved_lam_grid()`. Returns a dict with the same
    `vherm/sherm/h3/h4` point estimate as the winning (lowest-criterion)
    lam, plus `*_std` from the law-of-total-variance combination across
    the whole grid, `lam_weights` (softmax AIC weights, sums to 1),
    `lam_grid`, and `objective_gap_at_best` (the winning lam's own
    multi-start diagnostic -- see module docstring).
    """
    t0 = time.time()
    lam_grid = np.asarray(cfg.resolved_lam_grid(), dtype=float)
    keys = ("vherm", "sherm", "h3", "h4")

    results, all_moments, all_covs, criteria = [], [], [], []
    for lam in lam_grid:
        result, moments, cov, chi2, m_eff = laplace_covariance_at_lam(A_stack_w, y_w, cfg, lam)
        results.append(result)
        all_moments.append(moments)
        all_covs.append(cov)
        criteria.append(chi2 + w_m * m_eff)
    criteria = np.array(criteria)

    weights = np.exp(-0.5 * (criteria - criteria.min()))
    weights /= weights.sum()

    means = np.array([[m[k] for k in keys] for m in all_moments])  # (n_lam, 4)
    overall_mean = weights @ means  # (4,)

    # Law of total variance: E[Var(theta|lam)] + Var[E(theta|lam)].
    within = np.zeros(4)
    between = np.zeros(4)
    for i in range(len(lam_grid)):
        within += weights[i] * np.diag(all_covs[i])
        between += weights[i] * (means[i] - overall_mean) ** 2
    total_var = within + between

    best_idx = int(np.argmin(criteria))
    out = {f"{k}": overall_mean[i] for i, k in enumerate(keys)}
    out.update({f"{k}_std": float(np.sqrt(max(total_var[i], 0.0))) for i, k in enumerate(keys)})
    out["lam_weights"] = weights
    out["lam_grid"] = lam_grid
    out["lam_used"] = float(lam_grid[best_idx])
    out["objective_gap_at_best"] = results[best_idx].get("objective_gap")
    out["runtime_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation_suite"))
    from mocks import EMILES_GRID_DEFAULT, observed_wavelength_grid, draw_random_params, build_mock
    from kinextract import whiten
    from kinextract.uncertainty import bootstrap_losvd

    _repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    _emiles_path = os.path.join(_repo_root, EMILES_GRID_DEFAULT)
    grid_file = np.load(_emiles_path, allow_pickle=True)
    emiles_ages, emiles_metals = grid_file["ages"], grid_file["metals"]
    wave_fit = observed_wavelength_grid()

    cases = [(sigma, snr) for sigma in (50.0, 90.0, 150.0) for snr in (40.0, 60.0, 100.0)]
    n_truths_per_cell = 2  # 9 cells x 2 = 18 cases total, bounded per plan (~12-16 target)

    print(f"{'sigma':>6} {'snr':>6} {'method':>10} {'time_s':>8} "
          f"{'sherm_cov':>10} {'h4_cov':>8} {'sherm_std':>10} {'h4_std':>8}")
    rows = []
    for sigma, snr in cases:
        for ti in range(n_truths_per_cell):
            seed = int(sigma * 1000 + snr * 10 + ti)
            rng = np.random.default_rng(seed)
            params = draw_random_params(rng, fixed={"sigma": sigma, "snr": snr}, preset="safe",
                                        gaussian_only=False, n_templates=4,
                                        emiles_ages=emiles_ages, emiles_metals=emiles_metals)
            mock = build_mock(params, emiles_grid_path=_emiles_path, wave_fit=wave_fit)
            tm = mock.truth_moments
            noise_level = np.median(mock.galaxy_noiseless) / params["snr"]
            nrng = np.random.default_rng(seed)
            galaxy = mock.galaxy_noiseless + nrng.normal(0.0, noise_level, len(mock.wave_fit))
            errors = np.full(len(mock.wave_fit), noise_level)
            A_w, y_w, _ = whiten(mock.A_stack, galaxy, errors, rho=0.0)

            t0 = time.time()
            boot = bootstrap_losvd(A_w, y_w, mock.cfg, n_boot=12, seed=seed, apply_bias_correction=False)
            t_boot = time.time() - t0
            sherm_cov_boot = abs(boot["point0"]["sherm"] - tm["sherm"]) <= boot["sherm_std"]
            h4_cov_boot = abs(boot["point0"]["h4"] - tm["h4"]) <= boot["h4_std"]

            t0 = time.time()
            lap = laplace_aic_uncertainty(A_w, y_w, mock.cfg)
            t_lap = time.time() - t0
            sherm_cov_lap = abs(lap["sherm"] - tm["sherm"]) <= lap["sherm_std"]
            h4_cov_lap = abs(lap["h4"] - tm["h4"]) <= lap["h4_std"]

            print(f"{sigma:6.0f} {snr:6.0f} {'bootstrap':>10} {t_boot:8.1f} "
                  f"{str(sherm_cov_boot):>10} {str(h4_cov_boot):>8} "
                  f"{boot['sherm_std']:10.2f} {boot['h4_std']:8.4f}")
            print(f"{sigma:6.0f} {snr:6.0f} {'laplace':>10} {t_lap:8.1f} "
                  f"{str(sherm_cov_lap):>10} {str(h4_cov_lap):>8} "
                  f"{lap['sherm_std']:10.2f} {lap['h4_std']:8.4f}   "
                  f"(objective_gap={lap['objective_gap_at_best']})")
            rows.append(dict(sigma=sigma, snr=snr, t_boot=t_boot, t_lap=t_lap,
                             sherm_cov_boot=sherm_cov_boot, h4_cov_boot=h4_cov_boot,
                             sherm_cov_lap=sherm_cov_lap, h4_cov_lap=h4_cov_lap))

    n = len(rows)
    speedup = sum(r["t_boot"] for r in rows) / max(sum(r["t_lap"] for r in rows), 1e-9)
    print(f"\n{n} cases total.")
    print(f"total time: bootstrap={sum(r['t_boot'] for r in rows):.0f}s  "
          f"laplace={sum(r['t_lap'] for r in rows):.0f}s  speedup={speedup:.1f}x")
    print(f"sigma coverage: bootstrap={100*sum(r['sherm_cov_boot'] for r in rows)/n:.0f}%  "
          f"laplace={100*sum(r['sherm_cov_lap'] for r in rows)/n:.0f}%")
    print(f"h4 coverage:    bootstrap={100*sum(r['h4_cov_boot'] for r in rows)/n:.0f}%  "
          f"laplace={100*sum(r['h4_cov_lap'] for r in rows)/n:.0f}%")
