"""Laplace uncertainty for the non-parametric LOSVD, and the package's error-band recipe for real IFU data.

**The method.** The P-spline roughness penalty is the Bayesian equivalent
of a Gaussian prior on the spline coefficients (Eilers & Marx 1996), so
the Hessian of the negative log-posterior at the fitted solution
(`losvd._neg_log_post`, the exact objective `fit_log_spline_map`
minimizes, template weights held at their converged value) gives an
approximate posterior covariance in one linear solve -- jointly with the
continuum coefficients when continuum-cofitting is on (the default). Draws
`coeffs ~ N(coeffs_MAP, cov)` are pushed through `exp(basis @ coeffs)`
(unit-area normalized) to give sampled LOSVD curves, and each curve's
`(vherm, sherm, h3, h4)` comes from the same `compute_losvd_moments`
estimator used everywhere else. The same draws give the curve-level band
(`b_p16`/`b_p84`/`b_median`) and the moment errors.

The posterior is evaluated at one `lam`, the fit's own (`cfg.lam`; pass a
fit's `lam_used`). Marginalizing over `lam` was tested and barely changed
the errors (dev_notes/regularization_v2/FINDINGS.md, finding 14).

**Real IFU data: `losvd_error_bands`.** On real MUSE spectra the
residuals are dominated by sky-line residuals and template mismatch, not
photon noise, and Laplace bands from the pipeline errors under-cover. The
validated recipe (NGC 4751; FINDINGS findings 16-18 and 23) is:
1. fit with the plain errors (`fit_losvd_robust`);
2. fit the sky-line + template-mismatch variance model
   (`noise_model.noise_model_errors`) to a preliminary fit's residuals;
3. run Laplace with those errors (times the robust fit's own inflation),
   at the fit's `lam` and `rho`, without its clipped pixels;
4. move the draws onto the plain-error fit (`recenter_samples`) -- the
   point estimate keeps the better shape, the band carries the
   systematics.
`losvd_error_bands` does steps 2-4 for one bin, `losvd_error_bands_parallel`
for many. Do not *fit* with the noise-model errors: at a fixed `lam`,
near-uniformly larger errors act as a stronger penalty and over-smooth.

**Honest scope:**
- **Template-weight uncertainty is not modeled.** `w` is NNLS-constrained
  (non-Gaussian) and is held fixed at its converged value.
- **Template mismatch biases the point estimate**, which no band width
  fixes: on clean, high-S/N mocks built from different templates than the
  fit, the model-binned coverage is still about 0.42 with these bands.
- **Multi-start's own basin-selection risk is not modeled.** A different
  basin winning on a different noise draw is invisible to a unimodal
  Gaussian approximation. `objective_gap` (from `fit_log_spline_map`) is
  the closest diagnostic: a small gap means another basin was competitive.
  `nuts_uncertainty` samples the exact posterior instead (slower).
"""
from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .config import PENALTY_ORDER, FitConfig
from .gh_moments import compute_losvd_moments
from .losvd import _neg_log_post, build_bspline_basis, difference_matrix

_hessian_fn = jax.jit(jax.hessian(_neg_log_post))


def _neg_log_post_joint(theta, n_coef_losvd, basis, D, A_eff_w, y_w, lam_vec,
                         basis_c, D2_c, lam_c, ridge_c):
    """The joint `(LOSVD coeffs, continuum coeffs)` negative log-posterior
    -- `losvd._neg_log_post` and `continuum._neg_log_post_continuum`
    combined into one scalar function of the concatenated coefficient
    vector `theta`, template weights `w` held fixed (folded into
    `A_eff_w`), exactly the same simplification the no-continuum path
    already makes. `n_coef_losvd` is a plain Python int (closed over via
    `functools.partial`, not a traced argument) used to split `theta` --
    static under `jax.jit` by construction, no `static_argnums` needed.
    """
    coeffs = theta[:n_coef_losvd]
    coeffs_c = theta[n_coef_losvd:]
    b = jnp.exp(basis @ coeffs)
    C = jnp.exp(basis_c @ coeffs_c)
    resid = C * (A_eff_w @ b) - y_w
    chi2 = 0.5 * jnp.sum(resid ** 2)
    rough = 0.5 * jnp.sum(lam_vec * (D @ coeffs) ** 2)
    rough_c = 0.5 * lam_c * jnp.sum((D2_c @ coeffs_c) ** 2)
    ridge = 0.5 * ridge_c * jnp.sum(coeffs_c ** 2)
    return chi2 + rough + rough_c + ridge


def laplace_covariance_at_lam(A_stack_w: np.ndarray, y_w: np.ndarray, cfg: FitConfig, lam: float) -> dict:
    """Fit at exactly `lam` (no continuum: `cfg.continuum_knots` must be None), then return the Laplace covariance
    of the spline coefficients at that solution.

    Returns a dict: `result` (the `fit_losvd` result), `chi2`, `coeffs` (the converged coefficients, the Laplace
    mean), `cov_coeffs` (n_coef x n_coef Laplace covariance), `basis`, `v_grid` -- everything
    `_sample_curves_and_moments` needs.
    """
    from .multi_template import fit_losvd

    fit_cfg = dataclasses.replace(cfg, lam_rule="fixed", lam=lam)
    result = fit_losvd(A_stack_w, y_w, fit_cfg)

    v_grid = fit_cfg.build_velocity_grid()
    n_knots = fit_cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=fit_cfg.degree)
    dv = (fit_cfg.vmax - fit_cfg.vmin) / (n_knots + 1)
    D = difference_matrix(n_coef, order=PENALTY_ORDER, dv=dv, dv_ref=fit_cfg.knot_spacing_kms)
    lam_vec = np.full(D.shape[0], lam)  # uniform lam across every penalty row

    A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
    coeffs = np.asarray(result["coeffs"])
    chi2 = float(np.sum((A_eff_w @ result["b"] - y_w) ** 2))

    H = np.asarray(_hessian_fn(jnp.array(coeffs), jnp.array(basis), jnp.array(D),
                               jnp.array(A_eff_w), jnp.array(y_w), jnp.array(lam_vec)))
    try:
        cov_coeffs = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_coeffs = np.linalg.pinv(H)

    return dict(result=result, chi2=chi2, coeffs=coeffs, cov_coeffs=cov_coeffs, basis=basis, v_grid=v_grid)


def laplace_covariance_at_lam_with_continuum(A_stack_w: np.ndarray, y_w: np.ndarray, wave_fit: np.ndarray,
                                              cfg: FitConfig, lam: float) -> dict:
    """`laplace_covariance_at_lam`'s continuum-cofitting counterpart: fits at
    exactly `lam` with `fit_losvd` (continuum from `cfg.continuum_knots`/
    `lam_c`/`ridge_c`), then returns the Laplace covariance of the *joint*
    `(LOSVD coeffs, continuum coeffs)` vector, `w` held fixed at its
    converged value -- the same simplification `laplace_covariance_at_lam`
    already makes for `w` alone.

    Returns a dict with `result` (the `fit_losvd`-shaped fit result),
    `coeffs`/`continuum_coeffs` (the converged coefficients, split back
    out for convenience), `cov_joint` (the full joint covariance --
    `cov_joint[:n_coef, :n_coef]` is the LOSVD block,
    `cov_joint[n_coef:, n_coef:]` the continuum block, off-diagonal blocks
    the cross-covariance), `n_coef` (LOSVD coefficient count, for
    splitting `cov_joint`), `basis`/`basis_c` (LOSVD/continuum design
    matrices), `v_grid`.
    """
    from .multi_template import fit_losvd

    fit_cfg = dataclasses.replace(cfg, lam_rule="fixed", lam=lam)
    n_continuum_knots, lam_c, ridge_c = fit_cfg.continuum_knots, fit_cfg.lam_c, fit_cfg.ridge_c
    v_grid = fit_cfg.build_velocity_grid()
    n_knots = fit_cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=fit_cfg.degree)
    dv = (fit_cfg.vmax - fit_cfg.vmin) / (n_knots + 1)
    D = difference_matrix(n_coef, order=PENALTY_ORDER, dv=dv, dv_ref=fit_cfg.knot_spacing_kms)
    lam_vec = np.full(D.shape[0], lam)

    result = fit_losvd(A_stack_w, y_w, fit_cfg, wave_fit=wave_fit)

    basis_c, n_coef_c = build_bspline_basis(wave_fit, n_interior_knots=n_continuum_knots, degree=3)
    dw = (wave_fit.max() - wave_fit.min()) / (n_continuum_knots + 1)
    D2_c = difference_matrix(n_coef_c, order=2, dv=dw, dv_ref=dw)

    A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
    coeffs = np.asarray(result["coeffs"])
    coeffs_c = np.asarray(result["continuum_coeffs"])
    theta = np.concatenate([coeffs, coeffs_c])

    hessian_fn = jax.hessian(lambda th: _neg_log_post_joint(
        th, n_coef, jnp.array(basis), jnp.array(D), jnp.array(A_eff_w), jnp.array(y_w),
        jnp.array(lam_vec), jnp.array(basis_c), jnp.array(D2_c), lam_c, ridge_c))
    H = np.asarray(hessian_fn(jnp.array(theta)))
    try:
        cov_joint = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_joint = np.linalg.pinv(H)

    return dict(result=result, coeffs=coeffs, continuum_coeffs=coeffs_c, cov_joint=cov_joint,
                n_coef=n_coef, basis=basis, basis_c=basis_c, v_grid=v_grid, wave_fit=wave_fit)


def _sample_curves_and_moments(coeffs_map: np.ndarray, cov_coeffs: np.ndarray, basis: np.ndarray,
                                v_grid: np.ndarray, n_samples: int, rng: np.random.Generator,
                                fit_h3h4: bool = True, max_edge_amplitude_frac: float = 0.01,
                                max_draw_multiplier: int = 20) -> tuple:
    """Draw `n_samples` posterior-predictive LOSVD curves from
    `N(coeffs_map, cov_coeffs)`, push each through `exp(basis @ coeffs)`,
    unit-area normalize (removes the `(w, b) -> (c*w, b/c)` scale
    degeneracy), and read off
    `(vherm, sherm, h3, h4)` from each sampled curve via
    `compute_losvd_moments` (not JAX-traceable itself -- calls
    `scipy.optimize.least_squares` for h3/h4 -- so this loop is plain
    Python/numpy, not autodiff/vmap).

    **Rejects divergent-edge draws (`max_edge_amplitude_frac`, the
    threshold the lam rule uses for the point estimate) -- found to be
    necessary, not optional.** The order-3 curvature
    penalty's null space contains divergent solutions (e.g. the mirror-
    image `exp(+av^2)`, `VALIDATION_CAMPAIGN.md` Stage 23) that a weakly-
    regularized `lam`'s point estimate can already avoid by construction
    (the multi-start optimizer just doesn't converge there), but naive
    posterior-predictive *sampling* in coefficient space has no such
    protection: at low `lam`, several coefficient directions are only
    weakly constrained by the data (few effective degrees of freedom relative to `n_coef`),
    giving those directions a large Laplace variance, and pushing a
    large excursion in one of those directions through the highly
    nonlinear `exp()` link can land squarely in the divergent null-space
    direction. Measured directly on one representative low-`lam` case:
    26% of naive draws had a grid-edge amplitude above 1% of peak, and
    those draws alone inflated recovered sigma from ~88 km/s (every
    other draw) up to 312 km/s -- a small minority of pathological
    samples dominating the reported spread. Rejecting and redrawing
    (up to `max_draw_multiplier * n_samples` total draws, since some
    `lam`/covariance combinations produce a real, non-negligible
    pathological fraction) keeps the returned ensemble exactly
    `n_samples` valid draws rather than silently returning fewer.

    Returns `(b_samples (n_samples, n_bins), vherm, sherm, h3, h4)`, the
    latter four each shape `(n_samples,)`.
    """
    # method="eigh" (not the default "svd"/"cholesky"): robust to cov_coeffs being only
    # approximately PSD (a real possibility here -- cov_coeffs comes from inverting a
    # Hessian that JAX/float64 rounding can leave with tiny negative eigenvalues), and
    # cheaper than SVD at this matrix size. check_valid="ignore" since we've already
    # accepted the pinv fallback in laplace_covariance_at_lam for the genuinely singular case.
    kept_b = []
    n_drawn = 0
    max_draws = max_draw_multiplier * n_samples
    batch = n_samples
    while sum(len(b) for b in kept_b) < n_samples and n_drawn < max_draws:
        batch = min(batch, max_draws - n_drawn)
        coeffs_batch = rng.multivariate_normal(coeffs_map, cov_coeffs, size=batch,
                                                method="eigh", check_valid="ignore")
        b_batch = np.exp(coeffs_batch @ basis.T)  # (batch, n_bins)
        areas = np.trapezoid(b_batch, v_grid, axis=1)
        areas = np.where(areas > 0, areas, 1.0)
        b_batch = b_batch / areas[:, None]
        edge_frac = np.maximum(b_batch[:, 0], b_batch[:, -1]) / np.maximum(b_batch.max(axis=1), 1e-300)
        valid = edge_frac <= max_edge_amplitude_frac
        kept_b.append(b_batch[valid])
        n_drawn += batch
        batch = max(batch, n_samples)  # keep batches reasonably sized on retry

    b_samples = np.concatenate(kept_b, axis=0) if kept_b else np.zeros((0, basis.shape[0]))
    if len(b_samples) < n_samples:
        warnings.warn(
            f"_sample_curves_and_moments: only {len(b_samples)}/{n_samples} draws passed the "
            f"max_edge_amplitude_frac={max_edge_amplitude_frac} filter after {n_drawn} attempts "
            "-- this lam's own posterior is dominated by divergent-edge solutions. Using what "
            "was drawn rather than looping forever.",
            UserWarning, stacklevel=2,
        )
    else:
        b_samples = b_samples[:n_samples]
    n_samples = len(b_samples)  # may be smaller than requested if the filter above triggered

    vherm = np.empty(n_samples)
    sherm = np.empty(n_samples)
    h3 = np.empty(n_samples)
    h4 = np.empty(n_samples)
    for i in range(n_samples):
        m = compute_losvd_moments(v_grid, b_samples[i], fit_h3h4=fit_h3h4)
        vherm[i], sherm[i], h3[i], h4[i] = m["vherm"], m["sherm"], m["h3"], m["h4"]
    return b_samples, vherm, sherm, h3, h4


def _sample_joint_curves_and_moments(coeffs_map: np.ndarray, coeffs_c_map: np.ndarray,
                                      cov_joint: np.ndarray, n_coef: int, basis: np.ndarray,
                                      basis_c: np.ndarray, v_grid: np.ndarray, n_samples: int,
                                      rng: np.random.Generator, fit_h3h4: bool = True,
                                      max_edge_amplitude_frac: float = 0.01,
                                      max_draw_multiplier: int = 20) -> tuple:
    """`_sample_curves_and_moments`'s continuum-aware counterpart: draws
    the *joint* `(LOSVD coeffs, continuum coeffs)` vector from
    `N([coeffs_map, coeffs_c_map], cov_joint)` (not two independent draws
    -- the joint posterior has real LOSVD/continuum cross-covariance,
    since both blocks jointly explain the same residual), splits each
    draw, and pushes the LOSVD half through the same divergent-edge
    rejection filter `_sample_curves_and_moments` uses (the continuum's
    own, much coarser, order-2-penalized spline is not the failure mode
    that filter exists for, so only the LOSVD half is checked). Returns
    `(b_samples, C_samples, vherm, sherm, h3, h4)` -- `C_samples` are the
    continuum curves paired with each *kept* LOSVD sample, same length.

    **Renormalizing `b` to unit area requires compensating `C` by the
    same factor, not leaving it untouched.** `model = C * (A_eff_w @ b)`
    is linear in `b`, so `b -> b/area, C -> C*area` leaves the model (and
    therefore this draw's own likelihood) exactly invariant -- it is a
    change of gauge, not a change of fit. Skipping the `C` compensation
    was tried first and measured directly to produce absurd `C_std`
    (samples with `C`'s own mean up to ~15x the point estimate's, on a
    mock where the true amplitude never exceeds 1.15): `b`'s raw
    (pre-normalization) amplitude and `C`'s own mean are the *same*
    degenerate direction documented in `kinextract.continuum`'s own
    module docstring (root-caused 2026-08-01) -- with `w` held fixed
    (this module's own simplification), nothing else constrains that
    direction, and the continuum block's `ridge_c` alone only weakly
    bounds it (a real, but not tight, quadratic cost), so naive sampling
    explores it far beyond anything physically reasonable. Compensating
    `C` exactly cancels this spurious component of the draw-to-draw
    spread, leaving only the genuinely different-fit variation the
    Laplace covariance is meant to characterize.
    """
    theta_map = np.concatenate([coeffs_map, coeffs_c_map])
    kept_b, kept_c = [], []
    n_drawn = 0
    max_draws = max_draw_multiplier * n_samples
    batch = n_samples
    while sum(len(b) for b in kept_b) < n_samples and n_drawn < max_draws:
        batch = min(batch, max_draws - n_drawn)
        theta_batch = rng.multivariate_normal(theta_map, cov_joint, size=batch,
                                               method="eigh", check_valid="ignore")
        coeffs_batch = theta_batch[:, :n_coef]
        coeffs_c_batch = theta_batch[:, n_coef:]
        b_batch = np.exp(coeffs_batch @ basis.T)
        areas = np.trapezoid(b_batch, v_grid, axis=1)
        areas = np.where(areas > 0, areas, 1.0)
        b_batch = b_batch / areas[:, None]
        c_batch = np.exp(coeffs_c_batch @ basis_c.T) * areas[:, None]  # compensate, see docstring
        edge_frac = np.maximum(b_batch[:, 0], b_batch[:, -1]) / np.maximum(b_batch.max(axis=1), 1e-300)
        valid = edge_frac <= max_edge_amplitude_frac
        kept_b.append(b_batch[valid])
        kept_c.append(c_batch[valid])
        n_drawn += batch
        batch = max(batch, n_samples)

    b_samples = np.concatenate(kept_b, axis=0) if kept_b else np.zeros((0, basis.shape[0]))
    C_samples = np.concatenate(kept_c, axis=0) if kept_c else np.zeros((0, basis_c.shape[0]))
    if len(b_samples) < n_samples:
        warnings.warn(
            f"_sample_joint_curves_and_moments: only {len(b_samples)}/{n_samples} draws passed "
            f"the max_edge_amplitude_frac={max_edge_amplitude_frac} filter after {n_drawn} "
            "attempts -- see _sample_curves_and_moments's own docstring for the mechanism. "
            "Using what was drawn rather than looping forever.",
            UserWarning, stacklevel=2,
        )
    else:
        b_samples = b_samples[:n_samples]
        C_samples = C_samples[:n_samples]
    n_samples = len(b_samples)

    vherm = np.empty(n_samples)
    sherm = np.empty(n_samples)
    h3 = np.empty(n_samples)
    h4 = np.empty(n_samples)
    for i in range(n_samples):
        m = compute_losvd_moments(v_grid, b_samples[i], fit_h3h4=fit_h3h4)
        vherm[i], sherm[i], h3[i], h4[i] = m["vherm"], m["sherm"], m["h3"], m["h4"]
    return b_samples, C_samples, vherm, sherm, h3, h4


def _degenerate_point_fallback(fn_name: str, point0: dict, b_point: np.ndarray, v_grid: np.ndarray,
                                C_point: np.ndarray | None = None):
    """Last-resort fallback for when posterior-predictive sampling
    produces zero valid draws even after every retry budget is
    exhausted -- found directly running at real-data scale (2026-08-03):
    a bin can be so poorly constrained that literally every draw hits
    the divergent-edge rejection filter (`_sample_curves_and_moments`'s
    own docstring). Left unhandled, `np.percentile`/`.std(ddof=1)` on the
    resulting empty array raises `IndexError`, crashing an entire
    multi-bin parallel batch over one bad bin. Returns a degenerate,
    zero-spread "ensemble" (the point estimate repeated twice, so
    `ddof=1` std is well-defined as exactly 0.0 rather than NaN) --
    correctly signals "sampling couldn't characterize this bin" rather
    than silently fabricating a number or propagating a crash.
    """
    warnings.warn(
        f"{fn_name}: posterior-predictive sampling produced zero valid draws (every draw hit "
        "the divergent-edge rejection filter) -- this bin's fit is too poorly constrained for "
        "this method's sampling step to characterize. Falling back to the point estimate "
        "repeated as a degenerate ensemble (reported std will be 0, not a real measurement) "
        "rather than crashing.", UserWarning, stacklevel=2,
    )
    b_area = np.trapezoid(np.maximum(b_point, 0.0), v_grid)
    b_list = [np.tile(b_point / b_area, (2, 1))]
    vherm_list = [np.full(2, point0["vherm"])]
    sherm_list = [np.full(2, point0["sherm"])]
    h3_list = [np.full(2, point0["h3"])]
    h4_list = [np.full(2, point0["h4"])]
    if C_point is None:
        return b_list, vherm_list, sherm_list, h3_list, h4_list
    C_list = [np.tile(C_point, (2, 1))]
    return b_list, C_list, vherm_list, sherm_list, h3_list, h4_list


def laplace_losvd_uncertainty(A_stack_w: np.ndarray, y_w: np.ndarray, cfg: FitConfig,
                              wave_fit: np.ndarray | None = None, lam: float | None = None,
                              n_samples: int = 300, seed: int | None = None, fit_h3h4: bool = True) -> dict:
    """Laplace posterior of a LOSVD fit at one `lam` (default `cfg.lam`; pass the fit's `lam_used`).

    Refits the MAP at that `lam` on the given whitened `A_stack_w`/`y_w` (drop a robust fit's clipped pixels
    first), computes the Laplace covariance -- of the joint `(LOSVD, continuum)` coefficients when
    `cfg.continuum_knots` is set (then `wave_fit` is required) -- and draws `n_samples` posterior-predictive
    LOSVD curves from it, rejecting draws still at the grid edge (`cfg.max_edge_amplitude_frac`).

    Returns `result0` (the MAP fit), `point0` (its moments), `lam_used`, `b_samples` (unit area),
    `b_median`/`b_std`/`b_p16`/`b_p84`, `vherm_samples`/`sherm_samples`/`h3_samples`/`h4_samples` and their
    `*_std`; with continuum-cofitting also `C_samples`/`C_median`/`C_std`/`C_p16`/`C_p84`. **Read `C`'s shape
    as the meaningful uncertainty, not its absolute level** -- `C`'s mean is not identifiable (see
    `kinextract.continuum`), and `C_median`'s level will generally not match `result0['C']`'s: the draws are
    unit-area in `b` with `C` compensated by the same factor, while `result0['b']` is not unit-area. Compare
    continua mean-subtracted.
    """
    rng = np.random.default_rng(seed)
    lam_val = float(cfg.lam if lam is None else lam)
    C_samples = None
    if cfg.continuum_knots is not None:
        if wave_fit is None:
            raise ValueError("laplace_losvd_uncertainty: cfg.continuum_knots is set but wave_fit was not "
                             "provided -- the continuum spline is built on wavelength.")
        lc = laplace_covariance_at_lam_with_continuum(A_stack_w, y_w, np.asarray(wave_fit, float), cfg, lam_val)
        result0, v_grid = lc["result"], lc["v_grid"]
        point0 = compute_losvd_moments(v_grid, result0["b"], fit_h3h4=fit_h3h4)
        b_samples, C_samples, vherm_s, sherm_s, h3_s, h4_s = _sample_joint_curves_and_moments(
            lc["coeffs"], lc["continuum_coeffs"], lc["cov_joint"], lc["n_coef"], lc["basis"],
            lc["basis_c"], v_grid, n_samples, rng, fit_h3h4=fit_h3h4,
            max_edge_amplitude_frac=cfg.max_edge_amplitude_frac)
        if len(b_samples) == 0:
            b_list, C_list, vherm_list, sherm_list, h3_list, h4_list = _degenerate_point_fallback(
                "laplace_losvd_uncertainty", point0, result0["b"], v_grid, C_point=result0["C"])
            b_samples, C_samples, vherm_s, sherm_s, h3_s, h4_s = (
                b_list[0], C_list[0], vherm_list[0], sherm_list[0], h3_list[0], h4_list[0])
    else:
        lc = laplace_covariance_at_lam(A_stack_w, y_w, cfg, lam_val)
        result0, v_grid = lc["result"], lc["v_grid"]
        point0 = compute_losvd_moments(v_grid, result0["b"], fit_h3h4=fit_h3h4)
        b_samples, vherm_s, sherm_s, h3_s, h4_s = _sample_curves_and_moments(
            lc["coeffs"], lc["cov_coeffs"], lc["basis"], v_grid, n_samples, rng, fit_h3h4=fit_h3h4,
            max_edge_amplitude_frac=cfg.max_edge_amplitude_frac)
        if len(b_samples) == 0:
            b_list, vherm_list, sherm_list, h3_list, h4_list = _degenerate_point_fallback(
                "laplace_losvd_uncertainty", point0, result0["b"], v_grid)
            b_samples, vherm_s, sherm_s, h3_s, h4_s = (
                b_list[0], vherm_list[0], sherm_list[0], h3_list[0], h4_list[0])

    out = dict(
        method="laplace",
        result0=result0, point0=point0, lam_used=lam_val,
        objective_gap_at_best=result0.get("objective_gap"),
        b_samples=b_samples,
        b_median=np.median(b_samples, axis=0),
        b_std=b_samples.std(axis=0, ddof=1),
        b_p16=np.percentile(b_samples, 16, axis=0),
        b_p84=np.percentile(b_samples, 84, axis=0),
        vherm_samples=vherm_s, sherm_samples=sherm_s, h3_samples=h3_s, h4_samples=h4_s,
        vherm_std=float(vherm_s.std(ddof=1)), sherm_std=float(sherm_s.std(ddof=1)),
        h3_std=float(h3_s.std(ddof=1)), h4_std=float(h4_s.std(ddof=1)),
    )
    if C_samples is not None:
        out.update(C_samples=C_samples, C_median=np.median(C_samples, axis=0), C_std=C_samples.std(axis=0, ddof=1),
                   C_p16=np.percentile(C_samples, 16, axis=0), C_p84=np.percentile(C_samples, 84, axis=0))
    return out


def _laplace_losvd_uncertainty_worker(args):
    """Top-level (picklable) per-bin worker for `laplace_losvd_uncertainty_parallel`."""
    A_stack_w, y_w, cfg, wave_fit, lam, n_samples, seed, fit_h3h4 = args
    return laplace_losvd_uncertainty(A_stack_w, y_w, cfg, wave_fit=wave_fit, lam=lam, n_samples=n_samples,
                                     seed=seed, fit_h3h4=fit_h3h4)


def laplace_losvd_uncertainty_parallel(A_stack_w_list, y_w_list, cfg_list, wave_fit=None, lam=None,
                                        n_samples: int = 300, seed: int | None = None, fit_h3h4: bool = True,
                                        n_jobs: int | None = None, clipped_mask=None) -> list[dict]:
    """`laplace_losvd_uncertainty` for many independent bins in parallel (`ProcessPoolExecutor`, `spawn`
    context, BLAS threads pinned to 1 per worker -- see `multi_template.single_threaded_worker_env`).

    Parameters
    ----------
    A_stack_w_list, y_w_list : sequences of the same length, one entry per bin (whitened).
    cfg_list : a single `FitConfig` (reused for every bin) or a sequence of one per bin.
    wave_fit : a single array (shared) or a sequence of one per bin; required for bins with continuum-cofitting.
    lam : `None` (each bin's `cfg.lam`), a float, or a sequence of one per bin -- pass the fits' `lam_used`.
    seed : base seed. Bin `i` uses `seed + i` (or `None` for every bin if `seed` is `None`), so results match a
        serial loop with the same per-bin seeds.
    clipped_mask : optional boolean pixel mask (`True` = clipped), a single array or a sequence of one per bin
        (`None` entries allowed) -- pass each bin's `fit_losvd_robust(...)["clipped_mask"]`. Those pixels are
        dropped from `A_stack_w`/`y_w`/`wave_fit` before the Laplace fit, exactly as the robust fit dropped them.
        Without this the Laplace MAP is refit on the full pixel set, clipped outliers included, and can land on a
        different solution: found on real NGC 4751 WFM data (2026-10-15), 2-6 clipped pixels per bin pulled the
        MAP from sigma ~140-380 km/s to ~520-890 km/s, divergent at the v_grid edges.
    n_jobs : worker count; `None` uses `multi_template.resolve_n_jobs`'s memory-aware default, `1` runs serially.

    Returns a list of `laplace_losvd_uncertainty` result dicts, in input order.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from .multi_template import _limit_worker_threads, resolve_n_jobs, single_threaded_worker_env

    n_bins = len(A_stack_w_list)

    def per_bin(x, name):
        out = list(x) if isinstance(x, (list, tuple)) else [x] * n_bins
        if len(out) != n_bins:
            raise ValueError(f"{name}, given as a sequence, must have one entry per bin (got {len(out)} for "
                             f"{n_bins} bins).")
        return out

    if len(y_w_list) != n_bins:
        raise ValueError(f"A_stack_w_list and y_w_list must have the same length (got {n_bins} and {len(y_w_list)}).")
    cfgs, waves, lams = per_bin(cfg_list, "cfg_list"), per_bin(wave_fit, "wave_fit"), per_bin(lam, "lam")
    masks = per_bin(clipped_mask, "clipped_mask")
    seeds = [None] * n_bins if seed is None else [seed + i for i in range(n_bins)]
    A_stack_w_list, y_w_list = list(A_stack_w_list), list(y_w_list)   # copies: the caller's lists stay untouched
    for i, m in enumerate(masks):
        if m is None:
            continue
        m = np.asarray(m, dtype=bool)
        if m.shape != (A_stack_w_list[i].shape[1],):
            raise ValueError(f"clipped_mask for bin {i} has shape {m.shape}, expected "
                             f"({A_stack_w_list[i].shape[1]},) to match A_stack_w's pixel axis.")
        if m.any():
            A_stack_w_list[i] = A_stack_w_list[i][:, ~m, :]
            y_w_list[i] = np.asarray(y_w_list[i])[~m]
            if waves[i] is not None:
                waves[i] = np.asarray(waves[i])[~m]

    jobs = [(A_stack_w_list[i], y_w_list[i], cfgs[i], waves[i], lams[i], n_samples, seeds[i], fit_h3h4)
            for i in range(n_bins)]
    if n_jobs == 1:
        return [_laplace_losvd_uncertainty_worker(job) for job in jobs]
    # 4x the largest design matrix: draws live in the small coefficient/velocity space, so A_stack_w dominates.
    per_job_bytes = 4 * max((a.nbytes for a in A_stack_w_list), default=0)
    n_jobs = resolve_n_jobs(n_jobs, per_job_bytes)
    with single_threaded_worker_env(), ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("spawn"),
                                                           initializer=_limit_worker_threads) as pool:
        return list(pool.map(_laplace_losvd_uncertainty_worker, jobs))


def losvd_error_bands(A_stack: np.ndarray, flux: np.ndarray, errors: np.ndarray, wave_fit: np.ndarray, fit: dict,
                      cfg: FitConfig, sky_profile: np.ndarray | None = None, n_samples: int = 300,
                      n_moment_draws: int = 100, seed: int | None = 0, fit_h3h4: bool = True) -> dict:
    """Error bands for a `fit_losvd_robust` fit on real IFU data: the sky-line + template-mismatch noise model,
    Laplace at the fit's `lam` and `rho` without its clipped pixels, re-centered on the fit (module docstring).

    Parameters
    ----------
    A_stack, flux, errors, wave_fit : the unwhitened inputs of the fit.
    fit : the `fit_losvd_robust` result (`b`, `lam_used`, `rho_used`, `errors_inflated`, `clipped_mask`).
    cfg : the `FitConfig` of the fit; its `lam` (the lam rule's starting value) is the `lam` of the preliminary
        fit the noise model is measured on.
    sky_profile : the LSF-blended sky-line spectrum on `wave_fit` (`noise_model.sky_line_profile`); `None` for
        data without sky residuals (the mismatch term alone is fitted).
    n_samples : posterior draws; `n_moment_draws` of them give the moment errors.

    Returns
    -------
    dict
        `b_samples`, `b_median`, `b_p16`, `b_p84` (re-centered, unit area); `vherm_err`, `sherm_err`, `h3_err`,
        `h4_err` (half the 16-84 % range of the draws' moments -- robust to the few broad draws); `kappa`, `f`,
        `inflation` (the noise model: sky term in units of the median model, mismatch fraction, median error
        inflation); `lam_used`.
    """
    from .noise_model import noise_model_errors, whiten

    wave_fit = np.asarray(wave_fit, float)
    errors = np.asarray(errors, float)
    sky = np.zeros_like(wave_fit) if sky_profile is None else np.asarray(sky_profile, float)
    nm = noise_model_errors(A_stack, flux, errors, wave_fit, cfg, sky, lam=cfg.lam)
    c = dataclasses.replace(cfg, lam_rule="fixed", lam=float(fit["lam_used"]), rho=float(fit["rho_used"]))
    A_w, y_w, _ = whiten(A_stack, flux, np.asarray(fit["errors_inflated"], float) * nm["errors"] / errors, rho=c.rho)
    keep = ~np.asarray(fit["clipped_mask"], bool)
    lap = laplace_losvd_uncertainty(A_w[:, keep, :], y_w[keep], c, wave_fit=wave_fit[keep], n_samples=n_samples,
                                    seed=seed, fit_h3h4=fit_h3h4)
    v_grid = c.build_velocity_grid()
    r = recenter_samples(lap["b_samples"], v_grid, fit["b"])
    M = np.array([[m["vherm"], m["sherm"], m["h3"], m["h4"]]
                  for m in (compute_losvd_moments(v_grid, x, fit_h3h4=fit_h3h4)
                            for x in r["b_samples"][:n_moment_draws])])
    half = (np.percentile(M, 84, axis=0) - np.percentile(M, 16, axis=0)) / 2
    return dict(r, vherm_err=float(half[0]), sherm_err=float(half[1]), h3_err=float(half[2]), h4_err=float(half[3]),
                kappa=nm["kappa"], f=nm["f"], inflation=nm["inflation"], lam_used=c.lam)


def _losvd_error_bands_worker(args):
    """Top-level (picklable) per-bin worker for `losvd_error_bands_parallel`."""
    A_stack, flux, errors, wave_fit, fit, cfg, sky, kwargs = args
    return losvd_error_bands(A_stack, flux, errors, wave_fit, fit, cfg, sky_profile=sky, **kwargs)


def losvd_error_bands_parallel(A_stack_list, flux_list, errors_list, wave_fit_list, fit_list, cfg_list,
                               sky_profile_list=None, n_jobs: int | None = None, seed: int | None = 0,
                               **kwargs) -> list[dict]:
    """`losvd_error_bands` for many bins in parallel, one process per bin at a time (same pool setup as
    `fit_losvd_robust_parallel`). `cfg_list` and `sky_profile_list` take a single value or one per bin; bin `i`
    uses `seed + i`. Other keyword arguments go to every `losvd_error_bands` call. Returns a list in input order.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from .multi_template import _limit_worker_threads, resolve_n_jobs, single_threaded_worker_env

    n_bins = len(A_stack_list)
    cfgs = cfg_list if isinstance(cfg_list, (list, tuple)) else [cfg_list] * n_bins
    skies = sky_profile_list if isinstance(sky_profile_list, (list, tuple)) else [sky_profile_list] * n_bins
    if not (len(flux_list) == len(errors_list) == len(wave_fit_list) == len(fit_list) == len(cfgs) == len(skies)
            == n_bins):
        raise ValueError("losvd_error_bands_parallel: every per-bin sequence needs one entry per bin.")
    jobs = [(A_stack_list[i], flux_list[i], errors_list[i], wave_fit_list[i], fit_list[i], cfgs[i], skies[i],
             dict(kwargs, seed=None if seed is None else seed + i)) for i in range(n_bins)]
    if n_jobs == 1:
        return [_losvd_error_bands_worker(job) for job in jobs]
    per_job_bytes = 4 * max((a.nbytes for a in A_stack_list), default=0)
    n_jobs = resolve_n_jobs(n_jobs, per_job_bytes)
    with single_threaded_worker_env(), ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("spawn"),
                                                           initializer=_limit_worker_threads) as pool:
        return list(pool.map(_losvd_error_bands_worker, jobs))


def recenter_samples(b_samples: np.ndarray, v_grid: np.ndarray, b_point: np.ndarray, clip: float = 1e3) -> dict:
    """Move posterior draws onto a different point estimate, keeping their relative scatter.

    For bands taken from one posterior (e.g. Laplace with `noise_model.noise_model_errors` errors) around a fit made
    with other errors (the plain-error fit, whose shape is better at fixed lam): each draw is multiplied by
    `b_point / median(draws)` per velocity (both unit-area) -- a shift in log b, the LOSVD's own parametrization --
    clipped to [1/clip, clip], and renormalized to unit area. Returns `b_samples`, `b_median`, `b_p16`, `b_p84`."""
    v = np.asarray(v_grid, float)
    b0 = np.maximum(np.asarray(b_point, float), 0.0)
    b0 = b0 / np.trapezoid(b0, v)
    bs = np.asarray(b_samples, float)
    bs = bs / np.trapezoid(bs, v, axis=1)[:, None]      # unit-area first, so `clip` acts on shape ratios only
    ratio = np.clip(b0 / np.maximum(np.median(bs, axis=0), 1e-300), 1.0 / clip, clip)
    bs = bs * ratio[None, :]
    bs = bs / np.trapezoid(bs, v, axis=1)[:, None]
    return dict(b_samples=bs, b_median=np.median(bs, axis=0), b_p16=np.percentile(bs, 16, axis=0),
                b_p84=np.percentile(bs, 84, axis=0))
