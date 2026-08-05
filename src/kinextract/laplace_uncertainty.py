"""Laplace + AIC-weighted uncertainty -- the package's primary uncertainty
engine (Phase 1 of `dev_notes/V1_HARDENING_ROADMAP.md`), replacing
`bootstrap_losvd` as the recommended default. `bootstrap_losvd` remains
available as a cross-check: it has real properties this method
structurally cannot give (e.g. empirical skew/multimodality across
replicates, and genuine protection against landing in the wrong
optimization basin -- see its own module docstring and "Honest scope"
below).

**Why this replaces bootstrap as the default.** `bootstrap_losvd`
resamples noise around a single, fixed point estimate and refits --
structurally blind to that point estimate's own bias, which is exactly
why its empirical coverage is poor, especially for h4 (measured as low
as ~3-11% against a 68% target -- see `dev_notes/uncertainty/FINDINGS.md`).
This module instead treats `lam` itself as an uncertain hyperparameter:
it computes a Laplace (curvature) approximation to the posterior at
*each* candidate `lam` in `cfg.resolved_lam_grid()`, then combines them
with AIC-derived weights via posterior-predictive sampling. The
between-lam disagreement this picks up is exactly the kind of
uncertainty a single-lam-pinned bootstrap ensemble cannot see by
construction (regardless of `n_boot`) -- validated in the originating
prototype (`dev_notes/uncertainty/laplace_prototype.py`) to give h4
coverage of 56% vs. bootstrap's 6% on the same cases, at a fraction of
the runtime (bootstrap pays for `n_boot` full nonlinear refits; this
pays for one closed-form linear solve per `lam` in the grid, typically
6 values, plus cheap posterior-predictive sampling -- no new
optimizations at all beyond the `lam`-grid fits `aic_local` selection
already has to do).

**The math, in two ingredients:**

1. **Laplace covariance on the LOSVD spline coefficients, at a fixed
   `lam`.** The P-spline roughness penalty already *is* the Bayesian
   equivalent of a Gaussian prior on `coeffs` (Eilers & Marx 1996's own
   framing) -- so the negative-log-*posterior*'s Hessian at the converged
   solution (`losvd._neg_log_post`, the exact objective
   `fit_log_spline_map` minimizes, template weights held at their own
   converged value) gives an approximate posterior covariance in one
   linear solve. This reuses `aic_regularization.estimate_m_eff_local`'s
   own fit and linearization -- no extra fitting cost beyond what
   `select_lambda_aic_local` already pays to choose `lam` in the first
   place.
2. **Posterior-predictive sampling, AIC-weighted across the `lam`
   grid.** At each `lam`, draw `coeffs ~ N(coeffs_MAP, cov_coeffs)`,
   push each draw through `exp(basis @ coeffs)` (unit-area normalized,
   matching `bootstrap_losvd`'s own `b_boot` convention) to get sampled
   LOSVD curves, and read off `(vherm, sherm, h3, h4)` from each sampled
   curve via the same `compute_losvd_moments` estimator used everywhere
   else in this package. Samples are allocated across `lam` values in
   proportion to `softmax(-0.5*(chi2 + w_m*m_eff))` (the same criterion
   `select_lambda_aic_local` uses), then pooled into one combined
   ensemble -- percentiles/medians/stds of *that* pooled ensemble are
   this module's entire "law of total variance" mechanism, no separate
   closed-form propagation formula needed. This is a genuine (if
   approximate) parametric bootstrap from the Laplace-approximated
   posterior, not a linearized moment-propagation trick -- the same
   sampling machinery gives curve-level uncertainty (`b_p16`/`b_p84`/
   `b_median`) and moment-level uncertainty consistently, from one
   mechanism.

**Honest scope, read before trusting this over bootstrap:**

- **Template-weight uncertainty is not modeled.** `w` is NNLS-constrained
  (non-Gaussian) and is held fixed at its converged value for the
  covariance computation -- the same simplification
  `estimate_m_eff_local` already makes for effective degrees of freedom.
  No `w_std`/`w_p16`/`w_p84` fields exist here (unlike `bootstrap_losvd`).
- **Continuum-coefficient uncertainty is not yet supported.**
  `cfg.continuum_knots` must be `None` -- raises `ValueError` otherwise.
  Tracked as a Phase 2 follow-up in the roadmap, once continuum-
  cofitting's own vertical-offset bug is root-caused and fixed (the two
  are coupled: the Hessian needs to extend to the joint
  `(LOSVD coeffs, continuum coeffs)` block, which is more work with an
  unresolved continuum bug still in the picture).
- **Multi-start's own basin-selection risk is not modeled.** This
  computes the Laplace covariance around the single best-found optimum
  at each `lam`, exactly like the bootstrap point estimate's own
  optimizer -- a genuinely different basin winning on a different noise
  draw is invisible to a unimodal Gaussian approximation. `objective_gap`
  (from `fit_log_spline_map`, reported per `lam` via `result["objective_gap"]`)
  is the closest available diagnostic: a small gap means another basin
  was competitive, exactly when this assumption is least trustworthy.

Two entry points:

- `laplace_losvd_uncertainty` -- the full method described above.
  Mirrors `bootstrap_losvd`'s output-dict shape closely enough to be a
  near-drop-in swap: `result0`, `point0`, `b_median`, `b_std`, `b_p16`,
  `b_p84`, `vherm_std`, `sherm_std`, `h3_std`, `h4_std` all mean the same
  thing and can be read the same way.
- `quick_laplace_uncertainty` -- a cheap, single-`lam` variant (no
  AIC-weighting across the grid -- just one Laplace-covariance pass at
  whichever `lam` the point estimate actually used) for fast sanity
  checks, e.g. inside a notebook loop over many bins where the full
  method's cost isn't warranted. Explicitly the "quick error check"
  option, not the recommended default for a reported result.
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .config import FitConfig
from .gh_moments import compute_losvd_moments
from .losvd import _neg_log_post, build_bspline_basis, difference_matrix
from .aic_regularization import estimate_m_eff_local, select_lambda_aic_local

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
    """Fit at a fixed `lam` (reusing `estimate_m_eff_local`'s own fit --
    it already computes `chi2`/`m_eff`/the full `fit_losvd` result at
    this `lam`, so this is not an extra fit), then return the Laplace
    covariance of the spline coefficients at that solution.

    Returns a dict: `result` (the `fit_losvd`-shaped fit result),
    `chi2`, `m_eff`, `coeffs` (the converged coefficients, the Laplace
    mean), `cov_coeffs` (n_coef x n_coef Laplace covariance), `basis`,
    `v_grid` -- everything `_sample_curves_and_moments` needs.
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

    return dict(result=result, chi2=float(chi2), m_eff=float(m_eff), coeffs=coeffs,
                cov_coeffs=cov_coeffs, basis=basis, v_grid=v_grid)


def laplace_covariance_at_lam_with_continuum(A_stack_w: np.ndarray, y_w: np.ndarray, wave_fit: np.ndarray,
                                              cfg: FitConfig, lam: float, n_continuum_knots: int = 4,
                                              lam_c: float = 2.0, ridge_c: float = 0.1) -> dict:
    """`laplace_covariance_at_lam`'s continuum-cofitting counterpart --
    fits `fit_losvd_with_continuum` at a fixed `lam` (continuum-cofitting
    only ever supports a fixed `lam`, adaptive selection is not extended
    to it -- see this module's own "Honest scope" section), then returns
    the Laplace covariance of the *joint* `(LOSVD coeffs, continuum
    coeffs)` vector, `w` held fixed at its converged value -- the same
    simplification `laplace_covariance_at_lam` already makes for `w`
    alone.

    Returns a dict with `result` (the `fit_losvd_with_continuum`-shaped
    fit result), `coeffs`/`continuum_coeffs` (the converged coefficients,
    split back out for convenience), `cov_joint` (the full joint
    covariance -- `cov_joint[:n_coef, :n_coef]` is the LOSVD block,
    `cov_joint[n_coef:, n_coef:]` the continuum block, off-diagonal blocks
    the cross-covariance), `n_coef` (LOSVD coefficient count, for
    splitting `cov_joint`), `basis`/`basis_c` (LOSVD/continuum design
    matrices), `v_grid`.
    """
    from .continuum import fit_losvd_with_continuum

    v_grid = cfg.build_velocity_grid()
    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=cfg.degree)
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    D = difference_matrix(n_coef, order=cfg.penalty_order, dv=dv, dv_ref=cfg.knot_spacing_kms)
    lam_vec = np.full(D.shape[0], lam)

    result = fit_losvd_with_continuum(A_stack_w, y_w, wave_fit, basis, D, lam=lam,
                                       n_continuum_knots=n_continuum_knots, lam_c=lam_c, ridge_c=ridge_c)

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
    unit-area normalize (matching `bootstrap_losvd`'s `b_boot` convention
    -- removes the `(w, b) -> (c*w, b/c)` scale degeneracy), and read off
    `(vherm, sherm, h3, h4)` from each sampled curve via
    `compute_losvd_moments` (not JAX-traceable itself -- calls
    `scipy.optimize.least_squares` for h3/h4 -- so this loop is plain
    Python/numpy, not autodiff/vmap).

    **Rejects divergent-edge draws, matching `max_edge_amplitude_frac`'s
    own existing role in `select_lambda_aic_local` (Stage 23) -- found to
    be necessary, not optional, here too.** The order-3 curvature
    penalty's null space contains divergent solutions (e.g. the mirror-
    image `exp(+av^2)`, `VALIDATION_CAMPAIGN.md` Stage 23) that a weakly-
    regularized `lam`'s point estimate can already avoid by construction
    (multi-start L-BFGS-B just doesn't converge there), but naive
    posterior-predictive *sampling* in coefficient space has no such
    protection: at low `lam`, several coefficient directions are only
    weakly constrained by the data (small `m_eff` relative to `n_coef`),
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
            "was drawn rather than looping forever; consider excluding this lam from the "
            "ensemble (see min_weight_to_sample / max_edge_amplitude_frac in the caller).",
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
    Laplace covariance is meant to characterize -- the same principle
    `bootstrap_losvd`'s own unit-area `b_boot` convention already applies
    to the plain (no-continuum) `(w, b)` scale invariance.
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


def _check_no_continuum(cfg: FitConfig, fn_name: str) -> None:
    if cfg.continuum_knots is not None:
        raise ValueError(
            f"{fn_name}: cfg.continuum_knots is set, but this module does not support "
            "continuum-cofitting yet (see module docstring's 'Honest scope' section -- "
            "tracked as a Phase 2 follow-up in dev_notes/V1_HARDENING_ROADMAP.md)."
        )


def quick_laplace_uncertainty(A_stack_w: np.ndarray, y_w: np.ndarray, cfg: FitConfig,
                               n_samples: int = 200, seed: int | None = None,
                               fit_h3h4: bool = True) -> dict:
    """Cheap, single-`lam` Laplace uncertainty -- no AIC-weighting across
    `cfg.resolved_lam_grid()`, just one Laplace-covariance pass at
    whichever `lam` the point estimate actually uses. The fast "sanity
    check" option; use `laplace_losvd_uncertainty` for a reported result.

    If `cfg.lam_selection == 'aic_local'`, this still has to run the full
    adaptive selection once to find that `lam` (`select_lambda_aic_local`
    -- exactly the same cost `fit_losvd` itself already pays for its own
    point estimate, not extra work this function adds), but only computes
    *one* Laplace covariance/sampling pass afterward, skipping the other
    `len(lam_grid) - 1` grid points' own Hessian+sampling cost that
    `laplace_losvd_uncertainty` pays for its between-lam term.
    """
    _check_no_continuum(cfg, "quick_laplace_uncertainty")
    rng = np.random.default_rng(seed)

    if cfg.lam_selection == "aic_local":
        sel = select_lambda_aic_local(A_stack_w, y_w, cfg, np.asarray(cfg.resolved_lam_grid()))
        lam = sel.best_lam
    else:
        lam = cfg.lam

    lc = laplace_covariance_at_lam(A_stack_w, y_w, cfg, lam)
    result0 = lc["result"]
    v_grid = lc["v_grid"]
    point0 = compute_losvd_moments(v_grid, result0["b"], fit_h3h4=fit_h3h4)

    b_samples, vherm_s, sherm_s, h3_s, h4_s = _sample_curves_and_moments(
        lc["coeffs"], lc["cov_coeffs"], lc["basis"], v_grid, n_samples, rng, fit_h3h4=fit_h3h4,
        max_edge_amplitude_frac=cfg.max_edge_amplitude_frac)
    if len(b_samples) == 0:
        b_list, vherm_list, sherm_list, h3_list, h4_list = _degenerate_point_fallback(
            "quick_laplace_uncertainty", point0, result0["b"], v_grid)
        b_samples, vherm_s, sherm_s, h3_s, h4_s = (
            b_list[0], vherm_list[0], sherm_list[0], h3_list[0], h4_list[0])

    return dict(
        method="quick_laplace",
        result0=result0, point0=point0,
        lam_used=float(lam),
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


def laplace_continuum_uncertainty(A_stack_w: np.ndarray, y_w: np.ndarray, wave_fit: np.ndarray,
                                   cfg: FitConfig, lam: float | None = None, n_continuum_knots: int = 4,
                                   lam_c: float = 2.0, ridge_c: float = 0.1, n_samples: int = 300,
                                   seed: int | None = None, fit_h3h4: bool = True) -> dict:
    """Laplace uncertainty for a continuum-cofitting fit -- the
    continuum-aware counterpart to `quick_laplace_uncertainty` (single,
    fixed `lam` only, no AIC-weighting across `cfg.resolved_lam_grid()`),
    since continuum-cofitting itself only ever supports a fixed `lam`
    (adaptive selection is not extended to it -- see `fit_losvd`'s own
    `UserWarning` and this module's "Honest scope" section). Not a
    special case of `laplace_losvd_uncertainty`: that function's whole
    mechanism is AIC-weighted sampling *across* several `lam` values,
    which continuum-cofitting structurally cannot do yet.

    Computes the Laplace covariance of the *joint* `(LOSVD coeffs,
    continuum coeffs)` vector (`laplace_covariance_at_lam_with_
    continuum`), then draws posterior-predictive `(b, C)` curve pairs
    from it (`_sample_joint_curves_and_moments`) -- `w` held fixed at its
    converged value, matching every other Laplace function in this
    module.

    `lam` : if `None` (default), uses `cfg.lam` directly (continuum-
        cofitting's own convention -- see `fit_losvd_with_continuum`'s
        own docstring; `cfg.lam_selection` is not consulted, since
        adaptive selection doesn't apply here).

    Returns a dict shaped like `quick_laplace_uncertainty`'s own (same
    `result0`, `point0`, `lam_used`, `b_samples`/`b_median`/`b_std`/
    `b_p16`/`b_p84`, `vherm_samples`/etc. and their `_std` summaries),
    plus continuum-specific fields: `C_samples`, `C_median`, `C_std`,
    `C_p16`, `C_p84`. **Read `C`'s shape (relative variation) as the
    meaningful uncertainty here, not its absolute level** -- `C`'s mean
    is not identifiable from the model alone regardless of how well
    constrained its shape is (see `kinextract.continuum`'s own module
    docstring, root-caused in `dev_notes/continuum/FINDINGS.md`'s
    2026-08-01 entry) -- `C_std`/`C_p16`/`C_p84` inherit that same
    non-identifiability at the DC/mean level, not just the point estimate.

    **`C_median`'s own absolute level will generally NOT match
    `result0['C']`'s** -- expected, not a bug. `b_samples` are
    canonicalized to unit area (matching every other engine's `b_median`
    convention in this package), and `C_samples` are compensated by the
    exact same per-draw factor to keep each draw's own likelihood
    unchanged (`_sample_joint_curves_and_moments`'s own docstring has the
    full derivation) -- but `result0['b']` itself is *not* unit-area
    (`fit_losvd_with_continuum` never canonicalizes it), so `result0['C']`
    and `C_median` live on two different, equally arbitrary points along
    the same non-identifiable gauge direction. Comparing `C_median` to a
    known true continuum (e.g. a validation mock) should always be done
    mean-subtracted (`C_median - C_median.mean()` vs `true_C -
    true_C.mean()`), exactly like `continuum.py`'s own recommended
    diagnostic -- never by comparing raw levels.
    """
    rng = np.random.default_rng(seed)
    lam_val = cfg.lam if lam is None else lam

    lc = laplace_covariance_at_lam_with_continuum(
        A_stack_w, y_w, wave_fit, cfg, lam_val, n_continuum_knots=n_continuum_knots,
        lam_c=lam_c, ridge_c=ridge_c)
    result0 = lc["result"]
    v_grid = lc["v_grid"]
    point0 = compute_losvd_moments(v_grid, result0["b"], fit_h3h4=fit_h3h4)

    b_samples, C_samples, vherm_s, sherm_s, h3_s, h4_s = _sample_joint_curves_and_moments(
        lc["coeffs"], lc["continuum_coeffs"], lc["cov_joint"], lc["n_coef"], lc["basis"],
        lc["basis_c"], v_grid, n_samples, rng, fit_h3h4=fit_h3h4,
        max_edge_amplitude_frac=cfg.max_edge_amplitude_frac)
    if len(b_samples) == 0:
        b_list, C_list, vherm_list, sherm_list, h3_list, h4_list = _degenerate_point_fallback(
            "laplace_continuum_uncertainty", point0, result0["b"], v_grid, C_point=result0["C"])
        b_samples, C_samples, vherm_s, sherm_s, h3_s, h4_s = (
            b_list[0], C_list[0], vherm_list[0], sherm_list[0], h3_list[0], h4_list[0])

    return dict(
        method="laplace_continuum",
        result0=result0, point0=point0,
        lam_used=float(lam_val),
        b_samples=b_samples,
        b_median=np.median(b_samples, axis=0),
        b_std=b_samples.std(axis=0, ddof=1),
        b_p16=np.percentile(b_samples, 16, axis=0),
        b_p84=np.percentile(b_samples, 84, axis=0),
        C_samples=C_samples,
        C_median=np.median(C_samples, axis=0),
        C_std=C_samples.std(axis=0, ddof=1),
        C_p16=np.percentile(C_samples, 16, axis=0),
        C_p84=np.percentile(C_samples, 84, axis=0),
        vherm_samples=vherm_s, sherm_samples=sherm_s, h3_samples=h3_s, h4_samples=h4_s,
        vherm_std=float(vherm_s.std(ddof=1)), sherm_std=float(sherm_s.std(ddof=1)),
        h3_std=float(h3_s.std(ddof=1)), h4_std=float(h4_s.std(ddof=1)),
    )


def laplace_losvd_uncertainty(A_stack_w: np.ndarray, y_w: np.ndarray, cfg: FitConfig,
                               w_m: float = 2.0, n_samples: int = 300, seed: int | None = None,
                               fit_h3h4: bool = True, min_weight_to_sample: float = 0.01) -> dict:
    """The full method: AIC-weighted posterior-predictive sampling across
    `cfg.resolved_lam_grid()`. See module docstring for the math and
    "Honest scope" section.

    `w_m` : the same AIC criterion weight (`chi2 + w_m*m_eff`)
        `select_lambda_aic_local` uses -- kept consistent so the `lam`
        this function's own point estimate lands on matches what
        `fit_losvd(cfg)` (adaptive selection) would pick.
    `n_samples` : total posterior-predictive sample budget, allocated
        across `lam` grid points in proportion to their AIC weight (a
        `lam` with weight below `min_weight_to_sample` is skipped
        entirely -- not worth a Hessian/sampling pass for a candidate the
        criterion has already all but ruled out).

    Returns a dict shaped to be a near-drop-in swap for `bootstrap_losvd`'s
    return value: `result0`, `point0`, `b_median`, `b_std`, `b_p16`,
    `b_p84`, `vherm_std`, `sherm_std`, `h3_std`, `h4_std` mean the same
    thing and can be read the same way. `b_samples`/`vherm_samples`/etc.
    (not `b_boot`/`vherm_boot`) hold the raw pooled posterior-predictive
    draws -- named differently from `bootstrap_losvd`'s `*_boot` fields on
    purpose, since these are not bootstrap replicates. `lam_used`,
    `lam_grid`, `lam_weights` are new fields specific to this method.
    No bias-correction fields (`*_corrected`) -- `bias_correction.py`'s
    calibration tables were built against `bootstrap_losvd`'s own point
    estimate and have not been validated against this method's.
    """
    _check_no_continuum(cfg, "laplace_losvd_uncertainty")
    rng = np.random.default_rng(seed)
    lam_grid = np.asarray(cfg.resolved_lam_grid(), dtype=float)

    per_lam = [laplace_covariance_at_lam(A_stack_w, y_w, cfg, lam) for lam in lam_grid]
    criteria = np.array([lc["chi2"] + w_m * lc["m_eff"] for lc in per_lam])
    weights = np.exp(-0.5 * (criteria - criteria.min()))
    weights /= weights.sum()

    best_idx = int(np.argmin(criteria))
    result0 = per_lam[best_idx]["result"]
    v_grid = per_lam[best_idx]["v_grid"]
    point0 = compute_losvd_moments(v_grid, result0["b"], fit_h3h4=fit_h3h4)

    b_list, vherm_list, sherm_list, h3_list, h4_list = [], [], [], [], []
    n_sampled_lams = 0
    for i, lc in enumerate(per_lam):
        if weights[i] < min_weight_to_sample:
            continue
        n_i = max(1, int(round(weights[i] * n_samples)))
        b_s, vherm_s, sherm_s, h3_s, h4_s = _sample_curves_and_moments(
            lc["coeffs"], lc["cov_coeffs"], lc["basis"], v_grid, n_i, rng, fit_h3h4=fit_h3h4,
            max_edge_amplitude_frac=cfg.max_edge_amplitude_frac)
        # If the divergent-edge rejection filter (_sample_curves_and_moments's own
        # docstring) exhausted most of its draw budget, the surviving handful of
        # samples is a small, unrepresentative slice of this lam's own posterior --
        # its std can land anywhere, including artificially tight, which showed up
        # directly in a coverage-sweep standardized-residual breakdown as a small
        # number of wildly overconfident cells rather than a smooth trend (see
        # dev_notes/uncertainty/FINDINGS.md, 2026-08-01). Excluding this lam's
        # contribution entirely once too few of its own requested samples survived
        # (rather than pooling the unreliable partial sample) trades a slightly
        # smaller effective sample size for not letting one bad lam's own
        # sampling noise dominate the pooled ensemble's reported width.
        if len(b_s) < 0.5 * n_i:
            warnings.warn(
                f"laplace_losvd_uncertainty: lam={lam_grid[i]:g} only produced "
                f"{len(b_s)}/{n_i} requested posterior-predictive samples (its own "
                "posterior is dominated by divergent-edge solutions) -- excluding this "
                "lam's contribution from the pooled ensemble rather than pooling an "
                "unrepresentative partial sample. If this fires often, the affected "
                "lam values are contributing little useful signal to this fit's "
                "uncertainty band regardless.", UserWarning, stacklevel=2,
            )
            continue
        b_list.append(b_s); vherm_list.append(vherm_s); sherm_list.append(sherm_s)
        h3_list.append(h3_s); h4_list.append(h4_s)
        n_sampled_lams += 1

    if n_sampled_lams == 0:
        # min_weight_to_sample excluded every lam, or every lam's own sampling yield was
        # too low to trust (the per-lam loop above's own 0.5*n_i floor) -- fall back to
        # sampling the winner alone rather than returning an empty ensemble.
        warnings.warn(
            "laplace_losvd_uncertainty: every lam in the grid fell below min_weight_to_sample "
            "-- falling back to sampling only the winning lam. This should not normally happen; "
            "consider lowering min_weight_to_sample.", UserWarning, stacklevel=2,
        )
        lc = per_lam[best_idx]
        b_s, vherm_s, sherm_s, h3_s, h4_s = _sample_curves_and_moments(
            lc["coeffs"], lc["cov_coeffs"], lc["basis"], v_grid, n_samples, rng, fit_h3h4=fit_h3h4,
            max_edge_amplitude_frac=cfg.max_edge_amplitude_frac)
        b_list, vherm_list, sherm_list, h3_list, h4_list = [b_s], [vherm_s], [sherm_s], [h3_s], [h4_s]

    if sum(len(b) for b in b_list) == 0:
        # Even the winning lam's own posterior-predictive sampling produced zero valid
        # draws (found directly running at real-data scale, 2026-08-03) -- see
        # _degenerate_point_fallback's own docstring for why this needs handling here.
        b_list, vherm_list, sherm_list, h3_list, h4_list = _degenerate_point_fallback(
            "laplace_losvd_uncertainty", point0, result0["b"], v_grid)

    b_samples = np.concatenate(b_list, axis=0)
    vherm_samples = np.concatenate(vherm_list)
    sherm_samples = np.concatenate(sherm_list)
    h3_samples = np.concatenate(h3_list)
    h4_samples = np.concatenate(h4_list)

    return dict(
        method="laplace",
        result0=result0, point0=point0,
        lam_used=float(lam_grid[best_idx]), lam_grid=lam_grid, lam_weights=weights,
        objective_gap_at_best=result0.get("objective_gap"),
        b_samples=b_samples,
        b_median=np.median(b_samples, axis=0),
        b_std=b_samples.std(axis=0, ddof=1),
        b_p16=np.percentile(b_samples, 16, axis=0),
        b_p84=np.percentile(b_samples, 84, axis=0),
        vherm_samples=vherm_samples, sherm_samples=sherm_samples,
        h3_samples=h3_samples, h4_samples=h4_samples,
        vherm_std=float(vherm_samples.std(ddof=1)), sherm_std=float(sherm_samples.std(ddof=1)),
        h3_std=float(h3_samples.std(ddof=1)), h4_std=float(h4_samples.std(ddof=1)),
    )


def _laplace_losvd_uncertainty_worker(args):
    """Top-level (picklable) per-bin worker for
    `laplace_losvd_uncertainty_parallel` -- process pools need a
    module-level function, not a closure, since workers unpickle it by
    import path (same requirement as `uncertainty._bootstrap_losvd_worker`).
    Dispatches to the continuum-aware path when this bin's own `cfg` has
    `continuum_knots` set, and to the plain (AIC-weighted-across-lam)
    path otherwise -- see `laplace_losvd_uncertainty_parallel`'s own
    docstring for why these are genuinely different methods, not one
    function with a flag.
    """
    (A_stack_w, y_w, cfg, seed, w_m, n_samples, fit_h3h4, min_weight_to_sample,
     wave_fit, lam, n_continuum_knots, lam_c, ridge_c) = args
    if cfg.continuum_knots is not None:
        return laplace_continuum_uncertainty(
            A_stack_w, y_w, wave_fit, cfg, lam=lam, n_continuum_knots=n_continuum_knots,
            lam_c=lam_c, ridge_c=ridge_c, n_samples=n_samples, seed=seed, fit_h3h4=fit_h3h4,
        )
    return laplace_losvd_uncertainty(
        A_stack_w, y_w, cfg, w_m=w_m, n_samples=n_samples, seed=seed,
        fit_h3h4=fit_h3h4, min_weight_to_sample=min_weight_to_sample,
    )


def laplace_losvd_uncertainty_parallel(A_stack_w_list, y_w_list, cfg_list, w_m: float = 2.0,
                                        n_samples: int = 300, seed: int | None = None,
                                        fit_h3h4: bool = True, min_weight_to_sample: float = 0.01,
                                        n_jobs: int | None = None, wave_fit=None, lam=None,
                                        n_continuum_knots: int = 4, lam_c: float = 2.0,
                                        ridge_c: float = 0.1) -> list[dict]:
    """Laplace uncertainty for many independent bins in parallel via a
    process pool -- the Laplace analogue of `uncertainty.
    bootstrap_losvd_parallel`, same interface and parallelization
    strategy (`ProcessPoolExecutor`, `spawn` context, BLAS threads pinned
    to 1 per worker via `multi_template._limit_worker_threads` -- see
    that function's own docstring for why).

    Each bin's Laplace fit is already far cheaper per-bin than a full
    bootstrap ensemble (no `n_boot` refits -- one closed-form covariance
    per `lam` in the grid, typically 6 values, plus cheap posterior-
    predictive sampling -- see `laplace_losvd_uncertainty`'s own module
    docstring), so this parallelizes the *same* embarrassingly-parallel
    across-bins structure `bootstrap_losvd_parallel` does, for the same
    reason: many independent bins, zero shared state, pure wall-clock
    speedup with zero accuracy cost relative to a serial per-bin loop
    with the same per-bin seed.

    **Per-bin continuum dispatch, not a single uniform method.** A bin
    whose own `cfg.continuum_knots` is set is routed to
    `laplace_continuum_uncertainty` (fixed-`lam` joint LOSVD+continuum
    Laplace covariance) instead of `laplace_losvd_uncertainty` (AIC-
    weighted across `cfg.resolved_lam_grid()`) -- these are genuinely
    different methods (continuum-cofitting only ever supports a fixed
    `lam`; see `laplace_continuum_uncertainty`'s own docstring), not one
    function with a continuum flag. Mixing continuum and non-continuum
    bins in the same call is supported (each bin's own `cfg` decides),
    matching how `cfg_list` already lets every other per-bin setting vary.

    Parameters
    ----------
    A_stack_w_list, y_w_list : sequences of the same length, one entry per
        bin, each matching `laplace_losvd_uncertainty`'s own `A_stack_w`/
        `y_w`.
    cfg_list : a single `FitConfig` (reused for every bin) or a sequence
        of one per bin.
    seed : base seed. Bin `i` uses `seed + i` (or `None` for every bin,
        drawing fresh entropy each time, if `seed` itself is `None`) --
        matches `bootstrap_losvd_parallel`'s own convention, so results
        are directly comparable to (and, at `n_jobs=1`, bit-identical to)
        a serial `for i, ...: laplace_losvd_uncertainty(..., seed=seed+i)`
        loop.
    w_m, n_samples, fit_h3h4, min_weight_to_sample : shared across every
        non-continuum bin, forwarded to `laplace_losvd_uncertainty`.
    wave_fit, lam, n_continuum_knots, lam_c, ridge_c : shared across every
        continuum bin, forwarded to `laplace_continuum_uncertainty`.
        `wave_fit` accepts either a single array (shared across every
        continuum bin) or a sequence of one per bin, matching
        `bootstrap_losvd_parallel`'s own `wave_fit` convention. Ignored
        entirely for bins with `continuum_knots=None`.
    n_jobs : worker process count. `None` (default) uses `os.cpu_count()`,
        memory-permitting -- see `multi_template.resolve_n_jobs`'s own
        docstring. `1` disables the process pool and runs serially
        in-process.

    Returns
    -------
    list of result dicts, one per bin, in the same order as the inputs
    (each either `laplace_losvd_uncertainty`'s or `laplace_continuum_
    uncertainty`'s own shape, depending on that bin's `cfg`) -- not a
    generator, so a failure partway through surfaces before any results
    are used.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from .multi_template import _limit_worker_threads, resolve_n_jobs

    n_bins = len(A_stack_w_list)
    if len(y_w_list) != n_bins:
        raise ValueError(f"A_stack_w_list and y_w_list must have the same length "
                          f"(got {n_bins} and {len(y_w_list)}).")
    cfgs = cfg_list if isinstance(cfg_list, (list, tuple)) else [cfg_list] * n_bins
    if len(cfgs) != n_bins:
        raise ValueError(f"cfg_list, given as a sequence, must have one entry per bin "
                          f"(got {len(cfgs)} for {n_bins} bins).")
    waves = wave_fit if isinstance(wave_fit, (list, tuple)) else [wave_fit] * n_bins
    if len(waves) != n_bins:
        raise ValueError(f"wave_fit, given as a sequence, must have one entry per bin "
                          f"(got {len(waves)} for {n_bins} bins).")
    for cfg, wf in zip(cfgs, waves):
        if cfg.continuum_knots is not None and wf is None:
            raise ValueError(
                "laplace_losvd_uncertainty_parallel: a bin has cfg.continuum_knots set but no "
                "wave_fit was provided for it -- continuum-cofitting needs the observed "
                "wavelength grid (see laplace_continuum_uncertainty's own docstring)."
            )
    seeds = [None] * n_bins if seed is None else [seed + i for i in range(n_bins)]

    jobs = [
        (A_stack_w_list[i], y_w_list[i], cfgs[i], seeds[i], w_m, n_samples, fit_h3h4,
         min_weight_to_sample, waves[i], lam, n_continuum_knots, lam_c, ridge_c)
        for i in range(n_bins)
    ]

    if n_jobs == 1:
        return [_laplace_losvd_uncertainty_worker(job) for job in jobs]

    # 4x the largest single design matrix -- posterior-predictive samples are drawn
    # over the (small) coefficient/velocity-grid space, not copies of A_stack_w itself,
    # so the base design matrix still dominates a job's own memory footprint (see
    # resolve_n_jobs's own docstring for why this cap is memory-aware, not just cpu_count()).
    per_job_bytes = 4 * max((a.nbytes for a in A_stack_w_list), default=0)
    n_jobs = resolve_n_jobs(n_jobs, per_job_bytes)
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("spawn"),
                              initializer=_limit_worker_threads) as pool:
        return list(pool.map(_laplace_losvd_uncertainty_worker, jobs))
