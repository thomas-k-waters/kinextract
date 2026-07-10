"""
losvd_errors.py
================
Robust error bar estimation for non-parametric LOSVDs produced by the
spectral_fitting pipeline.

Three complementary methods are provided and can be combined:

  1. Laplace / penalized-likelihood covariance  (fast, ~seconds)
       Computes the posterior covariance of the full parameter vector at
       the MAP solution via finite-difference Hessian.  Accounts for the
       smoothing prior correctly in the Bayesian sense.  Underestimates
       frequentist errors if the noise model is wrong or the model is
       misspecified, but is cheap and useful as a lower bound.

  2. Residual bootstrap  (moderate cost, ~minutes with parallelism)
       Draws B synthetic spectra by adding rescaled, block-shuffled
       residuals to the best-fit model, then refits each one through the
       full pipeline (LOSVD + templates, optionally the joint continuum
       co-fit).  Gives frequentist intervals that automatically propagate
       regularization bias, continuum uncertainty, and template-weight
       uncertainty.
       This is the statistically most honest approach.

  3. Bias-corrected LOSVD  (analytic, nearly free once Hessian is known)
       The smoothing penalty shrinks b toward flatness.  The hat matrix
       H of the linearized penalized problem lets us estimate the
       regularization bias and subtract it from b_map.  The bias-
       corrected LOSVD is closer to the truth in expectation, though
       noisier.

  4. GH moment errors  (propagated from bootstrap or Laplace b-covariance)
       Propagates the LOSVD covariance through the Gauss-Hermite moment
       extraction via the delta method, giving sigma on (V, sigma, h3, h4).

Usage
-----
    from losvd_errors import LOSVDErrorEstimator

    # After running the main fit:
    fit = run_spectral_fit(cfg, gal_file=spectrum)

    est = LOSVDErrorEstimator(fit, cfg)

    # Fast lower bound (always run this first):
    laplace = est.laplace_covariance()

    # Gold standard (set n_bootstrap to at least 200 for publication):
    boot = est.residual_bootstrap(n_bootstrap=200, n_jobs=4)

    # Combine into a summary:
    summary = est.summarize(laplace_result=laplace, bootstrap_result=boot)
    est.plot_losvd_with_errors(summary)

Design notes
------------
- The Hessian is computed by *finite differences on the objective*, which
  is the only robust option given that _compute_smoothness is Numba-JIT
  and _convolve_losvd_numba is also JIT.  We use a central 2nd-order
  stencil with adaptive step size.

- For the bootstrap, each synthetic spectrum is fitted with write_outputs=False
  and a silenced logger so the run is quiet and fast.

- Block residual resampling (contiguous blocks of ~correlation_length pixels)
  preserves the correlated noise structure that arises from sky subtraction
  and interpolation onto a regular wavelength grid.

- Continuum-cofit bootstrap replicates (cfg.fit_continuum=True) refit via a
  single frozen-hyperparameter kinextract.joint.fit_joint call at the main
  fit's own converged xlam/sigl0/v_center rather than re-running the
  expensive auto-selecting drivers per replicate -- see
  _refit_one_bootstrap_joint's docstring.

- All results are in the native units of b (the LOSVD vector on the xl grid)
  and of the GH moments (km/s for V and sigma, dimensionless for h3/h4).
"""

from __future__ import annotations

import copy
import dataclasses
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    _THREADPOOLCTL_AVAILABLE = True
except ImportError:
    _threadpool_limits = None
    _THREADPOOLCTL_AVAILABLE = False
    warnings.warn(
        "threadpoolctl is not installed.  During parallel bootstrap, BLAS "
        "libraries (OpenBLAS/MKL) may spawn extra internal threads per worker, "
        "causing CPU oversubscription and highly variable per-replicate timing "
        "(observed range: 1-180 s per replicate on 10-core nodes).  "
        "Install with: pip install threadpoolctl",
        RuntimeWarning,
        stacklevel=1,
    )

# ── imports from the main pipeline ───────────────────────────────────────────
# These are imported at use-time inside functions so this module can be
# imported even if spectral_fitting is not on the path yet.
# ---------------------------------------------------------------------------
from .fitting import fit_state_map_with_optional_clean
from .losvd import fit_losvd_gauss_hermite, fit_losvd_gauss_hermite_higher
from .numerics import (
    _build_jax_objective_value_and_grad,
    evaluate_model_gp,
    jax,
    objective_map,
)
from .spectrum import build_initial_guess_nonparam
from .state import FitState

# =============================================================================
# Helpers
# =============================================================================

def _b_slice(st) -> slice:
    """Slice selecting the nl LOSVD bins from the full parameter vector."""
    return slice(0, st.nl)


def _w_slice(st) -> slice:
    """Slice selecting the nt template weights."""
    return slice(st.nl, st.nl + st.nt)


def _active_bound_mask(
    a: np.ndarray, bounds, atol: float = 1e-8, rtol: float = 1e-6,
) -> np.ndarray:
    """Boolean mask, True where a parameter sits at (or numerically
    indistinguishable from) its lower or upper optimization bound.

    Used to condition the Laplace covariance on the optimizer's active set
    (see :meth:`LOSVDErrorEstimator.laplace_covariance`): a parameter pinned
    at a bound is not free to vary in both directions, so a symmetric
    quadratic (Gaussian) approximation to its posterior is not meaningful,
    and including it in the joint Hessian inversion can corrupt the
    marginal variance of nearby *free* parameters through cross terms.
    """
    a = np.asarray(a, float)
    lb = np.array([lo for lo, _ in bounds], float)
    ub = np.array([hi for _, hi in bounds], float)
    at_lower = np.isclose(a, lb, atol=atol, rtol=rtol)
    at_upper = np.isclose(a, ub, atol=atol, rtol=rtol)
    return at_lower | at_upper


def _projected_gradient(
    grad: np.ndarray, a: np.ndarray, bounds, atol: float = 1e-8, rtol: float = 1e-6,
) -> np.ndarray:
    """KKT-projected gradient at a bound-constrained point.

    At an active lower (upper) bound, only a positive (negative) gradient
    component is consistent with a valid constrained optimum -- it pushes
    further into the infeasible region, which the bound correctly blocks,
    so that component is zeroed before taking the norm. Any other nonzero
    component (including a "wrong-signed" one at an active bound) reflects
    genuine remaining curvature the optimizer has not resolved. This is the
    standard stationarity measure for bound-constrained optima and is used
    here purely as a diagnostic: a large projected-gradient norm at the
    reported MAP solution means the Hessian computed there does not
    correspond to a true local minimum, so the Laplace approximation built
    from it should not be trusted.
    """
    grad = np.asarray(grad, float).copy()
    a = np.asarray(a, float)
    lb = np.array([lo for lo, _ in bounds], float)
    ub = np.array([hi for _, hi in bounds], float)
    at_lower = np.isclose(a, lb, atol=atol, rtol=rtol)
    at_upper = np.isclose(a, ub, atol=atol, rtol=rtol)
    grad[at_lower & (grad > 0)] = 0.0
    grad[at_upper & (grad < 0)] = 0.0
    return grad


def _require_genuine_map_fit(fit: dict, method_name: str) -> None:
    """Raise a clear error if ``fit`` came from the optional Bayesian path
    (:func:`kinextract.bayesian.fit_state_bayesian`) rather than a genuine
    MAP/L-BFGS-B optimization.

    ``laplace_covariance`` and ``bias_correction`` both require ``fit["a_map"]``
    to sit at (or very near) a true stationary point of the *full* penalized
    objective -- the former builds and inverts a Hessian there under that
    assumption, checking it via a projected-gradient diagnostic; the latter's
    hat-matrix bias estimate carries the same "this is what a MAP fit would
    have converged to" framing. The Bayesian path's output stores
    the NUTS **posterior mean** under the ``fit["a_map"]``/
    ``fit["outputs"]["b"]`` keys -- and by construction (see
    :mod:`kinextract.bayesian`'s module docstring), the posterior mean is
    generically *not* at a stationary point of the penalized objective
    whenever the posterior is skewed -- exactly the regime (near the
    instrumental resolution limit) where this distinction matters most.
    Calling these methods on such a fit does not crash outright: the
    existing projected-gradient check merely emits a generic "MAP did not
    fully converge" warning and produces a Hessian-based covariance that
    is not statistically meaningful, with no indication that the actual
    cause is conceptual rather than a convergence-tolerance problem
    (retrying with tighter ``map_ftol``/``map_gtol`` cannot fix it, since
    no MAP optimization is being run at all on that path).

    Detected via ``fit["result"].mcmc is not None`` (only set by the
    Bayesian path; a MAP-path ``scipy.optimize.OptimizeResult``-like
    object has no ``mcmc`` attribute at all).
    """
    result = fit.get("result")
    if getattr(result, "mcmc", None) is not None:
        raise ValueError(
            f"{method_name} requires a genuine MAP-mode fit, but this fit's "
            f"result carries posterior samples (fit['result'].mcmc is not "
            f"None) -- it came from the optional Bayesian path "
            f"(kinextract.bayesian.fit_state_bayesian), whose reported point "
            f"estimate is the posterior *mean*, not the MAP *mode*. A "
            f"Hessian computed at the posterior mean is not a valid Laplace "
            f"approximation (see the kinextract.bayesian module docstring). "
            f"For error bars on this fit, use "
            f"kinextract.plotting.plot_losvd_posterior() (built directly "
            f"from the NUTS posterior draws, no stationarity assumption "
            f"needed). If you specifically want a Laplace/hat-matrix "
            f"comparison, obtain a genuine MAP fit first via "
            f"kinextract.fitting.fit_state_map_with_optional_clean() and "
            f"pass that fit dict instead. residual_bootstrap() is unaffected "
            f"and can still be used directly on this fit."
        )


def _require_not_joint(estimator: "LOSVDErrorEstimator", method_name: str) -> None:
    """Raise a clear error if `estimator` wraps a joint-mode fit.

    Unlike :meth:`LOSVDErrorEstimator.residual_bootstrap` (which refits from
    scratch via :mod:`kinextract.joint` and so works for either fit type),
    ``laplace_covariance``/``bias_correction`` both need a Hessian or hat
    matrix of the *shipped* objective (:func:`kinextract.numerics.objective_map`)
    evaluated at ``a_map`` -- meaningless for a joint-mode ``a_map``, whose
    entries are ``[LOSVD bins, template weights, continuum P-spline
    coefficients]`` under a different objective entirely.
    """
    if getattr(estimator, "is_joint", False):
        raise NotImplementedError(
            f"{method_name} does not yet support joint-mode fits "
            f"(cfg.fit_continuum=True or cfg.joint_prenorm=True; "
            f"kinextract.joint): it needs a Hessian/hat-matrix of the "
            f"shipped objective_map at a_map, which doesn't apply to the "
            f"joint model's own objective. residual_bootstrap() is "
            f"unaffected and works directly on this fit."
        )


def _make_frozen_cfg(cfg):
    """
    Return a copy of cfg with relaxed tolerances for fast bootstrap refits.

    Bootstrap refits start from the MAP warm-start and just need a
    good-enough solution, not MAP-level precision -- tight tolerances cause
    slowdowns on pathological bootstrap spectra without improving error
    estimates.
    """
    c = copy.deepcopy(cfg)
    c.print_every = 0
    c.map_ftol = max(float(getattr(cfg, "map_ftol", 1e-12)), 1e-8)
    c.map_gtol = max(float(getattr(cfg, "map_gtol", 1e-10)), 1e-6)
    # Sigma-clipping converges in 1-2 passes from a warm start; the config
    # default (25) is appropriate for the initial MAP fit but wastes time
    # on bootstrap draws.
    c.clean_maxiter = min(int(getattr(cfg, "clean_maxiter", 25)), 3)
    # Auto-xlam selection runs once per spectrum on the real data.  Bootstrap
    # refits must use the same xlam (already stored in cfg.xlam after the main
    # fit), not re-search per replicate which would add noise to error estimates.
    c.xlam_auto = False
    # Workers never call JAX (use_jax_objective=False), so forked XLA locks are
    # never acquired and the inherited lock state cannot deadlock.  Keep False.
    c.use_jax_objective = False
    return c


def _make_frozen_cfg_joint(cfg):
    """Joint-mode analogue of :func:`_make_frozen_cfg`.

    Relaxes optimizer tolerances for speed on bootstrap replicates (a
    good-enough solution is fine; MAP-level precision on every one of
    hundreds of replicates isn't). There is no xlam/sigl0/v_center grid
    search to disable here -- those are frozen implicitly by reusing the
    already-converged ``st.xlam``/``st.sigl0``/``st.v_center`` from the
    main fit (see :func:`kinextract.joint.run_joint_fit`) and calling
    :func:`kinextract.joint.fit_joint` directly (a single fit at fixed
    hyperparameters, not the auto-selecting drivers) per replicate.
    """
    c = copy.deepcopy(cfg)
    c.print_every = 0
    c.map_ftol = max(float(getattr(cfg, "map_ftol", 1e-12)), 1e-8)
    c.map_gtol = max(float(getattr(cfg, "map_gtol", 1e-10)), 1e-6)
    return c


def _losvd_moments(xl: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return the flux-weighted centroid and dispersion of a LOSVD."""
    xl = np.asarray(xl, float)
    b = np.clip(np.asarray(b, float), 0.0, np.inf)
    total = float(np.sum(b))
    if not np.isfinite(total) or total <= 0:
        return float("nan"), float("nan")
    weights = b / total
    v = float(np.sum(xl * weights))
    sigma = float(np.sqrt(np.sum(weights * (xl - v) ** 2)))
    return v, sigma


# =============================================================================
# Section 1 – Hessian via finite differences
# =============================================================================

def compute_hessian_fd(
    objective_fn,
    a_map: np.ndarray,
    rel_step: float = 1e-4,
    abs_step_floor: float = 1e-8,
) -> np.ndarray:
    """
    Compute the Hessian of objective_fn at a_map by central finite differences.

    Uses a 2nd-order central stencil:
        H[i,j] = (f(+ei+ej) - f(+ei-ej) - f(-ei+ej) + f(-ei-ej)) / (4 hi hj)

    The diagonal is computed more accurately with the 3-point formula:
        H[i,i] = (f(+ei) - 2*f0 + f(-ei)) / hi^2

    Parameters
    ----------
    objective_fn : callable
        f(a) -> float.  Should be the full MAP objective including the
        smoothness and norm penalties.
    a_map : ndarray, shape (nparam,)
        MAP solution (the point at which to evaluate the Hessian).
    rel_step : float
        Step size as a fraction of |a_map[i]| (or abs_step_floor if smaller).
    abs_step_floor : float
        Minimum absolute step size.

    Returns
    -------
    H : ndarray, shape (nparam, nparam)
        Symmetric Hessian matrix.  Symmetrized as (H + H.T) / 2 before
        returning to suppress numerical asymmetry.
    """
    a = np.asarray(a_map, float).copy()
    n = len(a)
    H = np.zeros((n, n), float)
    f0 = float(objective_fn(a))

    steps = np.maximum(rel_step * np.abs(a), abs_step_floor)

    # Diagonal terms — 3-point formula (more accurate than cross-term formula)
    diag_vals = np.empty(n, float)
    for i in range(n):
        hi = steps[i]
        ai_orig = a[i]
        a[i] = ai_orig + hi
        fp = float(objective_fn(a))
        a[i] = ai_orig - hi
        fm = float(objective_fn(a))
        a[i] = ai_orig
        diag_vals[i] = (fp - 2 * f0 + fm) / (hi * hi)
        H[i, i] = diag_vals[i]

    # Off-diagonal terms — 4-point cross formula
    for i in range(n):
        hi = steps[i]
        for j in range(i + 1, n):
            hj = steps[j]
            ai_o, aj_o = a[i], a[j]

            a[i] = ai_o + hi; a[j] = aj_o + hj
            fpp = float(objective_fn(a))

            a[i] = ai_o + hi; a[j] = aj_o - hj
            fpm = float(objective_fn(a))

            a[i] = ai_o - hi; a[j] = aj_o + hj
            fmp = float(objective_fn(a))

            a[i] = ai_o - hi; a[j] = aj_o - hj
            fmm = float(objective_fn(a))

            a[i] = ai_o; a[j] = aj_o
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * hi * hj)

    # Symmetrize to suppress numerical noise
    H = (H + H.T) / 2.0
    return H


def compute_hessian_from_grad_fd(
    grad_fn,
    a_map: np.ndarray,
    rel_step: float = 1e-4,
    abs_step_floor: float = 1e-8,
) -> np.ndarray:
    """
    Compute the Hessian from central finite differences of an analytic gradient.

    This needs only ``2*n`` gradient evaluations versus the ``O(n^2)``
    objective evaluations used by :func:`compute_hessian_fd`, and is much
    faster whenever an analytic (e.g. JAX-computed) gradient of the MAP
    objective is available. Used by
    :meth:`LOSVDErrorEstimator.laplace_covariance` as the preferred path
    when a JAX gradient function can be built, falling back to
    :func:`compute_hessian_fd` otherwise.

    Parameters
    ----------
    grad_fn : callable
        Function ``grad_fn(a) -> ndarray`` returning the analytic gradient
        of the MAP objective at parameter vector ``a``.
    a_map : ndarray, shape (nparam,)
        MAP solution (the point at which to evaluate the Hessian).
    rel_step : float, optional
        Step size as a fraction of ``|a_map[i]|`` (or ``abs_step_floor`` if
        smaller).
    abs_step_floor : float, optional
        Minimum absolute step size.

    Returns
    -------
    H : ndarray, shape (nparam, nparam)
        Symmetric Hessian matrix, symmetrized as ``(H + H.T) / 2`` to
        suppress numerical asymmetry from the finite-difference evaluation.
    """
    a = np.asarray(a_map, float).copy()
    n = len(a)
    H = np.zeros((n, n), float)
    steps = np.maximum(rel_step * np.abs(a), abs_step_floor)

    for i in range(n):
        hi = steps[i]
        ai_orig = a[i]

        a[i] = ai_orig + hi
        gp = np.asarray(grad_fn(a), float)

        a[i] = ai_orig - hi
        gm = np.asarray(grad_fn(a), float)

        a[i] = ai_orig
        H[:, i] = (gp - gm) / (2.0 * hi)

    H = (H + H.T) / 2.0
    return H


# =============================================================================
# Section 2 – Laplace covariance
# =============================================================================

def laplace_posterior_covariance(
    H: np.ndarray,
    param_scale: Optional[np.ndarray] = None,
    regularize_nugget: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Compute the Laplace-approximation posterior covariance C = H^{-1}.

    The Hessian of the *penalized* objective at the MAP point is the
    precision matrix of the Laplace approximation to the posterior
    p(a | data) under the implicit prior encoded by the smoothing and
    norm penalties.

    Because b is constrained to be non-negative and some template weights
    may be at their lower bound, H can be singular or near-singular.  We
    regularize by adding a small nugget to the diagonal before inversion.

    Parameters
    ----------
    H : ndarray (n, n)
        Hessian of the penalized objective.
    param_scale : ndarray (n,), optional
        Parameter scales used during optimisation (xscale from
        build_parameter_xscale).  If supplied, H is assumed to be in
        the *scaled* parameter space and is transformed back before
        inversion:  H_phys[i,j] = H_scaled[i,j] / (s[i]*s[j]).
    regularize_nugget : float
        Fraction of the mean diagonal added to H before inversion.

    Returns
    -------
    C       : ndarray (n, n)  Posterior covariance in physical units.
    H_reg   : ndarray (n, n)  Regularized Hessian that was actually inverted.
    is_pd   : bool            True if H was positive definite before regularization.
    """
    H = np.asarray(H, float)
    n = H.shape[0]

    if param_scale is not None:
        s = np.asarray(param_scale, float)
        # H_phys[i,j] = H_scaled[i,j] / (s[i] * s[j])
        H = H / np.outer(s, s)

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(H)
    is_pd = bool(np.all(eigvals > 0))

    # Regularize: add nugget to diagonal
    nugget = regularize_nugget * max(float(np.mean(np.diag(H))), 1e-12)
    H_reg = H + nugget * np.eye(n)

    try:
        C = np.linalg.inv(H_reg)
        # Symmetrize
        C = (C + C.T) / 2.0
    except np.linalg.LinAlgError:
        warnings.warn(
            "Hessian inversion failed; returning diagonal approximation.",
            RuntimeWarning,
            stacklevel=2,
        )
        diag_safe = np.where(np.diag(H_reg) > 0, np.diag(H_reg), 1.0)
        C = np.diag(1.0 / diag_safe)

    return C, H_reg, is_pd


# =============================================================================
# Section 3 – Hat matrix and bias correction
# =============================================================================

def compute_losvd_hat_matrix(st, a_map: np.ndarray) -> np.ndarray:
    """
    Compute the linearized influence (hat) matrix H_b for the LOSVD subspace.

    For fixed template weights and continuum parameters, the model is
    *linear* in the LOSVD vector b (through the convolution).  In this
    linearized problem the penalized estimate satisfies:

        b_map = H_b @ b_true + noise_term

    where H_b = (J^T W J + P)^{-1} J^T W J and:
        J     = Jacobian of model w.r.t. b (npix × nl)
        W     = diagonal weight matrix (1/gerr^2 for good pixels, 0 for masked)
        P     = smoothness penalty Hessian w.r.t. b (the nl × nl matrix of
                second-difference weights, position-dependent per _compute_smoothness)

    The diagonal of H_b gives the effective degrees of freedom per LOSVD bin.
    The bias on b_map is:  bias = (H_b - I) @ b_true ≈ (H_b - I) @ b_map

    Parameters
    ----------
    st    : FitState
    a_map : ndarray  Full MAP parameter vector.

    Returns
    -------
    H_b : ndarray (nl, nl)
        Linearized influence matrix for the LOSVD bins.
    """
    from kinextract import evaluate_model_gp

    nl = st.nl
    a = np.asarray(a_map, float).copy()

    # Numerical Jacobian of the model spectrum w.r.t. b[0..nl-1]
    # Shape: (npix, nl)
    gp0, *_ = evaluate_model_gp(a, st)
    step = 1e-5
    J = np.zeros((st.npix, nl), float)
    for k in range(nl):
        a_p = a.copy(); a_p[k] += step
        a_m = a.copy(); a_m[k] -= step
        gp_p, *_ = evaluate_model_gp(a_p, st)
        gp_m, *_ = evaluate_model_gp(a_m, st)
        J[:, k] = (gp_p - gp_m) / (2 * step)

    # Weight matrix: only good pixels contribute
    good = (st.gerr < 1e9) & np.isfinite(st.gerr) & (st.gerr > 0)
    w = np.where(good, 1.0 / st.gerr ** 2, 0.0)

    # J^T W J  — (nl, nl)
    JtWJ = (J * w[:, None]).T @ J

    # Smoothness penalty Hessian w.r.t. b — (nl, nl)
    # This is the matrix form of _compute_smoothness.
    # Penalty = sum_j lam_j * (b[j+1] - 2*b[j] + b[j-1])^2 / resd
    # = b^T P b / resd   (up to boundary terms)
    # We build P by finite differencing the penalty gradient.
    # Build P analytically from the second-difference structure
    # smooth = sum_j lam_j * (D2 b)_j^2 / resd
    # P = D2^T * diag(lam_j) * D2 / resd
    sfac = 1.8
    lam_vec = np.array([
        st.xlam * (max(abs(st.xl[j]) / st.sigl0, sfac) / sfac) ** 4
        if abs(st.xl[j]) > sfac * st.sigl0
        else st.xlam
        for j in range(nl)
    ])
    # Build second-difference matrix D2 (nl-2, nl) with boundary modifications
    # matching _compute_smoothness exactly
    D2 = np.zeros((nl, nl), float)
    for j in range(nl):
        if j == 0:
            D2[j, 0] = -2.0
            D2[j, 1] =  1.0
        elif j == nl - 1:
            D2[j, nl - 2] =  1.0
            D2[j, nl - 1] = -2.0
        else:
            D2[j, j - 1] =  1.0
            D2[j, j    ] = -2.0
            D2[j, j + 1] =  1.0

    P = D2.T @ np.diag(lam_vec) @ D2 / st.resd

    # Hat matrix: H_b = (J^T W J + P)^{-1} J^T W J
    A_mat = JtWJ + P
    try:
        H_b = np.linalg.solve(A_mat, JtWJ)
    except np.linalg.LinAlgError:
        nugget = 1e-8 * np.trace(A_mat) / nl
        H_b = np.linalg.solve(A_mat + nugget * np.eye(nl), JtWJ)

    return H_b


def bias_corrected_losvd(
    b_map: np.ndarray, H_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a bias-corrected LOSVD and the estimated regularization bias.

    The MAP estimate satisfies (in expectation):
        E[b_map] = H_b @ b_true

    So an approximately unbiased estimate is:
        b_corr = H_b^{-1} @ b_map

    This can be numerically unstable when H_b is nearly singular (i.e.,
    when the smoothing is very strong).  We use a truncated pseudo-inverse
    via SVD, dropping singular values below tol * sigma_max.

    Warning
    -------
    Actively harmful when the true LOSVD is only weakly identified --
    specifically, when the velocity dispersion is comparable to the
    instrument's LSF width (see :class:`~kinextract.config.FitConfig`'s
    "Known limitations" section). In that regime this correction amplifies
    noise catastrophically (recovered h3/h4 pinned at their fit bounds,
    velocity bias several times *larger* than the uncorrected MAP estimate)
    rather than reducing bias.
    The truncated-SVD threshold does not adequately regularize ``H_b`` when
    it is this close to singular. Prefer this correction only when ``sigma``
    is comfortably above (roughly ``>= 2x``) the instrument's LSF width.

    Parameters
    ----------
    b_map : ndarray (nl,)   MAP LOSVD.
    H_b   : ndarray (nl,nl) Influence matrix from compute_losvd_hat_matrix.

    Returns
    -------
    b_corr : ndarray (nl,)  Bias-corrected LOSVD (clipped to >= 0).
    bias   : ndarray (nl,)  Estimated regularization bias = b_map - H_b @ b_corr.
    """
    # Pseudo-inverse via SVD with threshold
    U, s, Vt = np.linalg.svd(H_b)
    tol = 1e-3 * s[0]
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    H_b_pinv = Vt.T @ np.diag(s_inv) @ U.T
    b_corr = H_b_pinv @ b_map
    b_corr = np.maximum(b_corr, 0.0)  # enforce non-negativity
    bias = b_map - H_b @ b_corr
    return b_corr, bias


# =============================================================================
# Section 4 – Residual bootstrap
# =============================================================================

def _draw_bootstrap_spectrum(
    rng: np.random.Generator,
    g_data: np.ndarray,
    gp_model: np.ndarray,
    gerr: np.ndarray,
    block_size: int = 1,
) -> np.ndarray:
    """
    Draw one bootstrap replicate spectrum.

    Method: block resampling of standardized residuals.

    1. Compute standardized residuals r = (g - gp) / gerr on good pixels.
    2. Resample r in blocks of block_size (preserving local correlation).
    3. Construct g* = gp + gerr * r_resampled.

    Masked pixels (gerr >= BIG) keep their original data values — they
    will be masked again during the refit anyway.

    Parameters
    ----------
    rng        : numpy Generator
    g_data     : ndarray (npix,)  Observed spectrum.
    gp_model   : ndarray (npix,)  Best-fit model spectrum.
    gerr       : ndarray (npix,)  Noise spectrum (BIG for masked pixels).
    block_size : int  Block length for block bootstrap (>1 for correlated noise).

    Returns
    -------
    g_boot : ndarray (npix,)  Synthetic spectrum for one bootstrap replicate.
    """
    BIG = 1e9
    npix = len(g_data)
    good = (gerr < BIG) & np.isfinite(g_data) & np.isfinite(gp_model)
    n_good = int(np.count_nonzero(good))

    resid_std = np.zeros(npix, float)
    resid_std[good] = (g_data[good] - gp_model[good]) / gerr[good]

    # Block bootstrap on the good-pixel residuals
    good_idx = np.flatnonzero(good)
    if block_size <= 1:
        # IID resample
        chosen = rng.integers(0, n_good, size=n_good)
        resid_boot = resid_std[good_idx[chosen]]
    else:
        # Stationary block bootstrap (simplified: fixed blocks)
        n_blocks = int(np.ceil(n_good / block_size))
        starts = rng.integers(0, max(n_good - block_size + 1, 1), size=n_blocks)
        indices = np.concatenate([
            good_idx[s:s + block_size] for s in starts
        ])[:n_good]
        resid_boot = resid_std[indices]

    g_boot = gp_model.copy()
    g_boot[good] = gp_model[good] + gerr[good] * resid_boot

    # Leave masked pixels at their original values (they get masked again)
    g_boot[~good] = g_data[~good]

    return g_boot


def _refit_one_bootstrap(
    args: tuple,
) -> Optional[dict]:
    """
    Worker function for one bootstrap replicate.  Designed for use with
    multiprocessing.Pool.map (must be picklable, so it's a module-level
    function rather than a closure).

    Parameters (packed into args tuple for pickling)
    -------------------------------------------------
    g_boot  : ndarray (npix,)  Synthetic spectrum.
    cfg     : FitConfig        Configuration (relaxed tolerances; see _make_frozen_cfg).
    st_ref  : FitState         Reference FitState (used to copy fixed arrays;
                               g and gerr are replaced with g_boot and base_gerr).
    a_map   : ndarray          MAP solution (used as warm start).
    bounds  : list             Parameter bounds.
    seed    : int              RNG seed for this replicate (for reproducibility).

    Returns
    -------
    dict with keys: b, w, gh  (LOSVD bins, template weights, GH moments)
    or None on failure.
    """
    # sf is imported at module level from kinextract

    g_boot, cfg_frozen, base_gerr, a_map_ref, bounds, seed, st_fields = args

    rng = np.random.default_rng(seed)

    # Reconstruct a lightweight FitState from the serialized fields.
    # We shallow-copy the reference fields dict and replace the mutable ones.
    try:
        st = FitState(**st_fields)
        st.g = np.asarray(g_boot, float).copy()
        st.gerr = np.asarray(base_gerr, float).copy()
        st.ntot = 0

        # Warm-start from the MAP solution with a small random perturbation
        a0 = np.asarray(a_map_ref, float).copy()
        a0 += rng.normal(0, 1e-4, size=len(a0))
        a0 = np.clip(a0, [lo for lo, _ in bounds], [hi for _, hi in bounds])

        res = fit_state_map_with_optional_clean(
            st, cfg_frozen, a0, bounds,
        )

        if res is None:
            return None

        a_boot = np.asarray(res.x, float)
        gp, b, w, coff, coff2, A = evaluate_model_gp(a_boot, st)

        gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
        gh_ho = fit_losvd_gauss_hermite_higher(st.xl, b, max_order=6)

        result = {
            "b": b.copy(),
            "w": w.copy(),
            "gh_vherm": gh["vherm"],
            "gh_sherm": gh["sherm"],
            "gh_h3": gh["h3"],
            "gh_h4": gh["h4"],
            "gh_v1": gh["v1"],
            "gh_v2": gh["v2"],
            "converged": res.success,
        }
        for k in ["vherm", "sherm", "h3", "h4", "h5", "h6"]:
            result[f"gh_ho_{k}"] = gh_ho.get(k, np.nan)
        return result

    except Exception as exc:
        warnings.warn(f"Bootstrap replicate failed: {exc}", RuntimeWarning, stacklevel=1)
        return None


def _refit_one_bootstrap_joint(args: tuple) -> Optional[dict]:
    """
    Worker function for one bootstrap replicate under the joint
    continuum-in-the-model engine (:mod:`kinextract.joint`).

    Mirrors :func:`_refit_one_bootstrap`'s structure and return shape
    exactly, so :meth:`LOSVDErrorEstimator.residual_bootstrap`'s downstream
    percentile/GH-moment/``.sim``-file code needs no changes regardless of
    which worker produced the samples -- only *how one replicate is
    refit* differs.

    Refits via a single, frozen-hyperparameter
    :func:`kinextract.joint.fit_joint` call rather than the auto-selecting
    :func:`kinextract.joint.fit_joint_auto_xlam_sigl0` driver used for the
    original fit. The xlam/sigl0/v_center auto-selection machinery is
    deliberately *not* re-run per replicate: :func:`kinextract.joint.run_joint_fit`
    freezes the converged values onto ``st.xlam``/``st.sigl0``/``st.v_center``
    after the main fit, and re-selecting them per bootstrap replicate would
    cost ``n_sigl0_iter * len(xlam_grid)`` optimizations per replicate
    (hundreds of times the shipped path's single refit) while injecting
    extra hyperparameter-selection noise into the error bars rather than
    reflecting genuine data noise at fixed, already-decided regularization
    -- exactly the same rationale :func:`_make_frozen_cfg` already applies
    to the shipped path's ``xlam_auto`` search.

    Parameters (packed into args tuple for pickling)
    -------------------------------------------------
    g_boot       : ndarray (npix,)  Synthetic spectrum.
    cfg_frozen   : FitConfig        Relaxed-tolerance config (see
                                     :func:`_make_frozen_cfg_joint`).
    base_gerr    : ndarray (npix,)  Original per-pixel errors.
    st_fields    : dict             Serialized FitState fields (``xlam``/
                                     ``sigl0``/``v_center`` already carry the
                                     converged values from the main fit).
    joint_kwargs : dict             ``n_interior_knots``/``degree``/
                                     ``xlam_cont``/``cont_diff_order``/
                                     ``fit_continuum``, from
                                     ``LOSVDErrorEstimator.__init__``.
    seed         : int              Unused placeholder, kept only so this
                                     worker's args tuple has the same shape/
                                     position convention as
                                     :func:`_refit_one_bootstrap`'s.

    Returns
    -------
    dict with keys: b, w, gh_* , converged -- same shape as
    :func:`_refit_one_bootstrap`'s return value.
    or None on failure.
    """
    from .joint import evaluate_model_gp_joint, fit_joint, normalize_template_matrix

    g_boot, cfg_frozen, base_gerr, st_fields, joint_kwargs, seed = args

    try:
        st = FitState(**st_fields)
        st.g = np.asarray(g_boot, float).copy()
        st.gerr = np.asarray(base_gerr, float).copy()
        st.ntot = 0

        result, design = fit_joint(
            st,
            n_interior_knots=joint_kwargs["n_interior_knots"],
            degree=joint_kwargs["degree"],
            xlam_cont=joint_kwargs["xlam_cont"],
            cont_diff_order=joint_kwargs["cont_diff_order"],
            maxiter=cfg_frozen.map_maxiter, ftol=cfg_frozen.map_ftol,
            maxfun=cfg_frozen.map_maxfun, use_jax=cfg_frozen.use_jax_objective,
            auto_recenter_v=False, fit_continuum=joint_kwargs["fit_continuum"],
        )

        t_norm, _ = normalize_template_matrix(st)
        gp, b, w, cont_coeffs, cont = evaluate_model_gp_joint(result.x, st, design, t_norm)

        gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
        gh_ho = fit_losvd_gauss_hermite_higher(st.xl, b, max_order=6)

        result_dict = {
            "b": b.copy(),
            "w": w.copy(),
            "gh_vherm": gh["vherm"],
            "gh_sherm": gh["sherm"],
            "gh_h3": gh["h3"],
            "gh_h4": gh["h4"],
            "gh_v1": gh["v1"],
            "gh_v2": gh["v2"],
            "converged": result.success,
        }
        for k in ["vherm", "sherm", "h3", "h4", "h5", "h6"]:
            result_dict[f"gh_ho_{k}"] = gh_ho.get(k, np.nan)
        return result_dict

    except Exception as exc:
        warnings.warn(f"Bootstrap replicate failed: {exc}", RuntimeWarning, stacklevel=1)
        return None


def _fit_state_to_fields(st) -> dict:
    """
    Extract the picklable fields of a FitState for multiprocessing.

    All array-like values (including JAX DeviceArrays) are converted to plain
    numpy arrays before packing.  This is critical when using a spawn-context
    Pool: JAX arrays carry a reference to the XLA runtime, so unpickling them
    in a fresh worker process triggers JAX initialization.  When many workers
    start simultaneously they race to initialize XLA and deadlock.  Converting
    everything to numpy here breaks that dependency entirely.
    """
    d = {}
    for f in dataclasses.fields(st):
        val = getattr(st, f.name)
        if val is None or isinstance(val, (bool, int, float, str)):
            d[f.name] = val
        elif isinstance(val, list):
            d[f.name] = val
        else:
            # Coerce anything array-like (numpy OR jax) to a plain numpy copy.
            try:
                d[f.name] = np.asarray(val).copy()
            except Exception:
                d[f.name] = val
    return d


# =============================================================================
# Section 5 – Main estimator class
# =============================================================================

class LOSVDErrorEstimator:
    """
    Unified interface for estimating uncertainties on a recovered LOSVD.

    Wraps the output of a completed ``run_spectral_fit()`` -- a fit that
    recovers a non-parametric line-of-sight velocity distribution ``b`` on
    the velocity grid ``xl``, together with template weights ``w`` -- and
    provides three complementary, combinable error estimation strategies
    plus Gauss-Hermite (GH) moment error propagation. ``run_spectral_fit()``'s
    default output is a genuine MAP/penalized-likelihood fit, so
    :meth:`laplace_covariance` and :meth:`bias_correction` work directly on
    it. If instead handed a fit produced by the optional, non-default
    full-posterior path (:func:`kinextract.bayesian.fit_state_bayesian`,
    whose reported point estimate is the NUTS posterior *mean*, not a MAP
    *mode*), those two methods raise a clear ``ValueError`` rather than
    silently computing a meaningless Hessian at a non-stationary point
    (see :func:`_require_genuine_map_fit`) -- use
    :func:`~kinextract.plotting.plot_losvd_posterior` for error bars on
    such a fit instead. :meth:`residual_bootstrap` works either way (it
    always refits fresh via the MAP machinery internally, regardless of
    what kind of fit it was constructed from):

    1. **Laplace / penalized-likelihood covariance** (:meth:`laplace_covariance`)
       — fast (~seconds), computes a finite-difference Hessian of the MAP
       objective and inverts it to obtain the Bayesian posterior covariance
       under the Laplace approximation. Cheap and always worth running
       first, but likely underestimates true (frequentist) errors if the
       noise model or regularization strength is imperfect. Requires a
       genuine MAP fit (see above).

    2. **Residual bootstrap** (:meth:`residual_bootstrap`) — the most
       statistically honest but most expensive method (minutes, embarrassingly
       parallel via ``n_jobs``). Draws many synthetic spectra by adding
       resampled residuals (optionally in contiguous blocks, to preserve
       correlated noise from sky subtraction/resampling) to the best-fit
       model, refits each synthetic spectrum through the *full* pipeline
       (including, for continuum-cofit fits, the joint continuum), and uses
       the resulting ensemble of recovered LOSVDs/GH moments to build
       frequentist error bars and percentile intervals.

    3. **Bias-corrected LOSVD** (:meth:`bias_correction`) — analytic and
       nearly free once the Hessian is available. The roughness/smoothing
       penalty used during the MAP fit shrinks the recovered LOSVD toward
       flatness (regularization bias); the hat (influence) matrix of the
       linearized penalized problem is used to estimate and subtract this
       bias, at the cost of amplified noise in the corrected LOSVD. Requires
       a genuine MAP fit (see above); also see
       :class:`~kinextract.config.FitConfig`'s "Known limitations" section --
       this correction was separately found to be actively harmful near the
       instrumental resolution limit, independent of the MAP-vs-Bayesian
       question here.

    4. **Gauss-Hermite moment errors** — propagates the LOSVD covariance
       (from either the Laplace or bootstrap methods) through the GH moment
       extraction (:func:`kinextract.losvd.fit_losvd_gauss_hermite`) via the
       delta method, giving uncertainties directly on the physically
       interpretable moments (V, sigma, h3, h4) rather than just on the raw
       LOSVD bins.

    :meth:`summarize` combines whichever of these methods were run into a
    single dict with a recommended central estimate and error bar (preferring
    bootstrap over Laplace when both are available), and
    :meth:`plot_losvd_with_errors` visualizes the result.

    Parameters
    ----------
    fit : dict
        Output of ``run_spectral_fit()`` for the fit whose errors are to be
        estimated. Must contain ``fit["state"]`` (the
        :class:`~kinextract.state.FitState`), ``fit["a_map"]`` (the full MAP
        parameter vector), and ``fit["outputs"]`` with keys ``"b"``
        (recovered LOSVD), ``"w"`` (template weights), and ``"gp"``
        (best-fit model spectrum).
    cfg : FitConfig
        Configuration object used to produce the original fit. Reused as
        the starting point for bootstrap refits (with relaxed optimizer
        tolerances — see ``_make_frozen_cfg``/``_make_frozen_cfg_joint``)
        and to rebuild parameter bounds identical to those used in the
        original fit.

    Attributes
    ----------
    fit : dict
        The fit dict passed to the constructor.
    cfg : FitConfig
        The configuration passed to the constructor.
    st : FitState
        The FitState of the original fit (``fit["state"]``).
    a_map : ndarray
        Full MAP parameter vector (LOSVD bins, template weights, continuum
        offsets, etc.), as a flat array.
    b_map : ndarray, shape (nl,)
        MAP (best-fit) non-parametric LOSVD amplitudes on the velocity grid
        ``xl``.
    w_map : ndarray, shape (nt,)
        MAP template weights.
    gp_map : ndarray, shape (npix,)
        MAP best-fit model spectrum.
    xl : ndarray, shape (nl,)
        LOSVD velocity grid (km/s).
    bounds : list of (float, float)
        Parameter bounds for the full parameter vector, rebuilt identically
        to those used in the original MAP fit; used to keep bootstrap
        refits within the same feasible region.
    """

    def __init__(self, fit: dict, cfg):
        """Initialize the estimator from a completed fit and its configuration.

        Extracts the MAP parameter vector, recovered LOSVD, template
        weights, and best-fit model spectrum from ``fit``, and rebuilds the
        parameter bounds used during the original fit (via
        :func:`~kinextract.spectrum.build_initial_guess_nonparam`) so that
        subsequent bootstrap refits (:meth:`residual_bootstrap`) are
        constrained identically to the original MAP optimization.

        Parameters
        ----------
        fit : dict
            Output of ``run_spectral_fit()``. See the class docstring for
            required keys.
        cfg : FitConfig
            Configuration object used for the original fit.
        """
        # sf is imported at module level from kinextract

        self.is_joint = cfg.fit_continuum or getattr(cfg, "joint_prenorm", False)

        self.fit = fit
        self.cfg = cfg
        self.st: FitState = fit["state"]
        self.a_map: np.ndarray = np.asarray(fit["a_map"], float)
        self.b_map: np.ndarray = fit["outputs"]["b"].copy()
        self.w_map: np.ndarray = fit["outputs"]["w"].copy()
        self.gp_map: np.ndarray = fit["outputs"]["gp"].copy()
        self.xl: np.ndarray = self.st.xl.copy()

        if self.is_joint:
            # laplace_covariance/bias_correction still don't support the
            # joint parameter layout (see their own guards below) -- only
            # residual_bootstrap does, via _refit_one_bootstrap_joint. Build
            # the joint continuum design/bounds once here (design depends
            # only on st.x, which doesn't change across bootstrap replicates)
            # rather than every worker rebuilding it from scratch.
            from .joint import build_initial_guess as _joint_build_initial_guess
            from .joint import normalize_template_matrix as _joint_normalize_template_matrix

            fit_continuum = bool(cfg.fit_continuum)
            _, joint_bounds, _, joint_design = _joint_build_initial_guess(
                self.st, cfg.joint_n_interior_knots, cfg.joint_degree,
                fit_continuum=fit_continuum,
            )
            self.bounds = joint_bounds
            self._joint_design = joint_design
            self._joint_t_norm, _ = _joint_normalize_template_matrix(self.st)
            self._joint_fit_continuum = fit_continuum
            self._joint_kwargs = dict(
                n_interior_knots=cfg.joint_n_interior_knots, degree=cfg.joint_degree,
                xlam_cont=cfg.joint_xlam_cont, cont_diff_order=cfg.joint_cont_diff_order,
                fit_continuum=fit_continuum,
            )
        else:
            # Build bounds (same as used during the original fit)
            _, xlb, xub = build_initial_guess_nonparam(self.st, cfg.coff, cfg.coff2)
            self.bounds = list(zip(xlb, xub))

    # ------------------------------------------------------------------
    # Method 1: Laplace covariance
    # ------------------------------------------------------------------

    def laplace_covariance(
        self,
        rel_step: float = 1e-4,
        regularize_nugget: float = 1e-6,
        grad_warn_threshold: float = 0.05,
    ) -> dict:
        """
        Compute the Laplace-approximation posterior covariance at a_map.

        The Laplace approximation is only meaningful at a genuine local
        minimum of the *free* (non-bound-constrained) parameters: this
        method (1) checks that the MAP solution is actually converged in
        that sense and warns if not, since an under-converged solution can
        make the Hessian indefinite and its inverse statistically
        meaningless, and (2) conditions the covariance on the optimizer's
        active set -- LOSVD bins or template weights pinned at a bound are
        excluded from the matrix that gets inverted (see
        :func:`_active_bound_mask`), rather than being included and
        potentially corrupting the marginal variance of nearby free
        parameters through cross terms. Pinned parameters are reported with
        exactly zero variance and zero covariance with everything else --
        the standard "conditional on the active set" convention, not a
        claim of infinite precision (see ``pinned_mask_b``/``pinned_mask_w``
        in the returned dict to tell the two apart from a genuinely
        well-constrained free parameter).

        Any *free* parameter whose posterior variance still comes out
        negative (a residual numerical pathology, not expected once pinned
        parameters are excluded) is reported as ``NaN`` in ``b_err``/
        ``w_err`` with a warning, rather than being silently clipped to
        zero as in previous versions of this method -- a deceptively
        "perfectly precise" zero and a genuine "we don't have a reliable
        estimate here" are very different statistical statements.

        Parameters
        ----------
        rel_step : float, optional
            Finite-difference step size (as a fraction of ``|a_map[i]|``)
            used to build the Hessian; see :func:`compute_hessian_fd` /
            :func:`compute_hessian_from_grad_fd`.
        regularize_nugget : float, optional
            Diagonal regularization applied to the free-parameter
            sub-Hessian before inversion; see
            :func:`laplace_posterior_covariance`.
        grad_warn_threshold : float, optional
            If the largest KKT-projected gradient component among the free
            parameters exceeds this value, a warning is raised that the MAP
            solution may not be sufficiently converged for the Laplace
            approximation to be trustworthy (see :func:`_projected_gradient`).

        Returns
        -------
        dict with keys:
            C_full        : (nparam, nparam) full posterior covariance
                             (zero rows/columns at pinned parameters)
            C_b           : (nl, nl) marginal covariance of b
            C_w           : (nt, nt) marginal covariance of w
            b_err         : (nl,) posterior std dev of each LOSVD bin
                             (0 at pinned bins, NaN if inversion produced a
                             negative variance for a free bin)
            w_err         : (nt,) posterior std dev of each template weight
            H             : (nparam, nparam) Hessian (physical units)
            is_pd         : bool -- True if the *free-parameter* sub-Hessian
                             was PD before regularization
            pinned_mask_b : (nl,) bool -- True where a LOSVD bin is pinned
                             at a bound and excluded from the inversion
            pinned_mask_w : (nt,) bool -- same, for template weights
            free_grad_max_abs : float -- largest KKT-projected gradient
                             component among free parameters (convergence
                             diagnostic; see grad_warn_threshold)
            gh_err        : dict of GH moment errors via delta method
            moments_err   : dict of flux-weighted moment errors
        """
        _require_genuine_map_fit(self.fit, "laplace_covariance()")
        _require_not_joint(self, "laplace_covariance()")
        # sf is imported at module level from kinextract

        print("[LOSVDErrors] Computing Hessian...")
        t0 = time.perf_counter()

        H = None
        grad_at_map = None
        if _build_jax_objective_value_and_grad is not None:
            try:
                if getattr(self.cfg, "jax_enable_x64", True) and jax is not None:
                    jax.config.update("jax_enable_x64", True)
                vg = _build_jax_objective_value_and_grad(self.st)
                if vg is not None:
                    print("[LOSVDErrors] Using JAX gradient-backed Hessian FD (CPU)")

                    def grad_fn(a):
                        _g = np.asarray(self.st.g, float)
                        _gerr = np.asarray(self.st.gerr, float)
                        _cont = np.asarray(
                            getattr(self.st, "continuum_mult", np.ones(self.st.npix)), float
                        )
                        _, g = vg(np.asarray(a, float), _g, _gerr, _cont)
                        return g

                    grad_at_map = np.asarray(grad_fn(self.a_map), float)
                    H = compute_hessian_from_grad_fd(grad_fn, self.a_map, rel_step=rel_step)
            except Exception as exc:
                print(f"[LOSVDErrors] JAX Hessian path unavailable ({exc}); falling back to objective FD")

        if H is None:
            print("[LOSVDErrors] Using objective finite-difference Hessian")

            def obj(a):
                return objective_map(a, self.st)

            H = compute_hessian_fd(obj, self.a_map, rel_step=rel_step)

            # Central-difference gradient at a_map for the convergence check
            # (the objective-only Hessian path above doesn't produce one).
            a0 = np.asarray(self.a_map, float)
            steps = np.maximum(rel_step * np.abs(a0), 1e-8)
            grad_at_map = np.empty(len(a0), float)
            for i in range(len(a0)):
                a_p = a0.copy(); a_p[i] += steps[i]
                a_m = a0.copy(); a_m[i] -= steps[i]
                grad_at_map[i] = (obj(a_p) - obj(a_m)) / (2.0 * steps[i])

        # objective_map() returns chi2 + penalty terms.  The Laplace
        # approximation uses the Hessian (and gradient) of the negative log
        # posterior, which is 0.5 * (chi2 + penalty) under the usual
        # Gaussian-noise convention.  Without this factor the covariance is
        # too small by roughly a factor of 2.
        H = 0.5 * H
        grad_at_map = 0.5 * grad_at_map

        # -- Convergence diagnostic ------------------------------------------
        # A large projected gradient means the reported "MAP" point is not
        # actually a stationary point of the free parameters, so the Hessian
        # there is not a valid local curvature matrix for a Laplace
        # approximation (see _projected_gradient docstring).
        proj_grad = _projected_gradient(grad_at_map, self.a_map, self.bounds)
        free_grad_max_abs = float(np.max(np.abs(proj_grad))) if len(proj_grad) else 0.0
        if free_grad_max_abs > grad_warn_threshold:
            warnings.warn(
                f"Laplace covariance: the largest projected gradient component "
                f"at the MAP solution is {free_grad_max_abs:.3g}, above "
                f"grad_warn_threshold={grad_warn_threshold:.3g}. This suggests "
                f"the MAP optimization did not fully converge, which can make "
                f"the Hessian indefinite and the resulting error bars "
                f"unreliable (some may be silently near-zero). Consider "
                f"tightening map_ftol/map_gtol or enabling use_jax_objective "
                f"before re-fitting.",
                RuntimeWarning,
                stacklevel=2,
            )

        # -- Condition on the active set --------------------------------------
        # Exclude parameters pinned at a bound from the matrix that gets
        # inverted; see _active_bound_mask and the method docstring.
        pinned_mask = _active_bound_mask(self.a_map, self.bounds)
        free_mask = ~pinned_mask
        n = len(self.a_map)

        H_free = H[np.ix_(free_mask, free_mask)]
        C_free, H_free_reg, is_pd = laplace_posterior_covariance(
            H_free, regularize_nugget=regularize_nugget,
        )

        # Any remaining negative diagonal within the free block is a genuine
        # numerical anomaly (not expected once pinned parameters are
        # excluded) -- flag it honestly as NaN rather than clip to zero.
        diag_free = np.diag(C_free).copy()
        neg_free = diag_free < 0.0
        if np.any(neg_free):
            warnings.warn(
                f"Laplace covariance: {int(neg_free.sum())} free parameter(s) "
                f"have negative posterior variance after active-set "
                f"conditioning; reporting NaN for these rather than a "
                f"misleadingly precise value. This can happen if the MAP fit "
                f"has not fully converged (see the projected-gradient warning "
                f"above) or if rel_step needs adjusting.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Safe covariance for downstream linear propagation (GH/moment delta
        # method): treat anomalous entries as zero variance/covariance so a
        # few bad entries don't turn the whole propagated result into NaN.
        C_free_safe = C_free.copy()
        if np.any(neg_free):
            C_free_safe[neg_free, :] = 0.0
            C_free_safe[:, neg_free] = 0.0

        # Assemble full-size covariance: zero at pinned parameters (fixed,
        # contributes no variance/covariance -- the standard "conditional on
        # the active set" treatment), the free sub-block otherwise, with
        # NaN preserved (for user-facing arrays only) at anomalous entries.
        C = np.zeros((n, n), float)
        C[np.ix_(free_mask, free_mask)] = C_free_safe
        diag_full = np.zeros(n, float)
        diag_full[free_mask] = diag_free
        diag_full_reported = diag_full.copy()
        if np.any(neg_free):
            free_idx = np.where(free_mask)[0]
            diag_full_reported[free_idx[neg_free]] = np.nan

        nl, nt = self.st.nl, self.st.nt
        C_b = C[:nl, :nl]
        C_w = C[nl:nl + nt, nl:nl + nt]
        with np.errstate(invalid="ignore"):
            b_err = np.sqrt(diag_full_reported[:nl])
            w_err = np.sqrt(diag_full_reported[nl:nl + nt])

        gh_err = self._gh_errors_delta_method(C_b)
        moments_err = self._moment_errors_delta_method(C_b)

        print(f"[LOSVDErrors] Laplace covariance done in {time.perf_counter()-t0:.1f}s. "
              f"Hessian PD (free params): {is_pd}. "
              f"Pinned: {int(pinned_mask[:nl].sum())}/{nl} LOSVD bins, "
              f"{int(pinned_mask[nl:nl + nt].sum())}/{nt} template weights. "
              f"Max projected |grad|: {free_grad_max_abs:.3g}")

        return {
            "method": "laplace",
            "C_full": C,
            "C_b": C_b,
            "C_w": C_w,
            "b_err": b_err,
            "w_err": w_err,
            "H": H,
            "is_pd": is_pd,
            "pinned_mask_b": pinned_mask[:nl],
            "pinned_mask_w": pinned_mask[nl:nl + nt],
            "free_grad_max_abs": free_grad_max_abs,
            "gh_err": gh_err,
            "moments_err": moments_err,
        }

    # ------------------------------------------------------------------
    # Method 2: Residual bootstrap
    # ------------------------------------------------------------------

    def residual_bootstrap(
        self,
        n_bootstrap: int = 200,
        block_size: int = 1,
        n_jobs: int = 1,
        seed: int = 42,
        confidence: float = 0.68,
    ) -> dict:
        """
        Run a residual block bootstrap to estimate LOSVD error bars.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap replicates.  200 is a practical minimum
            for percentile intervals; 500+ for publication.
        block_size : int
            Block length for block bootstrap.  Set > 1 if you expect
            correlated noise (e.g. from sky subtraction).  A value of
            ~5-10 pixels is reasonable for typical spectra.
        n_jobs : int
            Number of parallel worker processes.  -1 = use all CPUs.
            Requires the main fit module to be importable in workers.
        seed : int
            Master RNG seed for reproducibility.
        confidence : float
            Confidence level for reported intervals (default 0.68 ≈ 1sigma).

        Returns
        -------
        dict with keys:
            b_samples    : (n_success, nl) array of bootstrap LOSVD draws
            b_err        : (nl,) bootstrap std dev of b
            b_lo, b_hi   : (nl,) lower/upper percentile bounds
            w_samples    : (n_success, nt)
            gh_samples   : dict of arrays of GH moment samples
            gh_err       : dict of GH moment std devs
            n_success    : number of successful replicates
            n_failed     : number of failed replicates
            confidence   : confidence level used
        """
        # sf is imported at module level from kinextract

        rng = np.random.default_rng(seed)

        if self.is_joint:
            worker_fn = _refit_one_bootstrap_joint
            cfg_frozen = _make_frozen_cfg_joint(self.cfg)
        else:
            worker_fn = _refit_one_bootstrap
            cfg_frozen = _make_frozen_cfg(self.cfg)

        print(f"[LOSVDErrors] Starting residual bootstrap "
              f"(n={n_bootstrap}, block={block_size}, jobs={n_jobs})...")

        # Build the base gerr (before any clean masking)
        base_gerr = self.st.gerr.copy()

        # Serialize FitState fields once (avoid repeated copying)
        st_fields = _fit_state_to_fields(self.st)

        # Generate bootstrap spectra
        g_boots = [
            _draw_bootstrap_spectrum(
                rng, self.st.g, self.gp_map, base_gerr, block_size,
            )
            for _ in range(n_bootstrap)
        ]

        seeds = rng.integers(0, 2**31, size=n_bootstrap)

        if self.is_joint:
            args_list = [
                (g_boots[i], cfg_frozen, base_gerr, st_fields,
                 self._joint_kwargs, int(seeds[i]))
                for i in range(n_bootstrap)
            ]
        else:
            args_list = [
                (g_boots[i], cfg_frozen, base_gerr, self.a_map,
                 self.bounds, int(seeds[i]), st_fields)
                for i in range(n_bootstrap)
            ]

        t0 = time.perf_counter()
        results = []

        if n_jobs == 1:
            # Serial execution — simple and avoids pickling issues
            for k, args in enumerate(args_list):
                r = worker_fn(args)
                results.append(r)
                if (k + 1) % max(1, n_bootstrap // 10) == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"[LOSVDErrors] Bootstrap {k+1}/{n_bootstrap} "
                          f"({elapsed:.0f}s elapsed)")
        else:
            import multiprocessing as _mp
            import sys as _sys
            from concurrent.futures import ThreadPoolExecutor as _TPE
            from concurrent.futures import as_completed as _as_completed

            n_workers = _mp.cpu_count() if n_jobs < 0 else n_jobs
            print(f"[LOSVDErrors] Using {n_workers} worker threads")

            # Threads share the parent's already-initialized JAX/XLA runtime.
            # No re-initialization, no fork, no broken thread-pool inheritance.
            # This allows us to re-enable JAX analytical gradients in workers
            # (use_jax_objective=True), giving ~50-100x speedup per replicate
            # over finite-difference gradients.
            #
            # _make_frozen_cfg/_make_frozen_cfg_joint force use_jax_objective=False
            # (safe default for fork workers), so we make a shallow copy and
            # override it here.
            cfg_frozen_t = copy.copy(cfg_frozen)
            cfg_frozen_t.use_jax_objective = bool(
                getattr(self.cfg, "use_jax_objective", False)
            )
            if self.is_joint:
                args_list_t = [
                    (g, cfg_frozen_t, ge, sf_, jk, s)
                    for (g, _, ge, sf_, jk, s) in args_list
                ]
            else:
                args_list_t = [
                    (g, cfg_frozen_t, ge, a, b, s, sf_)
                    for (g, _, ge, a, b, s, sf_) in args_list
                ]

            # Limit each BLAS library (OpenBLAS / MKL / OpenMP) to 1 internal
            # thread per Python thread.  With n_workers threads this gives
            # n_workers × 1 BLAS thread = n_workers CPUs fully utilized without
            # oversubscription.  _threadpool_limits is None when threadpoolctl
            # is absent (warned at import time above).
            if _THREADPOOLCTL_AVAILABLE:
                _blas_ctx = _threadpool_limits(limits=1, user_api="blas")
            else:
                from contextlib import nullcontext as _nc
                _blas_ctx = _nc()

            with _blas_ctx:
                with _TPE(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(worker_fn, a): i
                        for i, a in enumerate(args_list_t)
                    }
                    results = [None] * n_bootstrap
                    n_done = 0
                    report_every = max(1, n_bootstrap // 10)
                    for future in _as_completed(futures):
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception as _e:
                            print(f"[LOSVDErrors] Worker {idx} failed: {_e}",
                                  file=_sys.stderr)
                        n_done += 1
                        if n_done % report_every == 0:
                            elapsed = time.perf_counter() - t0
                            print(f"[LOSVDErrors] Bootstrap {n_done}/{n_bootstrap} "
                                  f"({elapsed:.0f}s elapsed)")

        # Collect results
        good_results = [r for r in results if r is not None]
        n_success = len(good_results)
        n_failed = n_bootstrap - n_success

        print(f"[LOSVDErrors] Bootstrap done in {time.perf_counter()-t0:.1f}s. "
              f"Success: {n_success}/{n_bootstrap}")

        if n_success == 0:
            raise RuntimeError(
                f"All {n_bootstrap} bootstrap replicates failed; cannot estimate errors. "
                "Check the per-replicate exceptions logged above."
            )

        if n_success < 10:
            warnings.warn(
                f"Only {n_success} bootstrap replicates succeeded. "
                "Error estimates will be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        b_samples = np.array([r["b"] for r in good_results])    # (n_success, nl)
        w_samples = np.array([r["w"] for r in good_results])    # (n_success, nt)

        alpha = (1.0 - confidence) / 2.0
        plo, phi = 100 * alpha, 100 * (1 - alpha)

        b_err = np.std(b_samples, axis=0, ddof=1)
        b_lo  = np.percentile(b_samples, plo, axis=0)
        b_hi  = np.percentile(b_samples, phi, axis=0)

        gh_keys = ["gh_vherm", "gh_sherm", "gh_h3", "gh_h4", "gh_v1", "gh_v2"]
        gh_samples = {k: np.array([r[k] for r in good_results]) for k in gh_keys}
        gh_err = {k: float(np.std(v, ddof=1)) for k, v in gh_samples.items()}
        gh_lo  = {k: float(np.percentile(v, plo)) for k, v in gh_samples.items()}
        gh_med = {k: float(np.percentile(v, 50))  for k, v in gh_samples.items()}
        gh_hi  = {k: float(np.percentile(v, phi)) for k, v in gh_samples.items()}

        gh_ho_keys = ["gh_ho_vherm", "gh_ho_sherm",
                      "gh_ho_h3", "gh_ho_h4", "gh_ho_h5", "gh_ho_h6"]
        gh_ho_samples = {k: np.array([r.get(k, np.nan) for r in good_results])
                         for k in gh_ho_keys}
        gh_ho_err = {k: float(np.nanstd(v, ddof=1)) for k, v in gh_ho_samples.items()}
        gh_ho_lo  = {k: float(np.nanpercentile(v, plo)) for k, v in gh_ho_samples.items()}
        gh_ho_hi  = {k: float(np.nanpercentile(v, phi)) for k, v in gh_ho_samples.items()}

        moments_v = np.array([_losvd_moments(self.xl, b)[0] for b in b_samples], float)
        moments_sigma = np.array([_losvd_moments(self.xl, b)[1] for b in b_samples], float)
        moments_err = {
            "v": float(np.nanstd(moments_v, ddof=1)),
            "sigma": float(np.nanstd(moments_sigma, ddof=1)),
        }
        moments_lo = {
            "v": float(np.nanpercentile(moments_v, plo)),
            "sigma": float(np.nanpercentile(moments_sigma, plo)),
        }
        moments_hi = {
            "v": float(np.nanpercentile(moments_v, phi)),
            "sigma": float(np.nanpercentile(moments_sigma, phi)),
        }

        return {
            "method": "bootstrap",
            "b_samples": b_samples,
            "b_err": b_err,
            "b_lo": b_lo,
            "b_hi": b_hi,
            "b_map": self.b_map,
            "w_samples": w_samples,
            "gh_samples": gh_samples,
            "gh_err": gh_err,
            "gh_lo": gh_lo,
            "gh_med": gh_med,
            "gh_hi": gh_hi,
            "gh_ho_samples": gh_ho_samples,
            "gh_ho_err": gh_ho_err,
            "gh_ho_lo": gh_ho_lo,
            "gh_ho_hi": gh_ho_hi,
            "moments_samples": {"v": moments_v, "sigma": moments_sigma},
            "moments_err": moments_err,
            "moments_lo": moments_lo,
            "moments_hi": moments_hi,
            "n_success": n_success,
            "n_failed": n_failed,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Method 3: Bias-corrected LOSVD
    # ------------------------------------------------------------------

    def bias_correction(self) -> dict:
        """
        Estimate and subtract the regularization bias from the MAP LOSVD.

        The roughness/smoothing penalty used in the MAP fit shrinks the
        recovered LOSVD ``b_map`` toward flatness — a classic bias-variance
        trade-off from penalized-likelihood estimation. This method builds
        the linearized influence ("hat") matrix ``H_b`` of the penalized
        problem restricted to the LOSVD subspace (see
        :func:`compute_losvd_hat_matrix`), then uses it to approximately
        invert the shrinkage and recover a less-biased (but noisier)
        estimate of the true LOSVD (see :func:`bias_corrected_losvd`). The
        trace of ``H_b`` gives the effective number of independent degrees
        of freedom retained by the smoothed fit, which is typically well
        below ``nl`` when strong regularization is used.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``H_b`` : ndarray, shape (nl, nl) — linearized influence
              (hat) matrix mapping the true LOSVD to its expected MAP
              estimate.
            - ``b_corr`` : ndarray, shape (nl,) — bias-corrected LOSVD,
              clipped to be non-negative.
            - ``bias`` : ndarray, shape (nl,) — estimated regularization
              bias, ``b_map - H_b @ b_corr``.
            - ``eff_dof`` : float — effective degrees of freedom,
              ``trace(H_b)``, out of a maximum of ``nl`` (the number of
              LOSVD velocity bins).
        """
        _require_genuine_map_fit(self.fit, "bias_correction()")
        _require_not_joint(self, "bias_correction()")
        print("[LOSVDErrors] Computing LOSVD influence matrix...")
        t0 = time.perf_counter()
        H_b = compute_losvd_hat_matrix(self.st, self.a_map)
        b_corr, bias = bias_corrected_losvd(self.b_map, H_b)
        eff_dof = float(np.trace(H_b))
        print(f"[LOSVDErrors] Bias correction done in {time.perf_counter()-t0:.1f}s. "
              f"eff_dof={eff_dof:.2f}/{self.st.nl}")
        return {
            "method": "bias_correction",
            "H_b": H_b,
            "b_corr": b_corr,
            "bias": bias,
            "eff_dof": eff_dof,
        }

    # ------------------------------------------------------------------
    # GH moment errors via delta method
    # ------------------------------------------------------------------

    def _gh_errors_delta_method(self, C_b: np.ndarray) -> dict:
        """
        Propagate b covariance through the GH moment extraction.

        Uses numerical Jacobian of (v1, v2, h3, h4) w.r.t. b.

        Parameters
        ----------
        C_b : (nl, nl)  Covariance matrix of b.

        Returns
        -------
        dict of GH moment standard deviations.
        """
        # sf is imported at module level from kinextract

        b0 = self.b_map.copy()
        xl = self.xl
        step = 1e-6

        def gh_vec(b):
            gh = fit_losvd_gauss_hermite(xl, b, fit_h3h4=True)
            return np.array([
                gh["vherm"], gh["sherm"], gh["h3"], gh["h4"],
                gh["v1"], gh["v2"],
            ])

        try:
            gh_vec(b0)  # validate that the baseline GH fit succeeds before differencing
        except Exception:
            return {k: np.nan for k in
                    ["gh_vherm", "gh_sherm", "gh_h3", "gh_h4", "gh_v1", "gh_v2"]}

        nl = len(b0)
        nm = 6
        Jgh = np.zeros((nm, nl), float)
        for k in range(nl):
            b_p = b0.copy(); b_p[k] += step
            b_m = b0.copy(); b_m[k] -= step
            try:
                Jgh[:, k] = (gh_vec(b_p) - gh_vec(b_m)) / (2 * step)
            except Exception:
                pass

        C_gh = Jgh @ C_b @ Jgh.T
        gh_std = np.sqrt(np.maximum(np.diag(C_gh), 0.0))

        keys = ["gh_vherm", "gh_sherm", "gh_h3", "gh_h4", "gh_v1", "gh_v2"]
        return dict(zip(keys, gh_std))

    def _moment_errors_delta_method(self, C_b: np.ndarray) -> dict:
        """Propagate b covariance to flux-weighted LOSVD moments (V, sigma)."""
        b0 = self.b_map.copy()
        step = 1e-6

        def mv_sigma(b):
            v, s = _losvd_moments(self.xl, b)
            return np.array([v, s], float)

        m0 = mv_sigma(b0)
        if not np.all(np.isfinite(m0)):
            return {"v": np.nan, "sigma": np.nan}

        nl = len(b0)
        J = np.zeros((2, nl), float)
        for k in range(nl):
            b_p = b0.copy(); b_p[k] += step
            b_m = b0.copy(); b_m[k] -= step
            m_p = mv_sigma(b_p)
            m_m = mv_sigma(b_m)
            if np.all(np.isfinite(m_p)) and np.all(np.isfinite(m_m)):
                J[:, k] = (m_p - m_m) / (2 * step)

        C_m = J @ C_b @ J.T
        m_std = np.sqrt(np.maximum(np.diag(C_m), 0.0))
        return {"v": float(m_std[0]), "sigma": float(m_std[1])}

    # ------------------------------------------------------------------
    # Summary and combination
    # ------------------------------------------------------------------

    def summarize(
        self,
        laplace_result: Optional[dict] = None,
        bootstrap_result: Optional[dict] = None,
        bias_result: Optional[dict] = None,
    ) -> dict:
        """
        Combine results from multiple methods into a unified summary.

        The recommended final error bar is:
          - If bootstrap is available: use bootstrap std dev on b
            (most honest frequentist estimate)
          - Otherwise: use Laplace posterior std dev
            (correct for the Bayesian interpretation, lower bound otherwise)

        The bias-corrected LOSVD is provided alongside the MAP LOSVD
        for comparison; its uncertainty is larger (we divided by H_b
        which amplifies noise), so report it alongside b_map rather
        than replacing it.

        Returns
        -------
        dict with human-readable summary of all error estimates.
        """
        # sf is imported at module level from kinextract

        gh_map = fit_losvd_gauss_hermite(self.xl, self.b_map, fit_h3h4=True)
        gh_ho_map = fit_losvd_gauss_hermite_higher(self.xl, self.b_map, max_order=6)

        summary = {
            "xl": self.xl,
            "b_map": self.b_map,
            "gh_map": gh_map,
            "gh_ho_map": gh_ho_map,
            "template_weights": self.w_map,
        }

        if bootstrap_result is not None:
            b_center = np.median(bootstrap_result["b_samples"], axis=0)
            gh_center = fit_losvd_gauss_hermite(self.xl, b_center, fit_h3h4=True)
            mv_center, ms_center = _losvd_moments(self.xl, b_center)
            summary["b_err_recommended"] = bootstrap_result["b_err"]
            summary["b_lo_recommended"] = bootstrap_result["b_lo"]
            summary["b_hi_recommended"] = bootstrap_result["b_hi"]
            summary["b_center_recommended"] = b_center
            summary["moments_center_recommended"] = {"v": mv_center, "sigma": ms_center}
            summary["moments_err_recommended"] = bootstrap_result.get("moments_err", {})
            summary["moments_lo_recommended"] = bootstrap_result.get("moments_lo", {})
            summary["moments_hi_recommended"] = bootstrap_result.get("moments_hi", {})
            summary["gh_center_recommended"] = gh_center
            summary["gh_err_recommended"] = bootstrap_result["gh_err"]
            summary["gh_lo_recommended"] = bootstrap_result["gh_lo"]
            summary["gh_hi_recommended"] = bootstrap_result["gh_hi"]
            summary["gh_ho_err_recommended"] = bootstrap_result.get("gh_ho_err", {})
            summary["gh_ho_lo_recommended"] = bootstrap_result.get("gh_ho_lo", {})
            summary["gh_ho_hi_recommended"] = bootstrap_result.get("gh_ho_hi", {})
            summary["method_recommended"] = "bootstrap"
            summary["bootstrap"] = bootstrap_result
            # include bootstrap template weight samples if present
            if "w_samples" in bootstrap_result:
                summary["template_weight_samples"] = bootstrap_result["w_samples"]
        elif laplace_result is not None:
            mv_center, ms_center = _losvd_moments(self.xl, self.b_map)
            summary["b_err_recommended"] = laplace_result["b_err"]
            summary["b_lo_recommended"] = self.b_map - laplace_result["b_err"]
            summary["b_hi_recommended"] = self.b_map + laplace_result["b_err"]
            summary["b_center_recommended"] = self.b_map
            summary["moments_center_recommended"] = {"v": mv_center, "sigma": ms_center}
            summary["moments_err_recommended"] = laplace_result.get("moments_err", {})
            summary["gh_center_recommended"] = gh_map
            summary["gh_err_recommended"] = laplace_result["gh_err"]
            summary["method_recommended"] = "laplace"

        if laplace_result is not None:
            summary["laplace"] = laplace_result

        if bias_result is not None:
            summary["bias_correction"] = bias_result
            summary["b_corr"] = bias_result["b_corr"]
            summary["b_bias"] = bias_result["bias"]
            summary["eff_dof"] = bias_result["eff_dof"]

        # Print a compact summary table
        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: dict) -> None:
        print("\n" + "=" * 70)
        print("LOSVD ERROR SUMMARY")
        print("=" * 70)
        gh = summary["gh_map"]
        b_center = summary.get("b_center_recommended", summary["b_map"])
        v_meas, sigma_meas = _losvd_moments(self.xl, b_center)
        moments = summary.get("moments_center_recommended", {"v": v_meas, "sigma": sigma_meas})
        moments_err = summary.get("moments_err_recommended", {})
        ghe = summary.get("gh_err_recommended", {})
        print("  Gauss-Hermite moments (MAP, consistent with pallmc.f):")
        print(f"    V    = {gh['vherm']:+.2f} km/s"
              + (f" ± {ghe['gh_vherm']:.2f}" if "gh_vherm" in ghe else ""))
        print(f"    σ    = {gh['sherm']:.2f} km/s"
              + (f" ± {ghe['gh_sherm']:.2f}" if "gh_sherm" in ghe else ""))
        print(f"    h3   = {gh['h3']:+.4f}"
              + (f" ± {ghe['gh_h3']:.4f}" if "gh_h3" in ghe else ""))
        print(f"    h4   = {gh['h4']:+.4f}"
              + (f" ± {ghe['gh_h4']:.4f}" if "gh_h4" in ghe else ""))
        print("  LOSVD moments (for reference):")
        print(
            f"    V    = {moments['v']:+.2f} km/s"
            + (f" ± {moments_err['v']:.2f}" if "v" in moments_err else "")
        )
        print(
            f"    σ    = {moments['sigma']:.2f} km/s"
            + (f" ± {moments_err['sigma']:.2f}" if "sigma" in moments_err else "")
        )

        if "eff_dof" in summary:
            nl = len(self.xl)
            print(f"\n  Regularization: eff_dof = {summary['eff_dof']:.2f} / {nl} "
                  f"(smoothing shrinks LOSVD toward {1 - summary['eff_dof']/nl:.0%} "
                  f"of its values)")

        if "bootstrap" in summary:
            b = summary["bootstrap"]
            print(f"\n  Bootstrap: {b['n_success']} successful / "
                  f"{b['n_success'] + b['n_failed']} total replicates")

        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_losvd_with_errors(
        self,
        summary: dict,
        fig=None,
        ax=None,
        show: bool = True,
    ):
        """
        Plot the recovered LOSVD together with its estimated error band.

        Draws the central LOSVD estimate (bootstrap median if a bootstrap
        was run, otherwise the MAP LOSVD) as a solid curve, shades the
        recommended 1-sigma (or configured confidence-level) uncertainty
        band from ``summary`` beneath it, overlays the Gauss-Hermite (GH)
        fit to the central LOSVD, and annotates the plot with the MAP GH
        moments (V, sigma, h3, h4) and their recommended uncertainties
        (bootstrap standard deviation if available, otherwise the Laplace
        delta-method standard deviation).

        Parameters
        ----------
        summary : dict
            Output of :meth:`summarize`. Must contain ``"xl"``, ``"b_map"``,
            and ``"gh_map"``; uses ``"b_center_recommended"``,
            ``"b_err_recommended"``/``"b_lo_recommended"``/``"b_hi_recommended"``,
            ``"gh_center_recommended"``, and ``"gh_err_recommended"`` when
            present to determine the central curve, error band, and
            annotated moment uncertainties.
        fig : matplotlib.figure.Figure, optional
            Existing figure to draw into. Ignored unless ``ax`` is also
            provided (a new figure/axes pair is created otherwise).
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into. If None, a new figure and axes are
            created.
        show : bool, optional
            If True (default), call ``plt.show()`` to display the figure
            immediately.

        Returns
        -------
        tuple
            ``(fig, ax)`` — the Matplotlib figure and axes used, so callers
            can further customize or save the plot.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        # sf is imported at module level from kinextract

        with mpl.rc_context({"text.usetex": False}):
            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 5))

            xl = summary["xl"]
            b_map = summary["b_map"]
            b_center = summary.get("b_center_recommended", b_map)
            gh = summary.get("gh_center_recommended", summary["gh_map"])

            # Error band
            if "b_err_recommended" in summary:
                b_lo = summary.get("b_lo_recommended", b_center - summary["b_err_recommended"])
                b_hi = summary.get("b_hi_recommended", b_center + summary["b_err_recommended"])
                method = summary.get("method_recommended", "")
                ax.fill_between(xl, b_lo, b_hi, alpha=0.25, color="tab:blue",
                                label=fr"$1\sigma$ ({method})")

            # Central LOSVD curve: bootstrap median when available, otherwise MAP.
            if np.allclose(b_center, b_map):
                center_label = "MAP LOSVD"
            else:
                center_label = "Bootstrap median LOSVD"
                # ax.plot(xl, b_map, "--", color="0.55", lw=1.1, ms=3, label="MAP LOSVD")
            ax.plot(xl, b_center, "o-", color="tab:blue", lw=1.5, ms=4, label=center_label)

            # Bias-corrected LOSVD
            # if "b_corr" in summary:
            #     ax.plot(xl, summary["b_corr"], "--", color="tab:orange",
            #             lw=1.5, label="Bias-corrected LOSVD")

            # GH fit
            if np.all(np.isfinite(gh["model"])):
                ax.plot(xl, gh["model"], "--", color="tab:red", lw=1, label="GH fit")

            # Annotate GH moments (MAP GH as central value, bootstrap GH std as error).
            # Consistent with original pallmc.f which fits GH to each bootstrap LOSVD
            # and reports MAP GH sigma with bias-corrected bootstrap error bars.
            ghe = summary.get("gh_err_recommended", {})
            gh_map_vals = summary["gh_map"]

            parts = [
                "Gauss-Hermite fit:",
                rf"  $V = {gh_map_vals['vherm']:+.2f}$ km/s" +
                (rf" $\pm {ghe['gh_vherm']:.2f}$" if "gh_vherm" in ghe else ""),
                rf"  $\sigma = {gh_map_vals['sherm']:.2f}$ km/s" +
                (rf" $\pm {ghe['gh_sherm']:.2f}$" if "gh_sherm" in ghe else ""),
                rf"  $h_3 = {gh_map_vals['h3']:+.3f}$" +
                (rf" $\pm {ghe['gh_h3']:.3f}$" if "gh_h3" in ghe else ""),
                rf"  $h_4 = {gh_map_vals['h4']:+.3f}$" +
                (rf" $\pm {ghe['gh_h4']:.3f}$" if "gh_h4" in ghe else ""),
            ]
            annotation = "\n".join(parts)
            ax.text(0.04, 0.96, annotation, transform=ax.transAxes,
                    ha="left", va="top", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="0.5", alpha=0.9))

            ax.axhline(0, color="0.6", lw=0.8, ls="--")
            ax.set_xlabel("Velocity [km/s]")
            ax.set_ylabel("LOSVD")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.2)
            plt.tight_layout()

            if show:
                plt.show()

            return fig, ax

    def plot_bootstrap_distribution(
        self, bootstrap_result: dict, show: bool = True,
    ):
        """
        Diagnostic plot of the full bootstrap ensemble (LOSVD bins and GH moments).

        Produces a 2x3 grid of diagnostic panels summarizing the residual
        bootstrap ensemble from :meth:`residual_bootstrap`: histograms of
        several individual LOSVD bins, an overplot of a subsample of the
        bootstrap LOSVD replicates against the MAP LOSVD and the bootstrap
        median, the per-bin standard deviation of the LOSVD as a function of
        velocity, and histograms of the bootstrap distributions of the
        Gauss-Hermite moments (V, sigma, h3, h4) with the MAP value and the
        16th/84th percentiles marked. Useful for sanity-checking that the
        bootstrap ensemble is well-behaved (e.g. roughly unimodal, no
        pathological outlier replicates) before trusting the resulting
        error bars.

        Parameters
        ----------
        bootstrap_result : dict
            Output of :meth:`residual_bootstrap`. Must contain
            ``"b_samples"``, ``"gh_samples"``, and ``"n_success"``.
        show : bool, optional
            If True (default), call ``plt.show()`` to display the figure
            immediately.

        Returns
        -------
        tuple
            ``(fig, axes)`` — the Matplotlib figure and 2x3 array of axes
            used, so callers can further customize or save the plot.
        """
        import matplotlib.pyplot as plt

        b_samples = bootstrap_result["b_samples"]
        gh_samples = bootstrap_result["gh_samples"]
        xl = self.xl

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(
            f"Bootstrap distributions  "
            f"(n={bootstrap_result['n_success']} replicates)",
            fontsize=13,
        )

        # Top row: selected LOSVD bins
        ax = axes[0, 0]
        for j in range(0, len(xl), max(1, len(xl) // 5)):
            ax.hist(b_samples[:, j], bins=20, alpha=0.5,
                    label=f"v={xl[j]:.0f}", density=True)
        ax.set_xlabel("b value")
        ax.set_title("LOSVD bin distributions")
        ax.legend(fontsize=7)

        # Top centre: bootstrap LOSVD ensemble
        ax = axes[0, 1]
        for k in range(min(50, len(b_samples))):
            ax.plot(xl, b_samples[k], color="tab:blue", alpha=0.1, lw=0.8)
        ax.plot(xl, np.median(b_samples, axis=0), "k-", lw=2, label="Median")
        ax.plot(xl, self.b_map, "--", color="0.5", lw=1.2, label="MAP")
        ax.set_xlabel("Velocity [km/s]")
        ax.set_title("Bootstrap LOSVD ensemble")
        ax.legend()

        # Top right: b std dev
        ax = axes[0, 2]
        ax.plot(xl, np.std(b_samples, axis=0, ddof=1), "o-", color="tab:blue")
        ax.set_xlabel("Velocity [km/s]")
        ax.set_ylabel("Std dev of b")
        ax.set_title("LOSVD uncertainty vs. velocity")

        # Bottom: GH moment histograms
        gh_plot = [
            ("gh_vherm", r"$V$ [km/s]"),
            ("gh_sherm", r"$\sigma$ [km/s]"),
            ("gh_h3",    r"$h_3$"),
            ("gh_h4",    r"$h_4$"),
        ]
        gh_map_vals = {
            "gh_vherm": self.fit["outputs"].get("gh", {}).get("vherm", np.nan),
            "gh_sherm": self.fit["outputs"].get("gh", {}).get("sherm", np.nan),
            "gh_h3": self.fit["outputs"].get("gh", {}).get("h3", np.nan),
            "gh_h4": self.fit["outputs"].get("gh", {}).get("h4", np.nan),
        }
        # sf is imported at module level from kinextract
        gh_map = fit_losvd_gauss_hermite(xl, self.b_map, fit_h3h4=True)
        gh_map_vals = {
            "gh_vherm": gh_map["vherm"], "gh_sherm": gh_map["sherm"],
            "gh_h3": gh_map["h3"], "gh_h4": gh_map["h4"],
        }

        for idx, (key, label) in enumerate(gh_plot):
            ax = axes[1, idx % 3] if idx < 3 else axes[1, 2]
            vals = gh_samples[key]
            finite = vals[np.isfinite(vals)]
            if len(finite) > 3:
                ax.hist(finite, bins=25, color="tab:green", alpha=0.7, density=True)
                ax.axvline(gh_map_vals[key], color="k", lw=2, label="MAP")
                ax.axvline(np.percentile(finite, 16), color="tab:red",
                           ls="--", lw=1.2, label="16/84%")
                ax.axvline(np.percentile(finite, 84), color="tab:red", ls="--", lw=1.2)
            ax.set_xlabel(label)
            ax.legend(fontsize=8)

        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_spectrum_with_errors(
        self,
        summary: dict,
        fig=None,
        ax=None,
        show: bool = True,
    ):
        """
        Plot the best-fit spectrum with observational and model error bands.

        Mirrors the standard kinextract data/model/residual diagnostic
        (compare :func:`kinextract.plotting.plot_fit`), but additionally
        propagates the LOSVD uncertainty into a shaded *model* uncertainty
        band on the spectrum itself: if bootstrap samples are present in
        ``summary``, the model spectrum is recomputed for a subsample of
        bootstrap LOSVD/template-weight draws and the 16th/84th percentile
        envelope is shaded; otherwise, if only Laplace/recommended
        diagonal LOSVD errors are available, the band is approximated by
        drawing independent Gaussian perturbations of the LOSVD bins using
        the recommended per-bin standard deviations. The top panel shows
        data, best-fit model, the observational 1-sigma error band, and
        masked pixels; the bottom panel shows fractional residuals with the
        RMS annotated.

        Parameters
        ----------
        summary : dict
            Output of :meth:`summarize`. Uses ``"bootstrap"``
            (with ``"b_samples"`` and optionally ``"w_samples"``) if
            present to build the model uncertainty band from actual
            refits; otherwise falls back to ``"b_center_recommended"`` and
            ``"b_err_recommended"`` for an approximate Gaussian-draw band.
        fig : matplotlib.figure.Figure, optional
            Existing figure to draw into. Ignored unless ``ax`` is also
            provided (a new figure/axes pair is created otherwise).
        ax : tuple of matplotlib.axes.Axes, optional
            Existing ``(ax0, ax1)`` pair (spectrum panel, residual panel) to
            draw into. If None, a new figure with two stacked axes is
            created.
        show : bool, optional
            If True (default), call ``plt.show()`` to display the figure
            immediately.

        Returns
        -------
        tuple
            ``(fig, ax)`` — the Matplotlib figure and the ``(ax0, ax1)`` axes
            pair used, so callers can further customize or save the plot.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        # sf is imported at module level from kinextract

        st = self.st
        gp = self.fit["outputs"]["gp"]
        wave = np.asarray(st.x, float)
        data = np.asarray(st.g, float)
        gerr = np.asarray(st.gerr, float)
        good = np.isfinite(data) & np.isfinite(gp) & np.isfinite(gerr) & (gerr < 1e8)
        resid = data - gp

        # Model uncertainty band: prefer bootstrap samples if available,
        # otherwise approximate by sampling b from recommended errors (diag only).
        gp_lo = gp_hi = None
        try:
            if "bootstrap" in summary and summary["bootstrap"] is not None and "b_samples" in summary["bootstrap"]:
                b_samples = np.asarray(summary["bootstrap"]["b_samples"])
                w_samples = np.asarray(summary["bootstrap"].get("w_samples")) if summary["bootstrap"].get("w_samples") is not None else None
                n_samps = b_samples.shape[0]
                n_plot = min(200, max(10, n_samps))
                idx = np.linspace(0, n_samps - 1, n_plot).astype(int)
                gps = []
                for k in idx:
                    a_s = self.a_map.copy()
                    a_s[_b_slice(self.st)] = b_samples[k]
                    if w_samples is not None and w_samples.shape[0] > k:
                        a_s[_w_slice(self.st)] = w_samples[k]
                    gp_s, *_ = evaluate_model_gp(a_s, self.st)
                    gps.append(gp_s)
                gps = np.vstack(gps)
                gp_lo = np.percentile(gps, 16, axis=0)
                gp_hi = np.percentile(gps, 84, axis=0)
            elif "b_err_recommended" in summary:
                # Approximate by independent Gaussian draws on b using recommended stddevs
                b_center = np.asarray(summary.get("b_center_recommended", self.b_map))
                b_err = np.asarray(summary.get("b_err_recommended", np.zeros_like(b_center)))
                n_draw = 100
                rng = np.random.default_rng(123456)
                gps = []
                for _ in range(n_draw):
                    b_draw = b_center + rng.normal(0.0, b_err)
                    a_s = self.a_map.copy()
                    a_s[_b_slice(self.st)] = b_draw
                    gp_s, *_ = evaluate_model_gp(a_s, self.st)
                    gps.append(gp_s)
                gps = np.vstack(gps)
                gp_lo = np.percentile(gps, 16, axis=0)
                gp_hi = np.percentile(gps, 84, axis=0)
        except Exception:
            gp_lo = gp_hi = None

        _show_errs = getattr(self.cfg, "use_spectrum_errors", True)

        with mpl.rc_context({"text.usetex": False}):
            if ax is None:
                fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

            ax0, ax1 = ax
            ax0.plot(wave, data, color="black", lw=1.0, label="data")
            ax0.plot(wave, gp, color="#bf5b17", lw=1.2, label="best fit")
            if gp_lo is not None and gp_hi is not None:
                ax0.fill_between(
                    wave, gp_lo, gp_hi,
                    color="#bf5b17", alpha=0.20,
                    label="model uncertainty",
                )
            if _show_errs:
                ax0.fill_between(
                    wave,
                    data - gerr,
                    data + gerr,
                    where=good,
                    color="0.7",
                    alpha=0.25,
                    label=r"$1\sigma$ data error",
                )
            if not np.all(good):
                ax0.scatter(wave[~good], data[~good], s=8, color="#7f7f7f", alpha=0.5, label="masked")
            ax0.set_ylabel("Flux")
            ax0.legend(loc="best", frameon=False)
            # ax0.set_title(summary.get("method_recommended", "best-fit spectrum"))

            frac_resid = resid / np.where(np.abs(gp) > 1e-12 * np.abs(gp[good]).max(),
                                         gp, np.nan)
            ax1.axhline(0.0, color="#444444", lw=0.8)
            ax1.plot(wave[good], frac_resid[good], color="#2c7fb8", lw=1.0)
            if good.any():
                rms = float(np.sqrt(np.nanmean(frac_resid[good] ** 2)))
                ax1.axhline( rms, color="0.6", lw=0.8, ls="--")
                ax1.axhline(-rms, color="0.6", lw=0.8, ls="--")
                ax1.text(0.01, 0.95, f"RMS={rms:.4g}", transform=ax1.transAxes, va="top",
                         fontsize=9)
            ax1.set_ylabel("(Data - model) / model")
            ax1.set_ylim(-0.05, 0.05)
            ax1.set_xlabel("Wavelength")
            ax1.grid(alpha=0.2)
            plt.tight_layout()

            if show:
                plt.show()

            return fig, ax


# -----------------------------------------------------------------------------
# Convenience plotting wrappers
# -----------------------------------------------------------------------------

def plot_losvd_with_errors(fit: dict, cfg, summary: dict, show: bool = True):
    """Convenience wrapper around :meth:`LOSVDErrorEstimator.plot_losvd_with_errors`.

    Constructs a fresh :class:`LOSVDErrorEstimator` from ``fit``/``cfg`` and
    immediately calls its ``plot_losvd_with_errors`` method. Useful when you
    already have a ``summary`` dict (e.g. from :func:`estimate_losvd_errors`)
    and just want a quick plot without keeping the estimator instance
    around.

    Parameters
    ----------
    fit : dict
        Output of ``run_spectral_fit()``.
    cfg : FitConfig
        Configuration object used for the original fit.
    summary : dict
        Output of :meth:`LOSVDErrorEstimator.summarize`.
    show : bool, optional
        If True (default), display the figure immediately.

    Returns
    -------
    tuple
        ``(fig, ax)`` — see :meth:`LOSVDErrorEstimator.plot_losvd_with_errors`.
    """
    est = LOSVDErrorEstimator(fit, cfg)
    return est.plot_losvd_with_errors(summary, show=show)


def plot_spectrum_with_errors(fit: dict, cfg, summary: dict, show: bool = True):
    """Convenience wrapper around :meth:`LOSVDErrorEstimator.plot_spectrum_with_errors`.

    Constructs a fresh :class:`LOSVDErrorEstimator` from ``fit``/``cfg`` and
    immediately calls its ``plot_spectrum_with_errors`` method.

    Parameters
    ----------
    fit : dict
        Output of ``run_spectral_fit()``.
    cfg : FitConfig
        Configuration object used for the original fit.
    summary : dict
        Output of :meth:`LOSVDErrorEstimator.summarize`.
    show : bool, optional
        If True (default), display the figure immediately.

    Returns
    -------
    tuple
        ``(fig, ax)`` — see :meth:`LOSVDErrorEstimator.plot_spectrum_with_errors`.
    """
    est = LOSVDErrorEstimator(fit, cfg)
    return est.plot_spectrum_with_errors(summary, show=show)


# =============================================================================
# Section 5 – File I/O for error estimates
# =============================================================================

def _write_sim_file(
    sim_path: "Path",
    velocity_grid: np.ndarray,
    losvd_map: np.ndarray,
    template_weights_map: np.ndarray,
    losvd_samples: np.ndarray,
    template_weight_samples: "np.ndarray | None" = None,
) -> None:
    """Write a .sim file in the format expected by pallmc.f.

    Format (all rows space-separated, nsim values per row):
      line 1 : nsim  n_vel  n_temp          (header)
      line 2 : nsim chi2 placeholder values  (fsim — read separately by pallmc)
      lines 3..n_vel+2     : n_vel LOSVD rows
      lines n_vel+3..n_vel+n_temp+2 : n_temp template-weight rows
      line n_vel+n_temp+3  : flux-weighted velocity centroid for each sample

    The MAP fit is prepended as sample 0 so pallmc's xsim(1,i) is the real
    MAP value (Fortran 1-indexed).
    """
    velocity_grid = np.asarray(velocity_grid, dtype=float)
    losvd_map = np.asarray(losvd_map, dtype=float)
    template_weights_map = np.asarray(template_weights_map, dtype=float)
    losvd_samples = np.asarray(losvd_samples, dtype=float)
    if losvd_samples.ndim == 1:
        losvd_samples = losvd_samples[None, :]

    n_vel = velocity_grid.size
    n_temp = template_weights_map.size

    if template_weight_samples is not None:
        wt_boot = np.asarray(template_weight_samples, dtype=float)
        if wt_boot.ndim == 1:
            wt_boot = wt_boot[None, :]
    else:
        wt_boot = np.zeros((losvd_samples.shape[0], n_temp), dtype=float)

    # Prepend MAP as sample 0
    losvd_mat = np.vstack([losvd_map[None, :], losvd_samples])      # (1+n_boot, n_vel)
    wt_mat    = np.vstack([template_weights_map[None, :], wt_boot])  # (1+n_boot, n_temp)
    nsim = losvd_mat.shape[0]

    # flux-weighted velocity centroids
    losvd_norm = losvd_mat.sum(axis=1)
    losvd_norm = np.where(losvd_norm > 0, losvd_norm, 1.0)
    vel_centers = losvd_mat.dot(velocity_grid) / losvd_norm          # (nsim,)

    with open(sim_path, "w") as fh:
        fh.write(f"{nsim:d} {n_vel:d} {n_temp:d}\n")
        fh.write("  ".join("0.0" for _ in range(nsim)) + "\n")
        for j in range(n_vel):
            fh.write("  ".join(f"{losvd_mat[i, j]:.8E}" for i in range(nsim)) + "\n")
        for k in range(n_temp):
            fh.write("  ".join(f"{wt_mat[i, k]:.8E}" for i in range(nsim)) + "\n")
        fh.write("  ".join(f"{vel_centers[i]:.8E}" for i in range(nsim)) + "\n")


def write_losvd_errors_to_files(
    summary: dict,
    prefix: str,
    outdir: str | Path = ".",
) -> dict:
    """
    Write LOSVD vector with error bars to file.

    Produces {prefix}.losvd_errs.out with columns:
        bin_index  velocity_km/s  amplitude_map  amplitude_err  amplitude_lo  amplitude_hi

    Parameters
    ----------
    summary : dict
        Output from LOSVDErrorEstimator.summarize()
    prefix : str
        Output file prefix (e.g., "bin0105sp")
    outdir : str or Path
        Output directory

    Returns
    -------
    dict
        {'losvd_errs_file': path_to_file}
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xl = summary["xl"]
    b_map = summary["b_map"]
    b_err = summary.get("b_err_recommended", np.zeros_like(b_map))
    b_lo = summary.get("b_lo_recommended", b_map - b_err)
    b_hi = summary.get("b_hi_recommended", b_map + b_err)
    method = summary.get("method_recommended", "unknown")

    outfile = outdir / f"{prefix}.losvd_errs.out"
    with open(outfile, "w") as f:
        f.write(f"# LOSVD error estimates (method: {method})\n")
        f.write("# bin_index  velocity_km/s  amplitude_map  amplitude_err  amplitude_lo  amplitude_hi\n")
        for i, (vel, amp, err, lo, hi) in enumerate(zip(xl, b_map, b_err, b_lo, b_hi)):
            f.write(f"{i+1:10d} {vel:15.6f} {amp:20.11E} {err:20.11E} {lo:20.11E} {hi:20.11E}\n")

    print(f"[losvd_errs] Wrote {outfile}")
    # If a bootstrap median center is available in the summary, write it to a separate file
    if "b_center_recommended" in summary:
        center = np.asarray(summary["b_center_recommended"])  # median from bootstrap
        center_file = outdir / f"{prefix}.losvd_center.out"
        with open(center_file, "w") as f:
            f.write("# LOSVD bootstrap median\n")
            f.write("# bin_index  velocity_km/s  amplitude_median\n")
            for i, (vel, amp_c) in enumerate(zip(xl, center)):
                f.write(f"{i+1:10d} {vel:15.6f} {amp_c:20.11E}\n")
        print(f"[losvd_errs] Wrote {center_file}")
        # Also write legacy mcfit summary if bootstrap samples available
        if "bootstrap" in summary and "b_samples" in summary["bootstrap"]:
            try:
                b_samps = np.asarray(summary["bootstrap"]["b_samples"])  # (n_success, nl)
            except Exception:
                b_samps = np.asarray(summary["bootstrap"]["b_samples"]) if isinstance(summary["bootstrap"].get("b_samples"), list) else None
        else:
            b_samps = None

        if b_samps is not None and b_samps.size:
            # Percentiles: 50, 5, 16, 84, 95
            p50 = np.percentile(b_samps, 50, axis=0)
            p5 = np.percentile(b_samps, 5, axis=0)
            p16 = np.percentile(b_samps, 16, axis=0)
            p84 = np.percentile(b_samps, 84, axis=0)
            p95 = np.percentile(b_samps, 95, axis=0)
            mcfile = outdir / f"{prefix}mc.mcfit"
            _n_temp = np.asarray(summary.get("template_weights", np.zeros(1))).size
            with open(mcfile, "w") as f:
                f.write(f"{len(xl)} {_n_temp}\n")
                for vel, _p50, _p5, _p16, _p84, _p95 in zip(xl, p50, p5, p16, p84, p95):
                    f.write(f"{vel:.8f} {_p50:.11E} {_p5:.11E} {_p16:.11E} {_p84:.11E} {_p95:.11E}\n")
            print(f"[losvd_errs] Wrote {mcfile}")
            # Write the .sim file using the actual MAP LOSVD as reference sample 0
            # (pallmc Fortran sample 1).  b_map is the MAP LOSVD from the spectral
            # fit; p50 is the bootstrap median.  pallmc's sigma bias correction
            # requires sample 1 = MAP fit, not bootstrap median.
            template_weights_arr = np.asarray(summary.get("template_weights", np.zeros(1)))
            template_weight_samples = summary.get("template_weight_samples", None)
            sim_path = outdir / (prefix + "mc.sim")
            _write_sim_file(
                sim_path,
                np.asarray(xl, float),
                b_map,
                template_weights_arr,
                b_samps,
                template_weight_samples=(
                    np.asarray(template_weight_samples)
                    if template_weight_samples is not None else None
                ),
            )
            print(f"[losvd_errs] Wrote {sim_path}")
            # Also write .mcfit / .mcfit2 via lib.scofit.core if importable.
            try:
                from lib.scofit.core import write_mcfit_files
                mc_stem = outdir / (prefix + "mc")
                losvd_low_for_mc = np.asarray(summary.get("b_lo_recommended", b_map))
                losvd_high_for_mc = np.asarray(summary.get("b_hi_recommended", b_map))
                mcfit_path, mcfit2_path, _sim = write_mcfit_files(
                    mc_stem,
                    np.asarray(xl, float),
                    b_map,
                    losvd_low_for_mc,
                    losvd_high_for_mc,
                    template_weights_arr,
                    template_weight_samples=(
                        np.asarray(template_weight_samples)
                        if template_weight_samples is not None else None
                    ),
                    losvd_samples=b_samps,
                )
                print(f"[losvd_errs] Wrote mcfit files: {mcfit_path}, {mcfit2_path}")
            except Exception as exc:
                print(f"[losvd_errs] Note: could not write .mcfit/.mcfit2 via lib.scofit.core: {exc}")

            return {"losvd_errs_file": str(outfile), "losvd_center_file": str(center_file),
                    "mcfit_file": str(mcfile), "sim_file": str(sim_path)}

        return {"losvd_errs_file": str(outfile), "losvd_center_file": str(center_file)}

    return {"losvd_errs_file": str(outfile)}


def write_gh_errors_to_files(
    summary: dict,
    prefix: str,
    outdir: str | Path = ".",
) -> dict:
    """
    Write Gauss-Hermite moment errors to file.

    Produces {prefix}.gh_errs.out with columns:
        parameter  value  error  error_lo  error_hi

    Moments: V (velocity), sigma (dispersion), h3, h4

    Parameters
    ----------
    summary : dict
        Output from LOSVDErrorEstimator.summarize()
    prefix : str
        Output file prefix (e.g., "bin0105sp")
    outdir : str or Path
        Output directory

    Returns
    -------
    dict
        {'gh_errs_file': path_to_file}
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gh_map = summary.get("gh_map", {})
    gh_err = summary.get("gh_err_recommended", {})
    gh_lo = summary.get("gh_lo_recommended", {})
    gh_hi = summary.get("gh_hi_recommended", {})
    method = summary.get("method_recommended", "unknown")

    params = ["V", "sigma", "h3", "h4"]
    keys = ["vherm", "sherm", "h3", "h4"]

    outfile = outdir / f"{prefix}.gh_errs.out"
    with open(outfile, "w") as f:
        f.write(f"# Gauss-Hermite moment errors (method: {method})\n")
        f.write("# parameter  value  error  error_lo  error_hi\n")
        for param, key in zip(params, keys):
            val = gh_map.get(key, np.nan)
            err_key = f"gh_{key}"
            err = gh_err.get(err_key, np.nan)
            lo = gh_lo.get(err_key, np.nan)
            hi = gh_hi.get(err_key, np.nan)
            f.write(f"{param:10s} {val:15.6f} {err:15.6f} {lo:15.6f} {hi:15.6f}\n")

    print(f"[losvd_errs] Wrote {outfile}")
    return {"gh_errs_file": str(outfile)}


def write_gh_higherorder_to_file(
    summary: dict,
    prefix: str,
    outdir: str | Path = ".",
) -> dict:
    """Write higher-order (h3..h6) Gauss-Hermite moment errors to a separate file.

    Uses the vdM1993 convention (fit_losvd_gauss_hermite_higher). Values from
    MAP LOSVD fit; errors from bootstrap distribution of each moment.

    Produces {prefix}.gh_ho.out with columns:
        parameter  value  error  error_lo  error_hi
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gh_ho_map = summary.get("gh_ho_map", {})
    gh_ho_err = summary.get("gh_ho_err_recommended", {})
    gh_ho_lo  = summary.get("gh_ho_lo_recommended", {})
    gh_ho_hi  = summary.get("gh_ho_hi_recommended", {})
    method = summary.get("method_recommended", "unknown")
    max_order = int(gh_ho_map.get("max_order", 6))

    params = ["V", "sigma"] + [f"h{n}" for n in range(3, max_order + 1)]
    map_keys = ["vherm", "sherm"] + [f"h{n}" for n in range(3, max_order + 1)]

    outfile = outdir / f"{prefix}.gh_ho.out"
    with open(outfile, "w") as f:
        f.write(f"# Higher-order Gauss-Hermite moments (vdM1993, h3..h{max_order}, method: {method})\n")
        f.write("# parameter  value  error  error_lo  error_hi\n")
        for param, mk in zip(params, map_keys):
            val = gh_ho_map.get(mk, np.nan)
            bk = f"gh_ho_{mk}"
            err = gh_ho_err.get(bk, np.nan)
            lo  = gh_ho_lo.get(bk, np.nan)
            hi  = gh_ho_hi.get(bk, np.nan)
            f.write(f"{param:10s} {val:15.6f} {err:15.6f} {lo:15.6f} {hi:15.6f}\n")

    print(f"[losvd_errs] Wrote {outfile}")
    return {"gh_ho_file": str(outfile)}


def write_bootstrap_diagnostics_to_file(
    bootstrap_result: dict,
    prefix: str,
    outdir: str | Path = ".",
) -> dict:
    """
    Write bootstrap convergence and diagnostic statistics.

    Produces {prefix}.bootstrap_diag.txt with summary statistics.

    Parameters
    ----------
    bootstrap_result : dict
        Output from LOSVDErrorEstimator.residual_bootstrap()
    prefix : str
        Output file prefix (e.g., "bin0105sp")
    outdir : str or Path
        Output directory

    Returns
    -------
    dict
        {'bootstrap_diag_file': path_to_file}
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_success = bootstrap_result.get("n_success", 0)
    n_failed = bootstrap_result.get("n_failed", 0)
    n_total = n_success + n_failed
    convergence_rate = 100.0 * n_success / n_total if n_total > 0 else 0.0

    outfile = outdir / f"{prefix}.bootstrap_diag.txt"
    with open(outfile, "w") as f:
        f.write("Bootstrap Convergence Diagnostics\n")
        f.write("==================================\n")
        f.write(f"Total replicates: {n_total}\n")
        f.write(f"Converged:        {n_success}\n")
        f.write(f"Failed:           {n_failed}\n")
        f.write(f"Convergence rate: {convergence_rate:.1f}%\n")
        if convergence_rate < 95.0:
            f.write("\nWARNING: Low convergence rate (<95%). Error bars may be unreliable.\n")

    print(f"[losvd_errs] Wrote {outfile}")
    return {"bootstrap_diag_file": str(outfile)}


# =============================================================================
# Section 6 – Convenience wrapper
# =============================================================================

def estimate_losvd_errors(
    fit: dict,
    cfg,
    *,
    run_laplace: bool = True,
    run_bootstrap: bool = True,
    run_bias: bool = True,
    n_bootstrap: int = 200,
    block_size: int = 1,
    n_jobs: int = 1,
    bootstrap_seed: int = 42,
    confidence: float = 0.68,
    plot: bool = True,
    write_to_files: bool = False,
    prefix: str | None = None,
    outdir: str | Path = ".",
) -> dict:
    """
    Convenience wrapper: run all requested error estimation methods and
    return the combined summary.

    Parameters
    ----------
    fit          : output of run_spectral_fit()
    cfg          : FitConfig used for the original fit
    run_laplace  : compute Laplace posterior covariance
    run_bootstrap: run residual bootstrap
    run_bias     : compute bias-corrected LOSVD
    n_bootstrap  : number of bootstrap replicates
    block_size   : block size for correlated-noise bootstrap
    n_jobs       : parallel workers for bootstrap (-1 = all CPUs)
    bootstrap_seed: RNG seed
    confidence   : confidence level for intervals (0.68 ≈ 1-sigma)
    plot         : show plots when done
    write_to_files: write error estimates to .losvd_errs.out, .gh_errs.out files
    prefix       : output file prefix (required if write_to_files=True)
    outdir       : output directory for files (default: current directory)

    Returns
    -------
    summary : dict (see LOSVDErrorEstimator.summarize)
        If write_to_files=True, also includes keys:
            'losvd_errs_file': path to {prefix}.losvd_errs.out
            'gh_errs_file': path to {prefix}.gh_errs.out
            'bootstrap_diag_file': path to {prefix}.bootstrap_diag.txt (if bootstrap run)
    """
    est = LOSVDErrorEstimator(fit, cfg)

    laplace = None
    bootstrap = None
    bias = None

    if run_laplace:
        try:
            laplace = est.laplace_covariance()
        except NotImplementedError as exc:
            print(f"[losvd_errs] Skipping Laplace covariance: {exc}")

    if run_bootstrap:
        bootstrap = est.residual_bootstrap(
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            n_jobs=n_jobs,
            seed=bootstrap_seed,
            confidence=confidence,
        )

    if run_bias:
        try:
            bias = est.bias_correction()
        except NotImplementedError as exc:
            print(f"[losvd_errs] Skipping bias correction: {exc}")

    summary = est.summarize(
        laplace_result=laplace,
        bootstrap_result=bootstrap,
        bias_result=bias,
    )

    if plot:
        est.plot_losvd_with_errors(summary)
        if bootstrap is not None:
            est.plot_bootstrap_distribution(bootstrap)
        est.plot_spectrum_with_errors(summary)

    if write_to_files:
        if prefix is None:
            raise ValueError("prefix must be provided if write_to_files=True")
        files_written = {}
        try:
            files_written.update(write_losvd_errors_to_files(summary, prefix, outdir) or {})
        except Exception as exc:
            print(f"[losvd_errs] Error writing losvd errors files: {exc}")
        try:
            files_written.update(write_gh_errors_to_files(summary, prefix, outdir) or {})
        except Exception as exc:
            print(f"[losvd_errs] Error writing GH errors file: {exc}")
        try:
            files_written.update(write_gh_higherorder_to_file(summary, prefix, outdir) or {})
        except Exception as exc:
            print(f"[losvd_errs] Error writing higher-order GH file: {exc}")
        if bootstrap is not None:
            try:
                files_written.update(write_bootstrap_diagnostics_to_file(bootstrap, prefix, outdir) or {})
            except Exception as exc:
                print(f"[losvd_errs] Error writing bootstrap diagnostics: {exc}")
        # Merge file paths into the returned summary for caller convenience
        if files_written:
            summary = dict(summary)
            summary.update(files_written)

    return summary
