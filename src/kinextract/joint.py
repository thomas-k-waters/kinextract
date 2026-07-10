"""Joint continuum-in-the-model LOSVD/template/continuum fit.

This is the (only) continuum-cofitting method for
:func:`~kinextract.fitting.run_spectral_fit`, used when ``cfg.fit_continuum=True``.
An earlier ALS/polynomial outer-loop continuum fit -- estimating the
continuum as a *separate* sub-fit with its own hyperparameter search and
overfitting heuristic -- was removed from the pipeline entirely because of a
structural problem: on a real MUSE spectrum, ALS's hyperparameter search
picked an oversmoothed, near-flat continuum despite a more flexible
smoothing strength giving a healthier chi2_red and visibly tracking the
data's real curvature -- every grid candidate failed the overfitting-floor
check, and the fallback logic is systematically biased toward the
*smoothest* available option. The same failure mode was confirmed for the
standalone polynomial continuum alternative (its own order-search grid falls
back the same way). The core ALS math
(:func:`kinextract.continuum.asymmetric_least_squares_continuum`) remains
available as a standalone, one-time pre-normalization utility (see
``examples/notebooks/06_prenormalized_workflow.ipynb``) -- just no longer as
an in-pipeline continuum-cofitting alternative.

Instead of a separate continuum sub-fit, a continuum is folded directly into
the same flat parameter vector as the LOSVD and template weights, applied
multiplicatively after LOSVD convolution (matching where the shipped ALS
continuum is applied), and optimized in the same single L-BFGS-B call. There
is no outer loop, no continuum hyperparameter grid search, and no
overfitting-floor heuristic.

**Continuum basis: penalized B-spline (P-spline), not a raw polynomial.** A
raw low-order Vandermonde polynomial continuum was tried first and found
wanting: (1) a low order (4-5) genuinely lacks the flexibility to represent
realistic continuum shapes -- on a noiseless mock with a real ALS-derived
continuum, the shipped ALS pipeline recovered V/sigma almost exactly while
the polynomial-continuum joint fit was off by ~10 km/s, forced into a wrong
template mixture to partially compensate for the ~1% continuum residual it
couldn't represent; (2) simply raising the polynomial order didn't fix this
-- going from order 5 to 8 made recovery *worse*, a signature of Vandermonde
ill-conditioning at higher order, not just insufficient degrees of freedom.
A B-spline basis (Eilers & Marx 1996's "P-spline": a modest number of
well-conditioned local basis functions plus an explicit discrete-difference
roughness penalty on their coefficients, mirroring the LOSVD's own
wing-tapered second-derivative penalty) fixes both.

Parameter vector layout: ``a = [b (nl), w (nt), cont_coeffs (n_coef)]`` --
deliberately dropping the shipped pipeline's ``icoff``/``coff``/``coff2``
continuum-offset parameterization (that exists for a narrower "pre-normalized
mode" purpose unrelated to full continuum fitting).

**Resolved false alarm -- not a real bias.** An initial real-MUSE-bin
validation (``bin1605sp``, compared against an entry in the original Fortran
pipeline's ``pallmc.out``) found the self-consistent ``sigl0`` fixed-point
iteration converging robustly but landing on a sigma ~11 km/s above the
"assumed truth." Root-caused: ``pallmc.out`` mixes two distinct binning
schemes (filenames prefixed ``bin`` vs. ``bnn``, ~40/38 split); the
comparison had matched ``bin1605sp`` against a *different* bin
(``bnn1605spmc.sim``) purely by a digit-substring coincidence, not the
correct ground truth for that spectrum. Confirmed by cross-checking against
the shipped ALS pipeline run directly on the actual ``bin1605sp.spec`` file,
which agrees with the joint fit (V~0-14, sigma~50-57) and disagrees with the
mismatched ``bnn1605spmc.sim`` entry (itself only weakly constrained:
sigma=40.7+/-8.9). The genuine residual bias at the ``sigl0`` fixed point is
the small ~2 km/s one already characterized in synthetic single-template
tests (see :func:`fit_joint_auto_xlam_sigl0`'s docstring).
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from ._utils import log
from .fitting import _discrepancy_principle_search, compute_losvd_n_peaks, compute_losvd_roughness
from .io import write_fitlov_outputs_from_model
from .losvd import fit_losvd_gauss_hermite
from .numerics import (
    _compute_smoothness,
    _convolve_losvd_numba,
    _wing_taper_lam_vec,
    estimate_velocity_xcorr,
    jax,
    jnp,
)
from .state import FitState, getnlosvd_fast_from_b

# Default P-spline continuum regularization weight. Chosen empirically: large
# enough to keep the ~14-16 coefficient continuum from overfitting individual
# noisy pixels, small enough to still track realistic multi-cycle continuum
# structure (verified against a flux-calibration-ripple stress test). Applied
# to *normalized* coefficients (see `build_jax_objective_value_and_grad`/
# `objective_joint`), so this default is portable across datasets with very
# different absolute flux units.
DEFAULT_XLAM_CONT = 3.0


def build_pspline_design(
    x: np.ndarray, n_interior_knots: int = 10, degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a B-spline design matrix for the continuum.

    Parameters
    ----------
    x : ndarray
        Wavelength (or pixel) array the continuum is defined on, length
        ``npix``.
    n_interior_knots : int, optional
        Number of interior knots spanning ``[x.min(), x.max()]``. More knots
        -> more local flexibility; ``n_coef = n_interior_knots + degree + 1``
        coefficients result. Default 10 (with a cubic spline, 14
        coefficients) comfortably resolves a handful of continuum "features"
        across a typical few-hundred-pixel fit window without approaching
        the LOSVD's own 29 degrees of freedom.
    degree : int, optional
        B-spline polynomial degree per segment. Default 3 (cubic), the
        standard smooth choice (continuous up to the second derivative).

    Returns
    -------
    design : ndarray, shape (npix, n_coef)
        B-spline design matrix (clamped at both boundaries, i.e. the first
        and last basis functions are supported exactly at the endpoints), so
        ``design @ cont_coeffs`` directly gives the continuum in the same
        (flux) units as the data.
    knots : ndarray
        The full (boundary-clamped) knot vector used to build `design`,
        returned for reference/reuse.
    """
    x = np.asarray(x, float)
    x_min, x_max = float(x.min()), float(x.max())
    interior = np.linspace(x_min, x_max, n_interior_knots + 2)
    knots = np.concatenate([np.full(degree, x_min), interior, np.full(degree, x_max)])
    design = BSpline.design_matrix(x, knots, degree).toarray()
    return design, knots


def difference_penalty_matrix(n_coef: int, diff_order: int = 2) -> np.ndarray:
    """Discrete difference operator for a P-spline roughness penalty.

    ``D @ coeffs`` gives the `diff_order`-th finite difference of the
    coefficient sequence; the roughness penalty is ``sum((D @ coeffs)**2)``,
    the standard P-spline penalty (Eilers & Marx 1996) and the direct
    coefficient-space analogue of the LOSVD's own second-derivative
    smoothness penalty.
    """
    D = np.eye(n_coef)
    for _ in range(diff_order):
        D = np.diff(D, axis=0)
    return D


def initial_continuum_coeffs(g: np.ndarray, good: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Least-squares initial guess for the continuum coefficients.

    Ignores absorption/emission features (an unweighted least-squares fit
    over the good pixels) -- this only needs to be a reasonable starting
    point for the optimizer, not an accurate continuum estimate on its own;
    the joint fit refines it using the actual template/LOSVD model.
    """
    coeffs, *_ = np.linalg.lstsq(design[good], g[good], rcond=None)
    return coeffs


def normalize_template_matrix(st: FitState) -> tuple[np.ndarray, np.ndarray]:
    """Rescale each template column to a common median level of 1.

    The bundled MUSE templates are NOT cross-normalized to a common flux
    level (per-template median ranges from ~0.5 to ~7.3, a 14x spread). In
    the retired ALS-outer-loop continuum-cofitting approach this didn't
    matter, because the continuum was re-fit via a separate, anchored
    regression at each outer iteration (against a *fixed* current template
    mixture) rather than varying jointly with the template weights. In a
    single joint
    optimization, though, leaving that spread in place opens a genuine
    degeneracy: shifting the template mix toward higher- or lower-level
    templates changes the weighted mixture's overall (multiplicative) level
    by more than an order of magnitude, which trades off against the
    continuum's own level in an essentially unconstrained way -- confirmed
    empirically: without this normalization, the joint fit converges to a
    continuum ~2-3x the data's actual scale with a suspiciously low chi2_red,
    i.e. overfitting via this degeneracy rather than genuinely explaining the
    spectrum. Normalizing here removes the degeneracy at its root.

    Returns
    -------
    t_norm : ndarray, shape (npix, nt)
        Template matrix with each column divided by its own median.
    template_scale : ndarray, shape (nt,)
        The per-template median divisor applied, returned so callers can
        convert normalized-basis template weights back to the original
        template files' own units if ever needed.
    """
    template_scale = np.median(st.t, axis=0)
    template_scale = np.where(template_scale == 0, 1.0, template_scale)
    t_norm = st.t / template_scale[None, :]
    return t_norm, template_scale


def evaluate_model_gp_joint(
    a: np.ndarray, st: FitState, design: np.ndarray, t_norm: np.ndarray,
    apply_continuum: bool = True,
) -> tuple:
    """Evaluate the joint forward model for a trial parameter vector.

    Parameters
    ----------
    a : ndarray
        Flat parameter vector: ``nl`` LOSVD bin values, ``nt`` template
        weights, then ``n_coef = design.shape[1]`` continuum P-spline
        coefficients.
    st : FitState
        Fit state supplying the data/template/LOSVD-grid machinery.
    design : ndarray, shape (npix, n_coef)
        Continuum B-spline design matrix from :func:`build_pspline_design`.
    apply_continuum : bool, optional
        If True (default), multiply the model by the continuum. Set False
        to get the continuum-free model (e.g. for diagnostics).

    Returns
    -------
    gp : ndarray
        Model spectrum, length ``npix``.
    b : ndarray
        LOSVD histogram used to build the model (floored at 1e-6).
    w : ndarray
        Template weights used to build the model (floored at 1e-12).
    cont_coeffs : ndarray
        Continuum P-spline coefficients used to build the model.
    cont : ndarray
        The continuum itself, ``design @ cont_coeffs``, length ``npix``.
    """
    nl, nt, npix = st.nl, st.nt, st.npix
    i = 0
    b = np.maximum(a[i:i + nl], 1e-6); i += nl
    # Hard-normalize b to sum to exactly 1, rather than relying on the shipped
    # pipeline's soft "0.1*|sum(b)-1|" penalty: with a free multiplicative
    # continuum in the same joint fit, that penalty is far too weak to anchor
    # sum(b) -- confirmed empirically, the optimizer let sum(b) drift to
    # ~0.4 and had the continuum's level compensate for the missing
    # amplitude. Hard-normalizing removes the degeneracy by construction.
    b = b / np.sum(b)
    w = np.maximum(a[i:i + nt], 1e-12); i += nt
    n_coef = design.shape[1]
    cont_coeffs = a[i:i + n_coef]; i += n_coef

    sum2 = float(np.sum(w)) or 1.0
    if st.fortran_template_mixture:
        tval = t_norm @ w / sum2
    else:
        valid_t = np.where(st.outside_tpl, 0.0, t_norm)
        s2o = sum2 - (st.outside_tpl * w[None, :]).sum(axis=1)
        s2o = np.where(s2o == 0, 1.0, s2o)
        tval = (valid_t * w[None, :]).sum(axis=1) / s2o

    _, ynew = getnlosvd_fast_from_b(st, b)
    gp, xs = _convolve_losvd_numba(tval, ynew, st.ip_map, st.ip_mask, npix, st.nlosvd)
    suml = float(np.sum(ynew))
    gp = np.where(xs != 0, gp * suml / xs, gp)

    # n_coef == 0 means continuum fitting is off (see fit_joint's
    # `fit_continuum` parameter, used for pre-normalized-mode fits where
    # there is no continuum to co-fit): design @ [] is a zero vector, not a
    # neutral one, so it must be special-cased rather than trusting the
    # matmul.
    cont = design @ cont_coeffs if design.shape[1] > 0 else np.ones(npix)
    if apply_continuum:
        gp = gp * cont

    return gp, b, w, cont_coeffs, cont


def objective_joint(
    a: np.ndarray, st: FitState, design: np.ndarray, t_norm: np.ndarray,
    D: np.ndarray | None = None, coeff_scale: float = 1.0,
    xlam_cont: float = DEFAULT_XLAM_CONT,
) -> float:
    """Scalar objective: chi2 + wing-tapered LOSVD smoothness + continuum roughness.

    Close in structure to ``kinextract.numerics.objective_map`` -- reuses
    :func:`kinextract.numerics._compute_smoothness` directly for the
    wing-tapered LOSVD regularization term. Differences from the shipped
    objective: the continuum is folded into :func:`evaluate_model_gp_joint`
    instead of being a separate co-fitted array; the LOSVD normalization is
    a hard constraint (``b`` is renormalized to sum to 1 inside
    :func:`evaluate_model_gp_joint`) rather than the shipped pipeline's soft
    ``0.1*|sum(b)-1|`` penalty; and a P-spline roughness penalty (``D``
    applied to the coefficients, normalized by `coeff_scale` so the penalty
    is dimensionless regardless of the data's absolute flux units) replaces
    the old polynomial continuum's implicit "few coefficients can't overfit"
    safety margin.
    """
    gp, b, w, cont_coeffs, cont = evaluate_model_gp_joint(a, st, design, t_norm)

    good = (
        (st.gerr > 0.0) & (st.gerr < 1.0e9)
        & np.isfinite(st.g) & np.isfinite(gp) & np.isfinite(st.gerr)
    )
    resid = (st.g - gp) / st.gerr
    chi2 = float(np.sum(np.where(good, resid * resid, 0.0)))

    v_center = float(getattr(st, "v_center", 0.0))
    smooth = _compute_smoothness(b, st.xl, st.xlam, st.sigl0, st.resd, st.nl, v_center)

    cont_smooth = 0.0
    if D is not None:
        d = D @ (cont_coeffs / coeff_scale)
        cont_smooth = float(xlam_cont * np.sum(d * d))

    return chi2 + smooth + cont_smooth


def build_jax_objective_value_and_grad(
    st: FitState, design: np.ndarray, t_norm: np.ndarray,
    D: np.ndarray | None = None, coeff_scale: float = 1.0,
    xlam_cont: float = DEFAULT_XLAM_CONT,
):
    """Build a JAX-jitted value-and-gradient function for the joint objective.

    Mirrors ``kinextract.numerics._build_jax_objective_value_and_grad``'s
    structure (same wing-taper precomputation, same discrete pixel-scatter
    convolution written directly in ``jax.numpy`` rather than calling the
    Numba kernel), adapted for this module's simplified parameter layout
    (no ``icoff``/``coff``/``coff2``/global-amplitude, hard-normalized `b`,
    and a design-matrix continuum instead of a separately co-fitted array),
    plus the P-spline continuum roughness penalty (see :func:`objective_joint`).

    Returns
    -------
    callable or None
        ``f(a, g, gerr) -> (value, grad)`` with `g`/`gerr` as explicit
        (non-differentiated) arguments so the same compiled kernel can be
        reused across calls that only change `g`/`gerr`. Returns None if JAX
        is not installed.
    """
    if jax is None or jnp is None:
        return None
    jax.config.update("jax_enable_x64", True)

    nl, nt, npix = int(st.nl), int(st.nt), int(st.npix)
    n_poly = design.shape[1]
    fortran_template_mixture = bool(st.fortran_template_mixture)
    D_j = None if D is None else jnp.asarray(D, dtype=jnp.float64)
    coeff_scale_j = float(coeff_scale)
    xlam_cont_j = float(xlam_cont)

    t = jnp.asarray(t_norm, dtype=jnp.float64)
    outside_tpl = jnp.asarray(st.outside_tpl, dtype=bool)
    losvd_j0 = jnp.asarray(st.losvd_j0, dtype=jnp.int32)
    losvd_j1 = jnp.asarray(st.losvd_j1, dtype=jnp.int32)
    losvd_w = jnp.asarray(st.losvd_w, dtype=jnp.float64)
    ip_mask_2d = jnp.asarray(st.ip_mask, dtype=jnp.float64)
    ip_safe = jnp.clip(jnp.asarray(st.ip_map, dtype=jnp.int32), 0, npix - 1)
    design_j = jnp.asarray(design, dtype=jnp.float64)

    xlam = float(st.xlam)
    sigl0 = float(st.sigl0)
    resd = float(st.resd)
    iskip = int(st.iskip)
    v_center = float(getattr(st, "v_center", 0.0))

    fit_mask = np.zeros(npix, dtype=bool)
    fit_mask[iskip:npix - iskip] = True
    fit_mask = jnp.asarray(fit_mask, dtype=bool)

    lam_vec = jnp.asarray(_wing_taper_lam_vec(st.xl, xlam, sigl0, v_center), dtype=jnp.float64)

    def _smooth_penalty(b):
        left = (b[1] - 2.0 * b[0]) ** 2
        right = (b[nl - 2] - 2.0 * b[nl - 1]) ** 2
        mid = (b[2:] - 2.0 * b[1:-1] + b[:-2]) ** 2
        terms = jnp.concatenate([jnp.array([left]), mid, jnp.array([right])])
        return jnp.sum(lam_vec * terms) / resd

    def _ynew_from_b(b):
        y2 = b[losvd_j0] + (b[losvd_j1] - b[losvd_j0]) * losvd_w
        s = jnp.sum(y2)
        sum_b = jnp.sum(b)  # == 1 here since b is hard-normalized before this point
        scale = jnp.where(s != 0.0, sum_b / s, 1.0)
        return y2 * scale

    def _objective(a, g, gerr_dyn):
        i = 0
        b_raw = jnp.maximum(a[i:i + nl], 1e-6)
        b = b_raw / jnp.sum(b_raw)  # hard normalization -- see evaluate_model_gp_joint
        i += nl
        w = jnp.maximum(a[i:i + nt], 1e-12)
        i += nt
        cont_coeffs = a[i:i + n_poly]
        i += n_poly

        sum2 = jnp.sum(w)
        sum2 = jnp.where(sum2 != 0.0, sum2, 1.0)
        if fortran_template_mixture:
            tval = (t @ w) / sum2
        else:
            valid_t = jnp.where(outside_tpl, 0.0, t)
            s2o = sum2 - jnp.sum(outside_tpl * w[None, :], axis=1)
            s2o = jnp.where(s2o == 0.0, 1.0, s2o)
            tval = jnp.sum(valid_t * w[None, :], axis=1) / s2o

        ynew = _ynew_from_b(b)
        contrib = tval[:, None] * ynew[None, :] * ip_mask_2d
        xs_contrib = ynew[None, :] * ip_mask_2d
        gp = jnp.zeros(npix, dtype=jnp.float64).at[ip_safe].add(contrib)
        xs = jnp.zeros(npix, dtype=jnp.float64).at[ip_safe].add(xs_contrib)

        suml = jnp.sum(ynew)
        gp = jnp.where(xs != 0.0, gp * suml / xs, gp)

        # n_poly == 0 means continuum fitting is off -- see the NumPy path's
        # identical special-case in evaluate_model_gp_joint.
        cont = design_j @ cont_coeffs if n_poly > 0 else jnp.ones(npix, dtype=jnp.float64)
        gp = gp * cont

        valid = (
            fit_mask
            & jnp.isfinite(g) & jnp.isfinite(gp) & jnp.isfinite(gerr_dyn)
            & (gerr_dyn > 0.0) & (gerr_dyn < 1.0e9)
        )
        resid = (g - gp) / gerr_dyn
        chi2 = jnp.sum(jnp.where(valid, resid * resid, 0.0))
        smooth = _smooth_penalty(b)
        cont_smooth = 0.0
        if D_j is not None:
            d = D_j @ (cont_coeffs / coeff_scale_j)
            cont_smooth = xlam_cont_j * jnp.sum(d * d)
        return chi2 + smooth + cont_smooth

    obj_vg = jax.jit(jax.value_and_grad(_objective))

    def _value_and_grad_np(a_np: np.ndarray, g_np: np.ndarray, gerr_np: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = obj_vg(
            jnp.asarray(a_np, dtype=jnp.float64),
            jnp.asarray(g_np, dtype=jnp.float64),
            jnp.asarray(gerr_np, dtype=jnp.float64),
        )
        return float(val), np.asarray(grad, dtype=np.float64)

    return _value_and_grad_np


def build_initial_guess(
    st: FitState, n_interior_knots: int = 10, degree: int = 3,
    b_bounds: tuple = (1e-6, 1.0),
    w_bounds: tuple = (1e-5, 1.0),
    cont_bound_factor: float = 8.0,
    fit_continuum: bool = True,
) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """Build the initial flat parameter vector, bounds, and xscale.

    Parameters
    ----------
    st : FitState
        Fit state.
    n_interior_knots, degree : optional
        Passed to :func:`build_pspline_design`.
    b_bounds, w_bounds : tuple of (float, float), optional
        Per-element bounds for the LOSVD histogram and template weights,
        matching ``kinextract.spectrum.build_initial_guess_nonparam``'s
        defaults.
    cont_bound_factor : float, optional
        Continuum coefficient bounds are ``coeffs0 +/- cont_bound_factor *
        max(|coeffs0|, floor)`` -- wide, since B-spline coefficients are
        local (each one only affects a small support region) and can
        legitimately need to move far from the unweighted least-squares
        initial guess to fit a real absorption-line-heavy window.
    fit_continuum : bool, optional
        If False, the continuum is not fit at all -- `design` is a
        ``(npix, 0)`` matrix and `x0`/`bounds`/`xscale` carry no continuum
        entries, so :func:`evaluate_model_gp_joint` treats the continuum as
        a fixed 1.0 (see its docstring). For pre-normalized input (the
        data's own continuum has already been divided out), a free
        P-spline continuum has nothing real to fit and is better left out
        entirely rather than letting it wander within its regularization
        penalty.

    Returns
    -------
    x0 : ndarray
        Initial flat parameter vector.
    bounds : list of (float, float)
        Per-parameter bounds, same order as `x0`.
    xscale : ndarray
        Per-parameter scale divisor for L-BFGS-B preconditioning.
    design : ndarray, shape (npix, n_coef)
        Continuum B-spline design matrix (needed by every subsequent call).
        ``n_coef == 0`` if `fit_continuum` is False.
    """
    nl, nt = st.nl, st.nt
    good = (st.gerr > 0.0) & (st.gerr < 1.0e9) & np.isfinite(st.g)

    if fit_continuum:
        design, _knots = build_pspline_design(st.x, n_interior_knots, degree)
        cont0 = initial_continuum_coeffs(st.g, good, design)
    else:
        design = np.zeros((st.npix, 0))
        cont0 = np.zeros(0)

    b0 = np.full(nl, 1.0 / nl)
    w0 = np.full(nt, 1.0 / nt)
    x0 = np.concatenate([b0, w0, cont0])

    cont_bound_width = cont_bound_factor * np.maximum(np.abs(cont0), 1e-3 * np.max(np.abs(cont0), initial=0.0))
    cont_bounds = list(zip(cont0 - cont_bound_width, cont0 + cont_bound_width))

    bounds = (
        [b_bounds] * nl
        + [w_bounds] * nt
        + cont_bounds
    )

    cont_scale = np.maximum(np.abs(cont0), 1e-6 * np.max(np.abs(cont0), initial=0.0) + 1e-12)
    xscale = np.concatenate([np.ones(nl), np.full(nt, 0.1), cont_scale])

    return x0, bounds, xscale, design


def fit_joint(
    st: FitState, n_interior_knots: int = 10, degree: int = 3,
    xlam_cont: float = DEFAULT_XLAM_CONT, cont_diff_order: int = 2,
    maxiter: int = 50000, ftol: float = 1e-12, maxfun: int = 500000,
    use_jax: bool = True,
    auto_recenter_v: bool = False,
    fit_continuum: bool = True,
    w_bounds: tuple = (1e-5, 1.0),
):
    """Run the joint LOSVD + template + continuum-P-spline MAP fit.

    Parameters
    ----------
    st : FitState
        Fit state to optimize (not modified unless `auto_recenter_v`).
    n_interior_knots, degree : optional
        Passed to :func:`build_pspline_design`. Defaults (10 interior knots,
        cubic -> 14 coefficients) give enough local flexibility to track
        realistic continuum structure without approaching the LOSVD's own
        29 degrees of freedom.
    xlam_cont : float, optional
        P-spline roughness penalty weight, applied to the continuum
        coefficients normalized by their own initial-guess scale (see
        :func:`objective_joint`) -- portable across datasets with very
        different absolute flux units.
    cont_diff_order : int, optional
        Order of the discrete difference penalty on the continuum
        coefficients (see :func:`difference_penalty_matrix`). Default 2
        (penalizes curvature, i.e. a straight-line continuum is free),
        matching the standard P-spline convention.
    maxiter, ftol, maxfun : optional
        Passed through to ``scipy.optimize.minimize(method="L-BFGS-B")``.
    use_jax : bool, optional
        If True (default), use exact analytic gradients via
        :func:`build_jax_objective_value_and_grad` instead of scipy's
        finite-difference approximation. Falls back to finite differences
        automatically if JAX isn't installed.
    auto_recenter_v : bool, optional
        If True, recenter the wing-taper's regularization on
        ``kinextract.numerics.estimate_velocity_xcorr(st)`` instead of the
        fixed ``v_center=0.0`` the shipped MAP path always uses. A dedicated
        stress test found this cuts V-recovery bias by ~90% in the regime it
        matters (clean/high-S/N data, xlam already tuned high). Default
        False here because :func:`fit_joint_auto_xlam` always recenters
        itself as a prerequisite step (see its docstring) -- this flag only
        matters for callers invoking :func:`fit_joint` directly at a single,
        fixed xlam.
    fit_continuum : bool, optional
        If False (default True), no continuum is fit at all -- the
        continuum is fixed at 1.0 (see :func:`build_initial_guess`'s
        `fit_continuum` parameter and :func:`evaluate_model_gp_joint`).
        Use this for pre-normalized input, where the data's own continuum
        has already been divided out and there is nothing genuine for a
        free P-spline to fit.
    w_bounds : tuple, optional
        Per-element template-weight bounds, passed through to
        :func:`build_initial_guess`. Default non-negative; widen to allow
        negative values (e.g. ``(-1.0, 1.0)``) when fitting with
        :func:`kinextract.templates.reduce_templates_svd` eigen-templates.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        ``result.x`` is in *physical* (unscaled) units.
    design : ndarray
        Continuum B-spline design matrix used for this fit.
    """
    if auto_recenter_v:
        st = dataclasses.replace(st, v_center=estimate_velocity_xcorr(st))

    x0, bounds, xscale, design = build_initial_guess(
        st, n_interior_knots, degree, fit_continuum=fit_continuum, w_bounds=w_bounds)
    t_norm, _template_scale = normalize_template_matrix(st)
    u0 = x0 / xscale
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

    value_and_grad = (
        build_jax_objective_value_and_grad(st, design, t_norm, D, coeff_scale, xlam_cont)
        if use_jax else None
    )

    if value_and_grad is not None:
        def _obj_jac(u):
            a = u * xscale
            val, grad = value_and_grad(a, st.g, st.gerr)
            return val, grad * xscale

        result = minimize(
            _obj_jac, u0, method="L-BFGS-B", bounds=u_bounds, jac=True,
            options={"maxiter": maxiter, "maxfun": maxfun, "ftol": ftol},
        )
    else:
        def _obj(u):
            a = u * xscale
            return objective_joint(a, st, design, t_norm, D, coeff_scale, xlam_cont)

        result = minimize(
            _obj, u0, method="L-BFGS-B", bounds=u_bounds,
            options={"maxiter": maxiter, "maxfun": maxfun, "ftol": ftol},
        )

    result.x = result.x * xscale
    return result, design


def fit_joint_auto_xlam(
    st: FitState, n_interior_knots: int = 10, degree: int = 3,
    xlam_grid: tuple = (10.0, 100.0, 1000.0, 10000.0, 100000.0),
    xlam_chi2_tolerance: float = 0.02, xlam_max_peaks: int = 1,
    xlam_peak_min_prominence: float = 0.1,
    recenter_v: bool = True,
    xlam_cont: float = DEFAULT_XLAM_CONT, cont_diff_order: int = 2,
    maxiter: int = 50000, ftol: float = 1e-12, maxfun: int = 500000,
    use_jax: bool = True,
    fit_continuum: bool = True,
    xlam_criterion: str = "chi2",
    xlam_discrepancy_nsigma: float = 1.0,
    w_bounds: tuple = (1e-5, 1.0),
):
    """Automatically select the joint model's LOSVD regularization strength.

    Two criteria are supported (``xlam_criterion``):

    ``"chi2"``/``"roughness"`` (default ``"chi2"``)
        Mirrors ``kinextract.fitting._auto_select_xlam``'s chi2 criterion
        exactly (same selection rule, reusing its ``compute_losvd_roughness``/
        ``compute_losvd_n_peaks`` helpers directly): run a full fit at every
        grid point, then among the *unimodal* candidates (``n_peaks <=
        xlam_max_peaks``) pick the **largest** xlam whose reduced chi2 is
        within ``xlam_chi2_tolerance`` of the grid minimum -- "use as much
        smoothing as possible without visibly hurting the fit."

    ``"discrepancy"``
        Delegates to :func:`kinextract.fitting._discrepancy_principle_search`
        (the same shared root-finder the shipped MAP path's
        :func:`kinextract.fitting._auto_select_xlam_discrepancy` uses),
        replacing the grid scan with a 1-D root-find over ``log(xlam)``
        targeting a known chi2 rise rather than a tolerance-from-minimum
        rule. Only ``xlam_grid``'s min/max are used, as the search's initial
        bracket. If `xlam_grid` has only one element (``cfg.xlam_auto=False``
        convention -- see :func:`run_joint_fit`), falls back to the
        ``"chi2"`` path's trivial single-candidate behavior instead of
        attempting a degenerate 1-D search.

    **Difference from the shipped selector, and why it matters:** this
    always recenters the wing-taper's ``v_center`` on
    ``kinextract.numerics.estimate_velocity_xcorr(st)`` *before* running the
    grid search (see `recenter_v`), rather than treating recentering as a
    separate, optional step layered on afterward. A dedicated diagnostic
    found the "prefer the largest xlam that doesn't hurt chi2" rule
    systematically walks toward high xlam whenever the data is clean enough
    not to visibly penalize it there -- and separately, that high xlam is
    exactly the regime where the wing-taper's fixed ``v_center=0``
    convention biases recovered V the most. Recentering first removes the
    asymmetry regardless of which xlam the search lands on.

    Parameters
    ----------
    st : FitState
        Fit state to optimize (not modified).
    n_interior_knots, degree, xlam_cont, cont_diff_order, maxiter, ftol,
    maxfun, use_jax : optional
        Passed through to each grid-point :func:`fit_joint` call.
    xlam_grid : tuple of float, optional
        Candidate xlam values, matching the shipped pipeline's typical
        grid span.
    xlam_chi2_tolerance : float, optional
        Fractional chi2 tolerance for the "largest xlam within tolerance
        of the minimum" rule.
    xlam_max_peaks : int, optional
        Maximum number of prominent LOSVD peaks (via
        ``compute_losvd_n_peaks``) for a candidate to count as unimodal.
    xlam_peak_min_prominence : float, optional
        Prominence threshold passed to ``compute_losvd_n_peaks``.
    recenter_v : bool, optional
        If True (default), recenter ``v_center`` on
        ``estimate_velocity_xcorr(st)`` once before the grid search.
    fit_continuum : bool, optional
        Passed through to each :func:`fit_joint` call -- see its docstring.
        Set False for pre-normalized input (no continuum to co-fit).

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The final fit at the selected xlam (physical units).
    design : ndarray
        Continuum B-spline design matrix for the final fit.
    best_xlam : float
        The selected xlam value.
    records : list of tuple
        ``(xlam, chi2_red, roughness, n_peaks)`` for every grid point, for
        diagnostics.
    """
    v_center_est = float(getattr(st, "v_center", 0.0))
    if recenter_v:
        v_center_est = estimate_velocity_xcorr(st)
        st = dataclasses.replace(st, v_center=v_center_est)

    grid = sorted(float(x) for x in xlam_grid)

    if str(xlam_criterion).lower() == "discrepancy" and len(grid) > 1:
        return _fit_joint_discrepancy_xlam(
            st, n_interior_knots, degree, xlam_grid=grid,
            xlam_max_peaks=xlam_max_peaks, xlam_peak_min_prominence=xlam_peak_min_prominence,
            v_center_est=v_center_est,
            xlam_cont=xlam_cont, cont_diff_order=cont_diff_order,
            maxiter=maxiter, ftol=ftol, maxfun=maxfun, use_jax=use_jax,
            fit_continuum=fit_continuum, nsigma=xlam_discrepancy_nsigma,
            w_bounds=w_bounds,
        )

    records = []
    fits = {}
    gh_by_xlam = {}
    for xlam in grid:
        st_try = dataclasses.replace(st, xlam=xlam)
        result, design = fit_joint(
            st_try, n_interior_knots, degree, xlam_cont=xlam_cont,
            cont_diff_order=cont_diff_order, maxiter=maxiter, ftol=ftol,
            maxfun=maxfun, use_jax=use_jax, auto_recenter_v=False,
            w_bounds=w_bounds,
            fit_continuum=fit_continuum,
        )
        t_norm, _ = normalize_template_matrix(st_try)
        gp, b, w, cont_coeffs, cont = evaluate_model_gp_joint(result.x, st_try, design, t_norm)
        good = (st_try.gerr > 0.0) & (st_try.gerr < 1.0e9) & np.isfinite(st_try.g) & np.isfinite(gp)
        resid = (st_try.g - gp) / st_try.gerr
        chi2_red = float(np.sum(np.where(good, resid * resid, 0.0))) / max(good.sum() - len(result.x), 1)
        roughness = compute_losvd_roughness(b)
        n_peaks = compute_losvd_n_peaks(b, xlam_peak_min_prominence)
        records.append((xlam, chi2_red, roughness, n_peaks))
        fits[xlam] = (result, design)
        gh_by_xlam[xlam] = fit_losvd_gauss_hermite(st_try.xl, b, fit_h3h4=True)

    def _select(candidates):
        """Same chi2-tolerance rule as the shipped pipeline, restricted to
        `candidates` (a subset of `records` still in play)."""
        chi2_reds = [r[1] for r in candidates if r[3] <= xlam_max_peaks]
        if not chi2_reds:
            return None
        chi2_min = min(chi2_reds)
        chi2_max_allowed = chi2_min * (1.0 + xlam_chi2_tolerance)
        for xlam, chi2_red, roughness, n_peaks in reversed(candidates):
            if n_peaks <= xlam_max_peaks and chi2_red <= chi2_max_allowed:
                return xlam
        return None

    def _candidate_ok(candidate, pool):
        """Two independent sanity checks against a spurious-overfit failure
        mode found on a real MUSE bin near the resolution limit: there, the
        smallest xlam in the grid found a genuinely different, lower-chi2
        local optimum (V off by >40 km/s) that the chi2-tolerance rule
        couldn't distinguish from the correct answer, since nothing else
        came within 2% of that spurious minimum.

        1. Compare against the pre-fit xcorr velocity estimate. Necessary
           but not sufficient: on the bin that motivated this, the xcorr
           estimate itself was *also* wrong in the same direction, so this
           check alone did not catch it.
        2. Compare against the *other* grid candidates' own median V
           (robustly, via MAD). This is what actually caught the failure:
           every candidate except the spurious one clustered within a few
           km/s of each other, while the spurious one was a >40 km/s
           outlier -- independent of whether the external xcorr estimate
           can be trusted.
        """
        v_candidate = gh_by_xlam[candidate]["vherm"]
        sigma_candidate = max(gh_by_xlam[candidate]["sherm"], 1.0)
        if abs(v_candidate - v_center_est) > 5.0 * sigma_candidate:
            return False

        other_vs = [gh_by_xlam[x]["vherm"] for x, _, _, npk in pool
                    if x != candidate and npk <= xlam_max_peaks]
        if len(other_vs) >= 2:
            median_other_v = float(np.median(other_vs))
            mad_other_v = float(np.median(np.abs(np.asarray(other_vs) - median_other_v)))
            scale = max(mad_other_v, 3.0)
            if abs(v_candidate - median_other_v) > 5.0 * scale:
                return False
        return True

    # Excluding a flagged candidate and re-running the same selection rule
    # on what's left lets a sensible higher-xlam solution win instead.
    remaining = list(records)
    best_xlam = grid[-1]
    while remaining:
        candidate = _select(remaining)
        if candidate is None:
            break
        if _candidate_ok(candidate, records):
            best_xlam = candidate
            break
        remaining = [r for r in remaining if r[0] != candidate]

    result, design = fits[best_xlam]
    return result, design, best_xlam, records


def _fit_joint_discrepancy_xlam(
    st: FitState, n_interior_knots: int, degree: int,
    xlam_grid: list, xlam_max_peaks: int, xlam_peak_min_prominence: float,
    v_center_est: float,
    xlam_cont: float, cont_diff_order: int,
    maxiter: int, ftol: float, maxfun: int, use_jax: bool,
    fit_continuum: bool, nsigma: float,
    w_bounds: tuple = (1e-5, 1.0),
):
    """Joint-model discrepancy-principle xlam selector -- the
    ``xlam_criterion="discrepancy"`` branch of :func:`fit_joint_auto_xlam`,
    factored out to keep that function's grid-search branch readable.

    Wraps :func:`kinextract.fitting._discrepancy_principle_search` (the
    same search the shipped MAP path's
    :func:`kinextract.fitting._auto_select_xlam_discrepancy` uses) with an
    ``evaluate_xlam`` closure that runs one :func:`fit_joint` call per
    trial xlam instead of :func:`kinextract.fitting._fit_map_once` --
    the two paths have entirely different objective functions and flat-
    parameter-vector layouts, so only the *search algorithm* is shared, not
    the per-trial fit mechanics (see the module docstring's plan reference).
    ``v_center_est`` is assumed already resolved by the caller (recentering,
    if requested, happens once in :func:`fit_joint_auto_xlam` before this is
    called, not per-trial here).

    Returns
    -------
    Same 4-tuple as :func:`fit_joint_auto_xlam`: ``(result, design,
    best_xlam, records)``.
    """
    fits: dict = {}

    def evaluate_xlam(xlam: float) -> dict:
        st_try = dataclasses.replace(st, xlam=xlam)
        result, design = fit_joint(
            st_try, n_interior_knots, degree, xlam_cont=xlam_cont,
            cont_diff_order=cont_diff_order, maxiter=maxiter, ftol=ftol,
            maxfun=maxfun, use_jax=use_jax, auto_recenter_v=False,
            fit_continuum=fit_continuum, w_bounds=w_bounds,
        )
        t_norm, _ = normalize_template_matrix(st_try)
        gp, b, w, cont_coeffs, cont = evaluate_model_gp_joint(result.x, st_try, design, t_norm)
        good = (st_try.gerr > 0.0) & (st_try.gerr < 1.0e9) & np.isfinite(st_try.g) & np.isfinite(gp)
        resid = (st_try.g - gp) / st_try.gerr
        chi2_total = float(np.sum(np.where(good, resid * resid, 0.0)))
        ngood = int(good.sum())
        n_peaks = compute_losvd_n_peaks(b, xlam_peak_min_prominence)
        roughness = compute_losvd_roughness(b)
        gh = fit_losvd_gauss_hermite(st_try.xl, b, fit_h3h4=True)
        fits[xlam] = (result, design, roughness, ngood, len(result.x))
        return dict(chi2_total=chi2_total, ngood=ngood, n_peaks=n_peaks,
                     v_rec=float(gh["vherm"]), sigma_rec=float(gh["sherm"]))

    best_xlam, trials = _discrepancy_principle_search(
        evaluate_xlam, xlam_lo=xlam_grid[0], xlam_hi=xlam_grid[-1],
        max_peaks=xlam_max_peaks, v_center_est=v_center_est, nsigma=nsigma,
        log_label="Auto-xlam (joint, discrepancy)",
    )

    result, design, _roughness, _ngood, _nparams = fits[best_xlam]
    # Same (xlam, chi2_red, roughness, n_peaks) shape as the grid-search
    # branch, for any caller relying on records' shape (diagnostics only).
    records = [
        (xlam, t["chi2_total"] / max(fits[xlam][3] - fits[xlam][4], 1), fits[xlam][2], t["n_peaks"])
        for xlam, t in sorted(trials.items())
    ]
    return result, design, best_xlam, records


def fit_joint_auto_xlam_sigl0(
    st: FitState, n_interior_knots: int = 10, degree: int = 3,
    sigl0_init: float = 100.0, n_sigl0_iter: int = 3, sigl0_tol: float = 2.0,
    xlam_grid: tuple = (10.0, 100.0, 1000.0, 10000.0, 100000.0),
    xlam_chi2_tolerance: float = 0.02, xlam_max_peaks: int = 1,
    xlam_peak_min_prominence: float = 0.1,
    recenter_v: bool = True,
    xlam_cont: float = DEFAULT_XLAM_CONT, cont_diff_order: int = 2,
    maxiter: int = 50000, ftol: float = 1e-12, maxfun: int = 500000,
    use_jax: bool = True,
    fit_continuum: bool = True,
    xlam_criterion: str = "chi2",
    xlam_discrepancy_nsigma: float = 1.0,
    w_bounds: tuple = (1e-5, 1.0),
):
    """Self-consistent fixed-point refinement of the wing-taper's ``sigl0``.

    This is the default top-level joint-fit entry point (see
    :func:`~kinextract.fitting.run_spectral_fit` with ``cfg.fit_continuum=True``).

    Rationale: a dedicated diagnostic found the sigma-recovery bias is a
    monotonic function of ``sigl0`` crossing zero almost exactly at
    ``sigl0 == true sigma``. That means the map ``sigl0 -> recovered
    sigma(sigl0)`` has a fixed point very close to the true sigma, and
    iterating -- fit once, read off the recovered sigma, use it as the next
    ``sigl0``, refit -- converges toward that fixed point regardless of how
    wrong the starting guess is. This replaces an earlier cross-correlation-
    based sigma pre-estimate that went through three failed implementation
    attempts, each validated cleanly on a single-template synthetic test but
    broken differently on the real, 35-template MUSE library (individual
    templates' own autocorrelation widths there span 43-385 km/s, so no
    single "reference template" construction generalized). Each iteration is
    a full :func:`fit_joint_auto_xlam` call (its own xlam grid search + wing-
    taper recentering), so this costs ``n_sigl0_iter`` times as much as a
    single auto-xlam fit.

    A validation sweep (sigl0_init from 20 to 500 km/s against a synthetic
    true sigma of 140) found one correction (2 total fits) already erases
    the overwhelming majority of the bias from a badly wrong initial guess
    (e.g. -37.45 -> -1.92 km/s for a 7x-too-small initial guess); a second
    correction (3 total fits, the default) tightens convergence further so
    that *every* tested starting point lands within ~0.3 km/s of the others,
    confirming genuine fixed-point convergence independent of the initial
    guess. On a real MUSE bin (``bin1605sp``), starting guesses of 20, 100,
    and 300 km/s all converged to sigma ~51.5-51.6 km/s within 2-3
    iterations, confirming the same robust convergence on real data as in
    the synthetic sweep above (an initial comparison against a ``pallmc.out``
    entry suggested an ~11 km/s residual here, but that entry turned out to
    be a different, mismatched bin -- see this module's docstring; the
    shipped ALS pipeline run on the actual ``bin1605sp.spec`` agrees with
    this result).

    Parameters
    ----------
    st : FitState
        Fit state to optimize (not modified).
    sigl0_init : float, optional
        Starting ``sigl0`` guess for iteration 1. Default 100 km/s,
        matching both this module's own and the original Fortran
        framework's fixed default (confirmed directly in the legacy
        pipeline's batch wrapper script: every bin uses the same hardcoded
        initial sigma guess regardless of radius or true dispersion).
    n_sigl0_iter : int, optional
        Number of fit-then-update rounds. Default 3 (two corrections after
        the initial guess). Reduce to 2 (or 1, to disable correction
        entirely) if runtime on a full batch matters more than the small
        extra convergence margin the third fit buys.
    sigl0_tol : float, optional
        Stop early (reuse the current iteration's result) if
        ``|sigl0 - recovered_sigma| <= sigl0_tol`` km/s, i.e. the fixed
        point has already been reached to within this tolerance.
    xlam_grid, xlam_chi2_tolerance, xlam_max_peaks, xlam_peak_min_prominence,
    recenter_v, xlam_cont, cont_diff_order, maxiter, ftol, maxfun, use_jax,
    xlam_criterion, xlam_discrepancy_nsigma :
        Passed through to each :func:`fit_joint_auto_xlam` call.
    fit_continuum : bool, optional
        Passed through to each :func:`fit_joint_auto_xlam` call -- see its
        docstring. Set False for pre-normalized input (no continuum to
        co-fit); ``sigl0`` fixed-point convergence still applies exactly
        the same way.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The best round's fit (physical units) -- see the divergence-guard
        note below, not necessarily the final round's.
    design : ndarray
        Continuum B-spline design matrix for the returned fit.
    best_xlam : float
        The returned fit's selected xlam.
    sigl0_trace : list of float
        ``[sigl0_init, sigl0_after_iter_1, sigl0_after_iter_2, ...]`` --
        the sequence of sigl0 values used/produced, for diagnosing
        convergence.

    Notes
    -----
    **Divergence guard.** This iteration is not guaranteed to converge
    monotonically on every chi2(xlam) curve: on an unusually flat one
    (confirmed on a clean, oversimplified synthetic mock -- an
    under-constrained problem for regularization selection in general, see
    ``examples/notebooks/01_basic_mock_fit.ipynb`` and
    :func:`kinextract.fitting._fit_map_sigl0_recenter`, the shipped-path
    analogue of this function), each round's recovered sigma can land
    *farther* from that round's ``sigl0`` input than the previous round
    did -- a positive feedback loop, not a fixed point. This function
    tracks the smallest ``|sigl0 - recovered_sigma|`` gap seen across all
    rounds and returns that round's fit instead of the final round's if
    the final round's gap is worse.
    """
    sigl0 = float(sigl0_init)
    sigl0_trace = [sigl0]
    result = design = best_xlam = None
    best = None  # (gap, result, design, best_xlam, sigl0_input)
    last_gap = None

    for _ in range(max(1, n_sigl0_iter)):
        sigl0_in = sigl0
        st_try = dataclasses.replace(st, sigl0=sigl0)
        result, design, best_xlam, _records = fit_joint_auto_xlam(
            st_try, n_interior_knots, degree, xlam_grid=xlam_grid,
            xlam_chi2_tolerance=xlam_chi2_tolerance, xlam_max_peaks=xlam_max_peaks,
            xlam_peak_min_prominence=xlam_peak_min_prominence,
            recenter_v=recenter_v,
            xlam_cont=xlam_cont, cont_diff_order=cont_diff_order,
            maxiter=maxiter, ftol=ftol, maxfun=maxfun, use_jax=use_jax,
            fit_continuum=fit_continuum,
            xlam_criterion=xlam_criterion, xlam_discrepancy_nsigma=xlam_discrepancy_nsigma,
            w_bounds=w_bounds,
        )
        t_norm, _ = normalize_template_matrix(st_try)
        _, b, *_ = evaluate_model_gp_joint(result.x, st_try, design, t_norm)
        gh = fit_losvd_gauss_hermite(st_try.xl, b, fit_h3h4=True)
        recovered_sigma = float(gh["sherm"])
        sigl0_trace.append(recovered_sigma)

        gap = abs(sigl0_in - recovered_sigma)
        if best is None or gap < best[0]:
            best = (gap, result, design, best_xlam, sigl0_in)
        last_gap = gap

        if gap <= sigl0_tol:
            break
        sigl0 = recovered_sigma

    if best is not None and last_gap is not None and best[0] < last_gap:
        gap_best, result, design, best_xlam, sigl0_best = best
        log(
            f"joint sigl0 fixed-point iteration diverged (final gap "
            f"{last_gap:.2f} > best gap {gap_best:.2f} km/s); reverting to "
            f"the round with the smallest |sigl0 - recovered_sigma| "
            f"(sigl0={sigl0_best:.2f}, xlam={best_xlam:.4g})"
        )

    return result, design, best_xlam, sigl0_trace


def run_joint_fit(st: FitState, cfg, write_outputs: bool = False, outdir: str = ".", prefix: Optional[str] = None) -> dict:
    """Run the joint fit and package results in the same shape as
    :func:`kinextract.fitting.run_spectral_fit`'s ``"outputs"`` dict.

    This is the entry point :func:`kinextract.fitting.run_spectral_fit`
    dispatches to whenever ``cfg.fit_continuum=True`` or ``cfg.joint_prenorm=True``.
    Whether a continuum is actually co-fit is controlled by
    ``cfg.fit_continuum``: if True, a free P-spline continuum is co-fit; if
    False (the pre-normalized-mode convention), the continuum is fixed at 1.0 (see
    :func:`fit_joint`'s `fit_continuum` parameter) -- there is nothing
    genuine for a free continuum to fit once the data's own continuum has
    already been divided out, and the sigl0/v_center/xlam-selection
    improvements apply identically either way. It reuses the corresponding
    ``FitConfig`` fields with equivalent semantics (``cfg.sigl`` as the
    initial sigl0 guess, ``cfg.xlam_auto_grid``/``xlam_chi2_tolerance``/
    ``xlam_max_peaks``/``xlam_peak_min_prominence`` for the xlam search,
    ``cfg.use_jax_objective``, ``cfg.map_maxiter``/``map_ftol``/
    ``map_maxfun``) plus the ``joint_*``-prefixed fields for concepts that
    have no shipped-pipeline analogue (P-spline knot count/degree/roughness
    weight, the sigl0 fixed-point iteration count/tolerance, whether to
    recenter ``v_center``).

    Parameters
    ----------
    st : FitState
        Fit state to optimize (mutated: ``continuum_mult`` is set to the
        recovered P-spline continuum so downstream plotting/ascii output
        reflects it).
    cfg : FitConfig
        Run configuration.
    write_outputs : bool, optional
        If True, write legacy-format output files via
        :func:`kinextract.io.write_fitlov_outputs_from_model`.
    outdir, prefix : optional
        Passed through when `write_outputs` is True.

    Returns
    -------
    dict
        Same keys as the shipped pipeline's ``outputs`` dict (``"gp"``,
        ``"b"``, ``"w"``, ``"wfrac"``, ``"tt"``, ``"coff"``, ``"coff2"``,
        ``"A"``, ``"rms"``, ``"chi2_red"``, ``"chi2_total"``, ``"nrms"``,
        ``"continuum"``, ``"paths"``), with ``"coff"=0.0``, ``"coff2"=0.0``,
        ``"A"=1.0`` as neutral placeholders (the joint model has no
        analogous parameters -- its continuum is the P-spline baked
        directly into ``"gp"``/``"continuum"``), plus ``"result"``,
        ``"design"``, ``"best_xlam"``, and ``"sigl0_trace"`` for the joint
        fit's own diagnostics.
    """
    from .numerics import compute_weighted_template_spectrum

    # cfg.xlam_auto=False means "use cfg.xlam as a single fixed regularization
    # strength, no grid search" -- matching the shipped path's own convention
    # (_auto_select_xlam is simply skipped there). fit_joint_auto_xlam_sigl0
    # has no separate "no search" mode of its own, but a single-element xlam
    # grid is exactly equivalent: the chi2-tolerance selection logic still
    # runs, but with only one candidate to "select." The sigl0 fixed-point
    # iteration and v_center recentering are independent improvements over
    # the legacy pipeline (not part of what xlam_auto governs) and stay
    # active either way.
    xlam_grid = cfg.xlam_auto_grid if cfg.xlam_auto else (float(cfg.xlam),)

    result, design, best_xlam, sigl0_trace = fit_joint_auto_xlam_sigl0(
        st,
        n_interior_knots=cfg.joint_n_interior_knots, degree=cfg.joint_degree,
        sigl0_init=cfg.sigl, n_sigl0_iter=cfg.joint_n_sigl0_iter, sigl0_tol=cfg.joint_sigl0_tol,
        xlam_grid=xlam_grid, xlam_chi2_tolerance=cfg.xlam_chi2_tolerance,
        xlam_max_peaks=cfg.xlam_max_peaks, xlam_peak_min_prominence=cfg.xlam_peak_min_prominence,
        recenter_v=cfg.joint_recenter_v,
        xlam_cont=cfg.joint_xlam_cont, cont_diff_order=cfg.joint_cont_diff_order,
        maxiter=cfg.map_maxiter, ftol=cfg.map_ftol, maxfun=cfg.map_maxfun,
        use_jax=cfg.use_jax_objective,
        fit_continuum=cfg.fit_continuum,
        xlam_criterion=getattr(cfg, "xlam_criterion", "discrepancy"),
        xlam_discrepancy_nsigma=getattr(cfg, "xlam_discrepancy_nsigma", 0.3),
        w_bounds=cfg.template_w_bounds if getattr(cfg, "template_w_bounds", None) is not None else (1e-5, 1.0),
    )

    t_norm, _ = normalize_template_matrix(st)
    gp, b, w, cont_coeffs, cont = evaluate_model_gp_joint(result.x, st, design, t_norm)
    st.continuum_mult = cont
    st.xlam = best_xlam
    # Freeze the converged sigl0/v_center onto st too (mirroring st.xlam
    # above) -- estimate_velocity_xcorr is a deterministic, pure function of
    # st.g/st.t, so recomputing it here reproduces exactly what the last
    # sigl0-iteration used internally, without needing fit_joint_auto_xlam_sigl0
    # to thread it back out as an extra return value. Needed so a frozen,
    # single-shot fit_joint(st, ...) call later (e.g. a bootstrap replicate
    # refit) reproduces this fit's regularization pivot instead of silently
    # reverting to the pre-fit defaults (sigl0=cfg.sigl, v_center=0.0).
    st.sigl0 = float(sigl0_trace[-1])
    st.v_center = estimate_velocity_xcorr(st) if cfg.joint_recenter_v else 0.0

    sw = float(np.sum(w))
    wfrac = w / sw if sw else w
    tt = compute_weighted_template_spectrum(st, w)

    good = (st.gerr > 0.0) & (st.gerr < 1.0e9) & np.isfinite(st.g) & np.isfinite(gp)
    sl = slice(st.iskip, st.npix - st.iskip)
    gfr = good[sl]
    rms = float(np.sqrt(np.mean((st.g[sl] - gp[sl])[gfr] ** 2))) if gfr.any() else np.nan
    nrms = int(gfr.sum())
    chi2_total = (
        float(np.sum((st.g[sl][gfr] - gp[sl][gfr]) ** 2 / st.gerr[sl][gfr] ** 2))
        if gfr.any() else 0.0
    )
    chi2_red = chi2_total / max(nrms - 1, 1)

    if write_outputs:
        written = write_fitlov_outputs_from_model(st, gp, b, w, 0.0, 0.0, 1.0, outdir, prefix)
        paths = written["paths"]
    else:
        paths = {}

    return {
        "gp": gp, "b": b, "w": w, "wfrac": wfrac, "tt": tt,
        "coff": 0.0, "coff2": 0.0, "A": 1.0,
        "rms": rms, "chi2_red": chi2_red, "chi2_total": chi2_total, "nrms": nrms,
        "continuum": cont, "paths": paths,
        "result": result, "design": design, "best_xlam": best_xlam, "sigl0_trace": sigl0_trace,
    }
