"""Fitting drivers that orchestrate the MAP LOSVD/template optimisation.

This module wraps the core objective/model functions in :mod:`kinextract.numerics`
into the higher-level machinery needed for a real spectral fit: building
parameter scaling for the optimizer (:func:`build_parameter_xscale`), running
a single bound-constrained L-BFGS-B optimisation (:func:`_fit_map_once`,
using either scipy finite-difference gradients or exact JAX-autodiff
gradients), iterative sigma-clipping of outlier pixels with protected windows
for real spectral features such as the Ca II triplet
(:func:`run_iterative_clean_map`), LOSVD shape diagnostics used to judge
whether a fit is over- or under-regularised (:func:`compute_losvd_roughness`,
:func:`compute_losvd_n_peaks`), an automatic regularisation-strength
(``xlam``) grid search (:func:`_auto_select_xlam`), and the sigma-clipped
MAP fit itself (:func:`fit_state_map_with_optional_clean`), used for
pre-normalized-mode fits (continuum-cofit fits are dispatched to
:mod:`kinextract.joint` instead). The top-level entry point
:func:`run_spectral_fit` ties all of this together into the primary public
API for fitting a single galaxy spectrum end-to-end, given a
:class:`~kinextract.config.FitConfig`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ._utils import BIG, Timer, log
from .config import FitConfig
from .io import infer_output_prefix, write_fitlov_outputs

# fit_losvd_gauss_hermite is not used inside this module, but is re-exported
# here because several example notebooks/scripts import it via
# `from kinextract.fitting import fit_losvd_gauss_hermite`; the canonical
# definition is in .losvd. The noqa keeps ruff/pyflakes from treating this
# as an unused import and silently deleting it on the next `ruff --fix`.
from .losvd import fit_losvd_gauss_hermite  # noqa: F401
from .masking import _bloom_rejected, _update_clean_mask, build_clean_protect_mask
from .numerics import (
    _get_or_build_jax_vg,
    estimate_velocity_xcorr,
    evaluate_model_gp,
    jax,  # module-level jax reference (may be None)
    objective_map,
)
from .spectrum import build_initial_guess_nonparam, make_fit_state
from .state import FitState

# =============================================================================
# Section 9 fitting functions
# =============================================================================

def build_parameter_xscale(st: FitState) -> np.ndarray:
    """Build a per-parameter scale vector to precondition the L-BFGS-B fit.

    L-BFGS-B (like most quasi-Newton methods) performs best when all
    parameters have comparable magnitude and sensitivity. The fit vector
    mixes LOSVD bin heights, template weights, continuum offsets, an
    optional global amplitude, and an optional polynomial coefficient,
    which naturally live on very different scales; this function returns a
    divisor for each parameter so that the optimizer works internally in
    rescaled units ``u = a / xscale`` (see :func:`_fit_map_once`), improving
    convergence without changing the physical solution.

    Parameters
    ----------
    st : FitState
        Fit state describing the parameter layout: number of LOSVD bins
        (``nl``), number of templates (``nt``), continuum-offset mode
        (``icoff``), and whether a global amplitude and/or polynomial
        continuum term are being fit.

    Returns
    -------
    ndarray
        Scale vector with the same length and ordering as the flat
        parameter vector `a` used throughout this module: ``nl`` LOSVD
        scales, ``nt`` template-weight scales, continuum-offset scales,
        an optional amplitude scale, and an optional polynomial-coefficient
        scale.
    """
    parts = [np.ones(st.nl), np.full(st.nt, 0.1)]
    if st.icoff == 1:
        parts.append(np.array([0.2, 0.2]))
    elif st.icoff == 2:
        parts.append(np.array([0.2]))
    if st.fit_global_amp:
        parts.append(np.array([1.0]))
    if getattr(st, "continuum_poly_mode", "none") != "none":
        bound = float(getattr(st, "continuum_poly_bound", 0.1))
        parts.append(np.array([max(bound, 1e-6)]))
    return np.concatenate(parts)


def _fit_map_once(
    st: FitState, a0: np.ndarray, bounds: list,
    map_maxiter: int, map_ftol: float, map_maxfun: int,
    print_every: int, label: str,
    map_gtol: float = 1e-10, map_maxls: int = 50,
    use_scaled_optimizer: bool = True,
    use_jax_objective: bool = False,
    jax_enable_x64: bool = True,
):
    """Run a single bound-constrained L-BFGS-B MAP optimisation.

    Minimises :func:`kinextract.numerics.objective_map` (chi2 + wing-tapered
    smoothness penalty + LOSVD normalization penalty) over the flat
    parameter vector, starting from `a0` and respecting box constraints
    `bounds`. This is the single-shot optimisation call used as a building
    block by :func:`run_iterative_clean_map` (once per sigma-clip
    iteration) and :func:`_auto_select_xlam` (once per candidate ``xlam``);
    the top-level looping/cleaning logic lives in those callers, not here.

    Parameters
    ----------
    st : FitState
        Fit state providing the data, template matrix, and regularisation
        settings used to evaluate the objective. Its ``xlam`` at call time
        determines the regularisation strength used for this fit.
    a0 : ndarray
        Initial guess for the flat parameter vector.
    bounds : list of tuple
        Per-parameter ``(lower, upper)`` bounds, same length/order as `a0`.
    map_maxiter : int
        Maximum number of L-BFGS-B iterations.
    map_ftol : float
        Relative function-value convergence tolerance passed to
        ``scipy.optimize.minimize`` (``options["ftol"]``).
    map_maxfun : int
        Maximum number of objective function evaluations.
    print_every : int
        Log a progress line every time the objective evaluation counter
        (``st.ntot``) advances by at least this many evaluations. Set to 0
        (falsy) to disable progress logging.
    label : str
        Label used in the timing log line (see :class:`~kinextract._utils.Timer`)
        to identify this optimisation call in the log output.
    map_gtol : float, optional
        Gradient-norm convergence tolerance passed to L-BFGS-B.
    map_maxls : int, optional
        Maximum number of line-search steps per L-BFGS-B iteration.
    use_scaled_optimizer : bool, optional
        If True (default), optimise in parameter units rescaled by
        :func:`build_parameter_xscale` for better-conditioned convergence,
        then rescale the solution back to physical units before returning.
    use_jax_objective : bool, optional
        If True, use the JAX-autodiff value-and-gradient function from
        :func:`kinextract.numerics._get_or_build_jax_vg` (exact analytic
        gradients) instead of scipy's finite-difference approximation of
        the NumPy/Numba objective. Falls back to the finite-difference path
        with a logged warning if JAX is unavailable or if
        ``continuum_poly_mode`` is enabled (unsupported by the JAX
        objective). This replaces the legacy Fortran solver's own
        finite-difference bound-constrained quasi-Newton scheme with exact
        gradients, giving a large speedup.
    jax_enable_x64 : bool, optional
        If True (default) and the JAX objective is used, enable JAX's
        float64 mode (``jax_enable_x64``) before building/using the
        compiled kernel, since JAX defaults to float32.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result of ``scipy.optimize.minimize`` with ``method="L-BFGS-B"``,
        with ``res.x`` rescaled back to physical parameter units if
        `use_scaled_optimizer` was True.

    Notes
    -----
    When `use_jax_objective` is requested and usable, the compiled
    value-and-gradient kernel is fetched from the process-wide cache
    (:func:`kinextract.numerics._get_or_build_jax_vg`) rather than rebuilt,
    so repeated calls for fits of the same problem shape (e.g. successive
    sigma-clip iterations or bootstrap replicates) reuse the same XLA
    compilation. A small local cache also avoids recomputing the
    value-and-gradient pair when the optimizer requests the function value
    and Jacobian at the same point in consecutive calls.
    """
    last = {"n": 0}

    def obj_phys(a):
        val = objective_map(a, st)
        if print_every and st.ntot - last["n"] >= print_every:
            last["n"] = st.ntot
            log(f"MAP ntot={st.ntot} obj={val:.8g}")
        return val

    opts = dict(
        maxiter=map_maxiter, ftol=map_ftol, gtol=map_gtol,
        maxfun=map_maxfun, maxls=map_maxls,
    )

    jax_vg = None
    if use_jax_objective:
        if jax is None:
            log("JAX objective requested but JAX is unavailable; falling back to NumPy/Numba objective")
        elif getattr(st, "continuum_poly_mode", "none") != "none":
            log("JAX objective does not support continuum_poly_mode; falling back to NumPy/Numba objective")
        else:
            if jax_enable_x64:
                try:
                    jax.config.update("jax_enable_x64", True)
                except Exception:
                    pass
            # Use the process-wide cache so the compiled XLA kernel is shared
            # across ALL callers: MAP outer iterations, bootstrap threads, etc.
            # _get_or_build_jax_vg is thread-safe (double-checked locking).
            jax_vg = _get_or_build_jax_vg(st)

    def _make_cached_fun_and_jac(vg_fn):
        cache = {"x": None, "val": None, "grad": None}

        def _eval(x):
            x_arr = np.asarray(x, float)
            if cache["x"] is None or not np.array_equal(x_arr, cache["x"]):
                val, grad = vg_fn(x_arr)
                cache["x"] = x_arr.copy()
                cache["val"] = float(val)
                cache["grad"] = np.asarray(grad, float)
            return cache["val"], cache["grad"]

        def _fun(x):
            val, _ = _eval(x)
            if print_every and st.ntot - last["n"] >= print_every:
                last["n"] = st.ntot
                log(f"MAP ntot={st.ntot} obj={val:.8g}")
            return val

        def _jac(x):
            _, grad = _eval(x)
            return grad

        return _fun, _jac

    with Timer(label):
        if jax_vg is not None and not use_scaled_optimizer:
            def _vg_phys(a):
                st.ntot += 1
                _g = np.asarray(st.g, float)
                _cont = np.asarray(
                    getattr(st, "continuum_mult", np.ones(st.npix)), float
                )
                return jax_vg(a, _g, np.asarray(st.gerr, float), _cont)

            fun, jac = _make_cached_fun_and_jac(_vg_phys)
            return minimize(fun, a0, jac=jac, method="L-BFGS-B", bounds=bounds, options=opts)

        if not use_scaled_optimizer:
            return minimize(obj_phys, a0, method="L-BFGS-B", bounds=bounds, options=opts)

        a0 = np.asarray(a0, float)
        xscale = build_parameter_xscale(st)
        if len(xscale) != len(a0):
            raise ValueError(
                f"xscale length {len(xscale)} != parameter length {len(a0)}"
            )
        u0 = a0 / xscale
        ubounds = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, xscale)]

        if jax_vg is not None:
            def _vg_scaled(u):
                st.ntot += 1
                a = np.asarray(u) * xscale
                _g = np.asarray(st.g, float)
                _cont = np.asarray(
                    getattr(st, "continuum_mult", np.ones(st.npix)), float
                )
                val, grad_a = jax_vg(a, _g, np.asarray(st.gerr, float), _cont)
                grad_u = np.asarray(grad_a) * xscale
                return val, grad_u

            fun, jac = _make_cached_fun_and_jac(_vg_scaled)
            res = minimize(fun, u0, jac=jac, method="L-BFGS-B", bounds=ubounds, options=opts)
            res.x = np.asarray(res.x) * xscale
            return res

        def obj_scaled(u):
            return obj_phys(np.asarray(u) * xscale)

        res = minimize(obj_scaled, u0, method="L-BFGS-B", bounds=ubounds, options=opts)
        res.x = np.asarray(res.x) * xscale
        return res


def run_iterative_clean_map(
    st: FitState, a0: np.ndarray, bounds: list,
    map_maxiter: int, map_ftol: float, map_maxfun: int, print_every: int,
    sigma_clip: float = 3.0, clean_maxiter: int = 5, clean_minpix: int = 10,
    protect_mask: Optional[np.ndarray] = None, protect_absorption_only: bool = True,
    bloom_pixels: int = 0,
    map_gtol: float = 1e-10, map_maxls: int = 50, use_scaled_optimizer: bool = True,
    use_jax_objective: bool = False, jax_enable_x64: bool = True,
):
    """Iteratively sigma-clip outlier pixels while refitting the MAP model.

    Alternates fitting the LOSVD/template model (:func:`_fit_map_once`) with
    identifying and masking outlier pixels (via
    :func:`kinextract.masking._update_clean_mask`) whose residuals exceed
    `sigma_clip`, until the set of good pixels stops changing (convergence),
    a maximum number of iterations is reached, or too few good pixels would
    remain. This "cleaning" process removes cosmic rays, sky-line residuals,
    bad columns, and other artifacts that would otherwise bias the LOSVD
    recovery, while `protect_mask` prevents known genuine spectral features
    (e.g. the Ca II triplet) from ever being clipped even if they are
    poorly fit.

    Parameters
    ----------
    st : FitState
        Fit state; ``st.gerr`` is temporarily modified pixel-by-pixel during
        cleaning (masked pixels get their error set to the large sentinel
        value ``BIG``) and is left consistent with the final good-pixel set
        on return.
    a0 : ndarray
        Initial guess for the flat parameter vector, used to start the
        first clean iteration.
    bounds : list of tuple
        Per-parameter ``(lower, upper)`` bounds passed through to
        :func:`_fit_map_once`.
    map_maxiter, map_ftol, map_maxfun, print_every
        Passed through to each call of :func:`_fit_map_once`; see that
        function for descriptions.
    sigma_clip : float, optional
        Number of standard deviations beyond which a pixel's residual is
        considered an outlier and masked on the next iteration.
    clean_maxiter : int, optional
        Maximum number of clean/refit iterations.
    clean_minpix : int, optional
        Minimum number of good pixels required; cleaning stops (with a
        warning logged) rather than masking below this floor.
    protect_mask : ndarray of bool, optional
        Boolean mask, length ``npix``, marking pixels that must never be
        clipped regardless of residual (e.g. a window around the Ca II
        triplet). If None, no pixels are protected.
    protect_absorption_only : bool, optional
        If True, the protection in `protect_mask` only prevents clipping of
        pixels where the data lies *below* the model (absorption-like
        residuals), still allowing clipping of emission-like outliers
        within the protected window.
    bloom_pixels : int, optional
        If > 0, grow (dilate) each newly-rejected pixel run by this many
        pixels on each side before finalising the mask for the next
        iteration, so that the wings of a bad feature are also excluded,
        not just its most deviant pixels. Growth is not applied within
        `protect_mask`.
    map_gtol, map_maxls, use_scaled_optimizer, use_jax_objective, jax_enable_x64
        Passed through to each call of :func:`_fit_map_once`; see that
        function for descriptions.

    Returns
    -------
    best_res : scipy.optimize.OptimizeResult
        The optimisation result from the final clean iteration.
    good_mask : ndarray of bool
        Final good-pixel mask, length ``npix``, after all clipping.
    """
    base_gerr = st.gerr.copy()
    good_mask = base_gerr < BIG
    protect = (
        np.asarray(protect_mask, bool) if protect_mask is not None
        else np.zeros(st.npix, bool)
    ) & good_mask
    best_res = None
    a_start = np.asarray(a0, float).copy()
    for it in range(clean_maxiter):
        st.gerr = np.where(good_mask, base_gerr, BIG)
        best_res = _fit_map_once(
            st, a_start, bounds,
            map_maxiter, map_ftol, map_maxfun, print_every,
            f"MAP clean iter {it + 1}",
            map_gtol, map_maxls, use_scaled_optimizer,
            use_jax_objective, jax_enable_x64,
        )
        a_start = np.asarray(best_res.x, float).copy()
        new_mask, sigma = _update_clean_mask(
            st, best_res.x, base_gerr, good_mask,
            sigma_clip, protect, protect_absorption_only,
        )
        if bloom_pixels > 0:
            newly_rejected = good_mask & ~new_mask
            if newly_rejected.any():
                bloomed = _bloom_rejected(newly_rejected, bloom_pixels)
                extra = bloomed & ~newly_rejected & ~protect
                if extra.any():
                    new_mask &= ~extra
        if np.array_equal(new_mask, good_mask):
            log(f"Clean converged after {it + 1} iter.")
            break
        if new_mask.sum() < clean_minpix:
            log(f"WARNING: clean would leave {new_mask.sum()} pixels; stopping.")
            break
        log(
            f"Clean iter {it + 1}: masked "
            f"{good_mask.sum() - new_mask.sum()} pixels "
            f"(sigma={sigma:.3g})"
        )
        good_mask = new_mask
    st.gerr = np.where(good_mask, base_gerr, BIG)
    return best_res, good_mask


def _chi2_stats(st: FitState, a_best: np.ndarray) -> tuple[float, int]:
    """Compute total chi2 and the number of good pixels for a fit result.

    Evaluates the forward model at `a_best`, then sums squared
    error-normalised residuals over the interior region (``iskip`` pixels
    trimmed from each end) restricted to pixels with finite, positive,
    non-sentinel (``< 1e9``) errors. Used to report final fit quality and
    to compute the reduced chi2 (``chi2 / ngood``) driving the ``xlam``
    selection logic in :func:`_auto_select_xlam`.

    Parameters
    ----------
    st : FitState
        Fit state providing the data, errors, and pixel-skip settings.
    a_best : ndarray
        Flat parameter vector at which to evaluate the model (typically the
        optimizer's best-fit solution).

    Returns
    -------
    chi2 : float
        Total chi-squared over good pixels; ``inf`` if no pixels are good.
    ngood : int
        Number of pixels included in the sum.
    """
    gp, *_ = evaluate_model_gp(a_best, st)
    return _chi2_stats_from_gp(st, gp)


def _chi2_stats_from_gp(st: FitState, gp: np.ndarray) -> tuple[float, int]:
    """Same computation as :func:`_chi2_stats`, given an already-evaluated
    model spectrum `gp` instead of a flat parameter vector.

    Factored out so callers whose parameter-vector layout ``evaluate_model_gp``
    cannot unpack (e.g. the joint continuum-in-the-model fit in
    :mod:`kinextract.joint`, which already evaluates its own forward model)
    can still get the same chi2/ngood definition without re-deriving it.
    """
    sl = slice(st.iskip, st.npix - st.iskip)

    good = (
        (st.gerr[sl] > 0.0)
        & (st.gerr[sl] < 1e9)
        & np.isfinite(st.g[sl])
        & np.isfinite(gp[sl])
        & np.isfinite(st.gerr[sl])
    )

    if not np.any(good):
        return np.inf, 0

    resid = (st.g[sl][good] - gp[sl][good]) / st.gerr[sl][good]
    return float(np.sum(resid ** 2)), int(good.sum())


def compute_losvd_roughness(b: np.ndarray) -> float:
    """Peak-normalised maximum absolute second difference of a LOSVD array.

    Measures local curvature: smooth LOSVDs (e.g. a Gaussian with sigma ~ 3
    LOSVD bins) have roughness ~ 0.10; jagged noise spikes produce values
    > 0.20. Used by :func:`_auto_select_xlam` (and logged as a diagnostic
    at every point of the ``xlam`` grid search) to judge whether the
    current regularisation strength is sufficient to suppress noise-driven
    wiggles in the recovered LOSVD.

    Parameters
    ----------
    b : ndarray
        LOSVD histogram to assess (negative values are clipped to zero
        before computing curvature).

    Returns
    -------
    float
        Roughness statistic: the maximum absolute second difference of the
        (clipped) LOSVD, divided by its peak value. Returns 0.0 if `b` has
        fewer than 3 elements, and ``inf`` if the peak is essentially zero
        (a degenerate LOSVD is treated as maximally rough so that
        ``xlam`` selection pushes toward stronger regularisation).
    """
    b_pos = np.maximum(np.asarray(b, float), 0.0)
    peak = float(np.max(b_pos))
    if len(b) < 3:
        return 0.0
    if peak < 1e-4:
        return np.inf  # degenerate LOSVD; treat as maximally rough so xlam increases
    return float(np.max(np.abs(np.diff(np.diff(b_pos))))) / peak


def compute_losvd_n_peaks(b: np.ndarray, min_prominence: float = 0.1) -> int:
    """Count topographically-prominent peaks in a LOSVD.

    Counts local maxima of the LOSVD whose *prominence* — height above the
    deepest valley ("key col") connecting the peak to any taller
    neighbouring peak, normalised by the global peak height — exceeds
    `min_prominence`. This correctly ignores wings and shoulders: features
    with only a shallow valley separating them from the main peak have low
    prominence and are not counted, while only genuine secondary peaks
    separated by a deep valley (e.g. a bimodal LOSVD produced by noise
    overfitting in the inner bins) exceed the threshold. Used by
    :func:`_auto_select_xlam` as a unimodality constraint: candidate `xlam`
    values that produce a multi-peaked (non-unimodal) LOSVD are rejected
    even if their chi2 is competitive.

    Parameters
    ----------
    b : ndarray
        LOSVD histogram to assess (negative values are clipped to zero).
    min_prominence : float, optional
        Minimum prominence, as a fraction of the global peak height,
        required for a local maximum to be counted as a distinct peak.

    Returns
    -------
    int
        Number of peaks meeting the prominence threshold; always at least
        1 (a LOSVD with no qualifying local maxima, or fewer than 3
        samples, or a non-positive peak, is counted as having a single
        peak).
    """
    b_pos = np.maximum(np.asarray(b, float), 0.0)
    peak = float(np.max(b_pos))
    if peak <= 0.0 or len(b_pos) < 3:
        return 1

    prom_threshold = min_prominence * peak

    # Find strict local maxima
    local_max = [i for i in range(1, len(b_pos) - 1)
                 if b_pos[i] > b_pos[i - 1] and b_pos[i] > b_pos[i + 1]]
    if not local_max:
        return 1

    def _key_col(idx: int, direction: str) -> float:
        """Min value between idx and the first sample taller than b_pos[idx]."""
        h = b_pos[idx]
        indices = range(idx - 1, -1, -1) if direction == "left" else range(idx + 1, len(b_pos))
        running_min = np.inf
        for j in indices:
            if b_pos[j] > h:
                return float(running_min) if np.isfinite(running_min) else 0.0
            running_min = min(running_min, float(b_pos[j]))
        return 0.0  # no taller point on this side; LOSVD boundary = 0

    n_peaks = 0
    for idx in local_max:
        key_col = max(_key_col(idx, "left"), _key_col(idx, "right"))
        if float(b_pos[idx]) - key_col >= prom_threshold:
            n_peaks += 1
    return max(n_peaks, 1)


def _auto_select_xlam(
    st: FitState,
    cfg: FitConfig,
    a0: np.ndarray,
    bounds: list,
    xlam_grid: tuple,
    smooth_threshold: float,
    map_kwargs: dict,
) -> float:
    """Automatically select the LOSVD regularisation strength ``xlam``.

    Runs a full MAP fit (:func:`_fit_map_once`) at each candidate value in
    `xlam_grid`, then picks the best value according to
    ``cfg.xlam_criterion``, storing the result in both ``st.xlam`` and
    ``cfg.xlam`` before returning. This automates a choice that otherwise
    requires manual per-galaxy tuning of the smoothness-penalty strength in
    the objective (see :mod:`kinextract.numerics` module docstring).

    Two selection criteria are supported:

    ``criterion="chi2"`` (default)
        Runs all grid fits, collects the reduced chi2 (via
        :func:`_chi2_stats`) at each point, then — among grid points that
        satisfy the unimodality constraint (``n_peaks <= xlam_max_peaks``,
        via :func:`compute_losvd_n_peaks`) — picks the *largest* ``xlam``
        (i.e. the most regularised, smoothest LOSVD) whose reduced chi2 is
        within ``cfg.xlam_chi2_tolerance`` of the grid minimum. This is
        scale-invariant and needs no per-galaxy tuning.
    ``criterion="roughness"`` (kept for backward compatibility)
        Iterates the grid from smallest to largest ``xlam`` and stops at
        the first value whose LOSVD roughness (via
        :func:`compute_losvd_roughness`) is at or below `smooth_threshold`
        AND satisfies the unimodality constraint. Less well calibrated for
        broad LOSVDs (large sigma).

    Both criteria always run and log the fit at every grid point (chi2,
    roughness, and peak count), producing a full diagnostic table in the
    log regardless of which criterion ultimately selects the answer.

    Parameters
    ----------
    st : FitState
        Fit state whose ``xlam`` is overwritten with each grid candidate
        during the search and set to the final selected value on return.
    cfg : FitConfig
        Run configuration; ``cfg.xlam_criterion``, ``cfg.xlam_max_peaks``,
        ``cfg.xlam_peak_min_prominence``, ``cfg.xlam_chi2_tolerance``, and
        ``cfg.xlam_auto_maxiter`` control the search. ``cfg.xlam`` is
        overwritten with the selected value on return.
    a0 : ndarray
        Initial guess for the flat parameter vector, used to start every
        grid-point fit.
    bounds : list of tuple
        Per-parameter ``(lower, upper)`` bounds passed to
        :func:`_fit_map_once`.
    xlam_grid : tuple of float
        Candidate ``xlam`` values to try (sorted ascending internally).
    smooth_threshold : float
        Roughness threshold used only when ``cfg.xlam_criterion ==
        "roughness"``.
    map_kwargs : dict
        Extra keyword arguments (e.g. ``map_gtol``, ``use_jax_objective``)
        forwarded to every :func:`_fit_map_once` call in the grid search.

    Returns
    -------
    float
        The selected ``xlam`` value. Also stored in ``st.xlam`` and
        ``cfg.xlam`` as a side effect.
    """
    grid = sorted(float(v) for v in xlam_grid)
    original_xlam = float(st.xlam)
    auto_maxiter = int(cfg.xlam_auto_maxiter or cfg.map_maxiter)
    max_peaks = int(getattr(cfg, "xlam_max_peaks", 1))
    min_prominence = float(getattr(cfg, "xlam_peak_min_prominence", 0.1))
    criterion = str(getattr(cfg, "xlam_criterion", "discrepancy")).lower()
    chi2_tolerance = float(getattr(cfg, "xlam_chi2_tolerance", 0.02))

    log(
        f"Auto-xlam search: grid={[f'{x:.0f}' for x in grid]}  "
        f"criterion={criterion}  "
        + (f"chi2_tolerance={chi2_tolerance:.3f}" if criterion == "chi2"
           else f"roughness_threshold={smooth_threshold:.3f}")
        + f"  max_peaks={max_peaks}  maxiter={auto_maxiter}"
    )

    # ── run all grid fits ────────────────────────────────────────────────────
    records: list[tuple[float, float, float, int]] = []  # (xlam, chi2_red, roughness, n_peaks)

    for xlam in grid:
        st.xlam = xlam
        res = _fit_map_once(
            st, np.asarray(a0, float), bounds,
            auto_maxiter, cfg.map_ftol, cfg.map_maxfun, cfg.print_every,
            f"auto-xlam {xlam:.0f}",
            **map_kwargs,
        )
        b = np.asarray(res.x[: st.nl], float)
        roughness = compute_losvd_roughness(b)
        n_peaks = compute_losvd_n_peaks(b, min_prominence)
        chi2_val, ngood = _chi2_stats(st, res.x)
        chi2_red = chi2_val / ngood if ngood > 0 else np.inf
        unimodal_tag = "" if n_peaks <= max_peaks else f"  [{n_peaks} peaks]"
        log(
            f"  xlam={xlam:8.0f}  chi2_red={chi2_red:.4f}  "
            f"roughness={roughness:.4f}  peaks={n_peaks}{unimodal_tag}"
        )
        records.append((xlam, chi2_red, roughness, n_peaks))

        # roughness criterion stops early; chi2 criterion always runs full grid
        if criterion == "roughness":
            if roughness <= smooth_threshold and n_peaks <= max_peaks:
                break

    # ── select best xlam ────────────────────────────────────────────────────
    best_xlam = float(grid[-1])

    if criterion == "chi2":
        # Among unimodal solutions, pick the largest xlam within chi2 tolerance.
        chi2_reds = [r[1] for r in records if r[3] <= max_peaks]
        if chi2_reds:
            chi2_min = min(chi2_reds)
            chi2_max_allowed = chi2_min * (1.0 + chi2_tolerance)
            log(
                f"  chi2_min={chi2_min:.4f}  "
                f"chi2_max_allowed={chi2_max_allowed:.4f}  (tolerance={chi2_tolerance:.3f})"
            )
            # Iterate largest-to-smallest: first match is the best (most-regularised)
            for xlam, chi2_red, roughness, n_peaks in reversed(records):
                if n_peaks <= max_peaks and chi2_red <= chi2_max_allowed:
                    best_xlam = xlam
                    break
            else:
                log("Auto-xlam (chi2): no unimodal solution within tolerance; using largest xlam")
        else:
            log(f"Auto-xlam (chi2): no unimodal solution found; using xlam={grid[-1]:.0f}")

    else:  # criterion == "roughness"
        last = records[-1]
        if last[2] <= smooth_threshold and last[3] <= max_peaks:
            best_xlam = last[0]
        else:
            log(
                f"Auto-xlam (roughness): no grid value satisfies roughness <= "
                f"{smooth_threshold:.3f} and peaks <= {max_peaks}; "
                f"using xlam={grid[-1]:.0f}"
            )

    st.xlam = best_xlam
    cfg.xlam = best_xlam
    log(
        f"Auto-xlam: selected xlam={best_xlam:.0f} "
        f"(original cfg.xlam was {original_xlam:.0f})"
    )
    return best_xlam


def _v_recovery_is_sane(
    candidate_xlam: float,
    trials: dict,
    v_center_est: float,
    max_peaks: int,
) -> bool:
    """Guard against a candidate xlam landing in a spurious local optimum.

    Adapted from :func:`kinextract.joint.fit_joint_auto_xlam`'s
    ``_candidate_ok`` (same two checks, same rationale -- found on a real
    MUSE bin near the resolution limit, where one xlam trial's optimizer
    landed on a genuinely different, lower-chi2 local optimum with V off by
    >40 km/s, indistinguishable from the correct answer by chi2 alone):

    1. Compare the candidate's recovered V against the pre-fit
       cross-correlation velocity estimate `v_center_est`.
    2. Compare against the *other* trials' own median V (robustly, via
       MAD) -- what actually catches the failure, since the spurious trial
       is typically an outlier relative to every other trial regardless of
       whether the external xcorr estimate itself can be trusted.

    Parameters
    ----------
    candidate_xlam : float
        The trial xlam being checked.
    trials : dict
        ``{xlam: trial_dict}`` for every xlam evaluated so far (each
        trial_dict has ``"v_rec"``, ``"sigma_rec"``, ``"n_peaks"`` keys).
    v_center_est : float
        Pre-fit cross-correlation velocity estimate (km/s).
    max_peaks : int
        Unimodality threshold; only unimodal trials count as "other"
        candidates for the MAD comparison.

    Returns
    -------
    bool
        True if `candidate_xlam` passes both checks (or there are too few
        other trials to run the MAD check).
    """
    cand = trials[candidate_xlam]
    v_candidate = cand["v_rec"]
    sigma_candidate = max(cand.get("sigma_rec", 1.0), 1.0)

    if abs(v_candidate - v_center_est) > 5.0 * sigma_candidate:
        return False

    other_vs = [
        t["v_rec"] for xlam, t in trials.items()
        if xlam != candidate_xlam and t["n_peaks"] <= max_peaks
    ]
    if len(other_vs) >= 2:
        median_other_v = float(np.median(other_vs))
        mad_other_v = float(np.median(np.abs(np.asarray(other_vs) - median_other_v)))
        scale = max(mad_other_v, 3.0)
        if abs(v_candidate - median_other_v) > 5.0 * scale:
            return False
    return True


def _discrepancy_principle_search(
    evaluate_xlam,
    xlam_lo: float,
    xlam_hi: float,
    max_peaks: int,
    v_center_est: float,
    nsigma: float = 1.0,
    log_label: str = "auto-xlam (discrepancy)",
    max_bracket_expansions: int = 6,
    max_bisections: int = 8,
    log_tol_dex: float = 0.05,
):
    """Generic pPXF-style discrepancy-principle 1-D search over ``xlam``.

    Adapted from Cappellari (2017, MNRAS 466, 798, "Improving the full
    spectrum fitting method: accurate convolution with Gauss-Hermite
    functions", Sec. 3.5)'s ``REGUL`` regularization-strength procedure for
    pPXF (Cappellari & Emsellem 2004, PASP 116, 138): increase the
    regularization strength until the total chi2 rises by ``nsigma * sqrt(2
    * ngood)`` above the chi2 achieved at (near-)zero regularization -- the
    standard deviation of a chi-squared distribution with ``ngood`` degrees
    of freedom, i.e. "regularize until the fit degrades by about one sigma
    of the noise's own chi2 fluctuation." Unlike a tolerance-from-minimum
    rule evaluated over a
    fixed grid (:func:`_auto_select_xlam`'s ``"chi2"``/``"roughness"``
    criteria), this target is tied to the *known noise level* rather than
    to the shape of the chi2(xlam) curve, so it degrades gracefully when
    that curve is nearly flat over many orders of magnitude of xlam -- the
    failure mode that motivated this function (see module docstring).

    This is a generic, fit-mechanics-agnostic search: the caller supplies
    `evaluate_xlam`, a callable ``xlam -> dict`` that runs one trial fit at
    that `xlam` and returns a dict with (at least) keys ``"chi2_total"``
    (float, *not* reduced), ``"ngood"`` (int), ``"n_peaks"`` (int, via
    :func:`compute_losvd_n_peaks`), ``"v_rec"`` (float, recovered velocity),
    and ``"sigma_rec"`` (float, recovered dispersion). This lets both the
    shipped MAP path (:func:`_auto_select_xlam_discrepancy`, below) and
    :func:`kinextract.joint.fit_joint_auto_xlam` share this exact search
    algorithm despite having entirely different fit mechanics (different
    objective functions and flat-parameter-vector layouts).

    Parameters
    ----------
    evaluate_xlam : callable
        ``xlam -> dict`` as described above. Expected to be expensive (a
        full MAP refit per call); every call is logged and cached in the
        returned `trials` dict, so no `xlam` is ever evaluated twice.
    xlam_lo, xlam_hi : float
        Initial bracket. `xlam_lo` should be small enough to be a good
        proxy for "negligible regularization" (used to establish the
        discrepancy-principle target); `xlam_hi` is expanded geometrically
        (x10, up to `max_bracket_expansions` times) if the target chi2 rise
        hasn't been reached by `xlam_hi` yet.
    max_peaks : int
        Unimodality constraint (:func:`compute_losvd_n_peaks`); candidates
        exceeding this are rejected in favor of a smaller (more
        regularized) xlam, same convention as :func:`_auto_select_xlam`.
    v_center_est : float
        Pre-fit cross-correlation velocity estimate, used by
        :func:`_v_recovery_is_sane` to guard against a spurious local
        optimum.
    nsigma : float, optional
        Multiplier on the ``sqrt(2*ngood)`` discrepancy-principle target
        (1.0 = Cappellari's own stated convention). Larger values select
        more regularization (a bigger allowed chi2 rise).
    log_label : str, optional
        Prefix for log lines, so callers (shipped vs. joint) can
        distinguish their search in the log.
    max_bracket_expansions : int, optional
        Cap on how many times `xlam_hi` is multiplied by 10 while
        searching for a bracket that reaches the target.
    max_bisections : int, optional
        Cap on bisection iterations once a bracket containing the target
        is found.
    log_tol_dex : float, optional
        Bisection convergence tolerance, in decades of `xlam`.

    Returns
    -------
    best_xlam : float
        The selected xlam.
    trials : dict
        ``{xlam: trial_dict}`` for every xlam evaluated, for diagnostics.
    """
    trials: dict = {}

    def _eval(xlam: float) -> dict:
        xlam = float(xlam)
        if xlam not in trials:
            trials[xlam] = evaluate_xlam(xlam)
            t = trials[xlam]
            log(
                f"  {log_label} xlam={xlam:10.4g}  chi2_total={t['chi2_total']:.4g}  "
                f"ngood={t['ngood']}  peaks={t['n_peaks']}  "
                f"V={t['v_rec']:+.2f}  sigma={t.get('sigma_rec', float('nan')):.2f}"
            )
        return trials[xlam]

    lo, hi = float(xlam_lo), float(xlam_hi)
    t_lo = _eval(lo)
    chi2_min = t_lo["chi2_total"]
    ngood = max(t_lo["ngood"], 1)
    # Cappellari (2017, Sec 3.5)'s stated procedure first rescales the input
    # errors so the *unregularized* fit has chi2/ngood == 1 exactly, then
    # targets chi2_rescaled = ngood + sqrt(2*ngood). Converting that target
    # back to this call's own (possibly miscalibrated) error scale without
    # literally mutating st.gerr: if errors are off by a factor `s` (chi2_min
    # = s^2 * ngood, i.e. s^2 = chi2_min/ngood = chi2_red_min), the same
    # rescaled-units target becomes chi2_min * (1 + nsigma*sqrt(2/ngood)) in
    # this call's own units -- reduces to the textbook chi2_min +
    # nsigma*sqrt(2*ngood) exactly when chi2_red_min == 1 (errors already
    # well calibrated), and scales the allowed chi2 rise down/up
    # proportionally when they are not.
    chi2_red_min = chi2_min / ngood
    target = chi2_min * (1.0 + nsigma * float(np.sqrt(2.0 / ngood)))
    log(f"{log_label}: chi2_min={chi2_min:.4g} (xlam={lo:.4g})  chi2_red_min={chi2_red_min:.4g}  "
        f"target={target:.4g}  (chi2_min * (1 + {nsigma:.2f}*sqrt(2/{ngood})))")

    t_hi = _eval(hi)
    expansions = 0
    while t_hi["chi2_total"] < target and expansions < max_bracket_expansions:
        lo = hi  # tighten: previous hi is still below target, so it's a valid new lower bound
        hi *= 10.0
        t_hi = _eval(hi)
        expansions += 1

    if t_hi["chi2_total"] < target:
        log(f"{log_label}: target not reached even at xlam={hi:.4g} after "
            f"{max_bracket_expansions} bracket expansions; using largest xlam tried")
        best_xlam = hi
    else:
        # Bisect in log10(xlam) space: lo always chi2 < target, hi always chi2 >= target.
        for _ in range(max_bisections):
            if np.log10(hi) - np.log10(lo) < log_tol_dex:
                break
            mid = float(10.0 ** (0.5 * (np.log10(lo) + np.log10(hi))))
            t_mid = _eval(mid)
            if t_mid["chi2_total"] < target:
                lo = mid
            else:
                hi = mid
        best_xlam = hi

    # Unimodality: if the converged answer is multi-peaked, fall back to the
    # smallest *evaluated* xlam that is both unimodal and meets the target;
    # expand further if no such trial exists yet.
    if trials[best_xlam]["n_peaks"] > max_peaks:
        candidates = sorted(
            (x for x, t in trials.items() if t["n_peaks"] <= max_peaks and t["chi2_total"] >= target),
        )
        if candidates:
            best_xlam = candidates[0]
            log(f"{log_label}: discrepancy-target xlam was multi-peaked; "
                f"using smallest unimodal candidate meeting target, xlam={best_xlam:.4g}")
        else:
            expand_hi = hi
            extra = 0
            while trials[best_xlam]["n_peaks"] > max_peaks and extra < max_bracket_expansions:
                expand_hi *= 10.0
                _eval(expand_hi)
                if trials[expand_hi]["n_peaks"] <= max_peaks:
                    best_xlam = expand_hi
                extra += 1
            if trials[best_xlam]["n_peaks"] > max_peaks:
                log(f"{log_label}: WARNING no unimodal candidate found up to "
                    f"xlam={expand_hi:.4g}; returning best available (still multi-peaked)")

    # Spurious-local-optimum guard: fall back through decreasing-chi2-rise
    # (i.e. smaller-xlam) candidates that still meet the target until one
    # passes, mirroring kinextract.joint.fit_joint_auto_xlam's own fallback.
    if not _v_recovery_is_sane(best_xlam, trials, v_center_est, max_peaks):
        fallback_pool = sorted(
            (x for x, t in trials.items()
             if t["n_peaks"] <= max_peaks and x != best_xlam),
            reverse=True,
        )
        for x in fallback_pool:
            if _v_recovery_is_sane(x, trials, v_center_est, max_peaks):
                log(f"{log_label}: xlam={best_xlam:.4g}'s recovered V failed the "
                    f"sanity check; falling back to xlam={x:.4g}")
                best_xlam = x
                break
        else:
            log(f"{log_label}: WARNING xlam={best_xlam:.4g}'s recovered V failed "
                f"the sanity check and no fallback candidate passed either; "
                f"using it anyway")

    log(f"{log_label}: selected xlam={best_xlam:.4g}")
    return best_xlam, trials


def _auto_select_xlam_discrepancy(
    st: FitState,
    cfg: FitConfig,
    a0: np.ndarray,
    bounds: list,
    xlam_grid: tuple,
    map_kwargs: dict,
    nsigma: float = 1.0,
) -> float:
    """Shipped (non-joint) MAP path's discrepancy-principle ``xlam``
    selector (``cfg.xlam_criterion="discrepancy"``) -- see
    :func:`_discrepancy_principle_search` for the algorithm and rationale.

    Recenters the wing-taper's ``v_center`` on
    :func:`kinextract.numerics.estimate_velocity_xcorr` once, up front
    (matching :func:`kinextract.joint.fit_joint_auto_xlam`'s own
    ``recenter_v`` -- high-xlam solutions are exactly where a
    zero-centered wing taper biases V the most, so recentering before the
    search removes that asymmetry regardless of where the search lands).

    Parameters
    ----------
    st : FitState
        Fit state whose ``xlam``/``v_center`` are overwritten with the
        selected values on return.
    cfg : FitConfig
        Run configuration; ``cfg.xlam_max_peaks``, ``cfg.xlam`` (overwritten
        on return) are used.
    a0 : ndarray
        Initial guess for the flat parameter vector.
    bounds : list of tuple
        Per-parameter bounds passed to :func:`_fit_map_once`.
    xlam_grid : tuple of float
        Only its min/max are used, as the search's initial bracket (kept as
        a grid-shaped argument so existing ``cfg.xlam_auto_grid`` configs
        need no changes to opt into this criterion).
    map_kwargs : dict
        Extra kwargs forwarded to every :func:`_fit_map_once` call.
    nsigma : float, optional
        See :func:`_discrepancy_principle_search`.

    Returns
    -------
    float
        The selected xlam. Also stored in ``st.xlam``/``cfg.xlam``.
    """
    v_center_est = estimate_velocity_xcorr(st)
    st.v_center = v_center_est

    max_peaks = int(getattr(cfg, "xlam_max_peaks", 1))
    min_prominence = float(getattr(cfg, "xlam_peak_min_prominence", 0.1))
    auto_maxiter = int(cfg.xlam_auto_maxiter or cfg.map_maxiter)
    a0 = np.asarray(a0, float)

    def evaluate_xlam(xlam: float) -> dict:
        st.xlam = float(xlam)
        res = _fit_map_once(
            st, a0, bounds, auto_maxiter, cfg.map_ftol, cfg.map_maxfun, cfg.print_every,
            f"auto-xlam(discrepancy) {xlam:.4g}", **map_kwargs,
        )
        b = np.asarray(res.x[: st.nl], float)
        chi2_total, ngood = _chi2_stats(st, res.x)
        n_peaks = compute_losvd_n_peaks(b, min_prominence)
        gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
        return dict(chi2_total=chi2_total, ngood=ngood, n_peaks=n_peaks,
                     v_rec=float(gh["vherm"]), sigma_rec=float(gh["sherm"]))

    grid = sorted(float(v) for v in xlam_grid)
    best_xlam, _trials = _discrepancy_principle_search(
        evaluate_xlam, xlam_lo=grid[0], xlam_hi=grid[-1],
        max_peaks=max_peaks, v_center_est=v_center_est, nsigma=nsigma,
        log_label="Auto-xlam (discrepancy)",
    )

    st.xlam = best_xlam
    cfg.xlam = best_xlam
    return best_xlam


def fit_state_map_with_optional_clean(
    st: FitState, cfg: FitConfig, a0: np.ndarray, bounds: list,
):
    """Run the MAP fit, optionally with sigma-clipping.

    This is the shipped (non-joint) fit path: a bound-constrained L-BFGS-B
    optimisation of ``chi2 + wing-tapered smoothness penalty + LOSVD
    normalisation penalty`` (:func:`kinextract.numerics.objective_map`), the
    same objective minimised by the original Fortran implementation this
    package is a port of. Used directly by pre-normalized-mode fits
    (``cfg.fit_continuum=False``) and internally by
    :mod:`kinextract.errors`'s residual-bootstrap error estimation, which
    needs hundreds to thousands of fast independent refits per fit -- a
    cost only a point-estimate optimisation, not full posterior sampling,
    can sustain. A full-posterior (NUTS/HMC) alternative is available via
    :func:`kinextract.bayesian.fit_state_bayesian` for users who want a
    sampled posterior instead of a point estimate; it is not used by
    default because a comprehensive recovery-accuracy comparison found the
    MAP point estimate matches or exceeds the posterior mean's LOSVD-shape
    recovery across tested instruments and LOSVD shapes, at a small
    fraction of the runtime.

    This is the mid-level orchestration function: it optionally runs the
    automatic ``xlam`` grid search (:func:`_auto_select_xlam`) once up
    front, then performs a single (optionally sigma-clipped) MAP fit.
    Continuum-cofit fits (``cfg.fit_continuum=True``) never reach this
    function -- :func:`run_spectral_fit` dispatches those to
    :mod:`kinextract.joint` instead.

    Parameters
    ----------
    st : FitState
        Fit state to optimise; ``st.xlam`` may be overwritten by the
        automatic regularisation search, and ``st.clean_good_mask`` is
        updated as a side effect when cleaning is enabled.
    cfg : FitConfig
        Run configuration controlling regularisation search
        (``xlam_auto`` and related settings) and cleaning (``clean`` and
        related settings).
    a0 : ndarray
        Initial guess for the flat parameter vector.
    bounds : list of tuple
        Per-parameter ``(lower, upper)`` bounds.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The optimisation result from the MAP fit (after any cleaning).
    """
    kwargs = dict(
        map_gtol=cfg.map_gtol,
        map_maxls=cfg.map_maxls,
        use_scaled_optimizer=cfg.use_scaled_optimizer,
        use_jax_objective=cfg.use_jax_objective,
        jax_enable_x64=cfg.jax_enable_x64,
    )

    original_clean = bool(cfg.clean)

    # Per-spectrum automatic smoothing: run before the MAP fit so that
    # st.xlam is correct for it. Bootstrap refits skip this
    # (xlam_auto=False in cfg_frozen; see losvd_errors._make_frozen_cfg) so
    # they use the xlam selected here.
    if getattr(cfg, "xlam_auto", False):
        xlam_grid = getattr(cfg, "xlam_auto_grid", (20., 50., 100., 250., 600., 1500., 4000.))
        if str(getattr(cfg, "xlam_criterion", "discrepancy")).lower() == "discrepancy":
            _auto_select_xlam_discrepancy(
                st, cfg, np.asarray(a0, float), bounds,
                xlam_grid=xlam_grid, map_kwargs=kwargs,
                nsigma=float(getattr(cfg, "xlam_discrepancy_nsigma", 0.3)),
            )
        else:
            _auto_select_xlam(
                st, cfg, np.asarray(a0, float), bounds,
                xlam_grid=xlam_grid,
                smooth_threshold=float(getattr(cfg, "xlam_smooth_threshold", 0.25)),
                map_kwargs=kwargs,
            )

    def one_fit(a_start, do_clean: bool, label: str = "MAP optimize"):
        if do_clean:
            pm = build_clean_protect_mask(st, cfg)
            log(f"Cleaning protection: {pm.sum()} pixels")
            res, good_mask = run_iterative_clean_map(
                st, a_start, bounds,
                cfg.map_maxiter, cfg.map_ftol, cfg.map_maxfun, cfg.print_every,
                cfg.clean_sigma, cfg.clean_maxiter, cfg.clean_minpix,
                pm, cfg.clean_protect_absorption_only,
                getattr(cfg, "clean_bloom_pixels", 0),
                **kwargs,
            )
            st.clean_good_mask = good_mask
            return res

        return _fit_map_once(
            st, a_start, bounds,
            cfg.map_maxiter, cfg.map_ftol, cfg.map_maxfun, cfg.print_every,
            label, **kwargs,
        )

    return one_fit(a0, do_clean=original_clean)


def _fit_map_sigl0_recenter(
    st: FitState, cfg: FitConfig, a0: np.ndarray, bounds: list,
):
    """Self-consistent ``sigl0``/``v_center`` fixed-point wrapper around
    :func:`fit_state_map_with_optional_clean`.

    Ports :func:`kinextract.joint.fit_joint_auto_xlam_sigl0`'s validated
    fixed-point iteration to the shipped (non-joint) MAP path -- see that
    function's docstring for the full rationale, validation history, and
    citations. Root cause: a real-data comparison between this path and the
    joint path's ``joint_prenorm=True`` mode (otherwise matched settings,
    same pre-normalized spectrum) found a large, unexplained V/sigma
    discrepancy, traced to this exact gap. The wing-tapered smoothness
    penalty (:func:`kinextract.numerics._wing_taper_lam_vec`) depends on
    both ``st.sigl0`` (where the taper begins ramping up) and ``st.v_center``
    (its pivot); the joint path always self-corrects both before returning,
    while this path previously left them at their initial values
    (``cfg.sigl``, 0.0) for the entire fit, however wrong those turned out to
    be -- most consequential at high ``xlam``, exactly where kinextract's own
    regularization search tends to land.

    Parameters
    ----------
    st : FitState
        Fit state to optimize; ``st.sigl0``, ``st.v_center``, and ``st.xlam``
        are overwritten as a side effect of the final iteration.
    cfg : FitConfig
        ``cfg.joint_recenter_v``, ``cfg.joint_n_sigl0_iter``, and
        ``cfg.joint_sigl0_tol`` control this wrapper exactly as they control
        :func:`kinextract.joint.fit_joint_auto_xlam_sigl0` -- shared fields,
        since the underlying wing-taper penalty and fixed-point rationale
        are identical between the two paths.
    a0, bounds :
        Passed through to :func:`fit_state_map_with_optional_clean` on every
        iteration.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The best round's fit result (see the divergence-guard note below --
        not necessarily the final round's).

    Notes
    -----
    **Divergence guard.** The fixed-point iteration is not guaranteed to
    converge monotonically on every chi2(xlam) curve: on an unusually flat
    one (confirmed on a clean, oversimplified synthetic mock -- an
    under-constrained problem for regularization selection in general, see
    ``examples/notebooks/01_basic_mock_fit.ipynb``), each round's recovered
    sigma can land *farther* from that round's ``sigl0`` input than the
    previous round did, instead of converging -- a positive feedback loop
    rather than a fixed point. This function tracks the smallest
    ``|sigl0_input - recovered_sigma|`` gap seen across all rounds and
    reverts to that round's fit/``sigl0``/``xlam`` if the final round's gap
    is worse, rather than returning a compounded-drift result.
    """
    if bool(getattr(cfg, "joint_recenter_v", True)):
        st.v_center = estimate_velocity_xcorr(st)

    n_iter = max(1, int(getattr(cfg, "joint_n_sigl0_iter", 3)))
    tol = float(getattr(cfg, "joint_sigl0_tol", 2.0))

    res = None
    best = None  # (gap, res, xlam, sigl0_input)
    last_gap = None
    for _ in range(n_iter):
        sigl0_in = float(st.sigl0)
        res = fit_state_map_with_optional_clean(st, cfg, a0, bounds)
        gp, b, *_ = evaluate_model_gp(res.x, st)
        gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
        recovered_sigma = float(gh["sherm"])
        if not np.isfinite(recovered_sigma) or recovered_sigma <= 0:
            break
        gap = abs(sigl0_in - recovered_sigma)
        if best is None or gap < best[0]:
            best = (gap, res, float(st.xlam), sigl0_in)
        reached = gap <= tol
        st.sigl0 = recovered_sigma
        last_gap = gap
        if reached:
            break

    if best is not None and last_gap is not None and best[0] < last_gap:
        gap_best, res, xlam_best, sigl0_best = best
        st.xlam = xlam_best
        st.sigl0 = sigl0_best
        log(
            f"sigl0 fixed-point iteration diverged (final gap {last_gap:.2f} > "
            f"best gap {gap_best:.2f} km/s); reverting to the round with the "
            f"smallest |sigl0 - recovered_sigma| (sigl0={sigl0_best:.2f}, "
            f"xlam={xlam_best:.4g})"
        )
    return res


# =============================================================================
# Section 14 - Top-level runner
# =============================================================================

def run_spectral_fit(
    cfg: FitConfig,
    gal_file: Optional[str] = None,
    *,
    gal_errors=None,
    write_outputs: Optional[bool] = None,
    output_prefix: Optional[str] = None,
) -> dict:
    """Run the full non-parametric LOSVD spectral fit end-to-end.

    This is the primary public entry point of ``kinextract``. Given a
    :class:`~kinextract.config.FitConfig` and a galaxy spectrum, it: loads
    and prepares the spectrum and stellar template library into a
    :class:`~kinextract.state.FitState` (:func:`~kinextract.spectrum.make_fit_state`);
    builds an initial non-parametric guess for the LOSVD and template
    weights (:func:`~kinextract.spectrum.build_initial_guess_nonparam`);
    runs a bound-constrained MAP optimisation
    (:func:`fit_state_map_with_optional_clean`) over that same flat
    parameter vector, with optional automatic regularisation (``xlam``)
    selection and sigma-clipping as configured -- or, when
    ``cfg.fit_continuum=True`` (or ``cfg.joint_prenorm=True``), dispatches
    to :mod:`kinextract.joint`'s continuum-cofit engine instead; and
    finally computes summary statistics and (by default) writes the
    standard fitlov/ascii/rms/template output files. It supports both
    notebook usage (set ``cfg.gal_file`` and call ``run_spectral_fit(cfg)``)
    and command-line/scripted usage (pass `gal_file` explicitly, overriding
    `cfg`).

    The non-joint MAP path (``cfg.fit_continuum=False`` and
    ``cfg.joint_prenorm=False``) self-corrects the wing-tapered smoothness
    penalty's ``v_center`` (recentered on a cross-correlation velocity
    estimate) and ``sigl0`` (fixed-point iteration toward the recovered
    sigma) via :func:`_fit_map_sigl0_recenter`, mirroring
    :func:`kinextract.joint.fit_joint_auto_xlam_sigl0`'s validated approach
    -- see its docstring for the rationale. This replaced an earlier
    zero-centered/fixed-``sigl0`` default after a real-data comparison found
    it caused a large, spurious V/sigma discrepancy against the joint path
    on otherwise-matched settings. Both corrections are controlled by
    ``cfg.joint_recenter_v``/``cfg.joint_n_sigl0_iter``/``cfg.joint_sigl0_tol``
    (shared fields with the joint path). A full-posterior (NUTS/HMC)
    alternative to this MAP
    point estimate is available via
    :func:`kinextract.bayesian.fit_state_bayesian` for users who want a
    sampled posterior instead; call it directly in place of this function
    if you need distributional uncertainty from a single fit rather than
    (or in addition to) the bootstrap error estimators in
    :mod:`kinextract.errors`. It is not the default because a
    comprehensive recovery-accuracy comparison found the MAP point
    estimate matches or exceeds the posterior mean's LOSVD-shape recovery
    across tested instruments and LOSVD shapes, at a small fraction of the
    runtime -- consistent with the original Fortran implementation this
    package is a port of, which also uses MAP + Monte Carlo resampling
    rather than full posterior sampling.

    For characterizing recovery bias for a specific target's instrument/
    S-N/template configuration -- e.g. near the instrumental resolution
    limit, where any point estimator has some residual bias -- see
    :func:`kinextract.validation.assess_recovery_bias`, which runs matched
    mock simulations and reports (and optionally corrects for) the
    empirical bias directly, rather than relying on generic multi-instrument
    numbers.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object specifying the wavelength/redshift window,
        kinematic grid, regularisation, cleaning, and continuum-cofit
        settings for the fit. In notebook usage, you may set
        ``cfg.gal_file`` directly and call ``run_spectral_fit(cfg)``.
    gal_file : str, optional
        Spectrum file path. If supplied, this overrides ``cfg.gal_file``
        (and `cfg` is updated in place to match, for later introspection).
        Useful for command-line/batch usage where the same `cfg` is reused
        across many spectra.
    gal_errors : array-like or float, optional
        Per-pixel error estimates to use instead of the errors read from
        the spectrum file. Accepts a 1-D array the same length as the full
        spectrum (before wavelength-range selection) or a scalar float for
        uniform errors. When provided, this overrides both the file errors
        and the ``use_spectrum_errors`` setting in `cfg`.
    write_outputs : bool, optional
        If True, write fitlov/ascii/rms/template output files to
        ``cfg.outdir``. If False, skip writing files and return the
        equivalent model products purely in memory (useful for
        notebook-driven exploration or bootstrap workers where per-replicate
        file I/O would be wasteful). Defaults to ``cfg.write_outputs`` (which
        itself defaults to False) when not given explicitly, so this can be
        set once per :class:`~kinextract.config.FitConfig` instead of on
        every call; pass an explicit True/False here to override the config
        for a single call.
    output_prefix : str, optional
        Prefix for output files. If None, inferred from `gal_file` via
        :func:`~kinextract.io.infer_output_prefix`.

    Returns
    -------
    dict
        Dictionary with keys:

        ``"state"``
            The fully-populated :class:`~kinextract.state.FitState` for
            this fit (velocity grid, data, best-fit LOSVD context, etc.).
        ``"template_files"``
            List of stellar template file paths used in the fit.
        ``"result"``
            The ``scipy.optimize.OptimizeResult`` from the final MAP
            optimisation.
        ``"a_map"``
            Best-fit flat parameter vector (ndarray).
        ``"f_map"``
            Best-fit objective value (float).
        ``"outputs"``
            Dictionary of derived model products: the model spectrum
            (``"gp"``), recovered LOSVD (``"b"``), template weights
            (``"w"`` and fractional ``"wfrac"``), weighted template
            spectrum (``"tt"``), continuum parameters (``"coff"``,
            ``"coff2"``, ``"A"``), fit RMS (``"rms"``, ``"nrms"``), the
            co-fit continuum (``"continuum"``), and, when `write_outputs`
            is True, the paths of the files written (``"paths"``).
        ``"chi2"``
            Final chi-squared over good pixels (float; see
            :func:`_chi2_stats`).
        ``"ngood"``
            Number of good (fitted, unmasked) pixels (int).
        ``"prefix"``
            Output file prefix used (or that would have been used).
        ``"gal_file"``
            Input spectrum file path actually used for this fit.

    Examples
    --------
    >>> from kinextract import FitConfig, run_spectral_fit
    >>> cfg = FitConfig(gal_file="bin0105.spec", template_list_file="Tlist",
    ...                 zgal=0.0016, wavefitmin=8400, wavefitmax=8800)
    >>> fit = run_spectral_fit(cfg)  # doctest: +SKIP
    >>> fit["chi2"]  # doctest: +SKIP
    """
    # Allow notebook usage:
    #
    #   cfg = FitConfig(gal_file="/path/to/spec", ...)
    #   fit = run_spectral_fit(cfg)
    #
    # and command-line/script usage:
    #
    #   fit = run_spectral_fit(cfg, gal_file=spectrum_path)

    if gal_file is None:
        gal_file = getattr(cfg, "gal_file", "")

    if gal_file is None or str(gal_file).strip() == "":
        raise ValueError(
            "No spectrum file supplied. Either call "
            "run_spectral_fit(cfg, gal_file='/path/to/spectrum') "
            "or set cfg.gal_file before calling run_spectral_fit(cfg)."
        )

    gal_file = str(gal_file)

    if write_outputs is None:
        write_outputs = cfg.write_outputs

    # Keep cfg.gal_file synchronized for notebook introspection/debugging.
    cfg.gal_file = gal_file

    log(f"==== spectral fitting START | {gal_file} ====")
    log(
        f"wavefit=[{cfg.wavefitmin}, {cfg.wavefitmax}] z={cfg.zgal} "
        f"sigl={cfg.sigl} xlam={cfg.xlam}"
    )
    log(
        f"fit_continuum={cfg.fit_continuum} "
        f"prenorm={not cfg.fit_continuum} "
        f"joint_prenorm={cfg.joint_prenorm}"
    )

    with Timer("build FitState"):
        st, tpl_files = make_fit_state(cfg, gal_file=gal_file, gal_errors=gal_errors)

    prefix = output_prefix if output_prefix is not None else infer_output_prefix(gal_file)

    use_joint = cfg.fit_continuum or cfg.joint_prenorm
    if use_joint:
        # kinextract.joint has its own parameter-vector layout and its own
        # initial-guess/optimization driver, so it bypasses
        # build_initial_guess_nonparam/fit_state_map_with_optional_clean/
        # evaluate_model_gp entirely rather than trying to fit through
        # machinery specific to the shipped layout. See
        # kinextract.joint.run_joint_fit's docstring.
        #
        # Only triggered in pre-normalized mode (fit_continuum=False) if
        # cfg.joint_prenorm is explicitly set -- joint's sigl0 fixed-point
        # iteration costs up to n_sigl0_iter * len(xlam_auto_grid) full
        # optimizations per fit, so this stays opt-in rather than silently
        # slowing down every default pre-normalized fit package-wide.
        from .joint import run_joint_fit

        if cfg.fit_continuum:
            log("Continuum method: joint (P-spline continuum co-fit in the model)")
        else:
            log(
                "Continuum method: none (prenormalized; joint_prenorm=True uses "
                "the joint engine's sigl0/v_center self-convergence, but the "
                "continuum stays fixed at 1.0, not co-fit)"
            )
        outputs = run_joint_fit(st, cfg, write_outputs=write_outputs, outdir=cfg.outdir, prefix=prefix)
        res = outputs["result"]
        if not res.success:
            log(f"WARNING: optimizer reported: {res.message}")
        a_map, f_map = np.asarray(res.x, float), float(res.fun)
        chi2, ngood = _chi2_stats_from_gp(st, outputs["gp"])

        log(f"Final chi2={chi2:.6g} ngood={ngood} xlam={st.xlam}")
        log("==== spectral fitting END ====")

        return {
            "state": st,
            "template_files": tpl_files,
            "result": res,
            "a_map": a_map,
            "f_map": f_map,
            "outputs": outputs,
            "chi2": chi2,
            "ngood": ngood,
            "prefix": prefix,
            "gal_file": gal_file,
        }

    a0, xlb, xub = build_initial_guess_nonparam(
        st, cfg.coff, cfg.coff2,
        w_bounds=cfg.template_w_bounds if cfg.template_w_bounds is not None else (1e-5, 1.0),
    )
    bounds = list(zip(xlb, xub))

    res = _fit_map_sigl0_recenter(st, cfg, a0, bounds)
    if not res.success:
        log(f"WARNING: optimizer reported: {res.message}")

    a_map, f_map = np.asarray(res.x, float), float(res.fun)

    if write_outputs:
        outputs = write_fitlov_outputs(st, a_map, cfg.outdir, prefix)
    else:
        # In-memory equivalent of the most useful pieces from write_fitlov_outputs(),
        # without touching the filesystem.
        from .numerics import compute_weighted_template_spectrum, evaluate_model_gp
        gp, b, w, coff, coff2, A = evaluate_model_gp(a_map, st)
        sw = float(np.sum(w))
        wfrac = w / sw if sw else w
        tt = compute_weighted_template_spectrum(st, w)
        cont = getattr(st, "continuum_mult", np.ones(st.npix))
        good = st.gerr < 1e9
        sl = slice(st.iskip, st.npix - st.iskip)
        gfr = good[sl]
        rms = (
            float(np.sqrt(np.mean((st.g[sl] - gp[sl])[gfr] ** 2)))
            if gfr.any()
            else np.nan
        )
        nrms = int(gfr.sum())
        chi2_total = (
            float(np.sum((st.g[sl][gfr] - gp[sl][gfr]) ** 2 / st.gerr[sl][gfr] ** 2))
            if gfr.any()
            else 0.0
        )
        chi2_red = chi2_total / max(nrms - 1, 1)

        outputs = {
            "gp": gp,
            "b": b,
            "w": w,
            "wfrac": wfrac,
            "tt": tt,
            "coff": coff,
            "coff2": coff2,
            "A": A,
            "rms": rms,
            "chi2_red": chi2_red,
            "chi2_total": chi2_total,
            "nrms": nrms,
            "continuum": cont,
            "paths": {},
        }

    chi2, ngood = _chi2_stats(st, a_map)

    log(f"Final chi2={chi2:.6g} ngood={ngood} xlam={st.xlam}")
    log("==== spectral fitting END ====")

    return {
        "state": st,
        "template_files": tpl_files,
        "result": res,
        "a_map": a_map,
        "f_map": f_map,
        "outputs": outputs,
        "chi2": chi2,
        "ngood": ngood,
        "prefix": prefix,
        "gal_file": gal_file,
    }
