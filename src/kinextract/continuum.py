"""ALS (asymmetric least-squares) continuum estimation for the kinextract fitter.

This module implements the "ALS mode" branch of kinextract's two-mode
continuum handling (the other being the legacy pre-normalized-spectrum mode,
where the continuum division already happened upstream in Fortran ``imfit``
and this module is not used at all). In ALS mode, a smooth Eilers (2003)
asymmetric-least-squares baseline is co-fit with the LOSVD/template model:
``asymmetric_least_squares_continuum`` solves the penalized banded system,
``optimize_als_hyperparams_for_target``/``score_als_target`` search and score
the smoothing hyperparameters ``als_lam`` (roughness penalty strength) and
``als_p`` (asymmetry weight) by reduced chi-squared or a BIC-like criterion
that penalizes the smoother's effective degrees of freedom (estimated with a
Hutchinson stochastic trace estimator), and ``fit_als_target_absorption_clean``
adds pPXF-like iterative sigma-clipping so that real absorption troughs (e.g.
the Ca II triplet) are excluded from the baseline estimate rather than pulling
it down. ``build_als_line_mask`` and the ``grow_boolean_mask*`` helpers build
and dilate the boolean masks that tell the ALS fit which pixels to ignore.
``init_als_continuum`` and ``update_als_continuum`` are the entry points called
once at :func:`~kinextract.spectrum.make_fit_state` construction time and then
once per outer LOSVD iteration, respectively. This is distinct from the
kinematic sigma-clipping/protect-mask system in ``masking.py``, which cleans
the LOSVD chi-squared fit itself rather than the continuum baseline.
"""

from __future__ import annotations
import warnings
from typing import Optional
import numpy as np
from scipy.linalg import solve_banded
try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else lambda f: f
    warnings.warn(
        "Numba is not installed or failed to import. JIT-compiled inner loops "
        "(chi-squared, convolution, ALS) will run as plain Python, which is "
        "10-50x slower. Install with: pip install numba",
        RuntimeWarning, stacklevel=1,
    )
from ._utils import CEE, BIG, log
from .config import FitConfig


# =============================================================================
# Section 1 - ALS continuum utilities
# =============================================================================

def robust_sigma(values: np.ndarray) -> float:
    """Robust standard-deviation estimate via the normalized MAD.

    Used throughout kinextract wherever a noise scale is needed but the
    sample may contain outliers (e.g. sigma-clipping continuum residuals
    or LOSVD-fit residuals) that would bias an ordinary ``np.std``.

    Parameters
    ----------
    values : ndarray
        Sample of residuals or values to estimate the scatter of.
        Non-finite entries are dropped before the estimate is computed.

    Returns
    -------
    float
        ``1.4826 * median(|values - median(values)|)``, which is a
        consistent estimator of the standard deviation for Gaussian data.
        Falls back to ``np.std`` if the MAD is exactly zero (e.g. a sample
        with many repeated values), and returns ``0.0`` for a single finite
        value or ``nan`` if no finite values remain.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    mad = np.median(np.abs(v - np.median(v)))
    return 1.4826 * mad if mad > 0 else (float(np.std(v)) if v.size > 1 else 0.0)


def _als_dtd_bands(L: int, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute the lam * D'D penalty bands for the ALS banded system.

    D is the (L-2) x L second-difference matrix.  D'D has an exact
    analytic form; all three non-zero band arrays are returned so that
    only the weight diagonal w needs to be added inside the iteration loop.

    Returns (main, off1, off2) with lengths L, L-1, L-2 respectively.
    The boundary values are derived analytically:
      main  : [1, 5, 6, ..., 6, 5, 1]  (4 for the centre pixel when L==3)
      off1  : [-2, -4, ..., -4, -2]
      off2  : [1, 1, ..., 1]
    All values are pre-multiplied by lam.
    """
    main = np.zeros(L, dtype=float)
    off1 = np.zeros(max(L - 1, 0), dtype=float)
    off2 = np.zeros(max(L - 2, 0), dtype=float)

    if L >= 3:
        main[0] = lam
        main[-1] = lam
        # Interior: 6 everywhere except second-from-edge which is 5.
        # Special case L==3: the single interior pixel is touched by only
        # one row of D so D'D[1,1] = (-2)^2 = 4, not 5.
        main[2:-2] = 6.0 * lam
        if L == 3:
            main[1] = 4.0 * lam
        else:
            main[1] = 5.0 * lam
            main[-2] = 5.0 * lam

        off1[0] = -2.0 * lam
        off1[-1] = -2.0 * lam
        if L > 3:
            off1[1:-1] = -4.0 * lam

    if L >= 4:
        off2[:] = lam

    return main, off1, off2


def asymmetric_least_squares_continuum(
    y: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
    lam: float = 1e7,
    p: float = 0.5,
    niter: int = 10,
    eps: float = 1e-6,
    return_weights: bool = False,
):
    """
    Fit a smooth asymmetric-least-squares baseline to a spectrum (Eilers 2003).

    The baseline ``z`` minimizes a weighted sum of squared residuals to ``y``
    plus a roughness penalty ``lam * ||D z||^2`` on the second difference of
    ``z``, with asymmetric weights that penalize the baseline lying above the
    data (``p``) much less than lying below it (``1 - p``). Iterating this
    reweighted least squares drives the baseline toward the lower envelope of
    the data by an amount controlled by ``p``, which is the desired behavior
    for continuum estimation when most deviations from the continuum are
    absorption (troughs), not emission (peaks). The linear system at each
    iteration is banded (bandwidth 4) and solved directly with
    ``scipy.linalg.solve_banded``, which is O(n) per iteration.

    Parameters
    ----------
    y : ndarray
        1-D flux array to fit the continuum baseline to.
    base_mask : ndarray of bool, optional
        Boolean mask, same length as `y`; True marks pixels that
        participate in the fit as continuum (e.g. non-absorption-line,
        non-emission-line pixels). Pixels outside the mask get only the
        floor weight `eps` and thus barely constrain the baseline directly,
        but the roughness penalty still interpolates the baseline smoothly
        through them. Defaults to all pixels participating.
    lam : float, optional
        Roughness-penalty strength (denoted ``als_lam`` in `FitConfig`).
        Larger values produce a smoother, stiffer baseline; smaller values
        let the baseline track the data more closely.
    p : float, optional
        Asymmetry weight (denoted ``als_p`` in `FitConfig`), in (0, 1).
        Small values (e.g. 0.001-0.01) push the baseline toward the upper
        envelope of the noise so that absorption troughs sit below it. A
        value of 0.5 is symmetric (ordinary smoothing spline).
    niter : int, optional
        Number of asymmetric-reweighting iterations.
    eps : float, optional
        Minimum weight floor applied everywhere, preventing the banded
        system from becoming singular where `base_mask` is False or where
        the reweighting would otherwise assign exactly zero weight.
    return_weights : bool, optional
        If True, also return the final iteration's weight vector and the
        precomputed penalty bands, so the caller (e.g. the Hutchinson trace
        estimator used for the BIC-like model-selection score) can compute
        ``trace(H_als)`` without re-running the ALS iteration from scratch.

    Returns
    -------
    ndarray or tuple
        If `return_weights` is False (default), returns the fitted
        continuum baseline `z` as a 1-D ndarray the same length as `y`.
        If `return_weights` is True, returns
        ``(z, w_final, dtd_main, dtd_off1, dtd_off2)`` where `w_final` is
        the converged weight vector and `dtd_main`/`dtd_off1`/`dtd_off2`
        are the precomputed ``lam * D'D`` band arrays.

    Notes
    -----
    The ``lam * D'D`` penalty bands depend only on `lam` and the array
    length, not on the data, so they are precomputed once by
    `_als_dtd_bands` outside the reweighting loop; only the diagonal
    weight vector `w` is rebuilt each iteration. If the banded solve fails
    (e.g. due to a degenerate mask), this falls back to a cubic smoothing
    spline through the unmasked, finite pixels, or to a constant (median)
    baseline if fewer than 6 good pixels remain.
    """
    from scipy.interpolate import UnivariateSpline
    y = np.asarray(y, float)
    L = y.size
    base_mask = np.ones(L, dtype=bool) if base_mask is None else np.asarray(base_mask, bool)

    # Precompute penalty bands once — only w changes each iteration.
    dtd_main, dtd_off1, dtd_off2 = _als_dtd_bands(L, lam)

    w = np.where(base_mask, 1.0, eps).astype(float)
    z = np.copy(y)
    for _ in range(niter):
        ab = np.zeros((5, L), dtype=float)
        ab[2] = w + dtd_main
        if L >= 2:
            ab[1, 1:] = dtd_off1
            ab[3, :-1] = dtd_off1
        if L >= 3:
            ab[0, 2:] = dtd_off2
            ab[4, :-2] = dtd_off2
        try:
            z = solve_banded((2, 2), ab, w * y, overwrite_ab=True, overwrite_b=True, check_finite=False)
        except Exception:
            good = base_mask & np.isfinite(y)
            if np.count_nonzero(good) < 6:
                fallback = np.full(L, np.nanmedian(y))
                if return_weights:
                    return fallback, w, dtd_main, dtd_off1, dtd_off2
                return fallback
            spline = UnivariateSpline(
                np.flatnonzero(good), y[good], k=3,
                s=max(1.0, np.count_nonzero(good) * 0.1),
            )
            fallback = spline(np.arange(L))
            if return_weights:
                return fallback, w, dtd_main, dtd_off1, dtd_off2
            return fallback
        w = np.where(
            y > z,
            base_mask * p + (~base_mask) * eps,
            base_mask * (1 - p) + (~base_mask) * eps,
        )
        w = np.maximum(w, eps)
    if return_weights:
        return z, w, dtd_main, dtd_off1, dtd_off2
    return z


def _als_hutchinson_trace(
    w: np.ndarray,
    dtd_main: np.ndarray,
    dtd_off1: np.ndarray,
    dtd_off2: np.ndarray,
    n_probes: int = 25,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate the ALS smoother's effective degrees of freedom via Hutchinson's
    stochastic trace estimator: trace(H_als) = trace((W + lam*D'D)^{-1} W).

    For each random Rademacher probe vector v ~ {-1, +1}^L,
    ``E[v^T (W + lam*D'D)^{-1} W v] = trace((W + lam*D'D)^{-1} W)``, so
    averaging (here, taking the median of) `n_probes` such quadratic forms
    gives an unbiased, low-variance estimate of the trace without forming
    the dense inverse. This ``k_eff`` is the "number of effective
    parameters" of the ALS baseline, used in the BIC-like model-selection
    penalty in `optimize_als_hyperparams_for_target` so that low-lambda
    (wigglier, higher-k_eff) continua are penalized appropriately. Reuses
    the same banded (bandwidth-4) structure as
    `asymmetric_least_squares_continuum`, so each probe costs one O(n)
    banded solve.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    L = w.size
    estimates = np.empty(n_probes, dtype=float)
    for k in range(n_probes):
        v = rng.choice(np.array([-1.0, 1.0]), size=L)
        rhs = w * v  # W @ v
        ab = np.zeros((5, L), dtype=float)
        ab[2] = w + dtd_main
        if L >= 2:
            ab[1, 1:] = dtd_off1
            ab[3, :-1] = dtd_off1
        if L >= 3:
            ab[0, 2:] = dtd_off2
            ab[4, :-2] = dtd_off2
        try:
            z = solve_banded((2, 2), ab, rhs, check_finite=False)
            estimates[k] = float(v @ z)
        except Exception:
            estimates[k] = float(L) / 2.0  # safe fallback: half of pixels
    # Median is more robust than mean against occasional bad solves.
    return float(np.median(estimates))

def score_als_target(
    y: np.ndarray,
    yerr: np.ndarray,
    continuum: np.ndarray,
    eval_mask: np.ndarray,
    *,
    templates: Optional[np.ndarray] = None,
    template_selection: str = "lsq",
    score_data: Optional[np.ndarray] = None,
    score_err: Optional[np.ndarray] = None,
    score_model0: Optional[np.ndarray] = None,
) -> tuple[float, int, str, int]:
    """
    Score how well a candidate ALS continuum baseline fits the data.

    Used by `optimize_als_hyperparams_for_target` to rank ``(als_lam,
    als_p)`` hyperparameter combinations. Automatically selects one of
    three scoring modes depending on which optional arguments are
    supplied, since the "right" comparison changes depending on whether
    the continuum is being fit before any LOSVD model exists (initial
    continuum) or refined against an existing model (outer-loop update).

    Parameters
    ----------
    y : ndarray
        Flux array the candidate continuum is being scored against
        (ignored in model-score mode).
    yerr : ndarray
        Flux uncertainty array corresponding to `y` (ignored in
        model-score mode).
    continuum : ndarray
        Candidate ALS continuum baseline to score.
    eval_mask : ndarray of bool
        Boolean mask selecting which pixels contribute to the score
        (typically the continuum-good/base pixel mask).
    templates : ndarray, optional
        2-D array of shape ``(npix, ntemplates)`` of stellar template
        fluxes. If supplied (and model-score mode is not active), the
        continuum-normalized spectrum is fit by a linear (or NNLS)
        combination of templates and scored against that fit. Useful
        mainly for scoring the very first, LOSVD-model-free continuum
        estimate.
    template_selection : str, optional
        ``"lsq"`` for ordinary (possibly negative-weighted) least squares
        or ``"nnls"`` for non-negative least-squares template weights.
    score_data : ndarray, optional
        Observed flux for the physically correct outer-update scoring mode
        (see Notes). Must be supplied together with `score_err` and
        `score_model0`.
    score_err : ndarray, optional
        Flux uncertainty corresponding to `score_data`.
    score_model0 : ndarray, optional
        Current LOSVD/template model evaluated *without* the continuum
        multiplied in, i.e. the shape kinextract expects the continuum to
        multiply onto to reproduce `score_data`.

    Returns
    -------
    chi2_red : float
        Mean squared standardized residual (reduced chi-squared) of the
        winning scoring mode; ``np.inf`` if too few valid pixels remain.
    n : int
        Number of pixels that contributed to `chi2_red`.
    score_kind : str
        Which mode was used: ``"model"``, ``"template"``, or ``"target"``.
    nt : int
        Number of templates used in template-score mode (0 otherwise).

    Notes
    -----
    Mode 1 (model-score, used during ALS outer updates): if `score_data`,
    `score_err`, and `score_model0` are all supplied, scores
    ``score_data ~= score_model0 * continuum``. This is the physically
    correct comparison once an LOSVD/template model exists, since it
    scores the continuum in the same multiplicative role it plays in the
    full spectral model.

    Mode 2 (template-score, used for the very first continuum estimate):
    if `templates` is supplied and mode 1's arguments are absent, scores
    the continuum-corrected spectrum ``y / continuum`` against a fitted
    linear combination of templates.

    Mode 3 (continuum-target, fallback): otherwise scores ``y ~=
    continuum`` directly using `yerr`. This is the least physically
    motivated mode (it rewards continua that hug the raw, possibly
    absorption-contaminated flux) and is used only when neither of the
    above is available.
    """
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    continuum = np.asarray(continuum, float)
    eval_mask = np.asarray(eval_mask, bool)

    # ------------------------------------------------------------
    # Mode 1: physically correct ALS update score:
    # data ~= model_without_continuum * continuum
    # ------------------------------------------------------------
    if (
        score_data is not None
        and score_err is not None
        and score_model0 is not None
    ):
        data = np.asarray(score_data, float)
        err = np.asarray(score_err, float)
        model0 = np.asarray(score_model0, float)

        m = (
            eval_mask
            & np.isfinite(data)
            & np.isfinite(err)
            & np.isfinite(model0)
            & np.isfinite(continuum)
            & (err > 0.0)
            & (err < BIG)
        )

        n = int(np.count_nonzero(m))
        if n < 2:
            return np.inf, 0, "model", 0

        r = (data[m] - model0[m] * continuum[m]) / err[m]
        r = r[np.isfinite(r)]

        if r.size < 2:
            return np.inf, 0, "model", 0

        return float(np.mean(r * r)), int(r.size), "model", 0

    # ------------------------------------------------------------
    # Mode 2: template-informed initial continuum score
    # ------------------------------------------------------------
    if templates is not None and np.size(templates) > 0:
        try:
            T = np.asarray(templates, float)

            if T.ndim == 2 and T.shape[0] == y.size:
                y_corr = np.full_like(y, np.nan, dtype=float)
                yerr_corr = np.full_like(yerr, np.nan, dtype=float)

                okc = np.isfinite(continuum) & (np.abs(continuum) > 0.0)
                y_corr[okc] = y[okc] / continuum[okc]
                yerr_corr[okc] = yerr[okc] / np.abs(continuum[okc])

                m = (
                    eval_mask
                    & np.isfinite(y_corr)
                    & np.isfinite(yerr_corr)
                    & (yerr_corr > 0.0)
                    & np.all(np.isfinite(T), axis=1)
                )

                n = int(np.count_nonzero(m))
                nt = int(T.shape[1])

                if n >= max(10, nt):
                    Tm = T[m]
                    ym = y_corr[m]
                    em = yerr_corr[m]

                    if template_selection == "nnls":
                        from scipy.optimize import nnls
                        w, _ = nnls(Tm / em[:, None], ym / em)
                    else:
                        w, *_ = np.linalg.lstsq(
                            Tm / em[:, None],
                            ym / em,
                            rcond=None,
                        )

                    model = Tm.dot(w)
                    r = (ym - model) / em
                    r = r[np.isfinite(r)]

                    if r.size >= 2:
                        return float(np.mean(r * r)), int(r.size), "template", nt

        except Exception:
            pass

    # ------------------------------------------------------------
    # Mode 3: direct continuum-target score
    # ------------------------------------------------------------
    m = (
        eval_mask
        & np.isfinite(y)
        & np.isfinite(yerr)
        & (yerr > 0.0)
        & np.isfinite(continuum)
    )

    n = int(np.count_nonzero(m))
    if n < 2:
        return np.inf, 0, "target", 0

    r = (y[m] - continuum[m]) / yerr[m]
    r = r[np.isfinite(r)]

    if r.size < 2:
        return np.inf, 0, "target", 0

    return float(np.mean(r * r)), int(r.size), "target", 0


def optimize_als_hyperparams_for_target(
    y: np.ndarray,
    yerr: np.ndarray,
    base_mask: np.ndarray,
    *,
    lam_grid: Optional[np.ndarray] = None,
    p_grid: Optional[np.ndarray] = None,
    lam_grid_fine_n: int = 5,
    p_grid_fine_n: int = 5,
    als_niter: int = 20,
    eps: float = 1e-6,
    verbose: bool = False,
    templates: Optional[np.ndarray] = None,
    template_selection: str = "lsq",
    score_data: Optional[np.ndarray] = None,
    score_err: Optional[np.ndarray] = None,
    score_model0: Optional[np.ndarray] = None,
    use_bic: bool = False,
    chisq_floor: float = 0.0,
    hutchinson_probes: int = 25,
    bic_dof_penalty: float = 1.0,
) -> tuple[np.ndarray, float, float, list]:
    """
    Search a grid of ALS smoothing hyperparameters and return the best fit.

    Performs a two-stage (coarse, then fine) grid search over
    ``als_lam`` (roughness penalty) and ``als_p`` (asymmetry weight),
    refitting `asymmetric_least_squares_continuum` at each grid point and
    ranking candidates by `score_als_target`'s reduced chi-squared, or by
    a BIC-like penalized score when `use_bic` is True. The fine stage
    refines the search only in the immediate neighborhood of the coarse
    best point, which is much cheaper than a single dense grid while still
    correcting for the coarse grid's resolution.

    Parameters
    ----------
    y : ndarray
        Flux (or continuum-target) array to fit the ALS baseline to.
    yerr : ndarray
        Corresponding flux uncertainty array.
    base_mask : ndarray of bool
        Boolean mask of continuum-good pixels passed through to
        `asymmetric_least_squares_continuum` at every grid point.
    lam_grid : ndarray, optional
        Candidate `als_lam` values for the coarse search stage. Defaults
        to a log-spaced grid from 1e2 to 1e9.
    p_grid : ndarray, optional
        Candidate `als_p` values for the coarse search stage. If it
        contains a single value, `als_p` is held fixed at that value
        (no p search is performed).
    lam_grid_fine_n : int, optional
        Number of `als_lam` points in the fine (refinement) grid.
    p_grid_fine_n : int, optional
        Number of `als_p` points in the fine (refinement) grid (ignored
        if `p_grid` is fixed).
    als_niter : int, optional
        Number of ALS reweighting iterations used at each grid point.
    eps : float, optional
        Minimum ALS weight floor, passed through to
        `asymmetric_least_squares_continuum`.
    verbose : bool, optional
        If True, print a running log of each grid point's score and a
        top-10 summary of the best candidates at the end.
    templates : ndarray, optional
        Template matrix forwarded to `score_als_target` for
        template-score mode (used only for the initial continuum).
    template_selection : str, optional
        ``"lsq"`` or ``"nnls"``, forwarded to `score_als_target`.
    score_data, score_err, score_model0 : ndarray, optional
        Forwarded to `score_als_target` for the physically correct
        outer-update scoring mode (``score_data ~= score_model0 *
        continuum``); supply these during ALS outer updates rather than
        letting the search fall back to fitting templates to the
        continuum-corrected target.
    use_bic : bool, optional
        If True, rank candidates by
        ``score = chi2_total + bic_dof_penalty * k_eff * ln(n)``, where
        ``chi2_total = chi2_red * n``, ``k_eff`` is the ALS smoother's
        effective degrees of freedom (see `_als_hutchinson_trace`), and
        `n` is the number of good pixels. This BIC-like penalty
        discourages low-``als_lam`` (wigglier, higher-``k_eff``) continua
        that would otherwise always win on raw chi-squared alone. If
        False, candidates are ranked purely by `chi2_red`.
    chisq_floor : float, optional
        If greater than 0, hard-reject any ``(lam, p)`` whose `chi2_red`
        falls below this floor, regardless of `use_bic`. Continua that
        absorb more variance than the noise budget allows produce
        ``chi2_red < 1``; setting e.g. ``chisq_floor=0.9`` prevents such
        overfit continua from ever being selected. Disabled (0.0) by
        default.
    hutchinson_probes : int, optional
        Number of random probe vectors used by the Hutchinson trace
        estimator when `use_bic` is True.
    bic_dof_penalty : float, optional
        Multiplicative weight on the ``k_eff * ln(n)`` BIC penalty term;
        larger values favor smoother (higher-``als_lam``) continua more
        strongly.

    Returns
    -------
    best_cont : ndarray
        ALS continuum baseline evaluated at the best-scoring
        ``(als_lam, als_p)``.
    best_lam : float
        Selected roughness-penalty strength.
    best_p : float
        Selected asymmetry weight.
    records : list of dict
        One diagnostic record per grid point evaluated (stage, lam, p,
        chi2_red, score, score_kind, n, k_eff, ...), useful for
        post-hoc inspection of the search.
    """
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    base_mask = np.asarray(base_mask, bool)

    if lam_grid is None:
        lam_grid = np.logspace(2, 9, 8)
    if p_grid is None:
        p_grid = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])

    lam_grid = np.atleast_1d(np.asarray(lam_grid, float))
    p_grid = np.atleast_1d(np.asarray(p_grid, float))

    p_fixed = p_grid.size == 1
    fixed_p = float(p_grid.item()) if p_fixed else None

    records: list[dict] = []

    best_score = np.inf
    best_cont: Optional[np.ndarray] = None
    best_lam = float(lam_grid[len(lam_grid) // 2])
    best_p = float(fixed_p if fixed_p is not None else p_grid[len(p_grid) // 2])

    _hutchinson_rng = np.random.default_rng(12345)

    def _eval(lam: float, p: float, stage: str):
        als_result = asymmetric_least_squares_continuum(
            y,
            base_mask=base_mask,
            lam=lam,
            p=p,
            niter=als_niter,
            eps=eps,
            return_weights=use_bic,
        )
        if use_bic:
            cont, w_final, dtd_main, dtd_off1, dtd_off2 = als_result
        else:
            cont = als_result

        chi2_red, n, score_kind, nt = score_als_target(
            y,
            yerr,
            cont,
            base_mask,
            templates=templates,
            template_selection=template_selection,
            score_data=score_data,
            score_err=score_err,
            score_model0=score_model0,
        )

        # Hard floor: reject continua that absorb more variance than the noise
        # budget allows (chi2_red < 1 signals overfitting the continuum).
        if chisq_floor > 0.0 and np.isfinite(chi2_red) and chi2_red < chisq_floor:
            score = np.inf
            k_eff = np.nan
        elif n < 2 or not np.isfinite(chi2_red) or chi2_red <= 0.0:
            score = np.inf
            k_eff = np.nan
        elif use_bic:
            k_eff = _als_hutchinson_trace(
                w_final, dtd_main, dtd_off1, dtd_off2,
                n_probes=hutchinson_probes,
                rng=_hutchinson_rng,
            )
            # BIC = chi2_total + bic_dof_penalty * k_eff * ln(n)
            score = float(chi2_red * n + bic_dof_penalty * k_eff * np.log(max(n, 2)))
        else:
            score = float(chi2_red)
            k_eff = np.nan

        records.append(
            dict(
                stage=stage,
                lam=float(lam),
                p=float(p),
                chi2_red=float(chi2_red),
                score=float(score),
                score_kind=score_kind,
                nt=int(nt),
                n=int(n),
                k_eff=float(k_eff) if np.isfinite(k_eff) else np.nan,
            )
        )

        if verbose:
            k_str = f"k_eff={k_eff:.1f}" if np.isfinite(k_eff) else "k_eff=n/a"
            print(
                f"  {stage:6s} "
                f"lam={lam:.3e} p={p:.4g} "
                f"score={score_kind:8s} "
                f"chi2_red={chi2_red:.6g} "
                f"{k_str} "
                f"target={score:.6g}"
            )

        return cont, score

    # Coarse grid
    if p_fixed:
        for lam in lam_grid:
            cont, score = _eval(float(lam), fixed_p, "coarse")
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_cont = cont
                best_lam = float(lam)
                best_p = fixed_p
    else:
        for lam in lam_grid:
            for p in p_grid:
                cont, score = _eval(float(lam), float(p), "coarse")
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_cont = cont
                    best_lam = float(lam)
                    best_p = float(p)

    if best_cont is None:
        best_cont = asymmetric_least_squares_continuum(
            y,
            base_mask=base_mask,
            lam=best_lam,
            p=best_p,
            niter=als_niter,
            eps=eps,
        )
        return best_cont, best_lam, best_p, records

    # Fine grid around coarse best
    li = int(np.argmin(np.abs(lam_grid - best_lam)))
    lam_lo = lam_grid[max(0, li - 1)]
    lam_hi = lam_grid[min(len(lam_grid) - 1, li + 1)]

    if lam_lo == lam_hi:
        lam_fine = np.array([best_lam], dtype=float)
    else:
        lam_fine = np.logspace(
            np.log10(lam_lo),
            np.log10(lam_hi),
            lam_grid_fine_n,
        )

    if p_fixed:
        for lam in lam_fine:
            cont, score = _eval(float(lam), fixed_p, "fine")
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_cont = cont
                best_lam = float(lam)
                best_p = fixed_p
    else:
        pi = int(np.argmin(np.abs(p_grid - best_p)))
        p_lo = p_grid[max(0, pi - 1)]
        p_hi = p_grid[min(len(p_grid) - 1, pi + 1)]

        if p_lo == p_hi:
            p_fine = np.array([best_p], dtype=float)
        else:
            p_fine = np.linspace(p_lo, p_hi, p_grid_fine_n)

        for lam in lam_fine:
            for p in p_fine:
                cont, score = _eval(float(lam), float(p), "fine")
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_cont = cont
                    best_lam = float(lam)
                    best_p = float(p)

    if verbose:
        finite_records = [r for r in records if np.isfinite(r["score"])]
        finite_records = sorted(finite_records, key=lambda r: r["score"])

        print("\nTop ALS candidates:")
        for r in finite_records[:10]:
            print(
                f"  {r['stage']:6s} "
                f"lam={r['lam']:.3e} "
                f"p={r['p']:.4g} "
                f"score={r['score_kind']:8s} "
                f"chi2_red={r['chi2_red']:.6g}"
            )

        print(
            f"Selected ALS: lam={best_lam:.3e}, "
            f"p={best_p:.4g}, score={best_score:.6g}\n"
        )

    return best_cont, best_lam, best_p, records


def fit_als_target(
    y: np.ndarray,
    yerr: np.ndarray,
    base_mask: np.ndarray,
    cfg,
    *,
    force_fixed: bool = False,
    templates: Optional[np.ndarray] = None,
    score_data: Optional[np.ndarray] = None,
    score_err: Optional[np.ndarray] = None,
    score_model0: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, float, list]:
    """
    Fit an ALS continuum baseline to `y`, optionally optimizing hyperparameters.

    Thin dispatch wrapper: if ``cfg.als_optimize`` is True and `force_fixed`
    is False, delegates to `optimize_als_hyperparams_for_target` to search
    for the best ``(als_lam, als_p)``; otherwise fits once at the fixed
    ``cfg.als_lam``/``cfg.als_p`` values via
    `asymmetric_least_squares_continuum`.

    Parameters
    ----------
    y : ndarray
        Flux (or continuum-target) array to fit.
    yerr : ndarray
        Corresponding flux uncertainty array.
    base_mask : ndarray of bool
        Boolean mask of continuum-good pixels.
    cfg : FitConfig
        Fit configuration; supplies ``als_optimize``, ``als_lam``,
        ``als_p``, ``als_lam_grid``, ``als_p_grid``, and related ALS
        settings.
    force_fixed : bool, optional
        If True, skip hyperparameter optimization even if
        ``cfg.als_optimize`` is True, and fit once at the current
        ``cfg.als_lam``/``cfg.als_p``. Used for cheaper refits after the
        hyperparameters have already been established (e.g. during
        absorption-cleaning iterations).
    templates : ndarray, optional
        Forwarded to `optimize_als_hyperparams_for_target` for
        template-score mode.
    score_data, score_err, score_model0 : ndarray, optional
        Forwarded to `optimize_als_hyperparams_for_target` so that,
        during ALS outer updates, the continuum is scored against the
        current spectral model (``score_data ~= score_model0 *
        continuum``) rather than by fitting templates directly.

    Returns
    -------
    continuum : ndarray
        Fitted ALS continuum baseline.
    lam : float
        Roughness-penalty strength used (optimized or fixed).
    p : float
        Asymmetry weight used (optimized or fixed).
    records : list of dict
        Diagnostic records from the hyperparameter search, or an empty
        list if the fixed-hyperparameter path was taken.
    """
    if cfg.als_optimize and not force_fixed:
        lam_grid = np.asarray(cfg.als_lam_grid, float)

        if getattr(cfg, "als_lambda_floor", None) is not None:
            floor = float(cfg.als_lambda_floor)
            lam_grid = lam_grid[lam_grid >= floor]
            if lam_grid.size == 0:
                raise ValueError(
                    "als_lambda_floor removed all ALS lambda grid values."
                )

        log(
            "ALS optimize: "
            f"lam_grid={tuple(np.asarray(lam_grid, float))} "
            f"p_grid={tuple(np.asarray(cfg.als_p_grid, float))} "
            f"templates_shape={None if templates is None else templates.shape} "
            f"model_score={score_model0 is not None}"
        )

        return optimize_als_hyperparams_for_target(
            y,
            yerr,
            base_mask,
            lam_grid=lam_grid,
            p_grid=np.asarray(cfg.als_p_grid, float),
            lam_grid_fine_n=cfg.als_lam_grid_fine_n,
            p_grid_fine_n=cfg.als_p_grid_fine_n,
            als_niter=cfg.als_niter,
            eps=cfg.als_eps,
            verbose=cfg.als_opt_verbose,
            templates=templates,
            template_selection=getattr(cfg, "als_template_selection", "lsq"),
            score_data=score_data,
            score_err=score_err,
            score_model0=score_model0,
            use_bic=getattr(cfg, "als_use_bic", False),
            chisq_floor=getattr(cfg, "als_chisq_floor", 0.0),
            hutchinson_probes=getattr(cfg, "als_hutchinson_probes", 25),
            bic_dof_penalty=getattr(cfg, "als_bic_dof_penalty", 1.0),
        )

    cont = asymmetric_least_squares_continuum(
        y,
        base_mask=base_mask,
        lam=cfg.als_lam,
        p=cfg.als_p,
        niter=cfg.als_niter,
        eps=cfg.als_eps,
    )

    return cont, float(cfg.als_lam), float(cfg.als_p), []


def grow_boolean_mask(mask: np.ndarray, grow_pix: int) -> np.ndarray:
    """
    Dilate a boolean mask by a fixed number of pixels on each side.

    Used to extend rejection/exclusion masks (e.g. absorption-clean
    rejections or emission-line masks) a few pixels beyond the originally
    flagged pixels, so that line wings adjacent to a flagged pixel are
    also excluded rather than leaving a single unmasked pixel at the edge
    of a feature.

    Parameters
    ----------
    mask : ndarray of bool
        Input mask where True means rejected/masked/excluded.
    grow_pix : int
        Number of pixels by which to grow the mask on each side. Values
        ``<= 0`` return a copy of `mask` unchanged.

    Returns
    -------
    ndarray of bool
        Grown mask, same shape as `mask`; True wherever any pixel within
        `grow_pix` samples was True in the input.
    """
    mask = np.asarray(mask, bool)
    if grow_pix <= 0 or mask.size == 0:
        return mask.copy()
    kernel = np.ones(2 * grow_pix + 1, dtype=int)
    grown = np.convolve(mask.astype(int), kernel, mode="same") > 0
    return grown


def grow_boolean_mask_A(mask: np.ndarray, x: np.ndarray, grow_A: float) -> np.ndarray:
    """
    Dilate a boolean mask by approximately a fixed wavelength on each side.

    Converts `grow_A` (in Angstroms) to a pixel count using the median
    pixel spacing of `x`, then calls `grow_boolean_mask`. This is the
    wavelength-aware counterpart of `grow_boolean_mask`, useful when the
    desired growth is naturally expressed in Angstroms (e.g. "grow the
    emission mask by 4 Å") rather than in pixels.

    Parameters
    ----------
    mask : ndarray of bool
        Input mask where True means rejected/masked/excluded.
    x : ndarray
        Wavelength array (Å) corresponding to `mask`, used only to
        estimate the pixel scale.
    grow_A : float
        Approximate growth distance in Angstroms on each side. Values
        ``<= 0`` return a copy of `mask` unchanged.

    Returns
    -------
    ndarray of bool
        Grown mask, same shape as `mask`.
    """
    if grow_A <= 0:
        return np.asarray(mask, bool).copy()
    dx = np.diff(np.asarray(x, float))
    dx = dx[np.isfinite(dx)]
    if dx.size == 0:
        grow_pix = 0
    else:
        step = float(np.median(np.abs(dx)))
        grow_pix = int(np.ceil(grow_A / max(step, 1e-12)))

    return grow_boolean_mask(mask, grow_pix)


def fit_als_target_absorption_clean(
    y: np.ndarray,
    yerr: np.ndarray,
    base_mask: np.ndarray,
    cfg,
    *,
    x: Optional[np.ndarray] = None,
    force_fixed: bool = False,
    skip_abs_clean: bool = False,
    templates: Optional[np.ndarray] = None,
    score_data: Optional[np.ndarray] = None,
    score_err: Optional[np.ndarray] = None,
    score_model0: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, float, list, np.ndarray]:
    """
    Fit the ALS continuum with pPXF-like iterative absorption-pixel rejection.

    This is the "good pixel" continuum-masking system referenced in the
    module docstring: it identifies which pixels the ALS baseline should
    treat as continuum versus absorption, so that real absorption troughs
    (e.g. the Ca II triplet) do not pull the fitted baseline down toward
    the trough flux. The procedure is:

      1. Fit the ALS baseline using the supplied `base_mask` (optionally
         performing full hyperparameter optimization via `fit_als_target`).
      2. Identify pixels with flux significantly below the fitted
         continuum (candidate absorption pixels), using a robust sigma
         computed from `robust_sigma`.
      3. Grow that rejection mask by `cfg.als_abs_clean_grow_A` Angstroms
         so absorption-line wings are also excluded.
      4. Refit the ALS baseline (at fixed lambda/p, for speed) using the
         updated continuum-good mask, and repeat for
         ``cfg.als_abs_clean_iter`` iterations or until convergence.
      5. Optionally re-optimize lambda/p once more on the final cleaned
         mask (``cfg.als_abs_clean_reoptimize_final``).

    This masking is for continuum estimation only: it must not be used to
    exclude absorption lines from the LOSVD/template fit itself, since
    those lines carry the kinematic information being recovered.

    Parameters
    ----------
    y : ndarray
        Flux (or continuum-target) array to fit.
    yerr : ndarray
        Corresponding flux uncertainty array.
    base_mask : ndarray of bool
        Initial boolean mask of continuum-candidate pixels (before
        absorption-pixel rejection).
    cfg : FitConfig
        Fit configuration; supplies ``als_abs_clean``,
        ``als_abs_clean_iter``, ``als_abs_clean_sigma``,
        ``als_abs_clean_grow_A``, ``als_abs_clean_two_sided``,
        ``als_abs_clean_minpix``, ``als_abs_clean_reoptimize_final``, and
        the general ALS settings used by `fit_als_target`.
    x : ndarray, optional
        Wavelength array (Å), used to convert `cfg.als_abs_clean_grow_A`
        from Angstroms to pixels when growing the rejection mask. If
        omitted, no wavelength-based growth is applied.
    force_fixed : bool, optional
        If True, skip hyperparameter optimization on the first fit and
        use the fixed ``cfg.als_lam``/``cfg.als_p`` values.
    skip_abs_clean : bool, optional
        If True, skip absorption-pixel rejection entirely and return
        after the first ALS fit (equivalent to `cfg.als_abs_clean` being
        False).
    templates : ndarray, optional
        Forwarded to `fit_als_target` for template-score mode.
    score_data, score_err, score_model0 : ndarray, optional
        Forwarded to `fit_als_target` so outer-update continuum refits
        are scored against the current spectral model.

    Returns
    -------
    continuum : ndarray
        Final ALS continuum baseline.
    lam : float
        Roughness-penalty strength used for the final fit.
    p : float
        Asymmetry parameter used for the final fit.
    records : list of dict
        Diagnostic records from every fit and absorption-clean iteration.
    final_base_mask : ndarray of bool
        Final continuum-good-pixel mask after absorption-pixel rejection.
    """
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    current_base = np.asarray(base_mask, bool).copy()
    records_all: list = []
    # First fit. This may do full hyperparameter optimisation.
    if force_fixed:
        cont = asymmetric_least_squares_continuum(
            y,
            base_mask=current_base,
            lam=cfg.als_lam,
            p=cfg.als_p,
            niter=cfg.als_niter,
            eps=cfg.als_eps,
        )
        lam = float(cfg.als_lam)
        p = float(cfg.als_p)
        records = []
    else:
        cont, lam, p, records = fit_als_target(
            y,
            yerr,
            current_base,
            cfg,
            templates=templates,
            score_data=score_data,
            score_err=score_err,
            score_model0=score_model0,
        )
    records_all.extend(records)
    if skip_abs_clean or not getattr(cfg, "als_abs_clean", False):
        return cont, float(lam), float(p), records_all, current_base
    # Iteratively reject absorption-like continuum outliers.
    # After the first fit, use fixed lam/p for speed and stability.
    n_iter = int(getattr(cfg, "als_abs_clean_iter", 2))
    clip_sigma = float(getattr(cfg, "als_abs_clean_sigma", 2.5))
    grow_A = float(getattr(cfg, "als_abs_clean_grow_A", 4.0))
    two_sided = bool(getattr(cfg, "als_abs_clean_two_sided", False))
    minpix = int(getattr(cfg, "als_abs_clean_minpix", 50))
    for it in range(n_iter):
        good = (
            current_base
            & np.isfinite(y)
            & np.isfinite(yerr)
            & (yerr > 0)
            & np.isfinite(cont)
        )
        if np.count_nonzero(good) < minpix:
            log(
                f"ALS absorption-clean iter {it + 1}: too few good pixels; "
                f"stopping."
            )
            break
        resid = np.full_like(y, np.nan, dtype=float)
        resid[good] = (y[good] - cont[good]) / yerr[good]
        sig = robust_sigma(resid[good])
        if not np.isfinite(sig) or sig <= 0:
            log(
                f"ALS absorption-clean iter {it + 1}: invalid residual sigma; "
                f"stopping."
            )
            break
        # Absorption-like pixels are below the continuum.
        reject = good & (resid < -clip_sigma * sig)
        # Optional: reject positive outliers too, e.g. sky residuals/emission.
        if two_sided:
            reject |= good & (resid > clip_sigma * sig)
        if x is not None and grow_A > 0:
            reject = grow_boolean_mask_A(reject, x, grow_A)
        new_base = current_base & ~reject
        n_rej = int(np.count_nonzero(current_base & ~new_base))
        n_new = int(np.count_nonzero(new_base))
        log(
            f"ALS absorption-clean iter {it + 1}: "
            f"sigma={sig:.4g}, rejected={n_rej}, base_pixels={n_new}"
        )
        if n_rej == 0:
            break
        if n_new < minpix:
            log(
                f"ALS absorption-clean iter {it + 1}: would leave only "
                f"{n_new} pixels; stopping."
            )
            break
        if np.array_equal(new_base, current_base):
            break
        current_base = new_base
        # Refit with fixed lam/p after rejection.
        cont = asymmetric_least_squares_continuum(
            y,
            base_mask=current_base,
            lam=float(lam),
            p=float(p),
            niter=cfg.als_niter,
            eps=cfg.als_eps,
        )
        records_all.append(
            {
                "stage": "abs_clean",
                "iter": it + 1,
                "lam": float(lam),
                "p": float(p),
                "sigma": float(sig),
                "n_rejected": n_rej,
                "n_base": n_new,
            }
        )

    # After absorption-cleaning changes the continuum-good mask,
    # optionally re-optimize lambda/p once on the final cleaned mask.
    if (
        getattr(cfg, "als_abs_clean_reoptimize_final", True)
        and getattr(cfg, "als_optimize", False)
        and not force_fixed
    ):
        cont2, lam2, p2, records2 = fit_als_target(
            y,
            yerr,
            current_base,
            cfg,
            templates=templates,
            score_data=score_data,
            score_err=score_err,
            score_model0=score_model0,
        )

        cont = cont2
        lam = float(lam2)
        p = float(p2)
        records_all.extend(records2)

        records_all.append(
            {
                "stage": "abs_clean_final_reopt",
                "lam": float(lam),
                "p": float(p),
                "n_base": int(np.count_nonzero(current_base)),
            }
        )

    return cont, float(lam), float(p), records_all, current_base


# =============================================================================
# Section 6 - ALS continuum within the LOSVD fit
# =============================================================================

# Common rest-frame features in the 8300-8900 Å region.
#
# These are intended for ALS continuum masking only.
# They should not remove pixels from the LOSVD/template fit.
DEFAULT_CAI_REGION_LINES = (
    # H I Paschen series, approximate rest air wavelengths in Å.
    ("paschen", "H I Pa25", 8323.42),
    ("paschen", "H I Pa24", 8333.78),
    ("paschen", "H I Pa23", 8345.55),
    ("paschen", "H I Pa22", 8359.00),
    ("paschen", "H I Pa21", 8374.48),
    ("paschen", "H I Pa20", 8392.40),
    ("paschen", "H I Pa19", 8413.32),
    ("paschen", "H I Pa18", 8437.96),
    ("paschen", "H I Pa17", 8467.25),
    ("paschen", "H I Pa16", 8502.49),
    ("paschen", "H I Pa15", 8545.38),
    ("paschen", "H I Pa14", 8598.39),
    ("paschen", "H I Pa13", 8665.02),
    ("paschen", "H I Pa12", 8750.47),
    ("paschen", "H I Pa11", 8862.79),

    # Ca II triplet.
    ("abs", "Ca II", 8498.02),
    ("abs", "Ca II", 8542.09),
    ("abs", "Ca II", 8662.14),

    # Fe I stellar absorption — wavelengths from Kurucz/VALD line lists,
    # selected for significant equivalent width in K-giant spectra.
    ("abs", "Fe I",          8327.06),
    ("abs", "Fe I",          8346.51),
    ("abs", "Fe I",          8387.77),
    ("abs", "Fe I / Ni I",   8471.84),
    ("abs", "Fe I",          8514.08),
    ("abs", "Fe I",          8536.87),
    ("abs", "Fe I",          8582.26),
    ("abs", "Fe I",          8611.80),
    ("abs", "Fe I",          8621.61),
    ("abs", "Fe I",          8643.63),
    ("abs", "Fe I",          8688.63),
    ("abs", "Fe I",          8699.46),
    ("abs", "Fe I",          8710.39),
    ("abs", "Fe I",          8824.22),
    ("abs", "Fe I",          8838.43),

    # Ti I — very strong in K-M giants; wavelengths from Kurucz/VALD.
    ("abs", "Ti I",          8357.68),
    ("abs", "Ti I",          8382.54),
    ("abs", "Ti I",          8408.90),
    ("abs", "Ti I",          8412.36),
    ("abs", "Ti I",          8435.65),
    ("abs", "Ti I",          8468.40),
    ("abs", "Ti I",          8510.74),
    ("abs", "Ti I",          8518.35),
    ("abs", "Ti I",          8521.13),
    ("abs", "Ti I",          8529.44),
    ("abs", "Ti I",          8556.98),
    ("abs", "Ti I",          8666.70),
    ("abs", "Ti I",          8682.98),
    ("abs", "Ti I",          8734.46),
    ("abs", "Ti I",          8759.84),
    ("abs", "Ti I",          8786.04),

    # Ca I
    ("abs", "Ca I",          8539.15),

    # Mg I
    ("abs", "Mg I",          8806.76),

    # Si I
    ("abs", "Si I",          8728.01),
    ("abs", "Si I",          8742.45),

    # Ni I
    ("abs", "Ni I",          8361.92),
    ("abs", "Ni I",          8476.85),
    ("abs", "Ni I",          8692.85),
    ("abs", "Ni I",          8703.25),

    # Cr I
    ("abs", "Cr I",          8308.45),
    ("abs", "Cr I",          8344.09),
    ("abs", "Cr I",          8465.42),

    # Ba II — strong in luminous giants due to s-process enhancement
    ("abs", "Ba II",         8559.60),

    # V I (Vanadium) — significant in K giants
    ("abs", "V I",           8675.20),

    # TiO δ (0,0) bandhead — strong in M2+ giants; sits just longward of CaT
    ("abs", r"TiO $\delta$",         8859.0),

    # Near-IR Ca I
    ("abs", "Ca I",          9257.0),   # moderately strong in K giants

    # CN red system A²Π–X²Σ⁺ band; significant in K giants and CN-rich stars
    ("abs", "CN",            9350.0),   # (0,0) band onset ~9100–9400 Å

    # Paschen series beyond Pa11: Pa8/Pa7 are the strongest in 8900–9500 Å
    ("paschen", "H I Pa9",   8863.0),   # Pa9 (n=12→3); often blended with TiO δ
    ("paschen", "H I Pa8",   9015.0),   # Pa8 (n=11→3)
    ("paschen", "H I Pa7",   9229.0),   # Pa-δ (n=10→3); strongest Paschen in this window

    # Possible emission/nebular lines or emission-like contaminants.
    ("em", "O I",            8446.36),
    ("em", "[Cl II]",        8578.70),
    ("em", "[Fe II]",        8616.95),
    ("em", "[N I]",          8680.28),
    ("em", "[N I]",          8683.40),
    ("em", "[C I]",          8727.13),
    ("em", "[S III]",        9068.60),
    ("em", "[S III]",        9530.60),
)

# Strong stellar absorption features in the Mg b / green region (rest air, Å).
# Tags: "abs" = stellar absorption, "em" = nebular/emission-line contaminant.
DEFAULT_MGB_REGION_LINES = (
    # Balmer series
    ("abs", "H I Hbeta",     4861.33),
    ("abs", "H I Hgamma",    4340.47),

    # Fe I / blends — strong in old stellar populations; Lick Fe5015 index at 5015
    ("abs", "Fe I",          5015.20),  # Fe5015 Lick index; Fe I + Ti I blend
    ("abs", "Fe I / blend",  4957.60),
    ("abs", "Fe I / blend",  5041.10),
    ("abs", "Fe I / blend",  5056.50),
    ("abs", "Fe I / blend",  5270.40),
    ("abs", "Fe I / blend",  5328.04),
    ("abs", "Fe I / blend",  5371.49),
    ("abs", "Fe I / blend",  5397.13),
    ("abs", "Fe I / blend",  5406.00),  # Fe5406 Lick index
    ("abs", "Fe I / blend",  5446.92),

    # Mg b triplet — almost always present in early-type spectra
    ("abs", "Mg b",          5167.32),
    ("abs", "Mg b",          5172.68),
    ("abs", "Mg b",          5183.60),

    # Mg I / MgH
    ("abs", "Mg I",          5528.41),
    ("abs", "MgH",           5138.70),  # A²Π-X²Σ⁺ (0,0) bandhead; gravity-sensitive

    # Common nebular emitters in this window (should be clipped, not protected)
    ("em", "[O III]",        4958.91),
    ("em", "[O III]",        5006.84),
    ("em", "He I",           5015.68),
    ("em", "[N I]",          5197.90),
    ("em", "[N I]",          5200.26),
)

# Stellar absorption and nebular emission features in the 5500–8300 Å gap
# between the Mg b and Ca II triplet windows.
DEFAULT_OPTICAL_RED_LINES = (
    # ── Fe I / blends (5600–6500 Å) ─────────────────────────────────────────
    # Fe5709/Fe5782 Lick indices; measurable in K giants
    ("abs", "Fe I",            5709.38),
    ("abs", "Fe I",            5780.38),  # Fe5782 Lick index
    ("abs", "Fe I",            6136.61),
    ("abs", "Fe I",            6191.56),
    ("abs", "Fe I",            6355.03),
    ("abs", "Fe I",            6393.60),
    ("abs", "Fe I",            6421.36),
    ("abs", "Fe I",            6430.85),
    ("abs", "Fe I",            7511.02),
    ("abs", "Fe I",            7531.15),

    # ── Na I D doublet — among the strongest features in K giants; very IMF-sensitive
    ("abs", "Na I D",          5889.95),
    ("abs", "Na I D",          5895.92),

    # ── Ca I — moderately strong in K/G giants ───────────────────────────────
    ("abs", "Ca I",            6122.22),
    ("abs", "Ca I",            6162.17),
    ("abs", "Ca I",            6347.23),  # stronger Ca I in K giants
    ("abs", "Ca I",            6373.48),

    # ── Ba II — s-process marker; strong in luminous K giants ────────────────
    ("abs", "Ba II",           6496.90),

    # ── Mg I ────────────────────────────────────────────────────────────────
    ("abs", "Mg I",            5528.41),

    # ── O I triplet — photospheric absorption (NOT the emission at 6300 Å) ───
    ("abs", "O I",             7771.94),
    ("abs", "O I",             7774.17),
    ("abs", "O I",             7775.39),

    # ── K I resonance doublet — gravity-sensitive; significant in K/M giants ─
    ("abs", "K I",             7664.91),
    ("abs", "K I",             7698.96),

    # ── Na I NIR doublet — IMF-sensitive indicator ───────────────────────────
    ("abs", "Na I",            8183.27),
    ("abs", "Na I",            8194.82),

    # ── TiO molecular bands (K5–M9 giants; multiple band systems) ────────────
    # γ' system (B³Π–X³Δ): bandheads at ~5847, ~6159, ~6651 Å
    ("abs", r"TiO $\gamma'$",          5847.0),   # (1,0) bandhead; onset K5 giants
    ("abs", r"TiO $\gamma'$",          6159.0),   # (0,0) bandhead; "TiO₁/₂" Lick region onset
    ("abs", r"TiO $\gamma'$",          6651.0),   # (0,1) bandhead; Lick TiO₃ region
    # γ system (A³Φ–X³Δ): prominent bandheads 7054–7676 Å
    ("abs", r"TiO $\gamma$",           7055.0),   # (0,0) bandhead; strongest optical TiO feature
    ("abs", r"TiO $\gamma$",           7087.0),   # secondary feature in (0,0) complex
    ("abs", r"TiO $\gamma$",           7126.0),   # (1,1) bandhead; Lick TiO₃/₄ region
    ("abs", r"TiO $\gamma$",           7589.0),   # (0,1) bandhead; Lick TiO₄ region
    ("abs", r"TiO $\gamma$",           7676.0),   # (1,2) bandhead
    # δ system: prominent bandhead at 8432 Å (in gap between Na I NIR and CaT)
    ("abs", r"TiO $\delta$",           8432.0),   # (1,0) bandhead

    # ── Common nebular emission lines in this window ──────────────────────────
    ("em",  "[O I]",           5577.34),  # sky/nebular [O I]
    ("em",  "He I",            5875.62),  # AGN/HII region; also stellar in hot types
    ("em",  "[O I]",           6300.30),
    ("em",  "[O I]",           6363.78),
    ("em",  "[N II]",          5754.60),  # faint but present in low-ionization nebulae
    ("em",  r"H$\alpha$",      6562.80),
    ("em",  "[N II]",          6548.05),
    ("em",  "[N II]",          6583.45),
    ("em",  "[S II]",          6716.44),
    ("em",  "[S II]",          6730.82),
    ("em",  "[Ar III]",        7135.80),
    ("em",  "[S III]",         9068.60),
    ("em",  "[S III]",         9530.60),
)

# Combined look-up: maps tag+name to rest wavelength for all windows.
_STELLAR_ABSORPTION_LINE_TABLES = (
    DEFAULT_CAI_REGION_LINES,
    DEFAULT_MGB_REGION_LINES,
    DEFAULT_OPTICAL_RED_LINES,
)

# Default set of stellar photosphere species to query from NIST.
_NIST_STELLAR_ELEMENTS: tuple[str, ...] = (
    "Fe I", "Fe II",
    "Ca I", "Ca II",
    "Ti I", "Ti II",
    "Mg I", "Mg II",
    "Si I", "Si II",
    "Cr I",
    "Ni I",
    "Mn I",
    "V I",
    "Ba II",
    "Co I",
    "Sc I", "Sc II",
)


def query_nist_lines(
    wavemin_A: float,
    wavemax_A: float,
    elements: tuple[str, ...] = _NIST_STELLAR_ELEMENTS,
    min_rel_intensity: float = 10.0,
    default_half_width_A: float = 3.0,
    cache_path: "str | None" = None,
    force_refresh: bool = False,
) -> "list[tuple[float, float, str]]":
    """Query NIST Atomic Spectra Database for stellar absorption lines.

    Returns a list of ``(center_A, half_width_A, name)`` tuples suitable for
    direct assignment to ``cfg.extra_absorption_lines``.

    Parameters
    ----------
    wavemin_A, wavemax_A
        Wavelength range in Angstroms (vacuum).
    elements
        Iterable of species strings such as ``"Fe I"``, ``"Ca II"``.
        Defaults to ``_NIST_STELLAR_ELEMENTS``.
    min_rel_intensity
        Minimum NIST relative intensity to include (rough line-strength proxy).
        Lines where intensity cannot be parsed are included by default.
    default_half_width_A
        Half-width assigned to every returned line for use in masking.
    cache_path
        If given, cache query results to this JSON file and re-use on subsequent
        calls (unless ``force_refresh=True``).
    force_refresh
        Ignore the cache file and re-query NIST.

    Returns
    -------
    list of (center_A, half_width_A, name) tuples.

    Notes
    -----
    Tries ``astroquery.nist`` first (most reliable).  Falls back to a direct
    HTTP request to ``physics.nist.gov`` with line-by-line HTML parsing when
    astroquery is unavailable.  If both fail, raises ``RuntimeError`` with a
    diagnostic message.

    The NIST ASD ``Rel.`` intensity column contains a mix of integers and
    annotated strings (e.g. ``"100*"``); non-numeric suffixes are stripped
    before comparison.
    """
    import json
    import re
    from pathlib import Path

    # ------------------------------------------------------------------ cache
    cache_key = {
        "wavemin": wavemin_A,
        "wavemax": wavemax_A,
        "elements": sorted(elements),
        "min_rel_intensity": min_rel_intensity,
    }

    if cache_path is not None and not force_refresh:
        cp = Path(cache_path)
        if cp.exists():
            try:
                stored = json.loads(cp.read_text())
                if stored.get("_cache_key") == cache_key:
                    lines = stored["lines"]
                    log(f"query_nist_lines: loaded {len(lines)} lines from cache {cp}")
                    return [(float(c), float(hw), str(n)) for c, hw, n in lines]
            except Exception:
                pass  # corrupt cache — fall through to query

    results: list[tuple[float, float, str]] = []
    errors: list[str] = []

    # ------------------------------------------------------------------ astroquery path
    try:
        from astroquery.nist import Nist  # type: ignore
        import astropy.units as u  # type: ignore

        _logged_cols: bool = False
        for species in elements:
            try:
                tbl = Nist.query(
                    wavemin_A * u.AA,
                    wavemax_A * u.AA,
                    linename=species,
                    wavelength_type="vacuum",
                )
                if tbl is None or len(tbl) == 0:
                    continue
                if not _logged_cols:
                    log(f"query_nist_lines: astroquery columns: {tbl.colnames}")
                    _logged_cols = True
                # astroquery.nist column names: "Observed", "Ritz", "Rel.", etc.
                # Older versions used names containing "wave"; check both.
                wave_col = next(
                    (
                        c for c in tbl.colnames
                        if c.lower() in ("observed", "ritz")
                        or "wave" in c.lower()
                        or "obs" in c.lower()
                    ),
                    None,
                )
                reli_col = next(
                    (c for c in tbl.colnames if c.strip(".").lower() == "rel"
                     or "rel" in c.lower()),
                    None,
                )
                if wave_col is None:
                    errors.append(
                        f"{species}: unrecognised column names {tbl.colnames!r}"
                    )
                    continue
                for row in tbl:
                    try:
                        raw_wav = row[wave_col]
                        # astroquery returns MaskedColumn rows; skip masked entries
                        import numpy.ma as _ma
                        if _ma.is_masked(raw_wav):
                            continue
                        wav = float(raw_wav)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(wav):
                        continue
                    if not (wavemin_A <= wav <= wavemax_A):
                        continue
                    if reli_col is not None:
                        raw = str(row[reli_col]).strip()
                        # Strip annotations like '*', '?', 'b', 'c', etc.
                        num = re.sub(r"[^\d.]", "", raw)
                        if num:
                            try:
                                if float(num) < min_rel_intensity:
                                    continue
                            except ValueError:
                                pass
                    results.append((wav, default_half_width_A, species))
            except Exception as e:
                errors.append(f"{species}: {e}")

        if results:
            log(
                f"query_nist_lines (astroquery): {len(results)} lines"
                + (f"; {len(errors)} species skipped" if errors else "")
            )
            results.sort(key=lambda t: t[0])
            _nist_cache_write(cache_path, cache_key, results)
            return results

    except ImportError:
        errors.append("astroquery not installed")

    # ------------------------------------------------------------------ HTTP fallback
    try:
        import urllib.request
        import urllib.parse

        for species in elements:
            try:
                params = urllib.parse.urlencode(
                    {
                        "spectra": species,
                        "limits_type": "0",
                        "low_w": f"{wavemin_A:.2f}",
                        "upp_w": f"{wavemax_A:.2f}",
                        "unit": "0",  # 0 = Angstrom
                        "de": "0",
                        "format": "3",  # tab-separated ASCII
                        "line_out": "0",
                        "en_unit": "0",
                        "output": "0",
                        "bibrefs": "0",
                        "show_obs_wl": "1",
                        "show_calc_wl": "0",
                        "order_out": "0",
                        "max_low_enrg": "",
                        "show_av": "2",  # vacuum wavelengths
                        "max_upp_enrg": "",
                        "tsb_value": "0",
                        "min_str": "",
                        "A_out": "0",
                        "intens_out": "on",
                        "allowed_out": "1",
                        "forbid_out": "1",
                        "conf_out": "0",
                        "term_out": "0",
                        "enrg_out": "0",
                        "J_out": "0",
                    }
                )
                url = f"https://physics.nist.gov/cgi-bin/ASD/lines1.pl?{params}"
                req = urllib.request.Request(
                    url, headers={"User-Agent": "kinextract/1.0 (science)"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = resp.read().decode("utf-8", errors="replace")

                for line in text.splitlines():
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    wav_str = parts[0].strip().strip('"')
                    if not wav_str or wav_str.lower().startswith("obs"):
                        continue
                    try:
                        wav = float(wav_str)
                    except ValueError:
                        continue
                    if not (wavemin_A <= wav <= wavemax_A):
                        continue
                    if len(parts) >= 2:
                        raw = parts[1].strip().strip('"')
                        num = re.sub(r"[^\d.]", "", raw)
                        if num:
                            try:
                                if float(num) < min_rel_intensity:
                                    continue
                            except ValueError:
                                pass
                    results.append((wav, default_half_width_A, species))
            except Exception as e:
                errors.append(f"{species}: {e}")

        if results:
            log(
                f"query_nist_lines (HTTP): {len(results)} lines"
                + (f"; {len(errors)} species had errors" if errors else "")
            )
            results.sort(key=lambda t: t[0])
            _nist_cache_write(cache_path, cache_key, results)
            return results

    except Exception as e:
        errors.append(f"HTTP fallback failed: {e}")

    raise RuntimeError(
        "query_nist_lines: all query methods failed.\n"
        "Install astroquery (pip install astroquery) for the most reliable path.\n"
        "Errors:\n" + "\n".join(f"  {e}" for e in errors)
    )


def _nist_cache_write(
    cache_path: "str | None",
    cache_key: dict,
    results: "list[tuple[float, float, str]]",
) -> None:
    """Write `query_nist_lines` results to `cache_path` as JSON, if given."""
    if cache_path is None:
        return
    import json
    from pathlib import Path

    try:
        data = {"_cache_key": cache_key, "lines": results}
        Path(cache_path).write_text(json.dumps(data, indent=2))
    except Exception as e:
        log(f"query_nist_lines: could not write cache: {e}")


def _velocity_pad_A(center_A: float, vpad_kms: float) -> float:
    """Convert velocity padding in km/s to wavelength padding in Å."""
    return abs(float(center_A)) * abs(float(vpad_kms)) / CEE


def _center_to_fit_frame(center: float, frame: str, cfg: FitConfig) -> float:
    """
    Convert a line center to the frame of st.x.

    For raw .spec files, st.x is rest-frame because load_spectrum_for_fit()
    divides wavelengths by 1 + zgal. Therefore observed-frame masks must be
    divided by 1 + zgal.
    """
    frame = frame.lower().strip()
    c = float(center)

    if frame == "observed":
        c = c / (1.0 + cfg.zgal)
    elif frame != "rest":
        raise ValueError(f"Mask frame must be 'rest' or 'observed', got {frame!r}")

    c += float(getattr(cfg, "als_mask_center_shift_A", 0.0))
    return c


def _estimate_caii_mask_shift(st, cfg) -> float:
    """Measure how far the Ca II triplet troughs sit from their rest positions.

    The input zgal is often an average redshift from bootstrap results and may
    be slightly off for individual spectra.  This function finds the actual
    trough position of each Ca II line (8498, 8542, 8662 Å) via parabolic
    sub-pixel interpolation and returns the median wavelength offset.

    Returns shift in Å (positive = observed features are redder than catalog).
    Applied to als_mask_center_shift_A only — no effect on velocity/LOSVD.
    """
    if not getattr(cfg, "als_auto_caii_shift", True):
        return 0.0

    search_hw = float(getattr(cfg, "als_auto_caii_search_hw_A", 12.0))
    centers_rest = list(getattr(cfg, "als_ca_centers", [8498.02, 8542.09, 8662.14]))
    wmin = float(getattr(cfg, "wavefitmin", float(st.x.min())))
    wmax = float(getattr(cfg, "wavefitmax", float(st.x.max())))
    good = (st.gerr < BIG) & np.isfinite(st.g)

    shifts: list[float] = []
    for cen_rest in centers_rest:
        if cen_rest < wmin or cen_rest > wmax:
            continue
        win = (st.x >= cen_rest - search_hw) & (st.x <= cen_rest + search_hw) & good
        if win.sum() < 5:
            continue
        x_win = st.x[win]
        g_win = st.g[win]
        idx_min = int(np.argmin(g_win))
        i0, i1, i2 = idx_min - 1, idx_min, idx_min + 1
        if i0 < 0 or i2 >= len(g_win):
            x_min = float(x_win[idx_min])
        else:
            ya = float(g_win[i0]); yb = float(g_win[i1]); yc = float(g_win[i2])
            xa = float(x_win[i0]); xb = float(x_win[i1]); xc = float(x_win[i2])
            # Parabola vertex: x_min = -B/(2A) where A,B are polynomial coefficients.
            denom = (xa - xb) * (xa - xc) * (xb - xc)
            if abs(denom) < 1e-30:
                x_min = xb
            else:
                A = (xc * (yb - ya) + xb * (ya - yc) + xa * (yc - yb)) / denom
                B = (xc**2 * (ya - yb) + xb**2 * (yc - ya) + xa**2 * (yb - yc)) / denom
                x_min = xb if abs(A) < 1e-30 else float(np.clip(-B / (2.0 * A), x_win[0], x_win[-1]))
        shifts.append(x_min - cen_rest)

    if not shifts:
        return 0.0
    return float(np.median(shifts))


def _window_to_fit_frame(
    lo: float,
    hi: float,
    frame: str,
    cfg: FitConfig,
) -> tuple[float, float]:
    """
    Convert a wavelength window to the frame of st.x.
    """
    frame = frame.lower().strip()
    lo = float(lo)
    hi = float(hi)

    if frame == "observed":
        lo = lo / (1.0 + cfg.zgal)
        hi = hi / (1.0 + cfg.zgal)
    elif frame != "rest":
        raise ValueError(f"Mask frame must be 'rest' or 'observed', got {frame!r}")

    shift = float(getattr(cfg, "als_mask_center_shift_A", 0.0))
    lo += shift
    hi += shift

    if hi < lo:
        lo, hi = hi, lo

    return lo, hi


def _apply_center_mask(
    mask: np.ndarray,
    x: np.ndarray,
    center: float,
    half_width: float,
    cfg: FitConfig,
) -> np.ndarray:
    """Apply center ± half_width, plus velocity padding."""
    vpad = float(getattr(cfg, "als_mask_velocity_pad_kms", 0.0))
    hw = float(half_width) + _velocity_pad_A(center, vpad)
    mask |= (x >= center - hw) & (x <= center + hw)
    return mask


def build_als_line_mask(st, cfg) -> np.ndarray:
    """
    Build the boolean mask of pixels to exclude from the ALS continuum fit.

    Combines several independent line/window sources into a single
    exclusion mask (True = exclude from the ALS baseline fit): the
    configured Ca II triplet windows, the optional built-in
    `DEFAULT_CAI_REGION_LINES` table (Paschen/absorption/emission
    features in the 8300-8900 Å Ca II region), user-specified extra line
    centers and wavelength windows, S/N-gated ``extra_absorption_lines``,
    and an automatic S/N-based detection pass over the built-in stellar
    absorption line tables (`_als_auto_abs_line_mask`). Each contributing
    window is optionally padded by a velocity-equivalent Angstrom width
    (``als_mask_velocity_pad_kms``) so masks scale with the galaxy's
    velocity dispersion rather than being fixed in wavelength.

    Parameters
    ----------
    st : FitState
        Fit state; supplies the wavelength grid ``st.x`` and flux/error
        arrays used for S/N-gating.
    cfg : FitConfig
        Fit configuration; supplies all ``als_mask_*``/``als_ca_*``/
        ``als_extra_mask_*``/``extra_absorption_lines`` settings that
        control which lines and windows are excluded.

    Returns
    -------
    ndarray of bool
        Mask of length ``st.npix``; True marks pixels excluded from the
        ALS continuum fit.

    Notes
    -----
    This mask affects continuum fitting only. Excluded pixels are still
    used in the LOSVD/template fit unless separately masked via
    ``st.gerr`` or a ``regions.bad`` file. For raw ``.spec`` inputs,
    ``st.x`` is rest-frame because `~kinextract.spectrum.load_spectrum_for_fit`
    divides wavelengths by ``1 + zgal``; observed-frame mask centers and
    windows are therefore converted by dividing by ``1 + zgal`` before
    being applied.
    """
    mask = np.zeros(st.npix, dtype=bool)
    x = st.x
    vpad = float(getattr(cfg, "als_mask_velocity_pad_kms", 0.0))

    if cfg.als_mask_ca:
        centers = np.asarray(cfg.als_ca_centers, float)
        widths = np.asarray(cfg.als_ca_half_widths, float)

        if widths.size == 1:
            widths = np.full_like(centers, float(widths[0]))

        for c_raw, hw in zip(centers, widths):
            c = _center_to_fit_frame(c_raw, cfg.als_ca_frame, cfg)
            mask = _apply_center_mask(mask, x, c, hw, cfg)

    if getattr(cfg, "als_use_default_caii_region_lines", False):
        abs_hw = float(getattr(cfg, "als_default_abs_half_width", 2.0))
        em_hw = float(getattr(cfg, "als_default_em_half_width", 2.5))
        pas_hw = float(getattr(cfg, "als_default_pas_half_width", 2.0))

        for kind, name, c_raw in DEFAULT_CAI_REGION_LINES:
            c = _center_to_fit_frame(c_raw, "rest", cfg)

            if kind == "paschen":
                hw = pas_hw
            elif kind == "em":
                hw = em_hw
            else:
                hw = abs_hw

            mask = _apply_center_mask(mask, x, c, hw, cfg)

    # User-specified extra line-center masks
    extra_line_frame = getattr(cfg, "als_extra_mask_line_frame", "rest")

    for item in getattr(cfg, "als_extra_mask_lines", ()):
        if not hasattr(item, "__len__"):
            raise ValueError(
                f"als_extra_mask_lines must be a tuple of (center, half_width) pairs, "
                f"e.g. ((8465.0, 5.0),). Got a bare {type(item).__name__} ({item!r}) — "
                f"did you forget the trailing comma? Write ((center, hw),) not ((center, hw))."
            )
        if len(item) < 2:
            raise ValueError(
                "Each als_extra_mask_lines entry must be (center, half_width)"
            )

        c_raw = float(item[0])
        hw = float(item[1])

        c = _center_to_fit_frame(c_raw, extra_line_frame, cfg)
        mask = _apply_center_mask(mask, x, c, hw, cfg)
    # User-specified extra wavelength-window masks
    extra_window_frame = getattr(cfg, "als_extra_mask_frame", "rest")

    for lo_raw, hi_raw in getattr(cfg, "als_extra_mask_windows", ()):
        lo, hi = _window_to_fit_frame(lo_raw, hi_raw, extra_window_frame, cfg)

        # Add velocity padding based on the window center.
        cen = 0.5 * (lo + hi)
        pad = _velocity_pad_A(cen, vpad)

        mask |= (x >= lo - pad) & (x <= hi + pad)

    # Shared extra absorption lines: S/N-gated so that lines absent from the
    # spectrum (e.g. bulk NIST results) are not masked unnecessarily.
    if getattr(cfg, "extra_absorption_lines", ()):
        snr_thr_extra = float(getattr(cfg, "als_auto_mask_snr_threshold", 5.0))
        snr_ctx_extra = float(getattr(cfg, "als_auto_mask_snr_context_A", 20.0))
        gerr_pos = np.where(st.gerr > 0, st.gerr, np.nan)
        for item in getattr(cfg, "extra_absorption_lines", ()):
            c_raw = float(item[0])
            hw = float(item[1])
            ctx = (st.x >= c_raw - snr_ctx_extra) & (st.x <= c_raw + snr_ctx_extra)
            if ctx.sum() >= 3:
                g_med = float(np.nanmedian(st.g[ctx]))
                e_med = float(np.nanmedian(gerr_pos[ctx]))
                if not (np.isfinite(g_med) and np.isfinite(e_med) and e_med > 0):
                    continue
                if g_med / e_med < snr_thr_extra:
                    continue
            c = _center_to_fit_frame(c_raw, "rest", cfg)
            mask = _apply_center_mask(mask, x, c, hw, cfg)

    # S/N-based auto-detection of strong stellar absorption lines.
    # Uses the same line tables as the kinematic cleaning protect mask so that
    # any feature strong enough to protect from clipping is also excluded from
    # the ALS continuum fit.
    if getattr(cfg, "als_auto_mask_abs_lines", True):
        mask |= _als_auto_abs_line_mask(st, cfg)

    return mask


def _als_auto_abs_line_mask(st, cfg) -> np.ndarray:
    """
    Build a boolean ALS-exclusion mask for strong stellar absorption lines
    detected at sufficient local S/N.

    Mirrors _auto_absorption_protect_mask but targets ALS continuum masking
    rather than kinematic-cleaning protection.  Half-width comes from
    als_default_abs_half_width (same as the manual als_use_default_caii_region_lines
    path) so the two modes are consistent.
    """
    mask = np.zeros(st.npix, dtype=bool)

    half_w = float(getattr(cfg, "als_auto_mask_half_width_A",
                           getattr(cfg, "als_default_abs_half_width", 5.0)))
    snr_ctx = float(getattr(cfg, "als_auto_mask_snr_context_A", 20.0))
    snr_thr = float(getattr(cfg, "als_auto_mask_snr_threshold", 5.0))
    mask_paschen = bool(getattr(cfg, "als_auto_mask_paschen", False))

    wmin = float(getattr(cfg, "wavefitmin", st.x.min()))
    wmax = float(getattr(cfg, "wavefitmax", st.x.max()))

    gerr_pos = np.where(st.gerr > 0, st.gerr, np.nan)
    vpad = float(getattr(cfg, "als_mask_velocity_pad_kms", 0.0))

    for table in _STELLAR_ABSORPTION_LINE_TABLES:
        for tag, name, cen_rest in table:
            if tag == "em":
                continue
            if tag == "paschen" and not mask_paschen:
                continue
            if cen_rest < wmin or cen_rest > wmax:
                continue

            ctx = (st.x >= cen_rest - snr_ctx) & (st.x <= cen_rest + snr_ctx)
            if ctx.sum() < 3:
                continue

            g_med = float(np.nanmedian(st.g[ctx]))
            e_med = float(np.nanmedian(gerr_pos[ctx]))
            if not (np.isfinite(g_med) and np.isfinite(e_med) and e_med > 0):
                continue
            if g_med / e_med < snr_thr:
                continue

            hw = half_w + _velocity_pad_A(cen_rest, vpad)
            cen_fit = _center_to_fit_frame(cen_rest, "rest", cfg)
            mask |= (st.x >= cen_fit - hw) & (st.x <= cen_fit + hw)

    return mask


def init_als_continuum(st, cfg, templates: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Initialize the ALS continuum baseline from the raw flux before LOSVD fitting.

    Called once from `~kinextract.spectrum.make_fit_state` when
    ``cfg.fit_als_continuum`` is True. This establishes ``st.continuum_mult``
    (the baseline the LOSVD/template model will be multiplied by) and
    ``st.continuum_mask`` (the continuum-good pixel mask) before any
    LOSVD/template optimization has taken place, using
    `fit_als_target_absorption_clean` for the fit itself. It also (1)
    optionally refines the Ca II mask center shift
    (``als_mask_center_shift_A``) by measuring the actual trough positions
    via `_estimate_caii_mask_shift`, (2) optionally freezes the
    hyperparameters found here for reuse in later updates
    (``als_optimize_init_only``), and (3) propagates template-library
    flux uncertainty into ``st.gerr`` in quadrature, scaled by the fitted
    continuum, since template noise is approximately proportional to
    the continuum level.

    Parameters
    ----------
    st : FitState
        Fit state to initialize in place; ``st.g``, ``st.gerr``, ``st.x``,
        and (if present) ``st.t`` are read, and ``st.continuum_mult``,
        ``st.continuum_mask``, ``st.continuum_mask_init``,
        ``st.als_lam_current``, ``st.als_p_current``, and
        ``st.als_records`` are set.
    cfg : FitConfig
        Fit configuration controlling ALS behavior (see `build_als_line_mask`,
        `fit_als_target_absorption_clean`, and ``als_clip``).
    templates : ndarray, optional
        Template flux matrix used for template-score mode during the
        initial hyperparameter search (see `score_als_target`). Typically
        ``st.t``.

    Returns
    -------
    ndarray
        The initial ALS continuum baseline (also stored as
        ``st.continuum_mult``).
    """
    # ── Ca II wavelength-shift refinement (masking only) ────────────────────
    # If zgal is slightly off, the Ca II troughs sit at a different position in
    # st.x than their catalog rest wavelengths.  Measure the offset and fold it
    # into als_mask_center_shift_A so all mask centers follow the spectrum.
    if getattr(cfg, "als_auto_caii_shift", True):
        caii_shift = _estimate_caii_mask_shift(st, cfg)
        if caii_shift != 0.0:
            cfg.als_mask_center_shift_A = float(getattr(cfg, "als_mask_center_shift_A", 0.0)) + caii_shift
            st.als_caii_refine_shift_A = caii_shift
            log(
                f"Ca II mask shift: {caii_shift:+.3f} Å  "
                f"(als_mask_center_shift_A → {cfg.als_mask_center_shift_A:.3f} Å)"
            )

    good = (
        (st.gerr < BIG)
        & np.isfinite(st.g)
        & np.isfinite(st.gerr)
        & (st.gerr > 0.0)
    )
    line_mask = build_als_line_mask(st, cfg)
    base = good & ~line_mask
    fill = np.nanmedian(st.g[base]) if np.any(base) else np.nanmedian(st.g[good])
    if not np.isfinite(fill):
        fill = 1.0
    y = np.where(np.isfinite(st.g), st.g, fill)
    err_fill = np.nanmedian(st.gerr[good]) if np.any(good) else 1.0
    if not np.isfinite(err_fill) or err_fill <= 0.0:
        err_fill = 1.0
    yerr = np.where(
        np.isfinite(st.gerr) & (st.gerr > 0.0),
        st.gerr,
        err_fill,
    )
    cont, lam, p, records, base = fit_als_target_absorption_clean(
        y,
        yerr,
        base,
        cfg,
        x=st.x,
        force_fixed=False,
        templates=getattr(st, "t", None),
    )
    # If requested, optimize ALS hyperparameters only once and reuse.
    if getattr(cfg, "als_optimize_init_only", False):
        cfg.als_lam = float(lam)
        cfg.als_p = float(p)
        log(
            f"ALS optimize-init-only: reusing "
            f"lam={cfg.als_lam:.3e}, p={cfg.als_p:.4g}"
        )
    lo, hi = cfg.als_clip
    cont = np.clip(cont, lo, hi) if np.isfinite(hi) else np.maximum(cont, lo)
    st.continuum_mult = cont
    st.continuum_mask = base
    st.continuum_mask_init = base.copy()
    st.als_lam_current = float(lam)
    st.als_p_current = float(p)
    st.als_records.append(
        {
            "kind": "init",
            "lam": float(lam),
            "p": float(p),
            "records": records,
        }
    )
    log(
        f"ALS init: lam={lam:.3e} p={p:.4g} "
        f"median={np.nanmedian(cont):.4g} "
        f"base_pixels={np.count_nonzero(base)} "
        f"line_mask_pixels={np.count_nonzero(line_mask)}"
    )

    # ── Template error propagation ───────────────────────────────────────────
    # Add template noise in quadrature to galaxy gerr.  The template
    # uncertainty is approximately f_template × continuum at each pixel.
    # This is computed once, after the initial continuum estimate is available.
    # Skipped when use_spectrum_errors=False: adding pixel-dependent template
    # noise would break the uniform-weighting guarantee.
    f_tpl = float(getattr(st, "f_template", 0.0))
    if f_tpl > 0.0 and getattr(cfg, "use_spectrum_errors", True):
        sigma_tpl = f_tpl * np.abs(cont)
        valid = st.gerr < BIG
        old_gerr_med = float(np.nanmedian(st.gerr[valid])) if valid.any() else 0.0
        st.gerr = np.where(valid, np.sqrt(st.gerr**2 + sigma_tpl**2), st.gerr)
        new_gerr_med = float(np.nanmedian(st.gerr[valid])) if valid.any() else 0.0
        log(
            f"Template error propagation: f_template={f_tpl:.4f}  "
            f"median(gerr): {old_gerr_med:.4g} → {new_gerr_med:.4g}"
        )

    return cont


def update_als_continuum(st, cfg, a_best: np.ndarray) -> float:
    """
    Refine the ALS continuum baseline after an LOSVD/template optimization step.

    Called once per outer iteration of the fit when ``cfg.fit_als_continuum``
    is True, alternating with LOSVD/template updates until convergence. The
    current best-fit parameters `a_best` are used to evaluate the
    LOSVD/template model *without* the continuum applied
    (``gp0 = evaluate_model_gp(..., apply_continuum=False)``), and the ALS
    baseline is refit against the target ``data / gp0`` — i.e. an absolute
    continuum estimate, not a multiplicative correction applied on top of
    the previous continuum. `fit_als_target_absorption_clean` performs the
    actual banded ALS solve and absorption-pixel cleaning, scored via
    `score_als_target`'s model-score mode
    (``score_data ~= score_model0 * continuum``).

    If ``cfg.als_abs_clean_init_only`` is True, the continuum-good pixel
    mask established once during `init_als_continuum` is reused rather than
    rerun, and absorption-pixel cleaning is skipped on subsequent updates
    for speed and stability.

    Parameters
    ----------
    st : FitState
        Fit state to update in place; ``st.continuum_mult``,
        ``st.continuum_mask``, ``st.als_lam_current``, ``st.als_p_current``,
        and ``st.als_records`` are overwritten with the refined values.
    cfg : FitConfig
        Fit configuration controlling ALS behavior (see
        `fit_als_target_absorption_clean` and ``als_clip``).
    a_best : ndarray
        Current best-fit parameter vector (LOSVD bins, template weights,
        and any continuum/offset parameters), used to evaluate the
        continuum-free model `gp0`.

    Returns
    -------
    delta : float
        Median absolute fractional change in the continuum relative to
        its previous value, ``median(|continuum_new - continuum_old| /
        |continuum_old|)``. Useful as an outer-loop convergence metric.
    """
    from .numerics import evaluate_model_gp
    gp0, *_ = evaluate_model_gp(a_best, st, apply_continuum=False)

    good = (
        (st.gerr < BIG)
        & np.isfinite(st.g)
        & np.isfinite(st.gerr)
        & (st.gerr > 0.0)
        & np.isfinite(gp0)
        & (np.abs(gp0) > 0.0)
    )

    line_mask = build_als_line_mask(st, cfg)
    init_mask = getattr(st, "continuum_mask_init", None)
    _abs_clean_init_only = getattr(cfg, "als_abs_clean_init_only", False)
    if _abs_clean_init_only and init_mask is not None:
        # Reuse the absorption base established during init; only update for any
        # new sigma-clip rejections that have set gerr=BIG since then.
        base = good & np.asarray(init_mask, bool)
        _skip_abs_clean = True
    else:
        base = good & ~line_mask
        _skip_abs_clean = False

    target = np.full(st.npix, np.nan, dtype=float)
    target_err = np.full(st.npix, np.nan, dtype=float)

    target[good] = st.g[good] / gp0[good]
    target_err[good] = st.gerr[good] / np.abs(gp0[good])

    fill = np.nanmedian(target[base]) if np.any(base) else np.nanmedian(target[good])
    if not np.isfinite(fill):
        fill = 1.0

    y = np.where(np.isfinite(target), target, fill)

    err_fill = np.nanmedian(target_err[good]) if np.any(good) else 1.0
    if not np.isfinite(err_fill) or err_fill <= 0.0:
        err_fill = 1.0

    yerr = np.where(
        np.isfinite(target_err) & (target_err > 0.0),
        target_err,
        err_fill,
    )

    old = getattr(st, "continuum_mult", np.ones(st.npix, dtype=float))

    cont, lam, p, records, base = fit_als_target_absorption_clean(
        y,
        yerr,
        base,
        cfg,
        x=st.x,
        force_fixed=getattr(cfg, "als_optimize_init_only", False),
        skip_abs_clean=_skip_abs_clean,
        templates=None,
        score_data=st.g,
        score_err=st.gerr,
        score_model0=gp0,
    )

    lo, hi = cfg.als_clip
    cont = np.clip(cont, lo, hi) if np.isfinite(hi) else np.maximum(cont, lo)

    st.continuum_mult = cont
    st.continuum_mask = base
    st.als_lam_current = float(lam)
    st.als_p_current = float(p)

    if not hasattr(st, "als_records") or st.als_records is None:
        st.als_records = []

    st.als_records.append(
        {
            "kind": "update",
            "lam": float(lam),
            "p": float(p),
            "records": records,
        }
    )

    denom = np.maximum(np.abs(old), 1e-12)
    delta = float(np.nanmedian(np.abs((cont - old) / denom)))

    log(
        f"  ALS update: lam={lam:.3e} p={p:.4g} "
        f"delta={delta:.4g} "
        f"base_pixels={np.count_nonzero(base)} "
        f"line_mask_pixels={np.count_nonzero(line_mask)}"
    )

    return delta
