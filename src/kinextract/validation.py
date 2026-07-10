"""Empirical, mock-based recovery-bias assessment and correction.

Complements :mod:`kinextract.errors`'s Laplace/bootstrap uncertainty
estimators (which characterize scatter around a single fit) by directly
measuring systematic bias: for a grid of known ground-truth (V, sigma)
values, :func:`assess_recovery_bias` generates matched mock spectra
(:func:`kinextract.mocks.build_matched_mock`) with the same instrument
resolution, template mixture, continuum, and noise level as a real
target, refits each with the MAP+bootstrap pipeline
(:func:`kinextract.fitting.fit_state_map_with_optional_clean`), and
compares recovered to true. This is the recommended way to characterize
bias for a specific target/instrument/S-N combination -- more informative
than a generic "near the resolution limit, expect roughly X km/s bias"
rule of thumb (see :class:`~kinextract.config.FitConfig`'s "Known
limitations" section for that generic guidance), and, unlike the retired
analytic :func:`kinextract.errors.bias_corrected_losvd`, it makes no
linearization assumption -- it just directly measures what the real
pipeline does to a spectrum with a known answer.

:func:`assess_recovery_bias` only supports pre-normalized-mode fits
(``cfg.fit_continuum=False`` and ``cfg.joint_prenorm=False``): it raises
``NotImplementedError`` for continuum-cofit fits, since
``build_matched_mock``/``evaluate_model_gp`` don't understand a joint
fit's ``[LOSVD, template weights, continuum P-spline]`` parameter
layout. Pre-normalize first (see
:func:`kinextract.continuum.asymmetric_least_squares_continuum` and
``examples/notebooks/06_prenormalized_workflow.ipynb``).

:func:`correct_recovered_losvd` then interpolates that empirical bias
table at a real fit's own recovered (V, sigma) to produce a
bias-corrected point estimate with an inflated uncertainty; this
correction is always opt-in, never applied automatically by
:func:`kinextract.fitting.run_spectral_fit`.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
from scipy.interpolate import griddata

from ._utils import log
from .config import FitConfig
from .errors import _fit_state_to_fields
from .fitting import fit_state_map_with_optional_clean
from .losvd import fit_losvd_gauss_hermite
from .mocks import build_matched_mock
from .numerics import evaluate_model_gp
from .spectrum import build_initial_guess_nonparam
from .state import FitState


def assess_recovery_bias(
    fit: dict,
    cfg: FitConfig,
    v_true_grid: Sequence[float],
    sigma_true_grid: Sequence[float],
    h3_true: float = 0.0,
    h4_true: float = 0.0,
    n_seeds: int = 8,
    seed0: int = 0,
) -> dict:
    """Measure empirical LOSVD recovery bias on a grid of known truths.

    For every combination of `v_true_grid` x `sigma_true_grid`, generates
    `n_seeds` independent matched mocks (:func:`kinextract.mocks.build_matched_mock`,
    reusing `fit`'s own instrument grid, template mixture, continuum, and
    per-pixel noise level), refits each with
    :func:`kinextract.fitting.fit_state_map_with_optional_clean` (the same
    pipeline :func:`kinextract.fitting.run_spectral_fit` uses by default),
    and reports the mean and seed-to-seed standard deviation of
    ``recovered - true`` for V, sigma, h3, and h4
    (:func:`kinextract.losvd.fit_losvd_gauss_hermite`).

    Parameters
    ----------
    fit : dict
        A completed fit dict, as returned by
        :func:`kinextract.fitting.run_spectral_fit` (or built directly via
        :func:`kinextract.spectrum.make_fit_state` +
        :func:`kinextract.fitting.fit_state_map_with_optional_clean`) on
        the real target being validated. Must contain ``"state"`` (the
        :class:`~kinextract.state.FitState`) and ``"a_map"`` (the best-fit
        flat parameter vector) -- these supply the template weights,
        continuum, and amplitude that matched mocks reuse, and the
        instrument grid/noise level they're built on. Not modified.
    cfg : FitConfig
        The same configuration used to produce `fit`. Every replicate
        refit runs with `cfg` exactly as configured -- including
        ``xlam_auto`` (so the bias table reflects the same
        regularisation-selection behavior a real fit would use) and
        ``clean`` sigma-clipping -- so the measured bias reflects the
        pipeline's actual end-to-end behavior on this target's
        configuration, not a simplified version of it. A copy of `cfg` is
        used internally, so the caller's own `cfg` object (e.g. its
        ``xlam`` field, which ``xlam_auto`` would otherwise overwrite on
        every replicate) is never mutated by this call.
    v_true_grid, sigma_true_grid : sequence of float
        Ground-truth velocity/dispersion grid points to test, in km/s.
        The full Cartesian product is evaluated.
    h3_true, h4_true : float, optional
        Ground-truth higher-order Gauss-Hermite moments to inject at
        every grid point (held fixed across the grid; this characterizes
        bias for one LOSVD shape at a time).
    n_seeds : int, optional
        Number of independent noise realizations per grid point.
    seed0 : int, optional
        Base seed; replicate `i` at each grid point uses seed
        ``seed0 + i``, so results are reproducible.

    Returns
    -------
    dict
        Keyed by ``(v_true, sigma_true)`` tuples. Each value is a dict
        with keys ``bias_v``, ``bias_v_std``, ``bias_sigma``,
        ``bias_sigma_std``, ``bias_h3``, ``bias_h3_std``, ``bias_h4``,
        ``bias_h4_std`` (all floats, mean/std of ``recovered - true``
        over the successful replicates), and ``n_seeds`` (int, the number
        of replicates that converged -- may be less than the requested
        `n_seeds` if some replicates fail). Grid points where every
        replicate failed are omitted (logged, not raised).
    """
    joint_active = cfg.fit_continuum or getattr(cfg, "joint_prenorm", False)
    if joint_active:
        raise NotImplementedError(
            "assess_recovery_bias only supports pre-normalized-mode fits "
            "(cfg.fit_continuum=False, cfg.joint_prenorm=False): fit['a_map']'s "
            "layout -- [LOSVD bins, template weights, continuum P-spline "
            "coefficients] for a joint fit -- is not what "
            "build_matched_mock/evaluate_model_gp expect, so mock generation "
            "would silently use the wrong model. Pre-normalize the spectrum "
            "first (see kinextract.continuum.asymmetric_least_squares_continuum "
            "and examples/notebooks/06_prenormalized_workflow.ipynb) and fit "
            "with fit_continuum=False before calling assess_recovery_bias."
        )

    st_ref = fit["state"]
    a_fit = np.asarray(fit["a_map"], float)
    st_fields = _fit_state_to_fields(st_ref)
    base_gerr = np.asarray(st_ref.gerr, float).copy()

    _, xlb, xub = build_initial_guess_nonparam(
        st_ref, cfg.coff, cfg.coff2,
        w_bounds=cfg.template_w_bounds if cfg.template_w_bounds is not None else (1e-5, 1.0),
    )
    bounds = list(zip(xlb, xub))

    # Work on a copy: _auto_select_xlam (invoked once per replicate below, if
    # cfg.xlam_auto is set) mutates cfg.xlam as a side effect -- across
    # dozens of replicates that would otherwise silently overwrite the
    # caller's own cfg.xlam with whatever the *last* mock happened to select.
    cfg = dataclasses.replace(cfg)

    results: dict = {}
    for v_true in v_true_grid:
        for sigma_true in sigma_true_grid:
            metrics = {"bias_v": [], "bias_sigma": [], "bias_h3": [], "bias_h4": []}
            for i in range(n_seeds):
                rng = np.random.default_rng(seed0 + i)
                g_mock = build_matched_mock(
                    st_ref, a_fit, v_true, sigma_true, h3_true, h4_true, rng
                )

                st = FitState(**st_fields)
                st.g = g_mock
                st.gerr = base_gerr.copy()
                st.ntot = 0
                # Zero-centered regardless of st_ref.v_center: a dedicated
                # validation sweep found data-driven recentering (now the
                # default for a real single fit via
                # kinextract.fitting._fit_map_sigl0_recenter) introduces
                # *worse* bias than a fixed zero point across this grid --
                # see test_assess_recovery_bias_uses_zero_centered_map_objective.
                st.v_center = 0.0

                try:
                    res = fit_state_map_with_optional_clean(st, cfg, a_fit.copy(), bounds)
                except Exception as exc:
                    log(
                        f"assess_recovery_bias: replicate failed "
                        f"(v_true={v_true}, sigma_true={sigma_true}, seed={seed0 + i}): {exc}"
                    )
                    continue
                if not getattr(res, "success", True):
                    continue

                a_best = np.asarray(res.x, float)
                _, b, *_ = evaluate_model_gp(a_best, st)
                gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
                metrics["bias_v"].append(gh["vherm"] - v_true)
                metrics["bias_sigma"].append(gh["sherm"] - sigma_true)
                metrics["bias_h3"].append(gh["h3"] - h3_true)
                metrics["bias_h4"].append(gh["h4"] - h4_true)

            n_ok = len(metrics["bias_v"])
            if n_ok == 0:
                log(
                    f"assess_recovery_bias: all {n_seeds} replicates failed at "
                    f"v_true={v_true}, sigma_true={sigma_true}; skipping this grid point"
                )
                continue
            entry = {"n_seeds": n_ok}
            for key, vals in metrics.items():
                entry[key] = float(np.mean(vals))
                entry[f"{key}_std"] = float(np.std(vals))
            results[(float(v_true), float(sigma_true))] = entry

    return results


def correct_recovered_losvd(v_recovered: float, sigma_recovered: float, bias_table: dict) -> dict:
    """Apply an empirical bias correction from an :func:`assess_recovery_bias` table.

    Interpolates `bias_table` at `(v_recovered, sigma_recovered)` (linear
    interpolation within the grid's convex hull, nearest-neighbor outside
    it or if fewer than 4 grid points are available) and returns
    bias-corrected point estimates plus an inflated uncertainty
    incorporating the bias table's own seed-to-seed scatter at that
    location -- an empirically-calibrated alternative to the retired
    analytic :func:`kinextract.errors.bias_corrected_losvd`, which is
    known to be unreliable near the instrumental resolution limit (see
    that function's docstring).

    Parameters
    ----------
    v_recovered, sigma_recovered : float
        A real fit's recovered velocity/dispersion
        (:func:`kinextract.losvd.fit_losvd_gauss_hermite`'s ``vherm``/
        ``sherm``) to correct.
    bias_table : dict
        Output of :func:`assess_recovery_bias`, built from mocks matched
        to the same target/instrument/configuration as the fit being
        corrected.

    Returns
    -------
    dict
        ``v_corrected``, ``sigma_corrected`` (bias-subtracted point
        estimates) and ``v_uncertainty_inflation``,
        ``sigma_uncertainty_inflation`` (the interpolated seed-to-seed
        standard deviation of the bias at this location, to be added in
        quadrature to whatever statistical uncertainty
        :class:`~kinextract.errors.LOSVDErrorEstimator` reports).
    """
    points = np.array(list(bias_table.keys()), dtype=float)
    if len(points) == 0:
        raise ValueError("bias_table is empty; run assess_recovery_bias first")
    query = np.array([[v_recovered, sigma_recovered]])

    def _interp(key: str) -> float:
        values = np.array([bias_table[tuple(p)][key] for p in map(tuple, points)])
        if len(points) < 4:
            val = griddata(points, values, query, method="nearest")[0]
        else:
            val = griddata(points, values, query, method="linear")[0]
            if not np.isfinite(val):
                val = griddata(points, values, query, method="nearest")[0]
        return float(val)

    bias_v = _interp("bias_v")
    bias_sigma = _interp("bias_sigma")

    return {
        "v_corrected": v_recovered - bias_v,
        "v_uncertainty_inflation": _interp("bias_v_std"),
        "sigma_corrected": sigma_recovered - bias_sigma,
        "sigma_uncertainty_inflation": _interp("bias_sigma_std"),
    }
