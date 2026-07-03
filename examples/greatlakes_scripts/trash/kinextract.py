"""
spectral_fitting.py
====================
Spectral fitting pipeline for LOSVD extraction.
 
Two operating modes, selected by the config file:
 
  1. PRE-NORMALISED  (fit_als_continuum = false in config)
       Reads a .norm file and fits only the LOSVD with no polynomial
       continuum.  The spectrum must already be continuum-normalised.
 
  2. ALS CONTINUUM   (fit_als_continuum = true in config)
       Reads a raw .spec file (or a .norm file with orig_flux columns).
       An asymmetric-least-squares (ALS) baseline is iteratively updated
       around the LOSVD model (outer loop).  The ALS continuum is stored
       in FitState.continuum_mult and applied multiplicatively inside
       evaluate_model_gp.
 
Usage
-----
    python spectral_fitting.py fit_config.toml
 
The TOML config file contains all tunable parameters; see fit_config.toml.
 
Changes vs. original notebook-merged version
--------------------------------------------
* Polynomial continuum mode (mdeg/adeg >= 0) removed entirely.
* All CaII-triplet continuum fitter code removed (fit_caii_continuum,
  optimize_als_hyperparams outer-loop version, caii_observed_centers,
  build_line_regions, detect_extra_lines, bloom_mask, score_continuum,
  make_line_mask, create_wavelength_grid).
* norm_continuum_flux config path (polynomial-from-.norm) removed.
* mdeg and adeg are now hard-wired to -1 internally; poly matrices are
  never built, saving memory and time.
* evaluate_model_gp: poly correction branches removed; bincount path
  cleaned (no redundant .ravel() on already-flat arrays).
* asymmetric_least_squares_continuum: DTD matrix now precomputed outside
  the iteration loop (was rebuilt every iteration).
* interp_template_tp_with_outside: outside-range pixels now tracked via
  a dedicated boolean mask instead of the fragile sentinel tp==1.0.
* build_initial_guess_nonparam: template weight upper bound raised to
  1.0 + epsilon to avoid the boundary-equals-initial-value degeneracy
  when nt == 1.
* Config is read from a TOML file (tomllib on Python >= 3.11, tomli
  package on older versions).  All FitConfig fields are settable from
  the config file with human-readable comments.
"""
 
from __future__ import annotations
 
import math
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares, minimize
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None
    warnings.warn(
        "JAX is not installed or failed to import.  The MAP optimizer and "
        "bootstrap will fall back to scipy finite-difference gradients, which "
        "are 20-100x slower per L-BFGS-B step.  Install with: pip install jax[cpu]",
        RuntimeWarning,
        stacklevel=1,
    )

import threading as _threading

# Process-wide cache for JAX JIT-compiled value+grad functions.
# Keyed by a shape-signature tuple so that the XLA kernel is compiled once per
# unique (nl, nt, npix, …) combination and then shared by ALL callers in the
# same process — including every bootstrap thread.  Without this, every call to
# _build_jax_objective_value_and_grad creates a new Python closure → new JAX
# trace → new XLA compilation (~30-120 s each).  With 1000 bootstrap replicates
# that cost dominates the runtime.
_JAX_VG_CACHE: dict = {}
_JAX_VG_CACHE_LOCK = _threading.Lock()


def _get_or_build_jax_vg(st):
    """Return the cached JAX value+grad function for this FitState shape.

    Thread-safe: uses double-checked locking so the first thread to encounter a
    new shape compiles and all others wait for the result rather than racing to
    compile simultaneously.
    """
    # When icoff==0 or icoff==2, coffi/coff2i are baked into the JAX closure as
    # Python float constants.  Include them in the key so two spectra with the
    # same shape but different fixed coff values don't share a compiled kernel.
    key = (
        int(st.nl), int(st.nt), int(st.npix), int(st.nlosvd),
        int(st.icoff), bool(st.fit_global_amp),
        bool(st.fortran_template_mixture), bool(st.fit_als_continuum),
        float(st.xlam), float(st.sigl0),
        float(st.coffi)  if int(st.icoff) in (0, 2) else 0.0,
        float(st.coff2i) if int(st.icoff) == 0       else 0.0,
    )
    if key in _JAX_VG_CACHE:
        return _JAX_VG_CACHE[key]
    with _JAX_VG_CACHE_LOCK:
        if key not in _JAX_VG_CACHE:
            _JAX_VG_CACHE[key] = _build_jax_objective_value_and_grad(st)
    return _JAX_VG_CACHE[key]

# ── TOML parsing (stdlib on 3.11+, falls back to third-party tomli) ──────────
try:
    import tomllib                # Python >= 3.11
except ModuleNotFoundError:
    try:
        import tomli as tomllib   # pip install tomli
    except ModuleNotFoundError:
        tomllib = None            # type: ignore[assignment]
        warnings.warn(
            "Neither tomllib (Python >= 3.11 stdlib) nor tomli (pip install tomli) "
            "could be imported.  Config files cannot be loaded — the framework "
            "will crash when attempting to read kinextract.config.",
            ImportWarning,
            stacklevel=1,
        )

try:
    plt.style.use("~/.mplstyle")
except Exception:
    pass

try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else lambda f: f
    warnings.warn(
        "Numba is not installed or failed to import.  JIT-compiled inner loops "
        "(chi-squared, convolution, ALS) will run as plain Python, which is "
        "10-50x slower.  Install with: pip install numba",
        RuntimeWarning,
        stacklevel=1,
    )
 
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CEE = 299792.458   # speed of light, km/s (IAU 2012)
BIG = 1e10     # sentinel value for masked pixels
 
_T0 = time.perf_counter()
 
 
def log(msg: str) -> None:
    print(f"[{time.perf_counter() - _T0:9.2f}s] {msg}", flush=True)
 
 
class Timer:
    def __init__(self, label: str):
        self.label = label
 
    def __enter__(self):
        self.t0 = time.perf_counter()
        log(f"START {self.label}")
        return self
 
    def __exit__(self, *_):
        log(f"END   {self.label} ({time.perf_counter() - self.t0:.2f}s)")
 
 
# =============================================================================
# Section 1 - ALS continuum utilities
# =============================================================================
 
def robust_sigma(values: np.ndarray) -> float:
    """Robust sigma via normalised MAD."""
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
    Asymmetric least-squares baseline (Eilers 2003).

    Parameters
    ----------
    y            : 1-D flux array.
    base_mask    : Boolean mask; True = pixel participates in the fit.
    lam          : Smoothing strength (larger -> smoother baseline).
    p            : Asymmetry weight.
    niter        : Number of re-weighting iterations.
    eps          : Minimum weight floor (prevents singular systems).
    return_weights: If True, return (z, w_final, dtd_main, dtd_off1, dtd_off2)
                   so the caller can compute trace(H_als) without re-running ALS.

    The lam * D'D penalty bands are precomputed once outside the loop;
    only the weight diagonal w is updated each iteration.
    """
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
    """
    Estimate trace(H_als) = trace((W + lam*D'D)^{-1} W) via the Hutchinson
    stochastic trace estimator.

    For each random Rademacher vector v ~ {-1, +1}^L:
        E[v^T (W + lam*D'D)^{-1} W v] = trace((W + lam*D'D)^{-1} W)

    Uses the same banded structure already in asymmetric_least_squares_continuum.
    Cost: n_probes banded solves (same bandwidth-4 system, O(n) each).
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
    Score an ALS continuum candidate.

    Modes
    -----
    1. model-score mode:
         If score_data, score_err, and score_model0 are supplied, score

             score_data ~= score_model0 * continuum

         This is the correct mode during ALS outer updates, where score_model0
         is the current LOSVD/template model without continuum.

    2. template-score mode:
         If templates are supplied, score the continuum-normalised spectrum
         against a weighted template combination.

         This is useful mainly for the initial raw-spectrum continuum.

    3. continuum-target mode:
         Otherwise score

             y ~= continuum

         using yerr.
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
    Optimize ALS lambda/p by reduced chi^2 (or BIC) on the chosen scoring target.

    When use_bic=True, the selection criterion becomes:

        score = chi2_total + bic_dof_penalty * k_eff * ln(n)

    where chi2_total = chi2_red * n is the total chi-squared, k_eff =
    trace((W + lam*D'D)^{-1} W) is the effective degrees of freedom of the
    ALS smoother estimated via the Hutchinson stochastic trace estimator
    (hutchinson_probes random Rademacher vectors), and n is the number of
    good pixels.  BIC penalises lower-lambda (wigglier) continua that would
    otherwise always win on raw chi-squared alone.

    chisq_floor (default 0.0 = disabled): hard-reject any (lam, p) where
    chi2_red < chisq_floor.  Continua that absorb more variance than the
    noise budget allows produce chi2_red < 1; setting chisq_floor = 0.9
    prevents this regardless of BIC.

    During ALS outer updates, use score_data / score_err / score_model0
    so candidate continua are scored as:

        score_data ~= score_model0 * continuum

    rather than by fitting templates to the continuum-corrected target.
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
    Fit ALS baseline to y.

    If cfg.als_optimize=True and force_fixed=False, optimize lambda/p.

    During ALS updates, pass score_data, score_err, score_model0 so the
    continuum is optimized against the current spectral model.
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
    Grow a boolean mask by grow_pix pixels on each side.

    Parameters
    ----------
    mask : array-like bool
        Input mask where True means rejected/masked.
    grow_pix : int
        Number of pixels by which to grow the mask on each side.

    Returns
    -------
    grown : ndarray bool
        Grown mask.
    """
    mask = np.asarray(mask, bool)
    if grow_pix <= 0 or mask.size == 0:
        return mask.copy()
    kernel = np.ones(2 * grow_pix + 1, dtype=int)
    grown = np.convolve(mask.astype(int), kernel, mode="same") > 0
    return grown


def grow_boolean_mask_A(mask: np.ndarray, x: np.ndarray, grow_A: float) -> np.ndarray:
    """
    Grow a boolean mask by approximately grow_A Angstroms on each side.
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
    Fit ALS continuum with pPXF-like absorption-pixel rejection.

    This does:

      1. Fit ALS using the supplied base_mask.
      2. Identify pixels significantly below the continuum.
      3. Grow that rejection mask.
      4. Refit ALS using the updated continuum-good mask.

    This is intended only for continuum estimation. It should not mask the
    absorption lines from the LOSVD/template fit.

    Returns
    -------
    continuum : ndarray
        Final ALS continuum.
    lam : float
        Lambda used.
    p : float
        Asymmetry parameter used.
    records : list
        Diagnostic records.
    final_base_mask : ndarray bool
        Final continuum-good-pixel mask.
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


def read_galaxy_params(path: str | Path) -> dict:
    """
    Read a legacy galaxy.params file.

    Expected format:
        First non-comment line = galaxy name
        Remaining lines = key value [optional comments]

    Example:
        NGC3258
        distanceMpc  31.9
        vmin        -1100.
        vmax         1100.

    Returns
    -------
    params : dict
        Parsed values. Numeric values are converted to int or float when
        possible. The first line is stored as params["galaxy_name"].
    """
    path = Path(path)

    # Accept a directory: look for galaxy.params inside it.
    if path.is_dir():
        path = path / "galaxy.params"

    if not path.exists():
        raise FileNotFoundError(f"galaxy.params file not found: {path}")

    params: dict = {}
    got_name = False

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            # First non-comment line is the galaxy name.
            if not got_name:
                params["galaxy_name"] = line.split()[0]
                got_name = True
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            key = parts[0]
            val_raw = parts[1]

            # Convert value to int/float where possible.
            try:
                val_float = float(val_raw)

                if np.isfinite(val_float) and val_float.is_integer():
                    val = int(val_float)
                else:
                    val = val_float

            except Exception:
                val = val_raw

            params[key] = val

    return params


# =============================================================================
# Section 2 - Configuration dataclass
# =============================================================================
 
@dataclass
class FitConfig:
    # ── Paths ────────────────────────────────────────────────────────────────
    # gal_file is optional for command-line mode, but useful in notebooks.
    gal_file: str = ""
    template_list_file: str = "Tlist"
    template_dir: Optional[str] = None
    regions_bad_path: str = "regions.bad"
    outdir: Optional[str] = "."

    # ── Wavelength / redshift ───────────────────────────────────────────────
    # wavemin_full / step are needed for raw index-format files.
    # For .norm files only step is needed when fortran_rebin_norm=True.
    wavemin_full: Optional[float] = None
    step: Optional[float] = None
    wavefitmin: float = -np.inf
    wavefitmax: float = np.inf
    zgal: float = 0.0

    # How to interpret the wavelength column of a .norm file:
    #   "auto"     - pick whichever frame puts more pixels inside fit window
    #   "rest"     - use the column as-is
    #   "observed" - divide by 1 + zgal
    norm_wave_frame: str = "auto"

    # ── Kinematic grid ──────────────────────────────────────────────────────
    galaxy_params_path: Optional[str] = None
    # If True and galaxy_params_path is supplied, use vmin/vmax from that file
    use_galaxy_params_velocity_bounds: bool = True

    sigl: float = 100.0
    xlam: float = 300.0
    smoothing: Optional[float] = None

    # ── Auto smoothing parameter selection ─────────────────────────────────
    # If True, search xlam_auto_grid (ascending) for the minimum xlam that
    # produces a smooth (non-jagged) LOSVD.  The selected value is stored in
    # both st.xlam and cfg.xlam so all subsequent MAP fits and bootstrap
    # refits use it.  Bootstrap workers always set xlam_auto=False (via
    # _make_frozen_cfg) so the search runs only once per spectrum.
    xlam_auto: bool = False
    xlam_auto_grid: tuple = (20., 50., 100., 250., 600., 1500., 4000.)
    # LOSVD roughness threshold: peak-normalised max second-difference.
    # A smooth Gaussian with σ bins has inherent roughness ≈ 2(1-exp(-1/2σ²)).
    # For the default 29-bin, ±250 km/s grid (17.9 km/s/bin) this is:
    #   σ=2.5 bins (44 km/s) → 0.16,  σ=3 bins (54 km/s) → 0.11
    #   σ=6 bins (107 km/s)  → 0.03
    # Threshold must exceed the inherent roughness for the narrowest LOSVD in
    # the dataset; 0.25 works for σ ≳ 40 km/s on the default grid.
    xlam_smooth_threshold: float = 0.25
    # Unimodality constraint: maximum number of peaks allowed in the LOSVD.
    # A peak counts only if it exceeds xlam_peak_min_prominence * global_peak.
    # Default is 1 (unimodal). A double-peaked LOSVD at low xlam is a fitting
    # artifact — real kinematic complexity is captured by Gauss-Hermite moments.
    xlam_max_peaks: int = 1
    xlam_peak_min_prominence: float = 0.1
    # Max iterations for each search fit (None → use map_maxiter).
    # The search only needs an approximate LOSVD shape, so a smaller budget
    # (e.g., 2000) speeds up the search without affecting accuracy.
    xlam_auto_maxiter: Optional[int] = None

    losvd_vmin: Optional[float] = None
    losvd_vmax: Optional[float] = None

    # Number of non-parametric LOSVD bins.
    n_losvd_bins: int = 29

    # ── Continuum mode ──────────────────────────────────────────────────────
    # False = pre-normalised mode
    # True  = ALS continuum mode
    fit_als_continuum: bool = False

    # ── Pre-normalised mode options ─────────────────────────────────────────
    norm_error_mode: str = "unit"  # "unit" or "file"
    fortran_rebin_norm: bool = True
    norm_apply_fortran_flux_mask: bool = True
    norm_fortran_flux_min: float = 0.0
    norm_fortran_flux_max: float = 1.3

    # ── ALS continuum options ───────────────────────────────────────────────
    als_lam: float = 1e7
    als_p: float = 0.05
    als_niter: int = 20
    als_outer_iter: int = 4
    als_outer_tol: float = 1e-3
    als_eps: float = 1e-6

    # If True, optimize lam/p only for the initial ALS continuum, then reuse.
    als_optimize_init_only: bool = False

    # Clamp ALS continuum values.
    als_clip: tuple = (0.0, float("inf"))

    # Whether column 3 of a raw .spec file is variance or sigma.
    spec_col3_is_variance: bool = False

    # When False, ignore the per-pixel input errors and use the median error
    # value uniformly across all pixels (equal weighting).  The error column
    # is still read; only the weights used in fitting are replaced.
    use_spectrum_errors: bool = True

    # ── ALS hyperparameter optimization ─────────────────────────────────────
    als_optimize: bool = False
    als_lam_grid: tuple = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
    als_p_grid: tuple = (0.5,)
    als_lam_grid_fine_n: int = 5
    als_p_grid_fine_n: int = 5
    als_opt_verbose: bool = False
    als_template_selection: str = "lsq"  # "lsq" or "nnls"
    # BIC-based continuum overfitting protection.
    # als_use_bic=True: score = chi2_total + als_bic_dof_penalty * k_eff * ln(n)
    #   where k_eff = trace((W + lam*D'D)^{-1} W) estimated via Hutchinson.
    # als_chisq_floor: hard-reject any (lam, p) where chi2_red < this value.
    als_use_bic: bool = True
    als_chisq_floor: float = 0.9
    als_hutchinson_probes: int = 25
    als_bic_dof_penalty: float = 1.0
    # Legacy field kept for config compatibility.
    als_bic_include_templates: bool = False
    # Optional minimum lambda to prevent pathological low-lambda fits
    als_lambda_floor: Optional[float] = None

    # ── ALS continuum line masks ────────────────────────────────────────────
    # These masks affect only the ALS continuum fit, not the LOSVD/template fit.
    als_mask_ca: bool = True
    als_ca_centers: tuple = (8498.02, 8542.09, 8662.14)
    als_ca_half_widths: tuple = (14.0, 16.0, 18.0)
    als_ca_frame: str = "rest"  # "rest" or "observed"
    mask_emission_line_velocity_pad_kms: float = 300.0

    # Robustness against imperfect redshift / systemic velocity.
    # At 8500 Å, 75 km/s ≈ 2.1 Å.
    als_mask_velocity_pad_kms: float = 0.0

    # Empirical rest-frame shift applied to all ALS line/window masks.
    # If all masks are too blue by 1 Å, set +1.0.
    als_mask_center_shift_A: float = 0.0

    # Auto-refine the mask shift from the Ca II triplet position so that mask
    # centers track the actual spectral features even when zgal is slightly off.
    # Masking-only: has NO effect on velocity/LOSVD measurements.
    als_auto_caii_shift: bool = True
    als_auto_caii_search_hw_A: float = 12.0

    # Extra wavelength windows to exclude from ALS continuum fitting.
    # Format: ((lo, hi), ...)
    als_extra_mask_windows: tuple = ()
    als_extra_mask_frame: str = "rest"  # "rest" or "observed"

    # Extra line-center masks (ALS only).
    # Format: ((center, half_width), ...) or ((center, half_width, "name"), ...)
    als_extra_mask_lines: tuple = ()
    als_extra_mask_line_frame: str = "rest"  # "rest" or "observed"

    # Convenience field: extra absorption lines applied to BOTH ALS masking and
    # kinematic cleaning protection so a single entry is sufficient.
    # Format: ((center_A, half_width_A), ...) or ((center_A, half_width_A, "name"), ...)
    # All wavelengths are rest-frame Angstroms.
    extra_absorption_lines: tuple = ()

    # Built-in common Ca-triplet-region features (manual, fixed half-widths).
    als_use_default_caii_region_lines: bool = False
    als_default_abs_half_width: float = 2.0
    als_default_em_half_width: float = 2.5
    als_default_pas_half_width: float = 2.0

    # S/N-based auto-detection of strong absorption lines for ALS masking.
    # Uses the same line tables as the kinematic clean-protect mask.
    # Half-width defaults to als_default_abs_half_width when not set.
    als_auto_mask_abs_lines: bool = False
    als_auto_mask_paschen: bool = False
    als_auto_mask_snr_threshold: float = 5.0
    als_auto_mask_half_width_A: float = 5.0
    als_auto_mask_snr_context_A: float = 20.0

    # ── pPXF-like ALS continuum good-pixel cleaning ─────────────────────────
    als_abs_clean: bool = True
    als_abs_clean_iter: int = 25
    als_abs_clean_sigma: float = 2.5
    als_abs_clean_grow_A: float = 3.0
    als_abs_clean_two_sided: bool = False
    als_abs_clean_minpix: int = 50
    als_abs_clean_reoptimize_final: bool = False
    # When True, abs_clean runs only during init_als_continuum; all outer
    # iterations reuse the absorption mask established at init.  Eliminates
    # O(als_outer_iter) redundant absorption-cleaning passes.
    als_abs_clean_init_only: bool = False

    # ── LOSVD / template options ────────────────────────────────────────────
    icoff: int = 1
    coff: float = 0.0
    coff2: float = 0.0
    fortran_nlosvd_full_x: bool = True
    fortran_template_mixture: bool = True
    fortran_mask_template_outside: bool = True
    fit_global_amp: bool = False
    continuum_poly_mode: str = "none"  # "none", "additive", or "multiplicative"
    continuum_poly_bound: float = 0.1

    # ── Optimizer settings ──────────────────────────────────────────────────
    map_maxiter: int = 10000
    map_maxfun: int = 200000
    map_ftol: float = 2e-14
    map_gtol: float = 1e-10
    map_maxls: int = 50
    use_scaled_optimizer: bool = True
    use_jax_objective: bool = False
    jax_enable_x64: bool = True
    print_every: int = 200

    # ── Kinematic-fit sigma-clipping / cleaning ─────────────────────────────
    # This is separate from ALS continuum good-pixel cleaning.
    clean: bool = False
    clean_sigma: float = 3.0
    clean_maxiter: int = 5
    clean_minpix: int = 10
    clean_bloom_pixels: int = 0

    clean_protect_ca_triplet: bool = True
    clean_ca_centers: tuple = (8498.02, 8542.09, 8662.14)
    clean_ca_half_width: float = 12.0
    clean_protect_absorption_only: bool = True
    clean_protect_ca_frame: str = "rest"
    clean_protect_windows: tuple = ()
    clean_protect_windows_frame: str = "rest"

    # ── Emission line pre-masking in spectral fit ───────────────────────────
    # Stellar templates model only absorption; emission lines must be excluded
    # from chi-squared before fitting.  Pixels are set to gerr=BIG in
    # make_fit_state so they are excluded from both ALS and LOSVD fits.
    # Detection uses the local excess above the flanking continuum so that
    # lines absent from the spectrum are not masked unnecessarily.
    mask_emission_lines_in_fit: bool = True
    mask_emission_line_half_width_A: float = 5.0
    # Only pre-mask emission lines whose excess above local flanking continuum
    # exceeds this S/N.  3.0 catches most real emission while avoiding noise spikes.
    mask_emission_line_snr_threshold: float = 3.0
    mask_emission_line_snr_context_A: float = 20.0  # ±Å window for local S/N estimate

    # After emission-line detection, grow the combined mask by this many Å on
    # each side.  This covers line wings that fall just below the SNR threshold
    # and merges closely-spaced emission lines into a single contiguous masked
    # region (no unmasked continuum between adjacent lines to confuse the ALS
    # or sigma-clipping).  Set to mask_emission_line_half_width_A or larger to
    # merge lines within ~2× half_width_A of each other.
    mask_emission_grow_A: float = 0.0

    # Segment-wise rolling-median upward-outlier detection.
    # For each CaT-free continuum segment, flags pixels above
    # local_rolling_median + n_sigma × MAD.  This catches emission wings and
    # features not in the known emission-line table that would otherwise
    # contaminate the ALS continuum base or bias the MAP fit.
    segment_emission_mask: bool = True
    segment_emission_n_sigma: float = 3.0
    segment_emission_win_A: float = 50.0  # rolling window half-width in Å

    # If True, treat H I Paschen lines as possible emission contaminants in the
    # spectral fit. This is especially important for Ca II triplet fitting because
    # Pa16, Pa15, and Pa13 overlap Ca II 8498, 8542, and 8662.
    mask_paschen_lines_in_fit: bool = False

    def __post_init__(self):
        if self.smoothing is not None:
            self.xlam = float(self.smoothing)

        # Read legacy galaxy.params if requested.
        self.galaxy_params = {}

        if self.galaxy_params_path is not None:
            _gp = Path(self.galaxy_params_path)
            if _gp.is_dir():
                _gp = _gp / "galaxy.params"
            if _gp.exists() and _gp.is_file():
                self.galaxy_params = read_galaxy_params(_gp)

            if self.use_galaxy_params_velocity_bounds:
                if "vmin" in self.galaxy_params:
                    self.losvd_vmin = float(self.galaxy_params["vmin"])

                if "vmax" in self.galaxy_params:
                    self.losvd_vmax = float(self.galaxy_params["vmax"])

        self.norm_error_mode = self.norm_error_mode.lower().strip()
        if self.norm_error_mode not in ("unit", "file"):
            raise ValueError("norm_error_mode must be 'unit' or 'file'")

        if self.als_clip[0] < 0:
            raise ValueError("als_clip[0] must be >= 0")

        self.continuum_poly_mode = self.continuum_poly_mode.lower().strip()
        if self.continuum_poly_mode not in ("none", "additive", "multiplicative"):
            raise ValueError(
                "continuum_poly_mode must be 'none', 'additive', or 'multiplicative'"
            )

        if self.fit_als_continuum and self.continuum_poly_mode != "none":
            raise ValueError(
                "continuum_poly_mode is only supported when fit_als_continuum=False"
            )

        if self.fit_als_continuum and self.fit_global_amp:
            raise ValueError(
                "fit_als_continuum=True should not be combined with "
                "fit_global_amp=True because the ALS continuum and global "
                "amplitude are degenerate."
            )

        # Normalize and validate frame strings.
        frame_attrs = [
            "norm_wave_frame",
            "als_ca_frame",
            "als_extra_mask_frame",
            "als_extra_mask_line_frame",
            "clean_protect_ca_frame",
            "clean_protect_windows_frame",
        ]

        for attr in frame_attrs:
            val = getattr(self, attr).lower().strip()
            setattr(self, attr, val)

        if self.norm_wave_frame not in ("auto", "rest", "observed"):
            raise ValueError("norm_wave_frame must be 'auto', 'rest', or 'observed'")

        for attr in [
            "als_ca_frame",
            "als_extra_mask_frame",
            "als_extra_mask_line_frame",
            "clean_protect_ca_frame",
            "clean_protect_windows_frame",
        ]:
            val = getattr(self, attr)
            if val not in ("rest", "observed"):
                raise ValueError(f"{attr} must be 'rest' or 'observed'")

        if self.losvd_vmin is not None and self.losvd_vmax is not None:
            if self.losvd_vmax <= self.losvd_vmin:
                raise ValueError(
                    "LOSVD velocity bounds invalid: "
                    f"losvd_vmin={self.losvd_vmin}, "
                    f"losvd_vmax={self.losvd_vmax}"
                )

        if int(self.n_losvd_bins) < 3:
            raise ValueError("n_losvd_bins must be >= 3")
 
 
# =============================================================================
# Section 3 - Config I/O (TOML)
# =============================================================================
 
def _lists_to_tuples(obj):
    """Recursively convert TOML lists to tuples."""
    if isinstance(obj, list):
        return tuple(_lists_to_tuples(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _lists_to_tuples(v) for k, v in obj.items()}
    return obj


def load_config_from_toml(path: str | Path) -> FitConfig:
    """
    Parse a TOML config file and return a FitConfig.

    Requires Python >= 3.11 or the tomli package.
    """
    if tomllib is None:
        raise ImportError(
            "TOML parsing requires Python >= 3.11 or 'pip install tomli'."
        )

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    # Flatten nested tables into FitConfig's flat namespace.
    flat: dict = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)

    # Allow top-level scalar keys too.
    flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})

    flat = _lists_to_tuples(flat)

    # Handle inf in als_clip.
    if "als_clip" in flat:
        als_clip_raw = flat["als_clip"]
        lo = float(als_clip_raw[0])
        hi_raw = als_clip_raw[1]
        hi = (
            float("inf")
            if isinstance(hi_raw, str) and hi_raw.lower() == "inf"
            else float(hi_raw)
        )
        flat["als_clip"] = (lo, hi)

    known = {f.name for f in FitConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    # Wrapper-level controls that may live in kinextract.config but are not
    # FitConfig fields (consumed by the shell wrapper / losvd_errs flow).
    allowed_wrapper_keys = {
        "run_errors_default",
        "run_bootstrap_default",
        "n_bootstrap",
        "n_jobs",
    }
    for k in list(flat.keys()):
        if k not in known:
            if k not in allowed_wrapper_keys:
                log(f"WARNING: unknown config key '{k}' ignored")
            del flat[k]

    return FitConfig(**flat)
 
 
# =============================================================================
# Section 4 - I/O helpers
# =============================================================================
 
def read_galaxy_index_flux_err(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"{path}: expected >= 3 columns (index-or-wave, flux, flux_err_or_variance)"
        )
    return arr[:, 0], arr[:, 1], arr[:, 2]
 
 
def read_norm_spectrum(path: str) -> dict:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path}: .norm file must have at least 3 columns")
    keys = ["wavelength", "normflux", "normflux_err",
            "orig_flux", "orig_flux_err", "continuum", "continuum_err"]
    data: dict = {k: None for k in keys}
    for i, k in enumerate(keys[:3]):
        data[k] = arr[:, i].astype(float)
    if arr.shape[1] >= 7:
        for i, k in enumerate(keys[3:], 3):
            data[k] = arr[:, i].astype(float)
    return data
 
 
def read_template_xy(path: str) -> tuple[np.ndarray, np.ndarray, "np.ndarray | None"]:
    """Load a template file and return (wavelength, flux, flux_err | None).

    Supports 2-column (wave, flux) and 3-column (wave, flux, flux_err) files.
    Files may be .txt (normalized flux) or .dat (physical flux; will be
    median-normalized before use).
    """
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Template {path} must have >= 2 columns")
    wave = arr[:, 0].astype(float)
    flux = arr[:, 1].astype(float)
    err = arr[:, 2].astype(float) if arr.shape[1] >= 3 else None
    return wave, flux, err
 
 
def read_template_list(list_file: str, template_dir: Optional[str] = None) -> list[str]:
    list_file = Path(list_file)
    base = Path(template_dir) if template_dir else list_file.parent
    files = []
    with open(list_file) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s)
            files.append(str(p if p.is_absolute() else base / p))
    if not files:
        raise ValueError(f"No templates found in {list_file}")
    return files
 
 
def infer_output_prefix(gal_file: str) -> str:
    p = Path(gal_file)
    for suf in [".norm", ".spec", ".txt", ".dat"]:
        if p.name.endswith(suf):
            return p.name[: -len(suf)]
    return p.stem
 
 
def nint_fortran(z: np.ndarray) -> np.ndarray:
    """Fortran NINT rounding (round half away from zero)."""
    z = np.asarray(z)
    return np.where(z >= 0, np.floor(z + 0.5), np.ceil(z - 0.5)).astype(int)
 
 
def setbadreg(x: np.ndarray, gerr: np.ndarray, iflip: int, regions_bad_path: str) -> np.ndarray:
    """Set gerr = BIG for pixels listed in a bad-region file."""
    if not regions_bad_path:
        return gerr
    p = Path(regions_bad_path)
    if not p.exists() or not p.is_file():
        return gerr
    for x1, x2 in np.atleast_2d(np.loadtxt(p)):
        m = (x >= iflip * x1) & (x <= iflip * x2)
        gerr[m] = BIG
    return gerr
 
 
def build_wavelength_from_index(npts: int, wavemin_full: float, step: float) -> np.ndarray:
    return wavemin_full + step * np.arange(npts)
 
 
def select_region_with_errors(
    x: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
    xmin: float, xmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = (x > xmin) & (x < xmax)
    return x[m].copy(), flux[m].copy(), flux_err[m].copy(), m
 
 
def estimate_step_from_wavelength(x: np.ndarray) -> float:
    dx = np.diff(np.asarray(x, float))
    dx = dx[np.isfinite(dx)]
    if dx.size == 0:
        raise ValueError("Could not estimate wavelength step: no finite differences.")
    step = float(np.median(dx))
    if step <= 0:
        raise ValueError(f"Non-positive wavelength step estimated: {step}")
    return step
 
 
def count_in_window(x: np.ndarray, xmin: float, xmax: float) -> int:
    return int(np.count_nonzero((np.asarray(x) > xmin) & (np.asarray(x) < xmax)))
 
 
# =============================================================================
# Section 5 - Template handling
# =============================================================================
 
def interp_template_tp_with_outside(
    xg: np.ndarray, xt: np.ndarray, ft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate template (xt, ft) onto galaxy grid xg.
 
    Returns
    -------
    tp      : Interpolated template values.  Pixels outside the template
              range are set to 1.0 (flat / neutral).
    outside : Boolean mask; True where xg is outside [xt[0], xt[-1]).
 
    Fix vs. original
    ----------------
    The outside mask is now a dedicated array instead of relying on the
    fragile ``T == 1.0`` sentinel, which could misidentify on-range pixels
    whose true value happens to be exactly 1.0.
    """
    xg, xt, ft = np.asarray(xg, float), np.asarray(xt, float), np.asarray(ft, float)
    tp = np.ones(len(xg), float)
    outside = (xg < xt[0]) | (xg >= xt[-1])
 
    j1 = np.searchsorted(xt, xg, side="right")
    j0 = j1 - 1
    inside = ~outside & (j0 >= 0) & (j1 < len(xt))
 
    jj = j0[inside]
    x0, x1 = xt[jj], xt[jj + 1]
    f0, f1 = ft[jj], ft[jj + 1]
    ok = (f0 > 0) & (f1 > 0) & (x1 != x0)
    val = np.ones(np.count_nonzero(inside), float)
    frac = np.zeros_like(val)
    frac[ok] = (xg[inside][ok] - x0[ok]) / (x1[ok] - x0[ok])
    val[ok] = f0[ok] + (f1[ok] - f0[ok]) * frac[ok]
    tp[inside] = val
    return tp, outside
 
 
def build_template_matrix_fortran(
    xg: np.ndarray, template_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (npix × ntemplates) template matrix on the galaxy grid.

    Each template is median-normalised so that values are ≈ 1.0.  This is
    required when templates are in physical flux units (e.g. the MUSE Library
    .dat files in erg cm⁻² s⁻¹ Å⁻¹) so the LOSVD fit can match the galaxy
    spectrum whose continuum level is also ≈ 1 after ALS normalisation.

    Returns
    -------
    T           : Template matrix (npix × ntemplates), values ≈ 1.
    T_err       : Template error matrix (npix × ntemplates), same normalisation.
                  Zero where the template file had no error column.
    outside_any : Boolean mask; True at pixels outside *all* templates.
    """
    T = np.empty((len(xg), len(template_paths)), float)
    T_err = np.zeros_like(T)
    outside_any = np.zeros(len(xg), dtype=bool)
    for k, p in enumerate(template_paths):
        wave, flux, err = read_template_xy(p)
        # Normalize to median positive flux so template ≈ 1.0 (shape only).
        pos = flux > 0
        med = float(np.nanmedian(flux[pos])) if pos.any() else 1.0
        if med > 0:
            flux = flux / med
            if err is not None:
                err = err / med
        tp, outside = interp_template_tp_with_outside(xg, wave, flux)
        T[:, k] = tp
        outside_any |= outside
        if err is not None:
            te, _ = interp_template_tp_with_outside(xg, wave, err)
            T_err[:, k] = te
    return T, T_err, outside_any
 
 
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
    if cache_path is None:
        return
    import json
    from pathlib import Path

    try:
        data = {"_cache_key": cache_key, "lines": results}
        Path(cache_path).write_text(json.dumps(data, indent=2))
    except Exception as e:
        log(f"query_nist_lines: could not write cache: {e}")


def _auto_absorption_protect_mask(st: "FitState", cfg: "FitConfig") -> np.ndarray:
    """
    Build a boolean protect mask for known strong stellar absorption lines that
    fall within the current fit window and have sufficient local S/N.

    Algorithm
    ---------
    For each "abs"-tagged line in the built-in tables (and optionally "paschen"
    lines if cfg.clean_protect_paschen is True):

    1.  Check the rest-frame line center is within the fit window
        [wavefitmin, wavefitmax].
    2.  Compute local continuum S/N in a ±snr_context_A window around the
        line center using  median(g) / median(gerr).  The median is robust to
        the absorption dip itself and reflects continuum quality.
    3.  If S/N >= clean_auto_line_snr_threshold, protect ±half_width_A pixels
        around the center.

    Returns a boolean array of length st.npix.
    """
    protect = np.zeros(st.npix, dtype=bool)

    if not getattr(cfg, "clean_protect_auto_lines", True):
        return protect

    half_w = float(getattr(cfg, "clean_auto_line_half_width_A", 4.0))
    snr_ctx = float(getattr(cfg, "clean_auto_line_snr_context_A", 20.0))
    snr_thr = float(getattr(cfg, "clean_auto_line_snr_threshold", 5.0))
    protect_paschen = bool(getattr(cfg, "clean_protect_paschen", False))

    wmin = float(getattr(cfg, "wavefitmin", st.x.min()))
    wmax = float(getattr(cfg, "wavefitmax", st.x.max()))

    gerr_pos = np.where(st.gerr > 0, st.gerr, np.nan)

    for table in _STELLAR_ABSORPTION_LINE_TABLES:
        for tag, name, cen_rest in table:
            if tag == "em":
                continue
            if tag == "paschen" and not protect_paschen:
                continue

            if cen_rest < wmin or cen_rest > wmax:
                continue

            # st.x is already rest-frame (load_spectrum_for_fit divides by 1+z)
            ctx = (st.x >= cen_rest - snr_ctx) & (st.x <= cen_rest + snr_ctx)
            if ctx.sum() < 3:
                continue

            g_ctx = st.g[ctx]
            e_ctx = gerr_pos[ctx]
            g_med = float(np.nanmedian(g_ctx))
            e_med = float(np.nanmedian(e_ctx))

            if not (np.isfinite(g_med) and np.isfinite(e_med) and e_med > 0):
                continue

            local_snr = g_med / e_med
            if local_snr < snr_thr:
                continue

            line_pix = (st.x >= cen_rest - half_w) & (st.x <= cen_rest + half_w)
            if line_pix.any():
                protect |= line_pix
                log(
                    f"  auto-protect {name} {cen_rest:.2f} Å  "
                    f"S/N={local_snr:.1f}  npix={line_pix.sum()}"
                )

    return protect


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
    Build a boolean mask of pixels to exclude from the ALS baseline fit.

    True = exclude from ALS continuum fit.

    Important
    ---------
    This mask is for continuum fitting only. These pixels are still used in the
    LOSVD/template fit unless separately masked via gerr/regions.bad.

    For raw .spec inputs, st.x is rest-frame because load_spectrum_for_fit()
    divides by 1 + zgal. Observed-frame mask wavelengths are therefore
    converted by dividing by 1 + zgal.
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
            name = item[2] if len(item) >= 3 else f"{c_raw:.1f} Å"
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
    if getattr(cfg, "als_auto_mask_abs_lines", False):
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
    Initialise the ALS continuum from the raw flux.

    The baseline is fit directly to the data so that the first outer iteration
    starts from a sensible continuum estimate.
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
    Refine the ALS continuum after an LOSVD optimisation step.

    This computes:

        gp0 = model_without_ALS_continuum
        target = data / gp0

    Because gp0 is evaluated with apply_continuum=False, target is an absolute
    continuum estimate, not a multiplicative correction to the previous
    continuum.

    Returns
    -------
    delta : float
        Median fractional change in continuum relative to previous continuum.
    """
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
 
 
# =============================================================================
# Section 7 - FitState dataclass and precomputation
# =============================================================================
 
@dataclass
class FitState:
    x: np.ndarray
    g: np.ndarray
    gerr: np.ndarray
    t: np.ndarray
    # outside_tpl: True where at least one template does not cover the pixel.
    # Replaces the fragile t_missing = (T == 1.0) sentinel from the original.
    outside_tpl: np.ndarray
    c1: float
    resd: float
    xlam: float
    xl: np.ndarray
    sigl0: float
    npix: int
    nl: int
    iskip: int
    nt: int
    icoff: int
    nlosvd: int
    scale: float
    coffi: float = 0.0
    coff2i: float = 0.0
    prenormalized: bool = False
    norm_extra: dict = field(default_factory=dict)
    fortran_template_mixture: bool = True
    fit_global_amp: bool = False
    vgrid0: Optional[np.ndarray] = None
    facnew0: Optional[np.ndarray] = None
    losvd_j0: Optional[np.ndarray] = None
    losvd_j1: Optional[np.ndarray] = None
    losvd_w: Optional[np.ndarray] = None
    ip_map: Optional[np.ndarray] = None
    ip_mask: Optional[np.ndarray] = None
    t_err: Optional[np.ndarray] = None          # template error matrix (npix × nt)
    f_template: float = 0.0                    # pooled median fractional template error
    als_caii_refine_shift_A: float = 0.0       # Ca II measured wavelength shift (masking only)
    emission_pre_mask: Optional[np.ndarray] = None   # pixels pre-masked for emission lines
    continuum_mult: Optional[np.ndarray] = None
    continuum_mask: Optional[np.ndarray] = None
    continuum_mask_init: Optional[np.ndarray] = None   # absorption mask from init (for als_abs_clean_init_only)
    fit_als_continuum: bool = False
    als_lam_current: Optional[float] = None
    als_p_current: Optional[float] = None
    als_records: list = field(default_factory=list)
    continuum_poly_mode: str = "none"
    continuum_poly_x: Optional[np.ndarray] = None
    continuum_poly_bound: float = 0.1
    ntot: int = 0
    
    def __getstate__(self):
        """Serialize FitState for multiprocessing by converting numpy arrays to tuples."""
        state = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                # Store as (bytes, shape, dtype) tuple for compact serialization
                state[k] = ("__ndarray__", v.tobytes(), v.shape, str(v.dtype))
            elif isinstance(v, dict):
                state[k] = v
            else:
                state[k] = v
        return state
    
    def __setstate__(self, state):
        """Deserialize FitState after unpickling, reconstructing numpy arrays."""
        for k, v in state.items():
            if isinstance(v, tuple) and len(v) == 4 and v[0] == "__ndarray__":
                # Reconstruct ndarray from (marker, bytes, shape, dtype)
                _, array_bytes, shape, dtype_str = v
                arr = np.frombuffer(array_bytes, dtype=np.dtype(dtype_str))
                self.__dict__[k] = arr.reshape(shape)
            else:
                self.__dict__[k] = v
 
 
def precompute_losvd_interp(st: FitState) -> None:
    """Precompute LOSVD interpolation indices and weights (stored on st)."""
    j1 = np.searchsorted(st.xl, st.vgrid0, side="right")
    j0 = j1 - 1
    j0 = np.clip(j0, 0, len(st.xl) - 1)
    j1 = np.clip(j1, 0, len(st.xl) - 1)
    denom = st.xl[j1] - st.xl[j0]
    w = np.zeros_like(st.vgrid0, float)
    m = (denom != 0) & (j0 != j1)
    w[m] = (st.vgrid0[m] - st.xl[j0][m]) / denom[m]
    st.losvd_j0, st.losvd_j1, st.losvd_w = j0, j1, w
 
 
def precompute_ip_map(st: FitState) -> None:
    """Precompute the (npix × nlosvd) pixel-mapping table for LOSVD convolution."""
    ip = nint_fortran(st.c1 + st.x[:, None] * st.facnew0[None, :]) - 1
    st.ip_map = ip
    st.ip_mask = (ip >= 0) & (ip < st.npix)
 
 
def getnlosvd_fast_from_b(st: FitState, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(b, float)
    y2 = b[st.losvd_j0] + (b[st.losvd_j1] - b[st.losvd_j0]) * st.losvd_w
    s = float(np.sum(y2))
    return st.facnew0, y2 * (float(np.sum(b)) / s if s else 1.0)
 
 
# =============================================================================
# Section 8 - Core numerics
# =============================================================================
 
@njit(cache=True)
def _compute_chi2(
    g: np.ndarray, gp: np.ndarray, gerr: np.ndarray,
    iskip: int, npix: int,
) -> float:
    chi2 = 0.0
    for i in range(iskip, npix - iskip):
        if gerr[i] > 0.0 and gerr[i] < 1.0e9:
            if np.isfinite(g[i]) and np.isfinite(gp[i]) and np.isfinite(gerr[i]):
                r = (g[i] - gp[i]) / gerr[i]
                chi2 += r * r
    return chi2
 
 
@njit(cache=True)
def _compute_smoothness(
    b: np.ndarray, xl: np.ndarray,
    xlam: float, sigl0: float, resd: float, nl: int,
) -> float:
    """
    Penalise roughness in the LOSVD vector b.
 
    The smoothing weight is increased by (|v| / sfac*sigl0)^4 outside the
    central region, strongly suppressing far wings.
    """
    sfac = 1.8
    smooth = 0.0
    for j in range(nl):
        lam = xlam
        if abs(xl[j]) > sfac * sigl0:
            lam = xlam * (abs(xl[j] / sigl0) / sfac) ** 4
        if j == 0:
            smooth += lam * (b[1] - 2 * b[0]) ** 2
        elif j == nl - 1:
            smooth += lam * (b[nl - 2] - 2 * b[nl - 1]) ** 2
        else:
            smooth += lam * (b[j + 1] - 2 * b[j] + b[j - 1]) ** 2
    return smooth / resd


@njit(cache=True)
def _convolve_losvd_numba(
    temp: np.ndarray,
    ynew: np.ndarray,
    ip_map: np.ndarray,
    ip_mask: np.ndarray,
    npix: int,
    nlosvd: int,
) -> tuple[np.ndarray, np.ndarray]:
    gp = np.zeros(npix, dtype=np.float64)
    xs = np.zeros(npix, dtype=np.float64)

    for i in range(npix):
        ti = temp[i]
        for j in range(nlosvd):
            if ip_mask[i, j]:
                ip = ip_map[i, j]
                yj = ynew[j]
                gp[ip] += ti * yj
                xs[ip] += yj

    return gp, xs
 
def evaluate_model_gp(
    a: np.ndarray, st: FitState, apply_continuum: bool = True,
) -> tuple:
    """
    Evaluate the model spectrum for parameter vector *a*.
 
    Returns
    -------
    gp   : Model spectrum (length npix).
    b    : LOSVD vector.
    w    : Template weights.
    coff, coff2 : Continuum offset parameters.
    A    : Global amplitude (1.0 if fit_global_amp=False).
 
    Changes vs. original
    --------------------
    * Polynomial correction removed (mdeg = adeg = -1 in kept modes).
    * bincount call cleaned: ip[mask] is already 1-D; redundant .ravel()
      on the mask removed.
    * fortran_template_mixture=False path uses outside_tpl (the boolean
      mask returned by build_template_matrix_fortran) instead of the
      fragile T==1.0 sentinel.
    """
    nl, nt, npix = st.nl, st.nt, st.npix
    i = 0
    b = np.maximum(a[i:i + nl], 1e-6);  i += nl
    w = np.maximum(a[i:i + nt], 1e-12); i += nt
 
    if st.icoff == 0:
        coff, coff2 = st.coffi, st.coff2i
    elif st.icoff == 1:
        coff, coff2 = a[i], a[i + 1]; i += 2
    elif st.icoff == 2:
        coff, coff2 = st.coffi, a[i]; i += 1
    else:
        raise ValueError(f"icoff must be 0, 1, or 2; got {st.icoff}")
 
    if st.fit_global_amp:
        A = a[i]; i += 1
    else:
        A = 1.0
    if st.continuum_poly_mode != "none":
        poly = a[i]; i += 1
    else:
        poly = 0.0
 
    sum2 = float(np.sum(w)) or 1.0
    if st.fortran_template_mixture:
        tval = st.t @ w / sum2
    else:
        # Exclude pixels outside any template's coverage.
        # Uses outside_tpl (bool) instead of the old T==1.0 sentinel.
        valid_t = np.where(st.outside_tpl[:, None], 0.0, st.t)
        s2o = sum2 - (st.outside_tpl[:, None] * w[None, :]).sum(axis=1)
        s2o = np.where(s2o == 0, 1.0, s2o)
        tval = (valid_t * w[None, :]).sum(axis=1) / s2o
 
    temp = (tval + coff) / (coff + 1.0) + coff2
 
    _, ynew = getnlosvd_fast_from_b(st, b)
 
    gp, xs = _convolve_losvd_numba(
        temp,
        ynew,
        st.ip_map,
        st.ip_mask,
        npix,
        st.nlosvd,
    )
 
    suml = float(np.sum(ynew))
    gp = np.where(xs != 0, gp * suml / xs, gp)
 
    gp = gp * A

    if st.continuum_poly_mode == "additive" and st.continuum_poly_x is not None:
        gp = gp + poly * st.continuum_poly_x
    elif st.continuum_poly_mode == "multiplicative" and st.continuum_poly_x is not None:
        gp = gp * (1.0 + poly * st.continuum_poly_x)
 
    if apply_continuum and st.fit_als_continuum:
        cont = getattr(st, "continuum_mult", None)
        if cont is not None:
            gp = gp * cont
 
    return gp, b, w, coff, coff2, A
 
 
def objective_map(a: np.ndarray, st: FitState) -> float:
    st.ntot += 1
    t0 = time.perf_counter()
    gp, b, *_ = evaluate_model_gp(a, st)
    if not hasattr(st, "time_eval_model"):
        st.time_eval_model = 0.0
    st.time_eval_model += time.perf_counter() - t0
    chi2 = _compute_chi2(st.g, gp, st.gerr, st.iskip, st.npix)
    smooth = _compute_smoothness(b, st.xl, st.xlam, st.sigl0, st.resd, st.nl)
    return float(chi2 + smooth + 1e-1 * abs(float(np.sum(b)) - 1.0))


def _build_jax_objective_value_and_grad(st: FitState):
    """Return a JAX-jitted value-and-grad callable in physical parameter units.

    Both g (observed spectrum) and cont (ALS continuum) are explicit arguments,
    not closure variables.  This is critical for performance:

    * cont changes between ALS outer iterations → making it explicit lets the
      same compiled kernel serve all outer iterations without recompilation.
    * g changes between bootstrap replicates → making it explicit means the
      module-level _JAX_VG_CACHE can share the single compiled kernel across
      all 1000+ bootstrap draws.  Without this, each replicate would trigger a
      fresh XLA compilation (30-120 s each).

    jax.value_and_grad differentiates w.r.t. the first positional argument (a)
    only; g and cont are treated as non-differentiated dynamic inputs.
    """
    if jax is None or jnp is None:
        return None

    nl = int(st.nl)
    nt = int(st.nt)
    npix = int(st.npix)
    nlosvd = int(st.nlosvd)
    icoff = int(st.icoff)
    fit_global_amp = bool(st.fit_global_amp)
    fortran_template_mixture = bool(st.fortran_template_mixture)
    fit_als_continuum = bool(st.fit_als_continuum)

    t = jnp.asarray(st.t, dtype=jnp.float64)
    outside_tpl = jnp.asarray(st.outside_tpl, dtype=bool)
    losvd_j0 = jnp.asarray(st.losvd_j0, dtype=jnp.int32)
    losvd_j1 = jnp.asarray(st.losvd_j1, dtype=jnp.int32)
    losvd_w = jnp.asarray(st.losvd_w, dtype=jnp.float64)
    ip_map = jnp.asarray(st.ip_map, dtype=jnp.int32)
    ip_mask = jnp.asarray(st.ip_mask, dtype=bool)
    # gerr is NOT captured here; it is passed explicitly at call time so that
    # sigma-clip updates (st.gerr changes) and bootstrap replicates all reuse
    # the single compiled kernel without recompilation.

    coffi = float(st.coffi)
    coff2i = float(st.coff2i)
    xlam = float(st.xlam)
    sigl0 = float(st.sigl0)
    resd = float(st.resd)
    iskip = int(st.iskip)

    fit_mask = np.zeros(npix, dtype=bool)
    fit_mask[iskip:npix - iskip] = True
    fit_mask = jnp.asarray(fit_mask, dtype=bool)

    sfac = 1.8
    lam_vec = np.array([
        xlam * (max(abs(float(st.xl[j])) / sigl0, sfac) / sfac) ** 4
        if abs(float(st.xl[j])) > sfac * sigl0
        else xlam
        for j in range(nl)
    ], dtype=float)
    lam_vec = jnp.asarray(lam_vec, dtype=jnp.float64)

    # Keep 2D shape (npix, nlosvd) to avoid jnp.tile/repeat inside the jitted function.
    ip_safe = jnp.clip(jnp.asarray(ip_map, dtype=jnp.int32), 0, npix - 1)
    ip_mask_2d = jnp.asarray(ip_mask, dtype=jnp.float64)

    def _smooth_penalty(b):
        left = (b[1] - 2.0 * b[0]) ** 2
        right = (b[nl - 2] - 2.0 * b[nl - 1]) ** 2
        mid = (b[2:] - 2.0 * b[1:-1] + b[:-2]) ** 2
        terms = jnp.concatenate([jnp.array([left]), mid, jnp.array([right])])
        return jnp.sum(lam_vec * terms) / resd

    def _ynew_from_b(b):
        y2 = b[losvd_j0] + (b[losvd_j1] - b[losvd_j0]) * losvd_w
        s = jnp.sum(y2)
        sum_b = jnp.sum(b)
        scale = jnp.where(s != 0.0, sum_b / s, 1.0)
        return y2 * scale

    # g, gerr_dyn, and cont are explicit arguments so JAX traces them as dynamic
    # arrays reused across outer iterations and bootstrap draws without recompile.
    # jax.value_and_grad differentiates w.r.t. the first argument (a) only.
    def _objective(a, g, gerr_dyn, cont):
        i = 0
        b = jnp.maximum(a[i:i + nl], 1e-6)
        i += nl
        w = jnp.maximum(a[i:i + nt], 1e-12)
        i += nt

        if icoff == 0:
            coff = coffi
            coff2 = coff2i
        elif icoff == 1:
            coff = a[i]
            coff2 = a[i + 1]
            i += 2
        else:
            coff = coffi
            coff2 = a[i]
            i += 1

        A = a[i] if fit_global_amp else 1.0

        sum2 = jnp.sum(w)
        sum2 = jnp.where(sum2 != 0.0, sum2, 1.0)
        if fortran_template_mixture:
            tval = (t @ w) / sum2
        else:
            valid_t = jnp.where(outside_tpl[:, None], 0.0, t)
            s2o = sum2 - jnp.sum(outside_tpl[:, None] * w[None, :], axis=1)
            s2o = jnp.where(s2o == 0.0, 1.0, s2o)
            tval = jnp.sum(valid_t * w[None, :], axis=1) / s2o

        temp = (tval + coff) / (coff + 1.0) + coff2
        ynew = _ynew_from_b(b)

        # Gather-style scatter: (npix, nlosvd) outer product, then scatter to output.
        # Broadcasting avoids the large jnp.tile/repeat intermediates.
        contrib = temp[:, None] * ynew[None, :] * ip_mask_2d   # (npix, nlosvd)
        xs_contrib = ynew[None, :] * ip_mask_2d                 # (npix, nlosvd)
        gp = jnp.zeros(npix, dtype=jnp.float64).at[ip_safe].add(contrib)
        xs = jnp.zeros(npix, dtype=jnp.float64).at[ip_safe].add(xs_contrib)

        suml = jnp.sum(ynew)
        gp = jnp.where(xs != 0.0, gp * suml / xs, gp)
        gp = gp * A

        if fit_als_continuum:
            gp = gp * cont

        valid = (
            fit_mask
            & jnp.isfinite(g)
            & jnp.isfinite(gp)
            & jnp.isfinite(gerr_dyn)
            & (gerr_dyn > 0.0)
            & (gerr_dyn < 1.0e9)
        )

        resid = (g - gp) / gerr_dyn
        chi2 = jnp.sum(jnp.where(valid, resid * resid, 0.0))
        smooth = _smooth_penalty(b)
        norm_pen = 1e-1 * jnp.abs(jnp.sum(b) - 1.0)
        return chi2 + smooth + norm_pen

    obj_vg = jax.jit(jax.value_and_grad(_objective))

    def _value_and_grad_np(
        a_np: np.ndarray,
        g_np: np.ndarray,
        gerr_np: np.ndarray,
        cont_np: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        val, grad = obj_vg(
            jnp.asarray(a_np, dtype=jnp.float64),
            jnp.asarray(g_np, dtype=jnp.float64),
            jnp.asarray(gerr_np, dtype=jnp.float64),
            jnp.asarray(cont_np, dtype=jnp.float64),
        )
        return float(val), np.asarray(grad, dtype=float)

    return _value_and_grad_np
 
def objective_components(a: np.ndarray, st: FitState) -> dict:
    """Return a breakdown of all objective terms for diagnostics."""
    gp, b, *_ = evaluate_model_gp(a, st)
    chi2   = _compute_chi2(st.g, gp, st.gerr, st.iskip, st.npix)
    smooth = _compute_smoothness(b, st.xl, st.xlam, st.sigl0, st.resd, st.nl)
    fadd   = 1e-1 * abs(float(np.sum(b)) - 1.0)
    return {
        "chi2": float(chi2), "smooth": float(smooth),
        "losvd_norm_penalty": float(fadd),
        "total": float(chi2 + smooth + fadd),
        "smooth_over_chi2": float(smooth / chi2) if chi2 else np.nan,
    }
 
 
def compute_weighted_template_spectrum(st: FitState, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    sum2 = float(np.sum(w))
    if sum2 <= 0 or not np.isfinite(sum2):
        return np.ones(st.npix)
    if st.fortran_template_mixture:
        return st.t @ w / sum2
    valid_t = np.where(st.outside_tpl[:, None], 0.0, st.t)
    s2o = np.where(
        (sum2 - (st.outside_tpl[:, None] * w[None, :]).sum(axis=1)) == 0,
        1.0,
        sum2 - (st.outside_tpl[:, None] * w[None, :]).sum(axis=1),
    )
    return (valid_t * w[None, :]).sum(axis=1) / s2o
 
 
# =============================================================================
# Section 9 - Sigma-clipping / cleaning
# =============================================================================
 
def _clean_center_to_fit_frame(center: float, frame: str, cfg: FitConfig) -> float:
    """
    Convert cleaning-protection line center into st.x frame.

    st.x is rest-frame for raw .spec files, so observed-frame centers are
    divided by 1 + zgal.  als_mask_center_shift_A (set by als_auto_caii_shift)
    is also applied so the protection window tracks the actual line position.
    """
    frame = frame.lower().strip()
    c = float(center)

    if frame == "observed":
        c = c / (1.0 + cfg.zgal)
    elif frame != "rest":
        raise ValueError(f"clean frame must be 'rest' or 'observed', got {frame!r}")

    c += float(getattr(cfg, "als_mask_center_shift_A", 0.0))
    return c


def _clean_window_to_fit_frame(
    lo: float,
    hi: float,
    frame: str,
    cfg: FitConfig,
) -> tuple[float, float]:
    """
    Convert cleaning-protection window into st.x frame.
    """
    frame = frame.lower().strip()
    lo = float(lo)
    hi = float(hi)

    if frame == "observed":
        lo = lo / (1.0 + cfg.zgal)
        hi = hi / (1.0 + cfg.zgal)
    elif frame != "rest":
        raise ValueError(f"clean frame must be 'rest' or 'observed', got {frame!r}")

    if hi < lo:
        lo, hi = hi, lo

    return lo, hi


def get_ca_centers_for_cleaning(st: FitState, cfg: FitConfig) -> np.ndarray:
    centers = np.asarray(cfg.clean_ca_centers, float)
    return np.array(
        [
            _clean_center_to_fit_frame(c, cfg.clean_protect_ca_frame, cfg)
            for c in centers
        ],
        dtype=float,
    )

def emission_half_width_A(cen_A, cfg):
    hw_A = float(getattr(cfg, "mask_emission_line_half_width_A", 5.0))
    vpad = float(getattr(cfg, "mask_emission_line_velocity_pad_kms", 300.0))
    return max(hw_A, cen_A * vpad / CEE)

def build_emission_line_mask(
    x: np.ndarray,
    g: np.ndarray,
    gerr: np.ndarray,
    cfg: "FitConfig",
) -> np.ndarray:
    """Boolean mask of emission line pixels to exclude from the spectral fit.

    Called in make_fit_state to set gerr=BIG before any fitting.  Emission
    lines are absent from stellar templates; pre-masking ensures they cannot
    bias the LOSVD even when too weak to be caught by sigma-clipping.

    Detection uses the *excess* above the local continuum, not the continuum S/N.
    For each candidate emission line:
      1. Estimate the local continuum from the flanking regions just outside
         the line window (±snr_context_A excluding ±half_width_A).
      2. Compute excess_snr = (max_flux_in_line - cont_est) / noise_est.
      3. Mask only if excess_snr >= snr_threshold.

    This prevents false positives in high-S/N regions (e.g. near Ca II) where
    the region S/N is high but the line itself is not in emission.
    """
    mask = np.zeros(len(x), dtype=bool)
    if not getattr(cfg, "mask_emission_lines_in_fit", True):
        return mask
    hw = float(getattr(cfg, "mask_emission_line_half_width_A", 5.0))
    snr_thr = float(getattr(cfg, "mask_emission_line_snr_threshold", 3.0))
    snr_ctx = float(getattr(cfg, "mask_emission_line_snr_context_A", 20.0))
    wmin = float(getattr(cfg, "wavefitmin", float(x.min())))
    wmax = float(getattr(cfg, "wavefitmax", float(x.max())))
    good = np.isfinite(g) & np.isfinite(gerr) & (gerr > 0) & (gerr < BIG)

    mask_paschen = bool(getattr(cfg, "mask_paschen_lines_in_fit", False))

    for table in _STELLAR_ABSORPTION_LINE_TABLES:
        for tag, name, cen_rest in table:
            is_emission_candidate = tag == "em"
            is_paschen_candidate = mask_paschen and tag == "paschen"

            if not (is_emission_candidate or is_paschen_candidate):
                continue

            if cen_rest < wmin - hw or cen_rest > wmax + hw:
                continue

            line_win = (x >= cen_rest - hw) & (x <= cen_rest + hw)

            if snr_thr <= 0:
                mask |= line_win
                log(
                    f"  emission pre-mask {name} {cen_rest:.2f} Å  "
                    f"unconditional  npix={line_win.sum()}"
                )
                continue

            ctx = (x >= cen_rest - snr_ctx) & (x <= cen_rest + snr_ctx) & good
            flanks = ctx & ~line_win
            if flanks.sum() < 5:
                continue

            cont_est = float(np.nanmedian(g[flanks]))
            noise_est = float(np.nanmedian(gerr[flanks]))
            if not (np.isfinite(cont_est) and np.isfinite(noise_est) and noise_est > 0):
                continue

            line_good = line_win & good
            if not line_good.any():
                continue

            peak_flux = float(np.nanmax(g[line_good]))
            excess_snr = (peak_flux - cont_est) / noise_est

            if not (np.isfinite(excess_snr) and excess_snr >= snr_thr):
                continue

            mask |= line_win
            log(
                f"  emission pre-mask {name} {cen_rest:.2f} Å  "
                f"excess_S/N={excess_snr:.1f}  cont_est={cont_est:.4g}  npix={line_win.sum()}"
            )
    return mask


def build_clean_protect_mask(st: FitState, cfg: FitConfig) -> np.ndarray:
    protect = np.zeros(st.npix, dtype=bool)

    if cfg.clean_protect_ca_triplet:
        for cen in get_ca_centers_for_cleaning(st, cfg):
            m = (
                (st.x >= cen - cfg.clean_ca_half_width)
                & (st.x <= cen + cfg.clean_ca_half_width)
            )
            protect |= m
            log(
                f"  protecting Ca II [{cen - cfg.clean_ca_half_width:.1f}, "
                f"{cen + cfg.clean_ca_half_width:.1f}] npix={m.sum()}"
            )

    frame = getattr(cfg, "clean_protect_windows_frame", "rest")

    for lo_raw, hi_raw in cfg.clean_protect_windows:
        lo, hi = _clean_window_to_fit_frame(lo_raw, hi_raw, frame, cfg)
        m = (st.x >= lo) & (st.x <= hi)
        protect |= m
        log(f"  protecting custom window [{lo:.1f}, {hi:.1f}] npix={m.sum()}")

    for item in getattr(cfg, "extra_absorption_lines", ()):
        c_raw = float(item[0])
        hw = float(item[1])
        c_fit = c_raw + float(getattr(cfg, "als_mask_center_shift_A", 0.0))
        m = (st.x >= c_fit - hw) & (st.x <= c_fit + hw)
        protect |= m
        name = item[2] if len(item) >= 3 else f"{c_raw:.2f} Å"
        log(f"  protecting extra line {name} [{c_fit - hw:.1f}, {c_fit + hw:.1f}] npix={m.sum()}")

    return protect


def _bloom_rejected(rejected: np.ndarray, n: int) -> np.ndarray:
    """Expand each rejected pixel by n neighbors on each side."""
    if n <= 0 or not rejected.any():
        return rejected.copy()
    result = rejected.copy()
    for i in range(1, n + 1):
        result[i:] |= rejected[:-i]
        result[:-i] |= rejected[i:]
    return result


def _update_clean_mask(
    st: FitState, a_best: np.ndarray, base_gerr: np.ndarray,
    good_mask: np.ndarray, sigma_clip: float,
    protect_mask: Optional[np.ndarray] = None,
    protect_absorption_only: bool = True,
) -> tuple[np.ndarray, float]:
    gp, *_ = evaluate_model_gp(a_best, st)
    resid = (st.g - gp) / np.where(base_gerr > 0, base_gerr, 1.0)
    sigma = robust_sigma(resid[good_mask])
    if not np.isfinite(sigma) or sigma <= 0:
        return good_mask.copy(), sigma
    # One-sided: only reject emission outliers (resid > N*sigma, data above model).
    # Absorption residuals (resid < 0) are never clipped — templates handle all
    # absorption features.  The protect_mask still prevents emission within
    # explicitly protected windows (e.g. Ca II) from being clipped when
    # protect_absorption_only=False.
    keep = resid <= sigma_clip * sigma
    if protect_mask is not None:
        pm = np.asarray(protect_mask, bool)
        keep = keep | (pm & (resid < 0) if protect_absorption_only else pm)
    return good_mask & keep, sigma
 
 
def run_iterative_clean_map(
    st: FitState, a0: np.ndarray, bounds: list,
    map_maxiter: int, map_ftol: float, map_maxfun: int, print_every: int,
    sigma_clip: float = 3.0, clean_maxiter: int = 5, clean_minpix: int = 10,
    protect_mask: Optional[np.ndarray] = None, protect_absorption_only: bool = True,
    bloom_pixels: int = 0,
    map_gtol: float = 1e-10, map_maxls: int = 50, use_scaled_optimizer: bool = True,
    use_jax_objective: bool = False, jax_enable_x64: bool = True,
):
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
 
 
def build_parameter_xscale(st: FitState) -> np.ndarray:
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
 
 
def _chi2_stats(st: FitState, a_best: np.ndarray) -> tuple[float, int]:
    gp, *_ = evaluate_model_gp(a_best, st)
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
    """
    Peak-normalised maximum absolute second difference of a LOSVD array.

    Measures local curvature: smooth LOSVDs (e.g., a Gaussian with σ ≈ 3
    LOSVD bins) have roughness ≈ 0.10; jagged noise spikes produce values
    > 0.20.  Used by _auto_select_xlam to determine whether the current
    regularisation strength is sufficient.
    """
    b_pos = np.maximum(np.asarray(b, float), 0.0)
    peak = float(np.max(b_pos))
    if len(b) < 3:
        return 0.0
    if peak < 1e-4:
        return np.inf  # degenerate LOSVD; treat as maximally rough so xlam increases
    return float(np.max(np.abs(np.diff(np.diff(b_pos))))) / peak


def compute_losvd_n_peaks(b: np.ndarray, min_prominence: float = 0.1) -> int:
    """
    Count peaks in a LOSVD whose prominence exceeds min_prominence * global_peak.

    Prominence of a peak = its height minus the deepest valley connecting it to
    any taller neighbouring peak (the "key col"), normalised by the global peak.

    This correctly ignores wings and shoulders — features with a shallow valley
    to the main peak have small prominence and are not counted.  Only genuine
    secondary peaks separated by a deep valley (e.g., bimodal LOSVDs from noise
    overfitting in inner bins) exceed the prominence threshold.
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
    """
    Find and set the minimum xlam in xlam_grid that yields a smooth LOSVD.

    Iterates the sorted grid from smallest to largest.  For each candidate,
    runs one MAP fit (no sigma-clipping, no ALS update) and measures the LOSVD
    roughness via compute_losvd_roughness.  Stops at the first xlam whose
    roughness is at or below smooth_threshold.

    Sets st.xlam = cfg.xlam = best_xlam before returning so that all
    subsequent MAP fits (ALS outer loop, final refit) and bootstrap refits
    use the selected regularisation strength.

    The search uses the ALS continuum already initialised by init_als_continuum
    and starts every trial from the same a0 so results are comparable.
    Bootstrap workers always have xlam_auto=False (set by _make_frozen_cfg)
    so this function runs only once per spectrum.
    """
    grid = sorted(float(v) for v in xlam_grid)
    original_xlam = float(st.xlam)
    auto_maxiter = int(cfg.xlam_auto_maxiter or cfg.map_maxiter)
    max_peaks = int(getattr(cfg, "xlam_max_peaks", 1))
    min_prominence = float(getattr(cfg, "xlam_peak_min_prominence", 0.1))

    log(
        f"Auto-xlam search: grid={[f'{x:.0f}' for x in grid]}  "
        f"threshold={smooth_threshold:.3f}  max_peaks={max_peaks}  "
        f"maxiter={auto_maxiter}"
    )

    best_xlam = float(grid[-1])
    found_smooth = False

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
        is_smooth = roughness <= smooth_threshold
        is_unimodal = n_peaks <= max_peaks
        status = ("smooth" if is_smooth else "jagged") + ("" if is_unimodal else f", {n_peaks} peaks")
        log(
            f"  xlam={xlam:8.0f}  roughness={roughness:.4f}  peaks={n_peaks}  {status}"
        )
        if is_smooth and is_unimodal:
            best_xlam = xlam
            found_smooth = True
            break

    if not found_smooth:
        log(
            f"Auto-xlam: no grid value satisfies roughness <= {smooth_threshold:.3f} "
            f"and peaks <= {max_peaks}; using xlam={grid[-1]:.0f} (most regularised)"
        )
        best_xlam = float(grid[-1])

    st.xlam = best_xlam
    cfg.xlam = best_xlam
    log(
        f"Auto-xlam: selected xlam={best_xlam:.0f} "
        f"(original cfg.xlam was {original_xlam:.0f})"
    )
    return best_xlam


def fit_state_map_with_optional_clean(
    st: FitState, cfg: FitConfig, a0: np.ndarray, bounds: list,
):
    """
    Run MAP optimisation, optionally with sigma-clipping and/or ALS outer loop.

    In ALS-continuum mode, full iterative cleaning is only done on the first
    outer iteration. Later ALS iterations reuse the resulting mask and perform
    ordinary MAP fits. This avoids clean_maxiter full refits per ALS iteration.
    """
    kwargs = dict(
        map_gtol=cfg.map_gtol,
        map_maxls=cfg.map_maxls,
        use_scaled_optimizer=cfg.use_scaled_optimizer,
        use_jax_objective=cfg.use_jax_objective,
        jax_enable_x64=cfg.jax_enable_x64,
    )

    original_clean = bool(cfg.clean)

    # Per-spectrum automatic smoothing: run before the first MAP fit so that
    # st.xlam is correct for the entire ALS outer loop that follows.
    # Bootstrap refits skip this (xlam_auto=False in cfg_frozen; see
    # losvd_errors._make_frozen_cfg) so they use the xlam selected here.
    if getattr(cfg, "xlam_auto", False):
        _auto_select_xlam(
            st, cfg, np.asarray(a0, float), bounds,
            xlam_grid=getattr(cfg, "xlam_auto_grid", (20., 50., 100., 250., 600., 1500., 4000.)),
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
    # Non-ALS mode: keep previous behavior.
    if not cfg.fit_als_continuum:
        return one_fit(a0, do_clean=original_clean)
    # ALS outer loop: alternate LOSVD fit and continuum update
    a_start = np.asarray(a0, float).copy()
    best = None
    last_delta = np.inf

    for k in range(cfg.als_outer_iter):
        log(f"ALS outer iteration {k + 1}/{cfg.als_outer_iter}")

        # If you implemented "clean only first outer", keep that logic here.
        do_clean_this_iter = cfg.clean and (k == 0)

        best = one_fit(
            a_start,
            do_clean=do_clean_this_iter,
            label=f"MAP optimize ALS outer {k + 1}",
        )

        last_delta = update_als_continuum(st, cfg, best.x)
        log(f"  ALS continuum median fractional change = {last_delta:.4g}")

        a_start = np.asarray(best.x, float).copy()

        if last_delta < cfg.als_outer_tol:
            log(
                f"  ALS outer loop converged "
                f"(delta={last_delta:.2e} < tol={cfg.als_outer_tol:.2e})"
            )
            break

    # After the final ALS continuum update, do one final fit with that continuum
    # fixed. Otherwise the returned parameters correspond to the previous
    # continuum, while the output model uses the newly updated continuum.
    log("Final MAP refit with final ALS continuum fixed")
    best = one_fit(
        a_start,
        do_clean=False,
        label="MAP final after ALS continuum update",
    )

    return best
 
 
# =============================================================================
# Section 10 - Spectrum loading
# =============================================================================
 
def choose_norm_wavelength_frame(
    wave_raw: np.ndarray, cfg: FitConfig,
) -> tuple[np.ndarray, str]:
    """Return wavelength array in the correct frame and a description string."""
    frame = cfg.norm_wave_frame.lower().strip()
    xa = np.asarray(wave_raw, float)
    xd = xa / (1 + cfg.zgal)
    na = count_in_window(xa, cfg.wavefitmin, cfg.wavefitmax)
    nd = count_in_window(xd, cfg.wavefitmin, cfg.wavefitmax)
    log(
        f".norm raw range=[{xa.min():.2f},{xa.max():.2f}] "
        f"in-window: raw={na}, raw/(1+z)={nd}"
    )
    if frame == "rest":
        return xa, "rest/as-given"
    if frame == "observed":
        return xd, "observed divided by 1+z"
    if frame == "auto":
        return (xd, "auto: /1+z") if nd > na else (xa, "auto: as-given")
    raise ValueError(f"norm_wave_frame must be 'auto'/'rest'/'observed', got {frame!r}")
 
 
def fortran_rebin_after_redshift(
    x: np.ndarray, flux: np.ndarray, step: float,
    extra_arrays: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x, flux = np.asarray(x, float), np.asarray(flux, float)
    x_new = x[0] + float(step) * np.arange(len(x), dtype=float)
    flux_new = np.interp(x_new, x, flux, left=flux[0], right=flux[-1])
    extra_new: dict = {}
    if extra_arrays:
        for k, v in extra_arrays.items():
            extra_new[k] = (
                None if v is None
                else np.interp(x_new, x, np.asarray(v, float), left=v[0], right=v[-1])
            )
    return x_new, flux_new, extra_new


def load_spectrum_for_fit(cfg: FitConfig, gal_file: Optional[str] = None):
    """
    Load the galaxy spectrum and return arrays ready for fitting.
    """
    if gal_file is None:
        raise ValueError("gal_file is required to load the spectrum for fitting.")

    prenorm = not cfg.fit_als_continuum
    is_norm = gal_file.lower().endswith(".norm")
    extra: dict = {}
    if is_norm:
        nd = read_norm_spectrum(gal_file)
        x, frame = choose_norm_wavelength_frame(nd["wavelength"], cfg)
        log(f".norm frame: {frame}")
        if prenorm:
            flux = nd["normflux"].copy()
            ferr = (
                np.ones_like(flux)
                if cfg.norm_error_mode == "unit"
                else nd["normflux_err"].copy()
            )
            log(f"PRE-NORMALIZED MODE; error_mode={cfg.norm_error_mode}")
        else:
            if nd["orig_flux"] is None:
                raise ValueError(
                    "fit_als_continuum=True with a .norm file requires orig_flux "
                    "(columns 4+ of the .norm file)."
                )
            flux = nd["orig_flux"].copy()
            ferr = (
                nd["orig_flux_err"].copy()
                if nd["orig_flux_err"] is not None
                else np.ones_like(flux)
            )
            if not cfg.use_spectrum_errors:
                med = float(np.nanmedian(ferr[ferr > 0])) if np.any(ferr > 0) else 1.0
                ferr = np.full_like(ferr, med)
                log("use_spectrum_errors=False: replaced per-pixel errors with uniform median")
            log("ALS CONTINUUM MODE from .norm; using orig_flux / orig_flux_err")
        if cfg.fortran_rebin_norm:
            if cfg.step is None:
                raise ValueError("cfg.step is required when fortran_rebin_norm=True")
            arrays = dict(
                fit_err=ferr,
                normflux=nd.get("normflux"),
                normflux_err=nd.get("normflux_err"),
                orig_flux=nd.get("orig_flux"),
                orig_flux_err=nd.get("orig_flux_err"),
                continuum=nd.get("continuum"),
                continuum_err=nd.get("continuum_err"),
            )
            x, flux, reb = fortran_rebin_after_redshift(
                x,
                flux,
                cfg.step,
                arrays,
            )
            ferr = reb["fit_err"]
            extra.update(reb)
        extra.update(
            wavelength=x.copy(),
            fit_flux=flux.copy(),
            fit_err=ferr.copy(),
        )
        extra["_x_full_for_nlosvd"] = x.copy()
        x_reg, g_reg, ge_reg, keep = select_region_with_errors(
            x,
            flux,
            ferr,
            cfg.wavefitmin,
            cfg.wavefitmax,
        )
        for k, v in list(extra.items()):
            if not k.startswith("_"):
                extra[k + "_fit"] = None if v is None else np.asarray(v)[keep].copy()
        if prenorm and cfg.norm_apply_fortran_flux_mask:
            bad = (
                ~np.isfinite(g_reg)
                | (g_reg <= cfg.norm_fortran_flux_min)
                | (g_reg > cfg.norm_fortran_flux_max)
            )
            if bad.any():
                log(f"Flux mask: {bad.sum()} pixels flagged")
            ge_reg = np.where(bad, BIG, ge_reg)
        if len(x_reg) < 2:
            raise ValueError(
                f"Too few pixels in [{cfg.wavefitmin}, {cfg.wavefitmax}]: {len(x_reg)}"
            )
        step_used = float(cfg.step) if cfg.step else estimate_step_from_wavelength(x_reg)
        log(f"fit pixels={len(x_reg)} step={step_used:g}")
        return x_reg, g_reg, ge_reg, step_used, prenorm, extra
    # Raw file
    col0, flux, ferr_raw = read_galaxy_index_flux_err(gal_file)
    ferr = (
        np.sqrt(np.clip(ferr_raw, 0.0, np.inf))
        if cfg.spec_col3_is_variance
        else ferr_raw
    )
    if not cfg.use_spectrum_errors:
        med = float(np.nanmedian(ferr[ferr > 0])) if np.any(ferr > 0) else 1.0
        ferr = np.full_like(ferr, med)
        log("use_spectrum_errors=False: replaced per-pixel errors with uniform median")
    is_integer_index = np.all(col0 == np.floor(col0)) and col0[0] < 100
    if is_integer_index:
        if cfg.wavemin_full is None or cfg.step is None:
            raise ValueError(
                "Raw integer-index .spec files require wavemin_full and step in the config."
            )
        x = build_wavelength_from_index(
            len(flux),
            cfg.wavemin_full,
            cfg.step,
        ) / (1.0 + cfg.zgal)
        step_used = float(cfg.step)
    else:
        x = col0.astype(float) / (1.0 + cfg.zgal)
        step_used = estimate_step_from_wavelength(x)
        log(f"Auto-detected wavelength from col 0; step={step_used:.4f}")
    extra["_x_full_for_nlosvd"] = x.copy()
    x_reg, g_reg, ge_reg, _ = select_region_with_errors(
        x,
        flux,
        ferr,
        cfg.wavefitmin,
        cfg.wavefitmax,
    )
    if len(x_reg) < 2:
        raise ValueError(
            f"Too few pixels in [{cfg.wavefitmin}, {cfg.wavefitmax}]: {len(x_reg)}"
        )
    log(f"fit pixels={len(x_reg)} step={step_used:g}")

    return x_reg, g_reg, ge_reg, step_used, False, extra
 
 
# =============================================================================
# Section 11 - FitState construction
# =============================================================================


def _segment_emission_mask(
    x: np.ndarray,
    g: np.ndarray,
    gerr: np.ndarray,
    cfg,
) -> np.ndarray:
    """
    Segment-wise rolling-median upward-outlier detection.

    Divides the wavelength range into CaT-free continuum segments, then for
    each segment flags pixels significantly ABOVE the local rolling median.
    Only upward deviations are flagged so absorption features are untouched.

    This catches:
    - Emission line wings that fall just below the SNR threshold used by
      build_emission_line_mask (the most common cause of ALS elevation in
      spectra with grouped emission near the blue/red edge).
    - Emission features entirely absent from the known line table.

    Controlled by cfg fields:
      segment_emission_mask     (bool,  default True)  – enable/disable
      segment_emission_n_sigma  (float, default 3.0)   – rejection threshold
      segment_emission_win_A    (float, default 50.0)  – rolling half-window (Å)

    Returns boolean mask; True = suspected emission / upward outlier.
    """
    if not bool(getattr(cfg, "segment_emission_mask", True)):
        return np.zeros(len(x), dtype=bool)
    if not getattr(cfg, "mask_emission_lines_in_fit", True):
        return np.zeros(len(x), dtype=bool)

    n_sigma = float(getattr(cfg, "segment_emission_n_sigma", 3.0))
    win_A = float(getattr(cfg, "segment_emission_win_A", 50.0))

    ca_centers = np.asarray(
        getattr(cfg, "als_ca_centers", (8498.02, 8542.09, 8662.14)), float
    )
    ca_hw_arr = np.asarray(
        getattr(cfg, "als_ca_half_widths", (14.0, 16.0, 18.0)), float
    )
    # Use 1.5× the broadest Ca mask half-width as the margin around each line
    # so that the strong Ca II absorption troughs do not inflate the rolling
    # median or the MAD in their respective segments.
    margin = float(np.max(ca_hw_arr)) * 1.5

    xarr = np.asarray(x, float)
    garr = np.asarray(g, float)
    earr = np.asarray(gerr, float)
    good = (earr < BIG) & np.isfinite(garr) & np.isfinite(earr) & (earr > 0)

    xmin, xmax = float(xarr.min()), float(xarr.max())
    ca_sorted = np.sort(ca_centers)

    # Build CaT-free segment boundaries: alternating (lo, hi) pairs.
    boundaries: list[float] = [xmin]
    for c in ca_sorted:
        boundaries.append(c - margin)
        boundaries.append(c + margin)
    boundaries.append(xmax)

    mask = np.zeros(len(xarr), dtype=bool)

    for i in range(0, len(boundaries) - 1, 2):
        lo, hi = boundaries[i], boundaries[i + 1]
        if lo >= hi:
            continue
        seg = good & (xarr >= lo) & (xarr <= hi)
        n_seg = int(np.count_nonzero(seg))
        if n_seg < 10:
            continue

        seg_idx = np.where(seg)[0]
        order = np.argsort(xarr[seg_idx])
        ordered_idx = seg_idx[order]       # global pixel indices, λ-sorted
        g_ordered = garr[ordered_idx]
        x_ordered = xarr[ordered_idx]
        n = len(g_ordered)

        # Rolling half-window in pixels
        dx = float(np.median(np.diff(x_ordered))) if n > 1 else 1.25
        half_win = max(5, int(np.ceil(win_A / (2.0 * max(dx, 1e-6)))))

        rolling_med = np.empty(n, dtype=float)
        rolling_mad = np.empty(n, dtype=float)
        for k in range(n):
            lo_k = max(0, k - half_win)
            hi_k = min(n, k + half_win + 1)
            w = g_ordered[lo_k:hi_k]
            m = float(np.median(w))
            rolling_med[k] = m
            rolling_mad[k] = 1.4826 * float(np.median(np.abs(w - m)))

        emit = (rolling_mad > 0) & (
            g_ordered > rolling_med + n_sigma * rolling_mad
        )
        if emit.any():
            n_new = int(emit.sum())
            log(
                f"Segment emission mask [{lo:.1f}–{hi:.1f} Å]: "
                f"{n_new} upward-outlier pixels (>{n_sigma:.1g}σ above rolling median)"
            )
            mask[ordered_idx[emit]] = True

    return mask


def make_fit_state(cfg: FitConfig, gal_file: Optional[str] = None):
    with Timer("read spectrum"):
        x_reg, g_reg, ge_reg, step_used, prenormalized, norm_extra = load_spectrum_for_fit(
            cfg,
            gal_file=gal_file,
        )
    if len(x_reg) < 10:
        raise ValueError(f"Too few pixels in fit region: {len(x_reg)}")

    em_mask = np.zeros(len(x_reg), dtype=bool)   # populated below when ALS mode is active

    with Timer("apply masks"):
        bad = (
            ~np.isfinite(x_reg)
            | ~np.isfinite(g_reg)
            | ~np.isfinite(ge_reg)
            | (ge_reg <= 0.0)
        )

        gerr = np.where(bad, BIG, ge_reg).astype(float)

        # Replace bad flux values so they cannot poison chi2/model diagnostics.
        # Their gerr=BIG means they are effectively masked.
        g_reg = np.where(np.isfinite(g_reg), g_reg, 0.0)

        xorig = x_reg * (1 + cfg.zgal)
        gerr = setbadreg(xorig, gerr, -1, cfg.regions_bad_path)
        gerr = setbadreg(x_reg,  gerr, +1, cfg.regions_bad_path)

        # Pre-mask emission lines: stellar templates contain no emission.
        em_mask = build_emission_line_mask(x_reg, g_reg, gerr, cfg)
        if em_mask.any():
            log(f"Emission line pre-mask: {em_mask.sum()} pixels set to gerr=BIG")
            gerr = np.where(em_mask, BIG, gerr)

        # Segment-wise rolling-median detection for emission wings and features
        # not in the known line table.  Unmasked emission wings are the most
        # common reason the ALS continuum rises above the full spectrum: they
        # stay in the ALS base as upward outliers, driving als_abs_clean to
        # reject the true continuum as "absorption", in a self-reinforcing loop.
        seg_em = _segment_emission_mask(x_reg, g_reg, gerr, cfg)
        if seg_em.any():
            new_pix = int(np.count_nonzero(seg_em & ~em_mask))
            if new_pix > 0:
                log(f"Segment emission pre-mask: {new_pix} additional pixels")
            em_mask = em_mask | seg_em
            gerr = np.where(seg_em, BIG, gerr)

        # Optionally grow the combined emission mask to cover line wings and
        # merge closely-spaced emission groups into a single contiguous region.
        # This prevents isolated unmasked continuum pixels between adjacent
        # emission lines from being clipped by the sigma-clip clean step.
        grow_A = float(getattr(cfg, "mask_emission_grow_A", 0.0))
        if grow_A > 0.0 and em_mask.any():
            em_mask_grown = grow_boolean_mask_A(em_mask, x_reg, grow_A)
            new_grow = int(np.count_nonzero(em_mask_grown & ~em_mask))
            if new_grow > 0:
                log(
                    f"Emission mask grow (+{grow_A:.1f} Å): "
                    f"{new_grow} additional pixels merged/covered"
                )
                em_mask = em_mask_grown
                gerr = np.where(em_mask, BIG, gerr)

    with Timer("read + interpolate templates"):
        tpl_files = read_template_list(cfg.template_list_file, cfg.template_dir)
        T, T_err, outside_any = build_template_matrix_fortran(x_reg, tpl_files)
        if cfg.fortran_mask_template_outside and outside_any.any():
            log(f"Template coverage mask: {outside_any.sum()} pixels masked")
            gerr = np.where(outside_any, BIG, gerr)

        # Pooled fractional template error: median(err/flux) across all templates
        # at pixels where both T and T_err are positive.
        valid_frac = (T > 0) & (T_err > 0)
        f_template = float(np.nanmedian(T_err[valid_frac] / T[valid_frac])) if valid_frac.any() else 0.0
        log(f"Template fractional error (pooled median): {f_template:.4f}")
 
    nl = int(getattr(cfg, "n_losvd_bins", 29))

    if cfg.losvd_vmin is not None and cfg.losvd_vmax is not None:
        xl = np.linspace(
            float(cfg.losvd_vmin),
            float(cfg.losvd_vmax),
            nl,
        )
        log(
            f"LOSVD velocity grid from galaxy.params/config: "
            f"[{xl[0]:.3f}, {xl[-1]:.3f}] km/s, nl={nl}"
        )
    else:
        xl = np.linspace(-4.5 * cfg.sigl, 4.5 * cfg.sigl, nl)
        log(
            f"LOSVD velocity grid from sigl: "
            f"[{xl[0]:.3f}, {xl[-1]:.3f}] km/s, nl={nl}"
        )

    c1 = 1.0 - x_reg[0] / step_used
 
    if cfg.fortran_nlosvd_full_x and "_x_full_for_nlosvd" in norm_extra:
        x_full = np.asarray(norm_extra["_x_full_for_nlosvd"], float)
        x_ref = float(x_full[max(len(x_full) // 2 - 1, 0)])
    else:
        x_ref = float(x_reg[len(x_reg) // 2])
    log(f"nlosvd reference wavelength: {x_ref:.4f}")
 
    vspan = float(xl[-1] - xl[0])
    nlosvd = max(
        3,
        int(np.rint(vspan / step_used * x_ref / CEE) * 2),)
    vgrid0 = xl[0] + (xl[-1] - xl[0]) * np.arange(nlosvd) / float(nlosvd - 1)
    facnew0 = (1 + vgrid0 / CEE) / step_used

    continuum_poly_x = None
    if cfg.continuum_poly_mode != "none":
        x_center = float(np.mean(x_reg))
        x_scale = float(np.ptp(x_reg))
        if not np.isfinite(x_scale) or x_scale <= 0.0:
            x_scale = 1.0
        continuum_poly_x = (x_reg - x_center) / x_scale
 
    st = FitState(
        x=x_reg, g=g_reg, gerr=gerr,
        t=T, t_err=T_err, f_template=f_template,
        outside_tpl=outside_any,
        c1=c1, resd=step_used ** 3,
        xlam=cfg.xlam, xl=xl,
        sigl0=cfg.sigl,
        npix=len(x_reg), nl=nl, iskip=0,
        nt=T.shape[1],
        icoff=cfg.icoff,
        nlosvd=nlosvd, scale=step_used,
        coffi=cfg.coff, coff2i=cfg.coff2,
        prenormalized=prenormalized,
        norm_extra=norm_extra,
        fortran_template_mixture=cfg.fortran_template_mixture,
        fit_global_amp=cfg.fit_global_amp,
        vgrid0=vgrid0, facnew0=facnew0,
        fit_als_continuum=cfg.fit_als_continuum,
        continuum_poly_mode=cfg.continuum_poly_mode,
        continuum_poly_x=continuum_poly_x,
        continuum_poly_bound=cfg.continuum_poly_bound,
    )
 
    st.emission_pre_mask = em_mask

    with Timer("precompute LOSVD + ip map"):
        precompute_losvd_interp(st)
        precompute_ip_map(st)
        if cfg.fit_als_continuum:
            init_als_continuum(st, cfg, templates=getattr(st, 't', None))
            log("ALS continuum initialised.")
        else:
            st.continuum_mult = np.ones(st.npix)
 
    log(
        f"STATE: npix={st.npix} nt={st.nt} nl={st.nl} nlosvd={st.nlosvd} "
        f"prenormalized={st.prenormalized} fit_als_continuum={st.fit_als_continuum}"
    )
    return st, tpl_files
 
 
# =============================================================================
# Section 12 - Initial guess
# =============================================================================
 
def build_initial_guess_nonparam(
    st: FitState,
    coff_init: float,
    coff2_init: float,
    b_bounds: tuple = (1e-6, 1.0),
    w_bounds: tuple = (1e-5, 1.0),
    coff_bounds: Optional[tuple] = None,
    coff2_bounds: tuple = (-0.7, 0.7),
    amp_bounds: tuple = (1e-8, 1e12),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the initial parameter vector and bounds.
 
    Fix vs. original: template weight upper bound is raised to 1.0 + 1e-6
    so that the initial value (1/nt) equals the lower bound only when
    nt == 1, not the *upper* bound.  This avoids a degenerate start.
    """
    nl, nt = st.nl, st.nt
    w_ub = max(w_bounds[1], 1.0 + 1e-6)  # ensure ub > initial = 1/nt for any nt
 
    parts = [np.full(nl, 1 / nl), np.full(nt, 1 / nt)]
    lb    = [np.full(nl, b_bounds[0]), np.full(nt, w_bounds[0])]
    ub    = [np.full(nl, b_bounds[1]), np.full(nt, w_ub)]
 
    if st.icoff == 1:
        cb = coff_bounds or (coff_init - 0.8, coff_init + 4.0)
        if cb[0] <= -1.0:
            raise ValueError(
                f"coff lower bound {cb[0]:.4g} ≤ -1: evaluate_model_gp divides by "
                f"(coff + 1), which would be zero or negative."
            )
        parts.append(np.array([coff_init, coff2_init]))
        lb.append(np.array([cb[0], coff2_bounds[0]]))
        ub.append(np.array([cb[1], coff2_bounds[1]]))
    elif st.icoff == 2:
        parts.append(np.array([coff2_init]))
        lb.append(np.array([coff2_bounds[0]]))
        ub.append(np.array([coff2_bounds[1]]))
 
    if st.fit_global_amp:
        tpl0 = st.t[:, 0]
        good = (st.gerr < 1e9) & np.isfinite(tpl0) & (tpl0 != 0)
        amp_init = (
            float(np.median(st.g[good]) / np.median(tpl0[good])) if good.any() else 1.0
        )
        if not np.isfinite(amp_init) or amp_init <= 0:
            amp_init = 1.0
        parts.append(np.array([amp_init]))
        lb.append(np.array([amp_bounds[0]]))
        ub.append(np.array([amp_bounds[1]]))

    if st.continuum_poly_mode != "none":
        parts.append(np.array([0.0]))
        lb.append(np.array([-getattr(st, "continuum_poly_bound", 0.1)]))
        ub.append(np.array([getattr(st, "continuum_poly_bound", 0.1)]))
 
    return np.concatenate(parts), np.concatenate(lb), np.concatenate(ub)
 
 
# =============================================================================
# Section 13 - Output writing
# =============================================================================
 
def write_fitlov_outputs(
    st: FitState, a_best: np.ndarray, fvalue: float,
    outdir: str = ".", prefix: Optional[str] = None,
    write_prefixed_copies: bool = True,
) -> dict:
    """
    Write spectral fitting outputs to ASCII files.

    Outputs are written with original framework-compatible naming:
    - {prefix}.fit    : LOSVD vector
    - {prefix}.temp   : Template weights
    - {prefix}.ascii  : Per-pixel wavelength, flux, model, template, error_flag
    - {prefix}.rms    : χ² statistics
    """
    outdir = Path("." if outdir is None else outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gp, b, w, coff, coff2, A = evaluate_model_gp(a_best, st)
    sw = float(np.sum(w))
    wfrac = w / sw if sw else w
    tt = compute_weighted_template_spectrum(st, w)
    good = st.gerr < 1e9
    ierr = (~good).astype(int)
    cont = getattr(st, "continuum_mult", np.ones(st.npix))

    # Use original naming scheme (compatible with downstream modeling)
    if prefix:
        fit_path    = outdir / f"{prefix}.fit"
        temp_path   = outdir / f"{prefix}.temp"
        ascii_path  = outdir / f"{prefix}.ascii"
        rms_path    = outdir / f"{prefix}.rms"
    else:
        fit_path    = outdir / "fitlov.fit"
        temp_path   = outdir / "fitlov.temp"
        ascii_path  = outdir / "fitlov.ascii"
        rms_path    = outdir / "fitlov.rms"
 
    # Write .fit file (LOSVD vector)
    # Header: coff, velocity_reference, nlosvd, n_templates
    with open(fit_path, "w") as f:
        f.write(f"{coff:13.8f} {st.xl[0]:16.9g} {st.nl:14d} {st.nt:12d}\n")
        for k in range(st.nl):
            f.write(f"{k + 1:12d} {st.xl[k]:12.6f} {b[k]:17.8E}\n")
 
    # Write .temp file (template weights)
    with open(temp_path, "w") as f:
        for k in range(st.nt):
            f.write(f"{wfrac[k]:17.8E} {k + 1:11d}\n")
 
    # Write .ascii file (per-pixel data for plotting with pspecfit).
    # Format: wavelength  data  model  continuum  template  ierr
    # Columns 1-3: standard spectrum/model vectors.
    # Column 4: ALS continuum (drawn in green by pspecfit as the baseline).
    # Column 5: template-only spectrum (model without continuum scaling).
    # Column 6: integer pixel mask (0=good, 1=masked).
    # pspecfit.f reads: read(2,*) x1,x2,x3,x4,x5,i6  (5 reals + 1 integer).
    with open(ascii_path, "w") as f:
        for i in range(st.iskip, st.npix - st.iskip):
            tt_out = tt[i] * cont[i] if st.fit_als_continuum else tt[i]
            f.write(
                f" {st.x[i]:12.5f}  {st.g[i]:15.8E}  {gp[i]:15.8E}"
                f"  {cont[i]:15.8E}  {tt_out:15.8E} {ierr[i]:1d}\n"
            )
 
    # Write .rms file (χ² and statistics)
    # Format: chi2_reduced  n_good  coff  coff2
    sl = slice(st.iskip, st.npix - st.iskip)
    gfr = good[sl]
    chi2_total = float(np.sum((st.g[sl][gfr] - gp[sl][gfr]) ** 2 / st.gerr[sl][gfr] ** 2))
    nrms = int(gfr.sum())
    chi2_red = chi2_total / max(nrms - 1, 1)
    with open(rms_path, "w") as f:
        f.write(f"{chi2_red:16.8E} {nrms:10d} {coff:14.8g} {coff2:14.8g}\n")
 
    log(f"Outputs written to {outdir}/ (prefix={prefix})")
    log(f"  .fit, .temp, .ascii, .rms files created")

    return {
        "gp": gp, "b": b, "w": w, "wfrac": wfrac, "tt": tt,
        "coff": coff, "coff2": coff2, "A": A,
        "chi2_red": chi2_red, "chi2_total": chi2_total, "nrms": nrms, "continuum": cont,
        "paths": {
            "fit": fit_path, "temp": temp_path, "ascii": ascii_path,
            "rms": rms_path,
        },
    }


def write_losvd_errors_file(
    summary: dict,
    prefix: str,
    outdir: str = ".",
) -> str:
    """
    Write LOSVD vector with error bars to a file for modeling pipeline.
    
    Parameters
    ----------
    summary : dict
        Output from losvd_errors.LOSVDErrorEstimator.summarize()
    prefix : str
        Output file prefix
    outdir : str
        Output directory
    
    Returns
    -------
    str
        Path to written file
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
    
    log(f"Wrote {outfile}")
    return str(outfile)


def write_gh_errors_file(
    summary: dict,
    prefix: str,
    outdir: str = ".",
) -> str:
    """
    Write Gauss-Hermite moment errors to a file.
    
    Parameters
    ----------
    summary : dict
        Output from losvd_errors.LOSVDErrorEstimator.summarize()
    prefix : str
        Output file prefix
    outdir : str
        Output directory
    
    Returns
    -------
    str
        Path to written file
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    gh_map = summary["gh_map"]
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
    
    log(f"Wrote {outfile}")
    return str(outfile)

 
# =============================================================================
# Section 14 - Top-level runner
# =============================================================================

def run_spectral_fit(
    cfg: FitConfig,
    gal_file: Optional[str] = None,
    *,
    write_outputs: bool = True,
    output_prefix: Optional[str] = None,
) -> dict:
    """
    Run the full spectral fit and return a results dict.

    This function supports both command-line and notebook usage.

    Parameters
    ----------
    cfg : FitConfig
        Configuration object. In notebook usage, you may set cfg.gal_file
        directly and call run_spectral_fit(cfg).

    gal_file : str, optional
        Spectrum file path. If supplied, this overrides cfg.gal_file.
        This is useful for command-line usage.

    write_outputs : bool, optional
        If True, write fitlov/ascii/rms/temp output files.
        If False, skip writing files and return model products in memory.

    output_prefix : str, optional
        Prefix for output files. If None, inferred from gal_file.

    Returns
    -------
    fit : dict
        Dictionary containing:
          state          : FitState object
          template_files : list of template files used
          result         : scipy OptimizeResult
          a_map          : best-fit parameter vector
          f_map          : best-fit objective value
          outputs        : output/model dictionary
          chi2           : final chi2
          ngood          : number of good fitted pixels
          prefix         : output prefix
          gal_file       : input spectrum path
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

    # Keep cfg.gal_file synchronized for notebook introspection/debugging.
    cfg.gal_file = gal_file

    log(f"==== spectral fitting START | {gal_file} ====")
    log(
        f"wavefit=[{cfg.wavefitmin}, {cfg.wavefitmax}] z={cfg.zgal} "
        f"sigl={cfg.sigl} xlam={cfg.xlam}"
    )
    log(
        f"fit_als_continuum={cfg.fit_als_continuum} "
        f"prenorm={not cfg.fit_als_continuum}"
    )

    with Timer("build FitState"):
        st, tpl_files = make_fit_state(cfg, gal_file=gal_file)

    a0, xlb, xub = build_initial_guess_nonparam(st, cfg.coff, cfg.coff2)
    bounds = list(zip(xlb, xub))

    res = fit_state_map_with_optional_clean(st, cfg, a0, bounds)
    if not res.success:
        log(f"WARNING: optimizer reported: {res.message}")

    a_map, f_map = np.asarray(res.x, float), float(res.fun)

    prefix = output_prefix if output_prefix is not None else infer_output_prefix(gal_file)

    if write_outputs:
        outputs = write_fitlov_outputs(st, a_map, f_map, cfg.outdir, prefix)
    else:
        # In-memory equivalent of the most useful pieces from write_fitlov_outputs(),
        # without touching the filesystem.
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
            "nrms": int(gfr.sum()),
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
 
 
# =============================================================================
# Section 15 - LOSVD analysis
# =============================================================================
 
def fh3_fortran_like(w: np.ndarray) -> np.ndarray:
    return (2 * w ** 3 - 3 * w) / np.sqrt(3.0)
 
 
def fh4_fortran_like(w: np.ndarray) -> np.ndarray:
    return (4 * w ** 4 - 12 * w ** 2 + 3) / np.sqrt(24.0)


def _gh_poly_vdm(n: int, w: np.ndarray) -> np.ndarray:
    """Normalized Gauss-Hermite polynomial of order n (van der Marel & Franx 1993).

    H_n(w) = He_n(w) / sqrt(n!) where He_n are probabilist Hermite polynomials
    generated by He_{k+1} = w*He_k - k*He_{k-1}. Standard pPXF/GH literature convention.
    """
    w = np.asarray(w, float)
    if n == 0:
        return np.ones_like(w)
    if n == 1:
        return w.copy()
    he_prev = np.ones_like(w)
    he_curr = w.copy()
    for k in range(1, n):
        he_next = w * he_curr - k * he_prev
        he_prev = he_curr
        he_curr = he_next
    return he_curr / np.sqrt(float(math.factorial(n)))


def gauss_hermite_losvd_model_ho(
    v: np.ndarray, amp: float, vel: float, sig: float, *hn: float,
) -> np.ndarray:
    """Gauss-Hermite LOSVD model with h3..h_{3+len(hn)-1} using vdM1993 polynomials."""
    sig = float(sig)
    if sig <= 0 or not np.isfinite(sig):
        return np.full_like(v, np.nan, float)
    w = (np.asarray(v, float) - vel) / sig
    poly_sum = np.ones_like(w, float)
    for i, h in enumerate(hn):
        poly_sum = poly_sum + float(h) * _gh_poly_vdm(i + 3, w)
    return amp * np.exp(-0.5 * w ** 2) / (np.sqrt(2 * np.pi) * sig) * poly_sum


def fit_losvd_gauss_hermite_higher(
    v: np.ndarray, y: np.ndarray, max_order: int = 6,
) -> dict:
    """Fit a Gauss-Hermite model including terms h3..h_{max_order}.

    Uses van der Marel & Franx (1993) normalized Hermite polynomials (vdM1993),
    the same convention as pPXF. Higher-order terms capture non-Gaussian wings,
    giving more accurate V and sigma than a truncated 4-moment fit.

    Returns a dict with keys vherm, sherm, h3..h_{max_order}, model, etc.
    Note: h3/h4 values use vdM1993 normalization and differ numerically from
    the fortran-convention h3/h4 returned by fit_losvd_gauss_hermite.
    """
    v, y = np.asarray(v, float), np.asarray(y, float)
    ok = np.isfinite(v) & np.isfinite(y)
    vfit, yfit = v[ok], y[ok]
    max_order = max(max_order, 4)
    n_hm = max_order - 2  # h3..h_{max_order}
    ho_keys = [f"h{n}" for n in range(3, max_order + 1)]
    nan_result: dict = {k: np.nan for k in ["vherm", "sherm", "amp"] + ho_keys}
    nan_result.update({"model": np.full_like(v, np.nan), "fit_success": False,
                       "fit_message": "Too few points", "max_order": max_order,
                       "convention": "vdM1993"})
    if vfit.size < 5:
        return nan_result

    f = getfwhm_fortran_like(vfit, yfit)
    fwhm_sigma = (
        f["fwhm"] / 2.35 if np.isfinite(f["fwhm"]) and f["fwhm"] > 0
        else max(float(np.std(vfit)), 1.0)
    )
    amp0 = f["ymax"] * np.sqrt(2 * np.pi) * fwhm_sigma
    if not np.isfinite(amp0) or amp0 <= 0:
        amp0 = max(float(np.sum(np.maximum(yfit, 0))), 1e-6)
    vel0 = f["vmax"] if np.isfinite(f["vmax"]) else float(vfit[np.argmax(yfit)])
    sig0 = max(fwhm_sigma, 1.0)
    vspan = max(float(np.ptp(vfit)), float(np.std(vfit)), 1.0)
    sigy = 0.01

    p0 = [amp0, vel0, np.log(sig0)] + [0.0] * n_hm
    lo = [0.0, vfit.min() - 2 * vspan, np.log(1e-6)] + [-0.5] * n_hm
    hi = [np.inf, vfit.max() + 2 * vspan, np.log(1e6)] + [0.5] * n_hm

    def _resid(p: np.ndarray) -> np.ndarray:
        return (
            gauss_hermite_losvd_model_ho(vfit, p[0], p[1], np.exp(p[2]), *p[3:]) - yfit
        ) / sigy

    try:
        res = least_squares(_resid, p0, bounds=(lo, hi), max_nfev=10000)
    except Exception as exc:
        nan_result["fit_message"] = str(exc)
        return nan_result

    amp_fit = float(res.x[0])
    vel_fit = float(res.x[1])
    sig_fit = float(np.exp(res.x[2]))
    hn_fit = res.x[3:]

    result: dict = {
        "vherm": vel_fit, "sherm": sig_fit, "amp": amp_fit,
        "convention": "vdM1993",
        "max_order": max_order,
        "model": gauss_hermite_losvd_model_ho(v, amp_fit, vel_fit, sig_fit, *hn_fit),
        "fit_success": bool(res.success), "fit_message": str(res.message),
    }
    for i, h in enumerate(hn_fit, start=3):
        result[f"h{i}"] = float(h)
    return result


def gauss_hermite_losvd_model(
    v: np.ndarray, amp: float, vel: float, sig: float, h3: float, h4: float,
) -> np.ndarray:
    sig = float(sig)
    if sig <= 0 or not np.isfinite(sig):
        return np.full_like(v, np.nan, float)
    w = (np.asarray(v, float) - vel) / sig
    return (
        amp * np.exp(-0.5 * w ** 2) / np.sqrt(2 * np.pi) / sig
        * (1 + h3 * fh3_fortran_like(w) + h4 * fh4_fortran_like(w))
    )
 
 
def getfwhm_fortran_like(x: np.ndarray, y: np.ndarray, frac: float = 0.5) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return {k: np.nan for k in ["fwhm", "vmax", "ymax", "x_left", "x_right", "v1", "v2"]}
    imax = int(np.argmax(y))
    ymax = float(y[imax]); vmax = float(x[imax]); yhalf = ymax * frac
    x1 = float(x[0])
    for i in range(imax):
        if yhalf >= y[i] and yhalf < y[i + 1]:
            x1 = x[i] + (yhalf - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])
    x2 = float(x[-1])
    for i in range(imax, len(x) - 1):
        if yhalf >= y[i + 1] and yhalf < y[i]:
            x2 = x[i + 1] + (yhalf - y[i + 1]) / (y[i] - y[i + 1]) * (x[i] - x[i + 1])
    sumy = float(np.sum(y))
    v1 = float(np.sum(x * y) / sumy) if sumy else np.nan
    v2 = float(np.sqrt(np.sum(y * (x - v1) ** 2) / sumy)) if sumy else np.nan
    return {
        "fwhm": float(x2 - x1), "vmax": vmax, "ymax": ymax,
        "x_left": x1, "x_right": x2, "v1": v1, "v2": v2,
    }
 
 
def fit_losvd_gauss_hermite(v: np.ndarray, y: np.ndarray, fit_h3h4: bool = True) -> dict:
    """Fit a Gauss-Hermite model to the non-parametric LOSVD vector."""
    v, y = np.asarray(v, float), np.asarray(y, float)
    ok = np.isfinite(v) & np.isfinite(y)
    vfit, yfit = v[ok], y[ok]
    nan_result = {k: np.nan for k in [
        "vherm", "sherm", "fwhm_sigma", "h3", "h4",
        "v1", "v2", "fwhm", "vmax", "ymax", "x_left", "x_right", "amp",
    ]}
    if vfit.size < 5:
        return {**nan_result, "model": np.full_like(v, np.nan), "fit_success": False,
                "fit_message": "Too few points"}
 
    f = getfwhm_fortran_like(vfit, yfit)
    fwhm_sigma = (
        f["fwhm"] / 2.35 if np.isfinite(f["fwhm"]) and f["fwhm"] > 0
        else max(float(np.std(vfit)), 1.0)
    )
    amp0 = f["ymax"] * np.sqrt(2 * np.pi) * fwhm_sigma
    if not np.isfinite(amp0) or amp0 <= 0:
        try:
            amp0 = np.trapezoid(np.maximum(yfit, 0), vfit)
        except AttributeError:
            amp0 = np.trapz(np.maximum(yfit, 0), vfit)
        if not np.isfinite(amp0) or amp0 <= 0:
            amp0 = max(float(np.sum(yfit)), 1e-6)
 
    vel0 = f["vmax"] if np.isfinite(f["vmax"]) else float(vfit[np.argmax(yfit)])
    sig0 = max(
        fwhm_sigma if np.isfinite(fwhm_sigma) and fwhm_sigma > 0 else float(np.std(vfit)),
        1.0,
    )
    vspan = max(float(np.ptp(vfit)), float(np.std(vfit)), 1.0)
    sigy = 0.01
 
    if fit_h3h4:
        p0 = [amp0, vel0, np.log(sig0), 0.0, 0.0]
        lo = [0.0, vfit.min() - 2 * vspan, np.log(1e-6), -1.0, -1.0]
        hi = [np.inf, vfit.max() + 2 * vspan, np.log(1e6),  1.0,  1.0]
 
        def resid(p):
            return (
                gauss_hermite_losvd_model(vfit, p[0], p[1], np.exp(p[2]), p[3], p[4]) - yfit
            ) / sigy
 
        res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=10000)
        amp, vel, log_sig, h3, h4 = res.x
        sig = np.exp(log_sig)
    else:
        p0 = [amp0, vel0, np.log(sig0)]
        lo = [0.0, vfit.min() - 2 * vspan, np.log(1e-6)]
        hi = [np.inf, vfit.max() + 2 * vspan, np.log(1e6)]
 
        def resid(p):
            return (
                gauss_hermite_losvd_model(vfit, p[0], p[1], np.exp(p[2]), 0.0, 0.0) - yfit
            ) / sigy
 
        res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=10000)
        amp, vel, log_sig = res.x
        sig = np.exp(log_sig)
        h3 = h4 = 0.0
 
    return {
        "vherm": float(vel), "sherm": float(sig), "fwhm_sigma": float(fwhm_sigma),
        "h3": float(h3), "h4": float(h4),
        "v1": float(f["v1"]), "v2": float(f["v2"]),
        "fwhm": float(f["fwhm"]), "vmax": float(f["vmax"]), "ymax": float(f["ymax"]),
        "x_left": float(f["x_left"]), "x_right": float(f["x_right"]), "amp": float(amp),
        "model": gauss_hermite_losvd_model(v, amp, vel, sig, h3, h4),
        "fit_success": bool(res.success), "fit_message": str(res.message),
    }
 
 
# =============================================================================
# Section 16 - Diagnostic plots
# =============================================================================
 
def plot_fit(fit: dict) -> None:
    """Plot data vs. model and residuals."""
    st = fit["state"]
    gp = fit["outputs"]["gp"]
    good = st.gerr < 1e9
    resid = (st.g - gp) / gp
 
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax = axes[0]
    ax.plot(st.x, st.g, "k", lw=1, label="data")
    ax.plot(st.x, gp, color="tab:red", lw=1.5, label="model")
    if (~good).any():
        ax.scatter(st.x[~good], st.g[~good], s=12, color="tab:orange",
                   label="masked", zorder=5)
    ax.set_ylabel("Normalised flux" if st.prenormalized else "Flux")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_ylim(st.g.min() * 0.9, st.g.max() * 1.1)
 
    ax = axes[1]
    ax.axhline(0, color="0.5", lw=1)
    ax.plot(st.x, resid, color="tab:blue", lw=1)
    if good.any():
        rms = np.sqrt(np.mean(resid[good] ** 2))
        ax.axhline( rms, color="0.7", ls="--", lw=1)
        ax.axhline(-rms, color="0.7", ls="--", lw=1)
        if (~good).any():
            ax.scatter(st.x[~good], resid[~good], s=12, color="tab:orange",
                       label="masked", zorder=5)
        ax.text(0.01, 0.95, f"RMS={rms:.4g}", transform=ax.transAxes, va="top")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("(Data - model) / model")
    ax.set_ylim(-0.05, 0.05)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
 
 
def plot_losvd(fit: dict) -> None:
    """Plot the non-parametric LOSVD and its Gauss-Hermite fit."""
    st = fit["state"]
    b = fit["outputs"]["b"]
    wfrac = fit["outputs"]["wfrac"]
    gh = fit_losvd_gauss_hermite(st.xl, b, fit_h3h4=True)
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(st.xl, b, "o-", color="tab:blue", label="LOSVD")
    if np.all(np.isfinite(gh["model"])):
        ax.plot(st.xl, gh["model"], color="tab:red", lw=1.8, label="GH fit")
    ax.axvline(gh["vherm"], color="tab:red", ls="--", lw=1.3)

    annotation = (
        rf"$V = {gh['vherm']:.2f}$, $\sigma = {gh['sherm']:.2f}$" "\n"
        rf"$h_3 = {gh['h3']:.3f}$, $h_4 = {gh['h4']:.3f}$" "\n"
        rf"(moments: $V = {gh['v1']:.2f}$, $\sigma = {gh['v2']:.2f}$)"
    )
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.4", alpha=0.85))
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("LOSVD")
    ax.legend()
    ax.grid(alpha=0.25)
 
    axes[1].bar(np.arange(1, len(wfrac) + 1), wfrac, color="tab:gray")
    axes[1].set_xlabel("Template index")
    axes[1].set_ylabel("Weight fraction")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
 
    print(
        f"vherm={gh['vherm']:.4f}  sherm={gh['sherm']:.4f}  "
        f"h3={gh['h3']:.4f}  h4={gh['h4']:.4f}  success={gh['fit_success']}"
    )
 
 
def _mask_to_spans(boolean_mask: np.ndarray, x: np.ndarray) -> list[tuple[float, float]]:
    """Convert a boolean pixel mask to a list of (x_lo, x_hi) contiguous spans."""
    spans = []
    i = 0
    n = len(boolean_mask)
    while i < n:
        if not boolean_mask[i]:
            i += 1
            continue
        i0 = i
        while i + 1 < n and boolean_mask[i + 1]:
            i += 1
        spans.append((float(x[i0]), float(x[i])))
        i += 1
    return spans


# Curated short list of the most prominent stellar features for reference overlays.
# Drawn as thin gray dashed lines when mark_prominent_lines=True.
# (center_A, label)
PROMINENT_STELLAR_LINES: tuple[tuple[float, str], ...] = (
    # ── H Balmer ─────────────────────────────────────────────────────────────
    (4861.33, r"H$\beta$"),
    # ── Fe I / blends (Lick indices) ─────────────────────────────────────────
    (5015.20, "Fe I"),          # Fe5015 Lick index
    (5270.40, "Fe I"),          # Fe5270 Lick index
    (5328.04, "Fe I"),          # Fe5335 Lick index
    (5406.00, "Fe I"),          # Fe5406 Lick index
    (5709.38, "Fe I"),          # Fe5709 Lick index
    (5780.38, "Fe I"),          # Fe5782 Lick index
    # ── Mg b triplet + MgH ───────────────────────────────────────────────────
    (5138.70, "MgH"),           # A²Π-X²Σ⁺ bandhead; gravity-sensitive
    (5167.32, "Mg b"),
    (5172.68, "Mg b"),
    (5183.60, "Mg b"),
    (5528.41, "Mg I"),
    # ── Na I D doublet — very strong in K giants; IMF-sensitive ─────────────
    (5889.95, "Na I D"),
    (5895.92, "Na I D"),
    # ── Ca I ─────────────────────────────────────────────────────────────────
    (6122.22, "Ca I"),
    (6162.17, "Ca I"),
    (6347.23, "Ca I"),
    (6373.48, "Ca I"),
    # ── Ba II (s-process; luminous K giants) ─────────────────────────────────
    (6496.90, "Ba II"),
    # ── TiO molecular bands ───────────────────────────────────────────────────
    # γ' system: ~5847, ~6159, ~6651 Å
    (5847.00, r"TiO $\gamma'$"),
    (6159.00, r"TiO $\gamma'$"),
    (6651.00, r"TiO $\gamma'$"),
    # γ system: 7054–7676 Å (strongest optical TiO complex)
    (7055.00, r"TiO $\gamma$"),
    (7126.00, r"TiO $\gamma$"),
    (7589.00, r"TiO $\gamma$"),
    (7676.00, r"TiO $\gamma$"),
    # δ system
    (8432.00, r"TiO $\delta$"),
    (8859.00, r"TiO $\delta$"),
    # ── K I resonance doublet — gravity-sensitive ─────────────────────────────
    (7664.91, "K I"),
    (7698.96, "K I"),
    # ── O I triplet (photospheric) ────────────────────────────────────────────
    (7771.94, "O I"),
    (7774.17, "O I"),
    (7775.39, "O I"),
    # ── Na I NIR doublet — IMF-sensitive ─────────────────────────────────────
    (8183.27, "Na I"),
    (8194.82, "Na I"),
    # ── Ca II infrared triplet ────────────────────────────────────────────────
    (8498.02, "Ca II"),
    (8542.09, "Ca II"),
    (8662.14, "Ca II"),
    # ── Fe I in CaT region ───────────────────────────────────────────────────
    (8514.08, "Fe I"),
    (8621.61, "Fe I"),
    (8688.63, "Fe I"),
    # ── Mg I ─────────────────────────────────────────────────────────────────
    (8806.76, "Mg I"),
    # ── Near-IR: Pa7, Ca I, CN ───────────────────────────────────────────────
    (9015.00, "Pa8"),
    (9229.00, r"Pa$\delta$"),
    (9257.00, "Ca I"),
    (9350.00, "CN"),
    # ── Common nebular emission ───────────────────────────────────────────────
    (5006.84, "[O III]"),
    (5875.62, "He I"),
    (6300.30, "[O I]"),
    (6562.80, r"H$\alpha$"),
    (6583.45, "[N II]"),
    (6716.44, "[S II]"),
    (6730.82, "[S II]"),
    (7135.80, "[Ar III]"),
    (9068.60, "[S III]"),
)


def plot_als_continuum(
    fit: dict,
    cfg=None,
    mark_prominent_lines: bool = True,
) -> None:
    """
    Diagnostic plot for ALS continuum mode: data, baseline, model, residuals.

    Parameters
    ----------
    fit : dict
        Output of run_spectral_fit.
    cfg : FitConfig, optional
        When supplied the plot additionally overlays masked regions and line
        labels.  Labels are drawn ONLY for lines whose pixels are actually in
        the corresponding mask.
    mark_prominent_lines : bool
        When True (default) draw thin gray reference ticks for the curated
        list of strongest stellar/nebular lines (PROMINENT_STELLAR_LINES)
        that fall in the fit window, regardless of masking status.
    """
    st = fit["state"]
    if not st.fit_als_continuum:
        print("Not in ALS continuum mode — nothing to plot.")
        return

    gp = fit["outputs"]["gp"]
    cont = fit["outputs"]["continuum"]
    good = st.gerr < 1e9

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1]},
    )

    # ── Build mask layers ────────────────────────────────────────────────────
    # all pixels with gerr >= BIG (after full fit: bad regions, emission pre-mask,
    # template gaps, sigma-clipped outliers).
    pre_masked = st.gerr >= BIG

    # Emission pre-mask (stored by make_fit_state before any fitting).
    em_mask_bool = getattr(st, "emission_pre_mask", None)
    if em_mask_bool is None:
        em_mask_bool = np.zeros(st.npix, dtype=bool)
    else:
        em_mask_bool = np.asarray(em_mask_bool, bool)

    # ALS continuum absorption mask: pixels excluded from ALS but still in spectral fit.
    cont_mask = getattr(st, "continuum_mask", None)
    if cont_mask is not None:
        als_abs_excl = ~np.asarray(cont_mask, bool) & ~pre_masked
    else:
        als_abs_excl = np.zeros(st.npix, dtype=bool)

    # "Other rejected" pixels: pre-masked for reasons other than detected emission
    # (bad regions, template coverage gaps, sigma-clipped outliers from fitting).
    other_rejected = pre_masked & ~em_mask_bool

    # ── Panel-specific shading helpers ───────────────────────────────────────
    def _shade_ax(ax, mask, color, alpha=0.20):
        for x0, x1 in _mask_to_spans(mask, st.x):
            ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0,
                       label="_nolegend_")

    # Panel 0 + 1: emission pre-mask (red)
    for x0, x1 in _mask_to_spans(em_mask_bool, st.x):
        axes[0].axvspan(x0, x1, color="tab:red", alpha=0.22, zorder=0,
                        label="_nolegend_")
        axes[1].axvspan(x0, x1, color="tab:red", alpha=0.22, zorder=0,
                        label="_nolegend_")
    # Panel 0 only: ALS absorption mask (purple)
    for x0, x1 in _mask_to_spans(als_abs_excl, st.x):
        axes[0].axvspan(x0, x1, color="tab:purple", alpha=0.22, zorder=0,
                        label="_nolegend_")

    # ── Helper: is this line center actually represented in a mask? ──────────
    def _line_is_masked(cen: float, hw: float, mask: np.ndarray) -> bool:
        pix = (st.x >= cen - hw) & (st.x <= cen + hw)
        return bool(pix.any() and mask[pix].any())

    # ── Collect line markers — ONLY for lines whose pixels are in the mask ───
    # Labels are tied to the actual mask arrays so they only appear when
    # there is corresponding shading.
    abs_lines:   list[tuple[str, float]] = []   # (label, center_rest)
    em_lines:    list[tuple[str, float]] = []
    extra_lines: list[tuple[str, float]] = []

    if cfg is not None:
        wmin = float(getattr(cfg, "wavefitmin", st.x.min()))
        wmax = float(getattr(cfg, "wavefitmax", st.x.max()))
        em_hw  = float(getattr(cfg, "mask_emission_line_half_width_A", 5.0))
        abs_hw = float(getattr(cfg, "als_auto_mask_half_width_A", 5.0))

        # Frame-conversion helper: apply the same als_mask_center_shift_A that
        # the masking functions use, so all plotted lines land in the same place.
        def _to_plot_frame(cen_rest: float) -> float:
            return _center_to_fit_frame(cen_rest, "rest", cfg)

        # Emission lines: label only if pixel actually in pre_masked.
        if getattr(cfg, "mask_emission_lines_in_fit", True):
            seen_em: set[float] = set()
            mask_paschen = bool(getattr(cfg, "mask_paschen_lines_in_fit", False))

            for table in _STELLAR_ABSORPTION_LINE_TABLES:
                for tag, name, cen_rest in table:
                    is_emission_candidate = tag == "em"
                    is_paschen_candidate = mask_paschen and tag == "paschen"

                    if not (is_emission_candidate or is_paschen_candidate):
                        continue

                    if cen_rest in seen_em:
                        continue

                    if cen_rest < wmin - em_hw or cen_rest > wmax + em_hw:
                        continue

                    cen_plot = _to_plot_frame(cen_rest)

                    if _line_is_masked(cen_plot, em_hw, pre_masked):
                        em_lines.append((name, cen_plot))
                        seen_em.add(cen_rest)

        # ALS absorption lines: label only if pixel actually in als_abs_excl.
        if getattr(cfg, "als_auto_mask_abs_lines", False) and als_abs_excl.any():
            mask_paschen = bool(getattr(cfg, "als_auto_mask_paschen", False))
            seen_abs: set[float] = set()
            for table in _STELLAR_ABSORPTION_LINE_TABLES:
                for tag, name, cen_rest in table:
                    if tag == "em" or cen_rest in seen_abs:
                        continue
                    if tag == "paschen" and not mask_paschen:
                        continue
                    if cen_rest < wmin or cen_rest > wmax:
                        continue
                    cen_plot = _to_plot_frame(cen_rest)
                    if _line_is_masked(cen_plot, abs_hw, als_abs_excl):
                        abs_lines.append((name, cen_plot))
                        seen_abs.add(cen_rest)

        # Extra user lines: label only if pixel in protect_extra or als_abs_excl.
        # protect_extra covers the user-supplied als_extra_mask_windows.
        protect_extra = np.zeros(st.npix, dtype=bool)
        extra_frame = getattr(cfg, "als_extra_mask_frame", "rest")
        for lo_raw, hi_raw in getattr(cfg, "als_extra_mask_windows", ()):
            lo = float(lo_raw) / (1.0 + cfg.zgal) if extra_frame == "observed" else float(lo_raw)
            hi = float(hi_raw) / (1.0 + cfg.zgal) if extra_frame == "observed" else float(hi_raw)
            protect_extra |= (st.x >= lo) & (st.x <= hi)
        seen_extra: set[float] = set()
        extra_combined = protect_extra | als_abs_excl
        extra_line_frame = getattr(cfg, "als_extra_mask_line_frame", "rest")
        for item in getattr(cfg, "als_extra_mask_lines", ()):
            c_raw = float(item[0])
            name = item[2] if len(item) >= 3 else f"{c_raw:.1f} Å"
            c_rest = c_raw / (1.0 + cfg.zgal) if extra_line_frame == "observed" else c_raw
            hw = float(item[1]) if len(item) >= 2 else abs_hw
            c_plot = _to_plot_frame(c_rest)
            if c_rest not in seen_extra and _line_is_masked(c_plot, hw, extra_combined):
                extra_lines.append((name, c_plot))
                seen_extra.add(c_rest)
        for item in getattr(cfg, "extra_absorption_lines", ()):
            c_raw = float(item[0])
            hw = float(item[1]) if len(item) >= 2 else abs_hw
            name = item[2] if len(item) >= 3 else f"{c_raw:.1f} Å"
            c_plot = _to_plot_frame(c_raw)
            if c_raw not in seen_extra and _line_is_masked(c_plot, hw, extra_combined):
                extra_lines.append((name, c_plot))
                seen_extra.add(c_raw)

    else:
        def _to_plot_frame(cen_rest: float) -> float:  # type: ignore[misc]
            return cen_rest

    # ── Prominent reference lines (independent of masking) ──────────────────
    ref_lines: list[tuple[str, float]] = []
    if mark_prominent_lines:
        xlo, xhi = float(st.x.min()), float(st.x.max())
        # Avoid duplicating lines already shown as masked labels
        already_labeled = {c for _, c in abs_lines + em_lines + extra_lines}
        for cen_rest, name in PROMINENT_STELLAR_LINES:
            cen_plot = _to_plot_frame(cen_rest)
            if xlo <= cen_plot <= xhi and cen_plot not in already_labeled:
                ref_lines.append((name, cen_plot))

    # ── Data, model, continuum (top panel) ──────────────────────────────────
    ax = axes[0]
    _show_errs = getattr(cfg, "use_spectrum_errors", True)
    ax.plot(st.x, st.g, "k", lw=0.8, label="data")
    if _show_errs:
        gerr_plot = np.where(st.gerr < BIG, st.gerr, np.nan)
        ax.fill_between(st.x, st.g - gerr_plot, st.g + gerr_plot,
                        color="0.8", step="mid", alpha=0.5, label="error")
    ax.plot(st.x, gp, color="C0", lw=1.5, label="model")
    ax.plot(st.x, cont, color="tab:orange", lw=1.2, ls="--", label="ALS continuum")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_patches = []
    if em_mask_bool.any():
        legend_patches.append(Patch(color="tab:red", alpha=0.4, label="Detected Emission"))
    if als_abs_excl.any():
        legend_patches.append(Patch(color="tab:purple", alpha=0.4, label="Detected Absorption (ALS)"))
    if other_rejected.any():
        legend_patches.append(Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="tab:red",
            markersize=4, alpha=0.5, label="rejected pixels",
        ))
    if ref_lines:
        legend_patches.append(Line2D([0], [0], color="0.65", lw=0.7, ls="--",
                                     label="prominent lines (ref)"))
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + legend_patches,
              labels=l + [p.get_label() for p in legend_patches],
              fontsize=8, loc="lower right", frameon=True, fancybox=True, framealpha=0.85)
    ax.set_ylabel("Flux")
    ax.grid(alpha=0.25)
    good_finite = good & np.isfinite(st.g)
    ylo = float(np.nanmin(st.g[good_finite])) if good_finite.any() else float(st.g.min())
    yhi = float(np.nanmax(st.g[good_finite])) if good_finite.any() else float(st.g.max())
    ax.set_ylim(ylo * 0.9, yhi * 1.1)

    for name, cen in ref_lines:
        ax.axvline(cen, color="0.60", lw=0.7, ls="--", alpha=0.7, zorder=1)
        ax.text(cen, yhi * 1.05, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="0.45", clip_on=True)
    for name, cen in em_lines:
        ax.axvline(cen, color="tab:red", lw=0.8, ls=":", alpha=0.85, zorder=3)
        ax.text(cen, yhi * 1.03, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="tab:red", clip_on=True)
    for name, cen in abs_lines:
        ax.axvline(cen, color="tab:purple", lw=0.8, ls=":", alpha=0.8, zorder=3)
        ax.text(cen, yhi * 1.03, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="tab:purple", clip_on=True)
    for name, cen in extra_lines:
        ax.axvline(cen, color="tab:green", lw=0.8, ls="--", alpha=0.85, zorder=3)
        ax.text(cen, yhi * 1.05, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="tab:green", clip_on=True)

    # ── Continuum-normalised (middle panel) ──────────────────────────────────
    ax = axes[1]
    cont_safe = np.where(np.abs(cont) > 0, cont, np.nan)
    g_norm = st.g / cont_safe
    ax.plot(st.x, g_norm, "k", lw=0.8, label="data / ALS continuum")
    ax.plot(st.x, gp / cont_safe, color="C0", lw=1.5, label="model / ALS continuum")
    if _show_errs:
        ax.fill_between(st.x, (st.g - gerr_plot) / cont_safe,
                        (st.g + gerr_plot) / cont_safe,
                        color="0.8", step="mid", alpha=0.5)
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    good_norm = good & np.isfinite(g_norm)
    ylo_n = float(np.nanmin(g_norm[good_norm])) if good_norm.any() else 0.8
    yhi_n = float(np.nanmax(g_norm[good_norm])) if good_norm.any() else 1.2
    ax.set_ylim(ylo_n * 0.95, yhi_n * 1.05)
    ax.set_ylabel("Continuum-normalised")

    em1_patches: list = []
    if em_mask_bool.any():
        em1_patches.append(Patch(color="tab:red", alpha=0.4,
                                 label="Detected Emission (spectral fitting)"))
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(handles=h1 + em1_patches,
              labels=l1 + [p.get_label() for p in em1_patches],
              fontsize=8)
    ax.grid(alpha=0.25)
    for _, cen in ref_lines:
        ax.axvline(cen, color="0.65", lw=0.7, ls="--", alpha=0.6, zorder=1)
    for name, cen in em_lines:
        ax.axvline(cen, color="tab:red", lw=0.8, ls=":", alpha=0.85, zorder=3)
        ax.text(cen, yhi_n * 1.03, name, fontsize=5, rotation=90,
                ha="center", va="bottom", color="tab:red", clip_on=True)
    for _, cen in abs_lines:
        ax.axvline(cen, color="tab:purple", lw=0.8, ls=":", alpha=0.8, zorder=3)
    for _, cen in extra_lines:
        ax.axvline(cen, color="tab:green",  lw=0.8, ls="--", alpha=0.8, zorder=3)

    # ── Residuals (bottom panel) ─────────────────────────────────────────────
    ax = axes[2]
    resid = (st.g - gp) / np.where(np.abs(gp) > 0, gp, np.nan)
    ax.plot(st.x, resid, color="C0", lw=0.8)
    ax.axhline(0, color="0.5", ls="--", lw=1)
    if good.any():
        rms = np.sqrt(np.nanmean(resid[good] ** 2))
        ax.text(0.01, 0.95, f"RMS={rms:.4g}", transform=ax.transAxes, va="top")
    ax.set_xlabel(r"Wavelength ($\AA$, rest frame)")
    ax.set_ylabel(r"(Data $-$ model) / model")
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.1, 0.1)
    for _, cen in ref_lines:
        ax.axvline(cen, color="0.65", lw=0.7, ls="--", alpha=0.6, zorder=1)
    for _, cen in em_lines:
        ax.axvline(cen, color="tab:red",    lw=0.8, ls=":", alpha=0.8, zorder=3)
    for _, cen in abs_lines:
        ax.axvline(cen, color="tab:purple", lw=0.8, ls=":", alpha=0.8, zorder=3)
    for _, cen in extra_lines:
        ax.axvline(cen, color="tab:green",  lw=0.8, ls="--", alpha=0.8, zorder=3)

    # ── Rejected-pixel scatter (all 3 panels) ────────────────────────────────
    # Show every pixel that is flagged as bad (not part of a labeled ALS
    # absorption or detected emission region) as a red dot so it's clear
    # which individual pixels were removed from the fit.
    if other_rejected.any():
        rej_x = st.x[other_rejected]
        rej_g = st.g[other_rejected]
        rej_gn = rej_g / cont_safe[other_rejected]
        rej_r  = resid[other_rejected]
        scatter_kw = dict(s=10, color="tab:red", alpha=0.35, zorder=4,
                          linewidths=0, label="_nolegend_")
        axes[0].scatter(rej_x, np.clip(rej_g,  ylo, yhi),    **scatter_kw)
        axes[1].scatter(rej_x, np.clip(rej_gn, ylo_n, yhi_n), **scatter_kw)
        axes[2].scatter(rej_x, np.clip(rej_r, -0.1,  0.1),   **scatter_kw)

    plt.tight_layout()
    plt.show()
 

        
# =============================================================================
# Section 17 - Entry point
# =============================================================================
 
def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python kinextract.py </your/path/spectrum_file> [</your/path/config_file>]\n"
            "\n"
            "  The TOML config file sets all fit parameters.\n"
            "  The spectrum file is passed separately on the command line.\n"
            "  If no config file is supplied, kinextract.config is searched for\n"
            "  in the same directory as the spectrum file.\n"
        )
        sys.exit(1)

    spectrum_path = sys.argv[1]

    config_path = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else str(Path(spectrum_path).with_name("kinextract.config"))
    )

    log(f"Reading spectrum from {spectrum_path}")
    log(f"Reading config file from {config_path}")

    cfg = load_config_from_toml(config_path)

    fit = run_spectral_fit(cfg, gal_file=spectrum_path)

    log(f"chi2={fit['chi2']:.4g}  ngood={fit['ngood']}  chi2_red={fit['outputs']['chi2_red']:.4g}")

    if cfg.fit_als_continuum:
        plot_als_continuum(fit)
    else:
        plot_fit(fit)

    plot_losvd(fit)

if __name__ == "__main__":
    main()