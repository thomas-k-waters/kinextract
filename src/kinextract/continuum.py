"""Standalone continuum-estimation utilities for kinextract.

kinextract's main fitting pipeline (``FitConfig.fit_continuum=True``) always
co-fits its continuum baseline via a penalized-B-spline folded directly into
the same optimization as the LOSVD and template weights -- see
:mod:`kinextract.joint`. An earlier separate-sub-fit approach (an
asymmetric-least-squares baseline, with a hyperparameter search and
pPXF-like iterative absorption-clipping around it) was removed from the
pipeline after being found prone to a real oversmoothing failure mode on
real data.

The core Eilers (2003) asymmetric-least-squares math
(:func:`asymmetric_least_squares_continuum`) is kept here as a standalone
utility: it's still useful for **one-time, manual continuum normalization**
of a raw spectrum outside the main fitting pipeline (fit a continuum once,
divide it out, save a normalized spectrum for later cheap fitting with
``fit_continuum=False``) -- see ``examples/notebooks/06_prenormalized_workflow.ipynb``
for a worked example. :func:`robust_sigma` (a normalized-MAD noise
estimate) is reused elsewhere in the package (e.g. sigma-clipping in
``masking.py``). The stellar/nebular line tables and
:func:`query_nist_lines` NIST line-list utility below are independent of
continuum fitting entirely and are reused by ``masking.py``/``plotting.py``
for line labeling and masking.
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
from ._utils import log
from .config import FitConfig

# =============================================================================
# Section 1 - Asymmetric least-squares continuum math (standalone utility)
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
        Roughness-penalty strength. Larger values produce a smoother,
        stiffer baseline; smaller values let the baseline track the data
        more closely.
    p : float, optional
        Asymmetry weight, in (0, 1). Small values (e.g. 0.001-0.01) push
        the baseline toward the upper envelope of the noise so that
        absorption troughs sit below it. A value of 0.5 is symmetric
        (ordinary smoothing spline).
    niter : int, optional
        Number of asymmetric-reweighting iterations.
    eps : float, optional
        Minimum weight floor applied everywhere, preventing the banded
        system from becoming singular where `base_mask` is False or where
        the reweighting would otherwise assign exactly zero weight.
    return_weights : bool, optional
        If True, also return the final iteration's weight vector and the
        precomputed penalty bands, e.g. for a caller that wants to inspect
        the converged fit's effective degrees of freedom without
        re-running the ALS iteration from scratch.

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


# =============================================================================
# Section 2 - Stellar/nebular line tables and NIST line-list lookup
# =============================================================================

# Common rest-frame features in the 8300-8900 Å region, reused by
# masking.py (build_emission_line_mask, the main pre-fit emission masking)
# and plotting.py (reference-line labeling).
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
        import astropy.units as u  # type: ignore
        from astroquery.nist import Nist  # type: ignore

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
        import urllib.parse
        import urllib.request

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

    return c


