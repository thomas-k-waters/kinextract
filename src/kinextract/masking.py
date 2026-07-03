"""Pixel-masking layers for kinextract, distinct from (but complementary to)
the ALS continuum masking in ``continuum.py``.

This module implements two separate masking systems used by the LOSVD fit.
The first is emission-line pre-masking (`build_emission_line_mask`,
`_segment_emission_mask`): known emission-line wavelengths (e.g. [N I],
[Cl II]) are flagged and set to ``gerr = BIG`` *before* fitting begins,
whenever a local excess signal-to-noise over the flanking continuum
indicates the line is actually in emission in this spectrum; a segment-wise
rolling-median check additionally catches emission features absent from the
line table. The second is the kinematic-fit cleaning/protection system
(`build_clean_protect_mask`, `get_ca_centers_for_cleaning`,
`clean_ca_half_width_A`, `emission_half_width_A`): this builds a "protect
mask" of windows (Ca II triplet, custom wavelength windows, extra absorption
lines) that must never be sigma-clipped out of the LOSVD chi-squared fit by
`_update_clean_mask`'s iterative pPXF-like outlier rejection, regardless of
residual sign, because they are known to contain real spectral features
rather than outliers. `_update_clean_mask` itself performs that iterative
sigma-clipping and is documented in place; it is not modified here.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from ._utils import CEE, BIG, log
from .config import FitConfig
from .state import FitState


# =============================================================================
# Section 9 - Sigma-clipping / cleaning (masking functions)
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
    """Return the Ca II triplet line centers in the ``st.x`` wavelength frame.

    Converts ``cfg.clean_ca_centers`` (given in the frame named by
    ``cfg.clean_protect_ca_frame``, typically rest-frame catalog
    wavelengths) into the frame of ``st.x`` via `_clean_center_to_fit_frame`,
    so that `build_clean_protect_mask` can build protection windows
    directly against ``st.x``.

    Parameters
    ----------
    st : FitState
        Fit state (only used indirectly, via `cfg`, for the redshift
        conversion; kept as a parameter for interface symmetry with other
        masking helpers).
    cfg : FitConfig
        Fit configuration; supplies ``clean_ca_centers`` (Å) and
        ``clean_protect_ca_frame`` (``"rest"`` or ``"observed"``).

    Returns
    -------
    ndarray
        Ca II triplet centers (Å) expressed in the ``st.x`` frame.
    """
    centers = np.asarray(cfg.clean_ca_centers, float)
    return np.array(
        [
            _clean_center_to_fit_frame(c, cfg.clean_protect_ca_frame, cfg)
            for c in centers
        ],
        dtype=float,
    )

def emission_half_width_A(cen_A, cfg):
    """Effective emission-line pre-mask half-width, widened for large velocity pad.

    Returns ``max(cfg.mask_emission_line_half_width_A, cen_A * vpad / c)``,
    i.e. the larger of a fixed Angstrom half-width and a velocity-equivalent
    half-width at the line's observed wavelength, so masking in
    `build_emission_line_mask`-adjacent logic scales with the requested
    velocity padding (``mask_emission_line_velocity_pad_kms``) at redder
    wavelengths.

    Parameters
    ----------
    cen_A : float
        Line center wavelength (Å) in the frame being masked.
    cfg : FitConfig
        Fit configuration; supplies ``mask_emission_line_half_width_A``
        (Å) and ``mask_emission_line_velocity_pad_kms`` (km/s).

    Returns
    -------
    float
        Effective half-width in Angstroms.
    """
    hw_A = float(getattr(cfg, "mask_emission_line_half_width_A", 5.0))
    vpad = float(getattr(cfg, "mask_emission_line_velocity_pad_kms", 300.0))
    return max(hw_A, cen_A * vpad / CEE)


def clean_ca_half_width_A(cen_A: float, cfg: "FitConfig") -> float:
    """Effective Ca II protect half-width (Å), widened for broad LOSVDs.

    Returns ``max(cfg.clean_ca_half_width, cen_A * vpad / c)`` so that a
    fixed Angstrom half-width tuned for a narrow LOSVD does not silently
    clip the true absorption wings of a broader one (large velocity
    dispersion ``sigl``). This scaling was added to fix protect windows
    that were too narrow for galaxies with broad LOSVDs, where the Ca II
    troughs are Doppler-broadened well beyond a fixed few-Angstrom window.

    Parameters
    ----------
    cen_A : float
        Ca II line center wavelength (Å), in the frame being protected
        (typically ``st.x``, i.e. already converted via
        `get_ca_centers_for_cleaning`).
    cfg : FitConfig
        Fit configuration; supplies ``clean_ca_half_width`` (Å, default
        6.0) and ``clean_protect_velocity_pad_kms`` (km/s, default 400.0).

    Returns
    -------
    float
        Effective half-width in Angstroms to use for the protect window
        around `cen_A`.
    """
    hw_A = float(getattr(cfg, "clean_ca_half_width", 6.0))
    vpad = float(getattr(cfg, "clean_protect_velocity_pad_kms", 400.0))
    return max(hw_A, cen_A * vpad / CEE)

def build_emission_line_mask(
    x: np.ndarray,
    g: np.ndarray,
    gerr: np.ndarray,
    cfg: "FitConfig",
) -> np.ndarray:
    """Build a boolean mask of emission-line pixels to exclude from the fit.

    Called from `~kinextract.spectrum.make_fit_state` to set ``gerr = BIG``
    at detected emission-line pixels before any LOSVD/template fitting
    begins. Stellar templates contain no emission lines, so an unmasked
    emission feature (e.g. [N I], [Cl II], or other nebular/AGN lines
    superposed on the stellar spectrum) would otherwise bias the recovered
    LOSVD; pre-masking ensures this even for lines too weak to later be
    caught by the kinematic sigma-clipping in `_update_clean_mask`.

    Detection is based on the *excess* flux above the local continuum
    within the line window, not on the raw continuum signal-to-noise of
    the region (which would produce false positives near strong, high-S/N
    absorption features like Ca II even when no emission is present). For
    each candidate line (drawn from the "em"-tagged, and optionally
    "paschen"-tagged, entries of
    `~kinextract.continuum._STELLAR_ABSORPTION_LINE_TABLES`):

      1. Estimate the local continuum and noise from the flanking regions
         just outside the line window (within ``±snr_context_A`` but
         excluding ``±half_width_A``), after excluding the Ca II triplet
         windows from the flanks so their absorption troughs cannot bias
         the continuum estimate downward.
      2. Compute ``excess_snr = (peak_flux_in_line - cont_est) / noise_est``.
      3. Mask the line window only if ``excess_snr >= snr_threshold``.

    Parameters
    ----------
    x : ndarray
        Wavelength array (Å), same frame as the fit state's ``st.x``.
    g : ndarray
        Flux array corresponding to `x`.
    gerr : ndarray
        Flux uncertainty array corresponding to `x`; used to estimate the
        local noise level and to identify already-bad pixels
        (``gerr >= BIG``).
    cfg : FitConfig
        Fit configuration; supplies ``mask_emission_lines_in_fit``,
        ``mask_emission_line_half_width_A``, ``mask_emission_line_snr_threshold``,
        ``mask_emission_line_snr_context_A``, ``mask_paschen_lines_in_fit``,
        ``wavefitmin``/``wavefitmax``, and the Ca II window settings
        (``als_ca_centers``, ``als_ca_half_widths``) used to protect the
        flanking-continuum estimate.

    Returns
    -------
    ndarray of bool
        Mask of length ``len(x)``; True marks pixels to exclude from the
        fit (by setting ``gerr = BIG``). All-False if
        ``cfg.mask_emission_lines_in_fit`` is False.
    """
    from .continuum import _STELLAR_ABSORPTION_LINE_TABLES
    mask = np.zeros(len(x), dtype=bool)
    if not getattr(cfg, "mask_emission_lines_in_fit", True):
        return mask
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

            # Widen the fixed Angstrom half-width for high velocity-pad
            # configs, same idiom as clean_ca_half_width_A for Ca II.
            hw = emission_half_width_A(cen_rest, cfg)

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

            # Exclude Ca II triplet absorption from the flank window.  Their
            # troughs pull cont_est down, making adjacent continuum pixels look
            # like emission (false positive for lines just redward of Ca II).
            abs_ctrs = np.asarray(
                getattr(cfg, "als_ca_centers", (8498.02, 8542.09, 8662.14)), float
            )
            abs_hws = np.asarray(
                getattr(cfg, "als_ca_half_widths", (14.0, 16.0, 18.0)), float
            )
            for _ac, _ahw in zip(abs_ctrs, abs_hws):
                flanks &= ~((x >= _ac - _ahw) & (x <= _ac + _ahw))

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
    """Build the boolean "protect" mask exempting known real features from clipping.

    This mask feeds directly into `_update_clean_mask`'s ``protect_mask``
    argument during the iterative kinematic sigma-clipping of the LOSVD
    chi-squared fit: pixels marked True here are fully exempt from being
    clipped as outliers, regardless of the sign of their residual, because
    they are known to contain real spectral signal (Ca II triplet
    absorption, user-specified windows, or extra absorption lines) rather
    than cosmic rays, sky-subtraction residuals, or emission contamination.
    Without this protection, iterative sigma-clipping could progressively
    erode real, deep absorption features that legitimately produce large
    residuals against an imperfect model.

    Combines three independent sources of protection:

      1. The Ca II triplet, at centers from `get_ca_centers_for_cleaning`
         with a per-line half-width from `clean_ca_half_width_A` (which
         widens the window for broad LOSVDs), if
         ``cfg.clean_protect_ca_triplet`` is True.
      2. User-specified custom wavelength windows (``cfg.clean_protect_windows``),
         converted to the ``st.x`` frame via `_clean_window_to_fit_frame`.
      3. User-specified extra absorption lines (``cfg.extra_absorption_lines``),
         each given as ``(center, half_width[, name])``.

    Parameters
    ----------
    st : FitState
        Fit state; supplies the wavelength grid ``st.x`` and pixel count
        ``st.npix``.
    cfg : FitConfig
        Fit configuration; supplies ``clean_protect_ca_triplet``,
        ``clean_protect_windows``, ``clean_protect_windows_frame``,
        ``extra_absorption_lines``, and ``als_mask_center_shift_A``.

    Returns
    -------
    ndarray of bool
        Mask of length ``st.npix``; True marks pixels that must never be
        rejected by the kinematic-fit sigma-clipping regardless of
        residual sign.
    """
    protect = np.zeros(st.npix, dtype=bool)

    if cfg.clean_protect_ca_triplet:
        for cen in get_ca_centers_for_cleaning(st, cfg):
            hw = clean_ca_half_width_A(cen, cfg)
            m = (st.x >= cen - hw) & (st.x <= cen + hw)
            protect |= m
            log(
                f"  protecting Ca II [{cen - hw:.1f}, "
                f"{cen + hw:.1f}] npix={m.sum()}"
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
    protect_absorption_only: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Update the good-pixel mask for one iterative-cleaning pass.

    The sigma-clip is one-sided: only pixels with *positive* residuals
    (``resid = (data - model) / gerr > sigma_clip * sigma``, i.e. the data
    reads higher than the model, as from emission contamination or a cosmic
    ray) are ever rejected.  Negative residuals (data below the model, i.e.
    deeper apparent absorption) are never clipped on their own, since real
    template absorption features can legitimately produce them.

    Because the clip is already one-sided, ``protect_mask`` pixels need an
    explicit exemption to be protected against a *positive*-residual clip.
    ``protect_absorption_only`` controls how far that exemption reaches:

    - ``False`` (default): every pixel in ``protect_mask`` is fully exempt
      from clipping, regardless of the sign of its residual.  Use this for
      windows you know contain real signal (e.g. the Ca II triplet) that
      must never be dropped, even if a positive residual there is caused by
      continuum or template mismatch rather than a genuine outlier.
    - ``True``: ``protect_mask`` pixels only gain protection when
      ``resid < 0`` — a case that is already exempt from the one-sided clip
      for *every* pixel, protected or not.  This setting therefore leaves
      ``protect_mask`` with no practical effect and is kept only for
      backward compatibility; new code should leave the default in place.
    """
    from .numerics import evaluate_model_gp
    from .continuum import robust_sigma
    gp, *_ = evaluate_model_gp(a_best, st)
    resid = (st.g - gp) / np.where(base_gerr > 0, base_gerr, 1.0)
    sigma = robust_sigma(resid[good_mask])
    if not np.isfinite(sigma) or sigma <= 0:
        return good_mask.copy(), sigma
    keep = resid <= sigma_clip * sigma
    if protect_mask is not None:
        pm = np.asarray(protect_mask, bool)
        keep = keep | (pm & (resid < 0) if protect_absorption_only else pm)
    return good_mask & keep, sigma


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
