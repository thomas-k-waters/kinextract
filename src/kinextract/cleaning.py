"""Iterative sigma-clipping ("cleaning") for a LOSVD fit -- robustly
excludes unmodeled contamination (cosmic rays, sky-subtraction residuals,
unexpected emission) from a fit without needing to know its wavelength in
advance, unlike `kinextract.forward_model.mask_wavelength_windows`
(which excludes *known*, fixed wavelength ranges before fitting at all).

Fit once, compute whitened residuals against the effective (weight-
collapsed) design matrix, mask any pixel with `|residual| > clip_sigma`,
refit on the reduced pixel set, repeat up to `max_iter` times or until no
new pixel is flagged. Standard practice (e.g. pPXF's own `clean` option
uses the same idea).

Reserve this for cases with a specific, independent reason to suspect
real per-pixel contamination (a visually obvious cosmic ray, a flagged
bad detector column) -- it is not a general-purpose fix for a fit that
simply looks biased, and can make a model/regularization bias worse
rather than better (the pixels that disagree most with an already-biased
fit are often exactly the ones correctly arguing against that bias).

This module also provides `detect_emission_lines`, a *pre-fit* mask for a
different problem than either of the above: real astrophysical emission
superposed on the stellar spectrum (e.g. [N I], [Cl II] in the Ca II
triplet region). Stellar templates contain no emission lines, so an
unmasked emission feature biases the recovered LOSVD -- inflated sigma in
particular, since the fitter tries to broaden the LOSVD to explain a
narrow emission spike the templates cannot otherwise produce. This is not
a fixed wavelength list (sky-subtraction residuals, `mask_wavelength_windows`'s
usual job) but a per-spectrum, data-driven detection of whether a known
candidate line is actually in emission here, adapted from the original
`kinextract` package's `masking.build_emission_line_mask`.

`detect_emission_from_residual` is a third, complementary approach: instead
of a curated line list, it fits once, finds local maxima in the whitened
*fit residual* above a S/N threshold, and characterizes each one with a
local Gaussian fit -- closer to pPXF's own fit-then-clean philosophy than
`detect_emission_lines`, but replacing `fit_losvd_clean`'s simple per-pixel
`|residual| > clip_sigma` test with a peak-shape-aware one, so it needs no
line list (catches serendipitous/unlisted/velocity-shifted emission) and
masks a window sized to the feature's own fitted width rather than a fixed
half-width or a scattered set of individually-clipped pixels.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .config import FitConfig
from .multi_template import fit_losvd

_DEFAULT_MAX_CLIP_FRACTION = 0.1

# Ca II triplet rest-frame centers/half-widths (Å) -- excluded from the
# flanking-continuum estimate in `detect_emission_lines` so the triplet's
# own absorption troughs cannot pull a candidate line's local continuum
# estimate down and produce a false-positive emission detection (this
# matters concretely here: [N I] 8680.28/8683.40's +/-20 Å default context
# window overlaps the CaT's own 8662.14 Å line).
CAII_TRIPLET_CENTERS_A = (8498.02, 8542.09, 8662.14)
CAII_TRIPLET_HALF_WIDTH_A = 8.0

# Emission-line candidates falling inside the CaII-triplet fit window
# (8400-8800 Å rest-frame) that every real-data notebook in this package
# uses -- the "em"-tagged subset of `kinextract`'s
# `continuum._STELLAR_ABSORPTION_LINE_TABLES` restricted to this range.
# Each entry is `(name, rest_wavelength_A)`.
DEFAULT_CAT_EMISSION_LINES = (
    ("O I", 8446.36),
    ("[Cl II]", 8578.70),
    ("[Fe II]", 8616.95),
    ("[N I]", 8680.28),
    ("[N I]", 8683.40),
    ("[C I]", 8727.13),
)


def detect_emission_lines(wave_rest: np.ndarray, flux: np.ndarray, error: np.ndarray,
                           lines=DEFAULT_CAT_EMISSION_LINES,
                           half_width_A: float = 4.0,
                           snr_threshold: float = 3.0,
                           context_width_A: float = 20.0) -> list[tuple[float, float]]:
    """Detect which candidate emission lines are actually in emission in
    this spectrum, and return their windows for `mask_wavelength_windows`.

    For each `(name, center)` in `lines`: estimate the local continuum and
    noise from the flanking region (within `+/-context_width_A` of `center`,
    excluding `+/-half_width_A` and the Ca II triplet windows), then compute
    `excess_snr = (peak_flux_in_line - cont_est) / noise_est`. The line's
    window is only returned if `excess_snr >= snr_threshold` -- i.e.
    detection is based on excess flux *above the local continuum*, not raw
    S/N, so a strong absorption feature is never mistaken for emission.

    This only flags candidates from a known line list; it will not catch
    contamination at an arbitrary, unlisted wavelength (that is what
    `fit_losvd_clean`'s iterative sigma-clipping is for -- the two are
    complementary, not alternatives).

    Parameters
    ----------
    wave_rest : rest-frame wavelength array (Å).
    flux, error : flux and per-pixel error arrays, same length as `wave_rest`.
    lines : `(name, rest_wavelength_A)` candidates to test. Default
        `DEFAULT_CAT_EMISSION_LINES`, the lines falling inside the
        CaII-triplet 8400-8800 Å window this package's real-data notebooks
        use; pass a different table for a different fit window/instrument.
    half_width_A : half-width (Å) of the window masked/excluded around each
        line center. Default 4.0.
    snr_threshold : minimum excess S/N (over the local continuum) required
        to flag a line as detected. Default 3.0.
    context_width_A : half-width (Å) of the flanking region used to
        estimate the local continuum/noise. Default 20.0.

    Returns
    -------
    list of `(wave_min, wave_max)` windows for lines detected in emission,
    directly usable as (or appended to) the `windows` argument of
    `kinextract.forward_model.mask_wavelength_windows`. Empty if none
    of the candidates are detected as being in emission.
    """
    wave_rest = np.asarray(wave_rest, float)
    flux = np.asarray(flux, float)
    error = np.asarray(error, float)
    good = np.isfinite(flux) & np.isfinite(error) & (error > 0)

    windows: list[tuple[float, float]] = []
    for name, cen in lines:
        line_win = (wave_rest >= cen - half_width_A) & (wave_rest <= cen + half_width_A)
        line_good = line_win & good
        if not line_good.any():
            continue

        ctx = (wave_rest >= cen - context_width_A) & (wave_rest <= cen + context_width_A) & good
        flanks = ctx & ~line_win
        for ca_cen in CAII_TRIPLET_CENTERS_A:
            flanks &= ~((wave_rest >= ca_cen - CAII_TRIPLET_HALF_WIDTH_A)
                        & (wave_rest <= ca_cen + CAII_TRIPLET_HALF_WIDTH_A))
        if flanks.sum() < 5:
            continue

        cont_est = float(np.nanmedian(flux[flanks]))
        noise_est = float(np.nanmedian(error[flanks]))
        if not (np.isfinite(cont_est) and np.isfinite(noise_est) and noise_est > 0):
            continue

        peak_flux = float(np.nanmax(flux[line_good]))
        excess_snr = (peak_flux - cont_est) / noise_est
        if np.isfinite(excess_snr) and excess_snr >= snr_threshold:
            windows.append((cen - half_width_A, cen + half_width_A))

    return windows


def detect_emission_from_residual(wave: np.ndarray, residual: np.ndarray,
                                   nsigma: float = 3.0, mask_sigma: float = 2.0,
                                   window_half_width_px: int = 6,
                                   min_peak_distance_px: int = 3,
                                   max_line_width_A: float = 5.0,
                                   isolation_check_sigma: float = 1.5,
                                   max_mask_fraction: float = 0.15) -> list[tuple[float, float]]:
    """Detect emission-line contamination from a fit's own whitened
    residual, then return windows for `mask_wavelength_windows`.

    Fit once (any `fit_losvd`-family call), form the whitened residual
    `(y_w - model_w)`, then: find local maxima above `nsigma`
    (`scipy.signal.find_peaks`), fit a local Gaussian to each one over a
    `+/-window_half_width_px`-pixel window, and mask
    `+/-mask_sigma * fitted_sigma` around its fitted center. A peak whose
    local Gaussian fit fails to converge, whose fitted width exceeds
    `max_line_width_A`, or that fails the isolation check below, is dropped
    rather than masked.

    The isolation check matters because a local window can only ever
    describe *local* structure: a broad, slowly-varying residual (e.g. a
    continuum/template-mismatch bump many pixels wide, not a genuine
    narrow line) will still contain individual noisy pixels that look like
    a plausible local Gaussian peak, and without this check would get
    fragmented into several spurious narrow "detections" instead of being
    recognized as one broad, non-line-like feature. After fitting, the
    residual is checked just beyond the masked window (at
    `+/-(mask_sigma + 2) * fitted_sigma`); if it hasn't dropped below
    `isolation_check_sigma` there, the feature is more consistent with an
    extended mismatch than an isolated line, and is dropped.

    This needs no line list (unlike `detect_emission_lines`): it flags
    whatever the fitted stellar LOSVD model actually cannot explain, so it
    also catches emission at an unlisted wavelength or an unexpected
    velocity offset from the systemic redshift. Only positive
    (emission-like) residual peaks are considered -- a negative dip (e.g.
    an oversubtracted sky line) is a different problem, already handled by
    a fixed sky-residual window list or `fit_losvd_clean`'s two-sided
    per-pixel clipping.

    Parameters
    ----------
    wave : wavelength array (Å) the residual corresponds to.
    residual : whitened fit residual, `(y_w - model_w)`, same length as
        `wave` -- already in units of sigma, so `nsigma`/`mask_sigma` are
        directly comparable to it.
    nsigma : minimum residual height to register as a candidate peak.
        Default 3.0.
    mask_sigma : mask `+/-` this many fitted-Gaussian sigmas around each
        detected peak's center. Default 2.0 (~95% of the fitted profile's
        flux).
    window_half_width_px : half-width (pixels) of the local window used to
        fit each peak's Gaussian. Default 6.
    min_peak_distance_px : minimum pixel separation between two accepted
        peaks (`find_peaks`' own `distance`) -- avoids fitting the same
        broad feature twice. Default 3.
    max_line_width_A : reject a peak if its fitted Gaussian sigma (Å)
        exceeds this. Default 5.0.
    isolation_check_sigma : reject a peak if the residual (linearly
        interpolated) at `+/-(mask_sigma + 2) * fitted_sigma` from its
        center is still above this value. Default 1.5.
    max_mask_fraction : warn (not error) if the detected windows would
        together cover more than this fraction of `wave`'s own range.
        Default 0.15. This method's whole premise -- that the residual is
        otherwise noise plus a few isolated narrow contaminants -- only
        holds when the underlying fit is already good; if the fit's own
        chi2_red is far from 1 (unmodeled continuum shape, template
        mismatch, mis-scaled noise), the residual is dominated by *that*
        instead, and this method will mistake broad, spatially-correlated
        mismatch for many separate "emission lines" rather than the
        isolated few it is designed to catch. A large flagged fraction is
        the diagnostic signal for exactly that failure mode -- check the
        fit's own chi2_red before trusting these windows when this warns.

    Returns
    -------
    list of `(wave_min, wave_max)` windows for detected peaks, directly
    usable as (or appended to) `mask_wavelength_windows`'s `windows`
    argument. Empty if no peak clears `nsigma` or none survive the local
    Gaussian fit and isolation check.
    """
    wave = np.asarray(wave, float)
    residual = np.asarray(residual, float)

    def _gauss(x, amp, mu, sig):
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    dw = float(np.median(np.diff(wave)))
    peaks, _ = find_peaks(residual, height=nsigma, distance=min_peak_distance_px)

    windows: list[tuple[float, float]] = []
    for p in peaks:
        lo = max(0, p - window_half_width_px)
        hi = min(len(wave), p + window_half_width_px + 1)
        x_local, y_local = wave[lo:hi], residual[lo:hi]
        if len(x_local) < 4:
            continue
        try:
            popt, _ = curve_fit(
                _gauss, x_local, y_local,
                p0=(residual[p], wave[p], 2 * dw),
                bounds=((0.0, x_local[0], 0.5 * dw), (np.inf, x_local[-1], max_line_width_A)),
            )
        except (RuntimeError, ValueError):
            continue
        amp, mu, sig = popt

        check_offset = (mask_sigma + 2.0) * sig
        resid_beyond = np.interp([mu - check_offset, mu + check_offset], wave, residual,
                                  left=0.0, right=0.0)
        if np.any(resid_beyond > isolation_check_sigma):
            continue

        windows.append((float(mu - mask_sigma * sig), float(mu + mask_sigma * sig)))

    wave_range = float(wave.max() - wave.min())
    total_width = sum(hi - lo for lo, hi in windows)
    if wave_range > 0 and total_width / wave_range > max_mask_fraction:
        warnings.warn(
            f"detect_emission_from_residual: {len(windows)} detected window(s) together cover "
            f"{100*total_width/wave_range:.0f}% of the wavelength range, above the "
            f"{100*max_mask_fraction:.0f}% sanity threshold -- this is more consistent with a "
            "poor overall fit (unmodeled continuum shape, template mismatch, mis-scaled noise) "
            "than several genuine isolated emission lines. Check chi2_red on the underlying fit "
            "before trusting these windows; masking this much of the spectrum is more likely to "
            "destabilize the fit than improve it.",
            UserWarning, stacklevel=2,
        )

    return windows


def _robust_sigma(x: np.ndarray) -> float:
    """1.4826*MAD -- a normal-consistent robust scale estimate, insensitive
    to the handful of genuine outliers a clip step is looking for (unlike
    `np.std`, which those same outliers inflate). Falls back to 1.0 if the
    MAD itself is exactly zero (e.g. a degenerate all-equal residual), so a
    caller dividing by this never hits a zero-division."""
    scale = 1.4826 * np.median(np.abs(x - np.median(x)))
    return float(scale) if scale > 0 else 1.0


def fit_losvd_clean(A_stack_w: np.ndarray, y_w: np.ndarray, cfg: FitConfig,
                     clip_sigma: float = 3.0, max_iter: int = 3,
                     max_clip_fraction: float = _DEFAULT_MAX_CLIP_FRACTION,
                     wave_fit: np.ndarray | None = None,
                     robust_scale: bool = False,
                     excluded_wavelength_windows: tuple[tuple[float, float], ...] | None = None) -> dict:
    """LOSVD (+ template-mixture-weight) fit with iterative sigma-
    clipping, for any `n_templates >= 1`. Each iteration's residual is
    computed against the effective (weight-collapsed) design matrix,
    `np.tensordot(w, A_stack_w, axes=([0],[0]))` (further multiplied by
    the continuum `C` first, when `cfg.continuum_knots` is set).

    Parameters
    ----------
    A_stack_w, cfg : as passed to `kinextract.multi_template.fit_losvd`.
    y_w : the observed, noise-whitened flux.
    clip_sigma : reject pixels with `|whitened residual| > clip_sigma`
        (`robust_scale=False`) or `> clip_sigma * robust sigma of the
        current residual` (`robust_scale=True`). Default 3.0, the
        conventional choice (e.g. matching pPXF's own default).
    max_iter : maximum refit iterations. Default 3; stops early if an
        iteration flags no new outliers.
    max_clip_fraction : warn (not error) if more than this fraction of
        pixels end up clipped across all iterations. Default 0.1.
    wave_fit : the observed wavelength grid `A_stack_w`/`y_w` correspond
        to -- required (and only used) if `cfg.continuum_knots` is set,
        exactly as for `kinextract.multi_template.fit_losvd`. Sliced
        by the same `keep` mask as `A_stack_w`/`y_w` each iteration, so
        the continuum's own wavelength basis always matches the current
        surviving pixel set, not the original full one -- a clipped
        pixel's wavelength must not silently leak into the continuum
        spline's own knot placement/support.
    robust_scale : if True, `clip_sigma` is measured against a MAD-based
        robust estimate of the *current* residual's own scale
        (recomputed every iteration from the surviving pixels), not the
        raw whitened residual directly. Default False, preserving this
        function's original behavior exactly -- that behavior silently
        assumes the input noise vector is correctly calibrated (chi2_red
        ~ 1), which real data with a systematic model-data mismatch
        (real chi2_red far from 1 -- unmodeled continuum shape, template
        mismatch, an under-scaled noise vector) violates badly: a nominal
        "3 sigma" cut can end up well under 1 *true* sigma, clipping
        nearly the whole spectrum on the first pass and refitting on an
        unconstrained handful of survivors -- confirmed directly on real
        MUSE data (`examples/notebooks/07_full_galaxy_analysis.ipynb`
        Section 4/8: clipped >90% of pixels and collapsed to a biased,
        unconstrained baseline at the default `clip_sigma=3.0`).
        `robust_scale=True` matches pPXF's own documented practice of
        rescaling the noise before ever clipping (there, by a single
        `sqrt(chi2/dof)` factor from an unclipped fit) -- the MAD-based
        version used here is outlier-robust rather than a single chi2-
        based scalar, and self-corrects each iteration as the surviving
        pixel set changes. Confirmed to behave far better than the
        default on the same real bins that broke it (RMS improved
        substantially, clip fractions realistic, not a runaway).
    excluded_wavelength_windows : optional sequence of `(lo, hi)` rest-frame
        wavelength ranges (same units/frame as `wave_fit`, which is
        required if this is set) that are never candidates for clipping,
        regardless of `clip_sigma` -- e.g. `CAII_TRIPLET_CENTERS_A` +/-
        `CAII_TRIPLET_HALF_WIDTH_A`, protecting real absorption features
        from being mistaken for outliers by an aggressive `clip_sigma`.
        Default `None` (no protected windows, original behavior). Checked
        directly on real data (2026-07-31,
        `dev_notes/real_data_sigma/FINDINGS.md`): CaII-region clipping is a
        small, not dominant, contributor to overall clip fraction there
        (~15% of clipped pixels within one triplet half-width vs. ~12%
        expected by chance) -- this option is a direct safety net against
        that specific risk, not a fix for clip fraction in general.
    """
    n_templates, n_pix, _ = A_stack_w.shape
    keep = np.ones(n_pix, dtype=bool)
    protected = np.zeros(n_pix, dtype=bool)
    if excluded_wavelength_windows:
        if wave_fit is None:
            raise ValueError(
                "fit_losvd_clean: excluded_wavelength_windows was given but wave_fit was not -- "
                "the protected windows are wavelength ranges and need wave_fit to locate them."
            )
        for lo, hi in excluded_wavelength_windows:
            protected |= (wave_fit >= lo) & (wave_fit <= hi)
    n_coef_min = cfg.resolved_n_interior_knots() + cfg.degree + 1  # LOSVD spline coefficient count
    result = None
    n_iter_run = 0
    for it in range(max_iter):
        n_iter_run = it + 1
        if keep.sum() < n_coef_min:
            # Clipping has removed more pixels than there are free LOSVD coefficients --
            # the fit below would be silently underdetermined (or, worse, "succeed" on a
            # meaningless result driven entirely by the roughness penalty, with no visible
            # red flag) rather than raise. Fail loudly and specifically instead: the
            # concrete real case this guards against (found 2026-08-03) is a spectrum
            # where sky-subtraction residuals wash out the real signal badly enough that
            # iterative clipping keeps removing pixels every iteration without converging.
            raise RuntimeError(
                f"fit_losvd_clean: only {keep.sum()} pixels survived clipping (fewer than "
                f"the {n_coef_min} free LOSVD coefficients) after {it} iteration(s) -- this "
                "spectrum's real signal is too contaminated/weak for this fit window and "
                "clip_sigma to recover, not a fit worth trusting. Inspect the input data "
                "(a common cause: sky-subtraction residuals dominating a low-S/N bin)."
            )
        wave_fit_kept = wave_fit[keep] if wave_fit is not None else None
        result = fit_losvd(A_stack_w[:, keep, :], y_w[keep], cfg, wave_fit=wave_fit_kept)
        A_eff = np.tensordot(result["w"], A_stack_w[:, keep, :], axes=([0], [0]))
        C = result.get("C")
        model = C * (A_eff @ result["b"]) if C is not None else (A_eff @ result["b"])
        resid = y_w[keep] - model
        scale = _robust_sigma(resid) if robust_scale else 1.0
        newly_bad_local = (np.abs(resid) > clip_sigma * scale) & ~protected[keep]
        if not np.any(newly_bad_local):
            break
        keep_idx = np.flatnonzero(keep)
        keep[keep_idx[newly_bad_local]] = False

    n_clipped = int((~keep).sum())
    frac_clipped = n_clipped / len(keep)
    if frac_clipped > max_clip_fraction:
        warnings.warn(
            f"fit_losvd_clean: clipped {100*frac_clipped:.1f}% of pixels "
            f"({n_clipped}/{len(keep)}), above the {100*max_clip_fraction:.0f}% "
            "sanity threshold -- this is more consistent with a large, systematic "
            "mismatch (wrong redshift, missing template coverage, wildly wrong "
            "noise level) than a few genuinely bad pixels. Inspect `clipped_mask` "
            "before trusting this result.",
            UserWarning, stacklevel=2,
        )

    result["clipped_mask"] = ~keep
    result["n_clipped"] = n_clipped
    result["n_iterations"] = n_iter_run
    return result


def _failed_fit_result(A_stack_w: np.ndarray, message: str) -> dict:
    """A NaN-filled, `success=False` result dict shaped like
    `fit_losvd_clean`'s own return value -- for a per-bin failure that
    should flag that one bin as unusable (moment maps render it grey,
    e.g. via `plot_bin_map` simply omitting it from `values_by_bin_id`)
    rather than crashing an entire multi-bin batch. A real spectrum can
    fail for reasons that have nothing to do with a coding bug --
    sky-subtraction residuals washing out the real signal at low S/N is
    the concrete case that motivated this (found 2026-08-03 on real
    data) -- so this is a genuine, expected outcome to handle gracefully,
    not a "should never happen" fallback.
    """
    n_templates, _n_pix, n_bins = A_stack_w.shape
    return dict(
        b=np.full(n_bins, np.nan), w=np.full(n_templates, np.nan), coeffs=None,
        success=False, message=message, n_iter=0, n_iterations=0,
        clipped_mask=np.zeros(A_stack_w.shape[1], dtype=bool), n_clipped=0,
    )


def _fit_losvd_clean_worker(args):
    """Top-level (picklable) per-bin worker for `fit_losvd_clean_parallel` --
    process pools need a module-level function, not a closure, since
    workers unpickle it by import path (same reason as `multi_template.
    _fit_losvd_worker`). Catches any exception from this bin's own fit
    and returns `_failed_fit_result` instead of propagating it -- see
    that function's own docstring for why a single bad bin must not
    crash an entire parallel batch."""
    (A_stack_w, y_w, cfg, clip_sigma, max_iter, max_clip_fraction, wave_fit, robust_scale,
     excluded_wavelength_windows) = args
    try:
        return fit_losvd_clean(A_stack_w, y_w, cfg, clip_sigma=clip_sigma, max_iter=max_iter,
                                max_clip_fraction=max_clip_fraction, wave_fit=wave_fit,
                                robust_scale=robust_scale,
                                excluded_wavelength_windows=excluded_wavelength_windows)
    except Exception as e:
        warnings.warn(
            f"fit_losvd_clean_parallel: a bin's fit raised {type(e).__name__}: {e} -- "
            "returning a NaN-filled, success=False result for this bin instead of "
            "crashing the whole batch. Inspect this bin's own input data (a common real "
            "cause: sky-subtraction residuals washing out the real signal at low S/N).",
            UserWarning, stacklevel=2,
        )
        return _failed_fit_result(A_stack_w, f"{type(e).__name__}: {e}")


def fit_losvd_clean_parallel(A_stack_w_list, y_w_list, cfg_list, clip_sigma: float = 3.0,
                              max_iter: int = 3, max_clip_fraction: float = _DEFAULT_MAX_CLIP_FRACTION,
                              wave_fit=None, robust_scale: bool = False, n_jobs: int | None = None,
                              excluded_wavelength_windows: tuple[tuple[float, float], ...] | None = None) -> list:
    """`fit_losvd_clean`, run across many independent bins in parallel --
    same idea and implementation (`ProcessPoolExecutor`, `spawn` context,
    `multi_template._limit_worker_threads`) as `kinextract.multi_template.
    fit_losvd_parallel`; see that function's own docstring for the
    threading/worker-startup details, which apply unchanged here.

    Needed once cleaning stops being an occasional add-on for a handful of
    flagged bins and becomes the default per-bin fitting step: each bin now
    pays up to `max_iter` refits instead of one, so running all of them in
    a serial loop would erase most of `fit_losvd_parallel`'s own speedup.

    Parameters
    ----------
    A_stack_w_list, y_w_list : sequences of the same length, one entry per
        bin, each matching `fit_losvd_clean`'s own `A_stack_w`/`y_w`.
    cfg_list : a single `FitConfig` (reused for every bin) or a sequence
        of one per bin -- see `fit_losvd_parallel`'s own note on this.
    clip_sigma, max_iter, max_clip_fraction, robust_scale : forwarded to
        every bin's own `fit_losvd_clean` call unchanged (not per-bin --
        pass a shared choice for the whole run).
    wave_fit : forwarded per bin, only needed if `cfg.continuum_knots` is
        set or `excluded_wavelength_windows` is given -- a single array
        (shared) or a sequence of one per bin.
    n_jobs : worker process count. `None` (default) uses `os.cpu_count()`,
        memory-permitting -- see `multi_template.resolve_n_jobs`'s own
        docstring. `1` disables the process pool and runs serially
        in-process.
    excluded_wavelength_windows : forwarded to every bin's own
        `fit_losvd_clean` call unchanged (a shared choice, not per-bin) --
        see its own docstring.

    Returns
    -------
    list of `fit_losvd_clean`'s own result dicts, one per bin, in the same
    order as the inputs.
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

    jobs = [(A_stack_w_list[i], y_w_list[i], cfgs[i], clip_sigma, max_iter, max_clip_fraction,
             waves[i], robust_scale, excluded_wavelength_windows) for i in range(n_bins)]

    if n_jobs == 1:
        return [_fit_losvd_clean_worker(job) for job in jobs]

    per_job_bytes = 4 * max((a.nbytes for a in A_stack_w_list), default=0)
    n_jobs = resolve_n_jobs(n_jobs, per_job_bytes)
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("spawn"),
                              initializer=_limit_worker_threads) as pool:
        return list(pool.map(_fit_losvd_clean_worker, jobs))


def compute_fit_rms(flux: np.ndarray, model: np.ndarray, errors: np.ndarray,
                     mask: np.ndarray | None = None) -> float:
    """Whitened-residual RMS for one fit: `sqrt(mean(((flux-model)/errors)**2))`.
    ~1 for a well-calibrated `chi2_red~1` fit; a much larger value points at
    unmodeled contamination (sky-subtraction residuals, emission spikes) or
    a genuine systematic model-data mismatch, not necessarily a bad LOSVD
    recovery -- a single severely-contaminated pixel can dominate this
    mean-of-squares even when the rest of the fit is fine. Use
    `fit_losvd_clean` to test whether iterative sigma-clipping recovers a
    specific flagged case; a large RMS is a flag to investigate, not a
    verdict on its own.

    `mask` : if given, only these pixels (e.g. the surviving, unclipped set
    from a `fit_losvd_clean` result's own `~clipped_mask`) are included --
    the fair comparison against a raw fit's RMS is over the *same* pixel
    set a cleaned fit was actually scored on, not a mix of kept and
    clipped pixels.
    """
    resid = (np.asarray(flux, float) - np.asarray(model, float)) / np.asarray(errors, float)
    if mask is not None:
        resid = resid[np.asarray(mask, bool)]
    return float(np.sqrt(np.mean(resid ** 2)))
