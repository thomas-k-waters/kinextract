"""Spectrum loading and FitState assembly for the kinextract fitter.

This module is the entry point that turns a galaxy spectrum file (MUSE or
STIS IFU spatial-bin spectrum, either a legacy Fortran-style pre-normalized
``.norm`` file or a raw ``.spec`` file) into the fully-populated
:class:`~kinextract.state.FitState` object consumed by the LOSVD-recovery
optimizer. ``load_spectrum_for_fit`` reads the file and selects the
fit-frame wavelength window, handling both continuum modes: pre-normalized
mode reads an already continuum-divided flux column, while continuum-cofit
mode (``cfg.fit_continuum=True``) reads the original (non-divided) flux so
that :mod:`kinextract.joint`'s P-spline continuum can be co-fit with the
LOSVD downstream. ``make_fit_state`` orchestrates loading, applies the
emission-line pre-masking layers from ``masking.py``
(`~kinextract.masking.build_emission_line_mask` and the segment-wise
rolling-median check), builds the interpolated stellar template matrix, and
constructs the LOSVD velocity grid. ``build_initial_guess_nonparam`` then
produces the starting parameter vector
and bounds for the non-parametric LOSVD optimization. If ``cfg.data_fwhm_A``
and ``cfg.template_fwhm_A`` are both set, ``make_fit_state`` also applies
instrumental LSF matching (see :mod:`kinextract.templates`) immediately
after building the template matrix, convolving whichever of the data or
templates has the sharper resolution down to match the coarser one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._utils import BIG, CEE, log
from .config import FitConfig
from .continuum import grow_boolean_mask_A
from .io import (
    build_wavelength_from_index,
    count_in_window,
    estimate_step_from_wavelength,
    read_galaxy_index_flux_err,
    read_norm_spectrum,
    read_template_list,
    select_region_with_errors,
    setbadreg,
)
from .masking import _segment_emission_mask, build_emission_line_mask
from .state import FitState, precompute_ip_map, precompute_losvd_interp
from .templates import (
    build_template_matrix_fortran,
    build_template_matrix_from_npz,
    convolve_gaussian_pixels,
    load_packed_templates,
    resolution_mismatch_sigma_A,
)

# =============================================================================
# Section 10 - Spectrum loading
# =============================================================================

def choose_norm_wavelength_frame(
    wave_raw: np.ndarray, cfg: FitConfig,
) -> tuple[np.ndarray, str]:
    """Determine whether a ``.norm`` file's wavelength column is rest- or
    observed-frame, and return it converted to the frame the fit expects.

    Legacy ``.norm`` files do not consistently record which frame their
    wavelength column is in. This resolves ``cfg.norm_wave_frame``:

    - ``"rest"``: assume the wavelengths are already rest-frame (as-given).
    - ``"observed"``: assume observed-frame and divide by ``1 + cfg.zgal``.
    - ``"auto"``: try both and pick whichever places more samples inside
      the fit window ``[cfg.wavefitmin, cfg.wavefitmax]``, on the
      assumption that the correct frame will have the fit window well
      covered by data.

    Parameters
    ----------
    wave_raw : ndarray
        Raw wavelength column read from the ``.norm`` file (Å).
    cfg : FitConfig
        Fit configuration; supplies ``norm_wave_frame``, ``zgal``,
        ``wavefitmin``, and ``wavefitmax``.

    Returns
    -------
    tuple of (ndarray, str)
        The wavelength array (Å) in the resolved frame, and a short
        human-readable description of which frame/branch was chosen
        (e.g. ``"auto: /1+z"``), used only for logging.
    """
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
    """Resample a de-redshifted spectrum onto a uniform wavelength grid.

    After dividing the observed wavelength array by ``1 + zgal``, the
    pixel spacing is no longer exactly uniform in the rest frame. This
    reproduces the legacy Fortran ``imfit`` behavior of re-gridding onto a
    strictly uniform step (starting at the first rest-frame wavelength) via
    linear interpolation, which several downstream steps (e.g. the LOSVD
    convolution machinery) assume.

    Parameters
    ----------
    x : ndarray
        De-redshifted (rest-frame) wavelength array (Å), non-uniformly
        spaced.
    flux : ndarray
        Flux array corresponding to `x`.
    step : float
        Desired uniform wavelength step (Å) of the output grid.
    extra_arrays : dict, optional
        Additional named arrays (e.g. flux errors, continuum estimates)
        to resample onto the same new grid using the same linear
        interpolation and edge-fill rule. Entries with value `None` are
        passed through as `None` in the output.

    Returns
    -------
    x_new : ndarray
        Uniformly spaced rest-frame wavelength grid (Å), same length as
        the input `x`, starting at ``x[0]``.
    flux_new : ndarray
        Flux linearly interpolated onto `x_new`, with edge values held
        constant beyond the original range.
    extra_new : dict
        Each array in `extra_arrays` similarly resampled onto `x_new`.
    """
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


def load_spectrum_for_fit(
    cfg: FitConfig,
    gal_file: Optional[str] = None,
    gal_errors=None,
):
    """
    Load a galaxy spectrum file and return arrays trimmed to the fit window.

    Supports two input formats and, for ``.norm`` files, both continuum
    modes:

    - Legacy pre-normalized ``.norm`` files (`~kinextract.io.read_norm_spectrum`):
      if ``cfg.fit_continuum`` is False (pre-normalized mode), the
      already continuum-divided ``normflux``/``normflux_err`` columns are
      used directly. If ``cfg.fit_continuum`` is True (continuum-cofit
      mode), the original (non-divided) ``orig_flux``/``orig_flux_err``
      columns are used instead so :mod:`kinextract.joint`'s continuum can
      be co-fit downstream; this requires the ``.norm`` file to include
      those columns (columns 4+). Optionally applies
      `fortran_rebin_after_redshift` to match the legacy Fortran ``imfit``
      uniform-grid convention, and (in pre-normalized mode) an additional
      Fortran-style flux-range mask.
    - Raw ``.spec`` files (`~kinextract.io.read_galaxy_index_flux_err`):
      always effectively continuum-cofit-compatible (the flux column is
      not pre-divided). Supports both integer-index columns (requiring
      ``cfg.wavemin_full``/``cfg.step``) and literal wavelength columns in
      the first field, with the wavelength converted to rest-frame by
      dividing by ``1 + cfg.zgal``.

    In all cases, `gal_errors`, if given, overrides the file's per-pixel
    flux errors (either a scalar broadcast to every pixel, or a full
    per-pixel array), and ``cfg.use_spectrum_errors=False`` (only
    consulted when `gal_errors` is not given) replaces per-pixel errors
    with their uniform median.

    Parameters
    ----------
    cfg : FitConfig
        Fit configuration; supplies ``fit_continuum``, ``zgal``,
        ``wavefitmin``/``wavefitmax``, ``norm_wave_frame``,
        ``norm_error_mode``, ``use_spectrum_errors``, ``fortran_rebin_norm``,
        ``step``, ``wavemin_full``, ``spec_col3_is_variance``,
        ``norm_apply_fortran_flux_mask``, and related range-mask settings.
    gal_file : str, optional
        Path to the spectrum file. Required; a ``.norm`` extension
        (case-insensitive) selects the pre-normalized-file path, anything
        else is treated as a raw ``.spec`` file.
    gal_errors : float or ndarray, optional
        Optional override for the flux uncertainty: a scalar to use a
        uniform error for every pixel, or an array to use per-pixel
        errors directly, bypassing whatever error column the file
        provides.

    Returns
    -------
    x_reg : ndarray
        Rest-frame wavelength array (Å) trimmed to
        ``[cfg.wavefitmin, cfg.wavefitmax]``.
    g_reg : ndarray
        Flux array corresponding to `x_reg` (continuum-divided in
        pre-normalized mode, raw/undivided in continuum-cofit mode).
    ge_reg : ndarray
        Flux uncertainty array corresponding to `x_reg`; pixels flagged
        bad by range or flux masks are set to ``BIG``.
    step_used : float
        Wavelength step (Å) used or estimated for this spectrum.
    prenorm : bool
        True if the pre-normalized (continuum-already-divided) path was
        used; False if this is continuum-cofit mode or a raw ``.spec`` file.
    extra : dict
        Auxiliary arrays useful to downstream code (e.g. the full
        unwindowed wavelength array for LOSVD-grid reference, and, for
        ``.norm`` files, copies of the various flux/error/continuum
        columns both full-length and windowed to the fit region).
    """
    if gal_file is None:
        raise ValueError("gal_file is required to load the spectrum for fitting.")

    prenorm = not cfg.fit_continuum
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
            if gal_errors is not None:
                _ge = np.asarray(gal_errors, dtype=float)
                ferr = np.full(len(flux), float(_ge)) if _ge.ndim == 0 else _ge
                log("gal_errors override applied (prenorm path)")
        else:
            if nd["orig_flux"] is None:
                raise ValueError(
                    "fit_continuum=True with a .norm file requires orig_flux "
                    "(columns 4+ of the .norm file)."
                )
            flux = nd["orig_flux"].copy()
            ferr = (
                nd["orig_flux_err"].copy()
                if nd["orig_flux_err"] is not None
                else np.ones_like(flux)
            )
            if gal_errors is not None:
                _ge = np.asarray(gal_errors, dtype=float)
                ferr = np.full(len(flux), float(_ge)) if _ge.ndim == 0 else _ge
                log("gal_errors override applied (.norm continuum-cofit path)")
            elif not cfg.use_spectrum_errors:
                med = float(np.nanmedian(ferr[ferr > 0])) if np.any(ferr > 0) else 1.0
                ferr = np.full_like(ferr, med)
                log("use_spectrum_errors=False: replaced per-pixel errors with uniform median")
            log("CONTINUUM-COFIT MODE from .norm; using orig_flux / orig_flux_err")
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
    if gal_errors is not None:
        _ge = np.asarray(gal_errors, dtype=float)
        ferr = np.full(len(flux), float(_ge)) if _ge.ndim == 0 else _ge
        log("gal_errors override applied (raw .spec path)")
    elif not cfg.use_spectrum_errors:
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

def make_fit_state(cfg: FitConfig, gal_file: Optional[str] = None, gal_errors=None):
    """
    Load a spectrum and assemble the fully-populated `FitState` for one fit.

    This is the top-level orchestration entry point that turns a single
    galaxy/IFU-bin spectrum file into everything the LOSVD-recovery
    optimizer needs. In order, it:

    1. Loads the spectrum and trims it to the fit window via
       `load_spectrum_for_fit`.
    2. Flags bad/non-finite pixels and out-of-range pixels
       (`~kinextract.io.setbadreg`) by setting their error to ``BIG``.
    3. Applies emission-line pre-masking: known emission-line wavelengths
       via `~kinextract.masking.build_emission_line_mask`, then
       segment-wise rolling-median detection of unlisted emission
       features via `~kinextract.masking._segment_emission_mask`, then
       optionally grows/merges the combined emission mask by
       ``cfg.mask_emission_grow_A``.
    4. Reads and interpolates the stellar template library onto the
       fit-region wavelength grid (`~kinextract.templates.build_template_matrix_fortran`),
       masking pixels outside template coverage and estimating the pooled
       fractional template flux error used later for error propagation.
    5. Builds the non-parametric LOSVD velocity grid (``st.xl``), either
       directly from ``cfg.losvd_vmin``/``cfg.losvd_vmax`` or spanning
       ``±4.5 * cfg.sigl``.
    6. Constructs the `FitState` object with all derived quantities
       (rebinning factors, continuum-polynomial coordinates, etc.).
    7. Precomputes the LOSVD interpolation tables and instrument-profile
       map, and sets a unit continuum placeholder (``st.continuum_mult``);
       when ``cfg.fit_continuum`` is True, :mod:`kinextract.joint` fits and
       overwrites this with the real co-fit continuum.

    Parameters
    ----------
    cfg : FitConfig
        Fit configuration controlling every stage above (redshift,
        wavelength window, continuum-cofit/pre-normalized mode selection,
        template list, LOSVD grid parameters, masking thresholds, etc.).
    gal_file : str, optional
        Path to the galaxy spectrum file, forwarded to
        `load_spectrum_for_fit`.
    gal_errors : float or ndarray, optional
        Optional flux-error override, forwarded to `load_spectrum_for_fit`.

    Returns
    -------
    st : FitState
        Fully populated fit state, ready to be passed to the optimizer
        (LOSVD bins, template matrix, masks, continuum initialization,
        and precomputed interpolation tables all set).
    tpl_files : list of str
        The template file paths used to build the template matrix, in
        the same order as the columns of ``st.t``. When
        ``cfg.template_npz_file`` is set, this is instead the packed
        template names (see :func:`kinextract.templates.load_packed_templates`),
        since there are no individual per-template file paths in that case.

    Raises
    ------
    ValueError
        If fewer than 10 pixels remain in the fit region after loading
        and trimming.
    """
    from ._utils import Timer
    with Timer("read spectrum"):
        x_reg, g_reg, ge_reg, step_used, prenormalized, norm_extra = load_spectrum_for_fit(
            cfg,
            gal_file=gal_file,
            gal_errors=gal_errors,
        )
    if len(x_reg) < 10:
        raise ValueError(f"Too few pixels in fit region: {len(x_reg)}")

    em_mask = np.zeros(len(x_reg), dtype=bool)   # populated below

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
        # not in the known line table -- catches upward outliers the fixed
        # line-table mask above would otherwise miss.
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
        if cfg.template_npz_file is not None:
            # Packed single-file grid (see kinextract.templates.pack_templates_to_npz)
            # takes priority over template_list_file/template_dir when set.
            T, T_err, outside_each, outside_all = build_template_matrix_from_npz(
                x_reg, cfg.template_npz_file, select=cfg.template_npz_select,
            )
            _, _, _npz_meta = load_packed_templates(cfg.template_npz_file, select=cfg.template_npz_select)
            tpl_files = list(_npz_meta["names"])
        else:
            tpl_files = read_template_list(cfg.template_list_file, cfg.template_dir)
            T, T_err, outside_each, outside_all = build_template_matrix_fortran(x_reg, tpl_files)
        if cfg.fortran_mask_template_outside and outside_all.any():
            log(f"Template coverage mask: {outside_all.sum()} pixels masked "
                f"(no template covers them)")
            gerr = np.where(outside_all, BIG, gerr)

        # Pooled fractional template error: median(err/flux) across all templates
        # at pixels where both T and T_err are positive.
        valid_frac = (T > 0) & (T_err > 0)
        f_template = float(np.nanmedian(T_err[valid_frac] / T[valid_frac])) if valid_frac.any() else 0.0
        log(f"Template fractional error (pooled median): {f_template:.4f}")

    if cfg.data_fwhm_A is not None and cfg.template_fwhm_A is not None:
        with Timer("LSF matching"):
            data_fwhm_rest = (
                cfg.data_fwhm_A / (1.0 + cfg.zgal)
                if cfg.data_fwhm_frame == "observed"
                else cfg.data_fwhm_A
            )
            sigma_diff_A, direction = resolution_mismatch_sigma_A(
                data_fwhm_rest, cfg.template_fwhm_A,
            )
            sigma_pix = sigma_diff_A / step_used
            if direction == "convolve_templates":
                log(
                    f"LSF matching: templates are sharper than the data "
                    f"(sigma_diff={sigma_diff_A:.4f} A = {sigma_pix:.3f} pix); "
                    f"convolving templates down to the data resolution"
                )
                T = convolve_gaussian_pixels(T, sigma_pix, axis=0)
                # Approximate error propagation (convolves sigma directly rather
                # than the exact sum-of-squared-weights variance formula);
                # T_err is a diagnostic only (pooled f_template above), not used
                # in the chi-squared objective, so this is a deliberately simple
                # and mildly conservative (slightly over-, not under-estimated)
                # approximation.
                T_err = convolve_gaussian_pixels(T_err, sigma_pix, axis=0)
            elif direction == "convolve_data":
                log(
                    f"LSF matching: data is sharper than the templates "
                    f"(sigma_diff={sigma_diff_A:.4f} A = {sigma_pix:.3f} pix); "
                    f"convolving the galaxy spectrum down to the template resolution"
                )
                # gerr is deliberately left unconvolved: (1) it holds the BIG
                # (1e10) mask sentinel at bad/excluded pixels, and a direct
                # convolution would bleed that into neighboring good pixels,
                # silently corrupting the mask; (2) the statistically correct
                # error propagation under a linear filter is quadrature
                # (variance convolved with the squared kernel weights, then
                # sqrt), not a direct convolution of sigma with the same
                # kernel -- and it also induces pixel-to-pixel noise
                # correlations that a diagonal gerr can't represent at all.
                # Only the flux is convolved; per-pixel weights are left as
                # measured on the original (sharper) data.
                g_reg = convolve_gaussian_pixels(g_reg, sigma_pix)
            else:
                log("LSF matching: data and template resolution already match; no convolution applied")

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
        outside_tpl=outside_each,
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
        fit_continuum=cfg.fit_continuum,
        continuum_poly_mode=cfg.continuum_poly_mode,
        continuum_poly_x=continuum_poly_x,
        continuum_poly_bound=cfg.continuum_poly_bound,
        xlam_wing_shrink=cfg.xlam_wing_shrink,
        xlam_wing_shrink_sfac=cfg.xlam_wing_shrink_sfac,
    )

    st.emission_pre_mask = em_mask

    with Timer("precompute LOSVD + ip map"):
        precompute_losvd_interp(st)
        precompute_ip_map(st)
        # kinextract.joint builds its own P-spline continuum initial guess
        # directly from the data (see kinextract.joint.build_initial_guess)
        # when cfg.fit_continuum=True -- no separate continuum pre-pass is
        # needed here. st.continuum_mult is overwritten with the recovered
        # P-spline continuum once the joint fit completes (see
        # kinextract.joint.run_joint_fit); left at all-ones for
        # pre-normalized-mode fits (cfg.fit_continuum=False).
        st.continuum_mult = np.ones(st.npix)

    log(
        f"STATE: npix={st.npix} nt={st.nt} nl={st.nl} nlosvd={st.nlosvd} "
        f"prenormalized={st.prenormalized} fit_continuum={st.fit_continuum}"
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
    Build the initial non-parametric-LOSVD parameter vector and its bounds.

    Assembles the flat parameter vector used by the optimizer, concatenating
    (in order): `nl` non-parametric LOSVD histogram bin weights (each
    initialized to a flat ``1/nl``, i.e. a uniform velocity distribution),
    `nt` stellar template weights (each initialized to a flat ``1/nt``),
    then optionally a continuum/wavelength-offset block (depending on
    ``st.icoff``), a global amplitude parameter (if
    ``st.fit_global_amp``), and a continuum-polynomial coefficient (if
    ``st.continuum_poly_mode`` is not ``"none"``). The corresponding lower-
    and upper-bound vectors are built in the same order.

    Parameters
    ----------
    st : FitState
        Fit state; supplies ``nl`` (number of LOSVD velocity bins), ``nt``
        (number of stellar templates), ``icoff`` (which
        continuum-offset parameterization is active), ``fit_global_amp``,
        ``continuum_poly_mode``, ``continuum_poly_bound``, ``t`` (template
        matrix, for the global-amplitude initial guess), and ``gerr``/``g``.
    coff_init : float
        Initial value for the primary continuum/wavelength offset
        coefficient (used when ``st.icoff == 1``).
    coff2_init : float
        Initial value for the secondary continuum/wavelength offset
        coefficient (used when ``st.icoff`` is 1 or 2).
    b_bounds : tuple of (float, float), optional
        ``(lower, upper)`` bounds applied to every LOSVD bin weight.
    w_bounds : tuple of (float, float), optional
        ``(lower, upper)`` bounds applied to every template weight; the
        effective upper bound is widened to ``max(w_bounds[1], 1 + 1e-6)``
        (see Notes).
    coff_bounds : tuple of (float, float), optional
        ``(lower, upper)`` bounds for the primary offset coefficient
        (only used when ``st.icoff == 1``). Defaults to
        ``(coff_init - 0.8, coff_init + 4.0)`` if not given. The lower
        bound must be greater than -1 (see Raises).
    coff2_bounds : tuple of (float, float), optional
        ``(lower, upper)`` bounds for the secondary offset coefficient
        (used when ``st.icoff`` is 1 or 2).
    amp_bounds : tuple of (float, float), optional
        ``(lower, upper)`` bounds for the global amplitude parameter
        (only used when ``st.fit_global_amp`` is True).

    Returns
    -------
    x0 : ndarray
        Initial flat parameter vector for the optimizer.
    lb : ndarray
        Lower bounds, same length and ordering as `x0`.
    ub : ndarray
        Upper bounds, same length and ordering as `x0`.

    Raises
    ------
    ValueError
        If the resolved lower bound on the primary continuum offset is
        ``<= -1``, since the model evaluation divides by ``(coff + 1)``
        and such a bound would allow a zero or negative denominator.

    Notes
    -----
    The template-weight upper bound is raised to at least ``1.0 + 1e-6``
    so that the initial value (``1/nt``) coincides with the lower bound
    only in the degenerate ``nt == 1`` case, never with the *upper*
    bound; starting exactly at an upper bound can cause some optimizers
    to treat the parameter as already converged.
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
