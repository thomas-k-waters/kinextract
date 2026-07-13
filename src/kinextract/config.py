"""
Fit configuration for the kinextract spectral-fitting pipeline.

Every tunable knob in the package -- wavelength/redshift setup, the
non-parametric LOSVD grid, continuum handling (pre-normalised vs.
continuum-cofit via :mod:`kinextract.joint`), regularisation (``xlam``)
and its automatic selection, the L-BFGS-B optimizer, iterative
sigma-clipping ("cleaning"), and emission-line masking -- is a field on
:class:`FitConfig`. There is no separate "advanced settings" object:
:func:`~kinextract.fitting.run_spectral_fit` and
:class:`~kinextract.errors.LOSVDErrorEstimator` both take a single
``FitConfig`` instance.

Because ``FitConfig`` has on the order of 100 fields grouped into more
than a dozen subsystems, two complementary ways to explore it are
provided:

- Static documentation: ``help(FitConfig)`` or ``FitConfig?`` in
  IPython/Jupyter shows the full class docstring below, which describes
  every field grouped by subsystem.
- Runtime introspection: :meth:`FitConfig.describe` prints the same
  grouping but resolved against either the dataclass defaults or a live
  instance's current values, and can be filtered with a substring, e.g.
  ``FitConfig.describe("joint")`` to see only joint-continuum-related knobs.

See Also
--------
load_config_from_toml : Build a FitConfig from a TOML file on disk.
"""
from __future__ import annotations

import warnings
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np

# ── TOML parsing (stdlib on 3.11+, falls back to third-party tomli) ──────────
try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # pip install tomli
    except ModuleNotFoundError:
        tomllib = None            # type: ignore[assignment]
        warnings.warn(
            "Neither tomllib (Python >= 3.11 stdlib) nor tomli (pip install tomli) "
            "could be imported.  Config files cannot be loaded — the framework "
            "will crash when attempting to read kinextract.config.",
            ImportWarning,
            stacklevel=1,
        )


# =============================================================================
# Section 2 - Configuration dataclass
# =============================================================================

# Maps each FitConfig field name to (section, one-line help).  Used only by
# FitConfig.describe() for grouped, filterable runtime introspection; it has
# no effect on parsing/validation.  Fields not listed here (should not
# happen in practice -- see the test that checks for full coverage) fall
# back to showing just their type and value.
_FIELD_HELP: dict[str, tuple[str, str]] = {
    # ── Paths ────────────────────────────────────────────────────────────
    "gal_file": ("Paths", "Spectrum file to fit. Optional here; usually passed to run_spectral_fit(cfg, gal_file=...)."),
    "template_list_file": ("Paths", "Path to the template-list file (one template filename per line). Ignored if template_npz_file is set."),
    "template_dir": ("Paths", "Directory the template list's relative paths are resolved against (defaults to the list file's directory). Ignored if template_npz_file is set."),
    "template_npz_file": ("Paths", "Path to a packed single-file template grid (see kinextract.templates.pack_templates_to_npz). Takes priority over template_list_file/template_dir when set."),
    "template_npz_select": ("Paths", "Optional list of (age_gyr, metallicity) pairs to select from template_npz_file. None (default) uses every template in the file. Ignored unless template_npz_file is set."),
    "regions_bad_path": ("Paths", "Path to a 'regions.bad' file of wavelength intervals to exclude from the fit."),
    "outdir": ("Paths", "Directory where .fit/.temp/.ascii/.rms output files are written. Only used if write_outputs is True."),
    "write_outputs": ("Paths", "Write {prefix}.fit/.temp/.ascii/.rms files to outdir after each run_spectral_fit() call. Defaults to False, so results stay in memory only (e.g. for scripted/notebook use or batch-fitting many spectra); set True (or pass run_spectral_fit(..., write_outputs=True)) to write files to disk."),

    # ── Wavelength / redshift ────────────────────────────────────────────
    "wavemin_full": ("Wavelength/redshift", "Wavelength (A) of pixel 1 of the FULL spectrum. Required for raw index-format .spec files."),
    "step": ("Wavelength/redshift", "Wavelength step (A/pixel). Required for raw .spec files, and for .norm files when fortran_rebin_norm=True."),
    "wavefitmin": ("Wavelength/redshift", "Lower bound (rest-frame A) of the wavelength range used in the fit."),
    "wavefitmax": ("Wavelength/redshift", "Upper bound (rest-frame A) of the wavelength range used in the fit."),
    "zgal": ("Wavelength/redshift", "Galaxy (systemic) redshift used to shift observed wavelengths to rest frame."),
    "norm_wave_frame": ("Wavelength/redshift", "How to interpret a .norm file's wavelength column: 'auto', 'rest', or 'observed'."),

    # ── Kinematic grid ───────────────────────────────────────────────────
    "galaxy_params_path": ("Kinematic grid", "Path to (or directory containing) a legacy galaxy.params file supplying vmin/vmax."),
    "use_galaxy_params_velocity_bounds": ("Kinematic grid", "If True and galaxy_params_path is set, adopt its vmin/vmax as losvd_vmin/losvd_vmax."),
    "sigl": ("Kinematic grid", "Initial guess for the LOSVD velocity dispersion (km/s); sets the default LOSVD grid width. If this is comparable to or below the instrument's LSF sigma, see the class docstring's 'Known limitations' section."),
    "xlam": ("Kinematic grid", "LOSVD roughness-penalty regularization strength. Larger = smoother LOSVD. Ignored if xlam_auto=True."),

    # ── Auto smoothing (xlam) selection ──────────────────────────────────
    "xlam_auto": ("Auto xlam selection", "If True, grid-search xlam_auto_grid and pick the best value per xlam_criterion, instead of using the fixed xlam."),
    "xlam_auto_grid": ("Auto xlam selection", "Candidate xlam values searched when xlam_auto=True. Only the min/max are used as the discrepancy search's initial bracket (expanded automatically if needed). Default spans 100-1e7 -- wide enough for the default 89-bin LOSVD grid to find its (much larger) natural regularization strength without needing bracket expansion; see n_losvd_bins."),
    "xlam_criterion": ("Auto xlam selection", "'discrepancy' (default: Cappellari/pPXF-style 1-D search targeting a known chi2 rise, robust to a flat chi2(xlam) curve), 'chi2' (legacy, scale-invariant grid+tolerance), or 'roughness' (legacy) selection rule; see class docstring."),
    "xlam_chi2_tolerance": ("Auto xlam selection", "Max fractional chi2_red increase over the grid minimum still considered acceptable (xlam_criterion='chi2')."),
    "xlam_smooth_threshold": ("Auto xlam selection", "Roughness threshold for xlam selection (xlam_criterion='roughness' only)."),
    "xlam_discrepancy_nsigma": ("Auto xlam selection", "Multiplier on the sqrt(2*ngood) chi2-rise target (xlam_criterion='discrepancy' only). Cappellari's own pPXF convention uses 1.0, but that value was empirically calibrated (benchmarks/calibrate_xlam.py) to be too large for kinextract's LOSVD-histogram parameterization; 0.3 is the calibrated default -- see class docstring."),
    "xlam_max_peaks": ("Auto xlam selection", "Reject xlam grid points whose recovered LOSVD has more than this many prominent peaks."),
    "xlam_peak_min_prominence": ("Auto xlam selection", "Minimum peak prominence (fraction of the global LOSVD peak) counted by xlam_max_peaks."),
    "xlam_auto_maxiter": ("Auto xlam selection", "Optimizer iteration budget for each xlam-grid trial fit (None -> map_maxiter)."),
    "xlam_wing_shrink": ("Auto xlam selection", "Opt-in L2 amplitude penalty on the LOSVD, active only in the wing region (default 0.0/off). The roughness penalty alone penalizes curvature, not amplitude, so it does little to suppress a flat, non-zero LOSVD pedestal far from the peak. This term adds a direct cost for that, but it cannot distinguish a genuine non-Gaussian wing/tail (a skewed or double-peaked LOSVD) from noise, so it will suppress real structure too; combined with xlam_auto=True it can also destabilize the auto-xlam search. Only enable it when there is independent reason to expect a compact true LOSVD, and check bootstrap error bars (not just the point estimate) afterward."),
    "xlam_wing_shrink_sfac": ("Auto xlam selection", "Onset of the xlam_wing_shrink taper, in units of sigl0 (default 1.8, matching the roughness penalty's own onset). Set larger (e.g. 3.0) to only suppress amplitude further from the peak than the roughness taper already reaches, avoiding genuine tail/edge-of-core signal."),

    "losvd_vmin": ("Kinematic grid", "Lower bound (km/s) of the non-parametric LOSVD velocity grid."),
    "losvd_vmax": ("Kinematic grid", "Upper bound (km/s) of the non-parametric LOSVD velocity grid."),
    "n_losvd_bins": ("Kinematic grid", "Number of bins in the non-parametric LOSVD histogram. Default 89 (the legacy Fortran pipeline uses 29). A coarser grid systematically overestimates sigma and overshoots the recovered LOSVD's peak height at fixed losvd_vmin/vmax, since bin width becomes a large fraction of sigma; 89 bins collapses both biases. Pairs with xlam_auto_grid's wider upper bound, since a finer LOSVD grid needs a proportionally larger regularization strength. Set to 29 to match the legacy Fortran pipeline's bin count exactly (e.g. for cross-validation against it)."),

    # ── Continuum mode ───────────────────────────────────────────────────
    "fit_continuum": ("Continuum mode", "False: input is already continuum-normalised. True: co-fit a P-spline continuum baseline with the LOSVD (see kinextract.joint)."),

    # ── Joint continuum-in-the-model options (kinextract.joint) ──────────
    "joint_n_interior_knots": ("Joint continuum", "Number of interior knots for the continuum's P-spline basis (n_coef = joint_n_interior_knots + joint_degree + 1)."),
    "joint_degree": ("Joint continuum", "B-spline polynomial degree per segment for the continuum P-spline basis (3 = cubic)."),
    "joint_xlam_cont": ("Joint continuum", "P-spline continuum roughness-penalty weight, applied to coefficients normalized by their own initial-guess scale."),
    "joint_cont_diff_order": ("Joint continuum", "Discrete difference order for the continuum P-spline roughness penalty (2 = penalize curvature)."),
    "joint_n_sigl0_iter": ("Joint continuum", "Number of fit-then-update rounds in the self-consistent sigl0 fixed-point iteration. Shared by both the joint path (kinextract.joint.fit_joint_auto_xlam_sigl0) and the shipped MAP path (kinextract.fitting._fit_map_sigl0_recenter)."),
    "joint_sigl0_tol": ("Joint continuum", "Stop the sigl0 fixed-point iteration early once consecutive sigl0 values agree within this tolerance (km/s). Shared by both the joint and shipped MAP paths."),
    "joint_recenter_v": ("Joint continuum", "If True, recenter the wing-taper's v_center on a cross-correlation velocity estimate before fitting. Shared by both the joint and shipped MAP paths."),
    "joint_prenorm": ("Joint continuum", "If True, use the joint engine's parameterization (P-spline continuum machinery present but fixed at 1.0) in pre-normalized mode (fit_continuum=False) instead of the shipped MAP path's own parameterization. Both paths now do v_center recentering and sigl0 fixed-point iteration by default; this flag only changes which model/optimizer is used, not whether those corrections happen. Opt-in: costs up to n_sigl0_iter * len(xlam_auto_grid) full optimizations per fit either way."),

    # ── Pre-normalised mode ───────────────────────────────────────────────
    "norm_error_mode": ("Pre-normalised mode", "'unit': use uniform errors on normalised flux. 'file': use the .norm file's own error column."),
    "fortran_rebin_norm": ("Pre-normalised mode", "Rebin a .norm file's wavelength grid onto an exact step grid, matching legacy Fortran behavior."),
    "norm_apply_fortran_flux_mask": ("Pre-normalised mode", "Apply the legacy [norm_fortran_flux_min, norm_fortran_flux_max] sanity clip to normalised flux."),
    "norm_fortran_flux_min": ("Pre-normalised mode", "Lower bound of the legacy normalised-flux sanity clip."),
    "norm_fortran_flux_max": ("Pre-normalised mode", "Upper bound of the legacy normalised-flux sanity clip."),
    "spec_col3_is_variance": ("Pre-normalised mode", "Whether column 3 of a raw .spec file holds variance (True) rather than sigma (False)."),
    "use_spectrum_errors": ("Pre-normalised mode", "If False, replace per-pixel errors with their median (equal weighting) while still reading the error column."),

    # ── Emission/absorption line masking ──────────────────────────────────
    "mask_emission_line_velocity_pad_kms": ("Emission masking", "Velocity padding (km/s) widening the fixed-Angstrom kinematic-fit emission-line pre-mask half-width for broad lines (see emission_half_width_A)."),
    "extra_absorption_lines": ("Emission masking", "((center_A, half_width_A[, name]), ...) rest-frame lines applied to kinematic clean-protection."),

    # ── LOSVD / template options ──────────────────────────────────────────
    "icoff": ("LOSVD/template", "Continuum-offset mode: 0 = fixed (coff, coff2), 1 = both float, 2 = coff fixed & coff2 floats."),
    "coff": ("LOSVD/template", "Additive continuum offset term (equivalent-width-like); meaning depends on icoff."),
    "coff2": ("LOSVD/template", "Second continuum offset term; meaning depends on icoff."),
    "fortran_nlosvd_full_x": ("LOSVD/template", "Match the legacy Fortran convolution-grid sizing convention exactly."),
    "fortran_template_mixture": ("LOSVD/template", "Use the legacy Fortran template-mixture formula (sum of weighted templates over total weight)."),
    "fortran_mask_template_outside": ("LOSVD/template", "Exclude pixels outside a template's wavelength coverage from that template's contribution."),
    "fit_global_amp": ("LOSVD/template", "Fit an overall multiplicative amplitude on the model spectrum. Not compatible with fit_continuum=True."),
    "continuum_poly_mode": ("LOSVD/template", "'none', 'additive', or 'multiplicative' low-order polynomial continuum correction (pre-normalised mode only)."),
    "continuum_poly_bound": ("LOSVD/template", "Bound on the fitted continuum_poly_mode coefficient."),
    "template_w_bounds": ("LOSVD/template", "Per-element template-weight bounds. None = ordinary non-negative (1e-5, 1.0); set e.g. (-1.0, 1.0) when fitting with reduce_templates_svd's eigen-templates, which need mixed-sign weights."),

    # ── Instrumental LSF matching ─────────────────────────────────────────
    "data_fwhm_A": ("Instrumental LSF", "Instrumental line-spread-function FWHM (A) of the galaxy spectrum. Must be set together with template_fwhm_A to enable LSF matching; leave both None (default) to assume the two are already matched. See the class docstring's 'Known limitations' section for validated recovery behavior when the true sigma is comparable to this LSF."),
    "template_fwhm_A": ("Instrumental LSF", "Instrumental line-spread-function FWHM (A) of the stellar template library, in the template's native (rest-frame) wavelength units. Must be set together with data_fwhm_A."),
    "data_fwhm_frame": ("Instrumental LSF", "'observed' (default) or 'rest': frame data_fwhm_A is quoted in. 'observed' is divided by (1+zgal) before comparing to template_fwhm_A, since LSF is a property of the instrument at observed wavelengths."),

    # ── Optimizer settings ────────────────────────────────────────────────
    "map_maxiter": ("Optimizer", "Max L-BFGS-B iterations for the MAP fit."),
    "map_maxfun": ("Optimizer", "Max objective-function evaluations for the MAP fit."),
    "map_ftol": ("Optimizer", "L-BFGS-B relative function-value convergence tolerance. At stricter values, fits landing on a large xlam (as the default 89-bin LOSVD grid often needs) can spuriously report success=False: floating-point noise in the objective at that scale prevents the relative-reduction check from cleanly triggering, even though the recovered V/sigma are correct. The default balances robust convergence reporting against precision."),
    "map_gtol": ("Optimizer", "L-BFGS-B gradient-norm convergence tolerance. See map_ftol for the validation behind the default."),
    "map_maxls": ("Optimizer", "Max line-search steps per L-BFGS-B iteration."),
    "use_scaled_optimizer": ("Optimizer", "Rescale parameters to comparable magnitudes before optimizing, improving L-BFGS-B conditioning."),
    "use_jax_objective": ("Optimizer", "Use a JAX-jitted analytic value-and-gradient instead of Numba/finite differences. Default True: exact gradients are strictly more accurate and, for realistic template counts, much faster (finite differences need ~n_params extra evaluations per gradient). Falls back to finite differences automatically if JAX is unavailable."),
    "jax_enable_x64": ("Optimizer", "Enable float64 precision in JAX (recommended; JAX defaults to float32)."),
    "print_every": ("Optimizer", "Log an objective-value progress line every N optimizer evaluations (0 disables)."),

    # ── Bayesian sampling (NUTS) ───────────────────────────────────────────
    # The defaults (50 warmup + 75 samples, 4 chains) are validated to
    # converge reliably (max R-hat < 1.1, negligible divergences) on typical
    # problem sizes in well under a minute. result["result"].success reports
    # whether the reliability gates actually passed for a given fit -- if
    # False, increase nuts_num_warmup/nuts_num_samples and refit rather than
    # trusting the point estimate. For an even faster preliminary look
    # (checking a wavelength window/template list/xlam choice before
    # committing to a full run), nuts_num_chains=2 is a legitimate, still-
    # checked option (R-hat remains computable, just a somewhat weaker
    # diagnostic with fewer chains) -- it is the same estimator (posterior
    # mean) at any of these settings, just noisier with less sampling, never
    # a different, systematically biased one.
    "nuts_num_warmup": ("Bayesian sampling", "NUTS warmup/adaptation steps per chain (step size and mass matrix tuning). Increase if result['result'].success is False."),
    "nuts_num_samples": ("Bayesian sampling", "NUTS post-warmup posterior samples per chain. Total posterior draws = nuts_num_samples * nuts_num_chains. Increase if result['result'].success is False."),
    "nuts_num_chains": ("Bayesian sampling", "Number of independent NUTS chains, run as one batched/vmap-ed computation (chain_method='vectorized'). Needed (>1) for R-hat convergence diagnostics; 2 is a legitimate faster option for a preliminary look, 4 (default) gives a stronger diagnostic."),
    "nuts_seed": ("Bayesian sampling", "PRNG seed for NUTS sampling."),

    # ── Kinematic-fit cleaning ────────────────────────────────────────────
    "clean": ("Kinematic cleaning", "Enable iterative sigma-clipping of outlier pixels from the LOSVD/template chi-squared fit itself."),
    "clean_sigma": ("Kinematic cleaning", "Sigma-clip threshold for kinematic-fit cleaning."),
    "clean_maxiter": ("Kinematic cleaning", "Max iterations of the kinematic-fit clean loop."),
    "clean_minpix": ("Kinematic cleaning", "Stop rejecting pixels once fewer than this many good pixels would remain."),
    "clean_bloom_pixels": ("Kinematic cleaning", "Grow each newly rejected pixel by this many samples on each side before the next clean iteration."),
    "clean_protect_ca_triplet": ("Kinematic cleaning", "Exempt the Ca II triplet windows from kinematic-fit clipping (see clean_ca_half_width)."),
    "clean_ca_centers": ("Kinematic cleaning", "Rest-frame Ca II triplet centers (A) used for the clean-protect windows."),
    "clean_ca_half_width": ("Kinematic cleaning", "Minimum Ca II protect half-width (A); widened automatically for broad LOSVDs, see clean_protect_velocity_pad_kms."),
    "clean_protect_velocity_pad_kms": ("Kinematic cleaning", "Velocity padding (km/s) that widens clean_ca_half_width so broad LOSVD wings stay inside the protected window."),
    "clean_protect_absorption_only": ("Kinematic cleaning", "If True, protect windows only shield resid<0 pixels (a no-op, since those are never clipped anyway). Default False fully protects them; kept for backward compatibility."),
    "clean_protect_ca_frame": ("Kinematic cleaning", "'rest' or 'observed': frame of clean_ca_centers."),
    "clean_protect_windows": ("Kinematic cleaning", "Extra ((lo, hi), ...) wavelength windows exempt from kinematic-fit clipping."),
    "clean_protect_windows_frame": ("Kinematic cleaning", "'rest' or 'observed': frame of clean_protect_windows."),

    # ── Emission line pre-masking ──────────────────────────────────────────
    "mask_emission_lines_in_fit": ("Emission masking", "Pre-mask known emission lines (set gerr=BIG) before any fitting, since templates model absorption only."),
    "mask_emission_line_half_width_A": ("Emission masking", "Half-width (A) of each pre-masked emission-line window."),
    "mask_emission_line_snr_threshold": ("Emission masking", "Excess-S/N threshold above local flanking continuum required to pre-mask a candidate emission line."),
    "mask_emission_line_snr_context_A": ("Emission masking", "Flanking window (A) used to estimate local continuum/noise for emission-line S/N."),
    "mask_emission_grow_A": ("Emission masking", "Grow the combined emission mask by this many Angstrom on each side after detection."),
    "segment_emission_mask": ("Emission masking", "Enable segment-wise rolling-median detection of emission features absent from the known line table."),
    "segment_emission_n_sigma": ("Emission masking", "Rejection threshold (in rolling MAD) for segment-wise emission detection."),
    "segment_emission_win_A": ("Emission masking", "Rolling-window half-width (A) for segment-wise emission detection."),
    "mask_paschen_lines_in_fit": ("Emission masking", "Treat H I Paschen lines (which overlap the Ca II triplet) as possible emission contaminants."),
}


def _print_field_table(
    target_fields, pattern: Optional[str], value_of,
) -> None:
    """Group ``target_fields`` by their _FIELD_HELP section and print them."""
    by_section: dict[str, list[tuple]] = {}
    for f in target_fields:
        section, help_text = _FIELD_HELP.get(f.name, ("Other", ""))
        if pattern is not None:
            needle = pattern.lower()
            haystack = " ".join([f.name, section, help_text]).lower()
            if needle not in haystack:
                continue
        by_section.setdefault(section, []).append((f.name, value_of(f), help_text))

    if not by_section:
        print(f"No FitConfig fields match {pattern!r}")
        return

    for section in sorted(by_section):
        print(f"\n=== {section} ===")
        for name, value, help_text in by_section[section]:
            print(f"  {name:<34s} = {value!r}")
            if help_text:
                print(f"      {help_text}")


class _HybridDescribe:
    """Descriptor giving FitConfig.describe(...) and cfg.describe(...) distinct behavior.

    A plain ``classmethod`` cannot tell these two call forms apart (it
    always receives the class, never the instance), and a plain instance
    method cannot be called on the class with no instance at all. This
    descriptor dispatches to the right behavior: printing dataclass
    defaults for ``FitConfig.describe()``, and printing the instance's
    actual current values for ``cfg.describe()``.
    """

    def __get__(self, instance, owner):
        def describe(pattern: Optional[str] = None) -> None:
            """See FitConfig class docstring / describe() usage examples."""
            if instance is None:
                def value_of(f):
                    return f.default if f.default is not MISSING else "(no default)"
            else:
                def value_of(f):
                    return getattr(instance, f.name)
            _print_field_table(fields(owner), pattern, value_of)

        describe.__doc__ = (
            "Print every tunable field, grouped by subsystem, with a one-line "
            "description.\n\n"
            "Parameters\n----------\npattern : str, optional\n"
            "    Case-insensitive substring filter matched against the field "
            "name,\n    its section name, or its description, e.g. "
            "FitConfig.describe('joint').\n\n"
            "Notes\n-----\nCalled on the class (FitConfig.describe()), shows "
            "dataclass defaults.\nCalled on an instance (cfg.describe()), shows "
            "that instance's actual\ncurrent values instead."
        )
        return describe


@dataclass
class FitConfig:
    """
    All tunable parameters controlling a single LOSVD spectral fit.
    Run FitConfig.describe() to see a grouped, filterable table of all
    fields and their one-line descriptions.

    ``FitConfig`` is the single configuration object passed to
    :func:`~kinextract.fitting.run_spectral_fit` and
    :class:`~kinextract.errors.LOSVDErrorEstimator`. Fields are grouped
    below by subsystem; call :meth:`describe` at any time (on the class or
    an instance) to print this same grouping resolved against actual
    values, optionally filtered to a subsystem, e.g. ``cfg.describe("joint")``.

    Field groups
    ------------
    Paths
        Spectrum/template/output file locations.
    Wavelength/redshift
        How raw pixel indices or wavelength columns map to rest-frame A.
    Kinematic grid
        The non-parametric LOSVD velocity grid (``losvd_vmin``/``vmax``,
        ``n_losvd_bins``) and the fixed regularization strength ``xlam``.
    Auto xlam selection
        Grid-search selection of ``xlam`` instead of a fixed value.
    Continuum mode
        Pre-normalised input vs. co-fit P-spline continuum baseline (see
        :mod:`kinextract.joint`).
    Joint continuum, Pre-normalised mode
        Options specific to each continuum-handling path.
    LOSVD/template
        Template-mixture and continuum-offset conventions inherited from
        the legacy Fortran objective function.
    Optimizer
        L-BFGS-B iteration/tolerance/JAX settings for the reported fit
        itself, and internally for the ``xlam`` search and ``clean``-mode
        outlier rejection.
    Bayesian sampling
        NUTS/HMC posterior-sampling settings, used only by the optional,
        non-default full-posterior path
        (:func:`kinextract.bayesian.fit_state_bayesian`, called directly
        rather than through :func:`~kinextract.fitting.run_spectral_fit`).
    Kinematic cleaning
        Iterative sigma-clipping of outliers from the LOSVD chi-squared
        fit itself.
    Emission masking
        Pre-fit exclusion of known and unknown emission-line pixels.

    Performance
    ---------------------------------------------------------------------
    The reported fit is a bound-constrained L-BFGS-B optimum (MAP point
    estimate) of ``chi2 + wing-tapered smoothness penalty + LOSVD
    normalization penalty``, the same objective minimised by the original
    Fortran implementation this package is a port of -- fast (typically a
    few seconds for a single-template mock or moderate real spectrum,
    including the ``xlam`` auto-selection grid search). A full-posterior
    (NUTS/HMC) alternative is available via
    :func:`kinextract.bayesian.fit_state_bayesian` for users who want a
    sampled posterior instead of a point estimate (see that function's
    module docstring); it is not the default because a comprehensive
    recovery-accuracy comparison (LOSVD-shape L1 distance to a known
    truth, across MUSE/STIS at and away from their resolution limits, for
    both Gaussian and moderately non-Gaussian LOSVDs) found the MAP point
    estimate matches or exceeds the posterior mean's shape recovery in
    every tested condition, at a small fraction of the runtime (seconds
    vs. tens of seconds to minutes for NUTS). Always check
    ``result["result"].success`` (the L-BFGS-B optimizer's own convergence
    flag) -- if ``False``, inspect ``result["result"].message`` and
    consider raising ``map_maxiter``/loosening ``map_ftol`` rather than
    trusting the point estimate.

    Known limitations: recovery near the instrumental resolution limit
    ---------------------------------------------------------------------
    Validated with a synthetic multi-instrument sweep (``benchmarks/``;
    MUSE and STIS setups, both at and away from each instrument's own
    resolution limit, for Gaussian and moderately non-Gaussian
    Gauss-Hermite LOSVD truths, 6 seeds per condition): once a real
    instrumental line-spread function is modeled and matched via
    ``data_fwhm_A``/``template_fwhm_A`` (rather than assumed
    already-matched), recovery degrades *gracefully* near the resolution
    limit rather than failing outright, but a real, condition-dependent
    bias remains -- there is no universal correction, which is exactly
    why :func:`~kinextract.validation.assess_recovery_bias` exists (see
    below).

    - V-bias is real but not one-signed or uniform across
      instruments/resolution regimes/LOSVD shapes: in a 216-fit
      validation sweep (2 instruments x 3 resolution regimes x 3 LOSVD
      shapes x 2 S/N levels x 6 seeds), mean bias_V per condition ranged
      from about -5.2 km/s (MUSE, 1.5x LSF sigma, Gaussian truth) to
      about +8.2 km/s (MUSE, 4x LSF sigma -- i.e. well-resolved --
      strongly non-Gaussian truth), with scatter of a similar magnitude.
      There is no single "expect X km/s bias" number that holds across
      setups, and bias does not simply vanish away from the resolution
      limit for strongly asymmetric LOSVDs (see the `xlam_auto` note
      below).
    - Sigma is more consistently **overestimated**, sometimes
      substantially: mean bias_sigma in the same sweep ranged from about
      -5.2 km/s (MUSE, well-resolved, Gaussian truth) up to about +13.3
      km/s (STIS, 1.5x LSF sigma, strongly non-Gaussian truth) --
      consistent with the well-known effect of noise amplification when
      deconvolving a velocity dispersion at or below an instrument's own
      LSF, not a bug or a tunable knob.
    - The MAP objective's wing-tapered smoothness penalty is always
      zero-centered (matching the original Fortran convention), rather
      than recentered on a data-driven velocity estimate for each fit.
      Recentering can amplify bias whenever the velocity estimate itself
      is imprecise -- a few to several km/s off, which is routine for a
      coarse cross-correlation -- so a fixed zero point is the more
      robust default across a broad range of targets. Use
      :func:`~kinextract.validation.assess_recovery_bias` (below) to
      check whether a specific target near its instrument's resolution
      limit would benefit from a more tailored treatment.
    - :func:`~kinextract.errors.bias_corrected_losvd` (analytic
      hat-matrix linearization) is a separate, older correction attempt
      that amplifies noise catastrophically (h3/h4 pinned at their fit
      bounds) when the LOSVD is this weakly identified. **Do not** use it
      as a resolution-limit correction.
    - This is distinct from -- and should not be confused with -- trying
      to recover a velocity dispersion *below* the detector's native
      pixel sampling with no LSF modeled at all (e.g. sub-pixel
      broadening): that regime showed much larger, less predictable
      bias in testing and is closer to an information-theoretic limit
      than a resolution-limit one.
    - **Recommended practical guidance**: since the direction and size of
      the bias is condition-dependent, don't rely on the generic numbers
      above for a specific target -- use
      :func:`~kinextract.validation.assess_recovery_bias` to measure the
      empirical bias directly, on mock spectra matched to that target's
      own instrument resolution, template mixture, continuum, and noise
      level (built automatically from a completed fit via
      :func:`~kinextract.mocks.build_matched_mock`), then optionally apply
      :func:`~kinextract.validation.correct_recovered_losvd` to correct
      for it. This is only worth the extra runtime (one MAP refit per
      mock seed) when the target's sigma is within about 2x the
      instrument's LSF sigma; well away from the resolution limit the
      bias shrinks toward zero.
    - ``xlam_auto`` (the default regularization-strength selector) can
      occasionally over-smooth strongly asymmetric (large ``h3``/``h4``)
      LOSVDs for specific noise realizations, biasing V by 15-30 km/s in
      roughly a quarter of tested cases for a strongly non-Gaussian truth
      -- tightening ``xlam_chi2_tolerance`` reduces this but was found to
      regress the common near-Gaussian case, so the default is
      unchanged. For targets suspected to have strongly asymmetric
      LOSVDs, cross-check with :func:`~kinextract.validation.assess_recovery_bias`
      rather than assuming the auto-selected ``xlam`` is well-calibrated.
      Insufficient regularization more generally (a fixed, low ``xlam``
      chosen by hand rather than ``xlam_auto``) reproduces the same
      failure mode and is not specific to strongly asymmetric truths --
      it is the standard bias-variance tradeoff of any penalized
      non-parametric fit, present in the original Fortran implementation
      too for an equally under-regularized choice, not a new behavior.
    - Fit cost scales strongly with template count, independent of
      pixel count -- e.g. 5 templates cost roughly 4-7x a single
      template, and a 35-template library roughly 12x (at fixed
      ``xlam``), mostly from ``xlam_auto``'s internal grid search
      needing a fresh JAX kernel compile per candidate value. This
      compounds multiplicatively with
      :meth:`~kinextract.errors.LOSVDErrorEstimator.residual_bootstrap`'s
      per-replicate cost. For large template libraries, consider a
      smaller, pre-selected subset, and set ``xlam_auto=False`` (with a
      pre-chosen ``xlam``) for repeated refits (bootstrap and
      :func:`~kinextract.validation.assess_recovery_bias` already do this
      internally where appropriate).

    Known limitations: joint-mode (``cfg.fit_continuum=True``) velocity bias
    ---------------------------------------------------------------------
    Validated with a synthetic E-MILES sweep (2-SSP and broad, ~20-SSP
    mixtures; ``n_losvd_bins=89`` -- the current default, see that field's
    own docstring for its recalibration history -- a 1000A fit window;
    sigma=30-350 km/s, 3-5 seeds per condition):

    - Sigma recovery is good across the full tested range (roughly
      +-1-3 km/s bias for sigma <= 200 km/s, growing to a still-modest
      +-1-4 km/s even at sigma=250-350).
    - V recovery carries two distinct biases. A small, roughly
      shrinkage-toward-zero offset (a few km/s, growing with the true
      velocity's distance from the LOSVD grid's own center at v=0) is
      present even at low sigma and is **not yet understood** -- ruled out
      as the cause: `xlam_criterion` choice, wing-taper `v_center`
      recentering (including recentering on the *exact* true velocity, not
      just a cross-correlation estimate), fitting-template count, and
      generating-population complexity (a broad, ~20-SSP mixture shows the
      same pattern as a simple 2-SSP one). Separately, a larger bias
      (growing to roughly -10 to -14 km/s by sigma=350) appears at broad
      sigma; this component also appears, at comparable magnitude, in a
      parametric (Gauss-Hermite) fit of the same mock, consistent with the
      well-documented V/sigma/h3 covariance at low per-resolution-element
      S/N (van der Marel & Franx 1993; Cappellari 2017) rather than being
      specific to this package's non-parametric LOSVD parameterization.
    - :func:`~kinextract.validation.assess_recovery_bias` and
      :func:`~kinextract.validation.correct_recovered_losvd` currently
      refuse joint-mode fits entirely (see :func:`~kinextract.validation.assess_recovery_bias`'s
      own guard) -- there is no automated bias-correction path for this
      mode yet. Until the small offset above is root-caused, treat
      recovered V at low-moderate sigma as accurate to a few km/s, and rely
      on :meth:`~kinextract.errors.LOSVDErrorEstimator.residual_bootstrap`'s
      error bars (not the point estimate) at sigma >~ 200 km/s.

    Examples
    --------
    >>> from kinextract import FitConfig
    >>> cfg = FitConfig(template_list_file="Tlist", zgal=0.0016,
    ...                  wavefitmin=8400.0, wavefitmax=8800.0,
    ...                  fit_continuum=True, clean=True)
    >>> cfg.describe("clean")  # doctest: +SKIP
    """
    # ── Paths ────────────────────────────────────────────────────────────────
    # gal_file is optional for command-line mode, but useful in notebooks.
    gal_file: str = ""
    template_list_file: str = "Tlist"
    template_dir: Optional[str] = None
    template_npz_file: Optional[str] = None
    template_npz_select: "Optional[list[tuple[float, float]]]" = None
    regions_bad_path: str = "regions.bad"
    outdir: Optional[str] = "."
    write_outputs: bool = False

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

    # ── Auto smoothing parameter selection ─────────────────────────────────
    # If True, search xlam_auto_grid for the best xlam according to
    # xlam_criterion.  The selected value is stored in both st.xlam and
    # cfg.xlam so all subsequent MAP fits and bootstrap refits use it.
    # Bootstrap workers always set xlam_auto=False (via _make_frozen_cfg)
    # so the search runs only once per spectrum.
    xlam_auto: bool = False
    # Upper bound spans 100-1e7: wide enough for the default 89-bin LOSVD
    # grid to find its natural regularization strength directly, without
    # relying on the discrepancy search's bracket-expansion fallback (a
    # finer LOSVD grid needs proportionally more regularization for the
    # same effective smoothness).
    xlam_auto_grid: tuple = (100., 1000., 10000., 100000., 1_000_000., 10_000_000.)

    # How to select xlam from the grid:
    #
    #   "discrepancy" (default) — Cappellari/pPXF's regularization-strength
    #       convention (Cappellari 2017, MNRAS 466, 798, Sec. 3.5; Cappellari
    #       & Emsellem 2004, PASP 116, 138): a 1-D search (not a grid scan)
    #       that increases xlam until chi2 rises by xlam_discrepancy_nsigma *
    #       sqrt(2 * ngood) above the chi2 at (near-)zero regularization --
    #       a target tied to the *known noise level*, not to the shape of
    #       the chi2(xlam) curve, so it does not need a sharp elbow to find
    #       a well-defined answer.  Only ``xlam_auto_grid``'s min/max are
    #       used, as the search's initial bracket (expanded automatically if
    #       needed).  See :func:`kinextract.fitting._discrepancy_principle_search`.
    #       An empirical calibration sweep (benchmarks/calibrate_xlam.py, 250
    #       mock fits across a sigma/snr/base scenario grid) found this
    #       criterion's default nsigma=0.3 beats "chi2" below by ~25-30% on
    #       RMS V/sigma recovery bias, and fixes a real near-resolution-limit
    #       failure mode "chi2" has (badly non-converged fits, chi2_red~2.7,
    #       at sigma_true near the instrumental resolution).
    #
    #   "chi2" — runs all grid fits, then picks the *largest* xlam
    #       (most regularised) whose chi2_red is within xlam_chi2_tolerance of
    #       the grid minimum.  Scale-invariant: works for any sigma without
    #       per-galaxy tuning.  Roughness is still logged at every grid point
    #       as a diagnostic.  Known weakness: on a spectrum whose chi2(xlam)
    #       curve is nearly flat over many orders of magnitude (e.g. a clean
    #       mock, or real data once the error scale is properly calibrated),
    #       this rule's answer is essentially arbitrary and highly sensitive
    #       to exactly which grid points were tried -- see "discrepancy"
    #       above, which was adopted as the default specifically to address
    #       this failure mode.  Kept available for backward compatibility.
    #
    #   "roughness" — original criterion: picks the *smallest* xlam whose
    #       LOSVD roughness falls at or below xlam_smooth_threshold.  Poorly
    #       calibrated for broad LOSVDs (large sigma); kept for backward compat.
    xlam_criterion: str = "discrepancy"
    # Maximum fractional increase in chi2_red relative to the grid minimum
    # that is still considered acceptable.  0.02 means "chi2 may increase by
    # at most 2%".  Only used when xlam_criterion="chi2".
    xlam_chi2_tolerance: float = 0.02

    # Roughness threshold (only used when xlam_criterion="roughness").
    xlam_smooth_threshold: float = 0.25

    # Multiplier on the discrepancy principle's sqrt(2*ngood) chi2-rise
    # target (only used when xlam_criterion="discrepancy"). Cappellari's own
    # stated pPXF convention uses 1.0 (one "sigma" of the noise's own chi2
    # fluctuation), but an empirical calibration sweep
    # (benchmarks/calibrate_xlam.py, 250 mock fits across the sigma/snr/base
    # scenario grid, 5 noise replicates each) found nsigma=1.0 gives far too
    # much regularization for kinextract's direct LOSVD-histogram
    # parameterization (as opposed to pPXF's many-template-weight
    # parameterization, where the convention originates) -- nsigma=0.3 gave
    # the best overall V/sigma bias & RMS across the grid, and also fixes a
    # real near-resolution-limit failure mode (chi2_red~2.7, V bias~-36 km/s
    # at sigma_true~15 km/s) that the "chi2" grid-tolerance criterion has.
    xlam_discrepancy_nsigma: float = 0.3

    # Unimodality constraint applied under both criteria: LOSVD must have at
    # most xlam_max_peaks prominent peaks (prominence >= xlam_peak_min_prominence
    # x global peak).  Multi-peaked solutions at low xlam are fitting artifacts.
    xlam_max_peaks: int = 1
    xlam_peak_min_prominence: float = 0.1
    # Max iterations for each search fit (None -> use map_maxiter).
    # The search only needs an approximate LOSVD shape, so a smaller budget
    # (e.g., 2000) speeds up the search without affecting the final result.
    xlam_auto_maxiter: Optional[int] = None

    # Opt-in L2 amplitude penalty on the LOSVD, active only in the wing
    # region (see kinextract.numerics._compute_wing_shrinkage). Off by
    # default: the roughness penalty above is wing-tapered in *strength*
    # but still only penalizes curvature, so a flat, non-zero pedestal far
    # from the peak -- which has near-zero curvature -- is nearly free for
    # the optimizer to leave in place. This term adds a direct cost for
    # that, but it cannot distinguish genuine non-Gaussian wing/tail
    # structure (a skewed or double-peaked LOSVD) from a noise pedestal, so
    # it suppresses both; combined with xlam_auto=True it can also
    # destabilize the auto-xlam search and understate bootstrap
    # uncertainty. Only enable it with independent reason to expect a
    # compact true LOSVD, and always check bootstrap error bars (not just
    # the point estimate) afterward. See _FIELD_HELP for more detail.
    xlam_wing_shrink: float = 0.0
    # Onset of the xlam_wing_shrink taper, in units of sigl0. Default 1.8
    # matches the roughness penalty's own taper onset. A larger value
    # (e.g. 3.0) suppresses amplitude only further from the peak than the
    # roughness taper already reaches, decoupled from the roughness
    # penalty's own onset.
    xlam_wing_shrink_sfac: float = 1.8

    losvd_vmin: Optional[float] = None
    losvd_vmax: Optional[float] = None

    # Number of non-parametric LOSVD bins. The legacy Fortran pipeline uses
    # 29; a coarser grid like that systematically overestimates sigma and
    # overshoots the LOSVD peak height at fixed losvd_vmin/vmax, since bin
    # width becomes a large fraction of sigma. 89 bins collapses that bias
    # (validated across a sigma=30-350 sweep and the package's own
    # regression-guardrail tests, tests/test_regression_convergence.py),
    # paired with xlam_auto_grid's wider upper bound above since a finer
    # LOSVD grid needs proportionally more regularization.
    n_losvd_bins: int = 29

    # ── Continuum mode ──────────────────────────────────────────────────────
    # False = pre-normalised mode
    # True  = co-fit a P-spline continuum baseline with the LOSVD, folded
    #         directly into the same L-BFGS-B optimization as the LOSVD and
    #         template weights (see kinextract.joint), rather than a
    #         separate continuum sub-fit with its own hyperparameter search.
    fit_continuum: bool = False

    # ── Joint continuum-in-the-model options (kinextract.joint) ────────────
    # See kinextract.joint.fit_joint_auto_xlam_sigl0/fit_joint_auto_xlam for
    # the full rationale behind each of these. Shared concepts (xlam grid
    # search, its chi2 tolerance/peak constraints, the initial sigma guess,
    # optimizer budgets) reuse the existing xlam_auto_grid/xlam_chi2_tolerance/
    # xlam_max_peaks/xlam_peak_min_prominence/sigl/map_maxiter/map_ftol/
    # map_maxfun/use_jax_objective fields above rather than duplicating them;
    # only genuinely new concepts get a joint_-prefixed field here.
    joint_n_interior_knots: int = 10
    joint_degree: int = 3
    joint_xlam_cont: float = 3.0
    joint_cont_diff_order: int = 2
    joint_n_sigl0_iter: int = 3
    joint_sigl0_tol: float = 2.0
    joint_recenter_v: bool = True

    # If True, use the joint fitting engine's own parameterization/optimizer
    # (P-spline continuum machinery present but fixed at 1.0) in
    # pre-normalized mode (fit_continuum=False), instead of the shipped MAP
    # path's own parameterization (icoff/coff/coff2 continuum-offset terms,
    # its own template-mixture handling). Both paths now do v_center
    # recentering and sigl0 fixed-point convergence by default (see
    # kinextract.fitting._fit_map_sigl0_recenter /
    # kinextract.joint.fit_joint_auto_xlam_sigl0) and cost the same
    # n_sigl0_iter * len(xlam_auto_grid) full optimizations per fit either
    # way -- this flag no longer trades off cost, only which model/optimizer
    # is used. Default False: the shipped path's own parameterization.
    joint_prenorm: bool = False

    # ── Pre-normalised mode options ─────────────────────────────────────────
    norm_error_mode: str = "unit"  # "unit" or "file"
    fortran_rebin_norm: bool = True
    norm_apply_fortran_flux_mask: bool = True
    norm_fortran_flux_min: float = 0.0
    norm_fortran_flux_max: float = 1.3

    # Whether column 3 of a raw .spec file is variance or sigma.
    spec_col3_is_variance: bool = False

    # When False, ignore the per-pixel input errors and use the median error
    # value uniformly across all pixels (equal weighting).  The error column
    # is still read; only the weights used in fitting are replaced.
    use_spectrum_errors: bool = True

    # ── Emission/absorption line masking ────────────────────────────────────
    mask_emission_line_velocity_pad_kms: float = 300.0

    # Convenience field: extra absorption lines applied to kinematic
    # cleaning protection.
    # Format: ((center_A, half_width_A), ...) or ((center_A, half_width_A, "name"), ...)
    # All wavelengths are rest-frame Angstroms.
    extra_absorption_lines: tuple = ()

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
    # Per-element bounds for template-mixture weights. None (default) keeps the
    # ordinary non-negative convention ((1e-5, 1.0)) appropriate for ordinary,
    # physical per-star templates. Set to something like (-1.0, 1.0) when
    # fitting with kinextract.templates.reduce_templates_svd's eigen-templates,
    # which are not physical flux spectra individually and generically need
    # mixed-sign weights to reconstruct a real template mixture -- see that
    # function's docstring.
    template_w_bounds: Optional[tuple] = None

    # ── Instrumental LSF matching ────────────────────────────────────────────
    # Both fields must be set together to enable LSF matching; the sharper
    # (narrower-FWHM) side is convolved down to match the coarser one before
    # fitting, following the standard convolution-matching convention (e.g.
    # Cappellari 2017, pPXF). See kinextract.templates.resolution_mismatch_sigma_A.
    data_fwhm_A: Optional[float] = None
    template_fwhm_A: Optional[float] = None
    data_fwhm_frame: str = "observed"  # "observed" or "rest"

    # ── Optimizer settings ──────────────────────────────────────────────────
    map_maxiter: int = 10000
    map_maxfun: int = 200000
    # Very strict tolerances (e.g. ftol=2e-14/gtol=1e-10) sit far past the
    # point where the objective has stopped changing at any physically
    # meaningful level: they force many thousands of extra L-BFGS-B
    # iterations chasing floating-point noise, without changing the
    # recovered LOSVD beyond ~1% of peak amplitude, and can leave the
    # optimizer far enough from a true stationary point that the Laplace
    # Hessian (see errors.py) is not positive-definite. At the larger xlam
    # an 89-bin LOSVD often needs, tolerances much stricter than the
    # default can also make result.success spuriously report False from
    # floating-point noise in the objective, even though the recovered
    # V/sigma are correct -- see map_ftol's own _FIELD_HELP entry.
    map_ftol: float = 1e-8
    map_gtol: float = 1e-8
    map_maxls: int = 50
    use_scaled_optimizer: bool = True
    # Analytic JAX gradients instead of scipy finite differences. This is
    # strictly more accurate (no finite-difference truncation error) and,
    # for realistic template-library sizes, dramatically faster: profiling
    # shows the finite-difference fallback needs ~(n_losvd_bins + n_templates
    # + 1) extra objective evaluations per gradient step, so runtime scales
    # with template count; the JAX path needs one evaluation regardless of
    # template count. Falls back to the finite-difference path automatically
    # if JAX is not installed (see numerics.py).
    use_jax_objective: bool = True
    jax_enable_x64: bool = True
    print_every: int = 200

    # ── Bayesian sampling (NUTS) ─────────────────────────────────────────────
    # Only consumed by the optional, non-default full-posterior path
    # (kinextract.bayesian.fit_state_bayesian, called directly rather than
    # through run_spectral_fit) -- the default MAP+bootstrap pipeline never
    # reads these.
    nuts_num_warmup: int = 50
    nuts_num_samples: int = 75
    nuts_num_chains: int = 4
    nuts_seed: int = 0

    # ── Kinematic-fit sigma-clipping / cleaning ─────────────────────────────
    clean: bool = False
    clean_sigma: float = 3.0
    clean_maxiter: int = 5
    clean_minpix: int = 10
    clean_bloom_pixels: int = 0

    clean_protect_ca_triplet: bool = True
    clean_ca_centers: tuple = (8498.02, 8542.09, 8662.14)
    # Minimum protect half-width in Angstrom.  The *effective* half-width used
    # is max(clean_ca_half_width, cen_A * clean_protect_velocity_pad_kms / c),
    # so broad LOSVDs (large sigl) automatically get a wider protected window
    # that still covers the true line wings; see clean_protect_velocity_pad_kms.
    clean_ca_half_width: float = 6.0
    # Velocity padding (km/s) used to widen clean_ca_half_width for broad
    # LOSVDs. At 8500 A, 400 km/s corresponds to ~11.3 A -- comfortably beyond
    # the ~3 sigma wing of most early-type-galaxy central LOSVDs.
    clean_protect_velocity_pad_kms: float = 400.0
    # If False (default), every pixel in a protect window (Ca II triplet,
    # clean_protect_windows, extra_absorption_lines) is fully exempt from the
    # sigma-clip regardless of residual sign.  If True, protection only
    # applies to negative residuals, which the one-sided clip already never
    # rejects -- i.e. True effectively disables protection.  Kept only for
    # backward compatibility; see _update_clean_mask in masking.py.
    clean_protect_absorption_only: bool = False
    clean_protect_ca_frame: str = "rest"
    clean_protect_windows: tuple = ()
    clean_protect_windows_frame: str = "rest"

    # ── Emission line pre-masking in spectral fit ───────────────────────────
    # Stellar templates model only absorption; emission lines must be excluded
    # from chi-squared before fitting.  Pixels are set to gerr=BIG in
    # make_fit_state so they are excluded from the fit.
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
    # region (no unmasked continuum between adjacent lines to confuse
    # sigma-clipping).  Set to mask_emission_line_half_width_A or larger to
    # merge lines within ~2× half_width_A of each other.
    mask_emission_grow_A: float = 0.0

    # Segment-wise rolling-median upward-outlier detection.
    # For each CaT-free continuum segment, flags pixels above
    # local_rolling_median + n_sigma × MAD.  This catches emission wings and
    # features not in the known emission-line table that would otherwise
    # bias the MAP fit.
    segment_emission_mask: bool = True
    segment_emission_n_sigma: float = 3.0
    segment_emission_win_A: float = 50.0  # rolling window half-width in Å

    # If True, treat H I Paschen lines as possible emission contaminants in the
    # spectral fit. This is especially important for Ca II triplet fitting because
    # Pa16, Pa15, and Pa13 overlap Ca II 8498, 8542, and 8662.
    mask_paschen_lines_in_fit: bool = False

    describe = _HybridDescribe()

    def __post_init__(self):
        """Normalize aliases and validate cross-field constraints after construction.

        Runs automatically after ``FitConfig(...)``. Loads ``galaxy_params_path``
        if given, lower-cases/validates the various ``*_frame`` string fields,
        and raises ``ValueError`` for a handful of known-invalid
        combinations (e.g. ``fit_continuum`` with ``fit_global_amp``,
        or ``losvd_vmax <= losvd_vmin``).
        """
        # Read legacy galaxy.params if requested.
        self.galaxy_params = {}

        if self.galaxy_params_path is not None:
            _gp = Path(self.galaxy_params_path)
            if _gp.is_dir():
                _gp = _gp / "galaxy.params"
            if _gp.exists() and _gp.is_file():
                from .io import read_galaxy_params
                self.galaxy_params = read_galaxy_params(_gp)

            if self.use_galaxy_params_velocity_bounds:
                if "vmin" in self.galaxy_params:
                    self.losvd_vmin = float(self.galaxy_params["vmin"])

                if "vmax" in self.galaxy_params:
                    self.losvd_vmax = float(self.galaxy_params["vmax"])

        self.norm_error_mode = self.norm_error_mode.lower().strip()
        if self.norm_error_mode not in ("unit", "file"):
            raise ValueError("norm_error_mode must be 'unit' or 'file'")

        self.continuum_poly_mode = self.continuum_poly_mode.lower().strip()
        if self.continuum_poly_mode not in ("none", "additive", "multiplicative"):
            raise ValueError(
                "continuum_poly_mode must be 'none', 'additive', or 'multiplicative'"
            )

        if self.fit_continuum and self.continuum_poly_mode != "none":
            raise ValueError(
                "continuum_poly_mode is only supported when fit_continuum=False"
            )

        if self.fit_continuum and self.fit_global_amp:
            raise ValueError(
                "fit_continuum=True should not be combined with "
                "fit_global_amp=True because the co-fit continuum and global "
                "amplitude are degenerate."
            )

        # Normalize and validate frame strings.
        frame_attrs = [
            "norm_wave_frame",
            "clean_protect_ca_frame",
            "clean_protect_windows_frame",
            "data_fwhm_frame",
        ]

        for attr in frame_attrs:
            val = getattr(self, attr).lower().strip()
            setattr(self, attr, val)

        if self.norm_wave_frame not in ("auto", "rest", "observed"):
            raise ValueError("norm_wave_frame must be 'auto', 'rest', or 'observed'")

        for attr in [
            "clean_protect_ca_frame",
            "clean_protect_windows_frame",
            "data_fwhm_frame",
        ]:
            val = getattr(self, attr)
            if val not in ("rest", "observed"):
                raise ValueError(f"{attr} must be 'rest' or 'observed'")

        if (self.data_fwhm_A is None) != (self.template_fwhm_A is None):
            raise ValueError(
                "data_fwhm_A and template_fwhm_A must be set together to enable "
                "instrumental LSF matching (or leave both None to disable it); "
                f"got data_fwhm_A={self.data_fwhm_A!r}, "
                f"template_fwhm_A={self.template_fwhm_A!r}."
            )

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
    Build a :class:`FitConfig` from a TOML configuration file.

    TOML tables are optional and purely organizational: any top-level
    scalar key, or any key nested one level inside a ``[section]`` table,
    is treated as a ``FitConfig`` field. Keys that are not recognized
    ``FitConfig`` fields are dropped with a warning (logged via
    :func:`~kinextract._utils.log`), except for a small set of
    wrapper-level keys (``run_errors_default``, ``run_bootstrap_default``,
    ``n_bootstrap``, ``n_jobs``) consumed by higher-level scripts rather
    than ``FitConfig`` itself. TOML arrays become tuples, matching the
    tuple-typed ``FitConfig`` fields (e.g. ``xlam_auto_grid``,
    ``extra_absorption_lines``).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TOML config file.

    Returns
    -------
    FitConfig
        A validated configuration object (``__post_init__`` runs as usual).

    Raises
    ------
    ImportError
        If neither the standard-library ``tomllib`` (Python >= 3.11) nor
        the ``tomli`` backport is available.

    Examples
    --------
    >>> from kinextract import load_config_from_toml
    >>> cfg = load_config_from_toml("kinextract.config")  # doctest: +SKIP

    See Also
    --------
    FitConfig : The configuration object this function constructs.
    FitConfig.describe : Print all tunable fields, e.g. after loading.
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

    from ._utils import log
    known = {f.name for f in FitConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    # Wrapper-level controls that may live in kinextract.config but are not
    # FitConfig fields (consumed by the shell wrapper / losvd_errs flow).
    # snr_override (float, e.g. under [wavelength]): a driver script may use
    # this to compute a uniform gal_errors = flux / snr_override array to
    # pass to run_spectral_fit(..., gal_errors=...), instead of the file's
    # own (possibly unreliable) per-pixel error column -- see the MUSE
    # example notebook, which does this with use_spectrum_errors=False.
    allowed_wrapper_keys = {
        "run_errors_default",
        "run_bootstrap_default",
        "n_bootstrap",
        "n_jobs",
        "snr_override",
    }
    for k in list(flat.keys()):
        if k not in known:
            if k not in allowed_wrapper_keys:
                log(f"WARNING: unknown config key '{k}' ignored")
            del flat[k]

    return FitConfig(**flat)
