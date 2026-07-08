"""
Fit configuration for the kinextract spectral-fitting pipeline.

Every tunable knob in the package -- wavelength/redshift setup, the
non-parametric LOSVD grid, continuum handling (pre-normalised vs. ALS),
regularisation (``xlam``) and its automatic selection, the L-BFGS-B
optimizer, iterative sigma-clipping ("cleaning"), and emission-line
masking -- is a field on :class:`FitConfig`. There is no separate
"advanced settings" object: :func:`~kinextract.fitting.run_spectral_fit`
and :class:`~kinextract.errors.LOSVDErrorEstimator` both take a single
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
  ``FitConfig.describe("als")`` to see only ALS-continuum-related knobs.

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
    "template_list_file": ("Paths", "Path to the template-list file (one template filename per line)."),
    "template_dir": ("Paths", "Directory the template list's relative paths are resolved against (defaults to the list file's directory)."),
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
    "smoothing": ("Kinematic grid", "Alias for xlam kept for legacy-script compatibility; if set, overrides xlam in __post_init__."),

    # ── Auto smoothing (xlam) selection ──────────────────────────────────
    "xlam_auto": ("Auto xlam selection", "If True, grid-search xlam_auto_grid and pick the best value per xlam_criterion, instead of using the fixed xlam."),
    "xlam_auto_grid": ("Auto xlam selection", "Candidate xlam values searched when xlam_auto=True."),
    "xlam_criterion": ("Auto xlam selection", "'chi2' (default, scale-invariant) or 'roughness' (legacy) selection rule; see class docstring."),
    "xlam_chi2_tolerance": ("Auto xlam selection", "Max fractional chi2_red increase over the grid minimum still considered acceptable (xlam_criterion='chi2')."),
    "xlam_smooth_threshold": ("Auto xlam selection", "Roughness threshold for xlam selection (xlam_criterion='roughness' only)."),
    "xlam_max_peaks": ("Auto xlam selection", "Reject xlam grid points whose recovered LOSVD has more than this many prominent peaks."),
    "xlam_peak_min_prominence": ("Auto xlam selection", "Minimum peak prominence (fraction of the global LOSVD peak) counted by xlam_max_peaks."),
    "xlam_auto_maxiter": ("Auto xlam selection", "Optimizer iteration budget for each xlam-grid trial fit (None -> map_maxiter)."),

    "losvd_vmin": ("Kinematic grid", "Lower bound (km/s) of the non-parametric LOSVD velocity grid."),
    "losvd_vmax": ("Kinematic grid", "Upper bound (km/s) of the non-parametric LOSVD velocity grid."),
    "n_losvd_bins": ("Kinematic grid", "Number of bins in the non-parametric LOSVD histogram."),

    # ── Continuum mode ───────────────────────────────────────────────────
    "fit_als_continuum": ("Continuum mode", "False: input is already continuum-normalised. True: co-fit a full-scale continuum baseline with the LOSVD (method set by continuum_method)."),
    "continuum_method": ("Continuum mode", "'joint' (default) -- single-shot P-spline-continuum-in-the-model fit (see kinextract.joint). 'als' or 'polynomial' -- legacy separate-continuum-sub-fit alternatives, kept opt-in for users who want that approach. Only used when fit_als_continuum=True."),

    # ── Joint continuum-in-the-model options (continuum_method="joint") ──
    "joint_n_interior_knots": ("Joint continuum", "Number of interior knots for the continuum's P-spline basis (n_coef = joint_n_interior_knots + joint_degree + 1)."),
    "joint_degree": ("Joint continuum", "B-spline polynomial degree per segment for the continuum P-spline basis (3 = cubic)."),
    "joint_xlam_cont": ("Joint continuum", "P-spline continuum roughness-penalty weight, applied to coefficients normalized by their own initial-guess scale."),
    "joint_cont_diff_order": ("Joint continuum", "Discrete difference order for the continuum P-spline roughness penalty (2 = penalize curvature)."),
    "joint_n_sigl0_iter": ("Joint continuum", "Number of fit-then-update rounds in the self-consistent sigl0 fixed-point iteration (see kinextract.joint.fit_joint_auto_xlam_sigl0)."),
    "joint_sigl0_tol": ("Joint continuum", "Stop the sigl0 fixed-point iteration early once consecutive sigl0 values agree within this tolerance (km/s)."),
    "joint_recenter_v": ("Joint continuum", "If True, recenter the wing-taper's v_center on a cross-correlation velocity estimate before each xlam grid search."),
    "joint_prenorm": ("Joint continuum", "If True, use the joint engine's v_center/sigl0/xlam-selection improvements even in pre-normalized mode (fit_als_continuum=False), with the continuum fixed at 1.0. Opt-in: costs up to n_sigl0_iter * len(xlam_auto_grid) full optimizations per fit."),

    # ── Pre-normalised mode ───────────────────────────────────────────────
    "norm_error_mode": ("Pre-normalised mode", "'unit': use uniform errors on normalised flux. 'file': use the .norm file's own error column."),
    "fortran_rebin_norm": ("Pre-normalised mode", "Rebin a .norm file's wavelength grid onto an exact step grid, matching legacy Fortran behavior."),
    "norm_apply_fortran_flux_mask": ("Pre-normalised mode", "Apply the legacy [norm_fortran_flux_min, norm_fortran_flux_max] sanity clip to normalised flux."),
    "norm_fortran_flux_min": ("Pre-normalised mode", "Lower bound of the legacy normalised-flux sanity clip."),
    "norm_fortran_flux_max": ("Pre-normalised mode", "Upper bound of the legacy normalised-flux sanity clip."),

    # ── ALS continuum options ────────────────────────────────────────────
    "als_lam": ("ALS continuum", "ALS baseline smoothing strength (larger = smoother continuum). See asymmetric_least_squares_continuum."),
    "als_p": ("ALS continuum", "ALS asymmetry weight (0-1); values well below 0.5 pull the baseline toward the upper envelope of the flux."),
    "als_niter": ("ALS continuum", "Number of ALS reweighting iterations."),
    "als_outer_iter": ("ALS continuum", "Max outer-loop iterations alternating the LOSVD fit and the ALS continuum re-estimate."),
    "als_outer_tol": ("ALS continuum", "Outer-loop convergence tolerance on the median fractional continuum change."),
    "als_eps": ("ALS continuum", "Minimum ALS weight floor, avoids a singular banded system."),
    "als_optimize_init_only": ("ALS continuum", "If True (default), optimize ALS (lam, p) only once at initialization and reuse them on later outer iterations -- validated to give the same accuracy as re-optimizing every iteration at lower cost."),
    "als_clip": ("ALS continuum", "(lo, hi) clamp applied to the fitted ALS continuum values."),
    "spec_col3_is_variance": ("ALS continuum", "Whether column 3 of a raw .spec file holds variance (True) rather than sigma (False)."),
    "use_spectrum_errors": ("ALS continuum", "If False, replace per-pixel errors with their median (equal weighting) while still reading the error column."),

    # ── ALS hyperparameter optimization ──────────────────────────────────
    "als_optimize": ("ALS hyperparameter search", "If True (default), grid-search (als_lam, als_p) instead of using the fixed als_lam/als_p values. A fixed als_lam is tuned for one instrument's own pixel scale and can be badly mismatched for others (e.g. it under-fits a narrower/finer-sampled spectrograph's continuum) -- searching per-target removes that mismatch at little to no extra cost (als_optimize_init_only limits the search to once per fit)."),
    "als_lam_grid": ("ALS hyperparameter search", "Coarse als_lam grid searched when als_optimize=True."),
    "als_p_grid": ("ALS hyperparameter search", "Coarse als_p grid searched when als_optimize=True. Defaults to a single value matching als_p's own default (0.05, asymmetric -- appropriate for real absorption-dominated spectra): if this contains only one value, als_p is held fixed at it and only als_lam is searched. Widen this (e.g. to include values up to 0.5) only if you specifically want the search to also consider symmetric/near-symmetric continuum fits -- a wide grid that includes symmetric values can otherwise let the search discard the physically-motivated asymmetric default in favor of a symmetric fit."),
    "als_lam_grid_fine_n": ("ALS hyperparameter search", "Number of points in the fine als_lam refinement grid around the coarse best value."),
    "als_p_grid_fine_n": ("ALS hyperparameter search", "Number of points in the fine als_p refinement grid around the coarse best value."),
    "als_opt_verbose": ("ALS hyperparameter search", "Log every (lam, p) trial's score during the ALS hyperparameter search."),
    "als_template_selection": ("ALS hyperparameter search", "'lsq' or 'nnls' template-weight solver used while scoring ALS hyperparameter trials."),
    "als_use_bic": ("ALS hyperparameter search", "Score ALS trials with chi2 + als_bic_dof_penalty * k_eff * ln(n) instead of raw chi2, penalising continuum overfitting."),
    "als_chisq_floor": ("ALS hyperparameter search", "Hard-reject any (lam, p) trial whose chi2_red falls below this floor (implausibly good fit -> overfit continuum)."),
    "als_hutchinson_probes": ("ALS hyperparameter search", "Number of Hutchinson stochastic-trace probes used to estimate the ALS effective degrees of freedom."),
    "als_bic_dof_penalty": ("ALS hyperparameter search", "Penalty weight multiplying k_eff*ln(n) in the BIC-like ALS score."),
    "als_lambda_floor": ("ALS hyperparameter search", "Optional minimum als_lam to prevent pathological low-lambda ALS fits."),

    # ── Polynomial continuum ──────────────────────────────────────────────
    "poly_continuum_order": ("Polynomial continuum", "Polynomial order for continuum_method='polynomial' (ignored if poly_continuum_optimize=True)."),
    "poly_continuum_p": ("Polynomial continuum", "Asymmetry weight for the polynomial's IRLS reweighting -- same convention/role as als_p."),
    "poly_continuum_niter": ("Polynomial continuum", "Number of asymmetric-reweighting iterations for the polynomial fit."),
    "poly_continuum_optimize": ("Polynomial continuum", "If True, grid-search poly_continuum_order_grid (scored like the ALS lam/p search) instead of using the fixed poly_continuum_order."),
    "poly_continuum_order_grid": ("Polynomial continuum", "Candidate polynomial orders searched when poly_continuum_optimize=True."),

    # ── ALS continuum line masks ─────────────────────────────────────────
    "als_mask_ca": ("ALS line masking", "Exclude the Ca II triplet windows from the ALS continuum fit (they are absorption, not continuum)."),
    "als_ca_centers": ("ALS line masking", "Rest-frame Ca II triplet centers (A) used for ALS masking and default cleaning-protect windows."),
    "als_ca_half_widths": ("ALS line masking", "Half-widths (A) of the Ca II exclusion windows used only in the ALS continuum fit."),
    "als_ca_frame": ("ALS line masking", "'rest' or 'observed': frame the ALS Ca II centers are given in."),
    "mask_emission_line_velocity_pad_kms": ("Emission masking", "Velocity padding (km/s) widening the fixed-Angstrom kinematic-fit emission-line pre-mask half-width for broad lines (see emission_half_width_A)."),
    "als_mask_velocity_pad_kms": ("ALS line masking", "Extra velocity padding (km/s) applied to all ALS line/window masks to cover systemic-velocity uncertainty."),
    "als_mask_center_shift_A": ("ALS line masking", "Empirical rest-frame shift (A) applied to all ALS line/window mask centers."),
    "als_auto_caii_shift": ("ALS line masking", "Auto-refine als_mask_center_shift_A from the measured Ca II triplet position. Masking only; does not affect the LOSVD fit."),
    "als_auto_caii_search_hw_A": ("ALS line masking", "Half-width (A) of the search window used to measure the Ca II shift for als_auto_caii_shift."),
    "als_extra_mask_windows": ("ALS line masking", "Extra ((lo, hi), ...) wavelength windows excluded from the ALS continuum fit."),
    "als_extra_mask_frame": ("ALS line masking", "'rest' or 'observed': frame of als_extra_mask_windows."),
    "als_extra_mask_lines": ("ALS line masking", "Extra ((center, half_width[, name]), ...) line masks for the ALS continuum fit."),
    "als_extra_mask_line_frame": ("ALS line masking", "'rest' or 'observed': frame of als_extra_mask_lines."),
    "extra_absorption_lines": ("ALS line masking", "((center_A, half_width_A[, name]), ...) rest-frame lines applied to BOTH ALS masking and kinematic clean-protection."),
    "als_use_default_caii_region_lines": ("ALS line masking", "Enable a built-in table of common Ca-triplet-region absorption/emission/Paschen lines."),
    "als_default_abs_half_width": ("ALS line masking", "Half-width (A) for the built-in default absorption lines."),
    "als_default_em_half_width": ("ALS line masking", "Half-width (A) for the built-in default emission lines."),
    "als_default_pas_half_width": ("ALS line masking", "Half-width (A) for the built-in default Paschen lines."),
    "als_auto_mask_abs_lines": ("ALS line masking", "Auto-detect and mask strong absorption lines (beyond Ca II) for the ALS continuum fit via local S/N."),
    "als_auto_mask_paschen": ("ALS line masking", "Include H I Paschen lines in the ALS auto-mask S/N detection."),
    "als_auto_mask_snr_threshold": ("ALS line masking", "S/N excess threshold that triggers auto-masking a candidate line for the ALS fit."),
    "als_auto_mask_half_width_A": ("ALS line masking", "Half-width (A) applied to lines caught by ALS auto-masking."),
    "als_auto_mask_snr_context_A": ("ALS line masking", "Flanking window (A) used to estimate local continuum/noise for ALS auto-mask S/N."),

    # ── ALS good-pixel cleaning ───────────────────────────────────────────
    "als_abs_clean": ("ALS good-pixel cleaning", "Enable pPXF-like iterative sigma-clipping of absorption outliers from the ALS continuum fit."),
    "als_abs_clean_iter": ("ALS good-pixel cleaning", "Max iterations of the ALS absorption-clean loop."),
    "als_abs_clean_sigma": ("ALS good-pixel cleaning", "Sigma-clip threshold for the ALS absorption-clean loop."),
    "als_abs_clean_grow_A": ("ALS good-pixel cleaning", "Grow each newly rejected ALS-clean pixel by this many Angstrom on each side."),
    "als_abs_clean_two_sided": ("ALS good-pixel cleaning", "If True, clip both positive and negative residual outliers in the ALS clean loop (default: absorption-side only)."),
    "als_abs_clean_minpix": ("ALS good-pixel cleaning", "Stop rejecting pixels once fewer than this many good pixels would remain."),
    "als_abs_clean_reoptimize_final": ("ALS good-pixel cleaning", "Re-run the ALS hyperparameter search once more after the final clean mask is fixed."),
    "als_abs_clean_init_only": ("ALS good-pixel cleaning", "Run absorption-cleaning only during the initial ALS continuum; reuse that mask on later outer iterations."),

    # ── LOSVD / template options ──────────────────────────────────────────
    "icoff": ("LOSVD/template", "Continuum-offset mode: 0 = fixed (coff, coff2), 1 = both float, 2 = coff fixed & coff2 floats."),
    "coff": ("LOSVD/template", "Additive continuum offset term (equivalent-width-like); meaning depends on icoff."),
    "coff2": ("LOSVD/template", "Second continuum offset term; meaning depends on icoff."),
    "fortran_nlosvd_full_x": ("LOSVD/template", "Match the legacy Fortran convolution-grid sizing convention exactly."),
    "fortran_template_mixture": ("LOSVD/template", "Use the legacy Fortran template-mixture formula (sum of weighted templates over total weight)."),
    "fortran_mask_template_outside": ("LOSVD/template", "Exclude pixels outside a template's wavelength coverage from that template's contribution."),
    "fit_global_amp": ("LOSVD/template", "Fit an overall multiplicative amplitude on the model spectrum. Not compatible with fit_als_continuum=True."),
    "continuum_poly_mode": ("LOSVD/template", "'none', 'additive', or 'multiplicative' low-order polynomial continuum correction (pre-normalised mode only)."),
    "continuum_poly_bound": ("LOSVD/template", "Bound on the fitted continuum_poly_mode coefficient."),

    # ── Instrumental LSF matching ─────────────────────────────────────────
    "data_fwhm_A": ("Instrumental LSF", "Instrumental line-spread-function FWHM (A) of the galaxy spectrum. Must be set together with template_fwhm_A to enable LSF matching; leave both None (default) to assume the two are already matched. See the class docstring's 'Known limitations' section for validated recovery behavior when the true sigma is comparable to this LSF."),
    "template_fwhm_A": ("Instrumental LSF", "Instrumental line-spread-function FWHM (A) of the stellar template library, in the template's native (rest-frame) wavelength units. Must be set together with data_fwhm_A."),
    "data_fwhm_frame": ("Instrumental LSF", "'observed' (default) or 'rest': frame data_fwhm_A is quoted in. 'observed' is divided by (1+zgal) before comparing to template_fwhm_A, since LSF is a property of the instrument at observed wavelengths."),

    # ── Optimizer settings ────────────────────────────────────────────────
    "map_maxiter": ("Optimizer", "Max L-BFGS-B iterations for the MAP fit."),
    "map_maxfun": ("Optimizer", "Max objective-function evaluations for the MAP fit."),
    "map_ftol": ("Optimizer", "L-BFGS-B relative function-value convergence tolerance. Default 1e-10 is validated for ~20x speedup over tighter values with no measurable LOSVD-shape change; do not loosen further without re-validating."),
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
            "FitConfig.describe('als').\n\n"
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
    values, optionally filtered to a subsystem, e.g. ``cfg.describe("als")``.

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
        Pre-normalised input vs. co-fit ALS (asymmetric least squares)
        continuum baseline.
    Pre-normalised mode, ALS continuum, ALS hyperparameter search, ALS line masking, ALS good-pixel cleaning
        Options specific to each continuum-handling path.
    LOSVD/template
        Template-mixture and continuum-offset conventions inherited from
        the legacy Fortran objective function.
    Optimizer
        L-BFGS-B iteration/tolerance/JAX settings for the reported fit
        itself, and internally for the ``xlam`` search, ALS continuum
        convergence, and ``clean``-mode outlier rejection.
    Bayesian sampling
        NUTS/HMC posterior-sampling settings, used only by the optional,
        non-default full-posterior path
        (:func:`kinextract.bayesian.fit_state_bayesian`, called directly
        rather than through :func:`~kinextract.fitting.run_spectral_fit`).
    Kinematic cleaning
        Iterative sigma-clipping of outliers from the LOSVD chi-squared
        fit itself (distinct from ALS good-pixel cleaning, which only
        affects the continuum estimate).
    Emission masking
        Pre-fit exclusion of known and unknown emission-line pixels.

    Performance
    ---------------------------------------------------------------------
    The reported fit is a bound-constrained L-BFGS-B optimum (MAP point
    estimate) of ``chi2 + wing-tapered smoothness penalty + LOSVD
    normalization penalty``, the same objective minimised by the original
    Fortran implementation this package is a port of -- fast (typically a
    few seconds for a single-template mock or moderate real spectrum,
    including the ``xlam`` auto-selection grid search and ALS continuum
    outer loop). A full-posterior (NUTS/HMC) alternative is available via
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
    - ``als_lam``'s fixed default is implicitly tuned for one
      instrument's own continuum shape/pixel scale, which can elevate
      chi2_red for narrower-window/finer-sampled instruments (e.g.
      STIS-like setups relative to MUSE) unless a per-target value is
      selected -- this is why ``als_optimize=True`` (per-target ALS
      hyperparameter search) is the default, together with
      ``als_p_grid`` defaulting to ``(0.05,)`` so the search covers
      ``als_lam`` without changing the continuum's asymmetry unless
      explicitly requested (a symmetric grid would bias fits toward the
      wrong side of absorption features). This addresses chi2_red
      specifically -- it does **not** mean the ALS continuum *shape* fit
      is always trustworthy: on complex, high-velocity-dispersion
      spectra, the fitted continuum can come out nearly linear even when
      a lower, more flexible ``als_lam`` would give a more realistic
      shape at *equal or better* final chi2_red, since the
      hyperparameter search's scoring can settle on an over-smoothed
      solution. Inspect ``plot_als_continuum(fit, cfg)``'s overlay
      directly for any fit where the continuum shape itself matters,
      rather than trusting chi2_red alone; ``continuum_method="polynomial"``
      (see below) is an alternative that is less prone to this
      over-smoothing failure mode for some spectra.
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

    Examples
    --------
    >>> from kinextract import FitConfig
    >>> cfg = FitConfig(template_list_file="Tlist", zgal=0.0016,
    ...                  wavefitmin=8400.0, wavefitmax=8800.0,
    ...                  fit_als_continuum=True, clean=True)
    >>> cfg.describe("clean")  # doctest: +SKIP
    """
    # ── Paths ────────────────────────────────────────────────────────────────
    # gal_file is optional for command-line mode, but useful in notebooks.
    gal_file: str = ""
    template_list_file: str = "Tlist"
    template_dir: Optional[str] = None
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
    smoothing: Optional[float] = None

    # ── Auto smoothing parameter selection ─────────────────────────────────
    # If True, search xlam_auto_grid for the best xlam according to
    # xlam_criterion.  The selected value is stored in both st.xlam and
    # cfg.xlam so all subsequent MAP fits and bootstrap refits use it.
    # Bootstrap workers always set xlam_auto=False (via _make_frozen_cfg)
    # so the search runs only once per spectrum.
    xlam_auto: bool = False
    xlam_auto_grid: tuple = (100., 1000., 10000., 100000.)

    # How to select xlam from the grid:
    #
    #   "chi2" (default) — runs all grid fits, then picks the *largest* xlam
    #       (most regularised) whose chi2_red is within xlam_chi2_tolerance of
    #       the grid minimum.  Scale-invariant: works for any sigma without
    #       per-galaxy tuning.  Roughness is still logged at every grid point
    #       as a diagnostic.
    #
    #   "roughness" — original criterion: picks the *smallest* xlam whose
    #       LOSVD roughness falls at or below xlam_smooth_threshold.  Poorly
    #       calibrated for broad LOSVDs (large sigma); kept for backward compat.
    xlam_criterion: str = "chi2"
    # Maximum fractional increase in chi2_red relative to the grid minimum
    # that is still considered acceptable.  0.02 means "chi2 may increase by
    # at most 2%".  Only used when xlam_criterion="chi2".
    xlam_chi2_tolerance: float = 0.02

    # Roughness threshold (only used when xlam_criterion="roughness").
    xlam_smooth_threshold: float = 0.25

    # Unimodality constraint applied under both criteria: LOSVD must have at
    # most xlam_max_peaks prominent peaks (prominence >= xlam_peak_min_prominence
    # x global peak).  Multi-peaked solutions at low xlam are fitting artifacts.
    xlam_max_peaks: int = 1
    xlam_peak_min_prominence: float = 0.1
    # Max iterations for each search fit (None -> use map_maxiter).
    # The search only needs an approximate LOSVD shape, so a smaller budget
    # (e.g., 2000) speeds up the search without affecting the final result.
    xlam_auto_maxiter: Optional[int] = None

    losvd_vmin: Optional[float] = None
    losvd_vmax: Optional[float] = None

    # Number of non-parametric LOSVD bins.
    n_losvd_bins: int = 29

    # ── Continuum mode ──────────────────────────────────────────────────────
    # False = pre-normalised mode
    # True  = full-scale continuum co-fit, method selected by continuum_method
    fit_als_continuum: bool = False

    # Which full-scale continuum model to co-fit when fit_als_continuum=True:
    # "joint" (default), "als", or "polynomial".
    #
    # "joint" folds a penalized-B-spline (P-spline) continuum directly into
    # the same single L-BFGS-B optimization as the LOSVD and template
    # weights (see kinextract.joint), rather than treating continuum
    # estimation as a separate sub-fit with its own hyperparameter search
    # and overfitting heuristic. This is the primary/recommended method:
    # on a real MUSE spectrum, ALS's hyperparameter search settled on an
    # oversmoothed, near-linear continuum despite a more flexible als_lam
    # giving a healthier chi2_red and visibly tracking the data's real
    # curvature -- every ALS grid candidate failed the overfitting floor
    # check, and the fallback logic is systematically biased toward the
    # *smoothest* available option. The same failure mode was confirmed
    # for the standalone polynomial alternative.
    #
    # "als" (asymmetric-least-squares smoothing spline) and "polynomial"
    # (asymmetric-reweighted polynomial -- see "Polynomial continuum
    # options" below) remain available as opt-in alternatives for users
    # who specifically want a separate continuum sub-fit instead.
    continuum_method: str = "joint"

    # ── Joint continuum-in-the-model options (continuum_method="joint") ────
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

    # If True, use the joint fitting engine (v_center recentering, sigl0
    # fixed-point convergence, safer xlam auto-selection) even in
    # pre-normalized mode (fit_als_continuum=False), with the continuum
    # fixed at 1.0 rather than co-fit (see kinextract.joint.fit_joint's
    # `fit_continuum` parameter). Default False: joint's sigl0 fixed-point
    # iteration costs up to n_sigl0_iter * len(xlam_auto_grid) full
    # optimizations per fit (up to 15x the shipped pre-normalized path's
    # single fit), so this stays opt-in rather than silently slowing down
    # every default (unconfigured) pre-normalized fit package-wide.
    joint_prenorm: bool = False

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
    als_optimize_init_only: bool = True

    # Clamp ALS continuum values.
    als_clip: tuple = (0.0, float("inf"))

    # Whether column 3 of a raw .spec file is variance or sigma.
    spec_col3_is_variance: bool = False

    # When False, ignore the per-pixel input errors and use the median error
    # value uniformly across all pixels (equal weighting).  The error column
    # is still read; only the weights used in fitting are replaced.
    use_spectrum_errors: bool = True

    # ── ALS hyperparameter optimization ─────────────────────────────────────
    als_optimize: bool = True
    als_lam_grid: tuple = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
    als_p_grid: tuple = (0.05,)
    als_lam_grid_fine_n: int = 5
    als_p_grid_fine_n: int = 5
    als_opt_verbose: bool = False
    als_template_selection: str = "lsq"  # "lsq" or "nnls"
    # BIC-based continuum overfitting protection.
    # als_use_bic=True: score = chi2_total + als_bic_dof_penalty * k_eff * ln(n)
    #   where k_eff = trace((W + lam*D'D)^{-1} W) estimated via Hutchinson.
    # als_chisq_floor: hard-reject any (lam, p) where chi2_red < this value.
    als_use_bic: bool = True
    als_chisq_floor: float = 1.0
    als_hutchinson_probes: int = 25
    als_bic_dof_penalty: float = 1.0
    # Optional minimum lambda to prevent pathological low-lambda fits
    als_lambda_floor: Optional[float] = None

    # ── Polynomial continuum options (continuum_method="polynomial") ────────
    # Same asymmetric-reweighting idea as the ALS baseline (see
    # asymmetric_least_squares_continuum), but with a low-order polynomial
    # basis instead of a penalized smoothing spline -- far fewer effective
    # degrees of freedom, so it cannot track noise/overfit a large template
    # mixture's scoring proxy the way an under-constrained ALS fit can.
    poly_continuum_order: int = 4
    poly_continuum_p: float = 0.05
    poly_continuum_niter: int = 20
    # If True, search poly_continuum_order_grid (scored the same way as the
    # ALS lam/p search, including als_use_bic/als_chisq_floor) instead of
    # using the fixed poly_continuum_order.
    poly_continuum_optimize: bool = True
    poly_continuum_order_grid: tuple = (2, 3, 4, 5, 6, 8)

    # ── ALS continuum line masks ────────────────────────────────────────────
    # These masks affect only the ALS continuum fit, not the LOSVD/template fit.
    als_mask_ca: bool = True
    als_ca_centers: tuple = (8498.02, 8542.09, 8662.14)
    als_ca_half_widths: tuple = (8.0, 8.0, 8.0)
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
    als_abs_clean_sigma: float = 3.0
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
    # ftol=2e-14/gtol=1e-10 (the previous defaults) sit far past the point
    # where the objective has stopped changing at any physically meaningful
    # level: profiling shows they force many thousands of extra L-BFGS-B
    # iterations chasing floating-point noise, without changing the
    # recovered LOSVD beyond ~1% of peak amplitude (numerical noise, not a
    # different solution). Tightening past this point can also leave the
    # optimizer far enough from a true stationary point that the Laplace
    # Hessian (see errors.py) is not positive-definite. 1e-10/1e-8 was
    # validated to give a ~20x speedup with no measurable loss of fit
    # quality; do not loosen further without re-validating against chi2
    # and LOSVD-shape stability.
    map_ftol: float = 1e-10
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
    # This is separate from ALS continuum good-pixel cleaning.
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

    describe = _HybridDescribe()

    def __post_init__(self):
        """Normalize aliases and validate cross-field constraints after construction.

        Runs automatically after ``FitConfig(...)``. Applies the
        ``smoothing`` -> ``xlam`` alias, loads ``galaxy_params_path`` if
        given, lower-cases/validates the various ``*_frame`` string fields,
        and raises ``ValueError`` for a handful of known-invalid
        combinations (e.g. ``fit_als_continuum`` with ``fit_global_amp``,
        or ``losvd_vmax <= losvd_vmin``).
        """
        if self.smoothing is not None:
            self.xlam = float(self.smoothing)

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

        if self.als_clip[0] < 0:
            raise ValueError("als_clip[0] must be >= 0")

        self.continuum_poly_mode = self.continuum_poly_mode.lower().strip()
        if self.continuum_poly_mode not in ("none", "additive", "multiplicative"):
            raise ValueError(
                "continuum_poly_mode must be 'none', 'additive', or 'multiplicative'"
            )

        self.continuum_method = self.continuum_method.lower().strip()
        if self.continuum_method not in ("joint", "als", "polynomial"):
            raise ValueError(
                "continuum_method must be 'joint', 'als', or 'polynomial'"
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
            "data_fwhm_frame",
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
    ``als_ca_centers``). ``als_clip`` may use the string ``"inf"`` for its
    upper bound.

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
