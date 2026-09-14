"""kinextract: a non-parametric LOSVD (line-of-sight velocity
distribution) extractor for galaxy spectroscopy.

Recovers the LOSVD as a smoothly-positive, low-dimensional log-density
spline directly from an absorption-line galaxy spectrum, without assuming
a Gauss-Hermite (or any other fixed-shape) functional form for the
kinematics themselves -- see `kinextract.losvd` for the method and
`kinextract.config.FitConfig` for the tunable parameters.

One fitting engine handles any number of templates (`kinextract.
multi_template.fit_losvd`, `n_templates >= 1`) -- there is no separate
single-template code path.

Submodules:
  losvd          -- the core LOSVD fitter (`fit_log_spline_map`).
  config         -- `FitConfig`, the tunable parameters.
  forward_model  -- template Doppler-shifting and design-matrix construction.
  gh_moments     -- summarizing a recovered LOSVD as (V, sigma, h3, h4).
  multi_template -- `fit_losvd`, the end-to-end fitting entry point.
  uncertainty    -- parametric-bootstrap uncertainty quantification.
  cleaning       -- pre-fit emission-line detection/masking, plus the
                    private hard-clip loop `robust_fit.fit_losvd_robust`
                    uses for its `cleaning_method="clip"`/`"inflate"` modes.
  robust_fit     -- `fit_losvd_robust`, the LOSVD fit with an optional
                    cleaning strategy against unmodeled contamination
                    (`cleaning_method="inflate"`, per-spectrum residual-based
                    error inflation, is the recommended default on real
                    data; `"clip"` is the older hard sigma-clip; `None`
                    is a plain fit with neither).
  continuum      -- shared multiplicative continuum, 3-block alternating fit.
  bias_correction -- synthetic-calibrated (Pinkney et al. 2003-style) correction
                     for regularization/shrinkage bias in recovered sigma/h4.
  lsf            -- instrument/template line-spread-function (LSF) matching,
                     needed before using any template library not already
                     natively at the target instrument's own resolution.
  aic_regularization -- data-driven regularization-strength selection via
                     Lipka & Thomas (2021)/Thomas & Lipka (2022)'s
                     bootstrap-estimated generalized AIC, as an alternative
                     to a fixed `cfg.lam`.
  noise_model    -- pluggable AR(1) noise-covariance whitening, replacing
                     the implicit independent-per-pixel-noise assumption;
                     `rho=0.0` (the default) reproduces old behavior exactly.
  emiles         -- E-MILES SSP template packing/loading, LSF-matched at
                     pack time; a first-class template source alongside
                     the real 33-star MUSE library.
  muse_cube      -- shared IFU-cube container + FITS I/O, for real or
                     synthetic MUSE-like data.
  mock_cube      -- synthetic Sersic-profile MUSE-like cube generator,
                     for testing spatial binning/template-reduction with
                     known ground truth.
  binning        -- "dartboard" spatial binning (log/linear-radial x
                     equal-angle sectors), for extracting per-bin spectra
                     from an IFU cube.
  template_reduction -- iterative full-grid-fit + NNLS-threshold
                     reduction to a representative template subset,
                     reusable across a whole cube.

See `examples/notebooks/` for worked examples and `README.md` for current
package status and scope.
"""
from __future__ import annotations

import os as _os

from ._version import __version__

# JAX float64 by default (opt out with KINEXTRACT_NO_X64=1). This is a
# correctness requirement, not a preference: under JAX's default float32,
# L-BFGS-B demonstrably stalls far from the optimum in the stiff
# heavy-penalty regime the (now default) third-difference penalty operates
# in on wide velocity grids -- measured at sigma=300/S N=30, lam=65536:
# 35% vs 6% shape residual, objective 80 units above the true minimum,
# reported as `success=True`. Float64 was validated as statistically
# neutral for the order-2 penalty path on the full 480-case grid before
# becoming the default (dev_notes/REGULARIZATION_INVESTIGATION_LOG.md,
# 2026-07-27; ~2x runtime cost). Must run before any jax-using submodule
# import below creates arrays.
if _os.environ.get("KINEXTRACT_NO_X64", "") != "1":
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)

from .aic_regularization import (
    AICSelectionResult,
    edge_amplitude_frac,
    estimate_m_eff,
    estimate_m_eff_local,
    select_lambda_aic,
    select_lambda_aic_local,
)
from .bias_correction import (
    BiasCalibration,
    PerLamCalibration,
    calibration_has_lam_axis,
    correct_losvd,
    correct_moments,
    estimate_snr_whitened,
    get_default_lam_axis_calibration,
    load_calibration_table,
    load_per_lam_calibration_table,
)
from .binning import (
    DartboardBins,
    assign_legacy_dartboard_bins,
    assign_spaxels_to_bins,
    build_reflected_sin_theta_edges,
    linear_radial_edges,
    load_legacy_bin_edges,
    log_radial_edges,
    pixel_aware_radial_edges,
    uniform_theta_edges,
)
from .cleaning import (
    CAII_PROTECTED_WINDOWS,
    DEFAULT_CAT_EMISSION_LINES,
    compute_fit_rms,
    detect_emission_from_residual,
    detect_emission_lines,
)
from .config import FitConfig
from .continuum import fit_continuum_block
from .emiles import load_emiles_templates, pack_emiles_templates, parse_emiles_filename
from .forward_model import (
    build_template_stack,
    build_velocity_grid,
    mask_wavelength_windows,
    shift_template,
)
from .gh_moments import (
    compute_direct_moments,
    compute_losvd_moments,
    fit_gauss_hermite,
    fit_h3_h4,
    gauss_hermite_model,
    shape_aware_taper_correction,
)
from .laplace_uncertainty import (
    laplace_continuum_uncertainty,
    laplace_covariance_at_lam,
    laplace_covariance_at_lam_with_continuum,
    laplace_losvd_uncertainty,
    laplace_losvd_uncertainty_parallel,
    quick_laplace_uncertainty,
)
from .legacy_io import write_fit_outputs
from .losvd import (
    build_bspline_basis,
    build_spline_truth,
    default_initial_coeffs,
    difference_matrix,
    fit_log_spline_map,
)
from .lsf import fwhm_to_sigma, match_gaussian_lsf, muse_lsf_fwhm, variable_gaussian_convolve
from .mock_cube import build_mock_cube
from .multi_template import fit_losvd, fit_losvd_basis, fit_losvd_parallel
from .muse_cube import (
    BACON2017_VARIANCE_CORRECTION_FACTOR,
    MuseCube,
    combine_bin_spectrum_robust,
    extract_aperture_spectrum,
    load_muse_cube,
    save_muse_cube_fits,
    spatial_correlation_inflation,
    white_light_image,
)
from .noise_model import (
    build_ar1_covariance,
    estimate_correlation_length,
    measure_and_whiten,
    whiten,
)
from .plotting import plot_bin_map, plot_moment_maps
from .robust_fit import (
    compute_local_inflation,
    fit_losvd_robust,
    fit_losvd_robust_parallel,
    full_pixel_model,
    reconstruct_continuum,
)
from .template_reduction import (
    TemplateReductionResult,
    load_template_subset,
    reduce_templates_iteratively,
    save_template_subset,
)
from .uncertainty import bootstrap_losvd, bootstrap_losvd_parallel, calibrate_noise_scale

__all__ = [
    "build_bspline_basis", "build_spline_truth", "difference_matrix", "default_initial_coeffs", "fit_log_spline_map",
    "build_velocity_grid", "shift_template", "build_template_stack",
    "mask_wavelength_windows",
    "gauss_hermite_model", "fit_gauss_hermite",
    "compute_direct_moments", "fit_h3_h4", "compute_losvd_moments", "shape_aware_taper_correction",
    "FitConfig",
    "fit_losvd", "fit_losvd_basis", "fit_losvd_parallel",
    "bootstrap_losvd", "bootstrap_losvd_parallel", "calibrate_noise_scale",
    "detect_emission_lines", "DEFAULT_CAT_EMISSION_LINES", "CAII_PROTECTED_WINDOWS",
    "detect_emission_from_residual", "compute_fit_rms",
    "fit_losvd_robust", "fit_losvd_robust_parallel", "compute_local_inflation",
    "reconstruct_continuum", "full_pixel_model",
    "fit_continuum_block",
    "BiasCalibration", "PerLamCalibration", "load_calibration_table", "load_per_lam_calibration_table",
    "estimate_snr_whitened", "correct_moments", "correct_losvd", "calibration_has_lam_axis",
    "get_default_lam_axis_calibration",
    "fwhm_to_sigma", "muse_lsf_fwhm", "variable_gaussian_convolve", "match_gaussian_lsf",
    "AICSelectionResult", "estimate_m_eff", "select_lambda_aic",
    "estimate_m_eff_local", "select_lambda_aic_local", "edge_amplitude_frac",
    "build_ar1_covariance", "whiten", "estimate_correlation_length", "measure_and_whiten",
    "laplace_covariance_at_lam", "laplace_losvd_uncertainty", "quick_laplace_uncertainty",
    "laplace_losvd_uncertainty_parallel", "laplace_covariance_at_lam_with_continuum",
    "laplace_continuum_uncertainty",
    "parse_emiles_filename", "pack_emiles_templates", "load_emiles_templates",
    "MuseCube", "white_light_image", "extract_aperture_spectrum", "combine_bin_spectrum_robust",
    "spatial_correlation_inflation",
    "load_muse_cube", "save_muse_cube_fits", "BACON2017_VARIANCE_CORRECTION_FACTOR",
    "build_mock_cube",
    "DartboardBins", "log_radial_edges", "linear_radial_edges", "pixel_aware_radial_edges",
    "uniform_theta_edges",
    "build_reflected_sin_theta_edges", "assign_spaxels_to_bins",
    "load_legacy_bin_edges", "assign_legacy_dartboard_bins",
    "TemplateReductionResult", "reduce_templates_iteratively", "save_template_subset", "load_template_subset",
    "write_fit_outputs",
    "plot_bin_map", "plot_moment_maps",
    "__version__",
]

