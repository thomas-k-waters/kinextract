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
  multi_template -- `fit_losvd`, the end-to-end fitting entry point, and
                    `with_lam_rule` (the sigma-scaled lam + edge retry).
  robust_fit     -- `fit_losvd_robust`, the LOSVD fit for real data:
                    per-spectrum residual-based error inflation against
                    unmodeled contamination, CaII cores protected.
  laplace_uncertainty -- Laplace posterior bands, and `losvd_error_bands`,
                    the error-band recipe for real IFU data.
  nuts_uncertainty -- NUTS sampling of the exact posterior (slow reference).
  cleaning       -- pre-fit emission-line detection/masking, plus the
                    private clip loop `fit_losvd_robust` uses for cleanup.
  continuum      -- shared multiplicative continuum, 3-block alternating fit.
  lsf            -- instrument/template line-spread-function (LSF) matching,
                     needed before using any template library not already
                     natively at the target instrument's own resolution.
  noise_model    -- AR(1) noise-covariance whitening (`rho`), and the
                     sky-line + template-mismatch variance model
                     (`noise_model_errors`) for uncertainty bands.
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
# objectives in the stiff heavy-penalty regime of the third-difference
# penalty on wide velocity grids stall the optimizer far from the optimum --
# measured at sigma=300/S N=30, lam=65536: 35% vs 6% shape residual,
# objective 80 units above the true minimum, reported as `success=True`
# (dev_notes/REGULARIZATION_INVESTIGATION_LOG.md, 2026-07-27). Must run
# before any jax-using submodule import below creates arrays.
if _os.environ.get("KINEXTRACT_NO_X64", "") != "1":
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)

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
from .config import PENALTY_ORDER, FitConfig, sigma_scaled_lam
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
)
from .laplace_uncertainty import (
    laplace_covariance_at_lam,
    laplace_covariance_at_lam_with_continuum,
    laplace_losvd_uncertainty,
    laplace_losvd_uncertainty_parallel,
    losvd_error_bands,
    losvd_error_bands_parallel,
    recenter_samples,
)
from .legacy_io import write_fit_outputs
from .losvd import (
    build_bspline_basis,
    build_spline_truth,
    default_initial_coeffs,
    difference_matrix,
    edge_amplitude_frac,
    fit_log_spline_map,
)
from .lsf import fwhm_to_sigma, match_gaussian_lsf, muse_lsf_fwhm, variable_gaussian_convolve
from .mock_cube import build_mock_cube
from .multi_template import fit_losvd, fit_losvd_basis, fit_losvd_parallel, with_lam_rule
from .muse_cube import (
    BACON2017_VARIANCE_CORRECTION_FACTOR,
    MuseCube,
    combine_bin_spectrum_robust,
    extract_aperture_spectrum,
    load_muse_cube,
    save_muse_cube_fits,
    spatial_correlation_inflation,
    trim_cube_wavelength_fits,
    white_light_image,
)
from .noise_model import (
    build_ar1_covariance,
    estimate_correlation_length,
    fit_noise_model,
    load_muse_sky_lines,
    measure_and_whiten,
    noise_model_errors,
    sky_line_profile,
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

__all__ = [
    "build_bspline_basis", "build_spline_truth", "difference_matrix", "default_initial_coeffs", "fit_log_spline_map",
    "edge_amplitude_frac",
    "build_velocity_grid", "shift_template", "build_template_stack",
    "mask_wavelength_windows",
    "gauss_hermite_model", "fit_gauss_hermite",
    "compute_direct_moments", "fit_h3_h4", "compute_losvd_moments",
    "FitConfig", "PENALTY_ORDER", "sigma_scaled_lam",
    "fit_losvd", "fit_losvd_basis", "fit_losvd_parallel", "with_lam_rule",
    "detect_emission_lines", "DEFAULT_CAT_EMISSION_LINES", "CAII_PROTECTED_WINDOWS",
    "detect_emission_from_residual", "compute_fit_rms",
    "fit_losvd_robust", "fit_losvd_robust_parallel", "compute_local_inflation",
    "reconstruct_continuum", "full_pixel_model",
    "fit_continuum_block",
    "fwhm_to_sigma", "muse_lsf_fwhm", "variable_gaussian_convolve", "match_gaussian_lsf",
    "build_ar1_covariance", "whiten", "estimate_correlation_length", "measure_and_whiten",
    "sky_line_profile", "load_muse_sky_lines", "fit_noise_model", "noise_model_errors",
    "laplace_covariance_at_lam", "laplace_covariance_at_lam_with_continuum",
    "laplace_losvd_uncertainty", "laplace_losvd_uncertainty_parallel",
    "losvd_error_bands", "losvd_error_bands_parallel", "recenter_samples",
    "parse_emiles_filename", "pack_emiles_templates", "load_emiles_templates",
    "MuseCube", "white_light_image", "extract_aperture_spectrum", "combine_bin_spectrum_robust",
    "spatial_correlation_inflation",
    "load_muse_cube", "save_muse_cube_fits", "trim_cube_wavelength_fits", "BACON2017_VARIANCE_CORRECTION_FACTOR",
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

