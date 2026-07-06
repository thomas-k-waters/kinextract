"""
kinextract — Non-parametric LOSVD spectral fitting package.

The top-level namespace (``import kinextract`` / ``from kinextract import
...``) exposes the public API: configuration (:class:`FitConfig`), the
main entry point (:func:`run_spectral_fit`), error estimation
(:class:`LOSVDErrorEstimator`), empirical recovery-bias validation
(:func:`assess_recovery_bias`), plotting, and commonly reused standalone
utilities (ALS continuum fitting, Gauss-Hermite LOSVD characterization,
legacy-format file I/O).

Internal implementation helpers (leading-underscore names such as
``_fit_map_once`` or ``_update_clean_mask``) are intentionally *not*
imported here; they remain documented and importable from their defining
submodule (e.g. ``from kinextract.fitting import _fit_map_once``) for
advanced use, but are left out of tab-completion/``dir(kinextract)`` and
``__all__`` so the public surface stays easy to scan. See
``FitConfig.describe()`` for a similar "what can I tune" overview of the
configuration object specifically.
"""
from __future__ import annotations

from ._version import __version__

# ── Utilities ──────────────────────────────────────────────────────────────
from ._utils import CEE, BIG, log, Timer

# ── Configuration ──────────────────────────────────────────────────────────
from .config import FitConfig, load_config_from_toml

# ── State ──────────────────────────────────────────────────────────────────
from .state import (
    FitState,
    precompute_losvd_interp,
    precompute_ip_map,
    getnlosvd_fast_from_b,
)

# ── I/O helpers ────────────────────────────────────────────────────────────
from .io import (
    read_galaxy_index_flux_err,
    read_norm_spectrum,
    read_template_xy,
    read_template_list,
    infer_output_prefix,
    nint_fortran,
    setbadreg,
    build_wavelength_from_index,
    select_region_with_errors,
    estimate_step_from_wavelength,
    count_in_window,
    read_galaxy_params,
    write_fitlov_outputs,
    write_losvd_errors_file,
    write_gh_errors_file,
)

# ── Templates ──────────────────────────────────────────────────────────────
from .templates import (
    interp_template_tp_with_outside,
    build_template_matrix_fortran,
    resolution_mismatch_sigma_A,
    convolve_gaussian_pixels,
)

# ── Continuum ──────────────────────────────────────────────────────────────
from .continuum import (
    robust_sigma,
    asymmetric_least_squares_continuum,
    score_als_target,
    optimize_als_hyperparams_for_target,
    fit_als_target,
    grow_boolean_mask,
    grow_boolean_mask_A,
    fit_als_target_absorption_clean,
    build_als_line_mask,
    init_als_continuum,
    update_als_continuum,
)

# ── Numerics ───────────────────────────────────────────────────────────────
from .numerics import (
    evaluate_model_gp,
    objective_map,
    objective_components,
    compute_weighted_template_spectrum,
)

# Expose jax at package level for `import kinextract as sf; sf.jax`
from .numerics import jax

# ── Masking ────────────────────────────────────────────────────────────────
from .masking import (
    build_emission_line_mask,
    build_clean_protect_mask,
)

# ── Spectrum loading ────────────────────────────────────────────────────────
from .spectrum import (
    choose_norm_wavelength_frame,
    fortran_rebin_after_redshift,
    load_spectrum_for_fit,
    make_fit_state,
    build_initial_guess_nonparam,
)

# ── Fitting ────────────────────────────────────────────────────────────────
from .fitting import (
    build_parameter_xscale,
    run_iterative_clean_map,
    compute_losvd_roughness,
    compute_losvd_n_peaks,
    fit_state_map_with_optional_clean,
    run_spectral_fit,
)

# ── LOSVD analysis ─────────────────────────────────────────────────────────
from .losvd import (
    fh3_fortran_like,
    fh4_fortran_like,
    gauss_hermite_losvd_model,
    gauss_hermite_losvd_model_ho,
    fit_losvd_gauss_hermite,
    fit_losvd_gauss_hermite_higher,
    getfwhm_fortran_like,
)

# ── Plotting ───────────────────────────────────────────────────────────────
from .plotting import (
    plot_fit,
    plot_losvd,
    plot_als_continuum,
    PROMINENT_STELLAR_LINES,
)

# ── Error estimation ───────────────────────────────────────────────────────
from .errors import LOSVDErrorEstimator, estimate_losvd_errors

# ── Recovery-bias validation ───────────────────────────────────────────────
from .mocks import build_matched_mock, true_losvd_on_grid
from .validation import assess_recovery_bias, correct_recovered_losvd

__all__ = [
    "__version__",
    # utils
    "CEE", "BIG", "log", "Timer",
    # config
    "FitConfig", "load_config_from_toml",
    # state
    "FitState", "precompute_losvd_interp", "precompute_ip_map",
    "getnlosvd_fast_from_b",
    # io
    "read_galaxy_index_flux_err", "read_norm_spectrum", "read_template_xy",
    "read_template_list", "infer_output_prefix", "nint_fortran", "setbadreg",
    "build_wavelength_from_index", "select_region_with_errors",
    "estimate_step_from_wavelength", "count_in_window", "read_galaxy_params",
    "write_fitlov_outputs", "write_losvd_errors_file", "write_gh_errors_file",
    # templates
    "interp_template_tp_with_outside", "build_template_matrix_fortran",
    "resolution_mismatch_sigma_A", "convolve_gaussian_pixels",
    # continuum
    "robust_sigma", "asymmetric_least_squares_continuum", "score_als_target",
    "optimize_als_hyperparams_for_target", "fit_als_target", "grow_boolean_mask",
    "grow_boolean_mask_A", "fit_als_target_absorption_clean",
    "build_als_line_mask", "init_als_continuum", "update_als_continuum",
    # numerics
    "evaluate_model_gp", "objective_map", "objective_components",
    "compute_weighted_template_spectrum", "jax",
    # masking
    "build_emission_line_mask", "build_clean_protect_mask",
    # spectrum
    "choose_norm_wavelength_frame", "fortran_rebin_after_redshift",
    "load_spectrum_for_fit", "make_fit_state", "build_initial_guess_nonparam",
    # fitting
    "build_parameter_xscale", "run_iterative_clean_map",
    "compute_losvd_roughness", "compute_losvd_n_peaks",
    "fit_state_map_with_optional_clean", "run_spectral_fit",
    # losvd
    "fh3_fortran_like", "fh4_fortran_like", "gauss_hermite_losvd_model",
    "gauss_hermite_losvd_model_ho", "fit_losvd_gauss_hermite",
    "fit_losvd_gauss_hermite_higher", "getfwhm_fortran_like",
    # plotting
    "plot_fit", "plot_losvd", "plot_als_continuum", "PROMINENT_STELLAR_LINES",
    # errors
    "LOSVDErrorEstimator", "estimate_losvd_errors",
    # recovery-bias validation
    "build_matched_mock", "true_losvd_on_grid",
    "assess_recovery_bias", "correct_recovered_losvd",
]
