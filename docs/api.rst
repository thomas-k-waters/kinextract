API reference
=============

This page covers the public surface exposed at the top level (``import
kinextract`` / ``from kinextract import ...``), grouped by the submodule
that defines each name. Internal helpers (leading-underscore names) are not
listed here but remain documented and importable from their defining
submodule for advanced use.

Configuration
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.FitConfig
   kinextract.sigma_scaled_lam
   kinextract.PENALTY_ORDER

The defaults are the recipe validated on NGC 4751 MUSE data: ``lam = 4096`` on the order-3 (third-difference)
penalty, a co-fitted continuum (4 knots), and ``lam_rule='sigma_scaled'``, which refits LOSVDs broader than
350 km/s at ``sigma_scaled_lam(sigma) = 4096 (sigma / 350)^6`` (capped at 131072) and retries fits still at the
velocity-grid edge at a larger ``lam``.

Core LOSVD fitter
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.build_bspline_basis
   kinextract.build_spline_truth
   kinextract.difference_matrix
   kinextract.default_initial_coeffs
   kinextract.fit_log_spline_map
   kinextract.edge_amplitude_frac

Forward model
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.build_velocity_grid
   kinextract.shift_template
   kinextract.build_template_stack
   kinextract.mask_wavelength_windows

Gauss-Hermite moments
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.gauss_hermite_model
   kinextract.fit_gauss_hermite
   kinextract.compute_direct_moments
   kinextract.fit_h3_h4
   kinextract.compute_losvd_moments

Multi-template fitting (main entry point)
------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fit_losvd
   kinextract.fit_losvd_basis
   kinextract.fit_losvd_parallel
   kinextract.with_lam_rule

Uncertainty
-----------

The Laplace (curvature) approximation to the posterior at the fit's own ``lam``, jointly over the LOSVD and continuum
coefficients. ``losvd_error_bands`` is the recipe for real IFU data: the sky-line + template-mismatch noise model
(below), Laplace with those errors at the fit's ``lam`` and ``rho`` without its clipped pixels, and the draws moved
onto the fit with ``recenter_samples``; ``losvd_error_bands_parallel`` runs it over many bins.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.losvd_error_bands
   kinextract.losvd_error_bands_parallel
   kinextract.laplace_losvd_uncertainty
   kinextract.laplace_losvd_uncertainty_parallel
   kinextract.laplace_covariance_at_lam
   kinextract.laplace_covariance_at_lam_with_continuum
   kinextract.recenter_samples

A NUTS/NumPyro reference mode also exists
(:mod:`kinextract.nuts_uncertainty`, ``nuts_losvd_uncertainty`` -- install
the optional ``nuts`` extra): it samples the exact posterior instead of its
Gaussian approximation, much more slowly, for spot-checks of the Laplace
bands. It is kept out of the top-level ``kinextract`` namespace so the
``numpyro`` dependency stays optional; import it from its own submodule.

Robust fitting, cleaning and emission-line masking
-----------------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fit_losvd_robust
   kinextract.fit_losvd_robust_parallel
   kinextract.compute_local_inflation
   kinextract.full_pixel_model
   kinextract.reconstruct_continuum
   kinextract.compute_fit_rms
   kinextract.CAII_PROTECTED_WINDOWS
   kinextract.detect_emission_lines
   kinextract.DEFAULT_CAT_EMISSION_LINES
   kinextract.detect_emission_from_residual

Continuum-cofitting
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fit_continuum_block

Continuum co-fitting is switched on with ``FitConfig.continuum_knots``; every fitting entry point
(``fit_losvd``, ``fit_losvd_robust``, the Laplace engines) then fits the shared multiplicative continuum jointly.

Line-spread-function (LSF) matching
--------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fwhm_to_sigma
   kinextract.muse_lsf_fwhm
   kinextract.variable_gaussian_convolve
   kinextract.match_gaussian_lsf

Correlated-noise (AR(1)) whitening
------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.build_ar1_covariance
   kinextract.whiten
   kinextract.estimate_correlation_length
   kinextract.measure_and_whiten

Systematic residuals: sky lines and template mismatch
--------------------------------------------------------

A per-pixel variance model ``err^2 + (kappa * s)^2 + (f * model)^2`` for the residual systematics of real IFU
spectra (``s`` = the LSF-blended sky-line spectrum), fitted to the residuals of a preliminary fit. Use it for the
uncertainty bands, not the fit -- :func:`kinextract.losvd_error_bands` does this. See ``05_noise_correlation.ipynb``,
Section 8.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.sky_line_profile
   kinextract.load_muse_sky_lines
   kinextract.fit_noise_model
   kinextract.noise_model_errors

E-MILES SSP templates
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.parse_emiles_filename
   kinextract.pack_emiles_templates
   kinextract.load_emiles_templates

IFU cube I/O
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.MuseCube
   kinextract.white_light_image
   kinextract.extract_aperture_spectrum
   kinextract.load_muse_cube
   kinextract.save_muse_cube_fits
   kinextract.trim_cube_wavelength_fits
   kinextract.combine_bin_spectrum_robust
   kinextract.spatial_correlation_inflation
   kinextract.BACON2017_VARIANCE_CORRECTION_FACTOR
   kinextract.build_mock_cube

Spatial binning
----------------

Two schemes for extracting per-bin spectra from an IFU cube: PA-aware
radial x angular sectors (log/linear-radial x equal-angle or
equal-sin(theta)), and a second scheme reading a legacy Fortran
``sco_framework``-pipeline ``bin_r.out``/``bin_v.out`` dartboard geometry
directly, for projects that need bit-identical bins to that downstream
toolchain rather than just a resembling scheme.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.DartboardBins
   kinextract.log_radial_edges
   kinextract.linear_radial_edges
   kinextract.pixel_aware_radial_edges
   kinextract.uniform_theta_edges
   kinextract.build_reflected_sin_theta_edges
   kinextract.assign_spaxels_to_bins
   kinextract.load_legacy_bin_edges
   kinextract.assign_legacy_dartboard_bins

Template reduction
-------------------

Iterative full-grid-fit + NNLS-threshold reduction to a representative
template subset, reusable across a whole cube.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.TemplateReductionResult
   kinextract.reduce_templates_iteratively
   kinextract.save_template_subset
   kinextract.load_template_subset

Legacy Fortran pipeline I/O
------------------------------

Writes a fit result in the exact ``.fit``/``.ascii``/``.rms``/``.mcfit``
formats the legacy ``sco_framework`` Fortran pipeline's own ``fitlovw.f``
produces, for handing kinextract's output to that downstream toolchain
(``pspecfit``/``pfitlov``/``pallmc``/``pfitlove``/``rtransvd``) unmodified.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.write_fit_outputs

``spmc_in`` files for the SCO Schwarzschild models (``model.x``), written directly from Laplace or NUTS
percentiles. With ``b_samples=`` and ``model_bins=`` the errors in each dynamical-model velocity bin are widened to at
least the binned posterior scatter: ``model.x`` sums per-point errors linearly over a bin, which understates it.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.legacy_io.write_spmc_in_laplace
   kinextract.legacy_io.transvd_bounds_laplace
   kinextract.legacy_io.spmc_side_inverts

Plotting
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.plot_bin_map
   kinextract.plot_moment_maps
