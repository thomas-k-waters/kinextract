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
   kinextract.tapered_lam_vector
   kinextract.penalty_row_velocities

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
   kinextract.shape_aware_taper_correction

Multi-template fitting (main entry point)
------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fit_losvd
   kinextract.fit_losvd_basis
   kinextract.fit_losvd_parallel

Uncertainty (parametric bootstrap)
------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.bootstrap_losvd
   kinextract.bootstrap_losvd_parallel
   kinextract.calibrate_noise_scale

Cleaning and emission-line masking
-------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fit_losvd_clean
   kinextract.detect_emission_lines
   kinextract.DEFAULT_CAT_EMISSION_LINES
   kinextract.detect_emission_from_residual

Continuum-cofitting
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fit_continuum_block
   kinextract.fit_losvd_with_continuum

Regularization-bias correction
---------------------------------

Synthetic-calibrated (Pinkney et al. 2003-style) correction for
regularization/shrinkage bias in recovered sigma/h4 -- opt-in, not the
default (see :mod:`kinextract.bias_correction`'s own module docstring for
current trust status).

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.BiasCalibration
   kinextract.PerLamCalibration
   kinextract.load_calibration_table
   kinextract.load_per_lam_calibration_table
   kinextract.estimate_snr_whitened
   kinextract.correct_moments
   kinextract.correct_losvd
   kinextract.calibration_has_lam_axis
   kinextract.get_default_lam_axis_calibration
   kinextract.run_calibration_sweep
   kinextract.save_calibration_table

Ablation / diagnostic sweeps
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.AblationConfig
   kinextract.fit_one
   kinextract.run_ablation_sweep
   kinextract.save_ablation_table
   kinextract.load_ablation_table

Line-spread-function (LSF) matching
--------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fwhm_to_sigma
   kinextract.muse_lsf_fwhm
   kinextract.variable_gaussian_convolve
   kinextract.match_gaussian_lsf

Adaptive regularization-strength selection
---------------------------------------------

Lipka & Thomas (2021)/Thomas & Lipka (2022)-style bootstrap-estimated
generalized AIC, as an alternative to a fixed ``cfg.lam``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.AICSelectionResult
   kinextract.estimate_m_eff
   kinextract.select_lambda_aic
   kinextract.estimate_m_eff_local
   kinextract.select_lambda_aic_local

Correlated-noise (AR(1)) whitening
------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.build_ar1_covariance
   kinextract.whiten
   kinextract.estimate_correlation_length

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
   kinextract.build_mock_cube

Spatial binning
----------------

"Dartboard" spatial binning (log/linear-radial x equal-angle or
equal-sin(theta) sectors), for extracting per-bin spectra from an IFU cube.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.DartboardBins
   kinextract.log_radial_edges
   kinextract.linear_radial_edges
   kinextract.uniform_theta_edges
   kinextract.build_reflected_sin_theta_edges
   kinextract.assign_spaxels_to_bins

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
