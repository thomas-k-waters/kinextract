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
   kinextract.load_config_from_toml

Top-level fitting API
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.run_spectral_fit
   kinextract.fit_state_map_with_optional_clean
   kinextract.build_parameter_xscale
   kinextract.run_iterative_clean_map
   kinextract.compute_losvd_roughness
   kinextract.compute_losvd_n_peaks

State
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.FitState
   kinextract.precompute_losvd_interp
   kinextract.precompute_ip_map
   kinextract.getnlosvd_fast_from_b

Spectrum loading
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.load_spectrum_for_fit
   kinextract.make_fit_state
   kinextract.build_initial_guess_nonparam
   kinextract.choose_norm_wavelength_frame
   kinextract.fortran_rebin_after_redshift

Continuum
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.asymmetric_least_squares_continuum
   kinextract.grow_boolean_mask
   kinextract.grow_boolean_mask_A
   kinextract.robust_sigma

``asymmetric_least_squares_continuum``/``robust_sigma`` are standalone,
one-time pre-normalization utilities -- not part of the main fitting
pipeline. See ``examples/notebooks/06_prenormalized_workflow.ipynb``.

Joint continuum-in-the-model fit
---------------------------------

The continuum-cofitting method (``FitConfig.fit_continuum = True``). See
:mod:`kinextract.joint` for the full rationale.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.run_joint_fit
   kinextract.fit_joint
   kinextract.fit_joint_auto_xlam
   kinextract.fit_joint_auto_xlam_sigl0
   kinextract.build_pspline_design
   kinextract.evaluate_model_gp_joint

LOSVD analysis
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.fit_losvd_gauss_hermite
   kinextract.fit_losvd_gauss_hermite_higher
   kinextract.gauss_hermite_losvd_model
   kinextract.gauss_hermite_losvd_model_ho
   kinextract.fh3_fortran_like
   kinextract.fh4_fortran_like
   kinextract.getfwhm_fortran_like

Numerics
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.evaluate_model_gp
   kinextract.objective_map
   kinextract.objective_components
   kinextract.compute_weighted_template_spectrum

Templates
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.interp_template_tp_with_outside
   kinextract.build_template_matrix_fortran
   kinextract.resolution_mismatch_sigma_A
   kinextract.convolve_gaussian_pixels

Masking
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.build_emission_line_mask
   kinextract.build_clean_protect_mask

Plotting
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.plot_fit
   kinextract.plot_losvd
   kinextract.plot_continuum

Error estimation
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.LOSVDErrorEstimator
   kinextract.estimate_losvd_errors

Recovery-bias validation
--------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.assess_recovery_bias
   kinextract.correct_recovered_losvd
   kinextract.build_matched_mock
   kinextract.true_losvd_on_grid

I/O
---

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.read_galaxy_index_flux_err
   kinextract.read_norm_spectrum
   kinextract.read_template_xy
   kinextract.read_template_list
   kinextract.read_galaxy_params
   kinextract.write_fitlov_outputs
   kinextract.write_losvd_errors_file
   kinextract.write_gh_errors_file

Optional Bayesian (NUTS) path
-------------------------------

Not part of the top-level namespace (requires ``pip install
kinextract[bayesian]``); import directly from :mod:`kinextract.bayesian`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   kinextract.bayesian.fit_state_bayesian
   kinextract.bayesian.build_numpyro_model
