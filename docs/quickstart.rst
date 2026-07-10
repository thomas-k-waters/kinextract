Quick start
===========

.. code-block:: python

   from kinextract import run_spectral_fit, load_config_from_toml

   cfg = load_config_from_toml("kinextract.config")
   fit = run_spectral_fit(cfg, gal_file="bin0105.spec")

   # Inspect the fit
   v  = fit["state"].xl           # velocity grid (km/s)
   b  = fit["outputs"]["b"]       # recovered LOSVD
   chi2_red = fit["outputs"]["chi2_red"]
   print(f"chi2_red = {chi2_red:.3f}")

   # Plot
   from kinextract import plot_fit, plot_continuum
   plot_fit(fit)
   plot_continuum(fit)   # only if cfg.fit_continuum = True

``cfg.fit_continuum = True`` co-fits the continuum baseline: a
penalized-B-spline continuum is folded directly into the same optimization
as the LOSVD and template weights (see :mod:`kinextract.joint`).
Alternatively, pre-normalize the spectrum once (e.g. via the standalone
:func:`kinextract.continuum.asymmetric_least_squares_continuum` utility --
see ``examples/notebooks/06_prenormalized_workflow.ipynb``) and fit with
``fit_continuum = False``.

See :meth:`~kinextract.config.FitConfig.describe` (or ``help(FitConfig)``)
for every tunable field, grouped by subsystem, e.g.
``FitConfig.describe("xlam")`` for just the regularization-selection
options.

Error estimation
-----------------

.. code-block:: python

   from kinextract import LOSVDErrorEstimator, load_config_from_toml, run_spectral_fit

   cfg  = load_config_from_toml("kinextract.config")
   fit  = run_spectral_fit(cfg, gal_file="bin0105.spec")
   est  = LOSVDErrorEstimator(fit, cfg)

   laplace = est.laplace_covariance()
   boot    = est.residual_bootstrap(n_bootstrap=200, n_jobs=4)
   summary = est.summarize(laplace_result=laplace, bootstrap_result=boot)
   est.plot_losvd_with_errors(summary)

Characterizing recovery bias
-----------------------------

Near the instrumental resolution limit, recovered velocity/dispersion carry
a real, condition-dependent bias. :func:`~kinextract.validation.assess_recovery_bias`
measures this directly for a specific target by fitting matched mock spectra
(same instrument, templates, continuum, and noise level) on a grid of known
truths:

.. code-block:: python

   from kinextract import assess_recovery_bias, correct_recovered_losvd

   bias_table = assess_recovery_bias(
       fit, cfg, v_true_grid=[0.0, 50.0], sigma_true_grid=[80.0, 150.0], n_seeds=8,
   )
   corrected = correct_recovered_losvd(v_recovered=42.0, sigma_recovered=90.0, bias_table=bias_table)

See ``examples/notebooks/05_recovery_validation.ipynb`` for a full worked
example.
