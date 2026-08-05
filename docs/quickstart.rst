Quick start
===========

.. code-block:: python

   import numpy as np
   from kinextract import FitConfig, build_template_stack, compute_losvd_moments
   from kinextract.multi_template import fit_losvd

   # wave_fit, galaxy, errors: your observed (rest-frame) wavelength grid,
   # normalized flux, and per-pixel noise, all the same length.
   # template_wave, templates: a template library's common wavelength grid
   # and a list of per-template flux arrays (see the E-MILES notebook for
   # how to build one, or `kinextract.emiles.load_emiles_templates`).

   cfg = FitConfig(vmin=-500.0, vmax=500.0)   # velocity grid bounds, km/s
   v_grid = cfg.build_velocity_grid()
   A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)

   A_stack_w = A_stack / errors[None, :, None]
   y_w = galaxy / errors
   result = fit_losvd(A_stack_w, y_w, cfg)

   print(f"success: {result['success']}  lam_used: {result['lam_used']:.3g}")

   moments = compute_losvd_moments(v_grid, result["b"])
   print(f"V={moments['vherm']:.1f}  sigma={moments['sherm']:.1f}  "
         f"h3={moments['h3']:.3f}  h4={moments['h4']:.3f}")

Regularization strength (``cfg.lam``) is selected automatically per fit via
``cfg.lam_selection="aic_local"`` (the default) -- pass
``FitConfig(lam_selection="fixed", lam=...)`` for a fixed value instead. See
:meth:`~kinextract.config.FitConfig.describe` (or ``help(FitConfig)``) for
every tunable field grouped by subsystem, e.g. ``FitConfig.describe("lam")``
for just the regularization options.

If the spectrum is not already continuum-normalized, either pre-normalize it
once, or co-fit a shared multiplicative continuum directly
(:func:`kinextract.fit_losvd_with_continuum`, :mod:`kinextract.continuum`) --
see ``02_realistic_mock_fit.ipynb``.

Error estimation
-----------------

.. code-block:: python

   from kinextract import laplace_losvd_uncertainty

   lap = laplace_losvd_uncertainty(A_stack_w, y_w, cfg)
   print(f"sigma = {lap['point0']['sherm']:.1f} +/- {lap['sherm_std']:.1f}")

``laplace_losvd_uncertainty`` is the default uncertainty engine -- a
production-scale coverage sweep across sigma 50-600 km/s found it matches
or beats parametric-bootstrap coverage in every tested regime, dramatically
so for h4, at a real speed advantage. ``bootstrap_losvd`` (same call
pattern, ``n_boot=200, n_jobs=4``) remains available as a cross-check for
real ensemble properties (empirical skew, multimodality) the Laplace
approximation can't give.

See ``01_basic_mock_fit.ipynb`` and ``02_realistic_mock_fit.ipynb`` for the
full worked pattern, including both engines and bias-corrected variants.

Characterizing recovery bias
-----------------------------

Regularization biases recovered sigma/h4, worse at low S/N. A
synthetic-calibrated correction is available
(:func:`kinextract.correct_moments`) but is opt-in, not the default -- see
:mod:`kinextract.bias_correction`'s own module docstring for its current
trust status, and ``04_template_calibration.ipynb`` for building a
calibration matched to your own template library.

.. code-block:: python

   from kinextract import correct_moments, estimate_snr_whitened

   snr = estimate_snr_whitened(model_w)   # model_w: the fit's own best-fit model, whitened units
   corrected = correct_moments(moments["vherm"], moments["sherm"], moments["h4"], snr)
