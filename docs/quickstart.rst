Quick start
===========

.. code-block:: python

   import numpy as np
   from kinextract import FitConfig, build_template_stack, compute_losvd_moments, fit_losvd_robust

   # wave_fit, flux, errors: your observed spectrum on its rest-frame wavelength grid, all the same length.
   # template_wave, templates: a template library's common wavelength grid and a list of per-template flux
   # arrays (see `kinextract.load_emiles_templates`, or 04_template_calibration.ipynb).

   cfg = FitConfig(vmin=-1500.0, vmax=1500.0)   # velocity grid bounds, km/s: about V +- 5-7 sigma
   v_grid = cfg.build_velocity_grid()
   A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)

   fit = fit_losvd_robust(A_stack, flux, errors, cfg, wave_fit=wave_fit)
   print(f"success: {fit['success']}  lam_used: {fit['lam_used']:.0f}")

   moments = compute_losvd_moments(v_grid, fit["b"])
   print(f"V={moments['vherm']:.1f}  sigma={moments['sherm']:.1f}  "
         f"h3={moments['h3']:.3f}  h4={moments['h4']:.3f}")

The defaults are the recipe validated on NGC 4751 MUSE data (``dev_notes/regularization_v2/FINDINGS.md``):

- **Regularization:** ``lam = 4096`` on the third-difference penalty. ``FitConfig.lam_rule='sigma_scaled'`` refits
  LOSVDs broader than 350 km/s at ``sigma_scaled_lam(sigma) = 4096 (sigma / 350)^6`` (capped at 131072), and refits
  a fit whose LOSVD is still at the velocity-grid edge at 2x, then a further 4x, its ``lam``. The ``lam`` used is
  ``fit['lam_used']``; ``FitConfig(lam=..., lam_rule='fixed')`` pins one.
- **The continuum:** a smooth multiplicative continuum (4 knots) is co-fitted, so ``wave_fit`` is needed;
  ``FitConfig(continuum_knots=None)`` switches it off for spectra that are exactly continuum-normalized.
- **Cleaning:** ``fit_losvd_robust`` inflates the errors wherever the residuals call for it (sky-line residuals,
  bad pixels), with the CaII triplet cores protected from clipping. ``cleaning_method=None`` is a plain fit, a
  quick check.

``fit_losvd`` is the same fit without the cleaning, on already-whitened inputs. See
:meth:`~kinextract.config.FitConfig.describe` (or ``help(FitConfig)``) for every tunable field grouped by
subsystem, e.g. ``FitConfig.describe("lam")``.

Error bands
-----------

On real IFU spectra the residuals are dominated by sky-line residuals and template mismatch, not photon noise.
``losvd_error_bands`` fits a per-pixel variance model for them (``err^2 + (kappa s)^2 + (f model)^2``, with ``s``
the LSF-blended sky-line spectrum), runs the Laplace posterior with those errors at the fit's ``lam``, and moves
the draws onto the fit:

.. code-block:: python

   from kinextract import load_muse_sky_lines, losvd_error_bands, sky_line_profile

   lines = load_muse_sky_lines("MU_GSKL_140101A_sky_lines.fits")       # MUSE pipeline static calibration
   bands = losvd_error_bands(A_stack, flux, errors, wave_fit, fit, cfg,
                             sky_profile=sky_line_profile(wave_fit, *lines, z=z))
   print(f"sigma = {moments['sherm']:.1f} +/- {bands['sherm_err']:.1f}")
   # bands["b_p16"], bands["b_p84"]: the LOSVD band; bands["b_samples"]: the posterior draws

Pass ``sky_profile=None`` for data without sky residuals. ``losvd_error_bands_parallel`` runs many bins at once.
For the posterior from the pixel errors alone, use ``laplace_losvd_uncertainty``; for its exact (non-Gaussian)
form, ``kinextract.nuts_uncertainty.nuts_losvd_uncertainty`` (slow; the ``nuts`` extra).

See ``01_basic_mock_fit.ipynb`` and ``02_realistic_mock_fit.ipynb`` for the worked pattern on mocks, and
``05_noise_correlation.ipynb`` (Section 8) and ``07_full_galaxy_analysis.ipynb`` for real data.

Inputs for SCO Schwarzschild models
-----------------------------------

``kinextract.legacy_io.write_spmc_in_laplace`` writes the ``spmc_in`` files SCO's ``model.x`` reads, from the bands.
With ``b_samples=`` and ``model_bins=`` each 300 km/s model velocity bin carries at least its binned posterior
scatter (``model.x`` sums per-point errors linearly, which understates it):

.. code-block:: python

   from kinextract.legacy_io import write_spmc_in_laplace

   write_spmc_in_laplace("bin0105", v_grid - v_sys, bands["b_median"], bands["b_p16"], bands["b_p84"], "spmc/",
                         vmax_out=1950.0, b_samples=bands["b_samples"],
                         model_bins=(np.linspace(-1800, 1800, 13), 150.0))
