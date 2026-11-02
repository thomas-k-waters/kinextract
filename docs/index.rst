kinextract
==========

Non-parametric binned-LOSVD spectral fitter for galaxy kinematics.

``kinextract`` recovers the line-of-sight velocity distribution (LOSVD)
directly from an absorption-line galaxy spectrum as a smoothly-positive,
low-dimensional log-density spline -- no Gauss-Hermite (or any other
fixed-shape) functional form assumed for the kinematics themselves. One
fitting engine (:func:`kinextract.fit_losvd`) handles any number of
templates. The defaults are the recipe validated on NGC 4751 MUSE data: a
third-difference roughness penalty at ``lam = 4096``, raised for broad
LOSVDs (:func:`kinextract.sigma_scaled_lam`), a co-fitted multiplicative
continuum (:mod:`kinextract.continuum`), and residual-based error inflation
against sky-line residuals and bad pixels
(:func:`kinextract.fit_losvd_robust`).

Beyond the core LOSVD fit, the package includes: Laplace-posterior error
bands with a sky-line + template-mismatch noise model
(:mod:`kinextract.laplace_uncertainty`, :func:`kinextract.losvd_error_bands`),
pre-fit emission-line masking (:mod:`kinextract.cleaning`), AR(1)
correlated-noise whitening (:mod:`kinextract.noise_model`), E-MILES SSP template packing/LSF-matching
(:mod:`kinextract.emiles`, :mod:`kinextract.lsf`), and IFU-cube spatial
binning + template reduction for full-galaxy analyses
(:mod:`kinextract.binning`, :mod:`kinextract.template_reduction`).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   configuration
   examples
   api

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
