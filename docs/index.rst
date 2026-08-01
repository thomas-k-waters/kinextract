kinextract
==========

Non-parametric binned-LOSVD spectral fitter for galaxy kinematics.

``kinextract`` recovers the line-of-sight velocity distribution (LOSVD)
directly from an absorption-line galaxy spectrum as a smoothly-positive,
low-dimensional log-density spline -- no Gauss-Hermite (or any other
fixed-shape) functional form assumed for the kinematics themselves. One
fitting engine (:func:`kinextract.fit_losvd`) handles any number of
templates; regularization strength is selected automatically via a
bootstrap-estimated generalized AIC (Lipka & Thomas 2021), with an optional
synthetic-calibrated correction for the residual shrinkage bias that
regularization itself introduces (see :mod:`kinextract.bias_correction`).

Beyond the core LOSVD fit, the package includes: parametric-bootstrap
uncertainty quantification (:mod:`kinextract.uncertainty`), iterative
sigma-clipping and pre-fit emission-line masking
(:mod:`kinextract.cleaning`), shared multiplicative continuum-cofitting
(:mod:`kinextract.continuum`), AR(1) correlated-noise whitening
(:mod:`kinextract.noise_model`), E-MILES SSP template packing/LSF-matching
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
