kinextract
==========

Non-parametric binned-LOSVD spectral fitter for galaxy kinematics.

``kinextract`` fits galaxy spectra using a non-parametric line-of-sight
velocity distribution (LOSVD) represented directly on a velocity grid -- no
Gauss-Hermite parametrisation assumed. When continuum co-fitting is enabled
(``FitConfig.fit_continuum = True``), a penalized-B-spline continuum is
folded directly into the same single optimization as the LOSVD and template
weights (see :mod:`kinextract.joint`), rather than fit as a separate
sub-fit. Alternatively, a spectrum can be pre-normalized once (e.g. via the
standalone asymmetric-least-squares utility,
:func:`kinextract.continuum.asymmetric_least_squares_continuum`) and fit
with ``fit_continuum = False``. Regularization strength is
selected automatically. The package uses spectroscopic data for use in
stellar dynamical modeling.

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
