kinextract
==========

Non-parametric binned-LOSVD spectral fitter for galaxy kinematics.

``kinextract`` fits galaxy spectra using a non-parametric line-of-sight
velocity distribution (LOSVD) represented directly on a velocity grid -- no
Gauss-Hermite parametrisation assumed. A continuum baseline (asymmetric
least-squares or a low-order polynomial) is co-fitted with the LOSVD, and
regularization strength is selected automatically. The package uses
spectroscopic data for use in stellar dynamical modeling.

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
