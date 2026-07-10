Configuration
=============

Copy ``examples/kinextract.config`` to your extraction directory and edit
the galaxy-specific parameters:

.. code-block:: toml

   [wavelength]
   zgal = 0.00157811      # galaxy redshift
   wavefitmin = 8400.0    # fit range (Angstrom)
   wavefitmax = 8800.0

   [kinematics]
   xlam_auto = true       # auto-select regularization
   xlam_smooth_threshold = 0.25

   [continuum]
   fit_continuum = true

TOML section names are purely organizational -- every field is flattened
into a single :class:`~kinextract.config.FitConfig` namespace regardless of
which section it's under. ``examples/kinextract.config`` documents the most
commonly-tuned fields inline; for the complete list, call
``FitConfig.describe()`` or read the class docstring below.

``fit_continuum = true`` turns on continuum co-fitting: a penalized-B-spline
continuum is folded directly into the same optimization as the LOSVD and
template weights (see :mod:`kinextract.joint`). Alternatively, pre-normalize
the spectrum once (e.g. via the standalone
:func:`kinextract.continuum.asymmetric_least_squares_continuum` utility --
see ``examples/notebooks/06_prenormalized_workflow.ipynb``) and fit with
``fit_continuum = false``.

.. autoclass:: kinextract.config.FitConfig
   :members: describe
   :noindex:
