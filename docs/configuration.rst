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
   fit_als_continuum = true

TOML section names are purely organizational -- every field is flattened
into a single :class:`~kinextract.config.FitConfig` namespace regardless of
which section it's under. ``examples/kinextract.config`` documents the most
commonly-tuned fields inline; for the complete list, call
``FitConfig.describe()`` or read the class docstring below.

.. autoclass:: kinextract.config.FitConfig
   :members: describe
   :noindex:
