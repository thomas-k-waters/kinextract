Configuration
=============

``FitConfig`` is a plain dataclass -- instantiate it directly with the
fields you want to override, e.g.:

.. code-block:: python

   from kinextract import FitConfig

   cfg = FitConfig(
       vmin=-500.0, vmax=500.0,      # velocity grid bounds, km/s
       lam_selection="aic_local",     # auto-select regularization strength (the default)
       penalty_order=3,               # third-difference, shrink-toward-Gaussian penalty (the default)
   )

There is no configuration-file loader -- every field is a normal
constructor argument or attribute. For the complete list of tunable
fields, grouped by subsystem, call ``FitConfig.describe()`` (or
``FitConfig.describe("lam")`` for just the regularization-selection
options) or read the class docstring below.

Regularization strength is selected automatically per fit
(``lam_selection="aic_local"``, the default) via a bootstrap-estimated
generalized AIC. Continuum handling: either pre-normalize the spectrum
once before fitting, or co-fit a shared multiplicative continuum directly
in the same alternating optimization
(:func:`kinextract.fit_losvd_with_continuum`, :mod:`kinextract.continuum`).

.. autoclass:: kinextract.config.FitConfig
   :members: describe
   :noindex:
