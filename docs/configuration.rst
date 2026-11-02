Configuration
=============

``FitConfig`` is a plain dataclass -- instantiate it directly with the
fields you want to override, e.g.:

.. code-block:: python

   from kinextract import FitConfig

   cfg = FitConfig(
       vmin=-1500.0, vmax=1500.0,    # velocity grid bounds, km/s (about V +- 5-7 sigma)
       rho=0.7,                      # AR(1) pixel-noise correlation, for whitening
   )

There is no configuration-file loader -- every field is a normal
constructor argument or attribute. For the complete list of tunable
fields, grouped by subsystem, call ``FitConfig.describe()`` (or
``FitConfig.describe("lam")`` for just the regularization options) or
read the class docstring below.

The defaults are the recipe validated on NGC 4751 MUSE data: ``lam = 4096``
on the third-difference penalty, raised for broad LOSVDs and for fits still
at the velocity-grid edge (``lam_rule='sigma_scaled'``, see
:func:`kinextract.sigma_scaled_lam` and
:func:`kinextract.with_lam_rule`), and a co-fitted multiplicative continuum
with 4 knots (``continuum_knots``, :mod:`kinextract.continuum`; ``None``
switches it off for exactly continuum-normalized spectra).
``FitConfig(lam=..., lam_rule='fixed')`` fits at exactly one ``lam``.

.. autoclass:: kinextract.config.FitConfig
   :members: describe
   :noindex:
