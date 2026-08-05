Example notebooks
==================

Interactive tutorials live in ``examples/notebooks/`` in the repository
(browsable directly on GitHub, or open locally with ``jupyter lab``/
``jupyter notebook`` after installing kinextract). Bundled real spectra and
templates used by several of these live alongside them in
``examples/data/``. A continuous pedagogical arc: basics -> increasingly
complex features -> real data -> a full galaxy analysis.

.. list-table::
   :header-rows: 1

   * - Notebook
     - What it demonstrates
   * - ``01_basic_mock_fit.ipynb``
     - The spectral model and LOSVD parameterization; core LOSVD +
       template-mixture recovery on a synthetic mock built from real
       E-MILES templates you choose
   * - ``02_realistic_mock_fit.ipynb``
     - A genuinely degenerate multi-template mixture, regularization
       trade-offs, an h4 noise-floor analysis, this package's two
       uncertainty engines (Laplace, the default, and bootstrap as a
       cross-check), and continuum-cofitting
   * - ``03_template_mismatch.ipynb``
     - Fits the same mock with a correct vs. a systematically wrong
       template library and compares the recovered LOSVD directly --
       motivates template calibration
   * - ``04_template_calibration.ipynb``
     - Get a MUSE cube, bin it, fit every bin against the full E-MILES
       grid, and iteratively reduce to a representative template subset
   * - ``05_noise_correlation.ipynb``
     - Real MUSE pixel-noise correlation (AR(1)), the Bacon et al. (2017)
       variance-underestimate factor, and correlated-noise whitening
   * - ``06_other_instruments.ipynb``
     - Adapting this package to a different instrument (HST/STIS):
       synthetic mocks and a real STIS spectrum fit end-to-end
   * - ``07_full_galaxy_analysis.ipynb``
     - The capstone: a full real-galaxy analysis start to finish --
       spatial binning, per-bin masking/cleaning/fitting/uncertainty in
       parallel, moment maps, and radial kinematic profiles
