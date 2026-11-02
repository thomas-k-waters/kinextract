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
     - A genuinely degenerate multi-template mixture, what the
       regularization costs and the h4 noise floor, the ``lam`` rule
       for broad LOSVDs, Laplace uncertainty, and the co-fitted
       continuum
   * - ``03_template_mismatch.ipynb``
     - Fits the same mock with a correct vs. a systematically wrong
       template library and compares the recovered LOSVD directly --
       motivates template calibration
   * - ``04_template_calibration.ipynb``
     - Get a MUSE cube, bin it, fit every bin against the full E-MILES
       grid, and iteratively reduce to a representative template subset
       (the dust-free half of NGC 4751's MUSE NFM cube)
   * - ``05_noise_correlation.ipynb``
     - Real MUSE pixel-noise correlation (AR(1)), the Bacon et al. (2017)
       variance-underestimate factor, correlated-noise whitening, and the
       sky-line + template-mismatch noise model behind ``losvd_error_bands``
   * - ``06_other_instruments.ipynb``
     - Adapting this package to a different instrument (HST/STIS):
       synthetic mocks and a real STIS spectrum fit end-to-end
   * - ``07_full_galaxy_analysis.ipynb``
     - The capstone: a complete extraction of a real cube (NGC 4751, MUSE
       NFM) -- the dust mask and a legacy dartboard, fits with the package
       defaults, ``losvd_error_bands``, flags and hand tuning, V_sys, moment maps, radial profiles, and a mirror-symmetry
       check of the errors
