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
     - Build your own synthetic spectrum and recover its LOSVD; all key
       outputs; ``FitConfig.describe()`` for config introspection
   * - ``02_realistic_mock_fit.ipynb``
     - A genuinely degenerate multi-template mixture, plus
       continuum-cofitting and Laplace + bootstrap uncertainty estimation
   * - ``03_emiles_templates.ipynb``
     - The full E-MILES SSP template API -- a real, dense
       stellar-population library
   * - ``04_template_calibration.ipynb``
     - Get a MUSE cube, bin it, and pick a representative template subset
   * - ``05_real_data_muse.ipynb``
     - Fitting real MUSE spectra end-to-end, across the S/N range
   * - ``06_real_data_stis.ipynb``
     - Fitting one real HST/STIS spectrum end-to-end
   * - ``07_noise_correlation.ipynb``
     - Real MUSE pixel-noise correlation and AR(1) whitening
   * - ``08_full_galaxy_analysis.ipynb``
     - Full galaxy analysis, start to finish: dartboard binning, bootstrap
       uncertainty, moment maps, radial kinematic profiles
