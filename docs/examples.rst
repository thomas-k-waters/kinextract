Example notebooks
==================

Interactive tutorials live in ``examples/notebooks/`` in the repository
(browsable directly on GitHub, or open locally with ``jupyter lab``/
``jupyter notebook`` after installing kinextract). Bundled real spectra and
templates used by several of these live alongside them in
``examples/data/``.

.. list-table::
   :header-rows: 1

   * - Notebook
     - What it demonstrates
   * - ``01_basic_mock_fit.ipynb``
     - Full pipeline on a normalized synthetic spectrum; all key outputs;
       ``FitConfig.describe()`` for config introspection
   * - ``02_realistic_mock_fit.ipynb``
     - Fitting a raw (non-normalized) spectrum with the joint
       continuum-cofitting method (``fit_continuum=True``), plus Laplace +
       bootstrap uncertainty estimation
   * - ``03_real_data_muse.ipynb``
     - Real NGC 5102 MUSE central bin -- bundled data, runs out of the box,
       plus error estimation
   * - ``04_real_data_stis.ipynb``
     - Real NGC 5102 HST/STIS inner-bin spectrum, plus error estimation
   * - ``05_recovery_validation.ipynb``
     - Measuring empirical recovery bias with ``assess_recovery_bias``/
       ``correct_recovered_losvd`` on matched mock spectra
   * - ``06_prenormalized_workflow.ipynb``
     - Manually pre-normalizing a raw spectrum with the standalone ALS
       utility (``kinextract.continuum.asymmetric_least_squares_continuum``),
       then fitting the resulting ``.norm`` spectrum with
       ``fit_continuum=False`` (joint mode off)

Supplementary notebooks (``S0_losvd_recovery_diagnostics.ipynb``,
``S1_regularization_demo.ipynb``) dig into non-obvious tool assumptions --
velocity-grid/forward-model sanity checks and the regularization
(``xlam``) bias-variance trade-off -- and are documented in
``examples/README.md``.
