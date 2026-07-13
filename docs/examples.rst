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
