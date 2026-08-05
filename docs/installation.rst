Installation
============

.. code-block:: bash

   pip install kinextract

This installs everything needed for the core fitting engine and both its
built-in uncertainty methods (Laplace, the default, and parametric
bootstrap), including JAX for fast fitting. ``pip install kinextract``
works the same way inside a conda environment; there is no separate conda
package.

Optional extras
----------------

.. code-block:: bash

   pip install "kinextract[eso]"    # ESO archive queries (astroquery), for downloading real MUSE cubes
   pip install "kinextract[nuts]"   # NUTS/NumPyro reference uncertainty mode (simulation-based-calibration checks)
   pip install "kinextract[dev]"    # test/lint tooling
   pip install "kinextract[docs]"   # this documentation's build tooling

From source
-----------

.. code-block:: bash

   git clone https://github.com/thomas-k-waters/kinextract
   cd kinextract
   pip install -e ".[dev]"

Requires Python >= 3.10.

Running tests
-------------

.. code-block:: bash

   pip install -e ".[dev]"
   pytest
