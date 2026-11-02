# kinextract examples

## Notebooks

A continuous pedagogical arc: basics -> increasingly complex features ->
real data -> a full galaxy analysis. Open with `jupyter lab` or
`jupyter notebook`; run each from its own directory (`examples/notebooks/`)
so relative data/template paths resolve correctly.

| Notebook | What it demonstrates |
| - | - |
| `01_basic_mock_fit.ipynb` | The spectral model and LOSVD parametrization; recover the LOSVD and template mixture of a synthetic spectrum built from E-MILES templates you choose |
| `02_realistic_mock_fit.ipynb` | A degenerate multi-template mixture, what the regularization costs and the h4 noise floor, the `lam` rule for broad LOSVDs (Section 5b), Laplace uncertainty, the co-fitted continuum |
| `03_template_mismatch.ipynb` | The same mock fitted with a matched and a mismatched template library |
| `04_template_calibration.ipynb` | Get a MUSE cube (NGC 4751 NFM, its dust-free half), bin it, and reduce the E-MILES grid to a representative template subset |
| `05_noise_correlation.ipynb` | Real MUSE pixel-noise correlation and AR(1) whitening; the sky-line + template-mismatch noise model and `losvd_error_bands` (Section 8) |
| `06_other_instruments.ipynb` | Adapting kinextract to HST/STIS: mocks and a real STIS spectrum |
| `07_full_galaxy_analysis.ipynb` | A complete extraction of a real cube (NGC 4751, MUSE NFM): the dust mask and a legacy dartboard, fits with the package defaults, `losvd_error_bands`, flags and hand tuning, V_sys, moment maps, radial profiles, a mirror-symmetry check of the errors; runs from shipped results by default |

Bundled data used by several notebooks lives in `examples/data/`; notebooks
that need a real MUSE cube either use a bundled subset or fetch one via
`kinextract.eso_archive.query_muse_phase3` (see each notebook's own setup
cell for its data mode).

```bash
pip install -e ".[dev,examples]"
jupyter lab examples/notebooks/
```
