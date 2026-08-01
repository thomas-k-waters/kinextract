# kinextract examples

## Notebooks

A continuous pedagogical arc: basics -> increasingly complex features ->
real data -> a full galaxy analysis. Open with `jupyter lab` or
`jupyter notebook`; run each from its own directory (`examples/notebooks/`)
so relative data/template paths resolve correctly.

| Notebook | What it demonstrates |
| - | - |
| `01_basic_mock_fit.ipynb` | Build your own synthetic spectrum and recover its LOSVD; all key outputs |
| `02_realistic_mock_fit.ipynb` | A genuinely degenerate multi-template mixture, plus continuum-cofitting |
| `03_emiles_templates.ipynb` | The full E-MILES SSP template API -- a real, dense stellar-population library |
| `04_template_calibration.ipynb` | Get a MUSE cube, bin it, and pick a representative template subset |
| `05_real_data_muse.ipynb` | Fitting real MUSE spectra end-to-end, across the S/N range |
| `06_real_data_stis.ipynb` | Fitting one real HST/STIS spectrum end-to-end |
| `07_noise_correlation.ipynb` | Real MUSE pixel-noise correlation and AR(1) whitening |
| `08_full_galaxy_analysis.ipynb` | Full galaxy analysis, start to finish: dartboard binning, bootstrap uncertainty, moment maps, radial kinematic profiles |

Bundled data used by several notebooks lives in `examples/data/`; notebooks
that need a real MUSE cube either use a bundled subset or fetch one via
`kinextract.eso_archive.query_muse_phase3` (see each notebook's own setup
cell for its data mode).

```bash
pip install -e ".[dev]"
jupyter lab examples/notebooks/
```
