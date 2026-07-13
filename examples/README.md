# kinextract examples

## Notebooks

Interactive tutorials covering the full workflow.
Open with `jupyter lab` or `jupyter notebook`.

| Notebook | What it demonstrates |
| - | - |
| `01_basic_mock_fit.ipynb` | Full pipeline on a normalized synthetic spectrum; all key outputs; `FitConfig.describe()` for config introspection |
| `02_realistic_mock_fit.ipynb` | Fitting a raw (non-normalized) spectrum with the joint continuum-cofitting method (`fit_continuum=True`), plus Laplace + bootstrap uncertainty estimation |
| `03_real_data_muse.ipynb` | Real NGC 5102 MUSE central bin — bundled data, runs out of the box, plus error estimation |

Plot styling is applied automatically by `kinextract.plotting` (no separate
style file to load), so no setup beyond installing `kinextract` is needed;
run notebooks with their own directory as the working directory (the default
when opening from `jupyter lab`/`jupyter notebook` in-place), since data/
template paths inside each notebook are relative to it.

```bash
pip install kinextract   # includes JAX + Numba for fast fitting by default
```

## Bundled data (`data/`)

Real spectra and stellar templates for notebooks 02 (template normalization
demo) and 03.

```text
data/
  muse/
    bin0105sp.spec   spaxel on the galactic center — VLT/MUSE WFM, NGC 5102 (notebook 03)
    bin0105sp.norm   the same spaxel, already continuum-normalized
    <template>.dat   one or more template spectra (any number supported)
    Tlist            template list file: one template filename per line
```

`Tlist` is just a plain-text list of template filenames (resolved relative to
`template_dir`) — kinextract places no constraint on the template count or which
library they come from, similar in spirit to how pPXF consumes a template set. The
exact templates bundled for these examples may change over time; check each
directory's `Tlist` for the current contents rather than relying on this table.

## Config file template (`kinextract.config`)

A fully-commented TOML template for CLI/scripted use (`python -m kinextract
spectrum.spec kinextract.config`) — see the top-level README's Configuration
section for how it's loaded from Python.
