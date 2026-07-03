# kinextract examples

## Notebooks

Interactive tutorials covering the full workflow.
Open with `jupyter lab` or `jupyter notebook`.

| Notebook | What it demonstrates |
| - | - |
| `01_basic_mock_fit.ipynb` | Full pipeline on a normalized synthetic spectrum; all key outputs |
| `02_realistic_mock_fit.ipynb` | Fitting a raw (non-normalized) spectrum using the ALS continuum fitter, plus Laplace + bootstrap uncertainty estimation |
| `03_real_data_muse.ipynb` | Real NGC 5102 MUSE central bin — bundled data, runs out of the box, plus error estimation |
| `04_real_data_stis.ipynb` | Real NGC 5102 HST/STIS inner-bin spectrum, plus error estimation |
| `05_real_data_muse_n4751.ipynb` | Real NGC 4751 MUSE NFM central bin — a **high velocity dispersion** ($\sigma \sim 350$ km/s) example; demonstrates why broad LOSVDs need much stronger/scale-invariant regularization than notebook 03's case |

### Supplementary

| Notebook | What it demonstrates |
| - | - |
| `S0_losvd_recovery_diagnostics.ipynb` | Diagnosing LOSVD recovery quality: velocity grid, forward-model accuracy, regularization bias, emission masking, S/N and true-V sweeps |
| `S1_regularization_demo.ipynb` | How `xlam` affects LOSVD smoothness; the chi² auto-selection criterion |

All notebooks load a shared plotting style (`kinextract.mplstyle`) via
`plt.style.use('kinextract.mplstyle')` in their first code cell, so run them
with the notebook's own directory as the working directory (the default when
opening from `jupyter lab`/`jupyter notebook` in-place).

Install the optional speed dependencies first for best performance:

```bash
pip install "kinextract[fast]"   # adds JAX + Numba
```

## Bundled data (`data/`)

Real spectra and stellar templates for notebooks 02 (template normalization
demo), 03, 04, and 05.

```text
data/
  muse/
    bin0105sp.spec   central Voronoi bin — VLT/MUSE WFM, NGC 5102 (notebooks 02, 03)
    <template>.dat   one or more template spectra (any number supported)
    Tlist            template list file: one template filename per line
  muse_n4751/
    bin0105sp.spec   central Voronoi bin — VLT/MUSE NFM, NGC 4751 (notebook 05)
                     reuses the templates/Tlist from muse/ above; no separate set
  stis/
    bin0105sp.spec   inner-bin spectrum — HST/STIS G750L
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
