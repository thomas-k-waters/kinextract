# kinextract

NOTE: This package is actively being developed. Please let me know if you encounter any issues or if you have suggestions for improvement. You can reach me via email at thomas.k.waters@gmail.com.

**Non-parametric binned-LOSVD spectral fitter for galaxy kinematics**

`kinextract` fits galaxy spectra using a non-parametric line-of-sight velocity distribution (LOSVD) represented directly on a velocity grid — no Gauss-Hermite parametrisation assumed. An asymmetric-least-squares (ALS) baseline is co-fitted with the LOSVD, and regularization strength is selected automatically via a roughness criterion. The package targets integral-field-spectroscopy data (MUSE, STIS) for black-hole mass measurements through Schwarzschild orbit modelling.

## Features

- **Non-parametric LOSVD**: recovers arbitrary shapes including double-peaked and asymmetric distributions
- **Automatic regularization**: grid-searches `xlam` via a chi-squared or roughness + unimodality criterion
- **Continuum co-fitting**: asymmetric least-squares (ALS) baseline by default, or a low-order asymmetric-reweighted polynomial (`FitConfig.continuum_method = "polynomial"`) as a less over-smoothing-prone alternative
- **JAX acceleration**: analytic JIT-compiled gradients, installed by default, for a large speedup and more reliable convergence over plain finite differences
- **Instrumental LSF matching**: optionally convolves the sharper of the data/templates down to match the coarser one before fitting, given both resolutions (see `FitConfig.data_fwhm_A`/`template_fwhm_A`)
- **Uncertainty estimation**: Laplace approximation (with active-set conditioning and a convergence diagnostic), residual bootstrap, and bias correction (via `LOSVDErrorEstimator`)
- **Empirical recovery-bias validation**: `assess_recovery_bias` fits matched mock spectra (same instrument, templates, continuum, and noise level as a real target) on a grid of known truths, so bias near the instrumental resolution limit can be measured directly rather than assumed
- **Self-documenting configuration**: `FitConfig.describe()` prints every tunable field, grouped by subsystem, optionally filtered by a substring (e.g. `FitConfig.describe("als")`)

## Installation

```bash
pip install kinextract
```

This installs everything needed for the primary MAP + bootstrap pipeline,
including JAX and Numba for fast fitting. The optional full-posterior
(NUTS/HMC) path in `kinextract.bayesian` needs NumPyro on top:

```bash
pip install "kinextract[bayesian]"
```

`pip install kinextract` works the same way inside a conda environment;
there is no separate conda package.

From source:

```bash
git clone https://github.com/thomas-k-waters/kinextract
cd kinextract
pip install -e ".[dev]"
```

Requires Python ≥ 3.10.

## Quick start

```python
from kinextract import run_spectral_fit, load_config_from_toml

cfg = load_config_from_toml("kinextract.config")
fit = run_spectral_fit(cfg, gal_file="bin0105.spec")

# Inspect the fit
v  = fit["state"].xl           # velocity grid (km/s)
b  = fit["outputs"]["b"]       # recovered LOSVD
chi2_red = fit["outputs"]["chi2_red"]
print(f"chi2_red = {chi2_red:.3f}")

# Plot
from kinextract import plot_fit, plot_als_continuum
plot_fit(fit)
plot_als_continuum(fit)   # only if cfg.fit_als_continuum = True
```

See `FitConfig.describe()` (or `help(FitConfig)`) for every tunable field,
grouped by subsystem, e.g. `FitConfig.describe("xlam")` for just the
regularization-selection options.

## Configuration

Copy [`examples/kinextract.config`](examples/kinextract.config) to your extraction directory
and edit the galaxy-specific parameters:

```toml
[wavelength]
zgal = 0.00157811      # galaxy redshift
wavefitmin = 8400.0    # fit range (Å)
wavefitmax = 8800.0

[kinematics]
xlam_auto = true       # auto-select regularization
xlam_smooth_threshold = 0.25

[continuum]
fit_als_continuum = true
```

TOML section names are purely organizational — every field is flattened into a single
`FitConfig` namespace regardless of which section it's under. `examples/kinextract.config`
documents the most commonly-tuned fields inline; for the complete list, call
`FitConfig.describe()` (see Quick start above) or read the `FitConfig` class docstring.

## Error estimation

```python
from kinextract import LOSVDErrorEstimator, load_config_from_toml, run_spectral_fit

cfg  = load_config_from_toml("kinextract.config")
fit  = run_spectral_fit(cfg, gal_file="bin0105.spec")
est  = LOSVDErrorEstimator(fit, cfg)

laplace = est.laplace_covariance()
boot    = est.residual_bootstrap(n_bootstrap=200, n_jobs=4)
summary = est.summarize(laplace_result=laplace, bootstrap_result=boot)
est.plot_losvd_with_errors(summary)
```

## Running tests

```bash
pip install -e ".[dev]"
pytest                      # fast tests only (~2 s)
pytest -m slow              # include slow integration tests
```

## Package structure

```
src/kinextract/
    config.py      FitConfig dataclass + TOML loader
    state.py       FitState dataclass (holds spectrum, templates, LOSVD)
    continuum.py   ALS continuum fitting (standalone math + fit-time)
    numerics.py    Objective function, JAX kernel, chi-squared, convolution
    masking.py     Emission-line masking, sigma-clipping, cleaning
    losvd.py       LOSVD analysis: roughness, peak counting, GH fitting
    spectrum.py    Spectrum loading, wavelength rebinning, FitState construction
    fitting.py     MAP fitting loop, automatic xlam selection, top-level API
    io.py          File I/O (spectra, templates, output .fit/.ascii/.rms files)
    templates.py   Template reading/interpolation, instrumental LSF matching
    plotting.py    Diagnostic plots (fit residuals, LOSVD, ALS continuum)
    errors.py      LOSVDErrorEstimator (Laplace + bootstrap + bias correction)
    mocks.py       Matched mock-spectrum generation for recovery-bias validation
    validation.py  assess_recovery_bias / correct_recovered_losvd
    bayesian.py    Optional full-posterior (NUTS/HMC) alternative to the MAP pipeline
```

## Known limitations

- Input spectra and templates must share the same logarithmic wavelength grid.
- Instrumental resolution (LSF) is assumed matched between the data and the templates unless
  `FitConfig.data_fwhm_A`/`template_fwhm_A` are set, in which case the sharper of the two is
  automatically convolved down to match the coarser one before fitting (see
  `kinextract.templates.resolution_mismatch_sigma_A`). Both fields must be supplied together
  (there is no built-in database of instrument/library LSF values — you must supply your own,
  e.g. from the instrument handbook and the template library's documentation).
- LOSVD recovery is sensitive to the velocity grid and regularization choice; inspect diagnostic plots.
- At low S/N, fine-scale LOSVD structure is not reliable — check `chi2_red` and the roughness value.
- Template mismatch can bias the LOSVD; run with multiple template libraries and compare.
- Emission lines are masked but not modelled; simultaneous emission fitting is not supported.
- Near the instrumental resolution limit, recovered V/sigma carry a real, condition-dependent bias
  (see `FitConfig`'s "Known limitations" section for details and typical magnitudes). Use
  `assess_recovery_bias` to measure this directly for a specific target's instrument/S-N/template
  configuration, rather than relying on generic numbers.

## Citation

If you use `kinextract` in published research, please cite:

```bibtex
@software{kinextract,
  author  = {Waters, Thomas K.},
  title   = {kinextract: Non-parametric binned-LOSVD spectral fitter for galaxy kinematics},
  year    = {2026},
  url     = {https://github.com/thomas-k-waters/kinextract},
}
```

See also `CITATION.cff` for the full citation metadata.

## License

MIT — see [LICENSE](LICENSE).
