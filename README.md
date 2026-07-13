# kinextract

NOTE: This package is actively being developed. Please let me know if you encounter any issues or if you have suggestions for improvement. You can reach me via email at [thomas.k.waters@gmail.com](mailto:thomas.k.waters@gmail.com).

Non-parametric binned-LOSVD spectral fitter for galaxy kinematics

`kinextract` fits galaxy spectra using a non-parametric line-of-sight velocity distribution (LOSVD) represented directly on a velocity grid — no Gauss-Hermite parametrisation assumed. When continuum co-fitting is enabled (`FitConfig.fit_continuum = True`), a penalized-B-spline continuum is folded directly into the same single optimization as the LOSVD and template weights (see `kinextract.joint`), rather than fit as a separate sub-fit. Alternatively, a spectrum can be pre-normalized once (e.g. via the standalone asymmetric-least-squares utility, `kinextract.continuum.asymmetric_least_squares_continuum`) and fit with `fit_continuum = False`. Regularization strength is selected automatically via a discrepancy-principle search (Cappellari/pPXF's `REGUL` convention, calibrated for this package's LOSVD parameterization; see References below), with legacy chi2-tolerance and roughness criteria also available. The package targets integral-field-spectroscopy data (MUSE, STIS) for black-hole mass measurements through Schwarzschild orbit modelling.

## Features

- **Non-parametric LOSVD**: recovers arbitrary shapes including double-peaked and asymmetric distributions
- **Automatic regularization**: selects `xlam` via a discrepancy-principle search (default), or legacy chi-squared/roughness grid criteria, all with a unimodality constraint
- **Joint continuum-in-the-model co-fitting** (`FitConfig.fit_continuum = True`): a penalized-B-spline continuum baseline is optimized in the same single fit as the LOSVD/template weights, fixing a real oversmoothing failure mode found when continuum estimation is a separate sub-fit with its own hyperparameter search; a self-consistent `sigl0` fixed-point iteration and `v_center` recentering further correct real velocity/dispersion recovery biases. Alternatively, pre-normalize a spectrum once (see `kinextract.continuum.asymmetric_least_squares_continuum`, a standalone utility -- not part of the main fitting pipeline) and fit with `fit_continuum = False`
- **JAX acceleration**: analytic JIT-compiled gradients, installed by default, for a large speedup and more reliable convergence over plain finite differences
- **Instrumental LSF matching**: optionally convolves the sharper of the data/templates down to match the coarser one before fitting, given both resolutions (see `FitConfig.data_fwhm_A`/`template_fwhm_A`)
- **Uncertainty estimation**: Laplace approximation (with active-set conditioning and a convergence diagnostic), residual bootstrap, and bias correction (via `LOSVDErrorEstimator`)
- **Empirical recovery-bias validation**: `assess_recovery_bias` fits matched mock spectra (same instrument, templates, continuum, and noise level as a real target) on a grid of known truths, so bias near the instrumental resolution limit can be measured directly rather than assumed
- **Self-documenting configuration**: `FitConfig.describe()` prints every tunable field, grouped by subsystem, optionally filtered by a substring (e.g. `FitConfig.describe("joint")`)

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

## Documentation

Documentation for kinextract is available at [Read the Docs](https://kinextract.readthedocs.io/en/latest/index.html).

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
from kinextract import plot_fit, plot_continuum
plot_fit(fit)
plot_continuum(fit)   # only if cfg.fit_continuum = True
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
fit_continuum = true
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

```text
src/kinextract/
    config.py      FitConfig dataclass + TOML loader
    state.py       FitState dataclass (holds spectrum, templates, LOSVD)
    continuum.py   Standalone continuum utilities (asymmetric-least-squares math)
    joint.py       Joint P-spline-continuum-in-the-model fit (the continuum-cofitting method)
    numerics.py    Objective function, JAX kernel, chi-squared, convolution
    masking.py     Emission-line masking, sigma-clipping, cleaning
    losvd.py       LOSVD analysis: roughness, peak counting, GH fitting
    spectrum.py    Spectrum loading, wavelength rebinning, FitState construction
    fitting.py     MAP fitting loop, automatic xlam selection, top-level API
    io.py          File I/O (spectra, templates, output .fit/.ascii/.rms files)
    templates.py   Template reading/interpolation, instrumental LSF matching
    plotting.py    Diagnostic plots (fit residuals, LOSVD, co-fit continuum)
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
- Continuum-cofit fits (`fit_continuum = True`) don't yet support `LOSVDErrorEstimator.laplace_covariance`/`bias_correction`
  (both raise `NotImplementedError`); `residual_bootstrap` is fully supported. Pre-normalize the spectrum
  (`fit_continuum = False`) instead if you need Laplace or bias-corrected error bars, or `assess_recovery_bias`.
- On real MUSE data, higher-order Gauss-Hermite moments (h3/h4) recovered by the joint method carry
  a small, consistent positive offset (~0.03-0.05) relative to the legacy Fortran pipeline, not yet
  root-caused (ruled out: a fixed-regularization-strength mismatch — the offset persists even at the
  legacy pipeline's own fixed `xlam` value). V and sigma do not show this issue once `xlam_auto_grid`
  is set wide enough for the data's actual error convention and `estimate_velocity_xcorr`'s
  reference-template/sub-pixel-refinement fixes are in place (both are default behavior as of this
  version).
- `joint_prenorm=True` and `joint_prenorm=False` (the shipped MAP path) are two genuinely different
  model parameterizations for pre-normalized-mode fitting (the shipped path has free `coff`/`coff2`
  continuum-offset nuisance parameters that joint's fixed-continuum-at-1.0 mode does not), not two
  implementations of the same model. Both now self-converge `sigl0`/`v_center` to a data-driven fixed
  point by default (see `kinextract.fitting._fit_map_sigl0_recenter` /
  `kinextract.joint.fit_joint_auto_xlam_sigl0`) — this fixed a real bug where the shipped path never
  did so at all (badly non-converged fits, chi2_red far from 1, were possible at some regularization
  settings) — but even with both self-converged, they can still land on different fixed points on the
  same spectrum (confirmed on real MUSE data: sigma agreed to ~3 km/s, V differed by ~20 km/s, using
  the recommended `xlam_auto=True` default). Pick one method per project rather than expecting the two
  to agree; the discrepancy reflects a genuine model choice, not an outstanding bug in either path.

## References

Methods and conventions `kinextract` builds on or reimplements from the literature
(all original code -- these are citations to published methods/formulas, not
copied source):

- Cappellari, M. & Emsellem, E. 2004, PASP, 116, 138 — pPXF, the penalized
  pixel-fitting method this package's overall approach (and several
  conventions below) draws on.
- Cappellari, M. 2017, MNRAS, 466, 798 — in particular Sec. 3.5's `REGUL`
  regularization-strength procedure (the "discrepancy principle": increase
  regularization until chi2 rises by `sqrt(2*N_good)` above the unregularized
  value), adapted as `FitConfig.xlam_criterion = "discrepancy"`
  (`kinextract.fitting._discrepancy_principle_search`).
- van der Marel, R. P. & Franx, M. 1993, ApJ, 407, 525 — the Gauss-Hermite
  polynomial normalization convention used throughout `kinextract.losvd`
  (matching pPXF's own convention).
- Eilers, P. H. C. 2003, Analytical Chemistry, 75, 3631 — asymmetric
  least-squares (ALS) baseline fitting
  (`kinextract.continuum.asymmetric_least_squares_continuum`), retained as a
  standalone, one-time pre-normalization utility.
- Eilers, P. H. C. & Marx, B. D. 1996, Statistical Science, 11, 89 —
  penalized B-splines ("P-splines"), the continuum basis used by the joint
  continuum-in-the-model fit (`kinextract.joint`).

The non-parametric LOSVD-fitting core this package ports to Python is
inherited from an original Fortran pipeline (Gebhardt & Waters) predating
this package; see `kinextract.io`'s module docstring.

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
