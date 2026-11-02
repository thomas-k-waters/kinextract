# kinextract

A non-parametric LOSVD (line-of-sight velocity distribution) extractor
for galaxy spectroscopy.

`kinextract` recovers the LOSVD directly from an absorption-line
galaxy spectrum as a smoothly-positive, low-dimensional log-density
spline -- without assuming a Gauss-Hermite (or any other fixed)
functional form for the underlying kinematics. Positivity is structural
(built into the representation via an `exp` link), not an afterthought
enforced after the fact, and the number of free parameters is chosen to
match how much velocity information a spectrum actually contains, not
fixed at an arbitrarily fine grid.

## Example notebooks

`examples/notebooks/` walks from simple synthetic data to real data, in
order:

- **`01_basic_mock_fit.ipynb`** -- foundations: the spectral model
  equation, the LOSVD's B-spline/`exp`-link parameterization, and the
  core LOSVD + template-mixture recovery machinery on a synthetic mock
  built from real E-MILES templates *you choose* (edit `TEMPLATE_RECIPE`:
  any number of templates, any ages/metallicities/weights -- no separate
  single-template-vs-mixture code path), no external files needed. Stops
  at the point estimate.
- **`02_realistic_mock_fit.ipynb`** -- picks up where 01 left off: a
  genuinely degenerate 4-template mixture, what the regularization costs
  and a Cramer-Rao-bound noise-floor analysis for `h4`, the `lam` rule
  for broad LOSVDs, Laplace uncertainty, and the co-fitted continuum --
  including why a recovered continuum's *shape* is trustworthy but its
  absolute *level* is not identifiable (a real, root-caused property of
  the model, not a bug).
- **`03_template_mismatch.ipynb`** -- fits the *same* mock spectrum with
  a template library that actually contains the true stellar population,
  and with one that systematically doesn't, and compares the recovered
  LOSVD directly -- motivates `04_template_calibration.ipynb`.
- **`04_template_calibration.ipynb`** -- run this first for any real MUSE
  target. Gets a MUSE datacube (a real ESO archive download -- here the
  dust-free half of NGC 4751's NFM cube -- a synthetic Sersic-profile mock
  with known ground truth, or shipped pre-computed products), bins it in
  radial annuli, fits every annulus against the full 636-template E-MILES
  grid, and iteratively reduces to a representative template subset
  (converges when the *union* of surviving templates across bins
  stabilizes) -- saved for reuse in `07_full_galaxy_analysis.ipynb`.
- **`05_noise_correlation.ipynb`** -- measures real MUSE pixel-noise
  correlation (AR(1) lag-1 correlation length) on a real spectrum,
  connects it to the Bacon et al. (2017) MUSE variance-underestimate
  factor, shows correlated-noise whitening (`FitConfig.rho`,
  `measure_and_whiten`) applied to a real fit and its uncertainty band,
  and confirms the whole mechanism on a synthetic mock with known,
  injected correlated noise. Section 8 adds the systematic residuals that
  dominate at high S/N (sky lines, template mismatch): the
  `noise_model_errors` variance model and `losvd_error_bands`.
- **`06_other_instruments.ipynb`** -- adapting this package to a
  different instrument (HST/STIS): follows the same 01 -> 02 -> real-fit
  progression, adapted to STIS -- a basic mock and a continuum-distorted
  mock built entirely from real STIS-observed stellar templates (no
  LSF-matching step needed -- templates and data share the same
  instrument), then a real HST/STIS NGC 5102 spectrum fit end-to-end --
  demonstrates this package's workflow is not MUSE-specific and shows how
  to adapt it to your own instrument.
- **`07_full_galaxy_analysis.ipynb`** -- the capstone: a complete
  extraction of a real cube (NGC 4751, MUSE NFM), from masks to maps.
  It runs:
  - masks -- the dusty half of the galaxy and defective spaxels -- and a
    legacy polar dartboard;
  - fits with the package defaults;
  - `losvd_error_bands_parallel`;
  - flags and hand tuning (`LAM_OVERRIDE`, `REVIEW`) and example fits;
  - V_sys, moment maps and radial profiles;
  - a mirror-symmetry check of the errors (bin against its mirror bin).

  It runs from shipped results by default, or from the cube with
  `FORCE_RECOMPUTE = True`.

## Features

The defaults are the recipe validated on NGC 4751 MUSE data (480 matched
mocks and the real bins; `dev_notes/regularization_v2/FINDINGS.md`): a
bare `FitConfig()` and `fit_losvd_robust` reproduce that extraction
exactly, and `losvd_error_bands` its errors.

- **LOSVD recovery** (`kinextract.losvd`, `kinextract.config`,
  `kinextract.multi_template`): one fitting engine (`fit_losvd`)
  recovers the LOSVD -- and, for more than one template, the mixture
  weights jointly via alternating minimization (exact NNLS for the
  weights, a multi-start exact-Hessian Newton MAP fit for the spline
  coefficients) -- for any `n_templates >= 1`. See `losvd.py`'s and
  `multi_template.py`'s module docstrings for the method, and
  `examples/notebooks/01_basic_mock_fit.ipynb` for a worked example with
  a configurable template recipe.
- **Regularization** (`FitConfig.lam`, `FitConfig.lam_rule`,
  `sigma_scaled_lam`, `with_lam_rule`): a third-difference roughness
  penalty on the log-LOSVD spline (its null space is any Gaussian, so
  heavy smoothing shrinks toward the best-fitting Gaussian, never toward
  fat tails) at `lam = 4096`. LOSVDs broader than 350 km/s are refitted at
  `4096 (sigma/350)^6` (capped at 131072), and fits still at the edge of
  the velocity grid (`edge_amplitude_frac`) at 2x, then a further 4x,
  their `lam`. On the mocks no `lam` selector beat this rule by more than
  a few per cent; adaptive selection failed on the real spectra.
- **Continuum co-fitting** (`kinextract.continuum`, on by default with 4
  knots): a shared, structurally-positive multiplicative continuum
  (`C(lambda) = exp(spline(lambda))`), jointly recovered with the LOSVD
  and template weights via 3-block alternating minimization. Validated on
  synthetic data, including the adversarial "does the continuum
  contaminate the weight recovery" check -- see `continuum.py`'s own
  docstring and `02_realistic_mock_fit.ipynb`/`06_other_instruments.ipynb`.
  The recovered continuum's *shape* is reliable; its absolute *mean* is
  not identifiable (`C`'s mean and the template weights' overall scale
  are the same degree of freedom -- `dev_notes/continuum/FINDINGS.md`,
  2026-08-01).
- **Robust fitting** (`kinextract.robust_fit`): `fit_losvd_robust`
  inflates the per-pixel errors wherever a preliminary fit's residuals
  call for it (sky-subtraction residuals, bad pixels; smoothed over 3 A,
  capped at 100x), with a loose cleanup clip for single-pixel outliers
  and the CaII triplet cores protected from clipping. Fixed bad-pixel
  ranges can be masked beforehand (`mask_wavelength_windows`), and
  emission lines detected (`detect_emission_lines`).
- **Error bands** (`kinextract.laplace_uncertainty`,
  `kinextract.noise_model`): the Laplace (curvature) approximation to the
  posterior at the fit's `lam`, jointly over the LOSVD and continuum
  coefficients (`laplace_losvd_uncertainty`). For real IFU data,
  `losvd_error_bands` first fits a per-pixel variance
  `err^2 + (kappa*s)^2 + (f*model)^2` -- `s` the LSF-blended sky-line
  spectrum (`sky_line_profile`, `load_muse_sky_lines`), `f` a
  template-mismatch term -- to a preliminary fit's residuals, runs
  Laplace with it at the fit's `lam` and `rho` without its clipped
  pixels, and moves the draws onto the fit (`recenter_samples`). On the
  mocks this kept the shapes unchanged and raised the coverage of the
  model-binned LOSVD under sky contamination from 0.48 to 0.78. An
  optional NUTS (NumPyro) reference mode (`kinextract.nuts_uncertainty`,
  the `nuts` extra) samples the exact posterior, for spot-checks.
- **LOSVD moment estimation** (`kinextract.gh_moments`): summarizes a
  recovered LOSVD as (V, sigma, h3, h4) via a smooth-taper direct moment
  for V and sigma and a fixed-center Gauss-Hermite fit for h3 and h4; the
  joint Gauss-Hermite fit (`fit_gauss_hermite`) is kept for comparison
  with the wider dynamics literature. On fits at the defaults the two
  agree to about 0.5 km/s on the validation grid.
- **Noise-correlation whitening** (`kinextract.noise_model`): AR(1)
  pixel-covariance whitening (`whiten`, `FitConfig.rho`), a lag-1 ACF
  estimator (`estimate_correlation_length`), `measure_and_whiten`, and
  per-bin measurement inside the robust fit
  (`fit_losvd_robust(..., measure_rho=True)`) -- see
  `05_noise_correlation.ipynb`.
- **IFU cubes** (`kinextract.muse_cube`, `kinextract.binning`,
  `kinextract.template_reduction`): MUSE cube I/O, robust per-bin spectrum
  combination with spatial-correlation-aware errors, dartboard binning
  (including the legacy SCO `bin_r.out`/`bin_v.out` geometry), and
  iterative template-subset reduction.
- **Schwarzschild-model inputs** (`kinextract.legacy_io`):
  `write_spmc_in_laplace`/`transvd_bounds_laplace` write SCO `spmc_in`
  files straight from Laplace/NUTS percentiles. With
  `b_samples=`/`model_bins=` each dynamical-model velocity bin carries at
  least its binned posterior scatter: `model.x` sums per-point errors
  linearly, which understated it by about 2x.

Not yet implemented: the template-continuum-trust hierarchy (library-
keyed presets for how much of a raw template's own continuum to trust --
see `continuum.py`'s docstring), wavelength-dependent LSF matching, an
end-to-end spectrum-file I/O and orchestration layer.

## Known limitations

- **The penalty shrinks genuine non-Gaussian structure toward a
  Gaussian.** h3 and h4 are attenuated wherever the data cannot hold
  them up -- at low S/N, and for h3 across a wide low-to-moderate sigma
  range, where recovered h3 stays close to 0. Much of this is the data's
  own limit (a Cramer-Rao analysis in `02_realistic_mock_fit.ipynb`), but
  the error bands describe noise, not this shrinkage.
- **Plain Laplace bands under-cover on real data, and the systematics
  model only partly fixes it.** With `losvd_error_bands` the coverage of
  sky-contaminated mocks is about right (0.78 for a 0.68 target, slightly
  conservative). On clean, high-S/N mocks built from different templates
  than the fit it is still about 0.42: the remaining error is
  template-mismatch bias in the point estimate, which no band width
  fixes.
- **Real chi2_red on real IFU data commonly runs well above 1** (double
  digits at high S/N) -- systematics-dominated residuals (sky-line
  residuals, template mismatch, correlated per-pixel noise), not just
  photon noise. The robust fit's error inflation and the noise-model
  bands exist for this; see `05_noise_correlation.ipynb` and
  `07_full_galaxy_analysis.ipynb`.
- **Below roughly the instrument's own resolution (MUSE: sigma ~36-38
  km/s in the CaT window), recovered sigma depends on `lam` with no
  plateau**, and the reported uncertainty, a spread around the fit at its
  `lam`, does not cover that dependence. Confirmed on real NGC 5102 bins:
  sigma varies by 9-16 km/s across a 64-65536 `lam` ladder against a
  Laplace `sherm_std` of 1-2 km/s (`dev_notes/uncertainty/FINDINGS.md`,
  2026-08-11). Not resolved.
- **Multi-template weight recovery is sensitive to template-library
  conditioning** -- a near-collinear template library gives genuinely
  unstable weight recovery across noise realizations, diagnosable via the
  condition number of the per-step NNLS design matrix. See
  `01_basic_mock_fit.ipynb`, `02_realistic_mock_fit.ipynb` and
  `03_template_mismatch.ipynb`. Degrades gracefully (sparser/noisier
  weight recovery, never a crash or NaN); mitigated by template
  conditioning (`04_template_calibration.ipynb`).
- **JAX runs in float64 by default** (opt out with `KINEXTRACT_NO_X64=1`).
  This is a correctness requirement: under float32 the objective in the
  stiff heavy-penalty regime of wide velocity grids stalls the optimizer
  far from the optimum, reporting `success=True` regardless.
- **Double-peaked/multi-component LOSVDs are not specifically handled.**
  The non-parametric spline recovery can represent them (unlike a fixed
  Gauss-Hermite parametrization), but no notebook in this sequence
  specifically validates that case.

## Installation

```
pip install -e ".[dev]"
```

Optional extras: `.[eso]` (ESO archive downloads), `.[nuts]` (the NUTS/
NumPyro reference uncertainty mode), `.[docs]` (building the docs),
`.[examples]` (the colormaps of the example notebooks' moment maps).

## Testing

```
pytest tests/
```

275 tests covering LOSVD recovery and the `lam` rule, the Newton
solver, moment estimation, multi-template fitting, robust fitting with
error inflation (serial and parallel), continuum co-fitting, Laplace
uncertainty and the noise-model error bands (serial and parallel), NUTS
reference-mode sampling, AR(1) correlated-noise whitening, wavelength
masking, spatial binning and template reduction, legacy I/O, and
input-validation regressions. These are unit/sanity tests (correct shapes,
right order of magnitude, no crashes, and several real before/after
demonstrations against known synthetic ground truth) -- the full,
wide-grid statistical validation is in `dev_notes/VALIDATION_CAMPAIGN.md`,
`dev_notes/validation_suite/` and `dev_notes/regularization_v2/`.

`A_w`/`y_w` (and `A_stack_w`) must not contain NaN or inf -- mask bad
pixels out of the arrays before calling any fit function; a clear
`ValueError` is raised otherwise rather than a fit silently returning a
NaN-filled LOSVD.

A handful of tests are marked `@pytest.mark.slow` (genuinely
multi-minute) and are skipped by default, both via `-m "not slow"` (this
project's own CI path) and by a bundled pytest-astropy-ecosystem plugin
that requires `--run-slow` to execute any `slow`-marked test at all, even
under a plain `pytest tests/` with no `-m` filter. Pass `--run-slow` to
actually run them, e.g. before a release: `pytest tests/ --run-slow`.

## Quick start

One fitting engine, for any number of templates (a single template is
just `template_fluxes = [template_flux]`):

```python
from kinextract import FitConfig, build_template_stack, compute_losvd_moments, fit_losvd_robust

cfg = FitConfig(vmin=-1500.0, vmax=1500.0)   # the velocity grid, about V +- 5-7 sigma; FitConfig.describe() lists the rest
v_grid = cfg.build_velocity_grid()
A_stack = build_template_stack(wave, template_wave, template_fluxes, v_grid)  # (n_templates, n_pix, n_bins)
fit = fit_losvd_robust(A_stack, flux, err, cfg, wave_fit=wave)
moments = compute_losvd_moments(v_grid, fit["b"])          # vherm, sherm, h3, h4
mixture_fractions = fit["w"] / fit["w"].sum()              # meaningful for n_templates > 1
```

Error bands, with the sky-line + template-mismatch noise model (`sky_profile=None` without sky residuals):

```python
from kinextract import load_muse_sky_lines, losvd_error_bands, sky_line_profile

sky = sky_line_profile(wave, *load_muse_sky_lines("MU_GSKL_140101A_sky_lines.fits"), z=z)
bands = losvd_error_bands(A_stack, flux, err, wave, fit, cfg, sky_profile=sky)
print(f"sigma = {moments['sherm']:.1f} +/- {bands['sherm_err']:.1f}")
```

Run `FitConfig.describe()` to see every tunable parameter grouped by
subsystem. `docs/quickstart.rst` has more, and `examples/notebooks/` has
complete, runnable, annotated examples.
