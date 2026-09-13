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

## Status

**v1.0.0** -- a complete, from-scratch rebuild of the fitting engine,
uncertainty quantification, and real-data pipeline, independently
validated end to end before being merged back in (27-stage validation
campaign, `dev_notes/VALIDATION_CAMPAIGN.md`). The default uncertainty
engine (Laplace + AIC-weighted) was validated via a production-scale
coverage sweep across sigma 50-600 km/s, matching or beating parametric
bootstrap's own coverage in every regime tested. Marked `Beta` (not
`Stable`) because it's still actively developed and because several
specific, real limitations are tracked openly below rather than hidden --
not because the validated parts are shaky.

The core fitting, uncertainty (Laplace and bootstrap), multi-template,
cleaning, and continuum machinery is implemented and validated on
synthetic data, and has also been run end-to-end on real VLT/MUSE and
HST/STIS spectra (see `examples/notebooks/05_noise_correlation.ipynb`,
`06_other_instruments.ipynb`, and `07_full_galaxy_analysis.ipynb` for a
full-galaxy analysis). **Real-data-specific finding worth knowing before
trusting a real fit**: real chi2_red on real IFU data commonly runs well
above 1 (systematics-dominated residuals -- template mismatch, unmodeled
continuum shape, correlated per-pixel noise -- not just photon noise),
which affects how much regularization gets selected and how well a
sigma-clipping/outlier-rejection scheme behaves; see
`05_noise_correlation.ipynb` and `07_full_galaxy_analysis.ipynb`'s own
Limitations section for the current, honest picture. Treat every
synthetic validation result in this package as solid, and every real-data
number as requiring the same care those notebooks show.

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
  genuinely degenerate 4-template mixture, `penalty_order`/`lam_wing`
  regularization trade-offs, a Cramer-Rao-bound noise-floor analysis for
  `h4`, this package's two uncertainty engines (Laplace, the default, and
  bootstrap as a cross-check), and continuum-cofitting -- including why
  a recovered continuum's *shape* is trustworthy but its absolute *level*
  is not identifiable (a real, root-caused property of the model, not a
  bug).
- **`03_template_mismatch.ipynb`** -- fits the *same* mock spectrum with
  a template library that actually contains the true stellar population,
  and with one that systematically doesn't, and compares the recovered
  LOSVD directly -- motivates `04_template_calibration.ipynb`.
- **`04_template_calibration.ipynb`** -- run this first for any real MUSE
  target. Gets a MUSE datacube (a real ESO archive download, a synthetic
  Sersic-profile mock with known ground truth, or shipped pre-computed
  products), bins it with the same PA-aware radial x angular scheme
  `07_full_galaxy_analysis.ipynb` uses for its real kinematic maps, fits
  every bin against the full 636-template E-MILES grid, and iteratively
  reduces to a representative template subset (converges when the
  *union* of surviving templates across bins stabilizes) -- saved for
  reuse in `07_full_galaxy_analysis.ipynb`.
- **`05_noise_correlation.ipynb`** -- measures real MUSE pixel-noise
  correlation (AR(1) lag-1 correlation length) on a real spectrum,
  connects it to the Bacon et al. (2017) MUSE variance-underestimate
  factor, shows correlated-noise whitening (`FitConfig.rho`,
  `measure_and_whiten`) applied to a real fit and its uncertainty band,
  and confirms the whole mechanism on a synthetic mock with known,
  injected correlated noise.
- **`06_other_instruments.ipynb`** -- adapting this package to a
  different instrument (HST/STIS): follows the same 01 -> 02 -> real-fit
  progression, adapted to STIS -- a basic mock and a continuum-cofitting
  mock built entirely from real STIS-observed stellar templates (no
  LSF-matching step needed -- templates and data share the same
  instrument), then a real HST/STIS NGC 5102 spectrum fit end-to-end --
  demonstrates this package's workflow is not MUSE-specific and shows how
  to adapt it to your own instrument.
- **`07_full_galaxy_analysis.ipynb`** -- the capstone: a full real-galaxy
  analysis start to finish -- PA-aware dartboard binning, per-bin
  masking/cleaning/fitting/Laplace uncertainty in parallel, moment maps,
  and radial kinematic profiles along the major axis.

## Features

- **LOSVD recovery** (`kinextract.losvd`, `kinextract.config`,
  `kinextract.multi_template`): one fitting engine (`fit_losvd`)
  recovers the LOSVD -- and, for more than one template, the mixture
  weights jointly via alternating minimization (exact NNLS for the
  weights, the same multi-start MAP fit for the spline coefficients) --
  for any `n_templates >= 1`. See `losvd.py`'s and `multi_template.py`'s
  module docstrings for the method, and
  `examples/notebooks/01_basic_mock_fit.ipynb` for a worked example with
  a configurable template recipe (any number of templates, one mechanism),
  including a template-conditioning diagnostic. Includes an opt-in wing ramp on the
  roughness penalty (`FitConfig.lam_wing`) -- see Known limitations.
- **Uncertainty quantification** (`kinextract.laplace_uncertainty`,
  `kinextract.uncertainty`): two engines. `laplace_losvd_uncertainty`
  (and its per-bin-parallel counterpart, `laplace_losvd_uncertainty_
  parallel`) is the default -- a Laplace (curvature) approximation to the
  posterior at each candidate regularization strength, combined via
  AIC-weighted posterior-predictive sampling; measured directly to give
  substantially better h4 coverage than bootstrap (39% vs. 4% against a
  68% target, production-scale coverage sweep at sigma 50-150 km/s) at a
  fraction of the runtime, though still below the target especially at
  low sigma -- see `dev_notes/uncertainty/FINDINGS.md`, which also
  breaks the residual gap down into a genuine point-estimate-bias
  component and a specific, identified sampling artifact. A separate
  sweep extending through sigma 250-600 km/s (real black-hole-mass
  target regime, e.g. NGC 4751) confirms the same pattern holds well
  beyond the original validation range: sigma coverage stays on-target
  for both engines, and Laplace's h4 advantage is if anything more
  decisive there (50-83% vs. 0-53%). `bootstrap_losvd`/`bootstrap_losvd_
  parallel` (parametric bootstrap, any `n_templates >= 1`, including
  mixture-weight uncertainty) remain available as a cross-check -- real
  ensemble properties (empirical skew/multimodality, basin-selection
  risk) Laplace structurally cannot give. An optional NUTS (NumPyro)
  reference mode (`kinextract.nuts_uncertainty`, the `nuts` extra) is
  available for spot-checking the Laplace approximation itself.
  `bootstrap_losvd` also supports an opt-in `calibrate_noise_scale`/
  `noise_scale` mechanism (not yet available on the Laplace engine) for
  when the assumed per-pixel noise itself is suspected wrong (validated
  on synthetic uniform noise mis-specification; **not yet safe on real
  data**, see Known limitations). See `laplace_uncertainty.py`'s and
  `uncertainty.py`'s own module docstrings for the full, nuanced coverage
  picture.
- **Continuum co-fitting** (`kinextract.continuum`): a shared,
  structurally-positive multiplicative continuum
  (`C(lambda) = exp(spline(lambda))`), jointly recovered with the LOSVD
  and template weights via 3-block alternating minimization. Cleanly
  validated on synthetic data, including the key "does the continuum
  contaminate the weight recovery" adversarial check -- see
  `continuum.py`'s own docstring, and `02_realistic_mock_fit.ipynb`/
  `06_other_instruments.ipynb` for worked demonstrations. The recovered
  continuum's *shape* is reliable; its absolute *mean* is not
  identifiable from the model alone (a real, root-caused mathematical
  property -- `C`'s mean and the template weights' overall scale are the
  same degree of freedom -- not a bug to fix, see `dev_notes/continuum/
  FINDINGS.md`'s 2026-08-01 entry). `penalty_order=3` with a fixed `lam`
  in the 4096-32768 range is the validated default whenever
  `continuum_knots` is set. **Tried on real data and found not to
  help** (see Status above) -- kept because the mechanism itself is
  solid, not because it's currently recommended for this dataset.
- **Outlier rejection** (`kinextract.forward_model.
  mask_wavelength_windows`, `kinextract.robust_fit`): fixed-wavelength
  masking for known bad-pixel ranges, and iterative sigma-clipping
  (`fit_losvd_robust(cleaning_method="clip")`) for contamination whose
  location isn't known in advance. The clip threshold can be measured
  against a robust (MAD-based) estimate of the residual's own scale
  (`robust_scale=True`) rather than the raw input noise vector -- matters
  on real data, where the input noise vector commonly underestimates the
  true residual scatter (see the chi2_red note above); confirmed to
  behave far better than the raw-threshold version on real MUSE bins that
  otherwise clip runaway. See `07_full_galaxy_analysis.ipynb` for this
  used as the default per-bin cleaning step, and its own Limitations
  section for what it still doesn't handle (continuum-*shape* mismatch,
  in particular).
- **LOSVD moment estimation** (`kinextract.gh_moments`): summarizes
  a recovered LOSVD as (V, sigma, h3, h4) via a direct-moment estimator
  that beats a full Gauss-Hermite refit on this package's own validation
  grid; the joint Gauss-Hermite fit is kept as a general-purpose
  alternative and for comparison against the wider dynamics literature.
- **Noise-correlation whitening** (`kinextract.noise_model`): AR(1)
  pixel-covariance whitening (`whiten`, `FitConfig.rho`), a lag-1 ACF
  correlation-length estimator (`estimate_correlation_length`), and a
  measure-and-whiten convenience function (`measure_and_whiten`) that
  does both in one call -- see `05_noise_correlation.ipynb` for the real
  measurement + consequence walkthrough this exists to support.

Not yet implemented: the template-continuum-trust hierarchy (library-
keyed presets for how much of a raw template's own continuum to trust --
see `continuum.py`'s docstring), wavelength-dependent LSF matching, real
MUSE/STIS pre-normalized data loading, an end-to-end spectrum-file I/O
and orchestration layer.

## Known limitations

- **`penalty_order=3` (third-difference roughness penalty) is the
  default**, replacing an earlier second-difference penalty. Its null
  space is any Gaussian (not a straight line in log-density), so heavy
  smoothing shrinks toward the best-fitting Gaussian rather than a
  slowly-decaying tail -- eliminates the wing-amplitude sigma inflation
  the earlier penalty suffered at low S/N. Real, structural cost in
  exchange: genuine non-Gaussian LOSVDs (nonzero h3/h4) at moderate
  sigma/S-N get their h3/h4 shrunk toward zero (`penalty_order=2` keeps a
  modest edge there instead -- see `02_realistic_mock_fit.ipynb`'s own
  regularization section). Neither is strictly better; which one you want
  depends on whether you're measuring moments or full-curve shape.
- **JAX runs in float64 by default** (opt out with `KINEXTRACT_NO_X64=1`).
  This is a correctness requirement, not a preference: under float32,
  L-BFGS-B can stall far from the optimum in the stiff heavy-penalty
  regime the third-difference penalty operates in on wide velocity grids,
  reporting `success=True` regardless.
- **Real chi2_red on real IFU data commonly runs well above 1** (double
  digits, not ~1) -- systematics-dominated residuals (template mismatch,
  unmodeled continuum shape, correlated per-pixel noise), not just photon
  noise. This affects more than the raw fit: it means a nominal "N sigma"
  threshold in any residual-based scheme (adaptive regularization
  selection, iterative cleaning) is measured against the wrong scale
  unless corrected for -- see the cleaning note below and
  `05_noise_correlation.ipynb`/`07_full_galaxy_analysis.ipynb` for the
  current, honest picture on real data.
- **Below roughly the instrument's own resolution (MUSE: sigma~36-38
  km/s in the CaT window most real notebooks here fit), recovered sigma
  is a lam-selection lottery, not a stable measurement -- for *any*
  `lam_selection`, and the reported uncertainty badly under-covers it.**
  Confirmed directly on real NGC 5102 bins (not just the synthetic
  sigma=20 mock this was first found on): sigma vs. `lam` is
  non-monotonic and has no plateau across a 64-65536 fixed-lam ladder
  (sometimes the package's own default `lam` gives that bin's *highest*
  ladder value, sometimes its lowest), `lam_selection="aic_local"` lands
  on an equally arbitrary point in the same ladder rather than a
  systematically better one, and the reported Laplace `sherm_std` (1-2
  km/s in the checked bins) is 5-10x smaller than the actual lam-to-lam
  swing (9-16 km/s) -- because the posterior is a curvature-based spread
  *around whichever lam won*, with no term for uncertainty *over which
  lam should have won*. See `dev_notes/uncertainty/FINDINGS.md`'s
  2026-08-11 (late) entry for the specific bins and numbers, and
  `07_full_galaxy_analysis.ipynb`'s own Section 7 for how this is
  flagged in-notebook. Not yet root-caused to a fix (switching
  `lam_selection` does not help -- confirmed, not assumed) -- flagged
  here rather than silently worked around.
- **Iterative sigma-clipping needs its threshold measured against the
  residual's own scale on real data, not the raw input noise vector.**
  `fit_losvd_robust(cleaning_method="clip")`'s default (`robust_scale=False`)
  compares the residual
  directly to the input per-pixel noise -- correct when that noise vector
  is well-calibrated (chi2_red ~ 1), but on real data where chi2_red
  commonly runs well into the double digits (see above), a nominal
  "3 sigma" cut is actually far less than 1 true sigma, and can clip the
  majority of a spectrum on the first pass, collapsing the fit to an
  unconstrained, biased baseline. `robust_scale=True` instead measures the
  clip threshold against a MAD-based robust estimate of the *current*
  residual's own scale (recomputed each iteration) -- confirmed
  substantially better-behaved on real bins that broke the default, and
  used as the default cleaning step in `07_full_galaxy_analysis.ipynb`.
  It still needs real per-target tuning (too aggressive a threshold risks
  clipping into genuine absorption features), and has no mechanism for a
  continuum-*shape* mismatch (as opposed to a localized outlier) -- see
  `cleaning.py`'s own docstring and `07_full_galaxy_analysis.ipynb`'s
  Limitations section.
- **The regularization-bias correction (`bias_correction.py`,
  `apply_bias_correction`) is not yet trustworthy enough to default on,
  and is not yet available for the Laplace uncertainty engine at all**
  (calibrated only against `bootstrap_losvd`'s own point estimate). A
  4-gate acceptance test (per-cell bias, held-out coverage, worst-cell
  regression, regression tests) finds sigma coverage passes but h4
  coverage does not -- h4's point estimate is severely shrunk under
  regularization and the reported uncertainty never widens to reflect
  that. `apply_bias_correction` stays `False` by default until this is
  resolved; see `dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md` and
  `dev_notes/uncertainty/FINDINGS.md`.
- **h3 recovery is severely suppressed at low-to-moderate sigma.** For
  genuinely nonzero true h3, recovered h3 stays close to 0 (essentially
  fully suppressed, not just biased) across a wide low-sigma range, only
  starting to recover at higher sigma -- a calibration table
  characterizing this (`regularization_bias_emiles_h3wing.txt`) exists,
  but the shipped bias-correction machinery does not yet consume it (its
  file format doesn't carry `h3_mean`/`h3_std` columns yet -- see
  `dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md`'s 2026-08-01
  entry). No correction attempted yet.
- **Multi-template weight recovery is sensitive to template-library
  conditioning** -- a near-collinear template library (differing mainly
  in overall depth scale, not shape) gives genuinely unstable weight
  recovery across noise realizations, diagnosable via the condition
  number of the per-step NNLS design matrix. See
  `examples/notebooks/01_basic_mock_fit.ipynb` and
  `02_realistic_mock_fit.ipynb` (a real, physically-motivated near-
  collinear example), and `03_template_mismatch.ipynb` for the related
  question of what happens when the library is wrong, not just
  degenerate. Degrades gracefully (sparser/noisier weight recovery, never
  a crash or NaN); mitigated but not eliminated by
  template-conditioning (`04_template_calibration.ipynb`'s iterative
  reduction).
- **No continuum model by default.** The core fit expects
  continuum-normalized input; continuum-cofitting (`kinextract.continuum`)
  is available but opt-in. Adaptive lambda selection supports it too
  (`aic_regularization.estimate_m_eff_local`'s joint LOSVD+continuum
  linearization), at a real per-candidate cost increase (each scanned
  `lam` costs a full 3-block alternation, not 2-block) -- its own
  validation record for the continuum case is newer and smaller than the
  no-continuum path's, so cross-check against a fixed `penalty_order=3,
  lam=4096.0` before trusting it in production.
- **Double-peaked/multi-component LOSVDs are not specifically handled.**
  The non-parametric spline recovery can represent them (unlike a fixed
  Gauss-Hermite parametrization), but no notebook in this sequence
  specifically validates that case.

## Installation

```
pip install -e ".[dev]"
```

Optional extras: `.[eso]` (ESO archive downloads), `.[nuts]` (the NUTS/
NumPyro reference uncertainty mode), `.[docs]` (building the docs).

## Testing

```
pytest tests/
```

329 tests covering LOSVD recovery, moment estimation, multi-template
fitting, Laplace and bootstrap uncertainty (single- and multi-template,
serial and parallel), NUTS reference-mode sampling, noise-scale
calibration and AR(1) correlated-noise whitening, wavelength masking,
iterative cleaning (serial and parallel, including the robust-scale
option), continuum co-fitting (including adaptive `lam_selection` under
continuum), the `edge_amplitude_frac` divergent-edge-pedestal diagnostic
and its use as an adaptive-selection guard, spatial binning and template
reduction, regularization-bias calibration, and feature-composition/
input-validation regressions. These are unit/sanity tests (correct shapes, right order of
magnitude, no crashes, and several real before/after demonstrations
against known synthetic ground truth) -- the full, wide-grid statistical
validation (bias and coverage across the entire physically-relevant
sigma/S-N/resolution range) is in `dev_notes/VALIDATION_CAMPAIGN.md` and
`dev_notes/validation_suite/`, referenced throughout the source
docstrings above.

`A_w`/`y_w` (and `A_stack_w`) must not contain NaN or inf -- mask bad
pixels out of the arrays before calling any fit function; a clear
`ValueError` is raised otherwise rather than a fit silently returning a
NaN-filled LOSVD.

A handful of tests are marked `@pytest.mark.slow` (genuinely
multi-minute -- e.g. a `lam_wing`-active fit refit inside every bootstrap
replicate) and are skipped by default, both via `-m "not slow"` (this
project's own CI path) and by a bundled pytest-astropy-ecosystem plugin
that requires `--run-slow` to execute any `slow`-marked test at all, even
under a plain `pytest tests/` with no `-m` filter. Pass `--run-slow` to
actually run them, e.g. before a release: `pytest tests/ --run-slow`.

## Quick start

One fitting engine, for any number of templates (a single template is
just `template_fluxes = [template_flux]`):

```python
from kinextract import FitConfig, build_template_stack
from kinextract.multi_template import fit_losvd

cfg = FitConfig()          # see FitConfig.describe() for every tunable parameter
v_grid = cfg.build_velocity_grid()
A_stack = build_template_stack(wave_obs, template_wave, template_fluxes, v_grid)  # (n_templates, n_pix, n_bins)
A_stack_w, y_w = A_stack / err[None, :, None], flux / err   # noise-whiten
result = fit_losvd(A_stack_w, y_w, cfg)
recovered_losvd = result["b"]
mixture_fractions = result["w"] / result["w"].sum()  # meaningful for n_templates > 1
```

Run `FitConfig.describe()` to see every tunable parameter grouped by
subsystem, or `FitConfig.describe("lam")` to filter by a substring. See
`examples/notebooks/` for complete, runnable, annotated examples
including uncertainty quantification and diagnostic plots.
