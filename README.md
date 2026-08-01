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

Beta-quality, actively developed research code. The core fitting,
uncertainty, multi-template, cleaning, and continuum machinery is
implemented and validated on synthetic data, and has also been run
end-to-end on real VLT/MUSE and HST/STIS spectra (see
`examples/notebooks/05_real_data_muse.ipynb`,
`06_real_data_stis.ipynb`, and `08_full_galaxy_analysis.ipynb` for a
full-galaxy analysis). **Real-data-specific finding worth knowing before
trusting a real fit**: real chi2_red on real IFU data commonly runs well
above 1 (systematics-dominated residuals -- template mismatch, unmodeled
continuum shape, correlated per-pixel noise -- not just photon noise),
which affects how much regularization gets selected and how well a
sigma-clipping/outlier-rejection scheme behaves; see
`07_noise_correlation.ipynb` and `08_full_galaxy_analysis.ipynb`'s own
Limitations section for the current, honest picture. Treat every
synthetic validation result in this package as solid, and every real-data
number as requiring the same care those notebooks show.

## Example notebooks

`examples/notebooks/` walks from simple synthetic data to real data, in
order:

- **`01_basic_mock_fit.ipynb`** -- the core LOSVD + template-mixture
  recovery machinery on a synthetic mock built from real E-MILES
  templates *you choose* (edit `TEMPLATE_RECIPE`: any number of
  templates, any ages/metallicities/weights -- no separate single-
  template-vs-mixture code path), no external files needed.
- **`02_realistic_mock_fit.ipynb`** -- composes the newer, more-realistic
  pieces on top of that same engine and the same configurable-recipe
  mechanism: correlated (AR(1)) pixel noise, continuum-cofitting, and
  adaptive regularization. Includes a real diagnostic section that
  isolates a visibly-biased LOSVD recovery down to its actual cause (a
  known, sigma-dependent regularization bias -- confirmed *not* a
  continuum-fitting defect by switching continuum/correlated-noise off
  one at a time and checking directly), not just an assertion. Also
  honest about two gaps found while building it: continuum-cofitting and
  adaptive regularization aren't wired together yet, and the bias-
  correction calibration table has no `rho` axis and no populated
  `m_eff` axis in its default form.
- **`03_emiles_templates.ipynb`** -- the full `kinextract.emiles` API
  (packing, loading, LSF matching at pack time) for the shipped 636-
  template E-MILES grid.
- **`04_template_calibration.ipynb`** -- run this first for any real MUSE
  target. Gets a MUSE datacube (a real ESO archive download, a synthetic
  Sersic-profile mock with known ground truth, or shipped pre-computed
  products), bins it with a "dartboard" scheme (log/linear-radial x
  equal-angle sectors, fully tunable bin edges), fits every bin against
  the full 636-template E-MILES grid, and iteratively reduces to a
  representative template subset (converges when the *union* of
  surviving templates across bins stabilizes) -- saved for reuse in
  `05_real_data_muse.ipynb`.
- **`05_real_data_muse.ipynb`** -- fitting real VLT/MUSE NGC 5102 spectra
  end-to-end, from raw file to bias-corrected LOSVD, using the template
  subset `04_template_calibration.ipynb` produced.
- **`06_real_data_stis.ipynb`** -- the same walkthrough on a real HST/STIS
  NGC 5102 spectrum instead, using real STIS-observed stellar templates
  (no LSF-matching step needed -- templates and data share the same
  instrument) -- demonstrates this package's real-data workflow is not
  MUSE-specific.
- **`07_noise_correlation.ipynb`** -- measures real MUSE pixel-noise
  correlation (AR(1) lag-1 correlation length), demonstrates the Bacon et
  al. (2017) MUSE variance-underestimate factor, and shows correlated-
  noise whitening (`FitConfig.rho`) applied to a real fit.
- **`08_full_galaxy_analysis.ipynb`** -- the capstone: a full real-galaxy
  analysis start to finish -- PA-aware dartboard binning, per-bin
  masking/cleaning/fitting/bootstrap uncertainty in parallel, moment maps,
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
- **Uncertainty quantification** (`kinextract.uncertainty`):
  parametric bootstrap over the recovered LOSVD (`bootstrap_losvd`, any
  `n_templates >= 1`, including mixture-weight uncertainty), giving
  per-bin uncertainty bands plus std on V/sigma/h3/h4, plus an opt-in
  `calibrate_noise_scale`/`noise_scale` mechanism for when the assumed
  per-pixel noise itself is suspected wrong (validated on synthetic
  uniform noise mis-specification; **not yet safe on real data**, see
  Known limitations). See `uncertainty.py`'s module docstring for the
  full, nuanced coverage picture across the sigma range.
- **Continuum co-fitting** (`kinextract.continuum`): a shared,
  structurally-positive multiplicative continuum
  (`C(lambda) = exp(spline(lambda))`), jointly recovered with the LOSVD
  and template weights via 3-block alternating minimization. Cleanly
  validated on synthetic data, including the key "does the continuum
  contaminate the weight recovery" adversarial check -- see
  `continuum.py`'s own docstring. **Tried on real data and found not to
  help** (see Status above) -- kept because the mechanism itself is
  solid, not because it's currently recommended for this dataset.
- **Outlier rejection** (`kinextract.forward_model.
  mask_wavelength_windows`, `kinextract.cleaning`): fixed-wavelength
  masking for known bad-pixel ranges, and iterative sigma-clipping
  (`fit_losvd_clean`/`fit_losvd_clean_parallel`) for contamination whose
  location isn't known in advance. The clip threshold can be measured
  against a robust (MAD-based) estimate of the residual's own scale
  (`robust_scale=True`) rather than the raw input noise vector -- matters
  on real data, where the input noise vector commonly underestimates the
  true residual scatter (see the chi2_red note above); confirmed to
  behave far better than the raw-threshold version on real MUSE bins that
  otherwise clip runaway. See `08_full_galaxy_analysis.ipynb` for this
  used as the default per-bin cleaning step, and its own Limitations
  section for what it still doesn't handle (continuum-*shape* mismatch,
  in particular).
- **LOSVD moment estimation** (`kinextract.gh_moments`): summarizes
  a recovered LOSVD as (V, sigma, h3, h4) via a direct-moment estimator
  that beats a full Gauss-Hermite refit on this package's own validation
  grid; the joint Gauss-Hermite fit is kept as a general-purpose
  alternative and for comparison against the wider dynamics literature.

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
  modest edge there instead -- see `02_realistic_mock_fit.ipynb` Section
  5). Neither is strictly better; which one you want depends on whether
  you're measuring moments or full-curve shape.
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
  `07_noise_correlation.ipynb`/`08_full_galaxy_analysis.ipynb` for the
  current, honest picture on real data.
- **Adaptive regularization selection (`lam_selection="aic_local"`, the
  default) can select heavy regularization even in the package's own
  validated velocity-grid convention, at low-to-moderate sigma.** Found
  while investigating a related warning in `FitConfig`: at sigma below
  roughly 30-50 km/s, `aic_local` can pick the top of its own candidate
  grid, producing real, sometimes severe sigma bias. Not yet root-caused
  or fixed -- flagged here rather than silently working around it.
- **Iterative sigma-clipping needs its threshold measured against the
  residual's own scale on real data, not the raw input noise vector.**
  `fit_losvd_clean`'s default (`robust_scale=False`) compares the residual
  directly to the input per-pixel noise -- correct when that noise vector
  is well-calibrated (chi2_red ~ 1), but on real data where chi2_red
  commonly runs well into the double digits (see above), a nominal
  "3 sigma" cut is actually far less than 1 true sigma, and can clip the
  majority of a spectrum on the first pass, collapsing the fit to an
  unconstrained, biased baseline. `robust_scale=True` instead measures the
  clip threshold against a MAD-based robust estimate of the *current*
  residual's own scale (recomputed each iteration) -- confirmed
  substantially better-behaved on real bins that broke the default, and
  used as the default cleaning step in `08_full_galaxy_analysis.ipynb`.
  It still needs real per-target tuning (too aggressive a threshold risks
  clipping into genuine absorption features), and has no mechanism for a
  continuum-*shape* mismatch (as opposed to a localized outlier) -- see
  `cleaning.py`'s own docstring and `08_full_galaxy_analysis.ipynb`'s
  Limitations section.
- **The regularization-bias correction (`bias_correction.py`,
  `apply_bias_correction`) is not yet trustworthy enough to default on.**
  A 4-gate acceptance test (per-cell bias, held-out coverage, worst-cell
  regression, regression tests) finds sigma coverage passes but h4
  coverage does not -- h4's point estimate is severely shrunk under
  regularization and the reported uncertainty never widens to reflect
  that. `apply_bias_correction` stays `False` by default until this is
  resolved; see `dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md` and
  `dev_notes/uncertainty/FINDINGS.md`.
- **h3 recovery is severely suppressed at low-to-moderate sigma.** For
  genuinely nonzero true h3, recovered h3 stays close to 0 (essentially
  fully suppressed, not just biased) across a wide low-sigma range, only
  starting to recover at higher sigma. No fix attempted yet.
- **Multi-template weight recovery is sensitive to template-library
  conditioning** -- a near-collinear template library (differing mainly
  in overall depth scale, not shape) gives genuinely unstable weight
  recovery across noise realizations, diagnosable via the condition
  number of the per-step NNLS design matrix. See
  `examples/notebooks/01_basic_mock_fit.ipynb` and
  `02_realistic_mock_fit.ipynb`. Degrades gracefully (sparser/noisier
  weight recovery, never a crash or NaN); mitigated but not eliminated by
  template-conditioning (`04_template_calibration.ipynb`'s iterative
  reduction).
- **No continuum model by default.** The core fit expects
  continuum-normalized input; continuum-cofitting (`kinextract.continuum`)
  is available but opt-in, and adaptive lambda selection does not yet
  support it (falls back to a fixed lambda with a warning).
- **Double-peaked/multi-component LOSVDs are not specifically handled.**
  The non-parametric spline recovery can represent them (unlike a fixed
  Gauss-Hermite parametrization), but no notebook in this sequence
  specifically validates that case.

## Installation

```
pip install -e ".[dev]"
```

## Testing

```
pytest tests/
```

200+ tests covering LOSVD recovery, moment estimation, multi-template
fitting, bootstrap uncertainty (single- and multi-template, serial and
parallel), noise-scale calibration and AR(1) correlated-noise whitening,
wavelength masking, iterative cleaning (serial and parallel, including
the robust-scale option), continuum co-fitting, spatial binning and
template reduction, regularization-bias calibration, and feature-
composition/input-validation regressions. These are unit/sanity tests
(correct shapes, right order of magnitude, no crashes, and several real
before/after demonstrations against known synthetic ground truth) -- the
full, wide-grid statistical validation (bias and coverage across the
entire physically-relevant sigma/S-N/resolution range) is in
`dev_notes/VALIDATION_CAMPAIGN.md` and `dev_notes/validation_suite/`,
referenced throughout the source docstrings above.

`A_w`/`y_w` (and `A_stack_w`) must not contain NaN or inf -- mask bad
pixels out of the arrays before calling any fit function; a clear
`ValueError` is raised otherwise rather than a fit silently returning a
NaN-filled LOSVD.

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
