# Parameter guide: what to set for your situation

`FitConfig.describe()` documents every field in isolation. This file is
the companion question: given a physical regime (true sigma, S/N, one
template vs. a mixture, known vs. unknown contamination), which knobs
actually matter, and what do they do to each other?

## The knobs, in one place

| Parameter | Default | Job | Interacts with |
|---|---|---|---|
| `penalty_order` | 3 | Which difference order the roughness penalty acts on. Order 3's null space is any *Gaussian* log-density, so heavy smoothing shrinks toward the best-fitting Gaussian instead of toward exponential tails -- this is what keeps noise-driven wing/edge-pedestal inflation under control by construction (see `max_edge_amplitude_frac` below for the direct diagnostic/guard), and won the 480-case paired validation over order 2 (23/32 cells better shape residual, overall \|sigma bias\| 5.6 vs 15.5 km/s). Order 2 keeps a modest shape edge for genuinely non-Gaussian LOSVDs (\|h3\|/h4 ~ 0.06+) at sigma 50-140, S/N~100, where order 3 shrinks real h4 by ~0.03-0.05 (a regime where h4 is near the measurement noise floor regardless -- see `dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`), but has no amplitude/edge guard of its own -- if you use it, watch `edge_amplitude_frac` directly. | `lam`/`lam_grid` (order-3 lam values sit orders of magnitude higher); activates two selection safeguards in `fit_losvd` (delta-AIC parsimony tiebreak, sub-knot-sigma candidate guard); requires the package's JAX-float64 default (on automatically) |
| `lam` | 2.0 (fallback value; see `lam_selection`; calibrated for `penalty_order=2` only -- order 3 needs values orders of magnitude higher, e.g. 4096) | Roughness penalty on the LOSVD spline. Always on. | `vmin`/`vmax`/`min_interior_knots` (a narrow grid can make `lam` behave far stronger than calibrated -- `FitConfig` warns if this happens); `penalty_order` (order 3 needs far larger values -- `FitConfig` warns on the stale combination) |
| `lam_selection` | `"aic_local"` | Whether `lam` is picked per fit (`"aic_local"`, minimizing chi2+2·m_eff -- measured 20-57% lower sigma bias than fixed `lam=2.0`) or held at the fixed `lam` value above (`"fixed"`). Supports `continuum_knots` too (`aic_regularization.estimate_m_eff_local`'s own joint LOSVD+continuum linearization) -- real cost increase there (~4-5x per fit, each scanned `lam` costs a full 3-block alternation) and its own "lam selection is a lottery at S/N 60-150" risk (this module's own module docstring) applies more sharply under continuum; see `dev_notes/continuum/FINDINGS.md`'s 2026-08-11 entry before switching a whole real-data population over to it. | `bootstrap_losvd` (`reselect_lam=True`, the default, re-runs selection per replicate, continuum-cofitting included) and auto-switches its default bias-correction table to the `m_eff`-axis one when this is `"aic_local"` |
| `max_edge_amplitude_frac` | 0.01 | During adaptive selection, excludes a candidate `lam` whose fitted LOSVD is still above this fraction of its own peak at either velocity-grid edge -- the divergent-edge-pedestal guard: a roughness penalty alone can't distinguish a compact LOSVD from one carrying a broad, low-amplitude pedestal across the grid, and that pedestal inflates every reported moment if left unconstrained. `aic_regularization.edge_amplitude_frac(b)` is the same check, exposed directly for diagnosing an *already-fitted* curve (e.g. a fixed-`lam` result) without refitting. Superseded an earlier per-row wing-taper penalty (`lam_wing`, removed) that needed per-target retuning to work at all; this constraint is inert on data with no pedestal and needs none. | Only active during `lam_selection="aic_local"`; has no effect on a `"fixed"` fit unless you check `edge_amplitude_frac` on its result yourself |
| `shape_aware_correction` | False | Post-fit correction for a moment-estimator windowing bias. | Helps below ~sigma=200 km/s, measurably *hurts* above it -- `compute_losvd_moments` warns if active and the recovered sigma lands above that boundary |
| `noise_scale` (via `calibrate_noise_scale`) | 1.0 (off) | Inflates bootstrap-replicate noise injection when the assumed per-pixel noise is wrong. | Only affects bootstrap replicates, never the point estimate or `lam`'s own behavior -- deliberately decoupled |
| `smooth_window_A` / `inflation_cap` (`fit_losvd_robust(cleaning_method="inflate")`) | 8.0 / 8.0 | **The recommended default** response to unmodeled contamination (sky-subtraction residuals in particular) whose wavelength isn't known in advance -- per-spectrum, per-wavelength error inflation from the fit's own residual, capped and floored, refit against the inflated errors. Real full-field validation (113 real MUSE bins, NGC 4751): median whitened RMS 1.41 -> 0.96, bins tripping `cleaning_method="clip"`'s own 10%-clipped sanity warning 22/113 -> 0/113, recovered kinematics essentially unchanged. Both defaults are starting points, not universal -- real per-target tuning found `smooth_window_A=3.0` needed at MUSE's ~1.25A/pix spacing, and `inflation_cap` needing to go as high as 100 on a target whose sky-line residuals otherwise saturated the cap (`dev_notes/real_data_sigma/FINDINGS.md`). | Uses `cleaning_method="clip"`'s own hard-clip loop underneath, twice, as a loose cleanup layer only (`prelim_clip_sigma`/`final_clip_sigma`, both loose by default) -- do not tighten those to do the inflation's own job. Not yet shown to reduce absolute kinematic bias, only residual calibration/robustness -- see `robust_fit`'s own module docstring |
| `clip_sigma` / `max_clip_fraction` / `robust_scale` (`fit_losvd_robust(cleaning_method="clip")`) | 3.0 / 0.1 / `True` | Iterative sigma-clipping, standalone. **Demoted from the previous default** -- real chi2_red on real data commonly runs well above 1 (11-28 measured), so a plain hard clip alone can compound bias rather than fix contamination; prefer `cleaning_method="inflate"` above unless you have a specific, independent reason to suspect a handful of genuine per-pixel outliers only. `robust_scale=True` (measures `clip_sigma` against a MAD-based robust scale of the current residual, not the raw whitened residual) is the default for exactly this reason -- confirmed to collapse to a biased, unconstrained fit at `robust_scale=False` whenever chi2_red is far from 1 (clipped >90% of pixels in the case that found this); `cleaning_method="inflate"`'s own internal clip passes have always hardcoded `robust_scale=True`. | Do not reach for this to fix sigma inflation from unmodeled contamination -- that's `cleaning_method="inflate"`'s job. Outlier rejection can *compound* a model bias, since the pixels disagreeing most with an already-biased fit are often exactly the ones correctly arguing against it |
| `mask_wavelength_windows` (`windows=`) | -- | Fixed-wavelength exclusion for *known* bad-pixel ranges (e.g. sky-subtraction residuals at specific wavelengths). | Complementary to `fit_losvd_robust`'s `cleaning_method="inflate"`/`"clip"` modes, not a substitute -- use this when the bad wavelengths are known and fixed across every spectrum, the per-spectrum methods when they aren't |
| `n_templates` (shape of `A_stack`) | -- | Not a `FitConfig` field -- the shape of the template stack you build and pass in. `fit_losvd`/`bootstrap_losvd`/`fit_losvd_robust` all accept any `n_templates >= 1`. | More templates costs identifiability: the same fixed-size LOSVD histogram now has to compete against more free template weights, which measurably worsens sigma/V bias at some sigma ranges relative to a single known template |

## Recommended starting points by regime

- **Current defaults (`penalty_order=3`, `lam_selection="aic_local"`,
  `max_edge_amplitude_frac=0.01`) are the validated starting point
  everywhere** the 480-case 2026-07 grid covers (sigma 20-300, S/N
  30-100, single template and mixtures) -- no per-regime tuning needed
  for the regularization strength itself.
- **Known, fixed bad-pixel wavelengths** (e.g. sky-subtraction residuals
  recurring at the same wavelengths across every spectrum in a survey):
  `mask_wavelength_windows`, applied before fitting.
- **Unmodeled contamination whose wavelength/severity varies per spectrum**
  (sky-subtraction residuals whose amplitude scales with S/N and radius,
  a bad detector column, real chi2_red structurally above 1): start with
  **`fit_losvd_robust(cleaning_method="inflate")`**, the recommended default --
  see its own table row above.
- **A specific, independent reason to suspect only a handful of genuine
  per-pixel outliers** (a visually obvious cosmic ray, a flagged bad
  detector column, nothing suggesting the noise vector itself is
  structurally miscalibrated): plain `cleaning_method="clip"` alone is still
  reasonable here -- reserve `"inflate"`'s extra machinery for when
  contamination is more pervasive or its wavelength isn't localized to a
  few pixels.
- **A recovered LOSVD looks visually skewed/asymmetric in a way that
  doesn't match physical expectations**: check `edge_amplitude_frac(b)`
  before suspecting a real regularization or fitting bug -- if it's ~0,
  the shape is noise-driven or a genuine physical signal near its own
  Cramer-Rao detection floor (see `examples/notebooks/02_realistic_mock_fit.ipynb`'s
  Section 5/6), not a divergent edge pedestal.
- **Suspected wrong per-pixel noise level**: `calibrate_noise_scale`,
  validated only for a uniformly-wrong noise level -- do not apply to
  data whose residuals show a handful of localized outlier pixels rather
  than a uniform mis-specification (check your own residuals first).

## Combining spaxels into one bin spectrum

`extract_aperture_spectrum` (flux sum, independent-noise error combine)
is fine for a synthetic mock or a quick look, but **for real MUSE data,
prefer `combine_bin_spectrum_robust`**: a biweight (not plain) mean for
the flux -- resistant to a single bad/cosmic-ray-contaminated spaxel
dominating the combined spectrum the way a plain mean would let it.

Error is the formal, independence-assuming inverse-variance combine
alone -- correct for spatially combining different spaxels, though (like
any such formula) it can under-report the true error if per-spaxel noise
is positively correlated (e.g. from resampling/drizzling during pipeline
reduction). That correlation is `noise_model.whiten`'s own AR(1) `rho`
job at fit time (on the wavelength axis), not this function's -- see
`05_noise_correlation.ipynb` for measuring a real target's own `rho`.
An earlier version of this function also added an "empirical" error term
(biweight scale of the raw per-spaxel flux, ported from the legacy SCO
Fortran framework's own repeat-exposure combiner, `imcmb.f`) meant to
catch that same correlation -- removed (`dev_notes/real_data_sigma/
FINDINGS.md`, 2026-08-09 entry): `imcmb.f` combines *repeat exposures of
the same sky position* (where flux scatter genuinely is noise), not
*spatially different spaxels* (where flux differences are real
surface-brightness structure) -- applied to the latter, that term ran
120-470x the actual formal noise on real data, i.e. it was measuring the
bin's own spatial gradient, not noise.

Two things worth knowing about the current, formal-only error:

- **Flux convention differs from `extract_aperture_spectrum`.** This
  function returns a *mean* (one spaxel's worth of surface brightness),
  not that function's own *sum* (total flux across the aperture) -- not
  interchangeable without rescaling by the spaxel count. This doesn't
  change the recovered LOSVD (the fit is scale-invariant -- template
  weights absorb a uniform rescale for free), but it does change any
  S/N number computed downstream from the combined flux/error.
- **The MUSE `STAT`-extension variance-underestimate correction is
  opt-in**, via `variance_correction_factor` (Bacon et al. 2017's own
  recommended value, also available as the package constant
  `BACON2017_VARIANCE_CORRECTION_FACTOR`, `1/0.6**2 ~= 2.78`) -- left at
  the default `1.0` (no correction) since the true factor is
  instrument/pipeline-version specific and should be measured for your
  own cube, not assumed. Do not apply it to a synthetic mock (its own
  noise model has no such resampling artifact to correct for).

Related: `assign_legacy_dartboard_bins`'s `fold_major_axis` (default
`True`, matching the legacy Fortran scheme exactly) controls whether a
bin's north/south halves (across the assumed kinematic major axis) get
merged or kept separate -- set `False` to preserve real north/south
asymmetry (e.g. from a warp, or a genuine failure of point-symmetry) the
folded scheme discards by construction.
