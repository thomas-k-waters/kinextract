# Changelog

## Unreleased

Nothing yet.

## 2.0.0 -- 2026-11-02 -- The NGC 4751 recipe as the defaults; inferior methods removed

The defaults are now the recipe validated on 480 NGC 4751-like MUSE mocks and the real NGC 4751 bins
(`dev_notes/regularization_v2/FINDINGS.md`): a bare `FitConfig()` with `fit_losvd_robust` reproduces that extraction
bit for bit, and `losvd_error_bands` its errors. Methods the validation found inferior -- slower and no more accurate,
or simply worse -- are removed. Kept as cheaper checks: `cleaning_method=None`, `laplace_losvd_uncertainty` with the
pipeline errors, `fit_gauss_hermite`, `extract_aperture_spectrum`, `detect_emission_from_residual` and
`compute_direct_moments`; kept as the slower, exact reference: `nuts_uncertainty`.

**Breaking: new defaults.** Fitted values change for anyone relying on the old defaults.
- `FitConfig.lam` 2.0 -> **4096**, with the new **`lam_rule="sigma_scaled"`** (replaces `lam_selection`): every fit is
  made at `lam`, refitted at `sigma_scaled_lam(sigma) = lam (sigma/350)^6` (capped at 131072) if that is larger, and
  refitted at `lam` x `edge_retry_factors` (2, then a further 4) while its LOSVD still holds more than
  `max_edge_amplitude_frac` (0.01) of its peak at the grid edge. `lam_rule="fixed"` fits at exactly `lam`. Results
  report `lam_used` and `lam_tried`. Implemented once, in `multi_template.with_lam_rule`, and applied by `fit_losvd`
  and `fit_losvd_robust` (the robust fit runs its whole cleaning procedure at each `lam`, re-measuring `rho` with
  `measure_rho=True`).
- `FitConfig.continuum_knots` None -> **4**: the multiplicative continuum is co-fitted by default, so fits need
  `wave_fit` (`continuum_knots=None` switches it off).
- `fit_losvd_robust`: `smooth_window_A` 8 -> **3** A, `inflation_cap` 8 -> **100**, `min_inflation_to_apply`
  None -> **1.1**, `excluded_wavelength_windows` None -> **`CAII_PROTECTED_WINDOWS`**.
- The roughness penalty is always third-difference (`kinextract.PENALTY_ORDER = 3`).
- Exact-Hessian trust-region Newton is the only solver (it became the default in development: 20-90x faster than
  L-BFGS-B on CaII-triplet mocks, 10-15x on real NGC 4751 bins, and a lower final objective -- L-BFGS-B's stopping
  rule ended short of the minimum in the stiff order-3 regime, biasing sigma by up to ~10 % at low S/N).
- `assign_legacy_dartboard_bins` folds the correct way (`-pa_deg`, as `plotbin3.f`); the previous fold was mirrored
  in x, so legacy-dartboard bins change.

**Removed** (with what replaces each):
- **Adaptive `lam` selection** -- `select_lambda_aic`, `select_lambda_aic_local`, `estimate_m_eff`,
  `estimate_m_eff_local`, `AICSelectionResult`, the `aic_regularization` module, and `FitConfig.lam_selection`,
  `lam_grid`, `self_normalize_lam_grid`. On real NGC 4751 spectra it failed in both directions (Gaussian LOSVDs at
  low S/N, broad pedestals at high S/N: one bin at sigma 648 km/s against 230 for its mirror); on the mocks it
  recovered shapes no better than the fixed rule, with slightly worse errors, at a multiple of the cost. Use the
  default `lam_rule`. `edge_amplitude_frac` moved to `kinextract.losvd` (still exported).
- **The `lam`-marginalized (AIC-weighted) Laplace bands** -- marginalizing over `lam` barely changed the errors
  (FINDINGS 14). `laplace_losvd_uncertainty` now evaluates the posterior at one `lam` (`cfg.lam`, or `lam=`), with
  the continuum included when `cfg.continuum_knots` is set; it absorbs `quick_laplace_uncertainty` and
  `laplace_continuum_uncertainty`, which are removed. `laplace_losvd_uncertainty_parallel` takes `wave_fit`, `lam`
  and `clipped_mask` and reads the continuum settings from the config (`w_m`, `min_weight_to_sample`,
  `n_continuum_knots`, `lam_c`, `ridge_c` are gone).
- **Bootstrap uncertainty** -- `bootstrap_losvd`, `bootstrap_losvd_parallel`, `calibrate_noise_scale`, the
  `uncertainty` module, and `FitConfig.losvd_single_start`. About 20x slower than Laplace, with worse coverage in
  every cell of the production sweep (h4: 0-53 % against Laplace's 50-83 % for a 68 % target). Use
  `laplace_losvd_uncertainty`, or `losvd_error_bands` on real data.
- **Regularization-bias correction** -- `correct_moments`, `correct_losvd`, `BiasCalibration`, `PerLamCalibration`,
  `load_calibration_table`, `load_per_lam_calibration_table`, `get_default_lam_axis_calibration`,
  `calibration_has_lam_axis`, `estimate_snr_whitened`, the `bias_correction` module and its calibration tables, and
  `compute_losvd_moments(..., regularization_bias_correction=, snr=, bias_calibration=, lam=)`. It never passed its own
  acceptance gates, its h4 correction was untrustworthy, and its tables were built from adaptive-`lam` fits.
- **`shape_aware_taper_correction`** and `compute_losvd_moments(..., shape_aware_correction=)` -- it made sigma worse
  at sigma >~ 300 km/s.
- **`FitConfig.optimizer`** and the L-BFGS-B path (`fit_log_spline_map`, `fit_losvd_basis`, `fit_continuum_block`
  lose `optimizer=`).
- **`FitConfig.penalty_order`** -- the second-difference penalty inflated wing mass and sigma at low S/N and gave
  3-4x larger h4 bias with the continuum.
- **`fit_losvd_robust(cleaning_method="clip")`** and its `clip_sigma`, `max_iter`, `max_clip_fraction`,
  `robust_scale` -- the inflate cleaning is better calibrated on real data (113 NGC 4751 bins: whitened RMS 1.41 ->
  0.96, bins with more than 10 % of pixels clipped 22 -> 0). Its loose cleanup clip remains inside the inflate passes.
- `fit_losvd_basis(..., losvd_warm_start=)`: the cold restart on every alternating iteration was ~3x slower for the
  same answer.

**New:**
- **`losvd_error_bands` / `losvd_error_bands_parallel`** -- the error-band recipe for real IFU data in one call:
  the sky-line + template-mismatch noise model on a preliminary fit's residuals, Laplace with those errors (times the
  robust fit's inflation) at the fit's `lam` and `rho` without its clipped pixels, the draws moved onto the fit, and
  moment errors as half the 16-84 % range of the draws' moments.
- **The systematics noise model**: `noise_model_errors`, `fit_noise_model`, `sky_line_profile`,
  `load_muse_sky_lines`, `recenter_samples`. Per-pixel variance `err^2 + (kappa*s)^2 + (f*model)^2`, with `s` the
  LSF-blended sky-line spectrum, fitted by maximum likelihood. For the bands only: fitting with these errors at a
  fixed `lam` over-smooths. On 480 NGC 4751-like mocks with real sky residuals: shapes unchanged (paired ratio
  1.001); coverage of the model-binned LOSVD 0.48 -> 0.78. On the real NGC 4751 bins: the bin/mirror pair chi2 per
  model bin 11.7 -> 0.74 (NFM) and 11.3 -> 0.59 (WFM).
- **`sigma_scaled_lam`**, **`with_lam_rule`**, **`PENALTY_ORDER`**, **`FitConfig.lam_rule`**,
  **`FitConfig.edge_retry_factors`**.
- **`transvd_bounds_laplace`/`write_spmc_in_laplace(..., b_samples=, model_bins=)`**: widen the per-velocity widths
  in each dynamical-model bin so that the error `model.x` builds for the bin (a linear sum, `velmtod.f`) is at least
  the posterior scatter of the binned LOSVD; the linear sum of correlated per-point widths understated it by about
  2x. Only ever widened, so the noise floor survives in faint wing bins. Also new: `legacy_io.write_spmc_in_laplace`,
  `transvd_bounds_laplace`, `spmc_side_inverts`, and `muse_cube.trim_cube_wavelength_fits`.

**Fixed:**
- **Parallel workers now really run single-threaded** (`multi_template.single_threaded_worker_env`, used by every
  `*_parallel` function): unpickling the pool initializer already imported numpy and JAX with the parent's unpinned
  thread settings, and nothing limited XLA's own thread pool -- 4 workers ran ~50 threads each. 16 real NGC 4751
  bins, 4 workers: 27.4 s -> 17.1 s, identical results.
- **`laplace_losvd_uncertainty_parallel(..., clipped_mask=)`** drops each bin's robust-fit clipped pixels before the
  Laplace fit, so the posterior is centered on the robust fit's solution (without it, NGC 4751 WFM bins refit their
  outliers and reported std=0 or moved by hundreds of km/s). `cleaning._fit_losvd_clip`'s `clipped_mask` is now
  exactly the pixel set the returned fit excluded (new `n_flagged_not_refit`, `clip_converged`).
- **A Gaussian start in the LOSVD multi-start pool** (`losvd.gaussian_initial_coeffs`): fixes fits that converged
  into an edge basin from the ridge and flat starts.
- The narrow-grid "effective lam" warning is measured against the top of the `lam` rule's range (3 x 131072).

**Documentation and examples:**
- `docs/api.rst` lists exactly the public API; the quickstart, configuration page, README and `PARAMETER_GUIDE.md`
  describe the new defaults and drop the removed methods.
- Every example notebook uses the defaults and contains no comparisons with removed methods:
  - `01`-`03`: fits pass `wave_fit` (continuum on); `02` loses its penalty-order section, gains Section 5b on the
    `lam` rule, and its uncertainty section is Laplace only.
  - `04`: fits with the defaults; explains why half of the NGC 4751 cube (the dusty side of the major axis) is
    masked; its binning figure shows the masked white-light image beside the bins.
  - `05`: Section 6-7 uncertainties from Laplace; Section 8 ends with `losvd_error_bands`.
  - `06`: Laplace only; its stale comparison with an NGC 5102 MUSE fit is gone.
  - `07` is rewritten as a complete extraction of a real cube (NGC 4751 MUSE NFM) with the defaults: the dust mask
    (explained) and 25 defective spaxels, a legacy dartboard (shipped as `examples/data/muse/ngc4751_bin_{r,v}.out`),
    `losvd_error_bands_parallel`, flags and hand tuning, V_sys, moment maps (cmasher iceburn/sunburst via the new
    optional `.[examples]` extra), radial profiles and a mirror-symmetry check of the errors. Its shipped results are
    regenerated.
  - Examples 04, 05 and 07 use the NGC 4751 MUSE NFM cube instead of NGC 5102.
- `compute_direct_moments` vs `fit_gauss_hermite`: on fits at the new defaults the two agree to about 0.5 km/s on the
  validation grid (the direct moment's clear win was measured on adaptive-`lam` fits); noted in `gh_moments`.

## 1.0.1 -- Consolidate robust-fit API; fix real error-budget and template-selection bugs; add continuum support to adaptive lambda
Breaking:

    fit_losvd_clean/_inflate/_with_continuum consolidated into
    fit_losvd_robust(cleaning_method=None|clip|inflate),
    continuum-cofitting now orthogonal (cfg.continuum_knots).
    robustness param renamed cleaning_method; robust_scale now
    defaults True for clip (False was biased on real MUSE data).
    cleaning_method=inflate now the recommended default over clip
    (validated on 113 NGC4751 bins: residual RMS 1.41->0.96).
    Dead/unvalidated code removed: lam_wing/taper_* mechanism,
    ablation.py, calibrate_bias.py.

Fixed:

    combine_bin_spectrum_robust no longer adds a spurious empirical
    error term that was the dominant driver of implausible error bars
    on real data (chi2/dof as low as 0.01).
    reduce_templates_iteratively was silently whitening as rho=0
    regardless of cfg.rho, and couldn't do continuum-cofitting.
    estimate_m_eff/select_lambda_aic didn't work under continuum
    (never reachable via FitConfig, so no real fit was affected).
    07_full_galaxy_analysis.ipynb's shipped-results fast path
    (FORCE_RECOMPUTE=False) was broken in 3 places.

New: lam_selection=aic_local now supports continuum-cofitting;
fit_losvd_robust gained measure_rho and min_inflation_to_apply
(opt-in, fixes an over-generous error budget found on NGC5102).

Detailed notes (written under "Unreleased" before the 1.0.1 tag):

- **`07_full_galaxy_analysis.ipynb`'s shipped-results fast path
  (`FORCE_RECOMPUTE=False`, the notebook's own advertised default) was
  broken in three separate places** -- found by a user hitting a
  `KeyError` on the very first cell that reads a fitted bin's spectrum,
  then confirmed to be systemic, not a one-off, by actually re-running
  the fast path end to end rather than assuming one fix covered it. (1)
  `errors_inflated`/`inflation_factor` were only ever attached under
  `result["..."]`, and the shipped-results loader never populated
  `bin_data[bid]["result"]` at all -- now stored as flat per-bin fields
  on both code paths, matching every other field's own convention, and
  persisted through the shipped `.npz` for the 4 example bins it keeps
  full spectra for. (2) The "5b. Bootstrap cross-check" cell referenced
  `A_stack_w_clean_list`/`y_w_clean_list`/`cfg_list_measured`, built only
  in the real-compute branch -- now skips with a clear explanation
  under shipped results (this cross-check's own inputs are deliberately
  too large to ship, not an oversight). (3) A new diagnostic added this
  session (below) referenced `WAVEFITMIN`/`WAVEFITMAX`, same
  branch-scoping bug, caught before it ever shipped. Verified via 3
  separate real runs: fresh compute (`FORCE_RECOMPUTE=True`, 275/275
  bins), the shipped path cold, and the shipped path after regenerating
  the `.npz` with the new fields -- all 0 errored cells.

- **New real-data finding: below roughly the instrument's own resolution,
  recovered sigma is a lam-selection lottery for *any* `lam_selection`,
  and the reported Laplace uncertainty badly under-covers it.** Raised
  by the user as a specific worry about `07_full_galaxy_analysis.ipynb`'s
  own radial profile (a sharp, non-smooth dip at low sigma) and whether
  `max_edge_amplitude_frac` has an analogue for the opposite
  (over-regularization) failure. Confirmed directly on this notebook's
  own real bins (not just the synthetic sigma=20 mock this was first
  suspected on, `dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`'s
  2026-07-26 entry): a fixed-lam ladder (64-65536) shows no monotonic
  trend and no plateau, `lam_selection="aic_local"` lands on an equally
  arbitrary point in the same ladder rather than a systematically better
  one, and reported `sherm_std` is 5-10x smaller than the actual
  lam-to-lam swing in the checked bins. All 275/275 bins in this
  notebook's own real target sit below the flagged threshold -- see the
  new Section 7b diagnostic and `dev_notes/uncertainty/FINDINGS.md`'s
  2026-08-11 (late) entry for the specific numbers. Not fixed (switching
  `lam_selection` does not help, confirmed not assumed) -- documented in
  `README.md`'s Known Limitations and flagged in-notebook instead of
  silently worked around.

- **`estimate_m_eff`/`select_lambda_aic` (the bootstrap-based reference
  implementation `estimate_m_eff_local`/`select_lambda_aic_local` are
  meant to be validated against, per both functions' own docstrings)
  did not actually work under continuum-cofitting at all** -- found
  while independently validating the new adaptive-lam-under-continuum
  support (below) against its own designated ground truth, which turned
  out to have never been checked there itself (no test exercised
  `estimate_m_eff`/`select_lambda_aic` with `continuum_knots` set before
  this). Two bugs, both silent: (1) `_model_prediction` (the model
  `estimate_m_eff`'s bootstrap procedure resamples around) never
  multiplied by the fitted continuum `C`, so its `chi2`/bootstrap noise
  were computed against the wrong model whenever continuum was active;
  (2) `estimate_m_eff` had no `wave_fit` parameter at all, so its
  internal `fit_losvd` calls raised `ValueError` the instant
  `cfg.continuum_knots` was set. Both fixed; `estimate_m_eff`/
  `select_lambda_aic` now accept `wave_fit` (threaded through, same
  contract as the `_local` versions) and agree with
  `estimate_m_eff_local`'s chi2 exactly at a shared `lam` under
  continuum, the same regression check already covered the no-continuum
  case. Neither bug affected any real fit -- `select_lambda_aic`/
  `estimate_m_eff` are standalone validation utilities, never reachable
  through `FitConfig.lam_selection` (`"aic_local"`/`"fixed"` only) -- but
  meant the closed-form continuum extension shipped without the one
  check its own docstring calls "its whole reason to exist."

- **Example notebooks (`examples/notebooks/01`-`07`): spring-cleaning
  pass.** Applied everything learned from the N4751 real-data work above
  to the tutorial notebooks themselves:
  - `02_realistic_mock_fit.ipynb`'s Section 5 "skewed LOSVD" example now
    prints `edge_amplitude_frac` for both fits shown, confirming the
    asymmetry there is a Cramer-Rao noise-floor effect (Section 6's own
    detectability bound), not a divergent edge pedestal -- explained
    in-notebook rather than left as an unexplained artifact.
  - `07_full_galaxy_analysis.ipynb`'s radial-profile cell now reports the
    ratio of the h3/h4 scatter-across-bins to the population's own
    astrophysical spread (18-32x measured) -- confirming the large h3/h4
    error bars there are a real consequence of low-sigma h3 suppression
    and Laplace uncertainty's own weakest-coverage regime, not a bug.
  - `06_other_instruments.ipynb`'s STIS/MUSE-disagreement diagnostic
    (previously `wing_mass`, a leftover from the removed `lam_wing`
    mechanism) now checks `edge_amplitude_frac` instead, and rules a
    divergent edge pedestal out as a contributor.
  - Stale cross-references and duplicated constants cleaned up throughout
    (broken `01_basic_mock_fit.ipynb` section-number citations that no
    longer matched that notebook's own headings; `05_noise_correlation.
    ipynb`'s inline Bacon et al. 2017 variance factor now imports
    `BACON2017_VARIANCE_CORRECTION_FACTOR` instead of recomputing it).
    `PARAMETER_GUIDE.md` also updated -- its `lam_wing` section and table
    row (referencing the removed mechanism) replaced with the
    `max_edge_amplitude_frac` guard that actually does that job, and its
    `combine_bin_spectrum_robust` section rewritten to match that
    function's current formal-error-only behavior.
  - `README.md`'s test-suite description updated from a stale "200+" to
    the current count (327 passed) and now mentions the
    `edge_amplitude_frac` diagnostic and continuum-cofitting coverage
    directly.

- **`lam_selection="aic_local"` (adaptive regularization) now supports
  continuum-cofitting.** Previously fell back to a fixed `lam` with a
  `UserWarning` whenever `cfg.continuum_knots` was also set.
  `aic_regularization.estimate_m_eff_local` now builds a joint
  LOSVD+continuum linearization instead of ignoring the continuum block
  entirely; `select_lambda_aic_local`/`fit_losvd` thread `wave_fit`
  through to it, and `bootstrap_losvd`'s `reselect_lam` (previously
  forced off under continuum for the same reason) now works there too.
  Real per-candidate cost increase (~4-5x wall-clock per fit, each
  scanned `lam` now costs a 3-block alternation) and its own "lam
  selection is a lottery at S/N 60-150" risk applies here too -- see
  `dev_notes/continuum/FINDINGS.md`'s 2026-08-11 entry for validation
  numbers and why a blanket switch to adaptive isn't recommended.
  `aic_regularization.edge_amplitude_frac` (the divergent-edge-pedestal
  diagnostic, previously private as `_edge_amplitude_frac`) is now a
  top-level export, for flagging which bins actually need it.

- **Release-readiness cleanup pass.** Removed confirmed-dead/orphaned code
  and fixed defaults known to underperform an available alternative,
  rather than keeping them around as untaken options:
  - `FitConfig.lam_wing`/`taper_sfac`/`taper_power`/`n_sigl0_iter` and the
    wing-taper machinery behind them (`losvd.tapered_lam_vector`,
    `greville_abscissae`, `penalty_row_velocities`, `core_width_sigma`)
    removed outright. This mechanism was ported from the original
    `kinextract` pipeline but never validated to fix the real-data sigma
    inflation it targeted (its own test/config docstrings said so
    explicitly); `FitConfig.max_edge_amplitude_frac` supersedes it and
    was already the default (see `dev_notes/PORT_STATUS.md`,
    `dev_notes/real_data_sigma/FINDINGS.md`). Breaking for any caller
    passing these fields.
  - `robust_scale` (`cleaning_method="clip"`'s own scale-calibration
    option, also used internally by `cleaning_method="inflate"`'s
    preliminary/final clip passes) now defaults to `True`, not `False`.
    `False` was confirmed to collapse to a biased, unconstrained fit on
    real MUSE data whenever chi2_red is far from 1 (clipped >90% of
    pixels in the case that found this); `True` was already what
    `cleaning_method="inflate"` hardcoded internally for exactly this
    reason. Breaking for any caller relying on the old `False` default
    with `cleaning_method="clip"` -- pass `robust_scale=False` explicitly
    to keep the old behavior.
  - `ablation.py` and `calibrate_bias.py` removed -- both confirmed
    orphaned (exercised only by their own unit tests, never imported by
    any notebook, script, or other package code, not part of
    `__init__.py`'s public API list). The calibration tables
    `calibrate_bias.py` originally generated (`bias_correction.py`'s
    shipped `PerLamCalibration`/`BiasCalibration` data) are unaffected --
    only the generating tool is gone; git history has it if a table ever
    needs regenerating.
  - Added `BACON2017_VARIANCE_CORRECTION_FACTOR` (`muse_cube.py`) and
    `CAII_PROTECTED_WINDOWS` (`cleaning.py`), both now top-level exports
    -- previously copy-pasted verbatim into every real-MUSE-data example
    notebook that needed them.

- **`fit_losvd_robust`/`fit_losvd_robust_parallel`'s `robustness` parameter
  renamed to `cleaning_method`** (same values: `None`/`"clip"`/`"inflate"`)
  -- `robustness` read as a generic quality knob, not what it actually
  selects (which cleaning strategy runs before the reported fit).
  Breaking, no deprecation shim -- update any external callers.

- **`combine_bin_spectrum_robust` no longer adds an "empirical" error
  term.** That term (`biweight_scale` of raw per-spaxel flux) was ported
  from the legacy SCO Fortran framework's `imcmb.f`, but `imcmb.f`
  combines *repeat exposures of the same already-spatially-binned
  spectrum* -- across exposures of one sky position, scatter in raw flux
  genuinely is noise. Applied to *spatially different spaxels within one
  bin* (this function's actual use), the same formula instead measures
  real surface-brightness structure, not noise -- confirmed directly on
  real N4751 data: 120-470x the actual (formal) noise level, meaning the
  combined error was, for practical purposes, just this spatial-gradient
  term. This was the dominant driver of implausibly large reported
  errors (whitened chi2/dof as low as 0.01, `1sigma` velocity bands
  spanning hundreds of km/s) on real data for both targets this project
  covers. Error is now the formal inverse-variance combine alone;
  `min_spaxels_for_empirical` (now meaningless) is removed. Any residual
  correlated-noise underestimate is `noise_model.whiten`'s job (already
  in the pipeline, applied at fit time), not this function's -- see
  `dev_notes/real_data_sigma/FINDINGS.md`'s 2026-08-09 entry.

- **`fit_losvd_robust` gained two opt-in parameters** (default off,
  existing callers/tests unaffected): `measure_rho=True` measures each
  bin's own pixel correlation (lag-1 ACF on a preliminary `rho=0`,
  continuum-off fit -- this pass ignores `cfg.continuum_knots`
  regardless of the real fit's own setting, avoiding a third
  continuum-cofit pass per bin on continuum-cofitting targets) instead
  of trusting one global `cfg.rho`, and returns `rho_used` so downstream
  re-whitening can match; `min_inflation_to_apply` skips
  `cleaning_method="inflate"`'s refit-with-inflated-errors step when the
  preliminary pass's own *local* inflation profile
  (`compute_local_inflation`'s max) shows nothing anywhere needs
  inflating (returns `chi2_dof_prelim`, a diagnostic only, not what this
  gates on -- an earlier version gated on a global chi2/dof average and
  was wrong, missing bins that are fine on average but have a real
  localized contaminated window). Fixes a real over-generous error
  budget (chi2/dof as low as 0.08) found on NGC 5102 -- see
  `dev_notes/real_data_sigma/FINDINGS.md`'s 2026-08-08 entries.

- **`fit_losvd_clean`/`fit_losvd_clean_parallel` (`cleaning.py`),
  `fit_losvd_inflate`/`fit_losvd_inflate_parallel` (`variance_inflation.py`,
  now removed), and `fit_losvd_with_continuum` (`continuum.py`) consolidated
  into `fit_losvd_robust`/`fit_losvd_robust_parallel` (`robust_fit.py`)**,
  a single entry point taking a `cleaning_method=None|"clip"|"inflate"`
  parameter instead of a separate function per strategy. Always takes
  unwhitened `(A_stack, flux_fit, errors, cfg, wave_fit=...)` and whitens
  internally regardless of mode (previously `fit_losvd_clean` took
  pre-whitened input while `fit_losvd_inflate` didn't -- one contract now).
  Continuum-cofitting has no separate function at all -- set
  `cfg.continuum_knots`, orthogonal to `cleaning_method`, on `fit_losvd`/
  `fit_losvd_robust` alike. `cleaning_method="inflate"` (per-spectrum,
  per-wavelength error inflation from the fit's own residual, floored at
  1x, capped, refit against the inflated errors) remains the recommended
  default response to unmodeled contamination (sky-subtraction residuals
  in particular), replacing `cleaning_method="clip"`'s standalone hard
  sigma-clip as the primary defense -- real full-field validation (113
  NGC 4751 MUSE bins): median whitened residual RMS 1.41 -> 0.96, bins
  tripping the clip mode's own 10%-clipped sanity warning 22/113 -> 0/113,
  recovered kinematics essentially unchanged. Not yet shown to reduce
  absolute kinematic bias (a calibration/robustness result, not yet a
  demonstrated accuracy one) -- see `dev_notes/variance_inflation/FINDINGS.md`
  for the full record and open items. The inflate mode's own preliminary
  pass now always fits at a single fixed `lam` regardless of
  `cfg.lam_selection` (its only job is a residual estimate for the
  inflation factor, not the reported LOSVD) -- ~27% faster on real
  `cfg.lam_selection="aic_local"` configs (profiled directly on real
  NGC 5102 bins), no effect when `cfg.lam_selection="fixed"` already.
  Both example notebooks (`04_template_calibration.ipynb`,
  `07_full_galaxy_analysis.ipynb`) use `cleaning_method="inflate"`.

- **`reduce_templates_iteratively` (`template_reduction.py`) now whitens
  via `noise_model.whiten(..., rho=cfg.rho)` and passes `wave_fit` to
  `fit_losvd`**, instead of a bare `A_stack/errors` division that
  silently whitened as if `rho=0.0` regardless of what `cfg.rho` actually
  held, and never supported `cfg.continuum_knots` at all (raised
  `ValueError` as soon as it was set, since `fit_losvd` needs `wave_fit`
  to build a continuum basis). Found running `n4751_template_calibration.
  ipynb` under the same `cfg` (including `continuum_knots`/measured
  `rho`) its own full-galaxy-analysis notebook uses -- template selection
  under a silently-different fitting method than the real science fit is
  exactly the inconsistency this change across both notebooks exists to
  remove. Two regression tests added
  (`test_cfg_rho_is_actually_respected_not_silently_whitened_as_zero`,
  `test_continuum_knots_runs_without_crashing`).

## 1.0.0 -- from-scratch rebuild

This release replaces the pre-1.0 `kinextract` (0.4.0) entirely with what
was developed independently as `kinextract_lite`: a from-scratch, more
thoroughly validated reimplementation of the same non-parametric LOSVD
fitting idea, not an incremental update. `src/`, `tests/`, and
`examples/` are wholesale replacements; `dev_notes/` is additive -- the
old project's investigation history (`stress_test/` harness, EMILES-vs-
real-star template comparisons, `n_losvd_bins`/auto-`xlam` regression
testing, MUSE/STIS real-data validation, joint-mode V-bias remediation)
is preserved alongside the new project's own (indexed in
`dev_notes/HISTORY.md`), not discarded.

**Not ported, by deliberate scope decision:**
- `kinextract.joint` (the P-spline-in-the-model continuum-cofitting
  method) -- superseded by a different architecture: a shared
  multiplicative continuum, alternated against the LOSVD/template fit in
  the same optimization (`kinextract.continuum`,
  `kinextract.multi_template`'s 3-block alternation).
- The TOML config loader (`load_config_from_toml`) and the `kinextract`
  CLI entry point -- `FitConfig` is now a plain dataclass, instantiated
  directly.
- `benchmarks/`'s old diagnostic suite (built against the removed
  `joint`/TOML API) -- removed; its role is now filled by
  `dev_notes/validation_suite/` (`sweep.py`, `coverage_sweep.py`), which
  is actively maintained and was rerun end-to-end as part of this merge.

**What's new relative to either prior codebase**, in brief (full detail in
`VALIDATION_CAMPAIGN.md`'s 27-stage record and `dev_notes/HISTORY.md`'s
topic index):
- A single fitting engine for any number of templates
  (`kinextract.fit_losvd`), replacing separate single/multi-template code
  paths.
- Data-driven regularization-strength selection via a bootstrap-estimated
  generalized AIC (`aic_regularization.py`), as an alternative to a fixed
  `lam`.
- A third-difference, shrink-toward-Gaussian penalty (`penalty_order=3`,
  the default) that measurably fixed a ~5%-everywhere LOSVD shape
  residual found in the prior architecture.
- Parametric-bootstrap uncertainty quantification with a documented,
  tested coverage story, plus an explicitly opt-in (not yet trusted for
  unsupervised use) synthetic-calibrated correction for regularization/
  shrinkage bias.
- A Laplace/curvature-approximation uncertainty engine
  (`laplace_uncertainty.py`: `laplace_losvd_uncertainty`,
  `quick_laplace_uncertainty`), the default uncertainty engine as of a
  production-scale coverage validation across sigma 90-600 km/s --
  matches or beats bootstrap's own coverage in every tested regime
  (dramatically so for h4) at a real speed advantage. Bootstrap remains
  available as a cross-check for real ensemble properties (skew,
  multimodality) the Laplace approximation can't give. A NUTS/NumPyro
  reference mode (`nuts_uncertainty.py`, optional `nuts` extra) exists
  alongside both, for simulation-based-calibration spot-checks rather
  than everyday use.
- AR(1) correlated-noise whitening (`noise_model.py`), closing a
  previously-unaddressed real-MUSE noise-correlation gap.
- E-MILES SSP template packing/LSF-matching as a first-class template
  source, IFU-cube spatial binning (PA-aware reflected `sin(theta)`
  sectors, plus a second scheme reading the legacy `sco_framework`
  pipeline's own `bin_r.out`/`bin_v.out` dartboard geometry directly for
  bit-identical bins where that matters) and iterative template-subset
  reduction, for full-galaxy analyses
  (`examples/notebooks/07_full_galaxy_analysis.ipynb`). `legacy_io.py`
  writes that same Fortran pipeline's own `.fit`/`.ascii`/`.rms`/`.mcfit`
  formats directly, for projects handing kinextract's output to that
  downstream toolchain unmodified.
- A randomized, multi-seed validation suite
  (`dev_notes/validation_suite/`) scoring recovered moments against a
  mock's own truth-curve moments (not nominal Gauss-Hermite parameters),
  covering GH and spline-shaped truths, run as this project's standing
  regression check rather than one-off validation scripts.

See `README.md` for current package scope/status and
`dev_notes/HISTORY.md` for the full index of closed/open investigations.
