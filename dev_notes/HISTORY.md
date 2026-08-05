# dev_notes index

`dev_notes/` has ~25 subdirectories accumulated over many sessions of
investigation, each with its own scripts, logs, and (usually) a
`FINDINGS.md`. This file is the map: one line per topic, its status, and
where to actually look. It does not replace the detailed files -- read
this to find the right one fast, not instead of it.

For the full chronological narrative (what was tried in what order, and
why), see **`VALIDATION_CAMPAIGN.md`** (27 numbered stages) -- this file
is organized by topic instead, and cross-references stage numbers where a
topic maps cleanly to one.

**Note (1.0 cleanup):** several directories this index originally pointed
to (`shape_residual/`, `stress_test/`, `coverage/`, `emiles/`, `noise/`,
`regularization/`'s older comparison scripts, `model_reference/`,
`AUDIT_BRIEFING.md`, `FULL_REPO_AUDIT_2026-07-29.md`) had their raw
scripts/logs/results removed as part of the 1.0 release cleanup, since
their conclusions are fully captured in the prose below and in the
`FINDINGS.md`/root docs that remain. The findings themselves (numbers,
root causes) are unchanged and still recorded here -- only the
now-superseded reproduction scripts are gone.

## Status legend

**CLOSED** -- root-caused and either fixed or deliberately not fixed with
the tradeoff documented. **OPEN** -- still an active question or a known
gap. **REJECTED** -- a real alternative that was tried and didn't work,
kept as a record so it isn't retried.

## Core LOSVD fitting / regularization

- **CLOSED** — root docs
  [REGULARIZATION_SHAPE_RESIDUAL_HANDOFF.md](REGULARIZATION_SHAPE_RESIDUAL_HANDOFF.md),
  [REGULARIZATION_INVESTIGATION_LOG.md](REGULARIZATION_INVESTIGATION_LOG.md)
  — the LOSVD shape-residual problem (mock fits showing a ~5%-everywhere
  shape mismatch). Root-caused and fixed: third-difference (order-3,
  shrink-toward-Gaussian) penalty promoted to default, JAX float64
  promoted to default. `VALIDATION_CAMPAIGN.md` Stage 22.
- **CLOSED** — [real_data_sigma/FINDINGS.md](real_data_sigma/FINDINGS.md),
  [PORT_STATUS.md](PORT_STATUS.md) — real NGC 5102 data recovered
  sigma ~2x an independent reference measurement. Root cause: the D3
  penalty's null space contains divergent edge solutions, and
  `aic_local` selects too little regularization on real data because
  real residuals are systematics-dominated (**chi2_red = 11-16, not
  ~1 -- itself a load-bearing finding, see "Real-data chi2_red" below**).
  Fixed via `max_edge_amplitude_frac`, a direct real-data sigma-inflation
  guard (ported from OG kinextract). `VALIDATION_CAMPAIGN.md` Stage 23.
- **CLOSED** — [skew/FINDINGS.md](skew/FINDINGS.md),
  [validation_suite/FINDINGS.md](validation_suite/FINDINGS.md) — the GH
  "truth" mismatch: nominal Gauss-Hermite parameters (V, sigma, h3, h4)
  are *not* the same as the moments of the curve actually built from
  them once h3/h4 are nonzero, so scoring bias against the nominal
  values overstates bias. Always score against the truth *curve's own*
  moments. Produced the randomized bias-validation suite now used
  throughout (`dev_notes/validation_suite/`). `VALIDATION_CAMPAIGN.md`
  Stage 24 + follow-up (GH-truth vs. spline-truth sweep, 525 fits, kept
  both as first-class tools -- `build_spline_truth` is now
  `build_mock_cube`'s default, see memory).
- **REJECTED** — [ghcore/FINDINGS.md](ghcore/FINDINGS.md) — a Gauss-
  Hermite-core-times-penalized-spline model (`log b(v) = GH(w) +
  spline(v;c)`, only `c` penalized) was tried as a structurally
  different fix for the shape-residual problem. Fails: worse max
  residual than both existing baselines, and degenerates completely
  (sigma collapses to its lower bound) at large lam. Not fixable by
  tuning -- the null space itself is the problem. Do not retry this
  architecture.
- **REJECTED** — Variable projection (VARPRO) as a
  replacement for the alternating-minimization scheme. Confirmed
  measurably better convergence in isolation (`nfail=0` vs `nfail=8`/15
  at S/N=60, sigma=150) but multi-start/initialization was never solved
  for it, and it was not pursued further -- **do not re-litigate this**. If revisited, the
  initialization gap is the actual open problem, not the projection
  method itself.
- **CLOSED** (superseded) — An early comparison of fixed-lam vs. AIC vs.
  `aic_local` regularization selection, precedes and feeds into `VALIDATION_CAMPAIGN.md` Stage 12
  (adaptive regularization promoted to default) and Stage 22 (penalty
  order). Numbers here predate the order-3 penalty and
  `max_edge_amplitude_frac`; treat as historical context, not current
  behavior.
- **CLOSED** — [shift_template/FINDINGS.md](shift_template/FINDINGS.md)
  — `shift_template`'s linear-interpolation (`np.interp`) Doppler-shift
  evaluation adds a fractional-shift-dependent effective broadening (0
  to ~17 km/s) invisible to synthetic mocks (which cancel it by
  construction, using the same design matrix for truth and fit) but real
  on real data. Quantified; fix is opt-in (does not change any default).

## Real-data behavior

- **CLOSED, load-bearing** — "Real-data chi2_red is 11-16, not ~1."
  First established in `real_data_sigma/FINDINGS.md` (see above);
  independently reconfirmed this session (2026-07-30) on
  `05_real_data_muse.ipynb`'s three representative NGC 5102 apertures
  (chi2_red 11-28 across all three, uncorrelated with S/N). **This is a
  known, structural property of real MUSE data under this fit, not a new
  anomaly** -- residuals are systematics-dominated (template mismatch,
  unmodeled continuum shape, and/or the MUSE per-pixel noise
  underestimate, see "noise/" below), not just Gaussian noise.
  `max_edge_amplitude_frac` exists specifically to keep the LOSVD fit
  robust to this fact rather than requiring chi2_red to be forced to ~1
  first. Relevant before trusting any residual-based cleaning scheme
  (`fit_losvd_clean`, `detect_emission_from_residual`, or ZAP -- see
  below) on real data: check chi2_red first, and don't expect it to
  reach 1.
- **MOSTLY CLOSED, real-data validated** — [zap/FINDINGS.md](zap/FINDINGS.md)
  (2026-08-04) — real-data ZAP (Zurich Atmosphere Purge) sky-residual
  cleaning, for both NGC 5102 (WFM) and NGC 4751 (NFM). The original
  hypothesis here (same-cube "sky" region contaminated by real galaxy
  light) was only half right: masking the SVD/eigenbasis via ZAP's own
  `mask` argument is real and does close that pathway (confirmed
  directly from `zap.py` source), but a **separate** corruption survives
  masking anyway — a small (~0.03-0.05% of the field), spatially compact
  cluster of spaxels comes out with implausible pixel-to-pixel flux
  swings (15-540 sigma vs. real measured noise), not fixed by `nevals`
  or `cfwidthSP` tuning. Fix: detect and revert those specific spaxels
  after the fact (`repair_corrupted_spaxels`,
  `kinextract_test_suite/notebooks/kinextract_scratch_notebooks/
  zap_cleaning.py`) rather than fight ZAP's internals further. Validated
  on real full-cube runs of both targets: galaxy-center S/N provably
  preserved (byte-identical at the true corrupted-and-reverted spaxels),
  real substantial sky-residual reduction in most (not quite all)
  flagged windows/apertures — 2-3 window/aperture combinations per
  target still come out mildly worse, reported honestly, not yet
  root-caused further. Not yet wired into any real-data notebook.
- **OPEN** — Every chi2, AIC criterion, and bootstrap variance in this project
  assumes independent per-pixel noise; MUSE spectra are
  resampled/drizzled during reduction, a known source of pixel-to-pixel
  correlation. `noise_model.py`'s AR(1) machinery
  (`estimate_correlation_length`, `whiten`) exists and is validated in
  isolation, but **is not wired into any real-data notebook or fit** --
  see memory `project-kinextract-open-problems`. Directly relevant: the
  Bacon et al. 2017 MUSE variance-underestimate factor `(1/0.6)^2` is
  demonstrated in `07_noise_correlation.ipynb`; `FitConfig.rho` threads a
  measured value through to a real fit, but it is not yet auto-measured
  and applied inside `fit_losvd` itself -- still a manual step.
- **CLOSED** — E-MILES template mismatch bias: quantifies bias when the fitting
  template library doesn't include the true stellar population, at a
  range of sigma/lam. Real, sigma-dependent effect (e.g. +204% sigma
  bias at sigma=30, mismatched case); mitigated by template-conditioning
  (`template_reduction.py`, `04_template_calibration.ipynb`'s iterative
  reduction), not eliminated.

## Uncertainty / calibration

- **CLOSED** — [uncertainty/FINDINGS.md](uncertainty/FINDINGS.md) — the
  bootstrap "lam-pin" (adaptive lam selection re-run per bootstrap
  replicate vs. pinned to the point estimate's own lam): quantified
  contribution to under-coverage, fixed via `reselect_lam`.
  `VALIDATION_CAMPAIGN.md` Stage 25. This session added a Laplace-
  mixture + AIC uncertainty prototype (~20x faster than bootstrap in
  isolation) and the `losvd_single_start` performance bug/fix -- see
  `dev_notes/speed/RESULTS.md`.
- **OPEN** — [calibration/CALIBRATION_TRUST_STRATEGY.md](calibration/CALIBRATION_TRUST_STRATEGY.md)
  — the strategy (not yet fully realized) for making
  `bias_correction.py`'s regularization-bias correction trustworthy
  enough to flip `apply_bias_correction` back to its default `True`.
  This session: replaced the flawed `m_eff`-axis 3-D interpolation with
  `PerLamCalibration` (dense per-canonical-lam tables, nearest-two-lam
  log-linear interpolation) and re-ran the 4-gate acceptance test --
  sigma coverage now passes (67%->83%), **h4 coverage still fails
  (~3%, essentially unchanged)** because h4's recovered point estimate
  is severely shrunk under regularization and the bootstrap SE never
  widens to reflect that. See `dev_notes/uncertainty/FINDINGS.md`'s
  final section for the full gate-by-gate breakdown and the concrete
  next step (an uncertainty-inflation mechanism, not just a point-
  estimate shift, is needed for h4). `apply_bias_correction` correctly
  stays `False` by default.
- **CLOSED** — [validation_suite/coverage_sweep.py](validation_suite/coverage_sweep.py)
  — end-to-end empirical coverage checks (does the reported +-1-sigma
  interval actually contain truth ~68% of the time). `VALIDATION_CAMPAIGN.md`
  Stage 27 first found this an honest negative result; re-run post-
  `PerLamCalibration` this session (see calibration entry above).

## Speed

- **CLOSED** — [speed/RESULTS.md](speed/RESULTS.md) — the full speed
  story across two sessions: iteration-level LOSVD warm-start + reusing
  the lam-selection scan's own fit (34.1s -> 5.0s per fit, 6.8x);
  lam-grid warm-start + `fit_losvd_parallel` for multi-bin fits (100
  bins: ~6.0 min -> ~1.1 min, 5.3x); and the `losvd_single_start` bug
  (silently 3-5x *slower* than multi-start due to disabling
  iteration-to-iteration warm-start on every iteration, not just the
  first) found and fixed while investigating a user-reported bootstrap
  slowdown. Honest pPXF comparison included (kinextract is ~2
  orders of magnitude slower per-fit than pPXF's fixed-4-parameter
  model -- a structural difference, not a bug; wins on sigma bias in
  most regimes despite this).

## Continuum-cofitting

- **OPEN, de-risked** — [continuum/FINDINGS.md](continuum/FINDINGS.md)
  — multi-seed validation sweep (288 fits) of `continuum.py`'s
  multiplicative continuum-cofitting: `penalty_order=3` with a fixed lam
  in 4096-32768 beats the previously-only-validated `penalty_order=2`/
  `lam=2.0` combination on both sigma and h4 bias at every S/N tested.
  Adaptive (`aic_local`) lam selection still does not support continuum-
  cofitting (falls back to fixed lam with a warning) -- not yet fixed,
  tracked as a real gap, not just an unvalidated corner.

## pPXF comparison

- **CLOSED** — [ppxf_comparison/FINDINGS.md](ppxf_comparison/FINDINGS.md)
  — synthetic-cube (known-truth) head-to-head against pPXF across four
  targeted cubes/S-N regimes, complementing the real-data-only
  comparison in the `ppxf_vs_kinextract_comparison.ipynb` scratch notebook
  (`VALIDATION_CAMPAIGN.md` Stage 13). kinextract wins on sigma
  bias in most regimes; pPXF is ~2 orders of magnitude faster per fit
  (see "Speed" above for the honest framing of that tradeoff).

## Stress testing

- **CLOSED** — Comprehensive 4480-fit
  grid across 4 template recipes (`VALIDATION_CAMPAIGN.md` Stage 19),
  plus the `spectral_prior` exploratory module's sigma-dependent
  crossover (Stages 20-21: found, then substantially (not universally)
  fixed via eigenvalue normalization).

## Scratch notebooks

The working sandbox notebooks (`ppxf_vs_kinextract_comparison.ipynb`,
`dartboard_pa.py`, `fit_all_bins.py`, and others) have moved out of this
package's `dev_notes/` entirely -- they were personal working tools, not
part of the package or its documentation. They now live in
`kinextract_test_suite/notebooks/kinextract_scratch_notebooks/`.
