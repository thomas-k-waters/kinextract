# The bootstrap lam-pin: a real, now-fixed contributor to under-coverage

**Date: 2026-07-28**

## The question

Earlier this project measured the bootstrap uncertainty band as roughly 2x
too narrow at sigma=50 and 4x too narrow at sigma=90 (coverage saturating
at 95%/89% instead of continuing to improve with S/N), attributed to
resampling around a *biased* point estimate (the bootstrap can't see an
offset shared by every replicate). Separately, this project documented
that adaptive lam selection is a lottery at S/N 60-150: the same real data
can pick lam=8 or lam=32768 depending only on the noise draw, because the
AIC criterion goes flat there. The open question: does `bootstrap_losvd`
propagate that lam-selection instability into its reported band, or does
it hide it?

## The mechanism (confirmed by direct code read, no ambiguity)

`bootstrap_losvd` fixed every one of its `n_boot` replicates at the point
estimate's own `lam_used`:

```python
cfg_boot = dataclasses.replace(
    cfg, lam=result0["lam_used"], lam_selection="fixed", ...
)
```

This is a deliberate, documented cost-saving choice ("re-running selection
for every replicate would cost n_boot x len(lam_grid) extra fits"), not a
hidden bug -- but given lam selection is a real lottery in exactly the
regime being bootstrapped, this pins the *entire ensemble* to whatever
regularization branch the point estimate's own noise draw happened to
land in, and structurally cannot explore "what if this data had pushed
selection somewhere else."

## Measured: how much of the gap does this actually explain

Three ensembles (`dev_notes/uncertainty/diag01_lam_lottery_vs_bootstrap.py`,
`diag01b_reselect_isolated.py`), same mock, n=30 each:

- **A**: `bootstrap_losvd` as shipped (lam pinned)
- **B**: same mechanism (noise injected around the point estimate's own
  model) but lam reselected every replicate, single-start held fixed to
  isolate this from a separate multi-start-vs-single-start confound the
  first version of this diagnostic missed (logged below)
- **C**: the gold standard -- independent noise draws around the TRUE
  model, each fit fresh with full adaptive selection (literally "repeat
  the real observation N times")

| regime | A: pinned | B: reselected | C: true sampling |
|---|---|---|---|
| sigma=50, S/N=60 | 4.119 | 3.377 | 3.178 |
| sigma=90, S/N=60 | 4.756 | 4.013 | 3.492 |

Reselecting closes ~75-80% of the sigma=50 gap and ~60% of the sigma=90 gap
between the pinned bootstrap and the true sampling distribution's own
spread. (B vs C's remaining difference is within n=30 sampling noise on a
std estimate, roughly +/-26% relative SE -- not read as a further
systematic direction.)

**This is real, but not the whole story.** The bias-blindness mechanism
already documented elsewhere is untouched by this fix and is still
expected to be the larger remaining contributor to the originally-measured
2x/4x under-coverage -- this fix closes a real, secondary piece of it.

## The fix

`bootstrap_losvd` gained `reselect_lam: bool = True` (new default). When
`cfg.lam_selection == "aic_local"`, every replicate now re-runs that same
adaptive selection independently instead of being pinned. `reselect_lam`
is forced off automatically when `cfg.continuum_knots` is set (adaptive
selection doesn't support continuum-cofitting yet and would just
re-trigger `fit_losvd`'s own fixed-lam fallback -- and its `UserWarning`
-- on every replicate for no behavioral difference), and is a no-op
whenever the caller already set `lam_selection="fixed"` explicitly (there
is nothing to reselect).

**Cost**: reselecting is ~2-2.5x the pinned cost per replicate (measured:
~7.3s vs ~3.2s for a single-start fit on a 4-template mock) -- real, but
affordable given this package's own ~6.8x fitting speed-up earlier this
project.

Two downstream `cfg_boot.lam` reads (the bias-calibration table-choice
logic, and the `m_eff` evaluation for the correction) were fixed to read
`result0["lam_used"]` instead -- `cfg_boot.lam` is no longer guaranteed to
equal the point estimate's own lam once reselection leaves it unset/stale.

## Verified

- `tests/test_uncertainty.py`: 4 new tests added
  (`test_reselect_lam_controls_the_actual_cfg_passed_to_every_replicate_fit`
  -- a monkeypatched structural check of the actual `cfg` objects passed to
  every replicate's `fit_losvd` call, not just a statistical consequence;
  plus basic-sanity, already-fixed-lam-is-a-no-op, and
  continuum-knots-is-a-no-op tests). All 25 tests in the file pass.
- Every pre-existing test that calls `bootstrap_losvd` without testing the
  lam mechanism itself was audited and pinned `reselect_lam=False`
  explicitly (`test_uncertainty.py`, `test_feature_composition.py`) --
  keeps the suite's runtime unchanged rather than silently paying the new
  default's cost everywhere it doesn't matter.
- Full package suite: **178 passed, 0 failed, 1948s (32:28)** -- matches
  the pre-change baseline (1970s) to within normal run-to-run variance.

## Process notes (logged per standing rule -- three real tooling mistakes, all self-corrected)

1. **A confound in the first version of this diagnostic.** The initial
   "reselect" comparison function left `cfg`'s own multi-start setting
   untouched while `bootstrap_losvd`'s shipped pinned path forces
   single-start on every replicate -- so the first A-vs-B comparison
   conflated "lam reselected" with "multi-start instead of single-start."
   Caught before drawing any conclusion; fixed by forcing
   `losvd_single_start=True` on the reselect variant too, isolating
   exactly one variable.
2. **An unguarded module-level loop.** `diag01_lam_lottery_vs_bootstrap.py`
   had its comparison loop at module level (no `if __name__ == "__main__":`
   guard), so importing it from a follow-up script silently re-ran the
   entire ~15-minute original diagnostic as a side effect, with the actual
   follow-up computation never being reached before a timeout killed it.
   Diagnosed by checking `ast.parse` and import timing directly rather
   than assuming a hang; fixed by guarding the loop.
3. **Undersized timeouts combined with buffered, unflushed prints.** Two
   separate runs produced empty output files and were initially
   (wrongly) read as "the script hung." In both cases the process was
   genuinely still running (or genuinely killed by an undersized
   `timeout`) and Python's default full-buffering when stdout is piped
   (not a tty) meant nothing had been flushed yet. Diagnosed by directly
   timing a single fit with `python3 -u` and explicit per-iteration
   `flush=True`, which reset the cost estimate from an assumed ~2-3s/fit
   to the real ~6-8s/fit -- explaining both timeouts without any actual
   hang. Fixed by using `-u`/`flush=True` and sizing timeouts from
   measured, not assumed, per-fit cost from then on.

None of these three affected the final reported numbers above -- each was
caught and corrected before being used to draw a conclusion, per this
project's standing rule to verify empirically before concluding anything.

---

## Addendum, 2026-07-29: end-to-end coverage check — reselect_lam helped, but did not fix coverage

The `reselect_lam` fix above was validated on its own narrow terms (does
reselecting close the gap between the pinned bootstrap's *spread* and the
true sampling distribution's spread -- yes, ~60-80%). What was NOT yet
checked: does the actual reported confidence interval contain the truth
at the rate it claims to? That is the number that actually matters, and
it is a different, harder question than "is the spread closer to
correct" -- an interval can have the right spread and still fail to
cover if it is not centered in the right place (exactly the bias-
blindness mechanism this project has documented separately).

Built `dev_notes/validation_suite/coverage_sweep.py`: for each of 9
(sigma, S/N) cells (sigma in {50,90,150}, S/N in {40,60,100}), drew 4
random truths (h3/h4/templates/weights randomized, `safe` preset,
`GH` generator) and ran one `bootstrap_losvd` call per truth (n_boot=12,
`reselect_lam=True`, the current default). Checked whether the true
sigma/h4 (the truth curve's own moments, never nominal) falls within
`point0['sherm'] +/- sherm_std` (nominal ~68% coverage for a correctly
calibrated 1-sigma interval).

**Result (n=36): sigma coverage = 47%, h4 coverage = 11%.** Both far
below the 68% target, and the shortfall is broad -- every one of the 9
cells under-covers sigma (25-75%, median well under 68%), and h4 coverage
is 0% in 7 of 9 cells. This is not a corner-case failure; it is the
general behavior across the tested regime.

| sigma | S/N | sigma coverage | h4 coverage | mean \|sigma bias\| |
|---|---|---|---|---|
| 50 | 40 | 75% | 0% | 5.51 |
| 50 | 60 | 50% | 0% | 3.98 |
| 50 | 100 | 50% | 0% | 2.58 |
| 90 | 40 | 50% | 0% | 6.12 |
| 90 | 60 | 25% | 0% | 5.44 |
| 90 | 100 | 50% | 50% | 2.16 |
| 150 | 40 | 75% | 0% | 9.09 |
| 150 | 60 | 25% | 50% | 15.00 |
| 150 | 100 | 25% | 0% | 6.10 |

**Honest conclusion: `reselect_lam` was a real, measured, partial fix --
not a solution.** It closed a genuine gap (the lam-selection-variance
blindness), but the dominant remaining mechanism is exactly the one
flagged and NOT addressed by this fix: resampling around a *biased*
point estimate cannot see the bias itself, so the interval is centered
wrong even when its width is closer to correct. h4's much worse coverage
than sigma's (11% vs 47%) is consistent with this -- h4 is more heavily
shrunk by the order-3 penalty (the noise-floor/shrinkage mechanism
documented in this file and in `dev_notes/ghcore/FINDINGS.md`), so its
point estimate carries a larger, more systematic offset for a
spread-only fix to miss.

**What this means for the open uncertainty-band problem**: it remains
open. The next real step is not another mechanical fix to
`bootstrap_losvd` but a genuine bias-correction-and-inflation approach
that shifts the reported center, not just its width -- i.e. the
`apply_bias_correction=True` path, now running against freshly
regenerated (order-3, current-default) calibration tables for the first
time. This exact coverage check (`coverage_sweep.py`, unchanged) should
be rerun with `--apply-bias-correction` as the next diagnostic, to
measure whether the newly-valid `_corrected` fields actually close the
remaining gap -- not yet done tonight, time-limited.

# Laplace + AIC-weighted uncertainty prototype: 20x faster, coverage at least as good on a small sample

**Date: 2026-07-29 (overnight)**

## The question

`bootstrap_losvd` refits `n_boot` times per LOSVD (full multi-start
optimization each time) -- real, but expensive: ~150-280s/case at
n_boot=12 in this run. Is there a cheaper approximation that doesn't
throw away the two things bootstrap actually buys (protection against
landing in a different local optimum, and propagation of `lam`-selection
instability into the reported band -- see the `reselect_lam` finding
above)?

## The approach

`dev_notes/uncertainty/laplace_prototype.py`, `laplace_aic_uncertainty()`:

1. **Laplace/curvature covariance at each `lam` in `cfg.resolved_lam_grid()`**
   (6 values for the order-3 default): reuses `estimate_m_eff_local`'s
   already-fitted `result` at that `lam` (no extra fit), takes
   `jax.hessian` of the exact objective `fit_log_spline_map` minimizes
   (`losvd._neg_log_post`) at the converged `coeffs`, inverts it for
   `cov_coeffs`, then propagates to `(vherm, sherm, h3, h4)` via a
   finite-difference Jacobian of `compute_losvd_moments` (that function
   itself isn't JAX-traceable, so this one step is numerical).
2. **AIC-weighted model averaging across the `lam` grid**: softmax
   weights from the same `chi2 + 2*m_eff` criterion
   `select_lambda_aic_local` uses internally, combined via the law of
   total variance (`Var_total = E_lam[Var(theta|lam)] + Var_lam[E(theta|lam)]`).
   This is what stands in for bootstrap's own implicit lam-selection-
   uncertainty propagation -- instead of resampling and re-selecting
   `lam` per replicate, it scores every canonical `lam` value once and
   lets between-lam disagreement contribute directly to the variance.

**What this does NOT do, honestly**: it computes the curvature around
the single best-found optimum at each `lam`, not a true multi-start
mixture -- `fit_log_spline_map` doesn't expose all multi-start
candidates, only the winner plus a scalar `objective_gap`. That gap is
reported per case (`objective_gap_at_best`) as a diagnostic for when this
matters, though in this run it came back `None` for every case (the
winning `lam`'s own multi-start pass apparently never had a second
competitive candidate at these settings -- not otherwise investigated
tonight). This is a screening estimator, not a validated drop-in
replacement.

## Benchmark: 18 cases (the 9-cell sigma x S/N grid from the coverage sweep above, 2 random truths/cell, `n_boot=12` for bootstrap)

| sigma | S/N | sigma cov. (boot / laplace) | h4 cov. (boot / laplace) |
|---|---|---|---|
| 50 | 40 | 50% / 100% | 0% / 100% |
| 50 | 60 | 100% / 100% | 0% / 50% |
| 50 | 100 | 50% / 50% | 0% / 50% |
| 90 | 40 | 0% / 50% | 0% / 50% |
| 90 | 60 | 50% / 100% | 0% / 50% |
| 90 | 100 | 50% / 50% | 50% / 50% |
| 150 | 40 | 100% / 100% | 50% / 50% |
| 150 | 60 | 100% / 100% | 0% / 50% |
| 150 | 100 | 100% / 100% | 0% / 100% |

**Aggregate (n=18): sigma coverage 67% (boot) vs. 83% (laplace); h4
coverage 6% (boot) vs. 56% (laplace). Total wall time: 3200s (boot) vs.
159s (laplace) -- a 20.1x speedup.** Laplace matched or beat bootstrap's
coverage in every one of the 9 cells, never worse.

## Reading this honestly

- **n=2 truths/cell is small** -- individual cell percentages (0/50/100%)
  are only 0, 1, or 2 successes out of 2 draws, not a stable estimate.
  The aggregate n=18 numbers are the more meaningful summary, and even
  those should be treated as a first look, not a validated result --
  this was deliberately kept small (per plan) to bound overnight runtime.
- **h4's gap is the more informative one.** Bootstrap's own `h4_std` was
  at or near 0.0000 in most cells (every replicate collapses to nearly
  the same order-3-shrunk h4 regardless of noise draw, since
  `reselect_lam` still pins the *same* biased mechanism each time) --
  structurally near-zero-width intervals around a biased point estimate
  cannot cover a nonzero true h4 except by luck. Laplace's h4_std was
  10-50x larger in the same cells (0.01-0.05 vs. 0.00-0.007) because the
  AIC-weighted between-lam term picks up real disagreement about how
  much the LOSVD's shape should be smoothed -- exactly the kind of
  uncertainty a single-lam-pinned bootstrap replicate set cannot see by
  construction, regardless of `n_boot`.
- **Some laplace sigma_std values are quite wide** (17-22 km/s at a few
  cells vs. bootstrap's 3-10 km/s) -- part of the coverage win could be
  "wide enough to rarely miss" rather than "correctly calibrated." This
  needs a wider run (more cells, more truths/cell) before trusting the
  method's precision, not just its coverage rate, particularly since a
  usefully narrow interval that still covers is the actual goal, not
  coverage alone.
- **This does not replace bootstrap's ensemble product.** `uncertainty.py`'s
  own docstring recommends the full replicate ensemble (`b_boot`,
  covariance across `(vherm,sherm,h3,h4)`, skew/multimodality) as the
  right thing to propagate forward into a downstream analysis, not a
  single point + symmetric std -- the Laplace prototype only produces
  the latter. It's a fast screening/triage tool (e.g. "is this fit's
  uncertainty band roughly sane before committing to a slow bootstrap"),
  not a proposed default replacement.

## Verdict

A genuinely promising prototype -- 20x faster and, on this small sample,
matching or beating bootstrap's own coverage in every cell tested,
particularly for h4 where bootstrap structurally cannot see the
lam-selection-driven shrinkage uncertainty. Not yet a case for switching
the default: needs the same scale of validation (full sigma/S-N grid,
more truths/cell, a check of interval width vs. bootstrap's ensemble
shape) before being trusted for real use. Next step, not done tonight:
rerun at the same 36-case scale as the main coverage sweep once that
run's compute budget is free.

# Coverage re-check with bias correction applied -- and the 4-gate acceptance test run against current tables

**Date: 2026-07-29 (overnight)**

## Setup

Reran `coverage_sweep.py` with the identical 36-case grid, seed, and
`n_boot=12` as the 47%/11% baseline above, this time with
`--apply-bias-correction`. The script records both `_raw` and
`_corrected` fields per case, so raw and corrected are a genuine paired
comparison from the same fits, not two separate runs.

## A number that needs flagging before anything else: this run's own raw coverage does not match the recorded 47%/11% baseline

Same seed, same grid, same `n_boot` -- a deterministic pipeline should
reproduce the old baseline's raw numbers exactly. It did not: this run's
raw coverage is **sigma=64%, h4=3%**, not 47%/11%. Checked
`git diff --stat` on the fitting engine: `multi_template.py`,
`aic_regularization.py`, `config.py`, and `uncertainty.py` all carry
large uncommitted diffs (100+ lines each) relative to the last commit --
consistent with `VALIDATION_CAMPAIGN.md` Stage 25 (`reselect_lam` fix)
and Stage 26 (calibration regeneration) having landed in the working
tree between when the original baseline was recorded and now, with
nothing committed in between to bisect against. Not re-investigated
line-by-line tonight (no prior commit to diff against, and the
regression-test suite covering these modules is green -- see below).
**Treat the 47%/11% figure as superseded, not a like-for-like comparison
point** -- this run's own raw column is the correct baseline for
everything below, since raw and corrected here share every seed and
code path except the correction step itself.

## Aggregate result

| | sigma coverage | h4 coverage |
|---|---|---|
| raw (this run) | 64% | 3% |
| corrected (this run) | 81% | 6% |
| nominal target | ~68% | ~68% |

Correction moves sigma coverage from below-target to above-target, and
leaves h4 coverage essentially unchanged and far below target.

## Per-cell breakdown (n=4 truths/cell)

| sigma | S/N | sigma cov. (raw/corr) | h4 cov. (raw/corr) | mean \|sigma bias\| (raw/corr) | mean \|h4 bias\| (raw/corr) |
|---|---|---|---|---|---|
| 50 | 40 | 75% / 75% | 0% / 0% | 5.97 / 6.26 | 0.0327 / 0.0327 |
| 50 | 60 | 100% / 75% | 0% / 0% | 3.05 / 2.75 | 0.0137 / 0.0138 |
| 50 | 100 | 75% / 75% | 25% / 25% | 2.12 / 1.15 | 0.0213 / 0.0203 |
| 90 | 40 | 50% / 75% | 0% / 0% | 5.16 / 5.17 | 0.0312 / 0.0312 |
| 90 | 60 | 50% / 50% | 0% / 0% | 3.68 / 3.67 | 0.0242 / 0.0242 |
| 90 | 100 | 0% / 100% | 0% / 0% | 2.73 / 2.75 | 0.0246 / 0.0245 |
| 150 | 40 | 50% / 75% | 0% / 0% | 7.57 / 6.93 | 0.0244 / 0.0250 |
| 150 | 60 | 100% / 100% | 0% / 0% | 2.57 / 2.47 | 0.0271 / 0.0271 |
| 150 | 100 | 75% / 100% | 0% / 25% | 3.61 / 3.64 | 0.0170 / 0.0167 |

## The 4-gate acceptance test (`CALIBRATION_TRUST_STRATEGY.md`), run against the current tables for the first time

1. **Per-cell bias check -- FAILS.** The gate requires `|bias_corrected| <
   |bias_raw|` in *every* cell, not on average. Sigma bias improved in
   5/9 cells and got (mildly) worse in 4/9 (50/40: 5.97->6.26; 90/40:
   5.16->5.17; 90/100: 2.73->2.75; 150/100: 3.61->3.64). h4 bias barely
   moves anywhere -- essentially inert (3 cells marginally better, 2
   marginally worse, 4 unchanged to 4 decimal places). Correction is a
   net-positive-on-average, not a strict per-cell improvement.
2. **Independent held-out coverage check -- passes for sigma, fails for
   h4.** `coverage_sweep.py`'s own seeds (base seed 0, per-cell/per-truth
   sub-seeds) are disjoint from the calibration table's own training seed
   (`WEIGHT_SEED=30000` in `build_emiles_calibration.py`), so this
   qualifies as a genuine held-out check. Sigma: 64%->81%, crossing the
   ~68% target. h4: 3%->6%, nowhere near it -- the correction's h4 term
   is calibrated as an additive floor at true h4=0, and does essentially
   nothing to fix coverage when the true h4 is nonzero (every truth here
   is drawn with random nonzero h3/h4, per this suite's own randomized-
   truth design).
3. **Worst-cell regression check -- PASSES.** Worst raw sigma-coverage
   cell: 0% (sigma=90, S/N=100). Worst corrected sigma-coverage cell: 50%
   (sigma=90, S/N=60) -- strictly better, not worse. Worst h4-coverage
   cell is 0% in both raw and corrected (tied, several cells) -- not
   regressed either.
4. **Regression test per bug found -- PASSES.** Both known overcorrection
   failure modes already have dedicated tests in
   `tests/test_bias_correction.py`
   (`test_bootstrap_correction_does_not_severely_overcorrect_the_01_notebook_mock`,
   `test_meff_axis_table_switch_uses_plain_table_when_lam_matches_and_meff_table_otherwise`),
   added earlier tonight, both passing.

**Overall: 2 of 4 gates pass outright (3, 4); gate 2 passes for sigma
only; gate 1 fails.** This matches `uncertainty.py`'s own current stance
(`apply_bias_correction` default `False`) -- the correction is a real,
measured net improvement for sigma specifically, not yet a case for
flipping the default, and h4 correction in particular is not doing its
job under the current calibration and should not be trusted for h4 at
all in its current form. Not attempted tonight (flagged in the original
plan as too large for unattended work): Fix #1 in
`CALIBRATION_TRUST_STRATEGY.md` -- rebuilding the tables with dense
per-canonical-lam sampling -- which is the actual precondition for
meaningfully passing gate 1.

# Re-run against `PerLamCalibration` (dense per-lam tables, nearest-two-lam interpolation)

**Date: 2026-07-30**

`bias_correction.py` was redesigned this session (`PerLamCalibration`
replacing the old single-table `m_eff`-axis design -- see this project's
own dev_notes/memory for the full root-cause writeup: the old design used
`m_eff` as a continuous 3rd `griddata` interpolation axis even though
`m_eff` depends on both `lam` and the local regime, so unrelated points
could interpolate together). The new design queries one dense 2-D
`(sigma, S/N)` table per canonical `lam`, via nearest-two-`lam`
log-linear interpolation. Same `coverage_sweep.py` harness, same protocol
(fresh seeds, `--apply-bias-correction`, `n_boot=12`, `n_truths=4`,
sigma in {50,90,150} x S/N in {40,60,100}, 36 cells).

## Per-cell breakdown (n=4 truths/cell)

| sigma | S/N | sigma cov. (raw/corr) | h4 cov. (raw/corr) | mean \|sigma bias\| (raw/corr) | mean \|h4 bias\| (raw/corr) |
|---|---|---|---|---|---|
| 50 | 40 | 50% / 50% | 0% / 0% | 8.46 / 8.81 | 0.0327 / 0.0327 |
| 50 | 60 | 100% / 100% | 0% / 0% | 2.32 / 1.73 | 0.0137 / 0.0138 |
| 50 | 100 | 75% / 75% | 25% / 25% | 2.12 / 0.88 | 0.0213 / 0.0216 |
| 90 | 40 | 50% / 75% | 0% / 0% | 5.16 / 5.16 | 0.0312 / 0.0312 |
| 90 | 60 | 75% / 75% | 0% / 0% | 2.94 / 2.35 | 0.0242 / 0.0242 |
| 90 | 100 | 25% / 100% | 0% / 0% | 2.73 / 2.75 | 0.0246 / 0.0246 |
| 150 | 40 | 50% / 75% | 0% / 0% | 7.57 / 5.17 | 0.0244 / 0.0244 |
| 150 | 60 | 100% / 100% | 0% / 0% | 2.57 / 2.46 | 0.0271 / 0.0271 |
| 150 | 100 | 75% / 100% | 0% / 0% | 3.61 / 3.45 | 0.0170 / 0.0168 |

Overall coverage: sigma 67% -> **83%** raw->corrected; h4 unchanged at
**2.8%** (identical to the pre-redesign run's 3%/6% within this smaller
sample's noise -- still nowhere near the ~68% target).

## The 4-gate test, re-run against the new tables

1. **Per-cell bias check -- still FAILS**, as expected (this is a
   strict per-cell bar the old table also failed). Sigma bias got
   *worse* in 9/36 cells, h4 in 15/36 -- both still net-positive on
   average, not universal.
2. **Independent held-out coverage -- unchanged verdict: passes for
   sigma, fails for h4.** Sigma coverage improved further versus the
   pre-redesign run (81%->83%, both comfortably past the ~68% target,
   arguably now over-conservative rather than under). h4 coverage is
   **identical to 4 decimal places in several cells** between raw and
   corrected (e.g. 0.0312 both ways at sigma=90/S-N=40) -- this is not
   "the correction barely helps," it is the correction being almost a
   complete no-op for h4 in most cells. Root cause, confirmed directly
   from the raw fit output: recovered h4 is essentially fully shrunk to
   ~0.0001 regardless of a true h4 of +-0.01-0.04, and the bootstrap
   standard error on h4 is itself tiny (~1e-6-1e-5) -- three to four
   orders of magnitude smaller than the actual bias. No point-estimate
   shift of the size this calibration applies (~0.0001) can fix a bias
   this large, and more importantly, **the reported uncertainty band
   itself never widens to reflect that shrinkage is severe here** --
   `correct_moments` only ever shifts the point estimate, it does not
   inflate `h4_std`. Coverage near 0% is therefore not really a
   calibration-accuracy problem at all; it is a missing mechanism
   (nothing in the current correction pipeline widens an interval to
   account for known severe shrinkage, only shifts its center).
3. **Worst-cell regression check -- now FAILS (a real regression from
   the pre-redesign run, where this gate passed).** Worst raw sigma
   bias across all 36 cells: 14.2. Worst *corrected* sigma bias: 16.4 --
   strictly worse than the worst raw case, the exact thing this gate
   exists to catch. (h4's worst case is a rounding-level tie, 0.0399 raw
   vs 0.0398 corrected -- not the source of the failure.) With only 4
   truths/cell this could be one unlucky draw rather than a systematic
   property of the new tables; worth re-checking with a larger
   `n_truths` before treating it as conclusive, but it must not be
   silently dropped.
4. **Regression tests per bug found -- still PASSES** (both tests from
   the earlier run were rewritten against the `PerLamCalibration` API
   during this session's redesign and are in the current 190-passing
   suite).

**Overall: gate 2 (sigma half) and gate 4 pass; gates 1 and 3 fail; gate
2 (h4 half) fails for a different, now much better-understood reason
than before.** The `PerLamCalibration` redesign accomplished what it set
out to do -- it fixed the actual architectural bug (the flawed `m_eff`-
axis 3-D interpolation) and measurably improved sigma's typical
performance and coverage -- but it was never expected to fix h4 on its
own, since h4's problem is upstream of calibration entirely (severe
regularization shrinkage the bootstrap SE doesn't reflect), and it
surfaced a new worst-cell regression on sigma that needs a larger-sample
re-check before deciding whether it is real. `apply_bias_correction`
correctly stays `False` by default; no table is close to a full-gate
pass yet. Next concrete step, if this is picked back up: either (a) a
larger `n_truths` re-run to confirm/refute the worst-cell finding, or
(b) a mechanism that inflates `h4_std` (not just shifts the h4 point
estimate) when the calibration table's own training data shows h4 is
severely shrunk at that (sigma, S/N, lam) cell -- gate 2's h4 half
cannot pass without one.

---

## 2026-08-01: Laplace/AIC-weighted h4 uncertainty -- reproduction check, still n=18

Overnight plan Phase 6 set out to validate the Laplace/AIC-weighted
prototype (`laplace_prototype.py`) at "the full 36-case `coverage_sweep.py`
scale it's never been run at." On closer look, that's a bigger lift than
it sounds: `coverage_sweep.py` computes genuine empirical coverage (the
fraction of *many independent truth draws per cell* whose interval
contains the truth) and is hardcoded to call `bootstrap_losvd` -- plugging
the Laplace method into that harness as a real alternative, then running
enough independent draws per cell for a trustworthy per-cell coverage
fraction, is a substantially larger task than time allowed tonight.

**What was actually done**: re-ran `laplace_prototype.py`'s own existing
`__main__` script (still n=18: 9 (sigma, S/N) cells x 2 truths each) as a
**reproduction check** under current code -- several regularization and
cleaning changes landed earlier this same session
(`dev_notes/regularization/FINDINGS.md`, `dev_notes/real_data_sigma/
FINDINGS.md`), worth confirming didn't disturb this result before trusting
it further.

Result, nearly identical to the previously-recorded numbers: speedup 6.8x
(746s bootstrap vs 110s Laplace, total across 18 cases -- somewhat lower
than the earlier "~20x" figure, plausibly because bootstrap itself got
faster this session via `warm_start_lam_grid`/`reselect_lam` improvements,
narrowing the relative gap without Laplace itself changing); sigma
coverage 61% (bootstrap) vs 83% (Laplace) against a 68% target; **h4
coverage 6% (bootstrap) vs 56% (Laplace)** -- matching the earlier
recorded 6%/56% almost exactly. The finding is robust to this session's
other changes, not a fluke of the original run.

**Still not done, left for a future session**: extending `coverage_sweep.py`
itself to support the Laplace method as a selectable uncertainty mechanism
(not just `bootstrap_losvd`), and running it at real scale (many
independent truth draws per cell, not 2) to get a trustworthy per-cell
coverage fraction rather than this n=18 pooled/aggregate estimate. Also
not attempted: wiring Laplace in as `bootstrap_losvd`'s (or a new
function's) actual h4 uncertainty mechanism in the shipped package --
this remains a prototype-only script in `dev_notes/`, not production code.

---

## 2026-08-01: Laplace promoted to a real package module; production-scale coverage sweep confirms it, per-cell

Both follow-ups flagged above are now done. `laplace_prototype.py`'s
`laplace_covariance_at_lam`/`laplace_aic_uncertainty` were ported into
`src/kinextract/laplace_uncertainty.py` as `laplace_covariance_at_lam`
(dict-returning) / `laplace_losvd_uncertainty` (full AIC-weighted-over-lam
mode) / `quick_laplace_uncertainty` (single-winning-lam cheap mode), with
two real gaps the prototype didn't have closed:

- **Curve-level uncertainty** (`b_samples`/`b_median`/`b_p16`/`b_p84`), not
  just moment propagation via a finite-difference Jacobian -- coefficients
  are drawn directly from the Laplace covariance, pushed through
  `exp(basis @ coeffs)`, at each lam in the AIC-weighted grid.
- **A real bug caught before shipping**: naive Gaussian coefficient
  sampling at weakly-regularized lam (the low end of the grid) produces
  draws that land in the curvature penalty's own divergent null-space
  direction once pushed through `exp()` -- 26% of draws at lam=1 in one
  smoke test, sigma inflated from ~88 to up to 312 km/s. Fixed with
  rejection-resampling in `_sample_curves_and_moments`, filtering on the
  same `max_edge_amplitude_frac` guard Stage 23 already validated for
  point estimates. Confirmed fixed: post-fix `sherm_std` matched
  bootstrap's own (3.20 vs 3.23 km/s) on the smoke-test case.

`coverage_sweep.py` (`dev_notes/validation_suite/`) is now pluggable
(`--method bootstrap|laplace|quick_laplace`). Ran the full 9-cell
`(sigma, S/N)` grid -- sigma in {50, 90, 150}, S/N in {40, 60, 100} -- at
n_truths=8 per cell (real per-cell rates, not the earlier n=18 pooled
estimate), same seeds for both methods so this is a paired comparison, not
independent draws:

| | sigma coverage (target 68%) | h4 coverage (target 68%) |
|---|---|---|
| bootstrap | 60% (per-cell range 38-75%) | **4%** (per-cell range 0-12%) |
| laplace | 72% (per-cell range 50-100%) | **39%** (per-cell range 12-75%) |

Laplace beat bootstrap in **every one of the 9 cells** on both sigma and
h4 coverage -- not an artifact of pooling. h4 is the one that mattered:
bootstrap essentially never captures the truth (0-12%), Laplace is a real,
substantial fix (up to ~10x better per-cell) though still short of the
68% target, weakest at low sigma (12-25%) and best at high sigma
(50-75%) -- plausibly because the low-sigma regime's LOSVD is more
sharply peaked, pushing more posterior mass toward the low-lam/divergent-
edge region the rejection-resampling has to filter out, which thins the
effective sample size feeding that regime's covariance estimate.

Raw sweep data: `dev_notes/validation_suite/results/coverage_sweep_bootstrap_v2.csv`,
`coverage_sweep_laplace_v1.csv`.

**Decision**: this clears the acceptance gate `dev_notes/V1_HARDENING_ROADMAP.md`
Phase 1b set out. Laplace (`laplace_losvd_uncertainty`) becomes the
uncertainty engine the revamped notebooks call by default; `bootstrap_losvd`
remains available as an alternative/cross-check (real ensemble properties
Laplace can't give, e.g. empirical skew/multimodality) per the locked-in
decision, not deprecated.

**Not done**: low-sigma h4 coverage is real progress, not a solved problem
-- still below the 68% target in that regime. A NumPyro/NUTS reference
mode (Phase 1c) would be the natural way to check whether the residual gap
is Laplace's own Gaussian-approximation limit or something structurally
missing (e.g. template-weight uncertainty, held fixed at MAP in both
engines) -- not yet built.

---

## 2026-08-01: NUTS reference mode built; the residual low-sigma h4 gap is probably not the Laplace linearization itself

Built `src/kinextract/nuts_uncertainty.py` (`nuts_losvd_uncertainty`,
Phase 1c) -- NUTS via NumPyro, sampling `losvd._neg_log_post` directly as
the log-density (`numpyro.factor` against an improper flat prior, the
standard pattern for reusing an existing potential function rather than
reformulating the model), at a single fixed `lam` (mirroring
`quick_laplace_uncertainty`'s single-lam mode, not the full
AIC-weighted-across-lam method -- jointly sampling `lam` itself is a
separate, bigger lift, not attempted). `w` held fixed at MAP, same as the
Laplace engine, same reason. New optional dependency (`pyproject.toml`'s
`nuts` extra); not imported from `kinextract/__init__.py` so it stays
genuinely optional.

**Spot-check**: same mock, same noise draw, `quick_laplace_uncertainty`
vs. `nuts_losvd_uncertainty` at the identical `lam`, at both a low-sigma
cell (sigma=50, S/N=60 -- one of the coverage sweep's weakest h4-coverage
regimes, 12-25%) and a high-sigma cell (sigma=150, S/N=60 -- one of its
strongest, 50-75%):

| regime | Laplace sherm_std | NUTS sherm_std | Laplace h4_std | NUTS h4_std |
|---|---|---|---|---|
| sigma=50 | 4.35 | 2.91 (0.67x) | 0.0229 | 0.0209 (0.91x) |
| sigma=150 | 4.29 | 2.34 (0.55x) | 0.0006 | 0.0006 (1.02x) |

(0 divergences in both NUTS runs.) The single-lam Laplace covariance
somewhat *overestimates* `sherm_std` relative to the true local posterior
(NUTS) at both sigma regimes -- not the direction that would explain
under-coverage -- while `h4_std` tracks NUTS closely (within ~10%) at
both. This is a targeted spot-check (2 cells, 1 seed each), not a full
SBC campaign, but it points away from "the Laplace linearization is
badly wrong at low sigma" as the explanation for the coverage sweep's
residual low-sigma h4 shortfall.

**More likely explanation, consistent with everything else this project
has already found about h4** (and now h3 -- see
`dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md`'s 2026-08-01 h3-wing
entry): empirical *coverage* depends on both the posterior's width AND
its center being right. If the *point estimate* itself is systematically
shrunk toward zero by the curvature penalty at low sigma/S-N (which the
h4wing and now h3wing calibration tables directly measure, independent of
either uncertainty engine), then even a perfectly-width-calibrated
posterior around that biased center will under-cover the true value some
fraction of the time -- no posterior-width method (Laplace, NUTS, or full
HMC) fixes a biased *center* by construction. Distinguishing "the
remaining gap is bias, not width" definitively would need combining a
per-cell coverage sweep with the calibration tables' own known bias at
the same cells -- not done here, flagged as the natural next step if this
is revisited.

---

## 2026-08-01: standardized-residual breakdown -- both a real bias component and a specific, fixable sampling artifact

Direct follow-up on the "bias, not width" question above, using data
already on disk (`coverage_sweep_laplace_v1.csv`, no new fits) -- for each
cell, computed the *standardized* residual `(h4_recovered - h4_true) /
h4_std_raw` (should average to ~0 with std ~1 for a well-calibrated
interval) and split it by sigma:

| sigma | n | mean standardized resid | std | mean raw h4 bias | mean h4_std (reported) |
|---|---|---|---|---|---|
| 50 | 24 | +1.45 | 4.13 | +0.0101 | 0.0115 |
| 90 | 24 | +4.84 | 26.32 | +0.0098 | 0.0180 |
| 150 | 24 | +0.15 | 1.45 | +0.0067 | 0.0366 |

**Two distinct things, not one, are visible here:**

1. **A real, genuine bias at low sigma.** At sigma=50, the mean raw h4
   bias (+0.0101) is comparable in *magnitude* to the mean reported
   `h4_std` itself (0.0115) -- h4 is systematically shrunk toward zero by
   about one reported-sigma on average, which alone depresses coverage
   for a method that (correctly) reports a *symmetric* interval around a
   *biased* center. This directly supports the "bias, not (only) width"
   hypothesis from the entry above -- at sigma=150, both the bias
   (+0.0067) is smaller and the reported `h4_std` is much larger (0.0366,
   3x the low-sigma value), so the same absolute bias matters far less
   relative to the band.
2. **A separate, specific sampling artifact, not a general bias/width
   story.** sigma=90's std of the standardized residual (26.32) is wildly
   larger than sigma=50 or 150's -- not a smoothly-varying trend, a sign
   that one or two cells have a `h4_std_raw` far too small relative to
   the actual error, consistent with the divergent-edge rejection filter
   (`_sample_curves_and_moments`) occasionally exhausting its draw budget
   at a low-weight lam and returning very few (or, per that function's
   own warning path, zero) valid samples for that lam's contribution to
   the pooled ensemble -- a small, unrepresentative sub-sample can land
   anywhere, including an artificially tight cluster that understates the
   true spread. This is a candidate root cause for at least part of the
   remaining coverage gap, and unlike the bias story above, is
   potentially directly fixable: e.g. excluding a lam's contribution
   entirely (not just down-weighting it) once its filtered sample count
   falls below some floor, rather than pooling in whatever small,
   possibly-unrepresentative sample survived.

**Not done**: a fix for either finding (rejection-sample-floor exclusion
for #2; nothing attempted yet for #1, since fixing bias at the point-
estimate level is a much larger question -- the regularization-bias
correction machinery exists but is calibrated against bootstrap, not
Laplace, and is not yet extended there). Flagged as the two concrete,
evidence-backed next steps for closing the low/moderate-sigma h4 coverage
gap, more specific than "coverage is still below target" alone.

---

## 2026-08-04: high-sigma coverage sweep -- data existed on disk, unanalyzed, until now

Real data for exactly the regime that matters for this project's actual
science target (NGC 4751, real central sigma ~350-650 km/s -- see
`dev_notes/broad_sigma_stress_test_log.md`) was sitting on disk from an
earlier, unattended run (`coverage_sweep.py --sigma-values 90,250,400,500,600
--snr-values 60 --n-truths 18 --n-boot 15`) but never analyzed or written
up -- the bootstrap half wasn't even finished (missing sigma=600 entirely,
sigma=500 only 15/18 truths). Computed directly from
`coverage_sweep_highsigma_{laplace,bootstrap}_v1.csv` (n=18/cell unless
noted, real per-cell rates, not pooled):

| sigma | Laplace sigma cov. | Laplace h4 cov. | Bootstrap sigma cov. | Bootstrap h4 cov. |
|---|---|---|---|---|
| 90 | 67% | 50% | 72% | 0% |
| 250 | 67% | 78% | 72% | 11% |
| 400 | 89% | 83% | 61% | 39% |
| 500 | 72% | 72% | 73% (n=15) | 53% (n=15) |
| 600 | 61% | 78% | not run | not run |

**Headline: sigma coverage stays roughly on-target (~68%) for both
methods across the entire high-sigma range** -- neither collapses at
N4751's real sigma, which was a real open question (nothing in this
project's validation history had tested past sigma=150 until this data).
**Laplace's h4-coverage advantage over bootstrap not only holds at high
sigma, it's the most decisive difference in the whole table**: 50-83%
(Laplace) vs. 0-53% (bootstrap), matching or exceeding the advantage
already documented at sigma<=150. This directly confirms
`V1_HARDENING_ROADMAP.md`'s Phase 1d decision (Laplace as the default
uncertainty engine) extends to the regime this project's real target
actually lives in, not just the synthetic-mock range it was originally
validated against.

**A rare failure mode found in this data -- and, checked directly, already
fixed in current code, not a live bug.** 2/90 Laplace rows (one each at
sigma=400 and sigma=500) have `sherm_std_raw`/`h4_std_raw` = NaN.
`coverage_sweep.py`'s own coverage check correctly treats a NaN interval
as non-covering (`sigma_covered_raw_1s=False` in both cases, not
silently miscounted as a pass), so this never distorted the reported
coverage rates above. Before writing this up as an open gap, checked
`laplace_uncertainty.py`'s current `_degenerate_point_fallback` directly
-- its own docstring states it was "found directly running at real-data
scale (2026-08-03)" for this *exact* mechanism (every posterior-
predictive draw hitting the divergent-edge rejection filter), has a
dedicated passing test
(`test_zero_valid_samples_falls_back_to_point_estimate_instead_of_crashing`,
confirmed green in this session's full-suite run), and now returns a
degenerate zero-spread ensemble (`std` exactly `0.0`, not `NaN`) with a
clear `UserWarning` instead. Since current code cannot produce a NaN
here anymore, **this sweep's 2 NaN rows almost certainly predate that
2026-08-03 fix** -- the sweep itself has no recorded run date to confirm
against, but the mechanism, the fix, and the test all line up. Not
independently re-verified by rerunning the exact 2 failing cases against
current code (would need the original per-case seeds, not recorded in
the CSV) -- flagged as the concrete way to close this out with full
certainty if picked back up, but not treated as an open bug in the
meantime.

**Not done**: complete the missing bootstrap sigma=600 cell and the 3
missing sigma=500 truths, for a fully symmetric comparison (lower
priority given Laplace is already the decided default and bootstrap's
disadvantage is already unambiguous in every cell tested); rerun the 2
NaN cases specifically against current code for full certainty the fix
covers them (see above).

---

## 2026-08-07: NGC 5102 regression (point estimate + uncertainty), and a
## package-level fix for the mechanism behind it

**Trigger**: user reported NGC 5102's `07_full_galaxy_analysis.ipynb`
output had "regressed pretty significantly" -- flat h3/h4 (~1e-4
everywhere) and, once that was fixed, visibly "non-sensical" error bars
on the radial profile plot (confirmed against a user-supplied screenshot).

**Bug 1 -- point estimate h3/h4 collapse.** A prior phase's fixed
`lam=4096, penalty_order=3` (validated only for N4751's real regime,
sigma 275-1186 km/s) was carried into NGC 5102's own notebook, whose real
sigma (14-95 km/s) gives a NSIG=7 velocity grid only ~200-700 km/s wide --
2-14 knots at `knot_spacing_kms=95`, well under `FitConfig`'s
`min_interior_knots=16` floor, inflating the *effective* lam by orders of
magnitude and crushing the LOSVD toward a pure Gaussian. Fix: reverted
NGC 5102's own notebook to `CONTINUUM_KNOTS=None, LOSVD_LAM=2.0,
LOSVD_PENALTY_ORDER=2` (its pre-regression defaults). N4751 unaffected
(its lowest real sigma, 275 km/s, stays safely above the knot floor even
at NSIG=7) and left unchanged.

**Bug 2 -- uncertainty instability, root cause.** `laplace_losvd_uncertainty`
(the base, non-continuum Laplace engine) ignores `cfg.lam_selection`/
`cfg.lam` entirely and *always* AIC-weight-scans the full
`cfg.resolved_lam_grid()`. At NGC 5102's low S/N this handed >90% of the
pooled weight to an unstable `lam=0.5` candidate whose own posterior is
dominated by divergent-edge solutions -- exactly finding #2 from the
2026-08-xx entry above (an unrepresentatively tight sample from a
low-weight, poorly-sampled lam candidate), just now observed distorting
*production* output rather than a synthetic sweep. Reproduced directly on
real cached bins: `h3_std`/`h4_std` inflated 2-40x versus a single-lam
evaluation across the 10 worst-affected bins.

**Fix, package-level.** `quick_laplace_uncertainty` (pre-existing,
previously underused) already does the right thing for a bin whose point
estimate was fit at one *known, fixed* lam: run adaptive selection once
if needed, then a single Laplace pass at that lam -- no grid ensemble.
Added a three-way dispatch to `_laplace_losvd_uncertainty_worker` (the
`laplace_losvd_uncertainty_parallel` worker): continuum-cofitting bins ->
`laplace_continuum_uncertainty` (unchanged); non-continuum, *fixed*-lam
bins -> `quick_laplace_uncertainty` (new route); non-continuum,
*adaptive*-lam bins -> `laplace_losvd_uncertainty` (unchanged, still
correct for genuine adaptive selection). This makes every future caller
of the parallel API safe against this class of bug, not just the two
call sites patched directly (NGC 5102's notebook, and
`sco_framework_updated/bin/run_losvd_errors.py`, the latter of which was
also found to be carrying *no* continuum-cofitting metadata at all across
the GreatLakes handoff -- a second, preventative fix, since N4751's own
GreatLakes uncertainty step hadn't been run yet and would have silently
refit the wrong model). Verified byte-identical `.kinematics` output
between the old `lam_grid=(lam,)` caller-side workaround and the new
dispatch, on real N4751 bin data, across all three routes.

**NGC 5102 end-to-end result** (full 275-bin re-run, both fixes applied):
`h4_std` max fell from 0.297 to 0.094 (median 0.027), 0 bins with
`h4_std > 0.1` (previously several), 0 NaN. Radial profile plot visually
confirmed dramatically tighter, sensible error bars with real V/sigma/h3/
h4 structure, matching the user's screenshot pixel-for-pixel in the
*pre*-uncertainty-fix state (confirming the screenshot was current, not
stale, before the fix landed).

**Validation sweep, to confirm no regression and characterize the fix's
tradeoffs** (`coverage_sweep.py`, sigma={20,50,90,300} x snr={40,80},
n_truths=8/cell for laplace/quick_laplace, reduced to sigma={20,90} x
snr={40,80}, n_truths=4/cell for bootstrap after it proved ~5x slower
than expected under CPU contention from the other concurrent jobs; all
three below use `mocks.py`'s bare-default `FitConfig`, i.e.
`lam_selection="aic_local"` -- an *adaptive*-selection mock, not the
fixed-lam production case the dispatch fix targets):

| method | sigma cov. (raw, pooled) | h4 cov. (raw, pooled) |
|---|---|---|
| laplace (adaptive ensemble) | 75% | 47% |
| quick_laplace (single-lam) | 52% | 16% |
| bootstrap | 38% | 0% |

Full test suite (342 passed, 1 skipped, 0 failed) confirms the dispatch
change itself introduces no regression.

**Reading the sweep correctly -- this is not "quick_laplace is broken."**
In this sweep, `quick_laplace_uncertainty` still uses `lam_selection=
"aic_local"` internally (per its own docstring: run adaptive selection
once, then commit to a *single* resulting lam). Its worse coverage here
versus full adaptive `laplace_losvd_uncertainty` is exactly the expected
cost of skipping AIC-ensemble marginalization over lam-selection
uncertainty -- there is real ambiguity in which lam the data support in
this mock, and only the full ensemble captures it. This is *not* the
same situation as NGC 5102/N4751's real bins, where lam is fixed by
design (not data-selected), so there is no lam-selection uncertainty to
marginalize and `quick_laplace_uncertainty`'s single pass at that fixed
lam is the *correct*, matching-the-actual-model choice -- which is why
the new dispatch routes only `lam_selection=="fixed"` bins to it, never
`"aic_local"` ones. Bootstrap's continued last-place finish (0% h4
coverage across all 16 cases, including the previously-untested sigma=20
point) is consistent with every previously-recorded cell in this file.

**sigma=20 specifically** (new, not previously tested in this project):
follows the same qualitative pattern as sigma=50/90 above -- adaptive
Laplace's h4 coverage (25% at sigma=20) is low but non-zero and still
clearly ahead of bootstrap's 0%, consistent with the low-sigma h4
undercoverage already flagged as an open, unresolved gap earlier in this
file (not a new regression from this session's changes).

**NUTS spot-check at sigma=20** (2 truths, same pattern as the earlier
sigma~50/90-ish 2-cell check referenced above; `quick_laplace_uncertainty`
vs `nuts_losvd_uncertainty` at the *same* `lam_used`, same noise draw):

| truth | true sherm | true h4 | point sherm | quick_laplace sherm_std | NUTS sherm_std | quick_laplace h4_std | NUTS h4_std |
|---|---|---|---|---|---|---|---|
| #0 (lam=8) | 18.52 | -0.039 | 11.85 | 5.43 | 2.18 | 0.0019 | 0.0020 |
| #1 (lam=32800) | 19.70 | -0.004 | 14.77 | 5.85 | 0.02 | 0.0004 | 0.0000 |

h4_std tracks closely between the two engines at both truths (as the
earlier spot-check also found). sherm_std does *not* -- NUTS's actual
posterior is narrower than the Laplace approximation's at both truths,
dramatically so at truth #1 (a very high `lam_used`, where heavy
curvature regularization pins the coefficient posterior tightly around
whatever the point estimate landed on). **This points away from, not
toward, "the Laplace approximation is too narrow" as an explanation for
low-sigma undercoverage** -- if anything the reverse holds here. Coverage
at both truths is a **point-estimate bias** story instead: truth #0's
point sherm (11.85) sits ~6.7 km/s below the true 18.52, more than one
quick_laplace sigma away -- checking directly, the reported interval
`[6.42, 17.28]` excludes the true value even though the band itself is
the *wider* of the two engines' bands; NUTS's tighter band excludes it by
more. Truth #1's much larger bias (14.77 vs true 19.70) happens to still
be covered by quick_laplace's wide band but would be missed by NUTS's
near-point posterior. **This confirms/extends finding #1 from the
2026-08-xx entry above** (regularization-driven point-estimate bias, not
approximation quality, is the dominant lever at low sigma) using the
actual gold-standard posterior rather than only a standardized-residual
argument -- n=2 is a spot-check, not a formal sweep, so treat this as
corroborating evidence, not new statistical proof.

**Not done**: a larger-n NUTS sweep at sigma=20 for a formal coverage
percentage (each case costs ~20-40s here, vs quick_laplace's ~3-4s, so a
full n_truths=8 x 2-snr sweep is feasible but was not run, given n=2
already reproduces the same qualitative conclusion as the prior
sigma~50/90-ish check); extending `coverage_sweep.py`'s
`--apply-bias-correction` path (present but not exercised this run -- the
`_corrected` columns in this run's CSVs are identical to `_raw` since the
flag was off, not because correction had no effect); a point-estimate
bias-correction fix for the Laplace engine specifically (the existing
regularization-bias correction machinery is calibrated against bootstrap
only, per the still-open item from the 2026-08-xx entry above -- this
remains the single most concretely-identified lever for closing the
low-sigma coverage gap, now corroborated twice).

---

## 2026-08-07 (later): NGC 5102's radial profile plot is still bad after
## both bugs above were fixed -- a real, deeper, still-open issue, NOT a
## regression from either fix

**PARTIALLY SUPERSEDED, then VINDICATED -- see
`dev_notes/real_data_sigma/FINDINGS.md`'s two 2026-08-08 entries.** The
first entry (morning) correctly diagnosed a *different*, real bug (a
fixed `lam=2.0` bypassing `max_edge_amplitude_frac`'s pedestal filter)
and its fix (restoring `lam_selection="aic_local"`) marked this entry
superseded that same night. That was premature: the wide error bars this
entry chased were never actually re-checked after the aic_local fix
landed, and turned out to still be present -- for a mechanism close to
what this entry's own steps 3/5 already found (chi2/dof far below 1, an
over-generous error budget), confirmed directly on 2026-08-08 (later) at
chi2/dof=0.08-0.13. The rho-measured-from-one-aperture concern (step
five) was real and is now fixed (`fit_losvd_robust(measure_rho=True)`);
the concurrent "no plateau anywhere in the lam grid" finding (step three)
turned out to be a *separate*, also-real multi-start instability problem
in adaptive lam selection itself, fixed by switching to a fixed,
pre-validated lam instead of chasing a plateau that isn't there to find.
See the newer entry for the full resolution and its own honest result:
point estimates improved substantially, but once the error budget is
genuinely correct, the remaining uncertainty width turned out to be real,
not a further-fixable artifact.

**Trigger**: user shared a fresh screenshot of the post-fix radial profile
plot (same notebook, both bugs above already fixed and re-verified) and
called it "still a severe regression... horrible plots" -- V/sigma error
bars nearly as large as the signal itself, and h4 suspiciously uniform
(~0.03-0.05) across nearly the whole galaxy instead of showing real
bin-to-bin structure.

**First, ruled out the two already-fixed bugs as the cause** (both
confirmed still fixed, neither explains this plot):
- h4 shows real bin-to-bin variation (0.011 to 0.206 across 275 bins, not
  uniformly ~1e-4) -- the point-estimate collapse bug is not back.
- Direct check: `quick_laplace_uncertainty` at the fixed lam=2 gives
  *smaller* sherm_std than the adaptive AIC-ensemble alternative would for
  the same real bins (13.25 vs 23.37 at one bin; AIC's own pick is
  lam=0.5, unstable) -- the lam-lottery fix is working as intended, not
  contributing to the large error bars.

**Second, checked whether the large error bars are simply real** (not a
bug at all). A much older, pre-variance-inflation cache on disk
(`clip3.0_iter3_robustTrue`, a naive hard-sigma-clip method) shows
error bars 30-40x tighter (sherm_std median 0.51 vs current 14.3) -- but
that old method's tiny errors are independently documented
(`dev_notes/variance_inflation/FINDINGS.md`) as *underestimates*: real
chi2_red on real MUSE data commonly runs 11-28 under that method, meaning
its reported errors were structurally too small, not a "good" baseline to
reproduce. Direct rho-sensitivity check on a real bin found the large
error bars persist even at `rho=0` (no correlation whitening at all),
ruling that specific knob out as the sole driver too.

**Third: a multi-lam bias scan on 4 real bins (spanning the full radius
range) found something genuinely new and concerning.** Scanning
lam=0.125 to 64 (order-2 penalty) on each bin: **neither sherm nor h4
plateaus anywhere in that range** -- both climb continuously and smoothly
with lam in every bin tested, with sherm reaching unphysical values
(200-330 km/s) at the high end. The current `LOSVD_LAM=2.0` default sits
at an arbitrary point on this continuously-drifting curve, not on any
validated plateau specific to NGC 5102's real data -- it was carried over
from validation on other targets/S-N regimes, never independently
confirmed here. The likely mechanism: **chi2/dof is well below 1 (0.30-
0.50) even at the weakest regularization tested**, meaning the fit
already matches the data better than the reported noise level should
allow -- exactly the AIC-keeps-rewarding-less-regularization signature of
an over-generous error budget, with nothing to anchor a "right" lam.

**Fourth: directly measured whether variance inflation (`fit_losvd_
inflate`, `INFLATION_CAP=8.0`) was the over-generous error source --
ruled out cleanly.** Across all 275 real bins: median inflation factor is
exactly 1.0 (no inflation for the typical pixel); even the per-bin
*maximum* inflation factor only reaches a median of 2.2, and essentially
no bins hit the cap (max 0.3% of pixels in the single worst bin).
Confirmed directly on one bin: inflated vs. original error ratio = 1.00.
**This was the leading hypothesis and it is wrong** -- variance inflation
is doing almost nothing for this target's real bins, so it cannot be
responsible for the over-generous error budget found in step three.

**Fifth: traced the over-generous error budget to `MEASURED_RHO=0.88`
itself, likely conflating genuine detector correlation with unmodeled
model-residual structure.** `05_noise_correlation.ipynb`'s own saved
output shows `measured_rho=0.88` was computed from a fit residual with
**chi2_red=10.2** (raw errors) -- meaning the residual used for the lag-1
ACF measurement is already ~3x larger than photon noise alone would
predict, before rho is even computed. A lag-1 autocorrelation cannot
distinguish genuine pixel-to-pixel detector correlation from smoothly-
varying unmodeled structure (template mismatch, imperfect continuum) --
both look identical to this estimator. Quick recheck requested by the
user: re-measuring rho after applying the Bacon-2017 MUSE variance
correction (`1/0.6^2`) to the input errors first -- **chi2_red drops from
13.2 to 5.07, but rho barely moves (0.892 -> 0.873)**. This shows the
excess correlated residual is *not* primarily an error-scale artifact --
it is real, structured residual (most likely template mismatch / an
imperfect P-spline continuum fit) that survives the correction, and that
same structure is what the lag-1 ACF is (at least partly) picking up as
"correlation."

**The real methodological gap this points to**: `MEASURED_RHO=0.88` was
measured once, from a single high-S/N representative aperture, then
applied *uniformly* to every one of 275 real dartboard bins spanning
`n_spaxels=1` to `n_spaxels=3366` -- a huge S/N range over which a
fixed-magnitude template-mismatch residual represents a very different
*fraction* of the total noise budget bin-to-bin. Applying the same rho
measured at high S/N (where that mismatch is most visible) uniformly to
much lower-S/N bins (where it would be proportionally less prominent)
plausibly over-corrects exactly the low/moderate-S/N bins -- consistent
with the chi2/dof<1 signature found in step three, and with V/sigma error
bars that look implausibly large relative to the recovered signal.

**Sixth: user asked directly whether `MUSE_VARIANCE_CORRECTION_FACTOR =
(1/0.6)^2` itself (not rho) might be the over-correction -- checked
model-free, and the result argues the opposite direction.** Independent
of any fit: measured the empirical pixel-to-pixel scatter (robust,
first-differences-based sigma) in 5 hand-picked "smooth continuum"
windows of the same representative spectrum, compared against the mean
raw pipeline error and the mean Bacon-corrected error in each window.
**The empirical scatter exceeds even the Bacon-corrected error in every
window tested (1.4x-7.5x, most consistently ~1.4-2.4x in 4 of 5
windows)** -- i.e. if anything the correction factor looks too *weak*,
not too aggressive, at least in these windows. Real but unverified
caveat: these windows were chosen by eye, not confirmed line-free -- a
real stellar spectrum has weak unresolved absorption features throughout,
and a first-differences estimator picks up genuine line curvature as if
it were noise, inflating the empirical number (the one 7.5x outlier
window is almost certainly contaminated this way). This result **does
not confirm** the correction factor is fine, but it clearly does not
support "the correction factor is too strong" either, and it complicates
(without resolving) the rho story in step five -- if per-pixel noise is
itself still underestimated even after both corrections, part of what
step five attributed to "structured model-residual correlation
masquerading as rho" could instead be genuine, still-underestimated
white noise, which would not inflate the lag-1 ACF the same way. **This
project does not yet have a clean, disambiguating measurement.**

**Not done / concrete next steps, in order of expected value**:
1. Get a genuinely line-free noise estimate -- either a real blank-sky/
   off-target spaxel (no galaxy model to mismatch or unresolved lines to
   contaminate a first-differences estimator), or a careful line-list
   cross-check before trusting any hand-picked "smooth" window again.
   This single measurement would disambiguate most of steps 3, 5, and 6
   above at once (genuine detector noise vs. model-residual-driven
   correlation vs. correction-factor calibration).
2. Measure rho as a function of S/N (repeat the same lag-1 ACF measurement
   on representative apertures/bins spanning the real S/N range this
   target actually has, not just the one high-S/N aperture) -- directly
   tests whether rho is S/N-dependent, which the step-five mechanism
   predicts, once step 1's contamination question is resolved.
3. Re-run `04_template_calibration.ipynb`'s own continuum/template
   fitting diagnostics on this representative aperture specifically, to
   check whether the imperfect-continuum-fit hypothesis for the excess
   chi2_red/correlated residual has direct, independent support (e.g. a
   visibly structured residual shape reproducible across apertures).
4. Only after 1-3: decide whether a single global rho (and/or the
   current `MUSE_VARIANCE_CORRECTION_FACTOR`) is defensible at all for
   this target, or whether a per-bin/per-S/N-regime approach (or a
   different noise model entirely) is needed -- a real methodology
   decision, not a parameter tweak, and should not be made by picking a
   new fixed constant without the above groundwork.

**Explicitly not concluded**: that `LOSVD_LAM=2.0`/`penalty_order=2`
(the "pre-Phase-0" defaults this notebook was reverted to earlier this
same day) are wrong in themselves -- the multi-lam scan shows the *whole
tested range* lacks a plateau, not that some other specific lam value
would fix it. The root identified here sits upstream, in the noise model
feeding the whitening step, not in the regularization choice itself.
**The notebook is left as-is** (`FORCE_RECOMPUTE=True`, `LOSVD_LAM=2.0`,
`MEASURED_RHO=0.88` unchanged) -- current output is confirmed free of the
two previously-fixed bugs, but should be understood as carrying this
open, likely-real over-correction in its V/sigma (and possibly h3/h4)
error bars until the noise-model work above is done. Do not treat the
current radial profile plot's error bars as final/trustworthy without
that follow-up.

## 2026-08-11 (late): real-data confirmation that low-sigma NGC 5102 bins are lam-selection lottery, and the reported error bar badly under-covers it

**Trigger**: user looked at `07_full_galaxy_analysis.ipynb`'s current
radial profile (fixed `lam=4096`/`penalty_order=3`, no continuum -- see
`dev_notes/real_data_sigma/FINDINGS.md`'s 2026-08-08/09 entries for why
fixed, not adaptive, is the current choice there) and flagged a real,
specific worry: the profile's sigma trends down toward the center to
~15-30 km/s with a sharp discontinuity, not a smooth trend, and asked
whether adaptive selection's own edge-pedestal guard
(`max_edge_amplitude_frac`, built earlier this session for N4751) has an
analogue that guards against the *opposite* failure -- over-regularization
quietly deflating sigma -- since MUSE's own instrumental resolution in
this fit window (8400-8800A, CaT triplet) works out to sigma~36-38 km/s
(`kinextract.lsf.muse_lsf_fwhm`), i.e. these bins are being measured at
or below the instrument's own resolution floor.

**Investigated three ways**: (1) confirmed the MUSE-resolution-floor
number above directly; (2) re-read this project's own Stage 22 sigma=20
finding (`dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`,
2026-07-26: "m_eff=3.000 at every lam... AIC criterion flat...
selection is a lottery... median residual >20% at EVERY lam") -- already
documented, but never directly demonstrated on this run's own real,
flagged bins; (3) ran a real-data lam ladder (`lam` = 64/256/1024/4096/
16384/65536 fixed, plus `aic_local`) on the 5 lowest-sigma bins from this
notebook's own freshly-regenerated cache (`full_galaxy_..._lam4096_
order3_caii.pkl`, 275/275 bins, current settings).

**Result: the hypothesis (fixed lam=4096 is systematically deflating
sigma) is NOT supported -- something more specific, and arguably worse
for public-release trustworthiness, is.** Sigma vs. lam is **not
monotonic** in any of the 5 bins -- e.g. bin 197: 6.1, 7.0, 6.8, 5.0,
14.7, 13.3 km/s across the ladder (lam increasing); bin 263: 1.7, 4.7,
7.8, **17.8**, 10.7, 12.3 (the *current* `lam=4096` choice gives the
*highest* value in that bin's own ladder, not the lowest). `lam=4096`
is sometimes the min, sometimes the max, sometimes in between across
these 5 bins -- there is no consistent direction, and `lam_selection=
"aic_local"` does not reliably land somewhere better: it picked
`lam=512` for bin 183 (giving 7.1 km/s, well outside that bin's own
64-65536 fixed-ladder range of 10.7-27.1) and `lam=64` for bin 238 (the
ladder's own lightest candidate). **This is exactly what Stage 22's
"selection is a lottery" finding predicts, now demonstrated on this
run's own real bins, not just a synthetic sigma=20 mock.**

**The more serious finding: reported Laplace `sherm_std` badly
under-covers this real instability.** For the same 5 bins, the ladder's
own sigma range spans ~9-16 km/s (e.g. bin 197: 5.0-14.7, a 9.7 km/s
swing; bin 195: 8.0-18.4, a 10.4 km/s swing including the adaptive
result) -- but this run's own reported `sherm_std` for these bins is
only **1.1-2.1 km/s** (13-41% relative to the point estimate, but
5-10x smaller than the actual lam-to-lam swing). Laplace's own posterior
-- computed as a curvature-based spread *around whichever single lam won
the criterion* -- has no mechanism to represent uncertainty *over which
lam should have won in the first place*, and that omitted term is the
dominant one here, not a second-order correction. This is the same
mechanism Stage 27 (`dev_notes/VALIDATION_CAMPAIGN.md`) already flagged
in the aggregate ("resampling around a biased point estimate... interval
can have closer-to-correct width and still be centered wrong") and the
same "not yet root-caused" gap README.md's own Known Limitations already
lists for `aic_local` at low sigma -- this entry is the first time either
has been pinned to specific, real, named bins with a concrete before/
after number, not just a suspected mechanism.

**What this is not**: not a new bug introduced this session (the
mechanism, and the open coverage problem, both predate tonight -- Stage
22/27 already flagged the pieces separately); not fixed by switching
07 to adaptive selection (the ladder shows adaptive is an equally noisy
draw from the same lottery here, not a systematically better one, and
switching would also reintroduce the real multi-start instability
`dev_notes/real_data_sigma/FINDINGS.md`'s 2026-08-08/09 entries already
root-caused and moved away from on this exact target); not something a
stronger or weaker single fixed `lam` would fix either (the ladder has
no plateau in this regime, matching the analogous "no plateau" finding
in the entry above this one).

**Recommendation, not yet implemented**: make the limitation visible
per-bin rather than only in prose. A cheap, real diagnostic is already
sitting in the data this notebook already computes: `losvd_objective_gap`
(multi-start disagreement at the winning lam) and/or a 2-3-point lam
perturbation check (refit at `0.25x`/`4x` the winning lam, see how much
`sherm` moves) could flag "regularization-limited, treat as illustrative"
bins directly on the radial profile plot, the same spirit as this
session's own `edge_amplitude_frac`-based flagging for the opposite
(pedestal) failure mode -- but this is a genuinely new, non-trivial
diagnostic to validate (false-positive/negative rate against a known
grid, not just "seems to flag the bins we already suspect"), not a
same-night fix. Deferred; see `07_full_galaxy_analysis.ipynb`'s own
Section 7 for where the honest caveat was added instead.
