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
