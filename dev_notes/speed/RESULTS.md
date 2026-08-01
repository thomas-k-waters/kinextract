# Speed regression: profiled and fixed (2026-07-27)

**Symptom** (user-reported): a single default `fit_losvd` on
`01_basic_mock_fit.ipynb`'s standard case took ~38 s; 50 bootstraps ~4 min.

## Where the time actually went

Profiled with call counters (`prof01_where_time_goes.py`), 3-template
E-MILES mock, 321 px, 149 velocity bins, 19 spline coefficients:

- **100%** of wall time was inside `losvd.fit_log_spline_map`.
- One default fit ran **114 L-BFGS-B optimizations** across 38 calls.
- Structure: 6 lam candidates x a full alternating fit each (for `m_eff`),
  **plus a 7th full refit at the winning lam**, and every one of the ~5
  alternating iterations inside each fit re-ran a *cold 3-way multi-start
  search*.

## Two fixes, both structural (no accuracy knob touched)

1. **Warm-start the LOSVD sub-fit across alternating iterations.** The
   multi-start pool exists to choose the right basin; that only has to
   happen on the first iteration, because `w` moves only slightly
   afterwards. Iterations 2+ now start from the previous iteration's
   coefficients. The first iteration's own multi-start gap is still what
   gets reported as `losvd_objective_gap`, since that is where the basin
   was actually chosen. Exposed as `losvd_warm_start` (default True) on
   `fit_losvd_basis`/`_alternate_once`, matching the existing
   `losvd_single_start` ablation-flag pattern.
2. **Reuse the selection scan's own fit at the winning lam.**
   `select_lambda_aic_local` must fit at every candidate lam anyway;
   `fit_losvd` then refit at the winner, repeating one of those fits
   exactly. `AICSelectionResult` now carries `best_fit` and `fit_losvd`
   reuses it (guarded off for the continuum path, which selection does not
   support).

## Result

| | before | after |
|---|---|---|
| one default `fit_losvd` | 34.1 s | **5.0 s** |
| L-BFGS-B optimizations | 114 | **36** |
| full alternating fits | 7 | 6 |

**6.8x faster.**

## Quality impact: measured, not assumed

Warm-starting changes which optimum the alternation settles into, so
results shift slightly (real data <0.5 km/s; the standard mock's sigma
moved 50.93 -> 48.98 while its shape residual *improved* 3.4% -> 2.1%).
Paired A/B on identical noise draws, 12 seeds x sigma in {50,140,300}
(`verify02_warmstart_ab.py`), scored against ground truth:

| sigma | d\|V bias\| | d\|sigma bias\| | d max-resid |
|---|---|---|---|
| 50  | +0.13 | -0.22 | -0.05% |
| 140 | +0.09 | -0.53 | -0.31% |
| 300 | -0.10 | +0.97 | +0.15% |

Mixed sign, no systematic direction, and every difference is far below the
fits' own scatter at the same points (mean \|V bias\| 2.4-9.4 km/s, mean
\|sigma bias\| 3-8 km/s). **Quality-neutral within noise.**

## Not done (further speed available, needs its own validation)

- `fit_log_spline_map` re-converts `basis`/`D2`/`A_w`/`y_w` to JAX arrays
  on every call (24x per fit). Bit-exact to hoist, but invasive.

# Warm-starting across the lam grid + parallel multi-bin fitting (2026-07-30)

The item flagged above as "not done" -- `select_lambda_aic_local`'s 6-point
scan cold-starting every candidate `lam` independently, previously measured
at ~9.5x a single fixed-`lam` fit -- plus a second, independent lever this
package never had: fitting many spatial bins (a full galaxy) in parallel
across processes, since every bin's fit is completely independent.

## Fix 1: `warm_start_lam_grid` on `select_lambda_aic_local` (now the default)

Same idea as the iteration-level warm-start above, one level up: seed each
sorted `lam` grid point's fit from its immediate neighbour's converged
coefficients (`multi_template._alternate_once`'s new `initial_coeffs`
parameter, threaded through `fit_losvd_basis`/`fit_losvd`/
`estimate_m_eff_local`), so only the smallest `lam` still pays a full cold
multi-start search.

**First validation pass found a real regression, not just a speed/quality
tradeoff**: `test_multi_start_escapes_a_known_bad_local_minimum` (`tests/
test_losvd.py`) started failing. Root cause: the *winning* `lam`'s own fit
is essentially never the smallest (coldest) grid point, so under the naive
warm-started scan its `losvd_objective_gap` (the "did multi-start actually
matter for this fit" diagnostic `fit_log_spline_map` reports) came back
`None` instead of a real value -- the winner had been warm-started, not
multi-started, even though it's the one result callers actually use.
**Fixed** by confirming the winner with one fresh, fully cold multi-start
fit after the scan picks it (`select_lambda_aic_local`'s own code, guarded
on `best_fit.get("losvd_objective_gap") is None`) -- costs at most one
extra full fit per selection call, not one per grid point, so this keeps
most of the lever's speedup while restoring the diagnostic (and, as a
side effect, makes the returned point estimate itself a genuine
from-scratch multi-start result again, not a warm-started shortcut).

**Validated quality-neutral-to-better with the winner-confirmation fix in
place** (`verify03_lamgrid_warmstart_ab.py`, same paired-A/B methodology
as above, 12 seeds x sigma in {50,140,300}):

| sigma | d\|V bias\| | d\|sigma bias\| | d max-resid | best_lam mismatches (of 12) | scan speedup |
|---|---|---|---|---|---|
| 50  | +0.00 | +0.00 | +0.00% | 0  | 1.8x |
| 140 | +0.00 | +0.00 | +0.00% | 0  | 1.8x |
| 300 | +0.16 | -0.62 | -0.48% | 1  | 1.6x |

At sigma 50/140 the result is now **bit-identical** to the original
fully-cold scan (0/12 `best_lam` mismatches, and the confirmed winner is a
real cold fit either way) -- a stronger guarantee than "quality-neutral,"
at the cost of the scan speedup dropping from the earlier (regression-
carrying) 3.7-4.2x measurement to a more modest but unconditionally-safe
1.6-1.8x. The one sigma=300 mismatch still recovered a *better* sigma/
shape residual, not worse. Now `select_lambda_aic_local`'s default
(`warm_start_lam_grid=True`); pass `False` to reproduce the original
independently-cold-started behavior.

## Fix 2: `fit_losvd_parallel` -- fit many bins across worker processes

New `multi_template.fit_losvd_parallel(A_stack_list, y_w_list, cfg_list,
wave_fit=None, n_jobs=None)`, exported from the package top level. Every
bin is embarrassingly parallel (no shared state) -- this is pure wall-clock
orchestration, zero accuracy cost (bit-identical to the same bins fit
serially, confirmed via `np.allclose`). Uses `concurrent.futures.
ProcessPoolExecutor` with `spawn` (not `fork`, the documented-safe choice
for JAX) and pins each worker's BLAS/OpenMP threads to 1 before import
(`_limit_worker_threads`) -- without this, `n_jobs` processes each
defaulting to all-core BLAS threading oversubscribes the machine by a
factor of `n_jobs` (measured directly: forcing single-threaded BLAS
improved an 8-worker/40-bin case from 1.53x to 1.70x).

**Machine matters more than expected.** On the development M1 (4
performance + 4 efficiency cores), `n_jobs=4` measurably *beat* `n_jobs=8`
(21.3s vs 24.8s for 80 bins) -- using the efficiency cores made things
worse, not better. No portable way to auto-detect performance-core count
without a new dependency, so `n_jobs` defaults to `os.cpu_count()` (right
for homogeneous cloud/cluster hardware) but is documented as worth
measuring explicitly on heterogeneous consumer hardware.

## Combined result: the actual user complaint ("over an hour for a full galaxy")

100-bin synthetic-cube benchmark (`FitConfig` order-3 default, adaptive lam
selection, 3-E-MILES-template mock):

| | time for 100 bins |
|---|---|
| pre-2026-07-30 (this file's earlier fix only) | ~6.0 min (extrapolated from a 25-bin sample) |
| + `warm_start_lam_grid` incl. winner-confirmation fix (serial) | 2.1 min -- **2.9x** |
| + `fit_losvd_parallel` (`n_jobs=4`) | 1.1 min -- **5.3x** |

(Numbers here are the final ones, measured *after* the winner-confirmation
fix above -- an earlier draft of this table, before that fix existed, quoted
a larger 5.1x/9.3x from the regression-carrying version of
`warm_start_lam_grid`; corrected here.)

A real galaxy analysis at ~150-180 bins (this project's own dartboard
notebooks) now finishes in a bit over a minute of pure fitting time, not
"over an hour" -- combined with the prior session's 6.8x per-fit fix, the
total improvement across both sessions is substantially larger still.
Bootstrap/bias-correction uncertainty estimation (many refits per bin) was
not re-benchmarked here -- `fit_losvd_parallel` parallelizes *across
bins*, not across bootstrap replicates within one bin, so a bootstrapped
full-galaxy run would need its own combined benchmark before quoting a
number.

**Honest comparison to pPXF, still not closed.** A single pPXF `moments=4`
fit on the same single-template spectrum measured at ~0.010 s -- roughly
two orders of magnitude faster than kinextract_lite's own ~1.2s/bin even
after every fix above. pPXF's much simpler fixed-4-parameter optimization
is fundamentally cheaper than kinextract_lite's non-parametric spline plus
adaptive regularization search; today's work closes most of the
*practical* gap for full-galaxy use (a couple of minutes, not hours)
without pretending to have closed the *per-fit* gap, which is a real,
structural difference between the two methods' cost, not a bug. See
`dev_notes/ppxf_comparison/FINDINGS.md` for the accompanying accuracy side
of this tradeoff (kinextract_lite wins on sigma bias in most regimes
despite being slower per fit).

# `losvd_single_start` was silently ~3-5x SLOWER than multi-start (found and fixed same day, 2026-07-30)

Found while investigating a user report that bootstrap fits with real
E-MILES templates (4 templates, genuine non-Gaussian truth) were taking
far longer than expected -- not stalled, but much slower than the mock
regime the earlier fixes on this page were validated against.

## Root cause

`FitConfig.losvd_single_start=True` (`bootstrap_losvd`'s own
`replicate_single_start=True` default forces this for every replicate) is
meant to ablate only the multi-start *search* -- fit once from a
deterministic ridge start instead of `fit_log_spline_map`'s 3-candidate
pool. Instead, `_alternate_once`'s branch fired on **every** alternating
iteration, not just the first:

```python
if losvd_single_start:
    coeffs0 = default_initial_coeffs(...)   # every iteration, not just #0
```

This silently disabled the iteration-to-iteration warm-start mechanism
(this file's very first fix, 2026-07-27) for the entire single-start
path: every iteration re-optimized from a generic ridge guess instead of
the previous iteration's own already-converged coefficients. Confirmed
directly on a real-templates (4 E-MILES templates), order-3-default,
non-Gaussian-truth (h3=0.05, h4=0.02) mock:

| | time per fit |
|---|---|
| multi-start (`losvd_single_start=False`) | 4.5-5.9s |
| single-start, buggy (forced every iteration) | 14.5-15.7s |
| single-start, fixed (forced iteration 0 only) | **3.2-3.4s** |

**~4.5x faster than the buggy version, and now correctly faster than
multi-start too** (matching the flag's own intent and its docstring's
original "~2.5x faster" claim, which this bug had quietly invalidated for
any case relying on the current order-3/adaptive-lam-grid default).

## Fix

`_alternate_once` now only takes the `losvd_single_start` branch when
`iteration == 0`; every later iteration warm-starts exactly as it would
with `losvd_single_start=False`. No accuracy change -- iteration 0's
starting point is identical either way, only iterations 1+ changed, and
they were always *supposed* to warm-start regardless of this flag (see
`FitConfig.losvd_single_start`'s own updated docstring).

## Lesson

The two per-fit timing numbers earlier on this page (the `warm_start_lam_grid`
scan benchmarks) were measured with `losvd_single_start` at its default
(`False`), so they're unaffected by this bug -- but `bootstrap_losvd`'s
own replicates, which default to `replicate_single_start=True`, were
hitting this exact broken path on every single call. This is why a user's
own bootstrap runs looked far slower than the mock-based benchmarks above
suggested they should be -- not a regression from today's other changes
(directly ruled out via A/B before finding this), but a latent bug this
investigation happened to surface.
