# Making the regularization-bias correction trustworthy: a strategy

**Status: proposed, not started.** Written 2026-07 after `bootstrap_losvd`'s
`apply_bias_correction` was flipped to default `False` (see
`VALIDATION_CAMPAIGN.md`, the stage documenting that decision) following two
distinct overcorrection failure modes found under real testing in short
succession. This document is the answer to "how do we make this trustworthy
enough to turn back on," not an implementation -- no code here yet.

## Why "helps on average" was never actually the right bar

Every validation of this correction so far (this project's own history, and
this session's) checked something like "does it reduce *median* bias across
a grid of cells." That is the wrong acceptance criterion for a tool meant to
run unsupervised on real data with no ground truth. Two concrete findings
this session make the point directly:

1. The `m_eff`-axis table's own overcorrection at `lam=0.5`
   (`01_basic_mock_fit.ipynb`'s own default mock: corrects 52.9 km/s sigma
   to 35.5 against a true 50, ~5x worse than raw) happened in a regime that
   would still show up as "helps on average" in a coarse aggregate check --
   it's a *local*, not global, failure.
2. An even earlier finding (documented in this project's own coverage
   testing, before this session) showed a Pinkney-style correction with
   **0% empirical coverage** at a specific (sigma, h4_true) cell despite
   passing an aggregate bias-reduction check elsewhere.

A correction that is right on average and confidently wrong in specific,
unflagged corners is worse than no correction, because a real user has no
way to know which regime they're in. **The actual bar is: every individual
cell in the calibration's own domain must pass, not just the average across
cells -- and cells that can't be made to pass must be explicitly excluded
from the domain, not silently left in.**

## The four gaps, and what closes each one

### 1. Grid density and axis independence (DONE -- data was already dense; the query design was the actual bug)

The sigma=80→140 gap (already found and fixed once this session) and the
old `m_eff`-axis table's thinner-per-cell sampling (found in an earlier
session, the direct cause of the lam=0.5 overcorrection) were originally
diagnosed as the same class of bug: spreading a fixed calibration-compute
budget across more grid points or axes reduces the reps-per-cell
everywhere. The proposed fix below called for building a full,
densely-sampled 2-D `(sigma, S/N)` table at each of a small number of
canonical `lam` values (this package's real grid, not the stale
`[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]` first written here -- see
`config.py`'s order-3 default: `[1, 8, 64, 512, 4096, 32768]`), each with
the same per-cell rep count the plain table uses, not divided by 6.

**2026-07-30 finding: that dense data already existed** (`dev_notes/
calibration/meff_checkpoints/`, built 2026-07-28/29, `n_reps=30` per
`(sigma, S/N, lam)` cell, confirmed by direct inspection of the shipped
table's own row count). Running the 4-gate test against it still failed
gate 1. **The actual bug was the query-time interpolation, not data
density**: the old design stored `m_eff` as a continuous third
interpolation axis and queried via 3-D `scipy.interpolate.griddata` over
`(recovered_sigma, snr, m_eff)` jointly -- but `m_eff` depends on both
`lam` and the local `(sigma, S/N)` regime, so two calibration points with
similar `m_eff` for unrelated reasons could get interpolated together.
**Fixed** by implementing the query design this section originally
proposed but the code never actually built: `PerLamCalibration`
(`bias_correction.py`) -- one 2-D `(sigma, S/N)` `BiasCalibration` per
canonical `lam` value, with nearest-two-`lam` log-linear interpolation at
query time. Reuses the existing dense data via
`load_per_lam_calibration_table` (no rebuild). Gate 1 has not yet been
rerun against this redesign -- that's the next real step, not another
data rebuild.

### 2. Truth-domain coverage (h3, h4 != 0)

The h4 correction is an additive floor tabulated at `h4_true=0` -- already
known to overshoot for nonzero true h4 (this project's own earlier "wing"
table, `regularization_bias_emiles_h4wing.txt`, covers exactly one
`h4_true=0.15` point at 4 sigma values, not a real grid). h3 has no
calibration at all beyond "not needed since true h3=0 shows no bias" --
which says nothing about nonzero-truth h3 recovery bias.

Proposed fix: extend the calibration sweep itself to vary `h3_true`/
`h4_true` over a real grid (not one extra point), e.g. `h4_true in
{0.0, 0.05, 0.10, 0.15, 0.20}` crossed with the existing sigma/S-N grid,
and same for h3. This is a straightforward extension of
`calibrate_bias.run_calibration_sweep` (already template-library-agnostic,
per its own docstring) -- the work is compute, not new design.

### 3. Template-mixture/library generality (the hard one)

Real users pick their own templates; the shipped calibration is trained on
one fixed Dirichlet-mixture recipe over one 40-template E-MILES subset.
This project's own `06_template_mismatch_diagnostic.ipynb` already computes
a real, meaningful diagnostic for a *different* purpose -- the per-step NNLS
design matrix's own **condition number** -- that plausibly matters here too:
this session found (`02_realistic_mock_fit.ipynb`) that a genuinely
degenerate 4-template mixture (condition number ~1785) produces
qualitatively different weight-recovery behavior than a well-conditioned
one, and it would be surprising if that had *no* effect on the sigma/h4
bias surface a correction is calibrated against.

Proposed fix, in order of effort:
- **(a) Test first, don't assume.** Run the existing calibration sweep
  machinery at a few condition-number regimes (well-conditioned single
  template through deliberately-collinear multi-template, already have code
  for both from this session's own diagnostics) and measure directly
  whether the bias surface actually shifts with conditioning, and by how
  much, before deciding this needs its own axis. If the effect is small
  relative to the sigma/S-N/lam effects already calibrated, the generic
  table may be more broadly trustworthy than assumed -- but that's an
  empirical question, not a guess either way.
- **(b) If it matters:** add design-matrix condition number as a third
  lookup axis (alongside sigma, S/N), the same way `m_eff` was added for
  `lam` -- same density-per-cell discipline as fix #1 above, not spread
  thin.
- **(c) Regardless of (a)/(b): make "build your own calibration" a first-
  class, actively-recommended path, not a fallback.** `calibrate_bias.py`'s
  own module docstring already frames `build_emiles_calibration.py` as "the
  literal, adaptable recipe" -- but no notebook currently tells a user with
  a materially different template library to actually go run it. Add that
  guidance explicitly wherever the correction is discussed.

### 4. A per-query trust signal (new, and probably the single highest-value piece)

None of the above eliminates residual risk -- extending coverage always
leaves *some* boundary. What actually addresses the user's real concern
("no way to know truth on real data") is not "the correction is right
everywhere" (unachievable) but **"the correction tells you, quantitatively,
when to trust it less."**

Concretely: `correct_moments` already clips to the calibration grid's own
range with a `UserWarning` outside it -- a binary in/out check. Extend this
to a **continuous confidence signal**: distance (in normalized
sigma/S-N/lam/condition-number units) from the query point to the nearest
well-sampled calibration cell, returned alongside the correction
(`sig_bias_confidence` or similar), not just used internally. A real-data
notebook could then print "this correction's own confidence is low for this
bin" instead of presenting a number with no indication of how far outside
the validated surface it sits. This is new machinery, not a bigger sweep --
worth prototyping early, in parallel with 1-3, since it changes what
"trustworthy" even means (a correction that honestly flags its own
unreliable regions is more trustworthy than one that doesn't, even before
those regions are fixed).

## The acceptance gate (applies to every table before it becomes a default)

No calibration table changes `apply_bias_correction`'s default, or replaces
a shipped table, until it passes all of:

1. **Per-cell bias check, not aggregate.** Every individual calibration
   cell (not the median/mean across cells) shows `|bias_corrected| <
   |bias_raw|`, or the cell is explicitly excluded from the table's
   declared domain (documented, not silently interpolated over).
2. **Independent held-out coverage check.** Fresh mocks with seeds disjoint
   from the calibration's own training seeds (already this project's own
   practice, e.g. `validate_emiles_calibration.py`'s `WEIGHT_SEED_BASE`
   discipline) -- but checking **empirical coverage** of the reported
   interval (does the ±1 sigma_corrected band actually contain truth ~68%
   of the time across many independent draws per cell), not just point-bias
   reduction. This is the exact test that would have caught the 0%-coverage
   finding before it shipped.
3. **Worst-cell regression check.** The single worst-performing cell in the
   held-out validation must not be worse than the single worst-performing
   *raw* (uncorrected) cell -- i.e., the correction is not allowed to make
   the worst case worse, even if it helps everywhere else on average.
4. **A regression test per bug already found.** `01_basic_mock_fit.ipynb`'s
   own lam=0.5 case and the m_eff-table-switch case both become permanent
   entries in `tests/test_bias_correction.py` (or a new
   `tests/test_calibration_trust.py`) -- specific, numbered regressions,
   not just "the aggregate number looks fine."

## Where the OG-kinextract comparison harness fits in

`kinextract_test_suite/notebooks/head_to_head_comparison/` already runs
both codebases against identical synthetic truth across a dense sigma/S-N/
shape/template grid, with curves saved (once the curve-saving extension
lands). Two direct uses for calibration trust-building, beyond its original
comparison purpose:
- It's a second, independent source of held-out validation cells (gate #2
  above) at essentially no extra harness-building cost, since the mock
  generator and grid already exist.
- Comparing kinextract_lite's *corrected* result against OG kinextract's
  own (differently regularized) result, on the same synthetic truth, is a
  useful cross-check on whether a correction is moving toward or away from
  a second, independently-biased estimate -- not a substitute for knowing
  the real truth, but a genuine extra data point.

## Suggested phase order (not a commitment, a starting proposal)

1. Fix #1 (dense per-lam tables) + the acceptance-gate test infrastructure
   (#4 above) + regression tests for the two bugs already found. This alone
   would have caught both.
2. Fix #2 (h3/h4 truth-domain grid), gated by the same acceptance criteria.
3. Fix #3(a) (test whether conditioning matters) -- decide 3(b) vs. 3(c)
   only once that's answered empirically.
4. Fix #4 (per-query confidence signal) -- can start in parallel with 1-3,
   since it's new machinery rather than a bigger sweep.
5. Only once a table passes the full acceptance gate: flip
   `apply_bias_correction`'s default back to `True`, and only for the
   specific domain that table actually passed for -- not a blanket
   re-enable.

## 2026-07-31/08-01: h3 sweep infrastructure added, no table built yet

`calibrate_bias.run_calibration_sweep` gained an `h3_true` parameter
(mirroring `h4_true`'s existing pattern exactly) so a future session can
build an h3 calibration table -- previously the sweep infrastructure
literally could not generate a nonzero-h3 truth grid at all
(`dev_notes/kinextract-open-problems` item 5). Unlike `h4_true`, `h3_true`
also shifts the true V away from `v_true` by the exact closed form
`sqrt(3)*sigma*h3` (`dev_notes/validation_suite/FINDINGS.md`) -- handled by
generalizing `_true_moments` to accept both `h3_true`/`h4_true` and always
scoring `v_bias`/`sig_bias` against the numerically-shifted true (V, sigma),
not the nominal `v_true`/`sigma`. Verified: `_true_moments(0, sigma=100,
h3_true=0.05)` gives true_v=8.22, matching the closed form's 8.66 to within
~5% (the small residual gap is the known clipping effect,
`dev_notes/validation_suite/FINDINGS.md`'s own <=4%-of-mass-at-extremes
caveat -- not a bug). `run_calibration_sweep` runs end-to-end with
`h3_true` set (new tests in `tests/test_calibrate_bias.py`); `save_
calibration_table`/`load_calibration_table` were deliberately NOT extended
to write/read the new `h3_mean`/`h3_std` columns yet -- out of scope for
this session's time budget, and the production h4/main tables must not
change format underneath already-shipped calibration files.

**Not done, left for a future session**: an actual h3 calibration sweep at
production scale (a `(sigma, S/N)` grid x several `h3_true` values x
enough seeds/lam values to be trustworthy -- likely comparable cost to the
existing h4wing/meff table builds, hours not minutes) and the joint
`(V, h3)` correction logic in `bias_correction.py` itself (currently only
shifts `sigma`/`h4`, per `correct_moments`'s own docstring). This entry
exists so a future session doesn't have to re-derive that the joint V-h3
coupling is real, exact-to-clipping, and already handled correctly by
`_true_moments` -- only the production sweep and the correction-application
side remain.
