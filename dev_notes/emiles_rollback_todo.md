# Notebook 03 rollback (2026-07-13): what was reverted, and why

`examples/notebooks/03_real_data_muse.ipynb` was rolled back from an E-MILES-template
+ auto-xlam design to (most of) the recipe last pushed to GitHub (`main`,
`b183b974`), because the E-MILES version's recovered LOSVD had a genuine
double-peaked core and a spurious wing "pedestal" that several in-place fix attempts
could not resolve without trading it for a different problem. See
`dev_notes/real_data_validation_muse_stis.md` for the full investigation (redshift
fix, nbins investigation, `xlam_wing_shrink` attempts and why they were rejected,
the residual-defect check).

This file exists so the E-MILES migration -- which was real, motivated work, not a
mistake -- can be reintroduced properly once the wing-pedestal issue is actually
understood, rather than lost. **Do not just flip these back without re-validating
against `dev_notes/real_data_validation_muse_stis.md`'s legacy comparison
(`bin0105spmc_in`) first** -- that's exactly the check that was skipped the first
time and cost real time to catch after the fact.

## What was rolled back in the notebook (all in `03_real_data_muse.ipynb`)

1. **Templates**: E-MILES SSP grid (`template_npz_file`/`template_npz_select`,
   `kinextract_emiles_grid.npz`) -> individual real-star MUSE library
   (`template_list_file`/`template_dir`, `examples/data/muse/Tlist`).
   - Motivation for the original switch (still valid, not disproven): a dense,
     physically-smooth SSP grid avoids template/LOSVD degeneracy at broad sigma
     that a library of ~35 idiosyncratic individual stars can show -- validated in
     a synthetic-mock stress-test sweep.
   - Why it's suspected in the real-MUSE regression: switching back to MUSE-star
     templates (with everything else held fixed -- corrected redshift, narrow
     window) made the wing pedestal disappear entirely. This was the single
     highest-leverage change found in the rollback investigation. Not fully
     root-caused *why* E-MILES specifically triggers it on this bin -- worth
     understanding before reintroducing.
2. **Regularization strategy**: `xlam_auto=True` (discrepancy-principle search,
   package default) -> `xlam_auto=False`, fixed `xlam=1e7` (hand-tuned by comparing
   directly against the legacy pipeline's own LOSVD, `bin0105spmc_in`).
   - With MUSE-star templates, `xlam_auto=True` undersmooths this spectrum: a
     small spiky secondary bump near v~-50 km/s, peak visibly narrower/taller than
     legacy (sigma ~45 vs. legacy's 54.8). This is presumably a mis-calibration of
     the discrepancy-principle target for this specific template library/window
     combination, not a fundamental problem with the auto-search itself.
3. **Error handling**: `use_spectrum_errors=True` (file's own error column) ->
   `gal_errors=flux/50` uniform S/N guess + one error-rescaling pass (refit at
   `errors *= sqrt(chi2_red)`), matching the last-pushed notebook's own approach.
4. **`sigl`**: 60.0 -> 100.0 (matches the last-pushed notebook; not independently
   re-tested against 60 in this rollback).
5. **`map_maxiter`**: kept at an increased value (20000 -> 40000) -- not itself a
   rollback, but required because the much larger fixed `xlam=1e7` needs more
   L-BFGS-B iterations to reach its own convergence criterion than the previous
   auto-selected (smaller) xlam did.

## What was deliberately KEPT from the (otherwise rolled-back) later work

- **`zgal=0.0016181`** (not the last-pushed notebook's generic-NED `0.001556`) --
  the redshift the legacy Fortran pipeline actually used for this exact spectrum
  (its own `.lastinput` file). Independently verified correct; biases V by ~19 km/s
  if wrong. Keep this regardless of what else gets reintroduced.
- **`FitConfig.n_losvd_bins=29`** (package default) -- this was never actually the
  cause of the wing pedestal (confirmed: present at both 29 and 89 bins with
  E-MILES templates); no reason to change it for this notebook specifically. The
  broader `n_losvd_bins=89` package-default story is unrelated and unaffected by
  this rollback -- see `dev_notes/nbins_xlam_regression_history.md`.
- **`map_ftol=1e-8`** package default (not explicitly set in the notebook, just
  inherited) -- established earlier in the same session as a general improvement
  (fixes flaky false-negative `success=False` flags at large xlam); unrelated to
  the wing-pedestal investigation and not implicated in it.
- **`LOSVDErrorEstimator.summarize()`-based bootstrap display** (MAP-anchored error
  bounds, matching the legacy `.mcfit2` convention) -- a presentation improvement
  independent of the regression; kept as-is.
- **`FitConfig.xlam_wing_shrink`/`xlam_wing_shrink_sfac`** -- the opt-in amplitude
  penalty added and ultimately not used for this notebook. Left in the codebase,
  off by default, documented with the negative results from trying it (see its own
  docstring and `real_data_validation_muse_stis.md`).

## Honest residual issue with the rolled-back state

Bootstrap sigma uncertainty (±1.2 km/s) is noticeably tighter than the legacy
pipeline's own (±2.0 km/s) at the large fixed `xlam=1e7` this recipe needs -- not
the severe collapse seen with `xlam_wing_shrink` + `xlam_auto` (which went to
<3% of a healthy value), but worth keeping an eye on. Plausibly just a property of
a large, fixed (not re-searched per bootstrap replicate) regularization strength;
not investigated further given time constraints.

## Suggested path back to E-MILES (when there's time)

1. Reproduce the wing-pedestal failure on a fresh mock or two using E-MILES
   templates specifically (not just this one real bin) to see if it's
   spectrum-specific or a general property of that template family in joint mode.
2. If general: investigate *why* -- template mixture degeneracy interacting with
   the P-spline continuum co-fit is the leading suspect (E-MILES's SSPs are smooth
   and mutually similar in a way individual real stars aren't, which could let the
   optimizer trade continuum shape against LOSVD wings more easily).
3. Re-validate any reintroduced auto-xlam behavior against `bin0105spmc_in`
   directly (shape, not just GH moments -- see the stage-1 mistake in
   `real_data_validation_muse_stis.md`), and check bootstrap error bars before
   calling anything fixed (see the stage-2/3 mistakes in the same file).
