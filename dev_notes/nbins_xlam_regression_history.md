# n_losvd_bins / auto-xlam regression: full history

This documents a multi-step regression-and-fix cycle around
`FitConfig.n_losvd_bins` and `FitConfig.xlam_auto_grid` so the reasoning
behind the current defaults doesn't live only in chat history. Read this
before changing either field again.

## Timeline

1. **29 -> 89 (initial change).** `n_losvd_bins` was raised from the
   legacy-Fortran-matching 29 to 89 to fix a real bias: a coarse LOSVD grid
   systematically overestimates sigma and overshoots the recovered LOSVD's
   peak height at fixed `losvd_vmin`/`losvd_vmax` (bin width becomes a
   large fraction of sigma). Validated via a synthetic E-MILES sweep,
   sigma=30-350 -- see `dev_notes/v_bias_remediation_plan.md` for the
   validation setup and the (separate, still-open) joint-mode V-bias
   findings from that same sweep. Shipped without re-checking the
   auto-xlam discrepancy-principle search (`xlam_auto_grid`,
   `xlam_discrepancy_nsigma`) against the new, 3x-larger bin count.
2. **The notebook-02 catastrophic regression.** A separate, same-session
   tuning change -- `xlam_discrepancy_nsigma` overridden from the package
   default 0.3 to 1.0 in `examples/notebooks/02_realistic_mock_fit.ipynb`,
   meant to reduce a mild LOSVD wiggle -- interacted catastrophically with
   `n_losvd_bins=89`: the auto-xlam search's bracket-expansion loop ran
   away to `xlam ~ 5e7` and the fit failed to converge
   (`result.success=False`). Confirmed via direct 2x2 bisection
   (nbins in {29,89} x nsigma in {0.3,1.0}) that only the (89, 1.0)
   combination is catastrophic; all three other combinations converge.
3. **Immediate fix: revert to 29.** `n_losvd_bins` reverted to 29 and the
   notebook's `nsigma=1.0` override removed, to get back to a known-good
   state. Added a proactive `RuntimeWarning` in
   `fitting._discrepancy_principle_search` that fires whenever the
   bracket-expansion search needs >=3 expansions (i.e. the final xlam is
   >=1000x the caller's original `xlam_hi`) -- this would have caught the
   regression above before it shipped. Added
   `tests/test_regression_convergence.py` (three fast guardrail scenarios:
   basic mock, realistic E-MILES mock, real MUSE data) asserting
   `success=True`, a sane xlam ceiling, and a wall-clock budget. Re-ran the
   sigma=30-350 sweep at nbins=29 on current code and confirmed **bit-
   identical** V/sigma/chi2_red vs. the pre-regression baseline
   (`results_nbins_sweep.json`) -- nothing else that changed this session
   affected numerics at nbins=29.
4. **Proper recalibration: 29 -> 89 again, this time with the auto-xlam
   grid fixed.** Reverting to 29 brought back the original sigma-
   overestimate/peak-too-tall bias (directly reproduced on notebook 01's
   own seed=42 mock: sigma recovered 134.9 vs truth 140, LOSVD peak 4.2%
   too tall at nbins=29; nbins=89 on the identical mock recovers
   sigma=142.8, peak ratio 0.98). Root cause of why 89 bins alone
   previously needed dangerous bracket expansion: `xlam_auto_grid`'s
   default topped out at 1e5, but 89 bins needs proportionally more
   regularization for the same effective smoothness (confirmed: the final
   selected xlam is independent of the initial bracket -- bisection finds
   the same root either way -- but a too-narrow grid forces multiple
   10x bracket expansions to get there, which is both slow and was the
   literal mechanism of the notebook-02 regression). Fix:
   `xlam_auto_grid`'s default upper bound raised from 1e5 to 1e7
   (`(100., 1000., 10000., 100000., 1_000_000., 10_000_000.)`), then
   `n_losvd_bins` raised back to 89.
5. **A second, milder issue found during this recalibration:**
   `examples/notebooks/02_realistic_mock_fit.ipynb`'s actual construction
   (a 2-SSP E-MILES mixture fit against a 20-template subgrid, harder/more
   degenerate than the single-real-template scenario in
   `test_regression_convergence.py`) selects `xlam=1e7` -- exactly the new
   grid's upper edge, via the "smallest unimodal candidate meeting the
   discrepancy target" fallback, since every evaluated candidate below
   1e7 was multi-peaked. At that extreme regularization strength,
   L-BFGS-B's default `map_ftol=1e-10` stopping check is exposed to
   floating-point noise in the objective: the *same* config produced
   `result.success=True` in a bare script (and even when that same script
   was run under `jupyter nbconvert`), but `result.success=False`
   ("TOTAL NO. OF ITERATIONS REACHED LIMIT") when run as part of the full
   notebook 02 -- with nearly identical recovered V/sigma either way
   (78.4-78.9 vs 141.2-141.8, truth 80/140). Raising `map_maxiter`
   10000 -> 20000 alone did *not* fix it (the notebook still hit the
   limit). Loosening `map_ftol` 1e-10 -> 1e-8 (matching notebook 01's
   already-looser setting) did fix it, confirmed stable across 2
   independent nbconvert re-runs. The recovered parameters were never
   actually wrong in the `success=False` runs -- only the convergence
   *flag* was a false negative -- but a user relying on `result.success`
   as a go/no-go signal would have been misled, so this was worth fixing
   rather than just documenting away.

## Current state (as of this writeup)

- `FitConfig.n_losvd_bins` default: **89**.
- `FitConfig.xlam_auto_grid` default: **(100, 1e3, 1e4, 1e5, 1e6, 1e7)**.
- `tests/test_regression_convergence.py` runs at these defaults (no
  per-test override) and passes, including the real-MUSE-data case.
- Re-validated on `examples/notebooks/01_basic_mock_fit.ipynb`,
  `02_realistic_mock_fit.ipynb` (now with `map_ftol=1e-8`,
  `map_maxiter=20000`), and `03_real_data_muse.ipynb` -- the latter now
  reproduces almost exactly the numbers it gave before this whole
  regression cycle started (V=+25.8, sigma=38.2, xlam~18434 vs. the
  earlier ~17783), strong evidence that 89 bins + the wider grid is what
  actually produced the originally-reported "nearly perfect" results.
- `dev_notes/stress_test/sweep_nbins89_recalibrated.py` re-runs the
  sigma=30-350 grid at nbins=89 with the new grid, for comparison against
  the historical nbins=89 rows in `results_nbins_sweep.json` (generated
  with the old, narrower grid) -- expected to match closely, since the
  bisection root doesn't depend on the initial bracket width.

## Lesson

Any future change to `n_losvd_bins` (or anything that changes the LOSVD
parameterization's dimensionality) must be re-validated jointly with
`xlam_auto_grid`/`xlam_discrepancy_nsigma`, not in isolation -- and, per
item 5 above, it's also worth sanity-checking `map_ftol`/`map_maxiter`
headroom at whatever extreme xlam the new bin count naturally selects,
since a stricter default tolerance can produce flaky false-negative
`success` flags at very large regularization strengths even when the
recovered parameters themselves are fine.
