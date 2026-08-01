# AIC-local low-sigma oversmoothing: mechanism confirmed, partial fix validated

## The bug

`lam_selection="aic_local"` selects `lam=32768` (the order-3 grid's max) at
sigma 25-55 km/s under the validated NSIG=7 velocity-grid convention, with
real sigma bias up to +178% observed in real-galaxy fitting (notebook 08,
2026-07-31). The knot-spacing `UserWarning` (`config.py`'s
`__post_init__`) fires constantly in this regime — confirmed via direct
testing this is **not spurious noise**, it flags a real, live oversmoothing
problem.

## Root cause (confirmed via code trace + exact-arithmetic test)

`config.py`'s `resolved_n_interior_knots()` floors the knot count at
`min_interior_knots=16` whenever `round((vmax-vmin)/knot_spacing_kms)` is
smaller — always true for sigma 25-55 under NSIG=7 (grid width 350-770 km/s,
auto count 4-8). The resulting actual knot spacing `dv` (20-45 km/s) is much
finer than the `knot_spacing_kms=95` reference the fixed `lam_grid`
`(1, 8, 64, 512, 4096, 32768)` was calibrated against.
`losvd.difference_matrix` scales the penalty operator `D` by
`(knot_spacing_kms/dv)` — **squared** once it appears in `D.T @ D` — so
every candidate in the grid is uniformly inflated in effective strength by
the same large factor before AIC ever compares them, and the criterion
(genuinely flat/information-limited at these sigmas — `m_eff` pins near its
floor at every candidate) ends up selecting the grid's own ceiling, which is
now a much stronger effective penalty than `32768` was ever calibrated to
mean. `config.py`'s own effective-lam warning threshold is linear
(`lam*(knot_spacing_kms/dv)`) and *understates* the true (quadratic)
inflation by another factor of `(knot_spacing_kms/dv)`.

## The fix: `FitConfig.self_normalize_lam_grid` (opt-in, not default)

Rescale the candidate `lam_grid` by `(dv/knot_spacing_kms)^2` before
`select_lambda_aic_local` compares candidates — exactly cancels
`difference_matrix`'s own scaling, so the *effective* penalty strength
explored stays centered on the calibrated range regardless of knot
spacing. Implemented in `aic_regularization.select_lambda_aic_local`
(new `self_normalize_lam_grid` parameter) and threaded through
`FitConfig`/`fit_losvd`. Verified exact by construction
(`tests/test_aic_regularization.py`,
`test_self_normalize_lam_grid_makes_effective_penalty_invariant_to_knot_spacing`):
`(lam_nominal * scale) * D_narrow.T @ D_narrow == lam_nominal * D_ref.T @ D_ref`
to `rtol=1e-10`.

**Why opt-in, not the new default**: turning this on changes the *numeric
value* `result['lam_used']` reports for narrow-grid fits (by design — that's
the point). `bias_correction.PerLamCalibration`'s tables were built by
sweeping sigma under the OLD (unscaled) convention, so their canonical `lam`
values assume today's `lam_used` numbers. Enabling this fix without
rebuilding those tables would silently feed the wrong canonical-`lam`
bracket to bias correction for exactly the narrow-grid (low-sigma) fits that
matter most — a second bug traded for fixing the first. Adopting this as
default requires rebuilding the calibration tables first (a separate,
multi-hour undertaking, see `dev_notes/calibration/`); flagged for the user
to decide on, not silently done.

## Empirical validation (`verify04_self_normalize_lam_grid_ab.py`)

Paired A/B, spline-truth mocks (`wiggle_scale=0.15`), sigma in
{25, 35, 45, 55}, S/N=60, 6 seeds each, identical mock/template/grid between
arms — only `self_normalize_lam_grid` differs:

| sigma | median bias, off | median bias, on | median &#124;bias&#124;, off | median &#124;bias&#124;, on |
|------:|------:|------:|------:|------:|
| 25 | +23.2% | +2.6% | 27.9% | **10.0%** |
| 35 | +4.4%  | -3.4% | 6.7%  | 5.4%  |
| 45 | -2.1%  | -1.7% | 3.5%  | 3.4%  |
| 55 | -0.3%  | -2.2% | 1.7%  | 2.7%  |

**Real, but partial and sigma-dependent.** At sigma=25 (the worst-affected
case, dv≈20.6 vs dv_ref=95, i.e. the most extreme mismatch), the fix cuts
median absolute bias roughly in half (28%→10%) — a genuine, large
improvement. The benefit shrinks rapidly with increasing sigma and is noise-
dominated (within seed-to-seed scatter, n=6) by sigma≈45-55, consistent with
`dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`'s prior finding (2026-07-26,
testing a different but related fix — varying `difference_matrix`'s own
scaling exponent directly) that this regime is genuinely information-limited
(`m_eff` pins near its floor at every candidate lam) and no single rescaling
cures it outright: in both the "on" and "off" arms, AIC always selects the
grid's own ceiling candidate at every sigma tested here — the fix changes
what absolute `lam` that ceiling corresponds to (correctly), it does not
make the selection itself less degenerate.

**Not yet done, left for a future session**: rebuilding the bias-correction
calibration tables under `self_normalize_lam_grid=True` so it can become the
default without desyncing `PerLamCalibration`; a proper validity-gate-style
fix (matching `max_edge_amplitude_frac`'s precedent) for the residual
information-limited degeneracy at sigma<~45, since rescaling alone doesn't
resolve that part.

See `dev_notes/regularization/aic_local_firming_up_results.txt` for the
original AIC-local-vs-fixed comparison this builds on, and
`dev_notes/REGULARIZATION_INVESTIGATION_LOG.md` for the 2026-07-26 prior
attempt at a related fix.
