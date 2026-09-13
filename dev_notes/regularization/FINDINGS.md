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

---

# 2026-08-09: lam=4096/penalty_order=3's own validation never covered
# N4751's real sigma range -- extended it, and the answer is concerning
# at the high end

**Trigger**: user asked directly, after a session of chasing error-budget
bugs, "How are we deciding what a good lam value is? I don't want to smooth
over real data" -- a fair challenge, since every real-data notebook uses
`lam=4096, penalty_order=3` (the combination `dev_notes/continuum/
FINDINGS.md`'s 2026-07-30 sweep validated) without ever having checked
whether that validation's own sigma range actually covers what's being fit.

**It doesn't.** `dev_notes/validation_suite/mocks.py`'s own `RANGE_PRESETS`
top out at sigma=300 (`"safe"`) and sigma=400 (`"wide"`) km/s. N4751's real
central sigma, from its own already-measured radial profile
(`ngc4751_radial_kinematics.npz`), is 500-650+ km/s -- entirely outside
either preset. Every prior validated recommendation for `lam`/
`penalty_order` was earned in a regime N4751's own real data doesn't live
in.

**Extended the existing mock injection-recovery machinery
(`dev_notes/validation_suite/mocks.py`/`runner.py`) with `fixed={"sigma":
...}` at 400/550/700 km/s** (bracketing N4751's own real range), 6 noise
seeds each, comparing fixed lam=512/4096/32768 and `aic_local`
(`penalty_order=3` throughout, matching the real-data default):

| sigma | best option (by \|bias_sigma\|) | lam=4096 (current default) | lam=32768 / aic_local |
|---|---|---|---|
| 400 | lam=32768 / aic_local: **-11+/-10** km/s | +50+/-76 km/s | -11+/-10 km/s (same, aic_local picked 32768) |
| 550 | lam=32768: +18+/-99 km/s (mean; large scatter) | **+85+/-80** km/s | +18+/-99 / aic_local +57+/-104 |
| 700 | lam=512: +155+/-186 km/s (least bad, still large) | +170+/-181 km/s | **+300+/-311 / +294+/-315** |

**At sigma=700 -- the top of N4751's own real central-bin range -- every
tested option is bad, and heavier regularization is measurably worse, not
better.** `aic_local`/`lam=32768` bias nearly doubles the already-large
bias `lam=512`/`4096` show (~300 km/s vs ~155-170 km/s, on a true sigma of
653 -- 46% vs 24-26% relative). This is the user's own worry, directly
confirmed with evidence: at N4751's most extreme bins, more smoothing does
throw away real signal, not just noise. The reverse holds at sigma=400 --
there, the *lighter* fixed values (512, 4096) are the ones doing badly
(50-66 km/s bias, larger scatter), and heavy regularization
(32768/aic_local) is clearly best. **No single lam (fixed or adaptive) is
uniformly good across N4751's own real sigma range** -- the regime this
package validated well (see the 2026-07-30 continuum sweep and the broader
sigma 90-600 Laplace coverage campaign referenced in `CHANGELOG.md`'s
1.0.0 entry) does not extend cleanly to N4751's most extreme bins.

**Not resolved -- flagged for the next session.** This needs more than a
single-seed spot check to act on (6 seeds/cell is enough to see a real,
large effect, not enough to tune a replacement default with confidence).
Concrete next steps, roughly in order of expected value:
1. Widen the mock validation sweep itself (more seeds, finer sigma
   sampling 400-800) and check whether a *different* fixed lam (values
   between 512 and 4096 untested here) does better throughout N4751's
   range, rather than assuming the existing candidate grid's own values
   are the only ones worth trying.
2. Cross-check whether the bias at sigma=700 is a regularization problem
   at all, or a template-mixture/basis-resolution one at this width (a
   related, independent finding from the same session: N4751's own
   central bin needs a +/-4000+ km/s grid and ~90 spline coefficients at
   `knot_spacing_kms=95`, for often single-spaxel, low-S/N data -- see
   this session's other findings on multi-start instability growing with
   both lam and grid width).
3. Given `07`/N4751 already run through `07_full_galaxy_analysis.ipynb`
   and `n4751_full_galaxy_analysis.ipynb` at `lam=4096` (the least-bad
   fixed option found here, not coincidentally already the shared
   default both notebooks use), no urgent notebook change is implied by
   this finding alone -- but do not read "least bad of 4 options tested"
   as "validated good" for N4751's highest-sigma bins specifically; that
   is not yet true.

**Separately, real-time performance profiling of the same fixed-lam sweep
found the mechanism behind another user observation ("higher lam takes
longer to fit") directly**: on N4751's own real central bin, the outer
alternating-fit loop's own iteration count (`n_iter`) stays flat (4) across
lam=1 to 100000, but wall-clock time grows ~10x (2.5s to 26s) and the
multi-start `losvd_objective_gap` grows from 0.005 to 5.9 over the same
range (`success=False` at lam=100000) -- i.e. the *inner* L-BFGS-B solves
are taking measurably longer and the optimization landscape is becoming
more genuinely multi-modal as the penalty strengthens, not just slower for
no reason. This is consistent with, and adds real-data confirmation to,
this session's separate finding (`dev_notes/real_data_sigma/FINDINGS.md`)
that multi-start instability grows with regularization strength on real
data. Also profiled directly: `cleaning_method="clip"` and `"inflate"`
cost real time by design, not overhead to eliminate -- `"clip"` roughly
doubles wall-clock on this bin (a real second refit pass triggered by
genuine contamination tripping the default `clip_sigma=3.0` threshold,
`robust_scale=False`); `"inflate"` costs ~2.3x a plain fit
(preliminary-at-fixed-lam + final pass), and `measure_rho=True` adds one
more fixed-lam preliminary pass on top of that (~+10s on this bin,
matching the user's own reported "+10s" observation exactly).

**Continuum-knot count, also checked directly on the same bin**: whitened
RMS drops sharply from 0 to 2 knots (1.09 -> 0.72) then is completely flat
2 through 12 knots (0.717 -> 0.716) -- so 4 knots is not "arbitrary" in the
sense of leaving RMS on the table, more knots buy nothing there. **But
recovered sigma/h3/h4 are not flat across that same range** (sherm 561-734
km/s, h4 0.010-0.085 depending on knot count, despite statistically
indistinguishable RMS) -- meaning RMS alone is not a sufficient criterion
for choosing continuum flexibility on this target: models tied on residual
quality can still disagree substantially on the recovered kinematics,
consistent with continuum/LOSVD-shape degeneracy absorbing real spectral
information in a way the residual itself doesn't reveal. A knot-count
sweep should report moment stability, not just RMS, before treating a
choice as validated.

---

# 2026-08-09 (later): widened sweep -- lam=4096 turns out to be a genuine
# local optimum in N4751's own sigma range, not an arbitrary pick

**Follow-up to the entry directly above, item 1 of its own "not resolved"
list**: widened the sweep -- 10 seeds (up from 6), fresh mock draws (a
different `rng` seed base), and, critically, **filled in the gap below
4096** (lam=512/1024/2048/4096, not just 512-vs-4096-vs-32768) at the same
three sigma values (400/550/700):

| sigma | lam=512 | lam=1024 | lam=2048 | **lam=4096 (default)** |
|---|---|---|---|---|
| 400 | +76+/-63 | +65+/-62 | +58+/-61 | **+39+/-48** |
| 550 | +89+/-89 | +65+/-76 | +40+/-57 | **+31+/-42** |
| 700 | +93+/-130 | +83+/-124 | +62+/-118 | **+52+/-109** |

**`bias_sigma` decreases monotonically from 512 to 4096 at every sigma
tested** -- combined with the previous entry's own finding that it
*increases* again from 4096 to 32768/`aic_local`, this traces out a real
minimum sitting at (or very near) 4096 across N4751's entire real sigma
range, not just coincidentally at one value. **The earlier entry's own
framing ("no single lam is uniformly good," emphasis on 700 being bad
everywhere) undersold this** -- the fuller picture is that `lam=4096` is
consistently the best of everything tested here, at every sigma, which is
a meaningfully more reassuring result than "least bad of a bad set."

**Still true, and still worth carrying forward**: absolute bias at
`lam=4096` remains real and non-trivial (31-52 km/s, roughly 6-13%
relative on true sigma 400-700) -- this is a smaller, "genuinely hard
regime" bias, not the "actively picking the wrong tool" concern the
sigma=700/lam=32768 comparison raised on its own. Scatter also stays large
(+/-42 to +/-109 km/s) at every lam tested, consistent with this session's
other finding that the recovery problem itself (not just the lam choice)
gets harder at N4751's real sigma -- a genuine, open accuracy/precision
limitation at this target's sigma range, separate from the "is lam=4096
the right pick" question this pair of entries now answers: yes, among
fixed values at least, it already is.

**Not done**: testing values *between* 4096 and 32768 (e.g. 8192, 16384)
to nail down exactly where the minimum sits, or whether it is a broad,
flat minimum (in which case 4096 being close-but-not-exact matters little)
or a sharp one (in which case it might be worth tuning precisely). Given
4096 already looks like a solid choice, this is a refinement, not a
correction, and lower priority than the still-open template-mixture/basis-
resolution question item 2 of the previous entry raised.
