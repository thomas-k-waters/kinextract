# Real-data sigma inflation: root cause found (2026-07-27)

**Symptom** (user-reported): on real NGC 5102 MUSE spectra, kinextract_lite
recovers sigma = 95.2 km/s (nuclear) and 105.6 km/s (mid-radius) where the
legacy Fortran pipeline gets ~54.3 and ~40-45. A ~2x failure.

## What it is NOT (each ruled out by direct measurement)

| hypothesis | test | result |
|---|---|---|
| the new D3 penalty | fit same data at penalty_order 2 vs 3 | 79.7 vs 81.4 -- **no difference**, predates Stage 22 |
| continuum mismatch | continuum-cofit, 0/6/12 knots | chi2_red 16 -> 5.6 but sigma **unchanged** (81->75 at lam=1; 50->49.6 at lam=512) |
| sky-line residuals | auto 3-sigma positive-outlier mask | 0 extra pixels flagged in the nuclear bin; sigma unchanged |
| template continuum normalization | normalize data AND templates | sigma 81.4 -> 81.1, **no effect** |
| the non-parametric machinery per se | 5-param parametric GH fit, same data/templates/design matrix | **sigma = 42.3** at comparable chi2 -- so the data does NOT want 81 |

## What it IS

Two stacked causes.

**1. The D3 penalty's null space contains divergent solutions.** Verified
directly: `||D3 c||^2` is *identical* (49.92) for a true Gaussian
`exp(-v^2/2s^2)` and for its mirror image `exp(+v^2/2s^2)`, which blows up
at the grid edges. The penalty cannot tell a compact LOSVD from a
divergent one. (Order 2 has the same disease one order down: its null
space is `exp(linear)`, also divergent.)

**2. `aic_local` selects the least-regularized lam on real data**, because
AIC's chi2-vs-m_eff trade assumes residuals are noise. Real residuals here
are **systematics-dominated** (chi2_red = 11-16, not ~1), so extra
flexibility keeps buying large raw chi2 gains that the `2*m_eff` penalty
cannot offset. Measured on the nuclear bin: lam=1 buys 1193 units of raw
chi2 over lam=512 while costing only ~5 effective dof.

The result: at the selected lam=1, the fit grows a **divergent edge
pedestal** -- 16% of peak amplitude at +-300 km/s (nuclear), 63% at +400
(mid-radius), 6.6% of total LOSVD mass beyond the taper onset -- which the
moment estimator integrates into sigma. The *core* is fine; the pedestal
is the entire error.

**lam is the controlling variable, and the answer is stable once it is
large enough:**

| bin | lam=1 (what aic_local picks) | lam=8 | lam=512 | lam=32768 | legacy Fortran |
|---|---|---|---|---|---|
| nuclear (high S/N) | 81.4 | 84.4 | **50.0** | 49.7 | ~54.3 |
| mid-radius (med S/N) | 97.5 | 39.3 | **39.3** | 40.8 | ~40-45 |

At sensible lam, kinextract_lite reproduces the legacy answer to within a
few km/s on both bins. The pathology is entirely a selection failure.

## Candidate fixes tested and REJECTED

- **Wing-mass-fraction penalty** `sum(taper*b)/sum(b)` (scale-invariant by
  construction, unlike the existing raw-amplitude `wing_lam`): fixes real
  data (nuclear 68 -> 49 at 1e4) but at that same strength **destroys the
  mock** (sigma 50 -> 31, shape residual 8.9% -> 59%). Same
  non-portability that killed `wing_lam`. Not a safe default.
- **One-sided convexity penalty** `relu(D2 c)^2`: dominated by
  clamped-spline boundary artifacts -- a true Gaussian scores *worse*
  (57.7) than the pathological inverted shape (44.8). Unusable as written.
- **Bounding the global quadratic null-space mode `a2 <= 0`** (exact
  blossom decomposition, no new tuning parameter): correct in principle
  and does help at large lam (nuclear 68.2 -> 49.9), but **no effect at
  the lam aic_local actually picks** (81.1 vs 80.5), because at lam=1 the
  pedestal is built from higher-order spline structure, not from the
  global quadratic mode. Necessary but far from sufficient.
- **QAIC / overdispersion correction** (divide chi2 by chi2_red before the
  m_eff penalty -- Burnham & Anderson's own prescription, already cited in
  this package for the parsimony tiebreak): arithmetic check shows it is
  **insufficient for the nuclear bin** -- 1193 raw chi2 units / c_hat=16 =
  75, still far larger than any 2*delta-m_eff. Would fix the mid-radius
  bin but not the worst one.

## Where this leaves the fix

No information criterion of the AIC family can resist this, because the
pedestal genuinely *does* improve the fit to data whose model is wrong.
The fix has to remove the LOSVD's ability to absorb model mismatch, not
re-referee the trade. Untested options, in the order I'd try them:

1. **Iterate the velocity grid to convergence.** The pedestal lives at
   |v| > 200 km/s on a +-400 grid whose true sigma is ~50. The notebook's
   two-stage V+-7sigma refinement is seeded by a *contaminated* stage-1
   sigma (81), so stage 2's grid stays far too wide and preserves the
   unconstrained range the pedestal lives in. Iterating to a fixed point
   should shrink it away, costs nothing, and is already half-implemented.
2. **A lam floor derived from the data's own overdispersion** -- e.g.
   require the selected lam to be the largest whose chi2 is within
   `c_hat * sqrt(2N)` of the minimum, rather than argmin of chi2 + 2 m_eff.
3. **Reconsider the moment estimator**: `compute_direct_moments`'s taper
   is amplitude-based and does not exclude a 2%-of-peak pedestal spanning
   half the grid. A core-restricted second moment would be far more robust
   to exactly this failure, independent of whether the fit is fixed.

---

## Update: structural fix identified (2026-07-27, later)

Confirmed first that **no lam-selection rule can work**, by arithmetic on
the measured chi2/m_eff table: QAIC (chi2/c_hat + 2 m_eff) and the
discrepancy principle (largest lam within c_hat*sqrt(2N) of chi2_min) both
still land on lam=8 -> sigma=84 for the nuclear bin. The pedestal buys
~1200 units of chi2 by absorbing template mismatch; no dof penalty of this
family offsets that. So the constraint has to be structural.

### Tested: unimodality of b(v)

A single-component LOSVD is unimodal, and B-splines are
**variation-diminishing** -- a unimodal coefficient sequence gives a
unimodal curve -- so unimodality can be imposed directly on the spline
coefficients. Implemented as a scale-invariant soft penalty added to the
existing objective (`diag07_soft_unimodal.py`):

    kappa * sum_j relu(-s_j * (c_{j+1} - c_j))^2,  s_j = +1 before the
                                                   peak, -1 after

Zero for any unimodal b; quadratic in the violation otherwise. Invariant
under `b -> c*b` (it sees only *differences* of log b), so unlike
`wing_lam` it needs no per-target amplitude tuning.

**Harness validated against the package's own `fit_losvd_basis` at
kappa=0 before believing any constrained number** -- agrees to 0.00 km/s
(45.62/45.62 and 45.36/45.36). This check is the reason the earlier
cumulative-sum + box-bounds attempt (`diag06_unimodal.py`) is discarded:
it FAILED the same control (41.1 vs the package's 49.0 with the
constraint off), because the lower-triangular reparameterization wrecks
the conditioning. None of diag06's numbers should be used.

Results, all at the lam `aic_local` actually selects:

| case | kappa=0 | kappa=1e4 | kappa=1e6 | target |
|---|---|---|---|---|
| real nuclear (lam=1) | 82.3 | 69.0 | 69.2 | ~54.3 |
| real mid-radius (lam=1) | 97.0 | 68.3 | 50.2 | ~40-45 |
| mock sigma=50 (lam=32768), 5 seeds | 49.86 | 50.57 | 50.50 | 49.39 |
| mock max shape residual | 5.18% | 6.39% | 5.89% | -- |

**Harmless on mocks** (sigma within 1.2 km/s of truth at every kappa,
shape residual essentially unchanged) and **cuts the real-data error
roughly in half** at the pathological lam. This is the first candidate
that does not trade one regime against the other.

Not sufficient alone: it caps the damage rather than removing it, because
lam=1 still leaves the core under-regularized. The combination that
reproduces legacy on both bins is unimodality **plus** a lam that is not
pinned to the bottom of the grid (at lam=512 the unconstrained fit is
already 50.5 / 39.5). Pairing the two, and choosing the lam rule, is the
remaining work.

### Known limitation to document if adopted

A unimodality constraint forbids genuinely bimodal LOSVDs (counter-
rotating cores, superposed components). It should be a default that can
be switched off, not an unconditional assumption.

---

# 2026-07-28: the fix, and two negative results on the way to it

## Negative result 1: linear-space D2 (OG's penalty) is NOT the cure

Hypothesis (mine): OG kinextract penalizes `b` directly while we penalize
`log b`; our D3 null space contains the divergent `exp(+a v^2)` while OG's
D2 null space is a bounded linear ramp; therefore linear space should kill
the pedestal. Implemented faithfully in `diag08_linear_space_d2.py`,
including OG's implicit `b=0` edge rows and the quartic wing taper, solved
exactly as an augmented NNLS (no optimizer, no local minima).

**Result: sigma = 88-97 km/s on BOTH bins at EVERY lam from 1 to 3.3e4**,
never approaching legacy 54.3 / 40-45. Also completely insensitive to the
taper seed -- starts of 30/40/50/60/80/120 km/s all converge to the same
value, so the suspected self-fulfilling `sigl0` fixed point does not exist.

The hypothesis was backwards. Heavy D2 in **linear** space drives `b` toward
its null space, a linear ramp, which is *broad*. Heavy smoothing in **log**
space drives `log b` toward a quadratic, i.e. a Gaussian, which is *compact*.
That is why our log-space fitter reaches 49.1 / 40.3 at large lam and the
linear-space one cannot get there at all. **Our parameterization is better
in the high-lam limit, not worse.**

## Negative result 2: OG's feasibility rule does not transfer either

OG selects the smallest lam that is unimodal (`xlam_max_peaks=1`) and
non-jagged (`xlam_smooth_threshold=0.25`). Applied to our log-space D3
fitter, it selects **lam=1 on both bins -- identical to what AIC picks.**
The reason is simple and worth remembering: the pedestal solutions *are*
unimodal (peaks=1) and *are* smooth (roughness 0.09-0.10, well under 0.25).
A smooth divergent pedestal passes both of OG's tests. Neither criterion
looks at where the LOSVD's mass actually is.

## The fix: an edge-amplitude feasibility constraint

The statistic that separates good from bad lam with no overlap at all is the
LOSVD's amplitude at the velocity-grid edge, as a fraction of its peak:

| lam | 1 | 8 | 64 | 512 | 4096 | 32768 |
|---|---|---|---|---|---|---|
| high bin edge | 21.9% | 29.4% | 26.0% | **0.0%** | **0.0%** | **0.0%** |
| med bin edge | 42.4% | 61.3% | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

Rule: **require the LOSVD to have decayed below 1% of its peak at the grid
edge.** This is not a tuning knob in any real sense -- it is the statement
that the velocity grid must contain the LOSVD, which is a precondition for
the moment integrals to mean anything. A fit that violates it is not
"differently regularized", it is being integrated over a range where the
model was never constrained.

Results:

| | AIC picks | sigma | edge rule picks | sigma | legacy |
|---|---|---|---|---|---|
| high bin | lam=1 | 84.5 | **lam=512** | **49.1** | 54.3 |
| med bin | lam=1 | 100.7 | **lam=64** | **40.3** | 40-45 |

**Portability -- the property every earlier candidate lacked.** On the NB02
mock the rule is *inert*: the +-660 km/s grid is already wide relative to
sigma=90, every lam is feasible, nothing is excluded, and the mock result is
unchanged (sigma 95-98 vs truth 98.4). The wing-mass penalty and `wing_lam`
both died because the strength that fixed real data destroyed the mock. This
one cannot, because on data with no pedestal it does nothing at all.

At S/N=200 the edge rule also recovers h4 = +0.017/+0.023 against AIC's
+0.003/+0.004 (truth +0.050) -- better, though both remain well inside the
CRLB (see dev_notes/ghcore/FINDINGS.md).

## Recommended implementation

Use the edge constraint as a **filter on the candidate grid, then let AIC
choose among the survivors** -- not "smallest feasible lam", which is what
the scan above did. On the mock AIC's picks (lam=32768 at S/N=60, lam=64 at
S/N=200) are already feasible, so AIC-among-feasible leaves mocks untouched;
on real data lam=1 is excluded and AIC falls to the feasible set that
reproduces legacy. This keeps AIC's good behavior where it works and removes
only the failure mode it provably cannot see.

## Harness bug found (logged per standing rule)

The first version of the lam scan produced *identical* results across five
orders of magnitude in lam. Cause: `cfg.lam_selection` defaults to
`'aic_local'`, which re-runs selection and overwrites `cfg.lam` before the
fit -- so setting `cfg.lam` alone is silently ignored. `lam_selection="fixed"`
is required. Documented behavior, my error, but an easy trap: anyone sweeping
`lam` without setting `lam_selection="fixed"` gets six copies of the same fit.

---

# 2026-07-31: what `robust_scale=True` cleaning actually clips on the real galaxy -- NOT dominantly CaII

Notebook 08's full-galaxy run (219 bins, `clip_sigma=3.0`, `max_iter=3`,
`robust_scale=True`) finished with aggregate clip fraction 8.9%
(61902 pixels total, 5492 clipped) -- much better than the pre-fix runaway
(>90% on some bins), but per-bin fraction is uneven: median 5.4%, mean 9.2%,
84/219 bins over the code's own 10% sanity threshold, max 32.8% on one bin.
This was the open question this session's plan flagged: is the residual
clipping eating into CaII (the user's suspected failure mode), or something
else entirely?

**Checked directly** (`dev_notes/real_data_sigma/analyze_clipped_wavelengths.py`,
run against the finished cache): pooled every clipped pixel's rest-frame
wavelength across all 219 bins (`wave_fit` is already rest-frame per bin --
each bin's own systemic+rotation velocity already divided out -- so a real
CaII effect should cluster tightly across bins despite each bin's different
velocity, while fixed-observed-frame sky-residual contamination should smear
out across different rest-frame positions bin to bin).

**Result: CaII is NOT the dominant driver.** Only 15.3% of clipped pixels
fall within 8 A of a CaII triplet center (`cleaning.CAII_TRIPLET_HALF_WIDTH_A`)
-- barely above the 12.0% expected from pure chance (a uniform-random
selection over the 8400-8800 A fit window would land within that same
distance of a center 12.0% of the time, since the three 16 A-wide windows
cover 12% of the 400 A range). A ~3 percentage-point excess is a real but
small effect, not the story.

**The dominant clustering is at the EDGES of the fit window instead**: the
two biggest histogram bins are 8400-8410 A (the blue edge) and
8740-8780 A (the red edge, 83-113 A from the nearest CaII center, i.e.
nowhere near it). This points at edge-of-fit-window model-data mismatch
(LSF/continuum shape breaking down near the window boundaries) as the
larger contributor to residual clip fraction -- plausibly the same root
cause behind the still-unexplained real-data chi2_red inflation (this
memory's own earlier finding, and the AIC-oversmoothing bug in
`dev_notes/regularization/FINDINGS.md`), not a separate problem.

**Practical upshot**: an excluded-wavelength-window option on
`fit_losvd_clean` (protecting known lines from ever being clipped,
regardless of `clip_sigma`) is still implemented as a direct, low-risk
safety net against the smaller CaII-specific risk the user flagged, but it
is not expected to meaningfully reduce the overall clip fraction -- the
bigger lead is the fit-window-edge mismatch, which is a job for the
continuum/chi2_red investigation, not for tuning the cleaning step further.

---

# 2026-08-08: the pedestal bug came back -- self-inflicted, via a fixed
# lam that bypassed this file's own already-validated fix

**Trigger**: user asked to audit every uncommitted change since the last
git push (`4473bd0`, 2026-08-05) and systematically find what broke
`07_full_galaxy_analysis.ipynb`'s NGC 5102 output between then and now --
"it was working pretty well at that time."

**Method**: `git worktree add` at `4473bd0` to get the exact pushed code,
diffed every changed file, and directly compared the push-time notebook's
own fitting cell against the current one line-by-line. This surfaced the
actual mechanism, not just a symptom.

**What changed, and which of it mattered**: 24 tracked files + several new
ones differ from the push. Systematically swept all of them (worker/CPU
resource management, binning.py's astropy-read fix, the Laplace dispatch
fix from earlier tonight, legacy_io.py's new pallmc/transvd port, download
idempotency, `combine_bin_spectrum_robust`/Bacon-2017 variance correction
replacing the old `extract_aperture_spectrum` in both this notebook and
`04_template_calibration.ipynb`) -- all either purely additive/opt-in or
independently re-verified harmless (`342 passed, 1 skipped, 0 failed`,
full suite, run again as part of this same audit). **Exactly one change
was the culprit, and it was a notebook setting, not a package bug**: at
the push, this notebook's fitting cell built `FitConfig(vmin=..., vmax=...,
rho=MEASURED_RHO)` with no override at all -- bare defaults, i.e.
`lam_selection="aic_local"`, `penalty_order=3`. Earlier the same day
(2026-08-07), fixing a *different*, real bug (Phase 0's fixed
`lam=4096`/`penalty_order=3` crushing h3/h4 to ~1e-4 via the
`min_interior_knots` floor -- `dev_notes/regularization/FINDINGS.md`,
`dev_notes/uncertainty/FINDINGS.md`'s own 2026-08-07 entries) replaced
that bare config with `lam_selection="fixed", lam=2.0, penalty_order=2`.
That correctly fixed the h3/h4 collapse, but it silently reintroduced
*this file's own* already-solved pedestal bug from 11 days earlier: a
**fixed** lam has no candidate grid, so `max_edge_amplitude_frac`
(the "Recommended implementation" section above, "filter the candidate
grid, then let AIC choose among the survivors") has nothing to filter.
`lam=2.0` sits inside the exact range this file already measured as
pedestal-forming (lam=1 gave 84.5/100.7 km/s; only lam>=64 recovered
legacy). Confirmed directly, live, on real bins before applying any fix:
sherm running ~2x above the legacy-consistent range (population median
~78 km/s vs ~34-46 km/s before/after), reported uncertainty correspondingly
large (median `sherm_std` ~14-15 km/s, up to 68 km/s, >35% relative in the
worst bins) -- this is what the user saw as "horrible plots" with
implausible error bars, not a new phenomenon, the *same* mechanism this
file first diagnosed on 2026-07-27, wearing a different fixed-lam value.

**The fix**: removed the `lam_selection`/`lam`/`penalty_order` override
from `07_full_galaxy_analysis.ipynb` entirely, restoring bare `FitConfig`
defaults -- i.e. exactly the push-time recipe, not a new setting. Full
275-bin re-run (81.7 min, adaptive selection scans a candidate grid per
bin so this is much slower than a fixed-lam pass -- budget accordingly).
Direct before/after, against this file's own legacy reference values:

| region | fixed lam=2 (broken) | restored aic_local (fixed) | legacy Fortran |
|---|---|---|---|
| nuclear bins (r<0.3") | ~85-95 km/s (extrapolated from population) | **47-53.5 km/s** | 54.3 |
| mid-radius (r~5") | ~85-95 km/s | **48.9 km/s** | 40-45 |
| population median sherm | ~78 km/s | **45.7 km/s** | -- |

Nuclear and mid-radius bins now land within a few km/s of the same legacy
values this file's original 2026-07-27/28 investigation targeted --
direct confirmation this is the same bug, and the same fix.

**Not fully resolved -- h4 collapses toward ~1e-4 again under restored
adaptive selection, but this is NOT the same bug, and predates tonight's
regression.** With `penalty_order=3` restored, population h4 sits at
9.06e-5 to 1.16e-4 (essentially flat, near-zero) across all 275 bins --
visually similar to Phase 0's own collapse symptom, but a different,
already-documented mechanism: order-3's null space is "any Gaussian log-
density" (`dev_notes/skew/FINDINGS.md`, 2026-07-27), so it structurally
shrinks h4 toward zero at whatever lam AIC selects, independent of
`min_interior_knots`. This is a real, known, order-3-specific tradeoff
(order 2 preserves h4 far better but is the lam-unstable option that
motivated Phase 0's switch to order 3 in the first place) -- present in
some form even at the "good" push-time state (the equivalent pre-push
cache's own h4 median was 0.0048, already close to zero, just with more
noise-driven scatter around it from an older, coarser 219-bin scheme).
**Sigma is fixed and verified against an independent reference; h4/h3
fidelity under order 3 remains the open, pre-existing tradeoff this
file's own "Known limitation to document if adopted" section already
flagged in 2026-07-27** (a unimodality constraint, or reconsidering the
moment estimator, were the untested candidate fixes named there). Not
attempted tonight -- flagged as the next concrete step if h3/h4 structure
(not just sigma) is needed from this target's real data.

**Process note for next time**: this file already contained the exact
mechanism and the exact validated fix (`max_edge_amplitude_frac` +
adaptive selection, legacy reference values) *before* today's "fix"
overrode it. The 2026-08-07 fix that broke this was reasoned from
`min_interior_knots`/`knot_spacing_kms` mechanics in isolation, without
first checking whether this target already had a documented, validated
lam-selection story on file. Check `dev_notes/*/FINDINGS.md` for the
specific target before changing `lam_selection` on real data again.

---

# 2026-08-08 (later): adaptive selection itself was the wrong fix --
# un-superseding `dev_notes/uncertainty/FINDINGS.md`'s 2026-08-07 rho/
# error-budget entry, and a real fix for what it found

**Trigger**: user compared a fresh radial-profile screenshot against an
older, tighter-error-bar version and flagged the regression directly.
Restoring `lam_selection="aic_local"` (this file's own fix, above) had
already landed and *was* still active -- this is a second, independent
problem the aic_local restoration did not solve, discovered by actually
re-checking the error bars rather than assuming the earlier fix covered
them.

**Diagnosed via `losvd_objective_gap` (multi-start's own built-in
diagnostic, already computed, never previously surfaced) across the full
population**: 262/275 bins (95%) have `gap > 0.1` -- different multi-start
seeds converging to meaningfully different objectives -- median 1.81, max
40.9. Directly on bin 315 (n_spaxels=3366, one of the worst): every one of
the 6 `aic_local` grid candidates was excluded by
`min_resolvable_sigma_frac`'s sub-knot guard (chi2/m_eff identical --
33.92/3.00 -- at *every* lam, meaning the fit genuinely cannot
discriminate lam here), so selection fell back to the grid's own max
(`lam=32768`) for 272/275 bins -- not a real AIC decision, the documented
"least bad of both failure modes" fallback. That fallback fit is itself
unstable: a fresh multi-start refit at the same lam landed on sherm=90.8
km/s vs. the cached run's 148.8 km/s, and an ad hoc *expanded* multi-start
pool (12 starts including physically-motivated Gaussian starts at sigma
40-400) found yet a third answer (134.9, worse objective than the
3-start pool's own 90.8) -- more starts did not converge toward a stable
value.

**Bootstrap independently confirmed this is real instability, not a
Laplace-approximation artifact**: at the same fixed lam, bootstrap's own
`sherm_std` came out ~5x *larger* than Laplace's (156 vs 32) on bin 315 --
an independent method finding *more* disagreement, not less.

**Root cause, once traced further: an over-generous error budget, not a
regularization/multi-start problem per se.** `chi2/dof` on the final
whitened+inflated fit was 0.08-0.13 for the worst bins -- confirming and
extending `dev_notes/uncertainty/FINDINGS.md`'s 2026-08-07 entry (marked
"superseded" that same night, prematurely -- restoring `aic_local` fixed
a *different* symptom, the pedestal bias, and the wide bars it also
described were never re-checked afterward). Two compounding causes,
isolated separately:
1. **`fit_losvd_robust`'s `robustness="inflate"` step, applied
   unconditionally, stacked on top of an already-adequate error budget.**
   Direct check: raw (Bacon-corrected, `combine_bin_spectrum_robust`)
   errors alone, no inflation, already give chi2/dof 0.4-1.3 on the same
   bins -- inflation was adding unneeded variance on bins that never
   showed evidence of the contamination it exists to correct for.
2. **`MEASURED_RHO=0.88` (one global value, from one high-S/N
   representative aperture) does not transfer across this target's real
   per-bin S/N range.** Measured directly on the shipped high/med/low
   apertures: rho = 0.892 / 0.676 / 0.730. Per-bin, on the same 3 real
   bins: 0.75-0.80, not 0.88 -- and that 0.88 was itself measured against
   a *different* error basis (raw single-spaxel pipeline errors) than the
   per-bin production pipeline actually uses (Bacon-corrected, robustly
   combined).

**The fix, implemented in `robust_fit.py` and
`07_full_galaxy_analysis.ipynb`** (not a notebook-only patch -- both
new behaviors are opt-in `fit_losvd_robust` parameters, default-off,
existing callers/tests unaffected; 345 passed, 1 skipped, full suite):
- `measure_rho=True`: one extra fixed-lam preliminary fit at `rho=0`,
  measures *this bin's own* lag-1 ACF (`noise_model.
  estimate_correlation_length`) instead of trusting a global `cfg.rho`,
  and uses it for every whitening step in the call. Returns `rho_used`
  so downstream re-whitening (plots, Laplace uncertainty) can match
  exactly rather than silently reverting to the placeholder.
- `min_chi2_dof_to_inflate=1.5`: the preliminary pass's own whitened
  chi2/dof is checked before inflating; at or below the threshold,
  inflation is skipped entirely (plain final fit instead), returning
  `chi2_dof_prelim` for inspection. 271/275 bins skip inflation under
  this gate on the real NGC 5102 run.
- Also switched from adaptive `aic_local` to a fixed `lam=4096,
  penalty_order=3` (the same combination already validated in
  `dev_notes/continuum/FINDINGS.md`'s 2026-07-30 sweep and already used
  by `n4751_full_galaxy_analysis.ipynb`) -- adaptive selection's own
  multi-start instability (above) was a separate, real problem on this
  target's data, not fixed by the rho/inflation work alone.

**Result, full 275-bin re-run (`FORCE_RECOMPUTE=True`, 53 min): point
estimates improved substantially, error bars did not shrink much --
and that split turned out to be the honest answer, not a partial fix.**
Direct before/after on the 3 flagged bins: sherm 148.8/87.9/133.1 ->
76.5/47.2/57.3 km/s (roughly half, far more plausible / legacy-
consistent) but `sherm_std` barely moved (29.1/29.4/34.8 ->
28.5/25.0/30.3). Population-wide: median sherm 45.4 -> 37.5 km/s;
median `sherm_std` 15.2 -> 18.4 km/s (not smaller). **Once chi2/dof is
genuinely ~1 (verified, not just asserted), the remaining uncertainty
width is real, honest statistical uncertainty given this data's actual
information content -- not a further-fixable calibration artifact.**
This matches the multi-start finding above: the optimization landscape
itself is genuinely ambiguous for most of these bins, independent of
getting the noise model right. Visually, the fix is still a clear win --
the previous plot's worst offenders were specifically the *highest*-S/N,
densest bins showing implausibly large bars (worse than low-S/N bins,
backwards from expectation); the new radial profile shows tight, sensible
error bars in the dense central region and appropriately wider ones only
where S/N is genuinely low, matching the pre-regression reference plot's
qualitative shape.

**Known small-scope remaining issue**: 2/275 bins (175, 217; both
n_spaxels 41-69, `chi2_dof_prelim` 0.02-0.07 -- notably lower than the
rest of the population) return `NaN` uncertainty -- their recovered `b`
collapses to an extremely narrow (~190-orders-of-magnitude dynamic range)
near-spike profile even at the fixed `lam=4096`, unlike the rest of the
population. Not yet root-caused (plausibly `measure_rho`'s own
preliminary pass, which fits at a lighter lam than the final `4096`,
landing in a bad basin for these two specific bins) -- affects population
aggregates computed with plain `np.median`/`np.percentile` (propagates
NaN) until filtered out; does not affect the other 273 bins' own values.

---

# 2026-08-08 (later still): the min_chi2_dof_to_inflate gate above was
# itself wrong -- global average is the wrong signal; local-max fixed it,
# and a bootstrap cross-check found the real dominant uncertainty driver

**Trigger**: user compared the fixed-lam/measure_rho plot against the
pre-regression reference again and flagged two remaining things directly:
sigma still looks inflated at large `r` specifically, and "a lot of
clipped pixels... but it shouldn't be clipping with variance inflation,
no?" -- a sharp, correct observation that motivated re-examining the gate
just added.

**The `min_chi2_dof_to_inflate` gate (previous entry) was a real design
flaw, not a re-emergence of an old bug.** Direct check on bin 315: global
`chi2_dof_prelim=0.72` ("fine" by that gate, correctly skipped
inflation) -- but the *local* inflation profile that would have been
computed shows 12% of pixels with `inflation_factor > 1.2` (up to 3x). A
global-average gate cannot see a bin that is fine on average but has a
real, localized contaminated window -- exactly the situation for the
`r=26"` outer-ring bins, which also turn out to be the *highest*-n_spaxels
bins in this run (dartboard annulus area grows outward, so "large r" and
"high S/N" are the same bins here, not opposite ends of a range).
**Fixed**: `robustness="inflate"`'s gate parameter renamed
`min_inflation_to_apply`, now checked against
`compute_local_inflation`'s own `.max()` (1.0-floored, rises only where
real local excess exists), not a global chi2/dof average.
`chi2_dof_prelim` is still always computed and returned, as a diagnostic
only. Confirmed: 158/275 bins now receive real local inflation (vs. 4/275
under the old global gate); test suite green (34/34 `test_robust_fit.py`,
unchanged).

**Even with the local gate fixed, clip fraction on the worst bins only
dropped modestly (15.0/15.3% -> 12.7/13.1% on bins 315/306) -- traced
directly, and it isn't a bug.** Clipped-pixel wavelengths for bin 315
cluster tightly: 8401-8420, 8436-8455, 8478-8489, 8523-8533, 8573-8582,
8744-8780 A -- and sit at the 89th flux percentile (near continuum, not
absorption troughs). All three CaT lines (8498/8542/8662, all
protected by `CAII_PROTECTED_WINDOWS`) show *zero* clipping -- ruling out
an unprotected-line-core explanation directly. This pattern (narrow,
high-flux, tightly wavelength-clustered, common across bins) is
consistent with genuine MUSE red-end sky-subtraction residuals (this
range is airglow-dense), not a spectral feature or a masking gap.
**Mechanism**: at `n_spaxels > 1000`, statistical error shrinks toward
zero while a sky residual's absolute amplitude stays roughly fixed, so
the same artifact invisible at low S/N becomes a formal >5-sigma outlier
specifically at the highest-S/N bins -- independent of `inflation_cap`
(confirmed: raising `8.0 -> 15.0 -> 50.0` changed nothing, because
`inflation_factor` never reached the cap in the first place).

**`smooth_window_A` was the actual lever, and it has a hard floor.**
`compute_local_inflation` enforces a 3-pixel minimum window regardless of
`smooth_window_A`; at this target's ~1.25A/pix spacing that floor is
~3.75A, reached already at `smooth_window_A=3.0` (confirmed: 3.0 and 1.5
give byte-identical `inflation_factor`/clip results -- no more headroom
below the floor). Narrowing `8.0 -> 3.0` gave a real, modest improvement
(bin 315: clip_frac 12.7% -> 10.2%, `inflation_factor.max()` 3.05 -> 4.36)
with no observed downside -- adopted as the new default for this
notebook. Does not eliminate the clipping; some of it is from spikes
sharp enough that no reasonable local-scale smoothing protects them
without over-fitting individual noise pixels.

**The dominant finding, by far: a systematic, large Laplace-vs-bootstrap
disagreement at high S/N, confirmed on 7 bins now (not a single
outlier).** Bootstrap (`n_boot=30`, same post-clip/post-inflation whitened
inputs the point estimate itself used) vs. Laplace `sherm_std`:

| bin | n_spaxels | Laplace | bootstrap | ratio |
|---|---|---|---|---|
| 305 | 3366 | 31.8 | 141.9 | 4.5x |
| 286 | 1807 | 23.1 | 181.9 | 7.9x |
| 304 | 1480 | 28.4 | 104.1 | 3.7x |
| 302 | 1100 | 23.0 | 264.5 | 11.5x |
| 285 | 1802 | 27.7 | 183.1 | 6.6x |
| 296 | 1807 | 23.8 | 163.8 | 6.9x |

`h4_std` shows the same pattern far more starkly: Laplace 0.0002-0.0009
vs. bootstrap 0.19-0.35 -- 2-3 orders of magnitude, meaning bootstrap
replicates are landing on wildly different LOSVD shapes across noise
resamples. At low/moderate S/N (bins 0/114/182/250, n_spaxels 1-160) the
two methods agree reasonably (within ~40%). **This is consistent with,
and now strong independent confirmation of, the multi-start optimization
instability found in the previous entry** (95% of bins with
`losvd_objective_gap > 0.1`) -- Laplace's local-curvature approximation
is blind to a landscape with several comparably-good, meaningfully
different optima; bootstrap's resampling exposes exactly that. **The
practical implication is the opposite of "inflated": if anything, the
currently-plotted (Laplace) error bars at the highest-S/N bins
understate the true uncertainty**, possibly by close to an order of
magnitude. Not yet resolved -- user is deciding between investigating
further (wider bootstrap sample, cheaper approximation of the multi-modal
spread) vs. switching high-S/N bins to bootstrap outright (correct but
`n_boot`x more expensive per bin) vs. leaving Laplace with this
documented caveat.

---

# 2026-08-09: the actual root cause of the huge error bars, on both
# targets -- combine_bin_spectrum_robust's "empirical" term was measuring
# real surface-brightness structure, not noise

**Trigger**: user built and ran the new `kinextract_feature_testing.ipynb`
(N4751_notebooks) -- a lite, 5-bin sandbox for toggling kinextract's
fitting options without a full cube reload -- and immediately saw central-
bin V/sigma 1-sigma bands spanning hundreds of km/s, called it unrealistic,
and asked directly: "you are assuming [the MUSE STAT extension] is
variance and not sigma, right?" They independently tested dividing the
combined spectral errors by 10 before fitting and got a much more sensible
result (smoothly declining sigma with radius, tight V/h3/h4 bands),
strongly suggesting the reported errors were too large by something like
an order of magnitude.

**First, checked the specific hypothesis directly: is the STAT (variance)
extension being sqrt'd correctly?** Yes -- `load_muse_cube` does
`error = np.sqrt(variance)` at load time (`muse_cube.py`), and
`combine_bin_spectrum_robust` correctly squares back to variance
(`error_pix ** 2`) before combining and takes `sqrt` again at the end. No
missing/duplicated sqrt anywhere in this path.

**The actual bug, found by inspecting `combine_bin_spectrum_robust`'s own
"empirical error" term directly.** That function computes
`error = sqrt(formal_error**2 + empirical_error**2)`, where
`empirical_error = biweight_scale(flux_masked, axis=1)` -- the biweight
*scatter of the raw per-spaxel flux values* across every spaxel in the
bin, at each wavelength. Measured directly on real N4751 data (a
representative wavelength, several bins spanning n_spaxels=80-4675):
`empirical_error` is **120x to 470x** the `formal_error` (the actual,
correct inverse-variance-weighted noise estimate) -- e.g. bin0605's raw
flux ranges from 45.5 to 213 across its 80 spaxels, a real ~5x brightness
gradient, not scatter. Since the two terms combine in quadrature and one
is 2-3 orders of magnitude larger, the reported error is for all practical
purposes *just this term*, having almost nothing to do with measurement
noise.

**Traced to a misapplied port.** The function's own docstring claimed
fidelity to the legacy SCO Fortran framework's `imcmb.f`
(`sco_framework/modprogs/imcmb.f`) -- read that source directly to check.
`imcmb.f` reads a *list of FITS files* (`filea(1)...filea(n)`, from
`imcmb.def`) and biweight-combines them pixel-by-pixel: `n` is the number
of **separate exposures/visits of the same already-spatially-binned
spectrum**, not spatial pixels within one exposure. Across repeat
exposures of one fixed sky position, scatter in raw flux genuinely *is*
noise (true flux shouldn't change between visits) -- `biwgt`'s use there
is entirely correct. kinextract's Python port applies the identical
formula to a completely different situation: spatially different spaxels
*within one bin, from a single MUSE exposure*. Different spaxels have
genuinely different true flux (surface-brightness gradient across the
bin's spatial extent), so the same statistic instead measures real
astrophysical structure. This explains the population pattern observed
all session (`chi2/dof` lowest, error bars worst, specifically for the
*largest*/highest-n_spaxels bins on both targets) -- bigger bins span more
physical extent, so they have more real internal gradient to
misattribute as noise, not less noise per the usual S/N-improves-with-
more-spaxels intuition.

**The fix (`combine_bin_spectrum_robust`, `muse_cube.py`)**: removed the
empirical term entirely. Error is now the formal inverse-variance combine
alone -- valid for spatial spaxel combination (unlike the empirical term),
though (like any independence-assuming formula) it can still under-report
error if per-spaxel noise is itself positively correlated from
resampling/drizzling. That specific failure mode already has a real fix
elsewhere in the pipeline (`noise_model.whiten`'s AR(1) `rho` machinery,
applied at fit time) -- restoring the *intended* division of labor this
function's own "what this does not do" section already described, rather
than trying to solve the same problem twice, badly, in one place.
`min_spaxels_for_empirical` (now meaningless) removed with it.
`biweight_location` (the robust flux combine itself, resistant to a
single cosmic-ray-contaminated spaxel) is unaffected and unchanged.

Full test suite: `tests/test_muse_cube.py` updated (the correlated-noise
test now asserts the formal-only error exactly matches a hand-computed
inverse-variance combine, rather than asserting the removed empirical
term inflates it; the variance-correction-factor test now asserts exact
2x scaling, no longer diluted by an empirical term) -- 14/14 passing.

**Scope and blast radius**: this affects every real-data notebook on both
targets (`07_full_galaxy_analysis.ipynb`/NGC 5102 and
`n4751_full_galaxy_analysis.ipynb`/N4751 both call
`combine_bin_spectrum_robust` directly), and very plausibly explains a
large fraction of *everything* investigated in this session's earlier
entries above under the "over-generous error budget" banner -- those
entries chased `rho`, inflation-gating, and `lam` selection as
contributing causes (all real, all independently confirmed, none of them
this large) without yet finding this one, which measured 100x+ where
those measured 3-10x. **Not yet re-validated end-to-end**: no full
275-bin (NGC 5102) or 115-bin (N4751) production re-run has been done
against this fix yet -- next step, once the smaller-scale check below (or
a fresh one) confirms the fix behaves as expected on real cached-vs-
regenerated bins.

---

# 2026-08-09 (later): full 275-bin NGC 5102 production re-run against the
# `combine_bin_spectrum_robust` fix -- confirmed, dramatic, population-wide

**Re-ran `07_full_galaxy_analysis.ipynb` fresh** (measure_rho=True,
min_inflation_to_apply=1.1, smooth_window_A=3.0, fixed lam=4096/
penalty_order=3 -- everything else this session already validated,
now combined with the `combine_bin_spectrum_robust` fix above). 25.7 min
for 275 bins (down from the 40-60 min pre-fix runs -- well-calibrated
errors converge faster, a second independent confirmation the fix is
doing what it should, not just a coincidence of machine load).

**Population-wide, before -> after this specific fix** (both runs already
had every other fix from this session's earlier entries):

| metric | before (empirical-term bug present) | after (fixed) |
|---|---|---|
| median `sherm_std` | 18.7 km/s | **1.49 km/s** |
| p90 `sherm_std` | 25.4 km/s | **2.49 km/s** |
| bins with finite `sherm_std` | 273/275 (2 NaN) | **275/275** |
| median sherm | 37.6 km/s | 33.8 km/s |

A ~12.5x reduction in the population's own median error bar, and the 2
previously-NaN bins (175, 217 -- degenerate near-spike fits, flagged as
unresolved in the 2026-08-08 entry above) now fit cleanly -- both were
downstream symptoms of the same over-generous error budget, not separate
bugs needing their own fix.

**The specific bins flagged throughout this session (305/306/315, the
r=26" outer-ring, highest-n_spaxels bins) are now stable and sensible**:
sherm 49.6-57.3 km/s (consistent across all three, previously 148.8/
87.9/133.1 at the very start of this investigation), `sherm_std` 2.9-3.3
km/s. Clip fraction there is still 16.5-17.9% -- unchanged, expected,
already diagnosed as the real MUSE red-end sky-residual pattern (2026-08-08
entry above), not something this fix touches.

**One thing worth a closer look, not urgent**: `chi2_dof_prelim` (the
preliminary pass's own diagnostic, not what `min_inflation_to_apply` gates
on) is population-median 8.2 but hits 2000+ at exactly those same 4
outer-ring bins (305/306/315/316) -- far higher than anywhere else in the
population (the next-highest bin is 1240). Plausibly an artifact of the
preliminary pass's own light, fixed mid-grid lam (`_prelim_cfg_for_inflate`,
~lam=64 for `penalty_order=3`) being a poor fit specifically for these
bins' real structure before the final pass's own `lam=4096` engages --
the *final*, reported numbers for these bins are sane (above), so this
does not look like a live problem, but the sheer size of the number is
worth understanding rather than dismissing outright if anyone revisits
this area.

**Visual confirmation**: the radial profile plot (V/sigma/h3/h4 vs. signed
distance along the major axis) now shows a clean rotation curve, a
sensible sigma profile (central peak ~30-47 km/s declining outward to
~25-36 km/s with a mild rise again at the largest radii, tight error bars
throughout) and near-zero h3/h4 with small error bars -- a complete,
qualitative transformation from every version of this plot shown earlier
in this session, and the first one that doesn't need a caveat attached.

**N4751's own production notebook** (`n4751_full_galaxy_analysis.ipynb`)
was updated the same way (measure_rho=True, min_inflation_to_apply=1.1,
smooth_window_A=3.0, same `combine_bin_spectrum_robust` fix, continuum-
cofitting kept on per that notebook's own established setup) and re-run
fresh, 115 bins, 48.4 min (down from 109-186 min pre-fix -- the same
faster-convergence signature seen on NGC 5102, plus the `measure_rho`
continuum-skip optimization). 0 errors.

**Clip fraction dropped sharply, population-wide**: 0.6% overall (median
0.0% per bin, max 5.9%) -- down from whatever the pre-fix run showed
(not directly comparable, since this run also carries every other fix
from this session, but qualitatively consistent with the same mechanism
that fixed NGC 5102's clip fraction). `bin0105` (the delta-spike bin from
the screenshot that kicked off this whole investigation) now fits with
`n_clipped=0` and sigma=685.3 km/s -- not obviously wrong (the radial
profile's own independent lookup already put this target's central sigma
at 597-630 km/s from a completely different measurement), and the LOSVD
itself (not shown here, see the notebook's own Section 5 diagnostic plots)
is a clean single-peaked profile, not a spike.

**h3/h4 are real, and noisier than NGC 5102's** -- population median h3
0.060, h4 0.018, but 17/115 bins (~15%) show `|h3| > 0.2`, well outside
the range Gauss-Hermite moments normally take for a physically reasonable
LOSVD. This is consistent with, not contradicted by, the regularization
validation work above: N4751's real sigma range (400-700+ km/s) is
exactly where that work found real, non-trivial recovery bias and large
scatter persist even at the best available `lam`, not a new problem this
run introduced. The radial profile itself (V, sigma, h3, h4 vs. signed
distance along the major axis -- no error bars, this notebook defers
uncertainty to the GreatLakes handoff) shows a physically sensible
rotation curve (smooth ~250 to -350 km/s S-curve) and a rough central
sigma peak (~600-1000 km/s near center, ~200-500 km/s further out), with
h3/h4 scatter concentrated exactly where the rotation curve is steepest --
a real result worth reporting, not a broken one, but one that should be
read with N4751's own harder-regime validation caveats attached, unlike
NGC 5102's now-clean plot above.

---

# 2026-08-10: N4751's own pipeline was never properly built for N4751 --
# noise correlation read NGC 5102's files, measure_rho regressed low-S/N
# bins, and the inflation cap was silently too low for this target

**Trigger**: user reported the previous entry's own N4751 radial profile
(V/sigma maps) had "regressed significantly... a ton of scatter and
nonsense" compared to an earlier, better-looking version, and separately
flagged that `n4751_noise_correlation.ipynb` "has ngc5102 files being
read in." Both turned out to be real, and mostly independent.

**Bug 1 -- `n4751_noise_correlation.ipynb` was a literal, unadapted copy
of NGC 5102's `05_noise_correlation.ipynb`.** `REP_BINS_PATH` pointed at
`ngc5102_representative_bins.npz`, `Z` was NGC 5102's systemic redshift,
every conclusion described "NGC 5102" spectra by name. N4751 never had an
equivalent representative-bins file, and `n4751_template_calibration.
ipynb` never produced one. Fixed both ends: added a new Section 11 to
`n4751_template_calibration.ipynb` that picks high/med/low-S/N bins
directly from that notebook's own already-computed `bin_table`/`snr`
(Section 5) and saves `ngc4751_representative_bins.npz` in the same
observed-frame, pre-windowing format `05_noise_correlation.ipynb`-style
notebooks expect; rebuilt `n4751_noise_correlation.ipynb` from NGC 5102's
own notebook as a base, systematically adapted (target name, `Z`,
`WAVEFITMIN`/`WAVEFITMAX` to match this target's own 8300-8850 A window,
file paths, no fixed `SKY_RESIDUAL_WINDOWS` list -- matching this
target's own established convention of letting `fit_losvd_robust`'s
per-spectrum inflation handle that adaptively instead), then re-ran it
fresh against real N4751 data. Real, validated result: **`rho = 0.708`**
on the aperture-summed high-S/N spectrum, cross-checked at `rho = 0.591`
on a single brightest spaxel (no aperture-summing at all) -- both real,
both in a similar range, confirming genuine pixel correlation and not an
aperture-summing artifact. `chi2_red` under the independence assumption
is 941.94 even on the high-S/N aperture -- far worse than NGC 5102's own
(11-28), consistent with everything else found on this target (huge
inflation needed, below).

**Bug 2 -- `measure_rho=True`'s own per-bin estimate is unreliable at low
S/N, and this measurably regressed results relative to the pre-`measure_
rho` state.** Direct comparison, same two nearly-identical mirror-image
bins, before vs. after `measure_rho` was added: `bin0403`/`bnn0403`
(n_spaxels=3 each) gave sherm 507.4/506.6 (agreeing to within 1 km/s)
under the old fixed-`rho=0` approach, vs. 996.2/411.8 (a 2.4x
discrepancy) once `measure_rho=True` was turned on -- traced directly to
`rho_used` itself: 0.584 vs. 0.261 for these two bins, a difference that
is measurement noise in the rho estimate (from a preliminary fit's
residual on a 3-spaxel-combined spectrum), not a real difference between
mirror-image bins at the same radius. This is the direct mechanism behind
the "checkerboard" scatter in the sigma/h3/h4 maps -- confirmed by
inspecting the full population: wild bin-to-bin swings appear specifically
in the smallest-n_spaxels bins, largest at exactly the radii where
`measure_rho`'s own precision is worst. **Fix**: gate `measure_rho`'s use
on `n_spaxels` (only available to the notebook, not to `fit_losvd_robust`
itself, so this is a notebook-level fix, not a package one) --
`MIN_SPAXELS_FOR_MEASURE_RHO=20`; bins at or above that get their own
per-bin measured rho as before, bins below it fall back to the newly-
validated `FALLBACK_RHO=0.708` from bug 1's own fix. Required restructuring
`n4751_full_galaxy_analysis.ipynb`'s own fitting loop from single
index-range batches into two groups (by this threshold), each with its
own `measure_rho` setting, reassembled by bin index afterward -- `fit_
losvd_robust_parallel`'s own `measure_rho` kwarg is uniform across an
entire parallel call, it cannot vary per-bin within one call.

**Bug 3 -- `INFLATION_CAP=8.0` (the package default, validated on NGC
5102 where it's never actually reached) was silently binding for the
large majority of N4751's own bins.** Direct check on the just-completed
run: **83/115 bins (72%) hit `inflation_factor.max() == 8.00` exactly** --
completely unlike NGC 5102, where this same cap was never reached in
practice. Tested directly on one heavily-affected bin (`bin1305`,
n_spaxels=2992): raising the cap 8 -> 30 -> 100 -> 300 shows inflation
naturally stabilizing at `max_infl=85.36` (no longer capped) by 100, with
no further change at 300 -- and recovered sigma for this bin drops from
1075.3 km/s (cached, cap=8) to 212.2 km/s (cap=100), a huge, direct
correction, not a marginal one. This is a real, target-specific
difference from NGC 5102 (much higher real sky/systematic contamination
amplitude relative to N4751's own, generally much brighter, spectra) --
confirmed the fix is *not* a universal one by re-testing the low-S/N
`bin0403`/`bnn0403` pair from bug 2 at the higher cap: their own
discrepancy was essentially unchanged (996->1006, 412->422), and their
own `max_infl` was already well under the old cap (7-8) -- i.e. bugs 2
and 3 are two independent, non-overlapping mechanisms, not the same bug
wearing two faces. **Fix**: `INFLATION_CAP` raised `8.0 -> 100.0`.

**Bug 4 (found while auditing, not reported directly)**: `n4751_full_
galaxy_analysis.ipynb`'s own `MUSE_VARIANCE_CORRECTION_FACTOR` was
silently `1.0` (`#/ 0.6 ** 2`, commented out, no explanation) while
`n4751_template_calibration.ipynb` applies the full Bacon et al. (2017)
correction (`1/0.6**2 ~= 2.78`) -- an unexplained inconsistency between
two notebooks processing the same target's same cube. Re-enabled to
match; `CACHE_TAG` updated to fingerprint it (it did not before, which
would have let a stale, wrong-correction-factor cache silently survive
this fix).

**Also fixed while touching this area**: `n4751_template_calibration.
ipynb`'s Section 5 markdown still described the since-removed "empirical"
error term (`combine_bin_spectrum_robust`'s own biweight-scale-of-raw-flux
term, removed in the entry above) as if it were still active -- updated
to describe the current, correct design.

**All four fixes combined, full 115-bin re-run**: in progress as of this
entry -- see this file's own next entry (or the notebook's own cache/
progress log directly) for the result. Given bugs 2 and 3 are both
demonstrated on individual bins to produce large, physically-implausible
sigma swings (2.4x and 5x respectively) that vanish once fixed, a
substantially cleaner population-wide result is expected, but "expected"
is not "confirmed" -- do not treat this entry alone as validation that
the full run is fixed.

---

# 2026-08-10 (later): full 115-bin re-run against all four fixes above --
# confirmed, real improvement, with an honest remaining limitation

**Re-ran `n4751_full_galaxy_analysis.ipynb` fresh** against all four
fixes above (measure_rho gated at `n_spaxels>=20` with `FALLBACK_RHO=
0.708` below it, `INFLATION_CAP=100`, `MUSE_VARIANCE_CORRECTION_FACTOR`
re-enabled, `CACHE_TAG` now fingerprints all of it). 51.0 min, 0 errors.

**Clip fraction dropped from 0.6%/5.9% (overall/max-per-bin) to
0.1%/0.7%** -- confirms the inflation-cap fix is doing its job at
population scale, not just on the one bin it was tested on: contamination
that used to need hard clipping is now being properly downweighted
instead.

**The two bins used to demonstrate bugs 2 and 3 individually, re-checked
in the full run**: `bin1305` (bug 3's own test case) sherm=219.0 km/s,
matching the single-bin cap=100 test (212.2) closely; `bin0403`/`bnn0403`
(bug 2's own test case) now both correctly get `rho_used=0.708`
(`FALLBACK_RHO`, since `n_spaxels=3 < 20`) and sherm 518.7/481.2 -- close
agreement, a world away from the old 996.2/411.8 split driven by noisy
per-bin rho estimates.

**Population-wide**: h3 `|h3|>0.2` count dropped 17/115 -> 9/115 (roughly
half); max `|h3|` 0.44 -> 0.40. Sherm's own overall range is similar
(211-1008 km/s vs. 204-1075 before) -- the fixes correct specific,
identifiable bins substantially (as shown directly above) without
uniformly shrinking the whole population's spread, which is the expected,
honest result: bugs 2 and 3 were real, identifiable, now-fixed mechanisms,
not a single global miscalibration.

**Visual confirmation**: the radial profile (V/sigma/h3/h4 vs. signed
distance along the major axis) now shows a clean, smooth rotation curve
and a coherent sigma profile (a real central peak, smooth decline, then a
rise again at the largest radii) -- a qualitative transformation from the
"checkerboard" scatter that triggered this entire investigation. Some
scatter remains, concentrated at the same extreme bins as before (the
very center, 1-4 spaxels each) -- **this is not evidence the fix failed**;
it is the same, already-documented, genuine low-information-content
limitation this session's earlier entries established directly (e.g.
`bin0105`'s own 92-spline-coefficient basis for a single spaxel, and the
regularization-validation entry's own finding that real recovery bias and
scatter persist at N4751's sigma range even at the best available `lam`).
Rho-measurement noise and an under-sized inflation cap were real,
fixable, and now fixed; a handful of single-digit-spaxel bins being
intrinsically hard to pin down is a different, deeper, and honestly
reported limitation, not a bug still waiting to be found.

**Scope note**: `n4751_noise_correlation.ipynb` and `n4751_template_
calibration.ipynb` (Section 11's new representative-bins extraction) are
both now real, N4751-specific notebooks producing real, validated
N4751-specific numbers -- the "properly organize and run the full
pipeline" ask this whole entry responds to. All three N4751 notebooks
(template calibration -> noise correlation -> full galaxy analysis) now
run in that order, each consuming the previous one's own real output,
with no cross-target file contamination anywhere in the chain.

---

# 2026-08-10 (later still): template-calibration notebooks were fitting
# under a different, stale method than their own full-galaxy notebooks --
# unified, plus a real package bug found doing it

**The problem.** `04_template_calibration.ipynb`/`n4751_template_
calibration.ipynb`'s own Section 7 fit (full-636-template NNLS pass that
picks the template subset every downstream fit uses) was running under
settings that had quietly drifted away from their own target's
`07_full_galaxy_analysis.ipynb`/`n4751_full_galaxy_analysis.ipynb`:
fixed `lam=2.0`/`penalty_order=2` (the pre-2026-07-30 default, since
superseded everywhere else by the validated `lam=4096.0`/`penalty_order=
3` combination -- see `dev_notes/continuum/FINDINGS.md`'s 2026-07-30
sweep), no `measure_rho` at all (`cfg.rho` silently `0.0`, i.e. assuming
zero pixel correlation on real MUSE data whose actual correlation runs
0.6-0.9), `SMOOTH_WINDOW_A=8.0` (pre-fix value, not the validated 3.0),
no emission-line masking, and target-specific gaps: N4751's own
`WAVEFITMIN/WAVEFITMAX` was `8000/9000` vs. the real fit's `8300/8850`,
`INFLATION_CAP=8.0` vs. the real fit's validated `100.0`, and no
continuum-cofitting at all vs. the real fit's `CONTINUUM_KNOTS=4`.
Template *selection* -- which stellar populations survive into every
downstream science fit -- was happening under a materially different,
less-validated fitting method than the fit those templates then get
used in. Flagged directly by the user as a scientific-integrity risk,
not a cosmetic inconsistency.

**The fix, both notebooks.** Matched every one of the above to each
target's own validated full-galaxy settings: `lam=4096.0`/`penalty_order=
3`, `SMOOTH_WINDOW_A=3.0`, emission-line masking
(`cleaning.detect_emission_lines`) added to the per-bin extraction step,
`measure_rho=True` gated at `n_spaxels>=20` with a `FALLBACK_RHO` below
that threshold (`0.88` for NGC 5102, from `05_noise_correlation.ipynb`;
`0.708` for N4751, from `n4751_noise_correlation.ipynb`) -- necessary
here specifically because these radial-annulus bins span a much wider
`n_spaxels` range than the kinematic bins measure_rho was originally
validated against (N4751's own nuclear annulus has `n_spaxels=1`).
`reduce_templates_iteratively` (Section 8) fits every bin at one shared
`cfg`, so it can't carry a per-bin measured rho -- uses the same
`FALLBACK_RHO` there instead of silently defaulting to `rho=0.0`. N4751
additionally: `WAVEFITMIN/WAVEFITMAX` -> `8300/8850`, `INFLATION_CAP` ->
`100.0`, and `CONTINUUM_KNOTS=4`/`LAM_C=2.0`/`RIDGE_C=0.1` added to match
the real fit's continuum-cofitting. The velocity-grid *extent* (a fixed,
generously wide range rather than the full-galaxy notebooks' per-bin
`NSIG*sigma0` grid) was deliberately left as-is in both notebooks -- a
structural necessity, not a fitting-method inconsistency: this section
is what *produces* the radial `(V, sigma)` profile the per-bin grids
center on, so there is no profile yet to center against here.

**A real package bug found making this consistent, not just a notebook
edit.** `reduce_templates_iteratively` whitened by hand
(`A_stack/errors`, `flux/errors`) instead of calling `noise_model.whiten
(..., rho=cfg.rho)`, so it silently whitened as if `rho=0.0` regardless
of what `cfg.rho` actually held -- a caller passing a real measured or
fallback correlation had it quietly ignored. It also never passed
`wave_fit` to `fit_losvd`, so it raised `ValueError` the instant
`cfg.continuum_knots` was set (`fit_losvd` needs the wavelength grid to
build a continuum B-spline basis) -- N4751's own re-run hit this
directly the first time it ran with the new, consistent `cfg`. Fixed in
`src/kinextract/template_reduction.py`: whitens via `noise_model.whiten`
and passes `wave_fit` through to `fit_losvd`/`full_pixel_model` (the
latter replacing a bare `A_eff_w @ b` chi2 calculation that also didn't
account for the continuum multiplicative factor under
`cfg.continuum_knots`). Two regression tests added
(`test_cfg_rho_is_actually_respected_not_silently_whitened_as_zero`,
`test_continuum_knots_runs_without_crashing`,
`tests/test_template_reduction.py`) -- both would have failed against
the pre-fix code. At `rho=0.0` (every existing caller/test before this
session) `whiten` reproduces the old bare-division path bit-for-bit by
its own explicit design, so this is a pure bug fix, not a behavior
change for anyone not setting `cfg.rho`.

**Also fixed in `n4751_template_calibration.ipynb`'s own Section 7
plotting cell**: its per-bin diagnostic plot de-whitened the fitted model
back to flux units via `model_w * errors_inflated`, which is only exact
at `rho=0.0` -- AR(1) whitening at `rho!=0` mixes neighboring pixels
through `noise_model.whiten`'s own Cholesky factor, so an elementwise
multiply can't undo it. Not a live bug until this session (`cfg.rho` was
always effectively `0.0` here before), but a latent one that
`measure_rho=True` would have hit immediately. Fixed by calling
`full_pixel_model` against the *unwhitened* `A_stack` directly -- exact
regardless of `rho`, since `reconstruct_continuum` (what `full_pixel_
model` uses internally) depends only on `wave_fit`/`cfg`, not on which
`A_stack` was passed in.

**Validated, full re-run of both targets' entire chains** (template
calibration -> noise correlation -> full galaxy analysis), all four
notebooks, 0 errored cells anywhere:

- `04_template_calibration.ipynb` (NGC 5102): decoy-drop-out still
  converges cleanly under the new `lam=4096`/`penalty_order=3` (30/636
  templates survive, `converged=True` in 5 iterations) -- confirms the
  old Section 13 markdown's warning was specifically about
  `penalty_order=3` *with the old* `lam=2.0` (essentially unregularized
  under that penalty order), not about `penalty_order=3` generally; the
  actual target combination (`lam=4096`) was never the risky one.
  `measure_rho` gating behaved exactly as designed: `rho_used=0.88`
  (fallback) for the `n_spaxels<20` bins, `0.70-0.89` (measured) smoothly
  varying by radius elsewhere. The radial sigma profile is now
  `~20-47 km/s` (physically plausible for this target) instead of the
  old fit's suspiciously flat `~100-130 km/s` wall.
- `n4751_template_calibration.ipynb`: decoy-drop-out converges (26/636,
  `converged=True` in 3 iterations). `rho_used` fallback `0.708` for
  `n_spaxels<20` bins, measured `0.70-0.93` elsewhere, smoothly varying
  by radius. Radial sigma profile now `~305-716 km/s`, consistent with
  this target's known high-dispersion regime (the same `400-700 km/s`
  range `dev_notes/regularization/FINDINGS.md`'s own sweep validated
  `lam=4096` against directly).
- `n4751_noise_correlation.ipynb`: measured rho `0.699` (aperture-summed)
  / `0.579` (single brightest spaxel) -- consistent with the prior
  `0.708`/`0.591` to within measurement noise; `FALLBACK_RHO` left at
  `0.708` in the downstream notebooks rather than churning a third
  re-run over a <2% shift.
- `n4751_full_galaxy_analysis.ipynb`: re-run with `refit=True` -- **the
  first attempt silently loaded a stale, pre-this-fix on-disk cache and
  finished in seconds**, because `CACHE_TAG` does not fingerprint the
  upstream template-subset/radial-kinematics files this notebook reads;
  caught by the suspiciously-fast completion, not by any error. The real
  re-run: 37.1 min, 115/115 bins finite, sigma median `401.5 km/s`
  (range `196.9-671.7`), `|h3|>0.2` count `5/115` (down from the
  already-fixed `9/115` two entries above -- a further improvement from
  this consistency pass alone), only 32 total clipped pixels across all
  115 bins.
- `07_full_galaxy_analysis.ipynb` (NGC 5102): `FORCE_RECOMPUTE=True`
  already set, no stale-cache risk. 275/275 bins finite, sigma median
  `30.2 km/s` (range `12.9-48.1`), `sherm_std` median `1.95 km/s`
  (consistent with this session's earlier `1.49 km/s` validated
  transformation), zero bins with `|h3|` or `|h4| > 0.2`.
- Full `kinextract` test suite: 347 passed (2 new), 1 skipped, 0 failed.

**Lesson for next time a notebook's own upstream `.npz`/cache dependency
changes**: `CACHE_TAG`-style fingerprinting only protects against the
parameters it enumerates -- neither `n4751_full_galaxy_analysis.ipynb`'s
nor (by the same construction) `07_full_galaxy_analysis.ipynb`'s own
`CACHE_TAG` includes a hash/mtime of the template-subset or
radial-kinematics files they load, so a cache built before those files
changed is silently treated as still valid. Always force a real refit
(`refit=True`/`FORCE_RECOMPUTE=True`) after regenerating anything
upstream, and don't trust a full-galaxy re-run that finishes in seconds
without checking which branch it actually took.
