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
