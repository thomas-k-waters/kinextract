# LOSVD shape-residual problem: handoff report

**Written**: 2026-07-26, for a fresh investigation (a different Claude
model, "Fable 5") to continue without re-deriving what's already been
tried. Everything in this document has been checked directly, with code
shown or referenced -- nothing here is a guess dressed up as a finding.

---

## STATUS UPDATE (2026-07-26, Fable 5 session — read this first)

The investigation below was continued the same day; chronological
detail in `REGULARIZATION_INVESTIGATION_LOG.md`, scripts/CSVs in
`dev_notes/shape_residual/`. Summary of what is now KNOWN (each item
verified directly, seeds/sweeps in the log):

1. **Root cause found.** The shape defect is a property of the
   *penalty form*, not the strength selection: the D2 (second-
   difference) penalty's null space is *linear* log-b, so heavy
   smoothing pushes toward exponential tails (the already-documented
   wing-inflation mechanism) while light smoothing overfits the core.
   The two failure modes squeeze `lam` from both sides — an oracle
   (truth-informed) lam under D2 still fails the 5% target at
   realistic S/N (only 2/15 seeds pass at S/N=30, 5/15 at S/N=60;
   sigma=50 single-template).
2. **The fix that works: a third-difference (D3) penalty** — null
   space = quadratic log-b = *any Gaussian*, so heavy smoothing
   shrinks toward the best-fitting Gaussian instead of fat tails.
   Wing-mass inflation disappears by construction (ratio ~1.0 even at
   lam=16384 where D2 hit 45x), sigma-moment bias at sigma=50 drops to
   the known -1.2% windowing floor, and paired oracle-vs-oracle D3
   beats D2 at essentially every S/N on Gaussian truth (e.g. 2.67% vs
   4.54% mean at S/N=100). `aic_local` (w_m=2) over an extended lam
   grid (1..32768) remains the selection rule — stiffer rules
   (discrepancy/1-SE) slam to the Gaussian limit and erase real h4.
3. **AIC-local under-smoothing explained**: m_eff varies by only ~1
   unit across a 256x lam range (the forward operator attenuates
   almost everything), so the chi2 term dominates and selection
   noise-chases to the grid bottom. No reweighting of the same
   criterion fixes D2 (w_m=4/8, GCV, discrepancy all tested — see
   log); changing the penalty form is what works.
4. **Hard information limits mapped — the 5%-everywhere goal is NOT
   achievable in three regimes, for any method** (parametric
   included): (a) sigma~20 at S/N<=30 (deconvolution through
   sigma_inst~35 km/s templates supports ~3 effective dof, m_eff=3.000
   at every lam); (b) non-Gaussian (|h3| or h4 ~ 0.1) truth at
   sigma~50, S/N<=100 — a direct 5-param parametric GH fit has
   sigma(h3)=0.15, sigma(h4)=0.19 at S/N=60, i.e. the distinguishing
   structure is below the estimation noise floor; (c) any truth with
   exact zeros inside the fit range (clipped-GH benchmark shapes) —
   the spectrum barely changes (chi2 ~ noiseless-perfect) while the
   LOSVD differs by ~8%, an ill-posedness floor.
5. **Two benchmark-methodology traps found** (both can corrupt future
   validation if not respected): (a) the "noise=1e-6 of signal"
   noiseless convention breaks the float32 JAX objective — whitened
   design values ~1e6 stall the optimizer (40% residual with chi2=8e6
   on data a 1e-4 whitening fits to 1.7%); use S/N~1e4 as the
   noiseless proxy. (b) At the `multi_template.fit_losvd` level the
   multi-start diagnostic is `losvd_objective_gap`, not
   `objective_gap`.
6. **Ruled out**: multi-start/local-basin accidents (no correlation of
   `losvd_objective_gap` with residual; gaps ~0.1-1 vs objective
   ~220); GH-anchored D3 (anchor noise dominates at low sigma — full
   negative result in the log); D3 knot-spacing scaling exponent 3
   (no better than the existing exponent-1 convention).
7. **RESOLVED (2026-07-27): the fix shipped as the package default.**
   The full paired validation ran three times (float32: `06_*`; x64
   both arms: `06b_*`; the production configuration end-to-end:
   `06c_*`/`06d_*` — 480 paired cases each, identical draws). Final
   configuration, now the default: **`FitConfig.penalty_order=3` + JAX
   float64 at package import (opt-out `KINEXTRACT_LITE_NO_X64=1`) +
   two order-3-gated selection safeguards** (delta-AIC<=2 parsimony
   tiebreak; sub-knot-scale-sigma candidate guard — the latter closes
   a delta-spike collapse mode where the linearized m_eff
   underestimates a spike's flexibility so badly that it wins the AIC
   comparison outright). Verdict vs the old default: better mean shape
   residual in 23/32 cells (332/480 paired wins), overall |sigma bias|
   5.6 vs 15.5 km/s, wing-mass ratios ~1.0 everywhere with `wing_lam`
   off. The sigma=300/S N=30 anomaly was two stacked causes — clamped-
   boundary null-space leakage (real, secondary) and, decisively, a
   float32 optimizer stall (fixed by x64). Documented tradeoff: order 3
   shrinks real h3/h4 by ~0.03-0.05 at sigma 50-140 / S/N<=100 (where
   h4 is near the information limit for any estimator);
   `penalty_order=2` reproduces the old behavior exactly.
   See `VALIDATION_CAMPAIGN.md` Stage 22 for the condensed record and
   `REGULARIZATION_INVESTIGATION_LOG.md` for the full chronology.

The rest of this document is the original handoff, unchanged, for the
pre-existing context it records.

---

## The goal, stated precisely

Recover the full **shape** of a line-of-sight velocity distribution
(LOSVD) `b(v)` from a galaxy spectrum, not just its summary moments
(V, sigma, h3, h4), with:

- **Residual `|b_recovered(v) - b_true(v)| / max(b_true)` under 5%
  everywhere** the true LOSVD carries meaningful density -- not just low
  bias in the four Gauss-Hermite moments computed *from* the recovered
  curve. A recovered curve can have deceptively good moment-level bias
  while still being visibly the wrong shape (see below) -- moments are
  not a sufficient proxy for shape fidelity, and should not be treated as
  one going forward.
- **Across the full physical regime** this package is meant to cover:
  true sigma from ~20 to ~400 km/s, S/N from ~15 to ~200+, single
  templates through real, uncurated multi-star template libraries.

This is a sharper, harder bar than anything previously validated in this
project. Prior validation (see "What's been tried" below) almost
entirely measured moment-level bias (`|sigma_recovered - sigma_true|`,
etc.), not full-curve shape residual. That gap is the direct motivation
for this document.

## The package and its current architecture

`kinextract_lite` (`/Users/waterstk/Documents/Gultekin_Astrophysics/kinextract_lite/`)
is a from-scratch, independent LOSVD extraction package (a companion,
non-parametric alternative to a separate legacy Fortran pipeline,
`sco_framework`). Read `VALIDATION_CAMPAIGN.md` (in the repo root) in
full -- it is the project's own chronological log of every stage,
finding, and decision, and this document does not repeat its contents,
only points at the relevant stages.

**Representation** (`src/kinextract_lite/losvd.py`): `b(v) =
exp(basis @ coeffs)`, `basis` a cubic B-spline design matrix
(`build_bspline_basis`), `coeffs` the free parameters. The `exp` link
makes positivity structural (no boundary at `b=0`). Knot count
auto-scales with grid width via a target absolute knot spacing
(`FitConfig.knot_spacing_kms`, default 95 km/s) rather than a fixed
count, with a `min_interior_knots` floor that binds at small sigma (this
floor binding is *itself* a documented source of a real scale confound --
see Stage 21 below, directly relevant to this investigation).

**Fitting**: a single L-BFGS-B optimization per fit (JAX autodiff for
exact gradients through the `exp` link), from a small multi-start pool
(`_default_multi_starts`: one data-driven ridge estimate, two flat
starts), keeping whichever converges to the lowest objective. Objective:
`0.5*||A_w @ b - y_w||^2 + 0.5*lam*||D2 @ coeffs||^2` (`D2` a discrete
second-difference operator on the coefficients -- a standard P-spline/
Eilers & Marx 1996 curvature penalty), plus an optional wing-amplitude
term (see below).

**Template mixtures** (`src/kinextract_lite/multi_template.py`): for
`n_templates > 1`, alternates between (a) fixing template weights `w`
and fitting the LOSVD as above against the effective design matrix
`sum_k w_k * A_stack[k]`, and (b) fixing `b` and solving `w` via NNLS
(convex, exact). Continuum-cofitting (`kinextract_lite.continuum`) adds
a third alternating block for a shared multiplicative continuum
`C(lambda)`.

**Regularization strength selection**:
- `lam` (curvature weight): default is **adaptive** per fit via
  `aic_regularization.select_lambda_aic_local` (minimizes chi2 + 2*m_eff
  over a small candidate grid, `cfg.lam_grid`, each candidate a full
  multi-start fit -- this is why an adaptive fit costs ~8x a fixed-`lam`
  fit). This is the current package-wide default
  (`FitConfig.lam_selection='aic_local'`).
- `wing_lam` (amplitude penalty in the far wings only, via a smooth
  taper `build_wing_taper`): **off by default** (`wing_lam=0.0`). Exists
  because the curvature-only penalty has no amplitude term at all, so it
  does nothing to suppress a noise-driven, non-zero pedestal far from
  the peak -- and *more* curvature smoothing makes this specific failure
  mode *worse*, not better (see mechanism below). No working
  auto-calibration rule exists; three well-tested approaches were tried
  and failed (see "What's been tried," `losvd.py`'s own docstring, and
  `FitConfig.wing_lam`'s help text for the full detail -- do not
  re-attempt these three without reading that first).
- `spectral_prior` (`src/kinextract_lite/spectral_prior.py`, new this
  session, exploratory/unvalidated): generalizes the curvature penalty
  into a 2-parameter spectral law over the penalty operator's own
  eigenmodes. See Stage 19-21 summary below -- substantial work, real
  partial progress, **not yet a solution to the problem this document is
  about** (see "The core new finding," which shows this method performs
  almost identically to the current default on the shape-residual
  metric).

## The core new finding of this diagnostic session

**Moment-level bias and shape-residual are not the same thing, and the
gap between them is large.** Concrete, fully reproducible example
(code in the Appendix): single real E-MILES template, true V=20 km/s,
true sigma=50 km/s, S/N=60, seed=0, fit both with the current default
(`multi_template.fit_losvd`, adaptive `lam`) and with
`spectral_prior.fit_losvd_with_spectral_prior` at its best validated
setting (`alpha=100, beta=1.5, normalize_eigenvalues=True`):

| | V | sigma | h3 | h4 |
|---|---|---|---|---|
| truth | +20.00 | 50.00 | 0.0000 | 0.0000 |
| default | +16.55 | 47.68 | -0.0379 | +0.0252 |
| spectral-prior | +16.43 | 46.61 | -0.0423 | +0.0191 |

Sigma bias here is small (~4-7%) -- by the project's own existing
moment-bias standard, this looks like a good fit. **But the recovered
curve's actual shape is not close to the truth**: plotted against the
true LOSVD, the recovered curve (both methods, visually indistinguishable
from each other) is visibly **taller and narrower at the peak** than the
truth. Quantified (`|recovered - true| / max(true)`, restricted to where
`true > 1%` of its own peak -- the same convention already used
elsewhere in this project, e.g. `01_basic_mock_fit.ipynb`):

- **Default**: max residual **10.9%**, with **32.3%** of that velocity
  range exceeding the 5% target. Worst point is right at the peak
  (v=24.7 km/s, near the true center): true density 0.00794, recovered
  0.00881 -- an ~11% overshoot exactly where it matters most.
- **Spectral-prior**: max residual **12.9%**, **33.8%** of the range over
  5% -- essentially the same defect, confirming this is not something
  the spectral-law generalization fixes (see below for why that's
  itself informative).

**This is not a template-mixture-degeneracy artifact** -- it's visible
with a single template, no weight-recovery ambiguity at all.

**Isolated the noiseless case, and this rules out a representation/basis
limitation**: fitting the *exact same setup* with noise reduced to
~1e-6 of signal (effectively noiseless) gives max residual **3.8%**,
**0%** of the range over 5% -- comfortably under the 5% target. **The
B-spline/curvature-penalty representation, at whatever `lam` `aic_local`
selects, can represent this true shape to well under 5% residual when
there is no noise to fight.** The problem is therefore in how the method
handles **noise**, not an inability to represent the true curve at all.

**S/N-dependence and seed-to-seed variance, both checked directly**
(3 seeds x 5 S/N values, single-template sigma=50 case, default vs.
spectral-prior):

| S/N | seed | default max resid % | default frac>5% | spectral max resid % | spectral frac>5% |
|---|---|---|---|---|---|
| 30 | 0 | 11.6 | 38.5 | 16.1 | 46.2 |
| 30 | 1 | 6.2 | 12.3 | 4.4 | 0.0 |
| 30 | 2 | 14.7 | 53.8 | 12.5 | 49.2 |
| 60 | 0 | 10.9 | 32.3 | 12.9 | 33.8 |
| 60 | 1 | 3.4 | 0.0 | 3.9 | 0.0 |
| 60 | 2 | 6.5 | 16.9 | 4.7 | 0.0 |
| 100 | 0 | 8.7 | 23.1 | 8.1 | 21.5 |
| 100 | 1 | 2.8 | 0.0 | 3.0 | 0.0 |
| 100 | 2 | 4.5 | 0.0 | 2.6 | 0.0 |
| 200 | 0 | 5.8 | 6.2 | 5.5 | 4.6 |
| 200 | 1 | 2.2 | 0.0 | 2.7 | 0.0 |
| 200 | 2 | 3.3 | 0.0 | 2.1 | 0.0 |
| 500 | 0 | 2.3 | 0.0 | 4.4 | 0.0 |
| 500 | 1 | 2.7 | 0.0 | 5.0 | 0.0 |
| 500 | 2 | 2.5 | 0.0 | 2.3 | 0.0 |

Three things worth noting in this table:
1. **Clear overall trend of decreasing residual with increasing S/N**,
   as expected.
2. **Large seed-to-seed variance at fixed S/N** -- e.g. at S/N=30, seed 1
   gives 6.2% while seed 2 gives 14.7% (more than 2x). The defect is not
   a smooth function of S/N alone; specific noise realizations matter a
   lot.
3. **A nonzero floor even at very high S/N** (~2-5% even at S/N=500,
   not zero) -- plausibly the already-documented, deterministic ~1.2%
   windowing bias (see Stage 1 of the deep-validation history below)
   compounding with residual optimizer/basin noise, but this specific
   floor has not been separately isolated in this session.
4. **`spectral_prior` tracks the default almost exactly at every S/N and
   seed** -- sometimes marginally better, sometimes marginally worse,
   never a qualitative difference. Whatever this shape defect's mechanism
   is, it is **not primarily about the curvature-penalty's power-law
   shape** (`beta`) -- both a linear (`beta=1`, today's default) and a
   `beta=1.5` steeper-relative-to-smooth-modes law produce the same
   defect at similar magnitude. This is a real, informative negative
   result: whatever is causing the~10%+ shape residual at realistic S/N
   is likely a different mechanism than the sigma-dependent bias
   `spectral_prior` was built to address (see Stage 20/21 below) --
   don't assume a fix for one fixes the other.

## What's been tried already (do not repeat blind)

### In `kinextract_lite` itself (this package)

Full detail in `VALIDATION_CAMPAIGN.md` and in the memory files at
`~/.claude/projects/-Users-waterstk-Documents-Gultekin-Astrophysics-kinextract/memory/`
(`project_kinextract_lite_deep_validation.md` especially). Summary,
in the order these were found:

1. **Deterministic windowing bias in the moment estimator itself**
   (`compute_direct_moments`'s smooth taper), independent of the fit:
   exactly -1.22% sigma bias for a pure Gaussian at the default
   `core_threshold`, confirmed analytically. Shape-dependent (varies with
   h3/h4). **Tested applying a universal correction and found it makes
   aggregate bias *worse*** (a 25-seed, 13-case grid: -0.03 -> +1.92 km/s
   mean bias) -- not shipped, documented as a known small contributor
   instead. This is a moment-estimator effect, not a `b(v)`-shape effect,
   and is **not** what this document's shape-residual finding is about
   (that finding persists even restricting attention to the recovered
   curve itself, before any moment is computed from it).

2. **Wing-amplitude inflation, root-caused and partially fixed
   (`wing_lam`).** The curvature-only penalty has no amplitude term;
   more smoothing pushes `log(b)` toward *linearity* (a slowly-decaying
   tail), not compactness, since a compact peak's log-density is a
   downward parabola with real, non-zero curvature that the penalty
   actively fights. This *inflates* wing mass (a documented 6x wing-mass
   inflation at S/N=30) and hence sigma, and gets *worse* with more
   curvature smoothing -- a real, counter-intuitive, directly-confirmed
   mechanism (`lam` swept 1->40 at S/N=30/sigma=50: bias +21.5%->+90.3%).
   `wing_lam` (an explicit amplitude penalty active only in the far
   wings) fixes this in the regime it's tuned for (S/N=30/sigma=50:
   +26.96%->+2.26% bias, *and* shape residual 8.47%->4.04% in the same
   test) but **does not generalize across sigma/grid-width without
   retuning** (the same value measurably *hurts* shape residual at a
   different sigma on the same S/N: 4.56%->7.95% at sigma=140). Three
   well-tested auto-calibration approaches for `wing_lam` were tried and
   **all failed** -- reference-model wing-mass matching, bootstrap
   bias-correction of the point estimate, and a lam-sensitivity
   diagnostic. Each failure mode is specific and instructive; see
   `losvd.py`'s own module docstring and `FitConfig.wing_lam`'s help
   text for the full mechanism of each failure before re-attempting.
   **This mechanism (wing amplitude, far from the peak) is a plausible
   *partial* contributor to this document's shape-residual finding, but
   is very unlikely to be the whole story** -- the worst residual in the
   concrete example above is *at the peak* (v=24.7, near true center),
   not in the far wings, which is a different location than `wing_lam`'s
   own targeted failure mode.

3. **Post-hoc regularization-bias correction** (`bias_correction.py`,
   Pinkney-style, calibrated via `calibrate_bias.py`): corrects
   *reported moments* after the fact, does not touch the recovered
   `b(v)` curve at all. **Two distinct overcorrection failure modes
   found**, now off by default (`apply_bias_correction=False`). Not
   relevant to a shape-residual fix by construction (it never touches
   the curve), but relevant context: this project has already found that
   moment-level post-hoc correction is fragile and easy to overcorrect
   with, which is a reason to be cautious about any *moment-level* fix
   proposed for this document's problem too -- the shape itself needs to
   be right, not just corrected on paper afterward.

4. **`spectral_prior.py`, this session (Stages 19-21 in
   `VALIDATION_CAMPAIGN.md`)**: generalizes the curvature penalty's
   implicit per-mode weight (`lam * mu_i^1`, writing the penalty in the
   eigenbasis of `D2'D2`) into `alpha * mu_i^beta`. Found and fixed a
   real, large (>30x) scale confound (`mu.max()` itself varies hugely
   with sigma because of how the knot-density floor interacts with the
   `dv_ref/dv` grid-width correction) via eigenvalue normalization.
   **Properly validated at scale** (20 seeds, full sigma x S/N grid, 4
   template recipes): `alpha=100, beta=1.5, normalize_eigenvalues=True`
   is a real net win on **sigma-moment bias** for a single template
   (5.7% vs. 6.6% median), roughly neutral for template mixtures, and
   still shows a real (if much smaller than before normalization)
   high-sigma degradation. **This document's new finding is that even
   at this best validated setting, the *shape* residual is essentially
   unchanged from the default** -- i.e. `spectral_prior` addresses a
   real, different, sigma-dependent moment-bias mechanism, and should
   not be assumed to address the shape-residual problem this document is
   about.

   Two methodological cautions worth inheriting, found the hard way in
   this same session: (a) a **quick, small-seed-count scan can and did
   produce a misleadingly good result** that did not hold up under a
   properly-powered re-test -- do not trust a scan with fewer than
   ~15-20 seeds per point in this problem, seed-to-seed variance is
   large; (b) a seed formula using Python's built-in `hash()` is
   **randomized per-process** and is not reproducible across separate
   script invocations -- use a deterministic hash (`zlib.crc32` or
   similar) for any cross-run "identical seed" comparison.

### In the parallel `kinextract_test_suite` project

`/Users/waterstk/Documents/Gultekin_Astrophysics/kinextract_test_suite/`
is a separate, independent (no shared code) reimplementation built
specifically to explore regularization choices from first principles,
reframing the question as "what statistical prior are we imposing, in
what basis, and is it the right one" rather than "how do we smooth."
This is **directly relevant prior art** -- read
`kinextract_test_suite/notebooks/ladder_iteration_02/`'s notebooks and
`kinextract_test_suite/lib/` before proposing a new regularization
scheme, since several plausible-sounding ideas were already tried there
in detail:

- **Tikhonov/GCV** and **TSVD** (truncated SVD, a "spectral filter" in
  Hansen's sense): both showed real wing artifacts on real data, checked
  directly to be genuine (not plotting bugs) via a synthetic ground-truth
  test -- a clean Gaussian recovered with real template/noise conditions
  still showed 8.5-20.2% (Tikhonov) / 10.8-16.2% (TSVD) of total `|b|`
  beyond `|v|=140` where truth has exactly zero power. **But truncating
  those wings and refitting only closed 14-30% of the sigma gap** -- most
  of the sigma inflation was *not* in the visible far wings, but in a
  subtler, unisolated distortion *within the core* (`|v|<=140`) itself.
  **This is likely the same phenomenon as this document's peak-region
  residual finding**, seen independently in a differently-built system --
  worth treating as a strong hint that the core-region distortion (not
  the far-wing pedestal) is the real target, and that it was *already*
  flagged as "still unisolated, unconfirmed mechanism" in that project
  too (leading hypothesis there: correlated bin-bin shrinkage flattening
  the peak -- never confirmed).
- **Non-negativity was found to be a real, separate, solved sub-problem**
  (NNLS-augmented Tikhonov) but exposed a *reliability* regression
  (~2.7x worse seed-to-seed sigma variance than plain Tikhonov) traced to
  standard GCV being invalid once the active set depends on the data --
  not fixed, a real open problem in that project too if non-negativity
  ever needs revisiting here.
- **Final adopted method there**: a low-dimensional, smoothly-positive
  **log-density spline** (`lib/log_spline_solve.py`) -- structurally very
  similar to `kinextract_lite`'s own `b=exp(basis@coeffs)` representation,
  arrived at independently. Full 5x5 (V, sigma) grid validation there
  found mean |sigma bias| 11.3 km/s, mean sigma std 11.3 km/s across that
  grid (their own scale/regime, not directly comparable in absolute
  units to this document's percentages, but the same order of
  "not solved" as what's found here).
- **A directly relevant methodological lesson from that project's own
  Bayesian-HalfNormal comparison**: aggregate bias/variance numbers
  looked only moderately worse than the adopted method, but plotting
  recovered-vs-true curves directly (not just summary statistics)
  revealed an "obvious, visually unmistakable defect in every single
  case" (a non-zero floor in the wings from each bin's independent prior
  mean) that the aggregate numbers alone did not surface. **This is the
  exact same lesson this document's own finding demonstrates** in
  `kinextract_lite`: check the shape directly, don't trust moment-level
  aggregates alone. If proposing any new method, plot recovered-vs-true
  curves for several representative cases before trusting aggregate
  statistics.

## Hypotheses for the shape-residual mechanism: status

- **Wing-amplitude inflation** (documented mechanism, `wing_lam`'s
  target): plausible partial contributor, but the worst residual in the
  concrete example is *at the peak*, not the far wings -- likely not the
  primary mechanism for the peak-region defect specifically. Untested:
  whether `wing_lam` (tuned for this exact case) improves the *peak*
  residual as a side effect, not just the sigma moment.
- **Curvature penalty pulling the recovered log-shape away from a true
  parabola** (a Gaussian's `log(b)` is an exact downward parabola with
  non-zero, roughly-constant curvature; the P-spline penalty has some
  real, non-zero cost even for a perfect parabola, since only
  constant+linear sequences are in `D2`'s null space): plausible
  mechanism for *some* systematic distortion, but the *direction*
  observed (recovered peak is **taller and narrower**, i.e. *more*
  curved, not less) is the opposite of the naive expectation from "more
  smoothing flattens curvature" -- not yet reconciled, worth investigating
  directly (e.g., does the recovered coefficient sequence's own curvature,
  measured directly via `D2 @ coeffs`, actually exceed the true
  sequence's, and if so why).
- **Multi-start optimizer landing in a real but different local basin**:
  a documented ~1-in-150 known risk per fit (`losvd.py`'s own docstring),
  but the defect in this document appears far more often than 1-in-150
  in the seed sweep above (multiple seeds out of ~15 total checks showed
  large residuals) -- either this specific defect is *not* the rare
  local-minimum failure mode, or that failure mode is much more common
  than 1-in-150 for this specific case (single template, sigma=50,
  moderate S/N) and the 1-in-150 figure doesn't generalize here. Worth
  checking directly: for a high-residual seed, does `objective_gap`
  (returned by `fit_log_spline_map`) show a close second-best start, or
  is the winning start clearly, robustly best?
- **Representation/knot-density limitation**: ruled out as the primary
  cause -- the noiseless case achieves 3.8% residual (under target) at
  the same knot count `aic_local` uses at this sigma, so the basis is
  not the bottleneck. (Not ruled out as a *secondary* contributor at
  other sigma/knot-count combinations -- only checked at this one case.)
- **Something in how noise interacts with the log-link specifically**
  (as opposed to a hypothetical linear-space spline): not tested at all
  this session. A clean way to test: fit the same noisy data with a
  *linear*-space (non-exponentiated) spline plus a non-negativity
  constraint (e.g. NNLS-augmented, as `kinextract_test_suite` already
  built and characterized -- see above) and compare shape residual
  directly. If the log-link is itself contributing to the peak
  over-sharpening, a linear-space fit at comparable effective smoothing
  should show a different residual pattern.

## Recommended next steps

1. **Before trying a new regularization scheme, isolate the mechanism
   directly** rather than another parameter sweep. Suggested diagnostic
   order:
   - Check `objective_gap` distributions across the seed sweep above --
     does a large shape residual correlate with a small gap (suggesting
     a real, close competing basin) or a large gap (suggesting the
     "winning" solution is robustly preferred despite being wrong)?
   - Directly compare `D2 @ coeffs_recovered` vs. `D2 @ coeffs_true`
     (projected onto the same basis) to see whether the recovered
     coefficient sequence is *more* or *less* curved than the truth --
     this directly tests the "curvature penalty pulls toward less
     curvature" hypothesis and would explain the observed
     taller-and-narrower-than-true direction if the finding is
     counter-intuitive (i.e., if recovered curvature is *higher*, the
     naive "smoothing reduces curvature" story is wrong and something
     else, likely noise-driven overfitting in the *core* specifically,
     is happening).
   - Repeat the noiseless-vs-noisy comparison at several different
     sigma/S-N combinations (not just sigma=50/S-N=60) to map out where
     the defect is worst, using the same core-restricted max-residual
     metric already established here.
2. **Cross-check against `kinextract_test_suite`'s own already-built
   diagnostic notebooks** (`ladder_iteration_02`) rather than rebuilding
   the same infrastructure -- that project already has an SVD/eigenmode
   decomposition of the forward design matrix (a different, complementary
   decomposition to `spectral_prior`'s curvature-operator eigenbasis)
   that was used successfully to diagnose the analogous core-region
   distortion there, even though the mechanism was never fully confirmed.
3. **When reporting any future result, always show a recovered-vs-true
   curve plot for at least 3-4 representative cases**, not just
   aggregate bias/variance tables -- both this session and the parallel
   `kinextract_test_suite` project independently learned this lesson the
   hard way (moment-level and even std-level aggregates can look
   deceptively fine while the shape is visibly wrong).
4. **Treat the 5%-everywhere target as regime-dependent until proven
   otherwise.** The noiseless check shows the representation itself can
   hit this bar; the S/N sweep shows realistic noise regimes currently
   cannot, consistently, at S/N below roughly 200 for even the easiest
   (single-template) case. Whether 5%-everywhere is achievable at, say,
   S/N=30 at all (with any regularization scheme) is an open question,
   not a given -- the noiseless result bounds the *best case*, not the
   achievable case at realistic S/N.

## Appendix: reproducible code for the core finding

```python
import numpy as np
from kinextract_lite import FitConfig, build_template_stack, gauss_hermite_model, load_emiles_templates
from kinextract_lite.multi_template import fit_losvd
from kinextract_lite.spectral_prior import fit_losvd_with_spectral_prior

EMILES_GRID = "src/kinextract_lite/template_data/emiles_muse_grid.npz"
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
NSIG = 7.0
TRUE_V, SIGMA, SNR = 20.0, 50.0, 60.0

templates, template_wave, names, ages, metals = load_emiles_templates(EMILES_GRID, select=[(5.0, 0.06)])
mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
wave_fit = template_wave[mask]
cfg = FitConfig(vmin=TRUE_V - NSIG * SIGMA, vmax=TRUE_V + NSIG * SIGMA)
v_grid = cfg.build_velocity_grid()
b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, TRUE_V, SIGMA, 0.0, 0.0), 0.0)
b_truth /= b_truth.sum()
A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
galaxy_noiseless = A_stack[0] @ b_truth

# --- noisy case (S/N=60, seed=0) ---
noise_level = np.median(galaxy_noiseless) / SNR
rng = np.random.default_rng(0)
galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
errors = np.full(len(wave_fit), noise_level)
A_stack_w = A_stack / errors[None, :, None]
y_w = galaxy / errors

result_base = fit_losvd(A_stack_w, y_w, cfg)
result_spec = fit_losvd_with_spectral_prior(A_stack_w, y_w, cfg, alpha=100.0, beta=1.5, normalize_eigenvalues=True)

b_truth_n = b_truth / np.trapezoid(np.maximum(b_truth, 0.0), v_grid)
peak = b_truth_n.max()
core = b_truth_n > 0.01 * peak

for name, result in [("default", result_base), ("spectral", result_spec)]:
    b_fit_n = result["b"] / np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    resid = np.abs(b_fit_n - b_truth_n) / peak * 100
    print(f"{name}: max resid (core) = {resid[core].max():.1f}%  frac(core)>5% = {100*np.mean(resid[core] > 5):.1f}%")

# --- noiseless case (isolates representation from noise) ---
errors_noiseless = np.full(len(wave_fit), np.median(galaxy_noiseless) * 1e-6)
A_stack_w_noiseless = A_stack / errors_noiseless[None, :, None]
y_w_noiseless = galaxy_noiseless / errors_noiseless
result_noiseless = fit_losvd(A_stack_w_noiseless, y_w_noiseless, cfg)
b_fit_noiseless_n = result_noiseless["b"] / np.trapezoid(np.maximum(result_noiseless["b"], 0.0), v_grid)
resid_noiseless = np.abs(b_fit_noiseless_n - b_truth_n) / peak * 100
print(f"noiseless: max resid (core) = {resid_noiseless[core].max():.1f}%  "
      f"frac(core)>5% = {100*np.mean(resid_noiseless[core] > 5):.1f}%")
```

Run from the repo root (`kinextract_lite/`) with the package installed
(`pip install -e .` or equivalent).
