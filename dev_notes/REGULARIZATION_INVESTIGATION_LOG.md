# Regularization shape-residual investigation — running log

Continuation of `REGULARIZATION_SHAPE_RESIDUAL_HANDOFF.md` (read that
first — it defines the problem, the 5%-everywhere shape target, and
everything already ruled out). This file is the chronological record;
the handoff doc stays the current-state summary and gets updated when a
real finding lands.

Convention: dated entries, appended as work happens (not
retrospectively). Scripts live in `dev_notes/shape_residual/`, writing
incrementally to CSV per the `dev_notes/stress_test/` convention.

---

## 2026-07-26 — session start (Fable 5)

Read the handoff report, `losvd.py`, `config.py`, `aic_regularization.py`
in full. Plan, in order:

1. **Reproduce the core finding** (appendix code: single E-MILES
   template, V=20, sigma=50, S/N=60, seed=0 → ~10.9% peak-region
   residual with the default adaptive-lam fit) and save
   recovered-vs-true curve plots, per the project's own "look at the
   curve first" rule.
2. **Oracle-lambda diagnostic** (my main addition to the handoff's
   suggested order): for each seed, fit at *every* fixed `lam` in
   `cfg.lam_grid` (plus a finer/wider grid) and record the shape
   residual at each, alongside what `aic_local` actually picked. This
   splits the problem cleanly in two:
   - If **no** fixed lam achieves <5% residual on a bad seed, the defect
     is in the penalty *form* (or the noise itself), and no selection
     rule can fix it.
   - If an oracle lam **does** achieve <5% but AIC picks a different
     one, the defect is in the *selection criterion*, and the fix is a
     better selection target.
3. Mechanism diagnostics from the handoff's list: `objective_gap`
   correlation with residual, `D2 @ coeffs` recovered-vs-truth curvature
   comparison, residual anatomy (where exactly the error lives:
   peak/shoulders/wings, signed not just absolute).
4. Whatever fix the mechanism points to, validated at proper power
   (>=15-20 seeds per point, deterministic zlib.crc32 seeding).

## 2026-07-26 — core finding reproduced; first mechanism clue

`dev_notes/shape_residual/00_reproduce_core_finding.py`, matches the
handoff numbers exactly (default adaptive: 10.9% max core residual,
32.3% of core over 5%; noiseless: 3.8%). **New observation the handoff
did not record**: on this exact case,

- the adaptive `aic_local` selection picked **lam=0.5 — the smallest
  value in the default `lam_grid`** — for the noisy S/N=60 data;
- a plain **fixed lam=2.0 fit on the same data gives 6.5%** max core
  residual (vs 10.9% for adaptive);
- the noiseless fit's adaptive selection picks lam=8.0.

So at realistic noise the AIC-local criterion is choosing *much less*
smoothing than either the fixed default or the noiseless-selected
value, and the resulting defect direction (peak taller/narrower = more
curvature than truth) is exactly what under-smoothing/noise-overfitting
produces. This reframes the handoff's unresolved "wrong direction"
puzzle in the curvature-penalty hypothesis: the recovered curve is
*more* curved than truth because lam was selected too low, not because
the penalty flattens it. Hypothesis now: **the shape-residual defect is
primarily a lam-*selection* failure (AIC-local systematically
under-smooths at realistic S/N), not a penalty-form failure.**
Next: oracle-lambda diagnostic to test this properly across seeds
(does some fixed lam reach <5%? does AIC systematically pick lower
than the shape-oracle?).

## 2026-07-26 — oracle-lambda sweep: the penalty FORM is the floor, not just selection

`dev_notes/shape_residual/01_oracle_lambda_sweep.py` (+`analyze_01.py`):
15 seeds x S/N {30,60,100,200} x lam {0.25..64}, single template
sigma=50. Per seed, compared residual at the AIC-local pick vs. the
truth-informed oracle lam (min max-core-residual over the whole grid):

| S/N | mean res @ AIC pick | mean res @ fixed lam=2 | mean res @ ORACLE lam | seeds <5%: AIC / lam2 / oracle |
|---|---|---|---|---|
| 30  | 11.5% | 11.4% | 9.2% | 0 / 1 / **2** of 15 |
| 60  | 8.4%  | 6.6%  | 5.8% | 2 / 3 / **5** of 15 |
| 100 | 6.6%  | 5.8%  | 4.5% | 4 / 5 / **10** of 15 |
| 200 | 4.7%  | 4.8%  | 3.3% | 8 / 7 / **15** of 15 |

Two conclusions, both firm:

1. **Selection is genuinely bad** — `aic_local` picks the bottom of the
   grid at low S/N (lam=0.5 in 13/15 seeds at S/N=30) and costs ~2-3
   points of residual vs. oracle. Offline re-scoring of the same fits
   shows no simple reweighting fixes it: w_m=4/8, GCV ≈ same;
   discrepancy-principle rules (chi2_min + sqrt(2N), Morozov) pick
   lam=8-16 and are FAR worse at S/N=30 (mean 27%!) because heavy D2
   smoothing inflates the wings (the documented log-linearity
   mechanism) — confirming lam is squeezed from both sides.
2. **Even the oracle lam cannot hit the 5% target at realistic S/N.**
   No global lam under the current D2-on-log-b penalty gets more than
   2/15 seeds under 5% at S/N=30. The defect is structural: the core
   needs heavy smoothing at low S/N, but the D2 penalty's null space is
   *linear* log-b, so heavy smoothing simultaneously drives the wings
   toward exponential tails (wing-mass inflation). One knob cannot
   serve both regions.

**Hypothesis for the fix (next test): change the penalty null space.**
A third-difference penalty (D3 on the log-spline coefficients) has
*quadratic* log-b in its null space — i.e. any Gaussian (any V, sigma,
amplitude) is penalty-free. Heavy smoothing then shrinks toward the
best-fitting Gaussian instead of toward exponential tails: wing
inflation mechanism eliminated by construction, and lam can be cranked
up at low S/N without flattening the core. For non-Gaussian truth it
shrinks h3/h4 toward 0 (a physical, Gauss-Hermite-like prior) instead
of toward log-linear tails (an unphysical one). Test: rerun the oracle
sweep with `difference_matrix(order=3)`.

Also noted: the `curv_fit/curv_truth` ratio recorded in the sweep is
contaminated by the truth-projection's wing clipping (log of clipped
zeros dominates the D2 norm) — not usable as recorded; don't reuse
without restricting to core knots.

## 2026-07-26 — D3 penalty test: hypothesis CONFIRMED on Gaussian truth

`dev_notes/shape_residual/02_penalty_order_test.py` (+`analyze_02.py`):
same 15 seeds x 4 S/N, same noise draws (paired), penalty swapped to
`difference_matrix(order=3)` (null space = quadratic log-b = any
Gaussian), lam grid extended to 16384 since D3 is weaker per unit lam.

Oracle-lam comparison, paired per seed (max core residual):

| S/N | D2 oracle mean | D3 oracle mean | D3 seeds <5% (D2) | D3 wing ratio @ heavy lam |
|---|---|---|---|---|
| 30  | 9.15% | 7.95% | 6/15 (2/15)  | 0.98 |
| 60  | 5.78% | 4.39% | 11/15 (5/15) | 0.97 |
| 100 | 4.54% | 2.67% | 14/15 (10/15)| 1.05 |
| 200 | 3.31% | 1.39% | 15/15 (15/15)| 1.03 |

Confirmed mechanisms:
- **Wing-mass inflation is gone by construction**: ratio ~1.0 even at
  lam=16384 (D2 hit 45x at lam=64, S/N=30). Heavy D3 smoothing shrinks
  toward the best-fitting Gaussian, not exponential tails.
- **Sigma moment bias collapses too**: mean sigma at D3 oracle is
  49.4-49.8 across all S/N (truth 50; the residual -0.5 km/s is
  consistent with the known -1.2% deterministic windowing bias of the
  moment estimator). Under D2 the median sigma at fixed lam ranged
  51-59 at S/N=30 depending on lam.
- At the heaviest lam (pure shrink-to-Gaussian limit), median residual
  at S/N=30 is ~9% -- i.e. much of the remaining S/N=30 error is the
  *statistical* error of estimating the 3 Gaussian parameters from data
  that noisy, which no regularizer can remove. The 5%-everywhere target
  at S/N=30 per-realization may be near the information limit (the
  handoff already flagged this possibility).
- **Selection is still the weak link at low S/N**: AIC-local (chi2 +
  2*m_eff computed with the D3 operator) lands mean 11.7% at S/N=30 vs
  oracle 7.95% (fine at S/N>=100). Next: offline rule comparison on the
  D3 sweep, then a non-Gaussian-truth test (the shrink-to-Gaussian
  prior must not erase real h3/h4 at the lam the rule picks) and other
  sigma regimes before any promotion.

## 2026-07-26 — non-Gaussian truth (03) + two structural discoveries

`dev_notes/shape_residual/03_nongaussian_d3.py` (+`analyze_03.py`):
truth (h3,h4) in {(0.10,0), (0,0.10), (0.06,-0.06)}, S/N {30,60,100},
15 seeds, both penalty orders, paired. Headlines:

- **No rule and no oracle reaches 5% on non-Gaussian truth** at
  realistic S/N (oracles sit at 8-14%). The stiff rules (1-SE,
  discrepancy) slam to the Gaussian limit under D3 and erase h4
  entirely (bias -0.1035 on an h4=+0.1 truth) — ruled out as defaults.
  `aic_w2` (the existing criterion) adapts sensibly under D3 (backs
  off to lam 4-64 at S/N=100 when real structure is present).
- D3 vs D2 at their own `aic_w2` picks on non-Gaussian truth: shape
  residual roughly ties (D3 better on (0.06,-0.06), slightly worse on
  (0,0.10)); sigma bias (vs the truth curve's own taper-moment sigma,
  NOT the GH parameter 50 — h4=+0.1 truth has taper-sigma 59.6!) is
  better under D3 (e.g. +0.2 vs +1.7 km/s at S/N=60, h4=+0.1). But D3
  attenuates h4 more than D2 at its picks (-0.087 vs -0.05 of a 0.104
  truth at S/N=60). Note D2 *also* erases half the h4 — neither is
  good at preserving h4 at these S/N.

**Structural discovery 1 — the "noiseless" benchmark convention is
corrupted by float32.** Fitting with errors = 1e-6 x signal (the
handoff appendix's noiseless convention) puts whitened design values
at ~1e6, where JAX's default float32 objective loses the optimizer:
the SAME noiseless h4=+0.1 data gives 40% residual at 1e-6 whitening
vs **1.70%** at 1e-4 whitening (chi2 = 7.9e6 vs 35 — the 1e-6 fit
never actually fit the data). All prior "noiseless floor" numbers
(including the handoff's 3.8%) are conservative upper bounds; the true
representation floor at sigma=50 is <2%. Use S/N ~ 1e4 as the
noiseless proxy from now on.

**Structural discovery 2 — clipped-GH truth shapes have an
ill-posedness floor no regularizer can cross.** h3=+0.1 truth (GH
clipped at zero) fits the *spectrum* essentially perfectly (chi2=161
over 444 px at 1e-4 whitening) while being 7.8% wrong in LOSVD space:
the exp-link cannot produce the clipped exact-zero region, and the
spectral signature of that difference is ~nothing. This portion of the
benchmark error is a property of the (artificial) clipped-GH truth
family near its own invalid-density boundary, not of the
regularization.

Next: test a **GH-anchored D3 penalty** (penalize `||D3(c - c_anchor)||`
with the anchor from a pilot fit's own GH moments) — it should keep
D3's Gaussian-truth wins while protecting real h3/h4 structure; then
the sigma-portability sweep (04), then full-scale validation.

## 2026-07-26 — GH-anchored D3: tried and REJECTED (negative result, documented in full)

`dev_notes/shape_residual/05_gh_anchored_d3.py` implements a
semi-parametric scheme: pilot 5-param GH fit to the spectrum ->
project its log-density onto the spline basis -> penalize
`||D3(c) - D3(c_anchor)||^2` (lam->inf recovers the parametric fit,
preserving pilot h3/h4). Two failure modes found, in order:

1. **Technical trap (fixable, worth remembering):** an *unweighted*
   least-squares projection of `log(max(b_anchor, 1e-6*peak))` onto the
   spline basis is dominated by the deep-wing floor plateau and can be
   catastrophically wrong in the core (anchor curve 13.9% residual, its
   projection 26.5% — and the anchored fit faithfully tracks the bad
   projection). Any future anchor/starting-point projection must weight
   by density (or restrict to where the anchor is meaningfully nonzero).

2. **Fundamental blocker (kills the approach at low sigma):** the pilot
   parametric GH fit itself, at sigma=50 / S/N=60 / single E-MILES
   template, has seed-to-seed scatter **sigma(h3)=0.15, sigma(h4)=0.19**
   (15 seeds, several hitting the +-0.3 fit bounds), i.e. the anchor's
   h3/h4 are noise. Anchoring to it just injects that noise as bias.
   Root reason: with sigma_inst ~ 30-35 km/s (E-MILES/MUSE around the
   Ca triplet) and sigma=50, h3/h4-scale structure is barely encoded in
   the spectrum at S/N<=60-100 — **an information limit of the data,
   not a regularization defect**. This also explains why *no oracle
   lam under any penalty* reaches 5% shape residual on non-Gaussian
   truth at sigma=50, S/N<=100 (03's result): the structure
   distinguishing those shapes from a Gaussian (~10% of peak at
   h4=0.1) is below the estimation noise floor for ANY method,
   parametric included. The 5%-everywhere target is genuinely
   infeasible in that corner; the practical goal is to be *at* the
   information limit, which plain D3 + AIC already approaches.

Decision: drop the anchored scheme; proceed with plain D3 + aic_local
(w_m=2, extended lam grid) as the candidate default. Remaining before
promotion: sigma-range portability / dv-scaling exponent (04), a check
that D3 doesn't erase *well-measurable* h3/h4 at large sigma (where
the information IS there), multi-template validation at scale.

## 2026-07-27 — final green test suite; investigation closed out

Full suite, run clean (no CPU contention with the notebook re-executions,
which were stopped mid-run at the user's request so they could run and
verify the notebooks themselves): **179 passed, 0 failed**, 81 minutes.
This is the complete validation of the default flip: `penalty_order=3`,
JAX float64 at import, the delta-AIC parsimony tiebreak, and the
sub-knot-sigma candidate guard, plus the 7 pre-existing tests updated to
pin `penalty_order=2` (or use `pytest.approx`/a rescanned seed) where
they specifically characterized order-2-only or precision-sensitive
behavior — all documented per-test in their own comments, not silently
patched.

Everything from root cause through shipped fix, including every
negative result and the general bias review, is recorded above and in
`VALIDATION_CAMPAIGN.md` Stage 22 / the handoff doc's status section.
Remaining open items for a future session, none blocking: (1) recalibrate
`bias_correction`'s tables for `penalty_order=3` (see the entry below —
currently actively harmful there, documented and guarded against, not
fixed); (2) the `shift_template` linear-interpolation broadening found
in the general bias review; (3) modernizing `dev_notes/scratch_notebooks/
bias_sweep_fixed_vs_adaptive.py` for the new default (currently kept as
a historical order-2-era record, per the sandbox notebook's own note).

## 2026-07-27 — new finding while updating notebooks: bias_correction is invalid under penalty_order=3

While refreshing `examples/notebooks/01_basic_mock_fit.ipynb` for the new
default, re-ran its bias-correction demo under `penalty_order=3`
(V=20/sigma=50/S-N=60, seed=42, same mock as the handoff's own worked
example). Result: raw point estimate is now excellent (sigma=50.9 km/s,
vs. the old default's 52.9) -- Section 9's fix already does the work the
old Section 10 leaned on a correction for. But `bootstrap_losvd(...,
apply_bias_correction=True)` on this same, already-good estimate pulls
sigma down to **~22 km/s** -- a severe overcorrection, worse than the
already-documented order-2 overcorrection case in the same notebook.
Cause: both `bias_correction` calibration tables (plain and the
`m_eff`-axis one `bootstrap_losvd` auto-switches to) were built against
order-2's own bias mechanism and `m_eff` scale; order-3's adaptively
selected `lam` sits far outside either table's calibrated regime.
**`apply_bias_correction=True` must not be used with the new
`penalty_order=3` default until these tables are recalibrated for it** --
documented in `uncertainty.py`'s own docstring, the notebook, and here.
Not yet started as a task; flagging for a future session.

## 2026-07-26 — handoff's local-basin hypothesis: ruled out

Using 02's per-fit `losvd_objective_gap` (D3 sweep, lam=1.0): no
consistent correlation with shape residual across S/N (r = -0.21,
+0.16, +0.65, +0.25 at S/N 30/60/100/200, n=15 each), and gaps are
tiny (median 0.1-1.2 against objective ~220). The high-residual fits
are not multi-start/basin accidents; every start agrees. (Note for
future scripts: at the `multi_template.fit_losvd` level the per-fit
diagnostic is `losvd_objective_gap`; `objective_gap` there is the
w0-pool gap and is None for a single weight start — 01's CSV recorded
the wrong key, 02's the right one.)

## 2026-07-26 — sigma-range portability (04): scaling exponent moot; two hard corners mapped

`dev_notes/shape_residual/04_sigma_range_d3.py` (+`analyze_04.py`):
Gaussian truth, sigma {20,50,140,300} x S/N {30,100}, 15 seeds, D3 with
knot-spacing scaling exponent 1 vs 3.

- **Scaling exponent: no winner, question moot in practice.** Neither
  exponent makes the oracle lam stationary across sigma (the
  `min_interior_knots` floor binding below sigma~90 breaks any single
  power law), and AIC-selected results differ only marginally between
  exponents. Keeping exponent 1 (the existing `difference_matrix`
  convention); per-fit adaptive selection over a wide grid does the
  real work.
- **sigma=20 is information-limited, like sigma=50-non-Gaussian:**
  m_eff = 3.000 at every lam (deconvolving sigma=20 through
  sigma_inst~35 km/s templates supports ~3 effective dof, period). The
  AIC criterion is flat to <1 unit while residuals across the same lam
  grid range 21-75% (S/N=30) — selection there is a lottery, and the
  median residual is >20% at EVERY lam, so no selection rule can fix
  sigma=20/S N=30. At S/N=100 best-case is ~6.5%.
- **sigma=300/S N=30 has a REAL residual wing problem even under D3**
  (wing-mass ratio 3-5 at the oracle lam, sigma_mom bias +40 km/s
  median at moderate lam), and the heavy-lam (Gaussian) limit
  *misbehaves* there (sigma_mom -> 400, chi2 rising — suspected
  optimizer stiffness at n_coef=48 with huge lam, since the D3
  Gaussian limit cannot legitimately prefer sigma=400). Not yet
  root-caused; flagged for the paired 06 run (does the current D2
  default do any better there? Stage 19's own numbers suggest not).

Decision going into 06 (final paired validation, running): candidate =
D3, exponent 1, aic_local w_m=2 over lam grid (1..32768), vs the
literal current default (`fit_losvd`, D2 + aic_local). sigma
{20,50,140,300} x S/N {30,100} x {Gaussian,(0.06,0.06)} x {single,
mix3} x 15 seeds, paired draws, zlib.crc32 seeding.

## 2026-07-26 — sigma=300 heavy-lam anomaly root-caused: TWO stacked causes, one decisive

Chased the 04 anomaly (D3 heavy-lam limit drifting to sigma~400+ at
sigma=300/S N=30, impossible for a shrink-to-Gaussian penalty):

1. **Boundary null-space leakage (real, but secondary/fixable):** the
   plain third-difference operator on the coefficient index is NOT
   zero for a true Gaussian at the clamped boundary (|D3c| rows 4.17
   at sigma=50, 0.72 at sigma=300 — interior rows are exactly 0). The
   coefficient sequence of a quadratic-in-v is its polar form
   (blossom) on knot triples, which is quadratic in *index* only where
   knots are uniform. Built a corrected operator (`gaussian_null_D3`
   in the scratch/log work: per-4-window null vectors of the blossom
   sequences of {1, v, v^2}, scaled to match the interior Delta^3
   convention) — achieves ||D3c||^2 ~ 1e-24 on a true Gaussian,
   boundary included. NOTE: a Greville-abscissae divided-difference D3
   is NOT sufficient (coefficients are blossoms, not function values
   at Greville points — tested, still leaks).
2. **The decisive cause: JAX float32 L-BFGS-B stalls in the stiff
   heavy-lam regime at n_coef=48.** With the exact-null-space operator
   the heavy-lam objective provably has a penalty-free Gaussian at
   chi2~445, yet the float32 fit converges "successfully" at chi2=772,
   sigma=1055 (lam=262144). Direct float32-vs-float64 comparison, same
   data/starts, plain D3, sigma=300/S N=30/seed 0:
   lam=16384: f32 obj=247.8/sigma=375 vs f64 obj=223.2/sigma=268;
   lam=65536: f32 obj=304/resid 35.3% vs f64 obj=225/resid **6.1%**.
   Float32 is 10-80 objective units above the true optimum — a
   systematic optimizer-precision failure, not basin luck. This is a
   substantial escalation of `losvd.py`'s documented "float32 can
   change which basin wins" caveat: at wide grids + heavy penalty it
   reliably fails to optimize at all.
   (Also tested and rejected: a "Gaussian start" from the ridge
   estimate's direct moments — the ridge estimate at sigma=300/S N=30
   is noise-dominated, V=72/sigma=1208, useless as an anchor.)

Consequence for the candidate: the D3 default must run with
`jax_enable_x64` (or the fits must be cast to float64) — D3 uses heavy
lam by design, exactly where float32 fails. The in-flight 06 run is
float32-vs-float32 (still a fair pairing, but its sigma=300 D3 cells
will understate the fixed method); plan: rerun affected cells at x64
and validate x64 as part of the package change.

## 2026-07-26 — 06 (float32) paired validation verdict

480 paired cases (sigma {20,50,140,300} x S/N {30,100} x
{Gaussian,(0.06,0.06)} x {single,mix3} x 15 seeds), full summary in
`dev_notes/shape_residual/06_summary.csv` (+`analyze_06.py` output).

- **Overall: D3 candidate wins 331/480 paired cases, better mean
  residual in 22/32 cells**; overall |sigma bias| 12.6 vs 15.0 km/s,
  |h4 bias| 0.020 vs 0.021.
- **Gaussian truth: decisive D3 win in 15/16 cells** — e.g. mix3
  sigma=140/S N=30: sigma bias 23.7 -> 10.0 km/s, wing ratio 5.7 ->
  1.3, mean residual 9.5% -> 6.1%; single sigma=140/S N=100: 3.7% ->
  1.6% with 15/15 under the 5% target.
- **Non-Gaussian (h3=h4=0.06) at sigma 50-140, S/N=100: D2 keeps a
  real edge in shape residual** (e.g. single sigma=50: 6.5% vs 8.1%)
  because D3's picks attenuate h4 by ~0.03-0.05 — the known
  shrink-to-Gaussian cost in the regime where h4 is near the
  information limit. At S/N=30 they roughly tie; at sigma=300 D3 wins
  even for this shape.
- **Both methods fail at sigma=300/S N=30** (sigma bias 44-91 km/s
  both arms) and D3 shows occasional catastrophic residuals at
  sigma=20 (max 119-160%) — both patterns match the float32 heavy-lam
  optimizer stall; 06b (identical grid, x64) is the test.
- Candidate runtime ~1.3-2x the default (heavier-lam fits iterate
  longer). Acceptable; a warm-start continuation across the lam grid
  is an obvious later optimization.

Package plumbing landed meanwhile (additive, defaults unchanged,
tests green): `FitConfig.penalty_order` (2 default | 3), `lam_grid`
default now None -> `resolved_lam_grid()` per order (order-2 grid
identical to the old default tuple), `fit_losvd` +
`estimate_m_eff_local` honor `penalty_order`; new
`tests/test_penalty_order.py`.

## 2026-07-26 — general bias review: findings beyond the regularization problem

Reviewed `forward_model.py`, `gh_moments.py`, `noise_model.py`,
`lsf.py`, `emiles.py`, `uncertainty.py` headers + measured one new
mechanism directly:

1. **NEW, measured: linear-interpolation shift smoothing in
   `shift_template` + benchmark circularity.** `np.interp` at a
   fractional-pixel Doppler shift acts as a triangular smoothing
   kernel: measured on the E-MILES grid (0.9 A sampling = 31.4 km/s at
   8600 A), a half-pixel shift removes 21% of the template's
   high-frequency power, equivalent to ~17 km/s of added Gaussian
   broadening — oscillating between 0 (integer-pixel shifts) and ~17
   km/s with a 31 km/s period across design-matrix columns (~13 km/s
   RMS-average). Two consequences: (a) **all synthetic benchmarks in
   this project generate the mock spectrum with the same design
   matrix (`A_stack @ b_truth`), so they inherit and thus HIDE this
   artifact entirely** — recovered-vs-true comparisons are internally
   consistent but the forward model differs from reality; (b) on real
   data the templates are effectively ~13 km/s broader (in quadrature)
   than the instrument alone, biasing recovered sigma LOW by roughly
   13^2/(2*sigma) (~1.7 km/s at sigma=50, less at larger sigma), with
   a v-periodic ripple. Recommended follow-ups (not done): generate
   benchmark spectra by convolving the template at native/finer
   sampling (breaking the circularity), and/or switch
   `shift_template` to band-limited (cubic/sinc) interpolation.
2. Float32 JAX optimizer stall (see earlier entry) — the largest
   newly-found non-regularization bias mechanism; promotes to a
   package-level x64 recommendation pending 06b.
3. Already-documented and confirmed still-accurate: -1.2% deterministic
   taper-windowing sigma bias in `compute_direct_moments` (geometry,
   not noise); Gaussian-LSF-shape assumption in `lsf.py` (h3/h4 risk
   up to +-0.1 per Simon 2026, needs measured template LSF shapes to
   fix); AR(1) `rho` measured at 0.3-0.5 on real MUSE residuals but
   whitening defaults to rho=0 — real-data fits should pass the
   measured rho (`noise_model.whiten`), else chi2/AIC/bootstraps are
   all mildly miscalibrated.
4. Reviewed and found clean at this level: `noise_model.py` (proper
   GLS), `forward_model.build_template_stack` structure, benchmark
   noise conventions, `emiles.py` pack-time LSF matching approach.

## 2026-07-26 — x64 partial results expose a SELECTION catastrophe at sigma=20; parsimony tiebreak adopted (order 3 only)

Partial 06b (x64) rows at sigma=20/S N=30: the D3-argmin candidate gets
WORSE under float64 (mean 65-143%, max 712% residual, recovered
sigma_mom 2-10 km/s = delta-spike collapse; picked lams 1-64). Float32's
stall had been accidentally masking it. Mechanism: with m_eff pinned at
~3, the AIC criterion is flat to <2 units, argmin degenerates to a
noise lottery, and a small-lam D3 fit leaves the sigma=20 deconvolution
effectively unregularized (D3 makes cubic wiggles cheap; float64
faithfully finds the resulting spike solution).

**Fix, tested offline against every recorded per-lam sweep (01-04)**:
Burnham & Anderson parsimony tiebreak — pick the LARGEST lam whose
criterion is within delta=2 of the minimum. Under D3 (where
oversmoothing is nearly free): sigma=20/30 mean 51->26%, max 167->48%;
sigma=300/30 max 45->15%; sigma=300/100 max 16->5.5%; Gaussian truth
elsewhere neutral-to-better; cost is a small extra h4 attenuation on
non-Gaussian truth at S/N 60-100 (e.g. -0.079 -> -0.092 of 0.104).
Under D2 the same rule measurably HURTS (heavier smoothing actively
wing-inflates): tested, mixed-to-worse — so the tiebreak is **gated on
penalty_order==3** (`select_lambda_aic_local(parsimony_delta=...)`,
default 0.0; `fit_losvd` passes 2.0 only for order 3).

Also: 06b's candidate arm (argmin, in-script) remains useful as the
x64-vs-float32 comparison, but the promoted configuration is now
D3 + x64 + tiebreak through the production `fit_losvd` path — being
validated end-to-end as `06c_candidate_production_x64.py` (chained to
start when 06b finishes), paired against 06b's `default_d2_aic_x64`
rows on identical draws. Full test suite after the plumbing: 171
passed. New selection-logic unit test added to
`tests/test_penalty_order.py`.

## 2026-07-27 — 06b (x64) complete: x64 is neutral for the D2 default

Full 480-case x64 rerun (`06b_final_validation_x64.csv`), compared
cell-by-cell against the float32 06 run (same seeds):

- **Current D2 default: float32 vs float64 is statistically neutral in
  every cell** (mean residual differences <1 point, sign varies).
  Historical float32 validations of the incumbent are therefore not
  corrupted, and x64 need not be forced for order-2 fits.
- **D3-argmin candidate**: x64 confirms the sigma=20/S N=30 selection
  catastrophe at full scale (mean 75%, max 712% — float32's stall had
  been masking it; the parsimony tiebreak is the fix, see previous
  entry) and mildly improves large-sigma cells (e.g. sigma=300/S N=100
  mean 4.2 -> 3.9, sigma bias 21 -> 16 km/s).
- sigma=300/S N=30 remains bad for BOTH methods (|sigma bias| ~60-68
  km/s) — dominated by wing noise at 44 knots; flagged as an open
  regime, not a D2-vs-D3 differentiator.

Verdict now rests on 06c (production D3 + x64 + tiebreak), running.

## 2026-07-27 — 06c production-path verdict + the last failure mode closed

`06c_candidate_production_x64.py` (production `fit_losvd`,
penalty_order=3, x64, parsimony tiebreak) vs 06b's x64 D2 default,
paired on identical draws (`06c_final_summary.csv`):

- **Overall: candidate better mean residual in 23/32 cells, 332/480
  paired wins; |sigma bias| 5.6 vs 15.5 km/s (3x better); wing ratios
  ~1 essentially everywhere** (D2: 3-28x at low S/N).
- Gaussian truth: dominant. sigma=300/S N=30: 13.3% -> 6.3% (mix3),
  sigma bias 63 -> 21 km/s; sigma=140/S N=100 single: 4.5% -> 1.4%
  with 15/15 under the 5% target (D2: 9/15).
- Known tradeoff confirmed at production scale: (h3,h4)=(0.06,0.06) at
  sigma 50 / S N=100 favors D2 (5.9% vs 9.5% mix3) via D3's h4
  attenuation (-0.04 vs -0.01). This is the info-limited regime; the
  cost is bounded and documented.
- **Residual failure mode found and closed**: on 4/60 sigma=20/S N=30
  draws the delta-spike solution beat the tiebreak window outright
  (its chi2 gain > 2; linearized m_eff underestimates a spike's true
  flexibility). Fix: candidates whose own fit's direct-moment sigma is
  below 0.5x the knot spacing (structure the basis cannot legitimately
  resolve) are excluded from selection
  (`select_lambda_aic_local(min_resolvable_sigma_frac=0.5)`, gated to
  order 3 with the tiebreak). Verified against all 960 production
  fits: flags exactly the 4 spikes (recovered sigma 2.4-4.4 km/s),
  zero healthy fits at any sigma/order. Unit-tested; sigma=20 cells
  re-running through the guarded production path (06d).

Curve plot (`dev_notes/shape_residual/00_curves.png`, from
`plot_00_curves.py`): the residual anatomy is a **sharpening pattern**
-- positive overshoot at the peak, negative lobes at the +-1.5-2 sigma
shoulders. Adaptive (lam=0.5) is the worst; fixed lam=2 shows the same
pattern at ~60% the amplitude; the noiseless fit (lam=8) shows only a
small, knot-scale oscillatory residual (~3.8% max, the representation
floor). Consistent with noise-overfitting, not oversmoothing.

## 2026-07-28 — SHASH core tested; the shape-residual investigation reaches its floor

Tried the sinh-arcsinh core as the successor to the divergent GH core. It solves
the divergence cleanly (SHASH reshapes a Gaussian's argument instead of adding a
polynomial to its exponent, so it cannot blow up), but it does **not** beat the
spline on NB02: best 12.77% max residual vs 7.73% (order 3) and 6.19% (order 2),
and it collapses entirely for lam >= 1000 via gradient underflow at small sigma.

Then ran the test that should have come first. Fitting the **4-parameter SHASH
core with no spline and no penalty at all** — a model with no null space for a
bad penalty to exploit — still gives 11.51% at S/N=60, while the same fit is
essentially exact when noiseless (1.77%, h4 = +0.050 vs truth +0.050). The S/N
ladder is monotonic: h4 = +0.050 (noiseless), +0.045 (S/N 1000), +0.028 (200),
-0.023 (60).

**This closes the shape-residual investigation.** The NB02 floor at S/N=60 is
photon noise plus near-collinear template degeneracy, not the roughness penalty's
null space. Every rejected attempt in this log — wing-mass penalty, one-sided
convexity, a2<=0, QAIC, unimodality, tapered wing penalty, GH core, SHASH core —
was aimed at a cause that is not the binding constraint. The existing spline at
6.19-7.73% already beats a maximally-constrained parametric fit at 11.5%.

Corollary for the stated goal ("true LOSVD within the error bars everywhere"):
that is achievable at S/N >= 200, and not achievable at S/N=60 for h4, because at
S/N=60 the data do not determine h4. The honest fix is to report h4 with an error
bar that covers zero at low S/N, not to keep redesigning the penalty.

Control note: the first S/N-ladder run had a broken fixed-weights control (unit-
integral vs unit-sum normalization, off by dv; NNLS masked it in the free-weight
rows). Corrected and re-run — control passes at 1.77%/h4=+0.049, so the ladder
stands. Details in dev_notes/ghcore/FINDINGS.md.

## 2026-07-28 (later) — real-data sigma inflation FIXED: edge-amplitude constraint

Chased the hypothesis that OG kinextract's linear-space D2 penalty was the
cure. **It is not.** Implemented faithfully (OG's implicit b=0 edge rows,
quartic wing taper, exact augmented-NNLS solve), it gives sigma = 88-97 km/s
on both real bins at every lam from 1 to 3.3e4, and is insensitive to the
taper seed, so the suspected self-fulfilling sigl0 fixed point does not exist
either. The reasoning was backwards: heavy D2 in linear space drives b toward
a linear ramp (broad); heavy smoothing in log space drives log b toward a
quadratic, i.e. a Gaussian (compact). Our parameterization is *better* in the
high-lam limit.

Also tested OG's selection rule (smallest lam that is unimodal and
non-jagged) on our log-space fitter: it picks **lam=1, identical to AIC**,
because the pedestal solutions are unimodal (1 peak) and smooth (roughness
0.09 vs its 0.25 threshold). A smooth divergent pedestal passes both tests.

What does separate the two classes, with no overlap: **LOSVD amplitude at
the velocity-grid edge** — 21.9%/42.4% at the failing lam, 0.0% at every good
one. Added as `FitConfig.max_edge_amplitude_frac` (default 0.01), applied as
a validity filter on the candidate grid inside `select_lambda_aic_local`,
with AIC still choosing among the survivors.

| | before | after | legacy |
|---|---|---|---|
| nuclear (high) | lam=1, sigma=84.5 | lam=512, **sigma=49.1** | 54.3 |
| mid-radius (med) | lam=1, sigma=100.7 | lam=32768, **sigma=42.4** | 40-45 |

**Portability, the property every earlier candidate lacked:** on the NB02
mock the constraint excludes nothing and results are *bit-identical* with it
on or off (S/N 60 and 200, two seeds each). The wing-mass penalty and
`lam_wing` both died because the strength that fixed real data wrecked mocks.
This cannot, because where there is no pedestal it does nothing.

Framed correctly it is not a tuning knob at all: it says the velocity grid
must contain the LOSVD, which is a precondition for the moment integrals to
mean anything — not a preference about smoothness.

Full detail, including both negative results and a harness bug (sweeping
`cfg.lam` without `lam_selection="fixed"` silently returns the same AIC fit
every time), in dev_notes/real_data_sigma/FINDINGS.md.

## 2026-07-28 (overnight) — bootstrap lam-pin fixed: a real, secondary contributor to under-coverage

Per the open-problems priority list, checked whether the previously-measured
too-narrow bootstrap band (~2x at sigma=50, ~4x at sigma=90) is partly caused
by `bootstrap_losvd` pinning lam at the point estimate's own selection for
every replicate. Confirmed directly by code read: it does, unconditionally,
by design (cost-saving, documented). Given lam selection is a real lottery at
S/N 60-150 (measured earlier: the same data can pick lam=8 or lam=32768 on
noise alone), this pins the whole bootstrap ensemble to one regularization
branch and cannot explore the alternative the real noise draw might have
produced.

Measured with 3 ensembles (pinned / reselected-isolated / true independent
sampling, n=30 each, two sigma regimes): reselecting lam per replicate closes
~60-80% of the gap between the pinned bootstrap's spread and the true
sampling distribution's own spread. Real, but the previously-documented
bias-blindness mechanism (bootstrap can't see an offset shared by every
replicate) is untouched by this and is still expected to be the larger
remaining contributor to the original 2x/4x under-coverage number.

Fix: `bootstrap_losvd(..., reselect_lam=True)` (new default) -- every
replicate now re-runs adaptive selection independently instead of being
pinned, at ~2-2.5x cost per replicate (affordable given the ~6.8x fitting
speed-up). Forced off automatically when continuum-cofitting is active
(adaptive selection doesn't support that combination yet and would just
re-warn n_boot times for no behavioral difference). Two downstream
`cfg_boot.lam` reads (bias-calibration table choice, m_eff evaluation) fixed
to use `result0["lam_used"]` instead, since `cfg_boot.lam` is no longer
guaranteed meaningful once reselection leaves it unset.

4 new tests added (including a monkeypatched structural check of the actual
`cfg` objects passed to every replicate's `fit_losvd` call -- not just a
statistical consequence). Every pre-existing bootstrap-calling test audited
and pinned `reselect_lam=False` where the mechanism itself wasn't what was
being tested (test_uncertainty.py, test_feature_composition.py), to keep
the suite's runtime unchanged. Full suite: 178 passed, 0 failed, 32:28 --
matches the pre-change 32:51 baseline.

Full writeup, including three self-corrected tooling mistakes along the way
(a confounded first comparison, an unguarded module-level import side
effect, and two undersized-timeout/buffered-output false "hangs"), in
dev_notes/uncertainty/FINDINGS.md.

Remaining on the error-bar problem: the bias-blindness mechanism itself is
still unfixed. Next candidate approach: a coverage-calibration study using
the randomized validation-suite mocks (dev_notes/validation_suite/) rather
than fixed test cases, to (a) confirm how much of the original 2x/4x gap
this fix actually closes in practice at scale, and (b) quantify what
residual inflation factor, if any, is still needed -- naturally folds into
the calibration-table regeneration (next item).

## 2026-07-28 (overnight, cont.) — shift_template interpolation bias quantified and fixed (opt-in)

Addressed the deferred item: `shift_template`'s linear interpolation
(`np.interp`) has a real, periodic-in-fractional-pixel-shift error --
quantified directly against an oversampled ground truth (no fitting
needed): max relative error swings 0.7-1.2% across one full ~31.7 km/s
period (matching the E-MILES native 0.9 Angstrom/pix sampling exactly),
confirming the earlier "0-17 km/s real-data effective broadening, period
~31 km/s" finding's mechanism precisely.

Fix: `shift_template`/`build_template_stack` gained `method="lanczos"`
(opt-in, default stays `"linear"`) -- a windowed-sinc resampling that
requires (and checks) a uniformly-sampled template grid. Verified correct
at exact grid alignment (reproduces source values to numerical precision)
and verified to remove the *periodic* component of the interpolation
error specifically (the part that can masquerade as shift-dependent
kinematic broadening) -- what remains is a smaller, phase-independent
residual, better explained by real spectra not being perfectly
band-limited than by any remaining periodic defect. Costs ~67x more than
linear in isolation, but `build_template_stack` runs once per fit setup
(not per iteration), so this adds ~6-9% to a typical total fit time.

6 new tests, all passing; `test_forward_model.py` full file (9 tests)
green. Not promoted to default -- same opt-in convention as every other
new mechanism in this project, pending real-data validation at scale.

Full writeup: dev_notes/shift_template/FINDINGS.md.

## 2026-07-29 (overnight, cont.) — calibration table regeneration complete

All three tables (main, h4wing, meff) regenerated under current defaults
(order-3 penalty, lam_grid matching select_lambda_aic_local's own
(1,8,64,512,4096,32768)) and verified: load correctly via
load_calibration_table (sigma_range/snr_range/m_eff-axis all sane), and
tests/test_bias_correction.py (10 tests) passes clean against them.

The meff table (8x8 sigma/S-N grid x 6 lam x 30 reps = 11520 fits) took
substantially longer than estimated -- real per-cell cost grows sharply
with sigma (linear-algebra cost of estimate_m_eff_local's own n_coef x
n_coef solve scales with grid width/knot count): sigma=20 took 515s total
(8 workers), sigma=300 took 2461s, sigma=400 took 4325s (~72 min alone).
Two full-grid attempts were killed by fixed timeouts with zero saved
progress before this was understood; rebuilt as a per-sigma-checkpointed
script (dev_notes/calibration/build_meff_incremental.py) that resumes from
disk if interrupted -- this is the version that actually completed
tonight, in roughly 2.5-3 hours wall time total for all 8 sigma values.

`_corrected` fields (bias_correction) are now valid again for the current
package defaults -- were stale since the order-2-to-3 flip.

## 2026-07-29 (overnight, cont.) — notebook 05 re-executed, confirms both fixes on real data

Re-ran examples/notebooks/05_real_data_muse.ipynb end-to-end (was stale:
cached sigma=95.19 predated the edge-amplitude fix, and the bootstrap cell
had a stored KeyboardInterrupt). Fresh execution: 0 errors.

Point estimates: high (nuclear) sigma=49.93 km/s, med (mid-radius)
sigma=41.16 km/s -- both now match the edge-amplitude-constraint fix
(legacy Fortran: 54.3 / 40-45), confirming the fix holds in the actual
notebook context, not just the isolated diagnostic.

Bootstrap (now reselect_lam=True by default): high sigma=49.9+/-0.3,
med=41.2+/-2.6, low=28.4+/-8.5 -- completed cleanly (previously crashed).

Also rewrote notebook 01's Section 10 and "Current limitations", which
had two now-false claims: "do not use apply_bias_correction with
penalty_order=3" (tables were stale, now rebuilt) and "every bootstrap
replicate pulled to the same lam" (reselect_lam=True is now default).
Replaced with real measured numbers on the notebook's own mock (raw sigma
48.3+/-2.8, corrected 46.9+/-4.2, truth 50 -- the old severe
overcorrection to ~22 is gone but correction is not shown to reliably
help either) and an honest reference to the coverage_sweep finding
(47%/11% vs 68% target).

## 2026-07-29 (overnight, cont.) — notebook 01 re-executed clean

examples/notebooks/01_basic_mock_fit.ipynb re-executed after the Section
10/limitations rewrite: 0 errors. Confirmed numbers match the standalone
verification exactly (raw sigma=48.3+/-2.8, corrected=46.9+/-4.2, truth
50). Notebook 06's title bug ("# 05" -> "# 06") fixed; full re-execution
launched to get fresh STIS numbers before updating its Section 5
cross-instrument comparison against notebook 05's new numbers.

## 2026-07-29 (overnight, cont.) — notebook 06 fixed and re-executed clean

examples/notebooks/06_real_data_stis.ipynb: fixed title bug ("# 05" ->
"# 06"), rewrote Section 5's cross-instrument comparison honestly (fresh
STIS: V=+68.3+/-2.6, sigma=61.7+/-3.4, h3=+0.028+/-0.011; fresh MUSE
nuclear: V=-7.1+/-0.2, sigma=49.9+/-0.3, h3=+0.001+/-0.004), and explained
-- rather than silently patched -- that the sigma-comparison DIRECTION
flipped: MUSE's old sigma (~81.5) was inflated by the edge-amplitude bug
this project fixed, so the old "STIS lower than MUSE" framing was an
artifact of a known bug, not a real finding; the fresh comparison is STIS
HIGHER by ~24%. Also softened the bootstrap-band markdown's stale
same-lam-every-replicate claim to reflect reselect_lam=True and added an
honest reference to the 47%/11% coverage finding. Re-executed twice (once
to get fresh numbers to write from, once after the edit) -- 0 errors both
times.
