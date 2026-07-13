# Real-data validation against the legacy SCO pipeline: MUSE (fixed) and STIS (open)

Both `examples/notebooks/03_real_data_muse.ipynb` and `04_real_data_stis.ipynb` fit
the same central bin (`bin0105sp`, r=0.01") of NGC 5102 that the legacy Fortran
pipeline (`pallmc`/`pfitlov`) already fit and validated. The legacy pipeline's own
outputs live in
`/Users/waterstk/Documents/Gultekin_Astrophysics/normalizing_flows/data/N5102/kinematics/{muse,stis}/kinematic_extraction/`
-- in particular `pallmc.out` (the final per-bin V/sigma/h3/h4 catalog, one row per
bin) and `.lastinput` (the redshift actually used in that instrument's line-fitting
run). This session compared kinextract's recovered LOSVDs against that catalog and
found real discrepancies -- most visibly, a spurious secondary bump in the recovered
LOSVD near v~+250 km/s in both instruments' fits.

## MUSE: resolved (in two stages -- the first "fix" was incomplete)

**Stage 1 -- redshift (necessary, not sufficient).** `kinextract`'s notebook used
`zgal=0.001556` (a generic NED redshift estimate). The legacy pipeline's own
`.lastinput` for this MUSE run records `last_redshift=0.0016181` -- a difference of
0.0000621, equivalent to ~18.6 km/s. Switching to that value moved the summary
Gauss-Hermite moments into good agreement with legacy's `pallmc.out`:

| | V (km/s) | sigma (km/s) | h3 | h4 |
|---|---|---|---|---|
| kinextract, before | +25.8 | 38.1 | -0.128 | +0.219 |
| kinextract, after z-fix (GH moments only) | +0.6 +/- 2.5 | 53.1 +/- 5.3 | -0.001 +/- 0.070 | -0.023 +/- 0.066 |
| legacy `pallmc.out` (bin0105) | -0.5 +/- 5.0 | 54.3 +/- 2.0 | +0.005 +/- 0.021 | -0.048 +/- 0.009 |

**This was reported as "resolved" and was wrong to call that.** The GH-moment summary
matched legacy well, but the underlying non-parametric LOSVD *shape* was not checked
before making that claim -- the user caught this directly by looking at the actual
plot: the recovered LOSVD had a genuine double-peaked core (confirmed via
`scipy.signal.argrelextrema` -- two real local maxima at v=-41 and v=+21 separated by
a real local minimum at v=-14, not a rendering artifact) plus a persistent, non-
decaying wing: **~25-30% of the total LOSVD probability mass sat at |v|>150 km/s**,
a region where the legacy fit has essentially zero signal. Low-order GH moments (up
to h4) can miss this kind of structure entirely -- matching summary statistics is
not the same as matching the shape a Schwarzschild-modeling stage would actually
consume. Lesson: for LOSVD shape questions, always look at (and ideally directly
diff against) the actual non-parametric array, not just the GH-fit summary.

**Stage 2 -- the double-peak/wing-pedestal root cause and fix.** Investigated in
order:

1. `n_losvd_bins=89` (this session's own earlier package-default change, made for
   an unrelated synthetic-mock sigma-bias fix) turned out to be the direct cause of
   the double-peaked *core*: reverting to `n_losvd_bins=29` on this exact spectrum
   gave a single-peaked core matching legacy's shape closely. Confirmed side-by-side
   (same redshift, same everything else): nbins=89 shows a real dip at v=-14 (local
   min 0.00491 between two local maxima 0.00556/0.00689); nbins=29 shows a single
   clean maximum at v=0 (0.00635) with no interior dip. This does *not* mean
   nbins=89 is universally wrong -- see `dev_notes/nbins_xlam_regression_history.md`
   for why it was raised in the first place -- just that it overfits this
   particular real (noisier, more model-imperfect) spectrum into spurious structure
   a coarser grid doesn't.
2. The wing pedestal was present at *both* bin counts and was **not** fixed by
   increasing regularization strength (`xlam`) -- swept 1e3 to 1e6 with
   `xlam_auto=False`; the wing mass shrank only slowly and the fit degraded
   (chi2_red crept from 0.25 up to 0.27, core shape barely changed). Root-caused by
   reading `kinextract.numerics.objective_map`/`objective_joint` directly: the
   objective is `chi2 + smoothness_penalty (wing-tapered 2nd-derivative roughness)
   + 0.1*|sum(b)-1|`. The wing-taper *scales up* the roughness penalty far from
   `v_center`, but roughness only penalizes *curvature* -- a flat, non-zero pedestal
   has near-zero curvature, so it is nearly free for the optimizer to leave in
   place regardless of how large the tapered weight gets, and doesn't violate the
   *global* sum-to-1 constraint either (mass can be borrowed from a slightly
   narrower core). No term in the objective ever penalizes a bin's raw amplitude
   just for being far from the peak. Checked the legacy Fortran source
   (`sco_framework_updated/modprogs/`) for a term kinextract's port might be
   missing; the actual fitting routine's source isn't present in that archive
   (only driver/output-writer code), so this couldn't be directly confirmed against
   the original, only inferred from kinextract's own code and behavior.
3. **Fix**: added `FitConfig.xlam_wing_shrink` (default `0.0`, opt-in) -- an L2
   amplitude penalty on `b`, active only in the wing-taper region, implemented in
   `kinextract.numerics._compute_wing_shrinkage` and wired through both the shipped
   (`objective_map`) and joint (`objective_joint`) objectives, NumPy and JAX paths,
   plus both JAX kernel caches' key functions (the value is baked into the compiled
   kernel, so it must be part of the cache key or two fits with different
   `xlam_wing_shrink` could silently share a wrongly-compiled kernel). An early
   version scaled the wing weight by `(lam_j - xlam)` (`lam_j` being the existing
   wing-tapered roughness weight) and badly destabilized the auto-xlam discrepancy
   search, since that search calibrates xlam against how chi2 rises with it and
   this made the new term's own strength scale with xlam too, entangling the two;
   fixed by making the wing weight dimensionless and independent of xlam's current
   value (`(d/sigl0/1.8)**4 - 1`, clipped at 0). Confirmed a true no-op at the
   default (`0.0`): full test suite (115 tests) passes identically with and without
   the new code present.

**First attempt -- `xlam_wing_shrink=1e6` at nbins=89, `xlam_auto=True`.** Result
looked excellent on the point estimate:

| | V (km/s) | sigma (km/s) | h3 | h4 | mass at \|v\|>150 | local maxima in core |
|---|---|---|---|---|---|---|
| kinextract, nbins=89, no wing_shrink | +0.6 | 53.1 | -0.001 | -0.023 | 30.5% | 3 (double/multi-peaked) |
| kinextract, nbins=29, no wing_shrink | +1.6 | 56.1 | +0.042 | +0.026 | 25.7% | 1 |
| kinextract, nbins=89, wing_shrink=1e6, xlam_auto | +2.3 | 54.5 | -0.033 | -0.059 | 0.7% | 1 |
| legacy `pallmc.out` (bin0105) | -0.5 +/- 5.0 | 54.3 +/- 2.0 | +0.005 +/- 0.021 | -0.048 +/- 0.009 | ~0% | 1 |

sigma matched legacy almost exactly, wing mass collapsed to <1%, core single-peaked.
**This was reported as fixed. It was not** -- the user asked to see the actual
bootstrap error bars, which had also collapsed: 30-replicate bootstrap sigma samples
ranged only 50.6-51.9 km/s (std=0.36, 2.4% of the MAP value) vs. 38.5-64.7 km/s
(std=5.57, healthy) with `wing_shrink=0`. **Every bootstrap replicate was landing on
nearly the same answer regardless of which noise realization was resampled** -- not
genuine statistical uncertainty, just the regularizer's own residual jitter. This
is dangerous specifically because it looks like a win on the point estimate while
silently producing overconfident, misleading error bars for downstream
Schwarzschild-modeling input.

Root cause: `xlam_auto=True`'s discrepancy-principle search doesn't just pick an
xlam value once -- every trial fit *inside* the search still carries the (fixed)
`xlam_wing_shrink` term, which changes what chi2 each candidate xlam actually
achieves, and the search compensated by pushing the selected xlam all the way to the
search grid's ceiling (1e7). Bootstrap replicates freeze `xlam` at that same
already-runaway value (by design -- see `_make_frozen_cfg`, so replicate-to-replicate
differences reflect noise, not a re-run search), so every replicate got hit with an
enormous *combined* regularization budget (huge `xlam` + `xlam_wing_shrink` alike),
leaving almost no freedom for noise to move the answer. This is a generic property of
heavily-regularized fits (well documented for e.g. ridge-regression bootstrap CIs),
not unique to this new term, but `xlam_wing_shrink` pushed this fit deep enough into
that regime to make it severe.

**Follow-up: does `xlam_wing_shrink` have a safe operating point at all?** Tested
with `xlam_auto=False` (a *fixed* xlam, avoiding the runaway-search interaction) across
a grid of fixed-xlam x wing_shrink combinations. Two findings killed the approach:
- Any `xlam_wing_shrink` strong enough to meaningfully clear the wing pedestal (mass
  at \|v\|>150 down to a few percent) also dragged sigma well away from the legacy
  value (up to sigma~65-67 vs. legacy's 54.3) -- even fairly small values (`ws=1e2`)
  already nudged sigma up with almost no wing-mass benefit. Mechanism: the wing
  taper's onset (`1.8*sigl0` ~ 108 km/s for `sigl=60`) sits too close to the peak --
  it starts suppressing amplitude right where genuine tail/edge-of-core signal still
  exists, not just the truly-empty far wings, so there is no clean separation between
  "kill the pedestal" and "distort the real kinematics."
- Even at fixed xlam, larger `xlam_wing_shrink` values still visibly narrowed the
  bootstrap spread (though less catastrophically than the auto-xlam case above).

**Final approach, shipped**: no `xlam_wing_shrink` in notebook 03 at all.
Instead, `xlam_auto=False` with a fixed `xlam=1e5` (vs. the auto search's own
~2.2e4) -- found during the fixed-xlam sweep above to *already* give a single-peaked
core closely matching legacy's shape, with no wing-shrink side effects:

| | V (km/s) | sigma (km/s) | h3 | h4 |
|---|---|---|---|---|
| **kinextract, fixed xlam=1e5, no wing_shrink** | **+1.0 +/- 2.0** | **55.0 +/- 4.2** | **+0.038 +/- 0.037** | **+0.003 +/- 0.039** |
| legacy `pallmc.out` (bin0105) | -0.5 +/- 5.0 | 54.3 +/- 2.0 | +0.005 +/- 0.021 | -0.048 +/- 0.009 |

Every value agrees with legacy within combined uncertainty, and the bootstrap error
bars are back to a healthy scale (sigma +/-4.2, comparable to legacy's own +/-2.0 --
not artificially narrow). chi2_red=0.257, essentially unchanged from the
unconstrained baseline. The v~+250 bump/wing pedestal is **still visually present**
(~25-28% of LOSVD mass at \|v\|>150 km/s) -- this is now accepted as a known,
cosmetic-only residual: it does not measurably affect the reported V/sigma/h3/h4 or
their bootstrap uncertainties, and every attempt to remove it introduced a worse,
substantive problem (either accuracy or uncertainty-quantification integrity). Fixing
it properly would need a more surgical version of the wing-shrinkage idea (e.g. a
taper onset decoupled from and set further out than the roughness penalty's own
`1.8*sigl0`) -- not attempted this session; `FitConfig.xlam_wing_shrink` remains in
the codebase as an opt-in, off-by-default feature for future use, with this session's
findings written into its own docstring.

**Lesson, on top of the "check the shape, not just the summary" one from stage 1**:
when validating a regularization change, check the *bootstrap error bars*, not just
the point estimate. A fix that visually cleans up a LOSVD while silently collapsing
its reported uncertainty is a worse outcome than the original problem, because it's
much harder to notice.

**Stage 3 (follow-up session) -- two more attempts, both negative results.** The
wing pedestal (still ~28% of LOSVD mass at |v|>150 km/s in the shipped fixed-xlam=1e5
configuration) was flagged by the user as still looking clearly wrong on a fresh look
at the plot, so it was investigated further rather than left as "accepted cosmetic."

1. **Decoupled wing-shrink onset.** Added `FitConfig.xlam_wing_shrink_sfac` (default
   1.8, matching the roughness penalty's own onset) so `xlam_wing_shrink`'s taper can
   be set to kick in further out (e.g. 2.5-3.0*sigl0) than the roughness penalty's,
   on the theory that the shared 1.8*sigl0 onset was touching genuine tail signal.
   Result: **worse, not better** -- at sfac=2.5-3.0, frac_far actually *increased*
   (up to 40%) and the core picked up 2-3 spurious peaks instead of 1, even at strong
   wing_shrink. Moving the onset out left a gap (roughly 108-160 km/s) where neither
   penalty's *amplitude* is constrained (only the separate, weaker roughness taper
   is), and that gap is apparently exactly where new spurious structure wants to
   form. The shared-onset (sfac=1.8, matching roughness) design from stage 2 remains
   the better of the two; `xlam_wing_shrink_sfac` is now a real, tested opt-in field
   (defaults to 1.8, a no-op) but this session's own attempt to use a larger value
   did not pan out.
2. **Real-data-defect check.** Inspected the actual per-pixel spectral residuals
   (data-model)/error directly rather than continuing to tune regularization blind.
   Found a genuine, unmasked ~3.6-sigma outlier at rest-frame ~8737 A, just past an
   already-(partially-)masked emission-affected segment (~8712-8713 A) near the red
   edge of the fit window -- exactly the kind of localized defect that could explain
   the model wanting spurious extreme-velocity LOSVD flux (a residual near the window
   edge can get "explained" via the wavelength<->velocity mapping as an extreme
   velocity shift). Tested narrowing `wavefitmax` from 8750 to 8710/8700 A to exclude
   it: frac_far only dropped from 0.28 to ~0.25-0.26 -- a real but small effect, not
   the dominant explanation. The wing pedestal is evidently not attributable to this
   one pixel; something more structural (the objective's lack of a genuine
   amplitude-toward-zero prior, as diagnosed in stage 2) remains the primary driver.

**Status at the end of this session**: the wing pedestal is a real, still-open
problem. Every fix attempted (wing-shrink at the shared onset, wing-shrink at a
decoupled onset, fixed vs. auto xlam, excluding the one confirmed bad pixel) either
didn't remove it, removed it at the cost of accuracy or bootstrap validity, or made
the core shape worse. The currently-shipped `xlam=1e5`, no-`xlam_wing_shrink`
configuration remains the best of the options actually tried -- accurate GH moments,
healthy bootstrap error bars, single-peaked core -- but the wing pedestal itself is
unresolved, not merely "cosmetic and accepted." Worth revisiting with fresh ideas
(e.g. a proper amplitude-toward-zero *prior* built into the objective from the start,
rather than a bolted-on penalty term; or a closer look at whether other real MUSE
bins show the same pedestal, which would help distinguish a data-specific issue from
a genuinely general one) when there's time to do it justice.

## `06_prenormalized_workflow.ipynb`: same bin, different code path, same symptoms -- resolved

Fits the identical MUSE bin0105 spectrum through a completely different route:
manually ALS-pre-normalize (divide out a fitted continuum once), then
`fit_continuum=False, joint_prenorm=False` -- the plain shipped non-parametric MAP
path, not `kinextract.joint` at all. The user noticed this notebook's LOSVD looked
similarly bad and flagged that as reassuring (same symptom, independent of the joint
continuum machinery -- rules out a joint-specific bug) but still clearly wrong. Three
separate problems, found in this order:

1. **Stale redshift.** This notebook was never updated with the `zgal=0.0016181` fix
   found for notebook 03 -- still had the old generic-NED `0.001556`.
2. **Wrong template source.** Still loaded E-MILES SSPs by directly importing `ppxf`
   and reading its bundled `spectra_emiles_9.0.npz` -- the exact thing notebooks 02/03
   were migrated away from earlier this session (that grid's license prohibits
   redistribution). Switched to kinextract's own packed grid
   (`template_npz_file`/`template_npz_select`), matching notebooks 02/03.
3. **Runaway auto-xlam**, same root cause as the joint-mode case but discovered
   independently: `xlam_auto=True` here has an even flatter chi2(xlam) curve than
   notebook 03 -- already 10 spurious peaks at the grid's lowest candidate
   (xlam=100), and chi2 barely rises even at xlam=1e8 -- so the discrepancy-principle
   search's bracket-expansion loop ran all the way to xlam~1.24e8, giving a badly
   oversmoothed LOSVD (sigma=77.1 km/s vs. legacy's ~54.8). Fixed the same way as
   notebook 03: `xlam_auto=False`, fixed `xlam=1e5` (the exact same value, though
   arrived at independently via its own small fixed-xlam sweep, not copied over).

Result after all three fixes, compared against the legacy `bin0105spmc_in` file (the
actual *_in modeling-input LOSVD the user pointed to -- 1000-point, finely-sampled,
independently confirms the same V=+0.5/sigma=54.8/h3=-0.004/h4=-0.045 already found
from `pallmc.out`/`.mcfit2`):

| | V (km/s) | sigma (km/s) | h3 | h4 |
|---|---|---|---|---|
| kinextract nb06, fixed xlam=1e5 | -8.3 | 53.9 | +0.053 | -0.032 |
| legacy `bin0105spmc_in` | +0.5 | 54.8 | -0.004 | -0.045 |

Single-peaked, cleanly decays to zero by ~|v|=150 km/s (no visible wing pedestal at
all -- cleaner than notebook 03's own residual pedestal, plausibly because this
fit's pixel count/masking/continuum treatment differ). sigma matches closely. There
is a real, not-yet-explained ~9 km/s V offset between this notebook's -8.3 and
notebook 03's own +1.0 for the *same physical bin* -- plausibly a real,
methodological difference (ALS pre-normalization vs. joint P-spline continuum
co-fit can each impose a slightly different effective tilt/weighting across the fit
window), but not confirmed. Flagged honestly here rather than chased further given
time constraints; worth a closer look if V accuracy at this level matters for a
specific downstream use.

## STIS: NOT resolved -- deliberately left out of this release

The analogous fix (STIS's own `.lastinput`: `last_redshift=0.00157136`, vs the
0.001556 previously used -- only a ~4.6 km/s difference) is applied in notebook 04,
but does not close the gap:

| | V (km/s) | sigma (km/s) | h3 | h4 |
|---|---|---|---|---|
| kinextract, after z-fix | +15.8 +/- 3.3 | 79.5 +/- 4.5 | +0.139 +/- 0.028 | +0.106 +/- 0.032 |
| legacy `pallmc.out` (bin0105) | +2.3 +/- 4.2 | 67.3 +/- 3.6 | +0.030 +/- 0.026 | -0.034 +/- 0.014 |

The v~+250 bump is still present in the STIS LOSVD. Per the user's own domain
knowledge: the legacy extraction tuned MUSE's and STIS's redshifts *separately* to
zero the central bin's velocity, so some residual V offset between instruments is
expected and not itself concerning -- but sigma should be consistent, and a ~12 km/s
(~18%) excess is a real, unexplained problem.

**Note for whoever picks this back up**: the MUSE investigation below found this
same v~+250 bump was a genuine wing-pedestal optimizer artifact (see the MUSE
section), fixable with the new `FitConfig.xlam_wing_shrink` opt-in penalty -- worth
trying on STIS too, but NOT assumed to transfer directly: STIS's sigma discrepancy
is in the *opposite* direction from MUSE's original one (kinextract overestimates
STIS's sigma; it underestimated MUSE's before the nbins/wing-shrink fix), so the
same value (or even the same sign of fix) is not guaranteed to help here, and this
was not tested this session per the decision below.

### Ruled out this session

1. **Redshift** -- only ~4.6 km/s for STIS; applied, insufficient.
2. **Regularization strength (`xlam`)** -- swept 5e4 to 1e7 with `xlam_auto=False`.
   Stronger regularization makes both V and sigma drift *further* from legacy
   (sigma 78.6 -> 96.0 as xlam goes 5e4 -> 1e7), and the bump never registers as a
   detected peak (`compute_losvd_n_peaks`, prominence 0.1) at any xlam tested -- it's
   a genuine sub-threshold shoulder, not an under-smoothing artifact.
3. **The legacy `regions.dat` file** (found in the STIS `kinematic_extraction`
   directory) -- its listed pixel segments have small gaps between them; converting
   those gaps to rest-frame wavelength (using `zgal=0.00157136`,
   `wave = 8275.0 + pix*0.5586`) puts them almost exactly on the three Ca II line
   cores (8498/8542/8662 A). Masking those gaps out via
   `FitConfig.regions_bad_path` (as if they were a bad-pixel list) makes the fit
   *much* worse (V jumps to +47, sigma to 131) -- strong evidence `regions.dat`
   belongs to an earlier continuum-fitting stage (`fitcontinuum`/`rimfit`, per
   `kinematic_extraction.info`), not the kinematic-fit stage, and should NOT be
   applied here.
4. **LSF/resolution mismatch** (`FitConfig.data_fwhm_A`/`template_fwhm_A`) -- tested
   a plausible range of template-vs-data FWHM combinations; only moves sigma by
   ~2 km/s (79.5 -> 77.0 at the most aggressive setting tested), far short of the
   ~12 km/s gap. Also, the template file `examples/data/stis/HR7615ecbig.dat` has a
   native pixel step of 0.5547 A, almost identical to STIS's own 0.5586 A/pixel --
   suggesting this template may already be at (or close to) STIS's own resolution,
   which would mean little or no real LSF mismatch exists to correct in the first
   place (undermining, though not fully disproving, this lever).

### Not checked (blocked on data/access, not further reasoning)

- The `.spec` file's own error column is a literal placeholder (`1.0` for every
  pixel) -- the current `flux/25` S/N guess is the best available from bundled data.
  The legacy pipeline's real per-pixel STIS errors (presumably derived upstream from
  the `calstis` reduction's own error array) were not found anywhere in the
  reference directory.
- The legacy fit's exact template (or template set) is untraceable from the files
  present in `kinematic_extraction/` -- no `Tlist`-equivalent exists there; the
  compiled Fortran pipeline likely referenced a hardcoded path outside this archive.

## Decision (this session)

Per explicit user instruction: ship the MUSE fix; leave STIS's notebook/config as-is
(z-fix applied, real discrepancy documented, not further tuned) and treat it as a
known, open issue to revisit later with more information -- not something to guess
at further right now.

**Update 1, same session**: the user then caught that the initial MUSE "fix" was
incomplete -- the redshift correction alone left a genuine double-peaked LOSVD and
a large spurious wing pedestal that the GH-moment summary happened to mask. This
was root-caused (see "MUSE: resolved" above, stage 2) to a real gap in the
regularization objective (no amplitude-shrinkage term in the LOSVD wings, only a
curvature/roughness penalty), and "fixed" with a new opt-in `FitConfig.xlam_wing_shrink`
penalty (`xlam_wing_shrink=1e6`), which cleaned up the point estimate dramatically.

**Update 2, same session**: the user then caught that this "fix" had *also* silently
collapsed the bootstrap error bars to an unrealistically narrow, unrepresentative
scale (sigma +/-0.3 km/s instead of a healthy +/-4-5 km/s) -- a second instance of
declaring success from a summary number (this time the point-estimate LOSVD shape)
without checking everything the change touched. Root-caused to the `xlam_wing_shrink`
term interacting with the auto-xlam discrepancy search (pushing `xlam` to the search
grid's ceiling) and, more fundamentally, to heavy total regularization compressing
bootstrap replicate-to-replicate variability regardless of noise. No safe operating
point for `xlam_wing_shrink` was found on this spectrum (see "MUSE: resolved" above
for the full grid search). **Final shipped fix**: a simple fixed `xlam=1e5` (no
`xlam_wing_shrink`, no `xlam_auto`) -- matches legacy's V/sigma/h3/h4 within combined
uncertainty, with healthy (non-collapsed) bootstrap error bars, at the cost of leaving
the wing pedestal cosmetically present but harmless. `xlam_wing_shrink` remains in the
codebase, off by default, for future work on a more surgical version. Full test suite
(115 tests) re-confirmed passing throughout every stage of this investigation.
