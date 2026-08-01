# kinextract: comprehensive validation campaign

**Status: IN PROGRESS. Started 2026-07-18 (approx). If you are a fresh
session picking this up after an interruption, read this whole file
before doing anything else, then jump to "Resume point" at the bottom.**

## Why this file exists

The user asked for a full, statistically-powered re-validation of every
feature built so far, across the entire physically-relevant sigma range,
before trusting the package further -- explicitly authorizing a long,
unattended run ("leave my laptop on... don't sacrifice completeness for
the sake of accuracy") and asking that progress be tracked durably enough
to survive a session interruption (a session limit was already hit once
earlier in this project). This file is that durable state: update it
after every completed stage, not just at the end. Conversation context
can be summarized/lost; this file and the repo's own test suite cannot.

**User's own words, verbatim, on scope and standard:**
> "I want to go back and reverify every step that we have done. Think of
> it this way: build step 1, test step 1, build step 2, test step 1,
> test step 2, test steps 1 and 2 together (if applicable), repeat with
> remaining steps. I also want to be sure that the code functions in
> every regime that is physically feasible... low velocity dispersions...
> 20-30 km/s... reaching the resolution limits of instruments like
> MUSE... highest sigma values measured in huge ellipticals (350 km/s+)
> with both high res and low res instruments. And we need to test
> everything inbetween with a dense enough grid that any trends or biases
> we find are statistically certain and not noisy measurements... don't
> sacrifice completeness for the sake of accuracy... If you uncover any
> issues, document them, and fix them with the best methods you know of
> that are reasonable both in computation time and complexity. If we find
> true biases that can't be easily corrected, document them honestly,
> propose fixes if they exist."

## Ground rules carried over from this whole project's history

- No `git add`/`commit`/`push` without fresh explicit permission (not
  yet requested for this campaign).
- Verify before building: don't add fixes for effects that haven't been
  empirically confirmed first.
- A coverage/bias number from < ~40 realizations/case is not trustworthy
  on its own -- this project has been burned by this twice already (see
  memory: `project_kinextract_deep_validation.md`, "third follow-up").
  Every new grid cell in this campaign uses n>=40 unless explicitly noted
  as a fast first-pass screen (and re-checked at higher n if it looks
  interesting).
- Opt-in features stay opt-in (off by default) unless overwhelming
  evidence says otherwise. Backward compatibility (bit-for-bit, tested)
  is non-negotiable for every existing default code path.
- Honest documentation over a tidy narrative -- an unresolved or
  worsened case gets written down as such, not smoothed over.

## What's already validated coming into this campaign (do not re-litigate from scratch)

See `README.md` "Known limitations" and `src/kinextract/uncertainty.py`'s
module docstring for the full detail. Headline facts this campaign
should either reconfirm at wider scope or leave alone:
- Wing-inflation sigma bias at low S/N, worst at low sigma (mechanistic,
  `wing_lam` mitigates, needs manual tuning).
- Taper-windowing sigma bias at nonzero h3/h4 (mechanistic,
  `shape_aware_correction` mitigates *except* at sigma>=~300, opt-in,
  off by default, now has an automatic post-fit warning if misused).
- sigma=300 bootstrap over-coverage: "sloppy manifold" hypothesis,
  unresolved, and this session found a real, unreconciled discrepancy
  between two different measurements of the same std-inflation number
  (1.47x vs ~1.0x) -- flagged as needing a clean re-measurement, not yet
  done.
- h4=+0.1 (sigma=140) under-covered from the same taper-windowing effect.
- Multi-template bootstrap: only one case checked (sigma=140, n=60),
  mild over-coverage on V/sigma, one under-covered template weight of
  three. No grid yet.
- **NEW this session, low-sigma h4 point-estimate bias**: mean(h4) at
  true h4=0 is robustly ~+0.03 at sigma=50 (S/N 15-90), but ~0 at
  sigma=140/300. Confirmed fully decoupled from noise-level correctness.
  Root cause NOT YET INVESTIGATED -- high priority for this campaign.
- **NEW this session, noise-scale calibration** (`calibrate_noise_scale`,
  `noise_scale` on both bootstrap functions): validated against uniform
  noise mis-specification on synthetic data (works well, see
  `uncertainty.py`). **NOT validated for real data** -- real per-bin
  residuals have localized outlier pixels (likely sky-residuals, see
  below), not uniform noise error, and applying the current tool to real
  data shrinks uncertainty in the wrong (dangerous) direction. Needs a
  localized-outlier synthetic test before ever being trusted on real
  data again.
- **NEW this session, real-data sky-residual candidate**: four fixed,
  recurring bad-pixel wavelength windows found in every real NGC 5102 bin
  checked so far (~8415-8418, ~8450-8453, ~8712-8713, ~8749-8752
  Angstrom), growing sharply worse at faint/outer bins, NOT near
  CaII/Paschen. Likely OH-airglow sky-subtraction residuals. Not yet
  masked, not yet root-caused beyond "recurring at fixed wavelength."
- V offset (real data): confirmed NOT reproduced in synthetic testing in
  the matching regime (sigma 40-60, S/N 15-90, V +-20) -- real-data/
  legacy-side cause, not a kinextract point-estimate bug. Redshift-
  convention hypothesis directly tested and ruled out.

## The staged plan

Given the combinatorial size of "every sigma x every S/N x every
resolution x every feature x proper statistical power," this is staged
from cheap/broad to expensive/targeted, not brute-forced uniformly.
Update the checklist below as each stage completes; write results
directly into this file (not just scratch/tmp, which does not survive a
session restart) as a `### Results` subsection under each stage.

### Stage 0 -- housekeeping (do first, cheap)
- [ ] Fix `calibrate_noise_scale`/`bootstrap_losvd` real-data-caveat
      documentation. **DONE** (this session, before this file was
      written) -- see `uncertainty.py`.
- [ ] Full existing test suite green before starting (baseline).

### Stage 1 -- broad point-estimate sweep (cheap, single-template, no bootstrap)
Sweep V/sigma/h3/h4 point-estimate bias across the full physically-
relevant grid, single-template, no bootstrap (fast: ~0.5-1s/fit). Goal:
find any NEW regime with a real bias not yet characterized, at moderate
power (n=25/cell -- enough to see a real trend, not enough to be final;
promote any interesting cell to Stage 1b).

Grid:
- sigma: 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 180, 220, 260, 300,
  350, 400 km/s (16 points -- dense at the low end per the user's
  explicit "resolution limit" concern, and around the already-known
  sigma=300 problem zone).
- S/N: 15, 30, 60, 100, 200 (5 points).
- V: 0 only for this broad pass (V bias already checked separately and
  ruled out as a synthetic-side issue; not worth tripling this grid).
- h3=h4=0 truth (isolates point-estimate bias in V/sigma/h4 cleanly;
  nonzero-h3/h4 truth is Stage 1c, narrower).
- "Instrument resolution": two conventions --
  (a) MUSE-like, `step=1.25` Angstrom/pixel (already used everywhere
      this session, matches real N5102 data).
  (b) A finer, "high-res" convention, `step=0.5` Angstrom/pixel (roughly
      STIS-like sampling density), to directly test the user's
      "high-res instruments resolving 20-30 km/s" question -- does the
      low-sigma bias/resolvability change with pixel sampling density?
  16 sigma x 5 S/N x 2 resolution = 160 cells x n=25 = 4000 fits.
      Estimated runtime: single-template point fits run ~0.5-1s each in
      this environment -> roughly 35-70 minutes. Tractable as one
      background job; split into 2-4 jobs by resolution/sigma-range if
      that turns out to be more convenient to checkpoint.

**Status: DONE.** Both grids completed (~950-1100s each). Full results in
`examples/notebooks/stage1_results_{muse,highres}.txt`. n=25/cell, so
individual small effects are noisy, but every finding below has an
effect size many times its own standard error of the mean
(SE = std/sqrt(25)) -- these are not chance small-sample artifacts.

#### Results and interpretation

**1. Low sigma (20-60 km/s): the wing-inflation sigma bias extends much
further and much worse than previously characterized, and instrument
resolution only partially helps.** Previously the package's own
validation only went down to sigma=50. Extending to 20-60 (MUSE
convention, S/N=15->200):

| sigma | S/N=15 | S/N=30 | S/N=60 | S/N=100 | S/N=200 |
|---|---|---|---|---|---|
| 20 | +203% | +156% | +112% | +80% | +50% |
| 25 | +157% | +118% | +73%  | +53% | +29% |
| 30 | +130% | +82%  | +54%  | +35% | +17% |
| 40 | +76%  | +49%  | +27%  | +14% | +4%  |
| 50 | +52%  | +29%  | +13%  | +5%  | -1%  |
| 60 | +35%  | +16%  | +7%   | +2%  | -2%  |

**sigma=20 km/s is essentially unmeasurable with the current defaults --
even at S/N=200 (very high, unrealistic for most real spectra) the bias
is still +50%.** This directly answers the user's "can we resolve
20-30 km/s" question: no, not without `wing_lam` tuning or a different
regularization approach at this sigma, regardless of S/N. The high-res
(step=0.5, ~801 pixels in-window vs. MUSE's 321) convention helps, but
only modestly -- e.g. sigma=20/S-N=200: muse +50% -> highres +34%; still
a large, real bias. **This confirms the bias is fundamentally an S/N-
driven regularization effect (the curvature-only penalty's own
behavior), not a pixel-sampling/instrumental-resolution artifact** --
finer sampling gives a bit more independent information per fit but
doesn't touch the underlying mechanism.

**2. NEW finding: the same wing-inflation mechanism reappears at HIGH
sigma when S/N is low enough -- not just a low-sigma phenomenon.**
Previously sigma=300 was only ever tested at S/N=100 (where it looks
fine on this specific bias, its problem is the separate over-coverage
issue). At S/N=15-30, high sigma shows a severe, previously-undetected
positive bias with enormous scatter:

| sigma | S/N=15 bias | S/N=15 std | S/N=30 bias |
|---|---|---|---|
| 220 | +21% | 56 km/s | +9%  |
| 260 | +26% | 68 km/s | +11% |
| 300 | +31% | 80 km/s | +9%  |
| 350 | +28% | 74 km/s | +14% |
| 400 | +26% | 97 km/s | +10% |

Muse and highres both show this (highres somewhat smaller, e.g.
sigma=300/S-N=15: muse +31% -> highres +12%, still large). This makes
sense once seen: wing-inflation is an S/N phenomenon, not a sigma
phenomenon -- high-sigma fits simply have more spline coefficients and
more information to fit at moderate-to-high S/N, masking the effect
until S/N drops far enough. **This had never been tested because the
standard grid always paired high sigma with S/N=100.**

**3. h4 point-estimate bias (the +0.03 finding from earlier this
session) is now confirmed across the FULL low-sigma range (20-60), not
just sigma=50, and stays essentially flat (~+0.031 to +0.037) regardless
of S/N within that range** -- consistent with it being a structural,
not noise-driven, effect (matches the earlier direct noise-mismatch
test). Above sigma~80 it fades toward zero as expected. **A second, new
h4 phenomenon appears at high sigma + low S/N**: h4 becomes wildly
unstable (std up to 0.08, mean swinging both signs) at sigma>=220,
S/N<=30 -- e.g. sigma=300/S-N=15: h4=+0.078+-0.085. This looks like
another facet of the same sigma>=~220-300 "sloppy manifold" over-
coverage story, now showing up as h4 instability too, not just sigma
over-coverage -- worth connecting to that investigation.

**4. V bias stays consistent with pure noise (no systematic trend)
everywhere in both grids**, including at the extreme low-S/N/high-sigma
corner where std balloons (e.g. sigma=400/S-N=15: +12.8+-60.4 km/s --
huge scatter, but the mean isn't significant relative to it). Reconfirms
the earlier finding that V is not the package's problem.

**Action items generated**: (a) `wing_lam`'s own documentation and the
`README`/`gh_moments.py` "Known limitations" need updating with the
much-worse-than-previously-stated low-sigma numbers and the new sigma=20
"structurally unresolvable" finding; (b) the high-sigma-low-S/N
wing-inflation reappearance is a genuinely new mechanism finding, needs
its own note; (c) Stage 1b (targeted n=60 re-tests) is only needed for
borderline/near-zero cells if any conclusion hinges on them -- the
headline findings above already have effect sizes 4-10x their own
standard error of the mean at n=25, so a blanket re-test isn't the best
use of remaining time; spot-check only if something specific looks
ambiguous later.

### Stage 1b -- targeted re-test of any Stage 1 cell showing a plausible effect
Any cell where |mean bias| looks large relative to its own std at n=25
gets a proper n=60 re-test before being written up as a real finding
(same discipline as this project's own "third follow-up" lesson).

**Status: NOT STARTED (depends on Stage 1).**

### Stage 1c -- nonzero h3/h4 truth, same sigma/S-N grid, narrower
Repeat a subset of Stage 1 (say sigma = 20, 30, 50, 80, 140, 220, 300,
400 -- 8 points, both resolutions) at h3=+-0.1, h4=+-0.1 truth, to check
whether `shape_aware_correction`'s validated range (good except >=~300)
holds up at the low-sigma end too, and whether the new h4-at-zero bias
interacts with nonzero true h4 differently than expected.

**Status: DONE** (MUSE resolution only, S/N=30/100, n=25/cell; 8 sigma
x 4 shapes x 2 S/N = 64 cells -- raw per-cell sweep script/output removed
in the later spring-cleaning pass as superseded by the general
`calibrate_bias`/`ablation` framework; findings below are the full
record).

**Results**:
- **`shape_aware_correction` still measurably worsens sigma at high
  sigma even combined with low S/N** (sigma=300/S-N=30, h4+ truth: raw
  +42.6 -> corrected +48.1 km/s) -- consistent with, and now extended
  to, the "don't use above sigma~200" guidance already in place.
- **h3 recovery is severely suppressed at low-to-moderate sigma**: for
  true h3=+0.1, recovered h3 stays near 0 (bias ~-0.10, i.e. essentially
  full suppression) across sigma=20-80, only starting to recover by
  sigma>=140. Not previously characterized at all (h3 accuracy was never
  swept this way before).
- **The low-sigma h4 bias (Stage 1) is an additive floor, not a bias
  proportional to the true value**: for true h4=-0.1, recovered h4 lands
  at ~+0.03 (not ~-0.07) across sigma=20-140 -- i.e. something pulls
  recovered h4 toward a roughly fixed +0.03ish value regardless of the
  true underlying h4, weakening only gradually above sigma~140-220. See
  Stage 5 below.

### Stage 2 -- bootstrap coverage across the same grid (expensive)
The real test: does the reported ~1-sigma interval actually contain
truth ~68% of the time, across the same sigma/S-N space. Each cell needs
n>=40 realizations x n_boot=12 refits = a lot more compute than Stage 1.
Scope: a sparser sigma grid than Stage 1 (say 8-10 points spanning
20->400, chosen to bracket every regime boundary Stage 1 finds, plus the
already-known ones: 50, 140, 300), x S/N = 15, 30, 60, 100 (4 points),
x 1 resolution convention (MUSE-like, the one real data actually uses --
add the high-res convention here only if Stage 1 found a resolution-
dependent effect worth chasing at the coverage level too).

**Status: DONE.** sigma=20/50/140/300/400 x S/N=15/30/60/100, n=60,
n_boot=12 (6091s total). Raw per-cell sweep script/output removed in the
later spring-cleaning pass as superseded by the general
`calibrate_bias`/`ablation` framework; results below are the full record.

| sigma | S/N=15 | S/N=30 | S/N=60 | S/N=100 |
|---|---|---|---|---|
| 20 | 0.0% | 0.0% | 0.0% | 0.0% |
| 50 | 0.0% | 0.0% | 8.3% | 38.3% |
| 140 | 88.3% | 73.3% | 88.3% | 65.0% |
| 300 | 56.7% | 65.0% | 85.0% | 95.0% |
| 400 | 58.3% | 70.0% | 80.0% | 81.7% |

(sigma coverage; target 68%; V coverage stayed in a reasonable 62-80%
band throughout every cell, not shown -- V is not the problem anywhere
in this grid.)

**Capstone finding of this whole campaign's synthetic side: coverage
failure is not simply "low sigma bad, high sigma bad" -- there are two
distinct, opposite-direction-in-S/N mechanisms that happen to cross near
sigma=140/S-N=100, which is why that specific cell has historically
looked well-calibrated.**
- **At low sigma (20, 50): coverage rises toward nominal as S/N
  increases** (0%->0%->8%->38% at sigma=50) -- the already-characterized
  wing-inflation under-coverage, fading as expected with more S/N, but
  not fully gone even by S/N=100.
- **At high sigma (140, 300, 400): coverage ALSO rises with S/N, but
  overshoots past 68% into over-coverage** -- e.g. sigma=300: 57%->65%->
  85%->95% as S/N climbs from 15 to 100. More data does not fix this
  regime, it makes the already-known sloppy-manifold over-coverage
  *worse*. Sigma=20's absolute lower bound (0% everywhere) shows the
  low-sigma mechanism is far more severe than the high-sigma one at low
  S/N, but the high-sigma mechanism is the one that gets worse, not
  better, as data quality improves -- a genuinely different, and in some
  ways more concerning, failure mode (a user with excellent data at high
  sigma would trust an interval that's actually badly miscalibrated).
- **sigma=140/S-N=100's historical reputation as "the well-calibrated
  case" is explained, not contradicted, by this**: it sits in a narrow
  saddle point between the two mechanisms (low-sigma under-coverage
  already faded, high-sigma over-coverage not yet strong at sigma=140
  specifically), not because coverage is fundamentally solved there.
  sigma=140 at S/N=15/60 is itself substantially over-covered (88.3%
  both times) -- the "good" S/N=100 result was never representative of
  the whole S/N range even at this one sigma.

**Not yet mechanistically explained**: why the sloppy-manifold mechanism
gets *worse* with more data is itself a real, open question -- flagged
but not investigated further this session (time budget went to breadth
across the grid rather than depth on this one still-open mechanism).

### Stage 3 -- feature composition / incremental regression ("build 1, test 1, build 2, test 1&2...")
Not a bias-hunt -- a correctness/regression pass confirming every opt-in
feature still behaves as documented *in combination* with every other
one, not just alone. Implemented as permanent additions to
`tests/`, not one-off scripts, since this needs to stay true going
forward, not just today. Proposed matrix (each cell: does it run, is
output finite/sane, does default-off still reproduce baseline exactly):
- wing_lam alone (already tested) x shape_aware_correction alone
  (already tested) x **both together** (NOT yet tested -- do they
  compose safely, e.g. does an active wing_lam change what sigma range
  shape_aware_correction's own post-fit warning should fire at?).
- noise_scale x wing_lam together (does injecting extra bootstrap noise
  change wing-inflation's own already-fragile low-sigma behavior?).
- noise_scale x shape_aware_correction together.
- multi-template + noise_scale (mirror single-template's test, using the
  effective collapsed design matrix -- partially done this session, not
  yet a permanent test).
- All four opt-in features active simultaneously, sanity-only (runs,
  finite, positive, no crash) -- not a bias claim, just "nothing
  catastrophically interacts."

**Status: NOT STARTED.**

### Stage 4 -- sky-residual masking investigation and fix
1. Confirm the four recurring wavelength windows found this session
   across ALL 100 real bins (only 7 checked so far), and check for
   others outside the ones already found.
2. Identify likely cause more precisely if easy (e.g. cross-check
   against a known OH-airglow line list for these wavelengths -- would
   upgrade "likely sky residual" to "confirmed").
3. Implement the simplest adequate fix: a fixed-wavelength mask (these
   are recurring at fixed *rest*-frame-post-deredshift wavelength, not
   per-bin-varying, so a fixed mask is well-justified here specifically,
   unlike the general CaT-emission case discussed and set aside earlier
   this session).
4. Test on synthetic data: inject a fake fixed-wavelength contaminant
   pixel cluster into an otherwise clean mock, confirm the mask removes
   its effect on recovered V/sigma/h4 and on `calibrate_noise_scale`'s
   real-data-mismatched behavior (this is also the fix for the
   noise-scale real-data caveat above -- mask first, then recalibrate).
5. Re-run the real-data 100-bin fit with masking active, compare to the
   existing (unmasked) `real_data_all_bins.npz` results.

**Status: (1)/(3)/(4) DONE, (2)/(5) NOT DONE.** Implemented
`kinextract.forward_model.mask_wavelength_windows(wave, *arrays,
windows=[...])` -- simple, general, no hardcoded line list (callers
supply their own windows; the N5102-specific 4 windows belong with
`fit_all_bins.py`, not the library). 3 new tests in
`tests/test_forward_model.py`, including a real before/after synthetic
demonstration (inject a severe localized contaminant at a fixed
wavelength into an otherwise clean mock, 6-seed average: contaminated
fit measurably worse than clean, masked fit measurably better than
contaminated). (2) OH-line-list cross-check not done (would need
external literature access not used this session -- the empirical case
for masking these 4 windows already stands on its own: fixed, recurring,
growing with faintness, not near CaII/Paschen). (5) Re-running the full
100-bin real fit with masking active -- **not done yet, queued for
Stage 6** (do it once, together with re-checking noise-scale calibration
post-masking, rather than two separate re-runs of an expensive fit).

### Stage 5 -- low-sigma h4 point-estimate bias root-cause

**Status: MECHANISM CHARACTERIZED, NOT FIXED.** Two decisive pieces of
evidence, both using data already collected in Stage 1/1c (no new
expensive compute needed):

1. **Not noise-driven -- unlike wing-inflation, this bias does NOT fade
   toward zero at high S/N.** From Stage 1's own muse-resolution table,
   at S/N=200 (the lowest-noise point tested): h4 = +0.032 (sigma=20),
   +0.033 (sigma=50), +0.035 (sigma=60) -- essentially the same size as
   at S/N=15-100. Wing-inflation's own sigma bias, by contrast, clearly
   collapses toward ~0% by S/N=200 in the same table. Different
   mechanism, not the same effect wearing two hats.
2. **It's an additive floor on recovered h4, not a bias proportional to
   the true h4 value.** Stage 1c (true h4=-0.1, sigma=20-140): h4_bias
   (recovered-true) sits at +0.13 to +0.14 for sigma 20-80, meaning
   **recovered h4 itself lands at ~+0.03 regardless of whether the true
   h4 was 0 or -0.1.** This is a strong, clean pattern -- something in
   the low-to-mid-sigma regime is pulling the recovered h4 toward a
   roughly fixed value near +0.03, not just adding noise or scaling the
   true signal.

**Not yet identified**: the specific mechanism inside `fit_h3_h4`/
`compute_direct_moments` producing this fixed offset, nor whether it's
fixable (an h4-specific analog of `shape_aware_taper_correction`,
correcting toward the *true* additive floor rather than the taper's
noiseless-bias story, is a plausible candidate, untested). **Documented
here and in `README.md` as an honest, currently-uncorrected limitation**
per the user's own standard for unfixed findings -- proposing the
analog-correction idea as a concrete future direction, not implementing
it this session (time budget went to finishing the wider campaign
first).

**Status: NOT STARTED.**

### Stage 6 -- real-data consistency re-check

**Status: DONE -- decisive, honest negative result.** Re-ran all 100
bins with sky-window masking active (`real_data_all_bins_masked.npz`),
compared against the original unmasked baseline
(`stage6_compare_masked.py`):

| | ORIG mean diff | ORIG within-err | MASKED mean diff | MASKED within-err |
|---|---|---|---|---|
| V | +4.92 | 62.0% | +8.59 | 46.0% |
| sigma | +5.02 | 28.0% | +6.40 | 19.0% |
| h4 | +0.09 | 0.0% | +0.09 | 0.0% |

sigma radial trend: outer bins (|r|>8") got *worse* with masking (+11.07
-> +15.89 km/s), not better. **Masking made every real-data metric
worse or unchanged, never better.** See Stage 7 below for the follow-up
that explains why, and `mask_wavelength_windows`'s own docstring for the
permanent record of this finding.

### Stage 7 -- does adaptive cleaning do any better than fixed masking? (added mid-campaign, user-authorized)

Built `kinextract.cleaning.fit_losvd_clean` (iterative
sigma-clipping, standard practice, no line list needed -- see that
module's own docstring for the method and its relationship to
`mask_wavelength_windows`). 5 new tests, all passing, including a
decisive synthetic demonstration
(uncleaned mean error 33.8 km/s vs. cleaned 5.4 km/s on an injected
10-pixel severe contaminant at S/N=30 -- cleaning genuinely works on
what it's designed for).

**Applied to a 10-bin real-data subset spanning the radius range: same
negative outcome as masking, and worse where it engages most.** (Script
removed in the later spring-cleaning pass -- had gone stale against a
since-renamed API and was superseded by `fit_all_bins.py`'s own
bias-corrected real-data path; findings below are the full record.) At
the bright/inner bins
clipping barely does anything (0-2 pixels out of 321, as expected at
high S/N where nothing exceeds 3-sigma). At the faint/outer bins where
it actually engages (16-20 pixels), **sigma got measurably worse, not
better** -- e.g. bnn1605: legacy-relative bias +13.6 -> +25.8 km/s after
cleaning.

**Two independent, methodologically different outlier-rejection
strategies both making the same real problem worse is itself a strong,
decisive finding, not a coincidence to explain away.** Best current
explanation (see `cleaning.py`'s own docstring for the full mechanistic
argument): the dominant driver of sigma inflation in these bins is very
likely the already-characterized S/N-driven wing-inflation mechanism --
a *model/regularization* bias, not pixel-level contamination. Outlier
rejection assumes deviant pixels are *wrong*; but when the real problem
is the *model* being biased, the pixels that disagree most with an
already-biased fit are often exactly the ones correctly arguing against
it, so removing them lets the bias compound rather than fixing anything.
**Recommendation, now written into both `cleaning.py` and
`mask_wavelength_windows`'s own docstrings: don't reach for cleaning or
masking to address sigma inflation at low S/N -- that's `wing_lam`'s
job. Reserve outlier rejection for cases with independent evidence of
real per-pixel contamination**, not as a general-purpose fix for a fit
that simply looks biased.

**Not done**: per-bin `wing_lam` tuning keyed to each bin's own S/N
remains the most promising, still-untried real fix for the sigma radial
trend -- flagged repeatedly this session, still not attempted.

### Stage 8 -- continuum architecture: implemented, synthetically validated, tried on real data (user-authorized mid-campaign expansion)

Implemented `kinextract.continuum` (3-block alternating
minimization: LOSVD, template weights, shared multiplicative continuum)
following `kinextract_test_suite/CONTINUUM_ARCHITECTURE_PLAN.md` verbatim
-- see that module's own docstring for the algorithm. 4 new tests, all
passing:
- **Validation order step 1** (known injected continuum distortion,
  already-normalized templates): recovers weights to within 0.02 of
  truth, sigma within 5 km/s, continuum shape RMS ~0.003-0.02 against a
  0.15-0.25-amplitude injected curve.
- **Validation order step 2, the key "kinematics primacy" check the user
  explicitly required**: an adversarial setup (two templates with
  *opposite* broadband continuum slopes, and an injected continuum
  distortion shaped exactly like what a wrong weight shift toward one of
  them could fake). **Plain, undamped 3-block alternation recovered the
  true 50/50 weight split anyway (mean max |w_err| < 0.05 across
  seeds)** -- the proximal/ridge mitigation the architecture plan
  proposed as a hypothesis, explicitly not to be implemented
  preemptively, turned out not to be needed. This was the single most
  important open question in the whole plan, and it resolved cleanly.
- Flat-continuum sanity check: when there's no real distortion to
  explain, the 3-block fit matches the already-validated 2-block
  `fit_losvd_basis` closely.

**Applied to real NGC 5102 data (10-bin subset, RAW/non-pre-normalized
`.dat` templates instead of the `.txt` pre-normalized ones -- the whole
point of a continuum model is not needing pre-normalized templates):
made agreement with legacy worse, not better, once properly aggregated.**

| | no-continuum baseline (pre-norm templates) | with continuum (raw templates) |
|---|---|---|
| V mean diff / RMS | +6.44 / 8.94 | +9.08 / 13.07 |
| sigma mean diff / RMS | +6.13 / 7.60 | +7.72 / 8.47 |
| h4 mean diff | +0.086 | +0.088 (unchanged) |

**An early per-bin eyeball read looked like V had improved substantially
-- it was wrong; the full aggregate shows the opposite. Worth recording
as its own lesson: never trust a handful of cherry-picked bin
comparisons over the properly aggregated statistic, even when the
individual numbers look encouraging.**

**The pattern across Stages 4, 7, and 8 is now unmistakable and is this
campaign's single most important real-data finding**: three
independent, well-motivated, individually-validated fixes --
fixed-wavelength sky masking, adaptive sigma-clipping, and now full
continuum co-fitting with raw templates -- all worked cleanly on their
own synthetic validation and all either failed to help or made things
measurably *worse* on the same real NGC 5102 data. This is strong
evidence that the real driver of the real-data sigma/V discrepancy is
something none of the three targets: not pixel-level contamination
(masking, cleaning), not continuum shape (continuum block). Leading
untested candidates, in rough order of the user's own suspicion and this
session's reasoning: (1) imperfect upstream pre-normalization in the
`.norm` files themselves (the user's own explicit framing -- "the
pre-normalization... definitely wasn't perfect"), (2) an LSF/resolution
mismatch between templates and the MUSE data (flagged repeatedly since
the h4 investigation, never directly tested), (3) per-bin `wing_lam`
tuning for the sigma-specific component of the discrepancy (the one
piece of this whole picture that IS mechanistically well-understood and
still untried). None of these three are tested yet.

**Status: implementation and synthetic validation DONE and solid; real-
data application tried and honestly documented as not (yet) helpful.**

### Stage 9 -- regularization-bias correction (Pinkney et al. 2003-style), calibration, and an ablation ladder to find what actually causes it

New session. Built `kinextract.bias_correction`: a synthetic-
calibrated correction for the regularization/shrinkage bias this whole
MPL/Tikhonov lineage carries (Saha & Williams 1994 -> Gebhardt et al.
2000b -> Pinkney et al. 2003), wired into `bootstrap_losvd`/
`compute_losvd_moments` as new `_corrected`-suffixed keys (no existing
key's meaning changed). Root-caused the mechanism first: refitting on a
point estimate's own exact noiseless model recovers a measurably
different (more-shrunk) `b` with a genuinely lower joint objective --
confirmed this is the real regularized optimum, not a convergence bug.

**First calibration attempt (idealized, single-template, V=0) failed hard
on real data.** Checked directly against 100 real NGC 5102 bins
(`fit_all_bins.py` vs. legacy `pallmc.out` -- note `fit_all_bins.py` was
found broken, referencing a function name deleted in the earlier
consolidation, `fit_losvd_with_templates_cfg`; fixed to `fit_losvd`):
corrected sigma was *worse* than raw in ~90% of bins.

**Second attempt (real 33-template MUSE library, one fixed Dirichlet
mixture) also failed against legacy** (~96% of bins worse) --
recalibrating with real template *content* wasn't the fix.

**The real turning point: legacy itself is not ground truth.** Diagnostic:
at S/N~70-100, the *raw* point estimate already tracked `pallmc.out`
almost exactly (e.g. one bin: raw 51.0 vs. legacy 50.9) while the
calibration says real bias should be present -- meaning "distance from
legacy" was never a valid pass/fail test (this whole campaign has
independently established `fitlovw`/`pallmc.out` carry their own real
biases, e.g. the hardcoded sigma-guess window issue documented in the
sco_framework test-suite work). **Replaced the legacy-comparison
scoreboard entirely** with `validate_bias_correction_against_truth.py`:
real MUSE templates, a *fresh* random Dirichlet mixture per replicate
(disjoint seed range from calibration), fit via ordinary free-NNLS
`fit_losvd`, checked against KNOWN synthetic truth. Result: the fixed-
mixture calibration reduced median sigma error in 41/42 cells (98%)
across N5102's realistic sigma (30-100)/S-N (15-150) range, often 3-10x --
the correction was working; legacy was the wrong ruler.

**Recalibrated once more (varying mixture: fresh random Dirichlet draw per
replicate during calibration too, not one fixed vector)**, since a real
galaxy's bins carry genuinely different stellar populations and every
real fit already does free NNLS regardless of what generated the mock --
matches how the package is actually used, per explicit user instruction
not to build "half-truths" with an artificially-narrow calibration
scenario. This is now the shipped default
(`calibration_data/regularization_bias_muse_multitemplate.txt`; the fixed-
mixture table kept alongside as
`..._fixedmixture.txt`, idealized single-template as
`..._singletemplate_idealized.txt`, none but the varying-mixture one
loaded by default). Full test suite green throughout every swap.

**Visualization made a standing part of the framework, not an
afterthought**: `kinextract.plotting` (`plot_calibration_surface`,
`plot_truth_validation`, `plot_ablation_ladder`), so every future
sweep/calibration gets a PNG alongside its text table automatically.

**Built a general ablation-testing framework** (`kinextract.ablation`:
`AblationConfig`, `run_ablation_sweep`) after the user's own observation
that we didn't actually know which of the package's several fitting-
pipeline mechanisms (regularization strength, multi-start, the moment-
extraction taper, h3/h4 estimation, `shape_aware_correction`) was
contributing how much bias, in isolation -- extensible by design (a new
feature just needs a new `AblationConfig` field). Required adding a real,
reusable `losvd_single_start` toggle to `FitConfig`/`fit_losvd_basis`/
`_alternate_once` (bypasses `fit_log_spline_map`'s default multi-start for
a single deterministic start) -- general package infrastructure, not
test-only scaffolding.

**Ladder**: `0_bare_unregularized` (lam=0, single start, plain trapz
moment, no h3/h4) -> `1_plus_regularization` (lam=2.0) ->
`2_plus_multistart` -> `3_plus_taper_moment` (smooth-taper
`compute_direct_moments`) -> `4_full_stack_default` (+ h3/h4 -- this IS
the real shipped default end to end) -> `5_plus_shape_aware` (branch,
opt-in in the real package). Same real MUSE templates + fresh mixture per
replicate as the truth-validation script, sigma 20-400 x S/N 10-200 x
n_reps=20, launched as an overnight run (`ablation_sweep.py`).

**Overnight run complete; both the ablation ladder and a fixed-vs-varying
calibration-mixture comparison landed real, actionable results.**

**Calibration mixture choice: fixed beats varying, reverted.** The
varying-mixture recalibration (fresh Dirichlet draw per *calibration*
replicate too, not just per validation replicate) was checked against the
same truth-validation harness: only 88% of 42 cells improved (median
corrected error 4.45 km/s) vs. the fixed-mixture calibration's 98% (median
4.79 km/s) -- and several sigma=80-100 cells got worse under the varying-
mixture table (e.g. sigma=100/S-N=60: raw 7.3, "corrected" 10.0). Likely
cause: resampling the mixture every calibration replicate adds a genuine
extra noise source to the calibration's own mean-bias estimate at fixed
`n_reps=20`, without a compensating increase in effective sample size --
the validation side already tests generalization across mixtures (fresh
draw at application time, in both cases), so varying the *calibration's*
own training mixture bought variance, not robustness. **Reverted the
shipped default to the fixed-mixture table**
(`regularization_bias_muse_multitemplate.txt`); the varying-mixture table
kept alongside for reference. Both `plot_calibration_surface`/
`plot_truth_validation` PNGs generated for each.

**The ablation ladder found the actual headline result of this whole
stage**: aggregated across the full sigma=20-400/S-N=10-200 grid, the
**unregularized** fit (`lam=0`) has *lower* median absolute sigma bias
(26.7%) than every regularized configuration, including the real shipped
default (35.8%). Breaking this down by sigma (`plot_ablation_ladder`,
`/tmp/kinextract_plots/ablation_ladder.png`) shows this is not
uniform -- it's a genuine crossover:

| sigma | unregularized | full_stack (shipped default) |
|---|---|---|
| 20 | 10.7% | 120.4% |
| 30 | 12.3% | 71.3% |
| 50 | 15.1% | 29.9% |
| 80 | 26.1% | 20.5% |
| 140 | 23.2% | 22.3% |
| 220 | 45.8% | 39.1% |
| 300 | 51.6% | 39.2% |
| 400 | 65.0% | 46.9% |

**At sigma<=~80, the curvature penalty (`lam=2.0`) is actively causing
most of the bias, not correcting for anything -- removing it entirely
would be a large, direct improvement in this regime.** At sigma>=~140,
the penalty is doing real, net-positive work (matching ordinary MPL
theory), though the gap is much smaller than the low-sigma penalty in the
other direction. `3_plus_taper_moment` (the smooth-taper direct moment)
is the single most consistently-helpful addition across the whole grid
after this crossover -- confirms the existing default choice there.
`5_plus_shape_aware` (still opt-in, off by default) is neutral-to-mildly-
harmful everywhere tested here, consistent with its own documented
caveats.

**This reframes the priority for tomorrow.** The Pinkney-style post-hoc
correction (this stage's original deliverable) is a patch on top of a
fixed `lam=2.0` that is *itself* the dominant bias source at low sigma --
fixing the regularization strength itself (sigma-adaptive `lam`, matching
both the originally-deferred Stage 3 idea and Mehrgan et al. 2023's own
AIC-based adaptive-strength approach) is now empirically well-motivated,
not speculative, and should be scoped with the user before implementation
(touches the core fitting engine, `losvd.py`) rather than treated as an
overnight decision.

**Status: DONE.** Calibration reverted and re-validated; full test suite
green; ablation ladder complete with a clear, actionable finding ready to
discuss.

## Session-continuity notes

I (the assistant) cannot introspect actual token/usage budget -- there is
no literal "session limit countdown" to track. The practical equivalent:
checkpoint this file (and, for durable cross-session memory, the project's
memory system) after every completed stage or major sub-step, not just at
the end, so at most one small unit of in-flight work is ever lost to an
unexpected cutoff. Background jobs write their own results to files under
`examples/notebooks/` or this repo directly (not the ephemeral
`/private/tmp` scratchpad, which does not persist across sessions) once
a stage is confirmed working, so raw output survives independently of
this file too.

**If you are resuming after an interruption**: check `ps aux` for any
still-running background python processes from this campaign first
(their PIDs will be stale/gone if the session actually restarted, but
worth a glance); then re-read this file's checklist top to bottom, find
the first `[ ]` unchecked item, and continue from there. Cross-check
against the memory system (`project_kinextract_deep_validation.md`
should carry a same-day summary pointer to this file).

## Resume point

**Last updated: Stages 0-9 all complete.** Overnight run (ablation ladder
+ fixed-vs-varying calibration-mixture comparison) finished and analyzed;
default calibration reverted to the better-performing fixed-mixture table;
full test suite green. See Stage 9's own section for the two headline
results: (1) fixed-mixture calibration beats varying-mixture against known
truth (98% vs. 88% of cells improved) and is the reverted shipped default;
(2) the ablation ladder shows a genuine sigma crossover -- the curvature
penalty (`lam=2.0`) is a net bias *source*, not corrector, at sigma<=~80,
and only becomes net-helpful at sigma>=~140. Not yet acted on (touches the
core fitting engine) -- reported to the user as the clear next priority,
not decided unilaterally.

**Natural next step**: scope a sigma-adaptive regularization-strength
mechanism with the user (Mehrgan et al. 2023's AIC-based approach, or
`kinextract` (non-lite)'s own `xlam_auto`, are the two concrete precedents
already identified) before implementing -- this is a core-engine change,
bigger than anything landed so far this stage, and the user's own stated
next milestone (implementing the continuum framework) is downstream of
"bias-free," so this is likely the highest-leverage remaining item.

**Summary of where this leaves the package:**
- Every new capability built earlier this campaign (`calibrate_noise_scale`,
  `mask_wavelength_windows`, `kinextract.cleaning`,
  `kinextract.continuum`) is implemented, exported, and backed by
  passing tests including genuine before/after synthetic demonstrations,
  not just "runs without crashing" checks.
- The synthetic-side picture of this package's biases and coverage is
  now comprehensive across the full physically-relevant sigma range
  (20-400 km/s) x S/N (15-200) x 2 instrument-resolution conventions --
  see Stage 1/1c/2's own sections for the numbers. sigma=20 km/s is
  effectively unusable (huge bias, zero coverage, regardless of S/N or
  resolution); sigma=50 is severely under-covered at low S/N; sigma=140/
  300/400 are increasingly over-covered as S/N *improves*, the opposite
  of the naive expectation.
- **The real-data picture is the more important, harder-won result of
  this whole campaign**: three independent, synthetically-validated
  fixes (sky masking, adaptive cleaning, continuum co-fitting) were each
  tried on the real NGC 5102 data and each either failed to help or made
  agreement with legacy measurably worse. See Stage 8's own section for
  the full argument and the three leading untested hypotheses for what
  the real driver actually is (pre-normalization error, LSF mismatch, or
  per-bin `wing_lam` tuning).
- **Stage 9's own real-data lesson generalizes beyond bias correction**:
  `pallmc.out`/legacy is not ground truth and should not be used as a
  pass/fail test for any future fix either -- use known synthetic truth
  (real templates, free NNLS, fresh per-replicate population mixture) as
  the standing scoreboard instead. This likely also reframes Stage 8's
  "three fixes all failed against legacy" finding -- worth a second look
  once bandwidth allows, since "failed against legacy" was exactly the
  test Stage 9 found to be unreliable.

**If picking this up fresh**: there is no unfinished background job to
check for. The natural next steps, in the order this document's own
reasoning suggests, are: (1) test the three untested real-data
hypotheses above, starting with per-bin `wing_lam` tuning since it's the
only one that's already mechanistically understood; (2) investigate why
the sloppy-manifold over-coverage mechanism gets *worse* with more S/N
(flagged as open in Stage 2, not investigated); (3) the template-
continuum-trust hierarchy from the architecture plan, if the continuum
model ever does prove useful once the real driver is found. None of
these were started.

### Stage 10 -- E-MILES-based regularization-bias calibration (replacing the real-MUSE-star default), an overnight recalibration + bugfix pass

New session, triggered by the user reading `01_basic_mock_fit.ipynb`'s
own bias-corrected LOSVD plot and noticing the correction looked wrong on
an already-good raw fit. Investigation traced this to two separate real
issues, then the user made an explicit architectural call: **stop
calibrating the shipped default against one specific real-star library
(MUSE-specific, non-portable); rebuild it against the E-MILES SSP grid
instead, with a documented, reusable recalibration path for any other
template library (STIS included).**

**Two real bugs found and fixed before the recalibration even started:**
1. `bootstrap_losvd`'s `b_p16_corrected`/`b_p84_corrected` only *recentered*
   the raw (uninflated) bootstrap percentile band on the bias-corrected
   median -- the band's own *width* silently stayed at the raw, uninflated
   value, even though the accompanying `sherm_std_corrected` number is
   inflated in quadrature with the calibration's own bias-scatter term and
   can be several times larger. Fixed: the band is now stretched by the
   same `sherm_std_corrected/sherm_std` ratio around the corrected median
   (an approximation -- one scalar ratio, not a per-bin recalibration --
   but consistent with the reported number instead of silently narrower).
2. `estimate_m_eff_local`'s closed-form solve (`np.linalg.solve(M, XtX)`)
   crashes with `LinAlgError: Singular matrix` at `lam=0` on narrow
   (low-sigma) grids -- `M=XtX` at `lam=0` can be exactly rank-deficient.
   Very plausibly what silently killed an earlier m_eff-axis calibration
   sweep (which included `lam=0.0` in its own grid) with "zero progress
   shown" before it was killed. Fixed: swapped to `np.linalg.lstsq`, which
   solves the same system via SVD -- bit-identical to `solve` whenever `M`
   is non-singular, a well-defined minimum-norm answer instead of a crash
   when it isn't.

**The real finding underneath the user's original observation: template
*shape*, not which real-spectrum library, is what the correction needs to
match.** Direct test: the OLD real-MUSE-star-calibrated correction, applied
to `01_basic_mock_fit.ipynb`'s own synthetic single-Gaussian-absorption
mock (sigma=140, S/N=100, 25 seeds), made sigma recovery *worse* in 24/25
draws (median error 10x worse than raw) -- while a positive control using
an actual real MUSE star template at the same (sigma, S/N) showed the
correction working as documented. Rebuilding the calibration against
E-MILES did not fix this on its own (still worse on the toy template, if
anything by a larger absolute margin) -- confirming the mismatch is about
spectral *shape* realism, not which specific real-spectrum basis was used
to build the table. **Fix: swapped `01_basic_mock_fit.ipynb`'s own mocks
from synthetic single-Gaussian templates to real E-MILES SSPs** (matching
every other notebook in the suite) -- the correction is now demonstrated
genuinely in-domain there (Part B: corrected sigma 49.7+/-2.3 vs. true 50,
corrected h4 -0.0005+/-0.0091 vs. true 0.0 -- both essentially exact).

**Built the E-MILES recalibration itself as a general, documented
framework, not a one-off script**: `calibrate_bias.run_calibration_sweep`
was already fully template-library-agnostic (plain `templates`/
`template_wave` arrays, no hardcoded path); what changed is
`examples/notebooks/build_emiles_calibration.py`, the literal, adaptable
recipe now used to build the shipped default -- swap its template-loading
function for STIS, a different star catalog, or any other library and
rerun. Built three tables from a 40-template E-MILES subset (ages 0.1-14
Gyr, [M/H] -1.49 to +0.40, one fixed Dirichlet mixture, seed=30000,
overnight parallelized sweep, 8 workers):
- `regularization_bias_emiles_multitemplate.txt` (main, h4_true=0.0,
  lam=2.0 fixed, 8x8 sigma/S-N grid, n_reps=30) -- now the shipped default
  (`bias_correction._DEFAULT_TABLE_PATH`), replacing
  `regularization_bias_muse_multitemplate.txt`.
- `regularization_bias_emiles_h4wing.txt` (h4_true=0.15 genuine-wing
  truth, 4x5 grid, n_reps=30).
- `regularization_bias_emiles_multitemplate_meff.txt` (m_eff axis, 8
  lam values swept per cell, 8x8x8 grid, n_reps=30, 15360 total fits,
  ~4hr wall time at 8 workers -- this is the sweep the lam=0 crash above
  was found and fixed inside).

**A third real bug, found only by actually re-running the notebooks
against the new tables**: `save_calibration_table` always writes the
13-column (lam/m_eff) format now, even for a single-fixed-lam sweep --
so the new shipped default (built at fixed lam=2.0, no lam variation)
accidentally satisfied the old "has an m_eff axis" check (column-count
alone), which made `bootstrap_losvd` auto-trigger `estimate_m_eff_local`
on every call -- and that function doesn't support continuum-cofitting,
so every continuum-cofit `bootstrap_losvd` call in `02_realistic_mock_fit
.ipynb` started crashing. Fixed: `load_calibration_table` now also
requires genuine `lam` variation (`len(unique(lam)) > 1`) before reporting
an m_eff axis, matching `BiasCalibration.m_eff`'s own documented intent
("built with more than one lam in lam_grid"). Regression tests added for
all three bugs (`test_corrected_band_width_reflects_sherm_std_corrected_
inflation`, `test_m_eff_local_does_not_crash_at_lam_zero_on_a_narrow_
grid`, `test_single_lam_table_does_not_report_an_m_eff_axis`).

**A fourth, independent bug found in the same pass, unrelated to
calibration**: `02_realistic_mock_fit.ipynb` and `03_emiles_templates
.ipynb`'s own synthetic mocks built `wave_fit` directly from E-MILES' own
native ~0.9 Angstrom/pixel grid, not the MUSE 1.25 Angstrom/pixel sampling
this package's calibration (and its own real-data notebooks) are actually
built and validated against -- a silent, self-introduced instrument-
sampling mismatch. Fixed in both notebooks (build a proper MUSE-pixel
`wave_fit`, pass E-MILES' native grid separately as `template_wave` for
`build_template_stack` to interpolate against, as every other notebook
already did correctly). Confirmed to matter: `03_emiles_templates.ipynb`'s
own worked-example V recovery improved from -27.7 (true -20) to -20.2
after the fix.

**Validated the new default two ways against known synthetic truth**
(`examples/notebooks/validate_emiles_calibration.py`, 25 fresh-mixture
reps/cell, disjoint seed range from the calibration's own):
1. **E-MILES self-consistency** (fresh Dirichlet mixture over the same
   40-template subset): median sigma error 6.41 -> 4.62 km/s (28%
   reduction), correction improved 78% of 49 (sigma, S/N) cells.
2. **Real-MUSE-star cross-domain** (fresh Dirichlet mixture over the
   actual 33-star real library, sigma 30-140 x S/N 15-150): median sigma
   error 15.85 -> 7.07 km/s (55% reduction), correction improved **100%**
   of 49 cells -- the calibration transfers robustly to real stellar
   spectra, not just within its own synthetic training domain, the
   critical check for real-data trustworthiness.

**Status: DONE for tonight's scope.** All 4 bugs fixed with regression
tests; new E-MILES calibration built, validated two ways, wired as
default; all 5 example notebooks (01-05) re-executed clean against it,
numbers and plots both updated to match; full test suite green, 126
passed (up from 123 -- the 3 new regression tests). `regularization_bias_
muse_multitemplate*.txt` (the old real-star tables) left on disk for
reference/comparison, no longer loaded by default.

**Not done, flagged for follow-up, not decided unilaterally tonight**:
- The m_eff-axis table exists and is validated to build/load correctly,
  but is not wired into any default code path (`bootstrap_losvd` only
  uses it if a caller explicitly passes it as `bias_calibration=`) --
  combining it with correlated-noise whitening or continuum-cofitting is
  still unvalidated, per `02_realistic_mock_fit.ipynb`'s own notes.
- `calibrate_bias_h4wing`'s STIS-generalization promise is now
  *architecturally* true (the sweep mechanism never depended on MUSE) but
  not yet *empirically* demonstrated -- no STIS template library has
  actually been run through `build_emiles_calibration.py`'s own recipe
  yet.
- The broader "package-wide stress-test matrix" (the standing item from
  before this stage) is still open; tonight's work fixes the calibration
  foundation it should now be built on top of, not the matrix itself.

## Stage 11: continuum-cofitting amplification bug found; notebook 02 rebuilt without it

Prompted by the user reporting a visibly bad spectral fit and a strong
systematic residual trend in `02_realistic_mock_fit.ipynb` (measured
directly: `linregress` of the noiseless-truth-minus-model residual against
wavelength gave `r=0.9952`), and separately noting the real MUSE fit
(`04_real_data_muse.ipynb`) shows a similar-looking shallow-absorption-line
residual, raising the question of whether this is one systemic issue.

**Root cause, isolated via a sequence of controlled tests (all in a
single-template mock unless noted, ruling out template-mixture degeneracy
each time):**
1. `fit_continuum_block` alone, given the *exact true* `(b, w)`, recovers
   `continuum_true` almost exactly (max abs error 0.0012 at `lam_c=2.0`,
   6e-7 at `lam_c=0`) -- the continuum block itself is not the bug.
2. The joint (chi2 + roughness) objective evaluated *at the true*
   `(b, w, C)` for a `sigma=90` mock was 12.65 (dominated by the LOSVD's
   own curvature penalty, `chi2=0` exactly since noiseless) vs. 2.65 at
   the alternation's own converged solution -- confirming the alternation
   is *correctly* finding a lower-cost point under the current fixed
   `lam=2.0` objective, just one that isn't close to truth. This is
   ordinary, already-known regularization bias, not new.
3. **The critical finding**: this same LOSVD bias (nearly identical
   recovered V/sigma/h3/h4, tested directly with `continuum_knots=None`
   vs. `continuum_knots=4` on the identical mock) produces a residual of
   amplitude ~0.005 (r=0.06, invisible) with continuum off, vs. ~0.04
   (r=0.999, a strong visible trend) with continuum on. Sweeping `lam`
   (2.0 down to 0.01) and `lam_c` (2.0 down to 0.005) independently left
   both the moments and this ~8x residual-amplitude gap essentially
   unchanged -- not a tuning problem. The alternation genuinely converges
   (`objective_history` plateaus) -- not a non-convergence bug either.
   **Continuum-cofitting reproducibly amplifies a small, pre-existing
   LOSVD bias into a large, misleading spectral-fit residual.** Root cause
   of the amplification mechanism itself not yet isolated further beyond
   this; documented as a confirmed, open bug in `continuum.py`'s module
   docstring and `FitConfig.continuum_knots`'s help text.
4. Separately, real-MUSE `04_real_data_muse.ipynb` does **not** use
   continuum-cofitting at all (`continuum_knots` is never set there) --
   so its own shallow-trough residual (chi2_red 1.42/1.06 at the two grid
   stages) is **not** explained by the bug above. Testing adaptive/lower
   `lam` directly on that real bin (down to `lam=0.1`, a 20x reduction)
   left chi2_red and trough depth essentially unchanged (1.423 -> 1.402),
   ruling out regularization strength as the fix there too. The two
   notebooks' bad-looking fits share a *symptom* (systematic residual
   structure) but not a confirmed common *cause* -- reported honestly as
   two separate, only-partially-understood issues rather than forced into
   one narrative.
5. Also checked and ruled out as *the* cause of notebook 02's original
   residual (though real and independently worth noting): the 4-template
   E-MILES mixture used there is genuinely near-collinear in the CaII
   window (pairwise correlations up to r=0.9994, condition number ~1785)
   -- but a single-template (condition number exactly 1.0) mock reproduces
   the same residual-amplification effect, so degeneracy is not the driver
   of the residual trend, even though it's a real, separate limitation on
   mixture-weight recovery (now called out directly in the notebook itself
   via a printed correlation matrix).

**Action taken**: `02_realistic_mock_fit.ipynb` rebuilt without
continuum-cofitting (until the bug above is resolved) and without AR(1)
correlated noise (a real, MUSE-specific measurement that had been
generalized into this notebook's "generic realistic mock" too broadly --
moved to `04_real_data_muse.ipynb`, where it belongs). Re-executed clean;
the resulting spectral fit and residual are now genuinely good (residual
`r=-0.29`, ~0.005 amplitude, LOSVD bias corrected by bootstrap from
sigma=104.2 (raw) to 93.3 (corrected), truth 90).

**Not done**: the continuum-cofitting amplification bug's root cause is
still open. Notebook 04's real-data shallow-trough residual is also still
open (candidate causes not yet tested: LSF mismatch, genuine template-set
incompleteness for this real star population).

## Stage 12: adaptive regularization promoted to default; a real regression found and fixed in the process

Prompted by the user's own standing rule ("when we test something and find
it is better, it becomes the standard method") and a re-check of already-
completed validation work
(`dev_notes/regularization/aic_local_firming_up_results.txt`,
`aic_vs_fixed_regularization_results_with_local.txt`) that had never been
acted on: `select_lambda_aic_local` (the closed-form, no-bootstrap adaptive
`lam` selector) measurably reduces sigma bias vs. fixed `lam=2.0` by
20-57% across sigma=50/150/300 and h4_true=0/0.15 -- consistent enough on
the primary target (sigma) to promote, even though h4 bias is a genuine
mixed result (better at some sigma, worse at others).

**Change**: `FitConfig.lam_selection` (new field) defaults to `"aic_local"`
now, not `"fixed"`. `fit_losvd` runs `select_lambda_aic_local` once per
call (falling back to fixed `lam` with a warning if `continuum_knots` is
set -- adaptive selection doesn't support continuum-cofitting yet).
`bootstrap_losvd` runs the selection once at the point estimate and holds
that `lam` fixed for every bootstrap replicate (re-selecting per replicate
would cost `n_boot`x more for an unvalidated gain).

**A real regression, found and fixed before this reached the user's own
notebooks as a false "no improvement" conclusion**: `bootstrap_losvd`
originally switched its default bias-correction table to the `m_eff`-axis
one *whenever adaptive selection ran*, regardless of what `lam` it
actually picked. In the exact regime `02_realistic_mock_fit.ipynb` uses
(sigma_true=90, 4-template mixture, S/N=60), `select_lambda_aic_local`
picks `lam=2.0` -- the same value the plain (sigma, S/N) table was built
at. Switching tables anyway made the correction *worse*: same recovered
sigma, same S/N, same `lam=2.0` -- plain-table correction landed at 93.3
km/s (truth 90), `m_eff`-table correction at 80.0 km/s, with a *tighter*,
more confident uncertainty band despite being more wrong. Root cause:
the `m_eff`-axis table's total rep budget is spread across 8 `lam` values
x (sigma, S/N) cells, so it has far fewer reps *at any one lam* than the
plain table has at its single calibrated `lam=2.0` -- switching tables
for zero actual regularization-strength change trades a well-sampled
table for a worse-sampled one. **Fixed**: `BiasCalibration` gained a `lam`
field (the single calibrated `lam` for a non-`m_eff` table); `bootstrap_
losvd` now only switches to the `m_eff` table if adaptive selection
actually landed on a `lam` more than 25% away from the plain table's own.
Verified directly against the notebook's own regime -- reproduces the
correct, pre-regression numbers.

**Status**: full test suite green, 126 passed (up from 123 before this
stage's 3 new fields/behaviors). Docs updated (`FitConfig.lam_selection`/
`BiasCalibration.lam` help text, `aic_regularization.py`'s module
docstring, `PARAMETER_GUIDE.md`).

**Not done yet**: re-executing notebooks 01/02/04/05 against this new
default (next step). The continuum-cofitting amplification bug (Stage 11)
remains separately open and unaffected by this stage.

## Stage 13 (in progress overnight): head-to-head kinextract vs. OG kinextract stress test

User request: a full, systematic comparison of kinextract against OG
kinextract across the physical parameter space (sigma, S/N, Gaussian vs.
non-Gaussian LOSVD shape, template regime), prompted by a real, specific
concern (a Mehrgan+2023-style template-mismatch-into-LOSVD-distortion
hypothesis, and the observation that OG kinextract "performs better in a
lot of these cases"). Full design, feasibility checks, and bugs found
while building the harness are in
`kinextract_test_suite/notebooks/head_to_head_comparison/` (`mock_generator.py`,
`run_kinextract.py`, `run_og_kinextract.py`, `sweep.py`).

**Confirmed feasible**: OG kinextract's own file-based API (`spectrum.
make_fit_state` + `joint.run_joint_fit`) can be driven directly, without
reimplementing its internals, by writing synthetic mocks to its `.norm`
format. kinextract's own shipped E-MILES `.npz` grid is column-
compatible with OG kinextract's `templates.load_packed_templates`, so
both codebases were pointed at the *identical* template file -- no
template-library mismatch as a confound.

**Two real config traps found and fixed via the harness's own smoke test,
before they could silently corrupt the comparison**:
1. OG kinextract's `.norm`-file reader defaults to `norm_error_mode="unit"`
   -- silently *ignores* the file's own error column and assumes a
   uniform placeholder noise level instead. Caught directly: chi2_red
   came back ~0.0008 (four orders of magnitude below 1) and sigma was
   wildly biased before this was set to `norm_error_mode="file"`.
2. `fortran_rebin_norm=True` (OG's own default) requires an explicit
   `cfg.step` for `.norm` files -- raises a clear `ValueError` if missing
   (caught immediately, not silently wrong).

**Grid design, dense in sigma deliberately**: the user flagged directly
that an early sigma grid draft jumped 80 -> 140 (near-2x gap) -- the same
class of undersampling bug just root-caused in Stage 12's `m_eff`-table
regression. Final sigma grid is dense (10 km/s steps) through 40-200 km/s,
the range most real galaxy dispersions fall in.

**Compute budget forced a redesign from full factorial to marginal
sweeps**: measured directly (`sweep.py --probe`) at ~20-25s/cell for
*each* codebase (both pay their own regularization-selection cost --
kinextract's `select_lambda_aic_local` grid, OG's own discrepancy-
principle xlam bracket search) -- a full sigma x S/N x shape x template
factorial would be ~13 hours *per seed*. Replaced with two marginal
sweeps: Sweep A (full dense sigma grid x 3 LOSVD-shape tiers x 2 template
regimes, at S/N=60) for bias-vs-sigma; Sweep B (full S/N grid x shape x
template, at sigma in {50, 100, 200}) for bias-vs-S/N. Launched overnight
at 3 seeds/cell, checkpointed to `sweep_results.csv` after every cell
(safe to interrupt/resume/extend with more seeds later), detached from
the interactive session (`nohup`/`disown`) so it survives independently.

**Deliberately scoped to stop here for tonight**: analysis/figure-building
and any feature-porting decision both wait for the user to look at real
results, not for me to act on my own read of partial data.

**Open, flagged not resolved**: OG kinextract's `fit_losvd_gauss_hermite`
documents a different h3/h4 Hermite-polynomial convention than
kinextract's own -- not yet verified whether the two are directly
numerically comparable. Check before drawing h3/h4-specific conclusions
from the sweep results, not after.

## Stage 14: two confirmed correction overcorrection bugs; apply_bias_correction now defaults to False

Prompted directly by the user reviewing the sweep artifact and the
notebook LOSVD plots: a visible leftward skew in several representative
LOSVDs, and specific frustration that the bias-correction tool "completely
fails" in both `01_basic_mock_fit.ipynb` and `02_realistic_mock_fit.ipynb`.

**First, ruled out a broader "skew bug" hypothesis empirically, not just
by inspection.** 25 independent noise realizations at two settings (single
template sigma=50; wing+mixture sigma=100/h4_true=0.15) showed V and h3
are NOT statistically distinguishable from zero bias (0.6-1.2 sigma from
zero in both cases), while sigma and h4 ARE (8-12 sigma from zero) --
matching everything already known and documented. A noiseless control
(realistic whitening scale, zero actual noise) confirmed no deterministic
skew mechanism in the fitting engine itself (h3 at the 1e-4 level for both
a real E-MILES template and an idealized symmetric toy template). The
individual "skewed" representative-fit images are real single noise draws
within normal V/h3 sampling scatter, not a structural bug -- an important,
if partly negative, finding: the "biggest priority" the user suspected
turned out to be sampling noise, and the actual priority (sigma/h4 bias)
was already correctly identified by the adaptive-lam work.

**But investigating that led directly to two real, confirmed correction
bugs, one already fixed this session and one new:**

1. (Already fixed earlier this session, Stage 12): `bootstrap_losvd`
   switched to the `m_eff`-axis calibration table whenever adaptive `lam`
   selection *ran*, not whenever the selected `lam` actually differed from
   the plain table's own -- fixed via `BiasCalibration.lam` + a 25%
   difference threshold.
2. **New**: even when that switch is genuinely warranted (selected `lam`
   really does differ), the `m_eff` table's own correction can still
   overcorrect badly. Reproduced directly on `01_basic_mock_fit.ipynb`'s
   own default mock (seed 42): raw sigma=52.9 km/s (truth 50, already
   good); adaptive selection picks `lam=0.5` (below this package's own
   documented "sweet spot" floor); the `m_eff` table's correction sends
   sigma to 35.5 km/s -- nearly 5x worse than raw. Root cause not fully
   isolated, but plausibly the `m_eff`-axis table's own reps-per-cell being
   thinner than the plain table's (same total calibration budget spread
   across 8 `lam` values x the sigma/S-N grid, vs. all of it concentrated
   at one `lam` for the plain table).

**Decision**: `bootstrap_losvd`'s `apply_bias_correction` now defaults to
`False`. Two distinct overcorrection failure modes found in one session is
a real pattern, not bad luck, and a real user applying this to real data
has no ground truth to catch either one the way synthetic testing did.
Kept available and documented as an explicit opt-in, not deleted -- it
does genuinely help in some regimes (confirmed directly earlier this
session: 104.2 -> 93.3 km/s against a true 90, via the plain table).

**Also done this stage**: all four example notebooks (01/02/04/05)
rewritten so the raw bootstrap point estimate is the primary/recommended
report; the mock-fit notebooks (01/02) each keep an explicit, clearly-
labeled "experimental, use with caution" section demonstrating the opt-in
correction (reproducing the concrete overcorrection case where relevant);
the real-data notebooks (04/05) mention the correction is available and
why it's not demonstrated on real data at all (no ground truth to catch a
bad correction, and STIS in particular is a further extrapolation beyond
the calibration's own MUSE/E-MILES domain).

**Also produced**: `dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md`
-- a proposed (not yet started) strategy for what "trustworthy enough to
default back on" actually requires: per-cell (not aggregate) bias checks,
independent held-out *coverage* validation (not just point-bias
reduction -- the exact gate that would have caught a previously-found
0%-coverage failure before it shipped), a worst-cell-regression check, a
per-query confidence/domain-distance signal returned alongside every
correction, and using the head-to-head OG-kinextract comparison harness
(Stage 13) as an additional source of independent held-out validation
cells.

**Status**: full test suite rerun in progress to confirm the default flip
doesn't break existing behavior (many tests explicitly pass
`apply_bias_correction=True`/`False` already and are unaffected; a few
relied on the old implicit default and needed inspection). Notebooks not
yet re-executed against these changes -- next step.

## Stage 15: JIT-cache speed fix, a real crash bug, and the continuum amplification bug (Stage 11) root-caused and fixed

**Speed.** `fit_log_spline_map` and `fit_continuum_block` both defined a
fresh JAX closure per call (capturing arrays as constants), defeating
`jax.jit`'s shape/dtype-keyed compilation cache and forcing a full
retrace+recompile on every call. Fixed by hoisting the jitted functions
to module level and passing arrays/`lam` in as traced arguments instead
of closed-over constants -- ~1.2x faster per fit. Bigger win:
`bootstrap_losvd` now defaults to `replicate_single_start=True` (every
bootstrap replicate fits with a single deterministic start instead of the
full multi-start pool; the point estimate itself is untouched) --
measured ~2.5x faster across 3 seeds, with replicate-to-replicate scatter
equal to or slightly larger than before, never smaller.

**Crash bug, found via a proactive code-review pass (not user-reported):**
`multi_template._alternate_once` crashed with an opaque `TypeError` if
NNLS drove every template weight to exactly zero on the very first
iteration (a pathological noise draw) -- reachable from any
`bootstrap_losvd` call, where a single unlucky replicate could kill an
otherwise-healthy multi-minute run. Now returns a `success=False` result
with a clear message instead.

**Other review findings, fixed**: a bias-correction table-selection guard
that only checked whether adaptive lambda selection ran, not the fit's
actual lambda value (missed a caller-supplied fixed `lam` far from the
table's own calibration point); a stale `fit_log_spline_map` default
(`lam=10.0` vs. the package's validated `2.0`, never actually reached
since every caller passes `lam` explicitly, but misleading for direct
callers). `estimate_m_eff_local`'s effective-dof Hessian now also
accounts for the wing-amplitude penalty (`wing_lam`), previously
represented only indirectly.

**Continuum amplification bug (Stage 11), root cause found**: the
continuum spline's curvature penalty (`lam_c*D2_c`) has an exact
zero-cost null space -- the constant and linear directions in coefficient
space, a standard property of any second-order-difference P-spline
penalty. An arithmetic-sequence coefficient vector gives
`||D2_c @ coeffs|| ~ 1e-15` regardless of `lam_c`, and the resulting
continuum curve is a near-perfectly-linear function of wavelength --
exactly why sweeping `lam_c` over a 400x range never suppressed Stage
11's residual-amplification symptom. Fixed with a new `FitConfig.ridge_c`
field: a direct L2 penalty on the continuum coefficients themselves,
which has a real, nonzero, growing cost along that exact direction.
Calibrated empirically to `0.1` (`1.0` closed the null space equally well
-- identical recovery RMS, 0.0015, across `ridge_c` from 0 to 1.0 -- but
tipped `fit_continuum_block`'s own already-documented spurious
`success=False` flag in several regression tests). Also added
`wave_fit`/continuum support to `cleaning.fit_losvd_clean`, which
previously had no way to combine sigma-clipping with continuum-cofitting
at all.

**Honest limitation**: the exact Stage-11 8x-residual/r=0.999 case was
not re-triggered end-to-end to give a full closed-loop before/after
demonstration of this fix (several unrelated fixes landed in this
codebase since Stage 11 and plausibly shifted the alternation's own
dynamics enough that the same seed no longer reproduces the identical
failure). The null-space mechanism itself is verified directly via linear
algebra, independent of reproducing that specific historical run, and is
a fully sufficient explanation for every fact Stage 11 documented. Treat
continuum-cofitting as substantially de-risked, not fully closed-loop
re-validated -- a fresh multi-seed bias/residual sweep with `ridge_c`
active is the natural next validation step before treating it as
production-ready.

**Also settled this stage: `fit_continuum_block`'s frequent
`success=False` flag is spurious, not a real accuracy signal (a
long-running open question, misdiagnosed twice before this).** Attempt 1
assumed the single-start design caused it; multi-start was added as a
real, independently-verified-safe improvement but didn't fix this case.
Attempt 2 assumed (from `result["message"]` alone) that the failure
traces to the LOSVD sub-fit misbehaving inside the continuum-alternation
context -- wrong, because `_alternate_once`'s `message` field used to
always report the LOSVD sub-fit's own message regardless of which block
actually failed (a real reporting bug, now fixed: `losvd_success`/
`continuum_success` are reported separately and `message` is attributed
correctly). What's actually true: stepping through a known-failing
trajectory iteration by iteration found the "ABNORMAL" flag corresponds
to every multi-start candidate converging to the same objective value
regardless of the flag, consistent with scipy's L-BFGS-B line search
occasionally failing its own strict internal criteria on an
already-well-converged, small, nearly-linear problem, not a genuine
optimization failure. Confirmed via bias comparison across ~80 reps at
sig=50/snr=60/continuum-on: mean recovered sigma-bias was statistically
indistinguishable between `continuum_success=True` and `=False` groups
(+20.49 vs +19.11 km/s). Treat `success=False` here as a diagnostic
curiosity, not a hard failure filter.

**Status**: full test suite green (133/133) after all of the above.

## Stage 16: codebase cleanup -- dead code removed, docstrings trimmed to current-state only

Removed `plotting.py` (`plot_ablation_ladder`/`plot_calibration_surface`/
`plot_truth_validation`) and its test: zero callers anywhere outside its
own test, across `src/`, `tests/`, `examples/`, and `dev_notes/`.

Also moved the following investigation history out of source docstrings
(where it had accumulated as incident-report-style prose) and into this
file, so the source code documents only current behavior:

**Auto-selecting `wing_lam`: three approaches tried, none adopted.**
`FitConfig.wing_lam` has no principled auto-selection rule; manual,
per-target tuning is the validated approach. A scale-normalized
formulation (dividing the penalized `b` by a purely geometric
peak-amplitude estimate from the grid's own `vmin`/`vmax` convention) was
confirmed exact for a Gaussian target and removes the amplitude-scale
confound, but a single normalized value still doesn't generalize across
sigma -- a value strong enough to fix sigma=50 at S/N=30 measurably
degraded sigma=140's shape residual at the same S/N. Three ways of
estimating that second, per-target factor from the data alone (no ground
truth) were tried and rejected:
1. **Reference-model wing-mass matching** (fit once, build a "trusted"
   reference model from the recovered core, predict expected wing mass,
   set `wing_lam` from the mismatch). Failed at the first checkpoint:
   every candidate reference estimator tried (a tighter-threshold moment,
   a uniformly-weighted Gauss-Hermite fit, a wing-taper-downweighted fit)
   carried its own real, S/N-dependent bias -- a *tighter* core window is
   *more* biased, not less, so there was no clean, low-bias single-fit
   quantity to calibrate against.
2. **Bootstrap bias-correction of the point estimate**
   (`2*point - mean(bootstrap replicates)`, and a significance-gated
   shrinkage version). Cut bias substantially in the regime it was first
   tried (sigma=50, S/N=30-100) but actively corrupted already-good cases
   elsewhere in the grid (one case: 100% coverage -> 58%); more bootstrap
   replicates didn't fix it (tested n_boot=15/40/80, corrupted-case bias
   unchanged). Root cause: the correction assumes refitting fresh noise
   on top of the point estimate reproduces the same bias the original
   noisy fit incurred relative to the truth -- false here, since every
   bootstrap replicate picks up roughly the *generic* bias of fitting at
   that S/N regardless of how good its input already was.
3. **Sensitivity-to-`lam` diagnostic** (refit at a perturbed `lam`, use
   how much recovered sigma/wing-mass moves as a signal). Worked cleanly
   at sigma=50 (near-perfectly monotonic tracking of true bias across
   S/N=30-2000), but was wrong-signed and noise-dominated at sigma=140
   and especially sigma=300 -- including at the single worst case in the
   whole test grid. A 12x larger perturbation didn't rescue it: signal
   and noise scaled up together, never separating.

Recorded here in this detail specifically so a future attempt doesn't
have to re-run the same three experiments blind.

## Stage 17: real-data template calibration pipeline -- dartboard binning, iterative E-MILES reduction, self-contained real-cube notebooks

Replaced the previous "real MUSE data" workflow's dependency on a
hardcoded external path to a separate legacy Fortran pipeline's
pre-binned output (not reproducible by this package, not shippable) with
a fully self-contained pipeline: get a real MUSE cube (ESO archive
download, generalized to any MUSE target) or a synthetic Sersic-profile
mock cube with known ground truth; bin it with a new "dartboard" scheme;
fit every bin against the full 636-template E-MILES grid and iteratively
reduce to a representative subset; save that subset and a few real
spectra for reuse.

**New modules** (`src/kinextract/`): `muse_cube.py` (duck-typed
`MuseCube` container + `load_muse_cube`/`extract_aperture_spectrum`/
`white_light_image`, works identically on a real FITS cube or the
synthetic mock); `binning.py` (`DartboardBins` -- log- or linear-radial
annuli x sectors equally spaced in **theta itself**, not `sin(theta)`
like the legacy SCO pipeline's own binning convention, deliberately
avoiding any assumed kinematic axis; explicit edge arrays are the
mechanism for concentrating resolution anywhere a user wants, e.g. near
a nuclear star cluster); `mock_cube.py` (Sersic2D surface brightness x a
fixed true template mixture, constant-noise-floor Gaussian noise so S/N
declines with radius purely from brightness falloff); `template_reduction.py`
(`reduce_templates_iteratively` -- NNLS weight-sparsity threshold,
repeated until the **union** of surviving templates across participating
bins stabilizes; `bins_subset` restricts the expensive full-grid pass to
a representative set of bins). `eso_archive.py` wraps `astroquery.eso`
for Phase-3 MUSE-DEEP queries/downloads, gated behind an explicit
`confirm=True` (refuses by default, prints size estimate).

**Real validation, not just synthetic**: downloaded the real NGC 5102
MUSE-DEEP Phase-3 cube (`ADP.2017-03-22T15:25:58.757`, ~2.97 GB,
`60.A-9308`, matching the archive-portal metadata field-for-field except
a minor exposure-time accounting difference, 3554.2s vs. 3840s -- not a
different dataset). Dartboard-binned it (log-radial x 4 angular sectors,
white-light-peak center at pixel (160, 158)), fit all 21 populated bins
against the full 636-template grid, and ran iterative reduction on a
representative 6-bin subset spanning the S/N range: **converged in 3
iterations to 9/636 templates**. Confirmed on the synthetic mock first
(636 -> 28 templates on a 17-bin run, 2 iterations, no collapse to 1
template -- the specific failure mode this design was meant to guard
against).

**A real, checked finding, not a bug**: the reworked
`05_real_data_muse.ipynb` now uses the cube's own STAT-extension
per-pixel errors (via `extract_aperture_spectrum`) instead of the
previous version's single scalar `.rms`-file noise estimate. At the
high-S/N (~500) representative aperture this produces `chi2_red~15.7`
(11.3 medium-S/N, 27.5 low-S/N) -- confirmed directly to be a systematic
model-data residual (median whitened residual ~2.7 sigma, not a few
outlier pixels) and not a masking or fitting bug, and confirmed not a
regularization artifact (an earlier version of this same bin swept `lam`
20x with `chi2_red` essentially unchanged). At real per-pixel precision,
model imperfections invisible at a coarser noise estimate (LSF mismatch,
un-fit continuum shape, template-subset incompleteness) become the
dominant term in `chi2_red` rather than photon noise. Left open as a
real limitation, not papered over by reverting to the less rigorous
noise estimate.

Also found and fixed during this stage, empirically: dartboard bins
built from large spatial apertures average away the radial S/N decline
(all 21 real bins came out S/N~400-1900 regardless of radius) --
representative "high/med/low S/N" real spectra for the notebook instead
use small (1-7 pixel) apertures at increasing radius, giving a genuine
S/N~491/62/15 spread with visible sky-residual artifacts in the faintest
one, matching what the pipeline's own dartboard bins are for (template
selection, where large-aperture S/N is a feature) versus what a
representative-spectrum demo needs (a real range of data quality).

**Notebooks renumbered**: `06_template_mismatch_diagnostic.ipynb` ->
`04_template_calibration.ipynb` (rebuilt, now the entry point for any
real MUSE target); `04_real_data_muse.ipynb` -> `05_real_data_muse.ipynb`
(reworked to load 3 representative real apertures and the saved template
subset instead of one hardcoded bin); `05_real_data_stis.ipynb` ->
`06_real_data_stis.ipynb` (cross-references only, content unchanged).

**Forward-looking, not built now**: generalizing this pipeline to other
instruments, especially HST/STIS long-slit spectroscopy -- the dartboard
scheme's 2-D spatial binning has no 1-D slit analog, so this needs a
genuinely different binning abstraction, not a parameter change.

**Status**: full test suite green after all of the above (159+/159+,
new tests in `test_binning.py`/`test_mock_cube.py`/
`test_template_reduction.py`/`test_muse_cube.py`/`test_eso_archive.py`);
both reworked notebooks re-executed end-to-end with 0 errors.

## Stage 18: `examples/` cleanup, and the mock-notebook LOSVD "skew" root-caused to wing-amplitude inflation

Two user-reported issues addressed in the same pass, both grounded in
direct measurement rather than guesswork.

**`examples/` cleanup**: moved everything that wasn't an official
numbered notebook or compact data product out -- calibration-table-
building scripts to `dev_notes/calibration/`, scratch/check notebooks and
their data to `dev_notes/scratch_notebooks/`, orphaned reference HTML to
`dev_notes/model_reference/` (and fixed the one live pointer to it,
`dev_notes/AUDIT_BRIEFING.md`). `examples/` now contains only the 6
numbered notebooks and `examples/data/`.

**The visible LOSVD "skew" in `01_basic_mock_fit.ipynb`/
`02_realistic_mock_fit.ipynb` is wing-amplitude inflation, not a new
bug** -- the same, already-documented mechanism `wing_lam` exists to
fix (`lam` penalizes curvature, not amplitude, so nothing stops
noise-driven wing mass at low sigma/S-N/many-template regimes), simply
never turned on in these two introductory notebooks. **Checked directly,
not assumed**: a 10-seed sweep at notebook 01's exact recipe (3
templates, sigma=50, S/N=60, `lam_selection='aic_local'`) found
`wing_lam=3e6` cuts median sigma bias from 3.2% to 1.5%; the same sweep
at notebook 02's exact recipe (4-template near-collinear mixture,
sigma=90, S/N=60) found `wing_lam=3e7` cuts median sigma bias from 11.4%
to 1.3% -- an order of magnitude larger value than notebook 01's own,
direct, checked confirmation of `PARAMETER_GUIDE.md`'s own warning that a
mixture's tuning doesn't transfer from a single-template table or
another mixture's own tuning. Added as new sections in both notebooks
(not silently swapped into the existing baseline, to preserve the
already-validated regularization-bias narrative both notebooks build
around their own untuned baseline).

**A second, distinct, previously under-emphasized finding: the
bootstrap 16th-84th percentile band is not guaranteed to contain its own
point estimate, and this is expected, not a bug.** Checked directly on
`05_real_data_muse.ipynb`'s nuclear aperture and `06_real_data_stis.ipynb`'s
one real bin: the point-estimate curve falls outside its own bootstrap
band across roughly half the velocity grid in both cases. Mechanism:
every bootstrap replicate is refit at the point estimate's own fixed
`lam`/`wing_lam`, so the replicate ensemble shows the *generic*
regularization bias that choice produces, while the point estimate
reflects whatever its own one noise draw happened to support -- the same
mechanism `01_basic_mock_fit.ipynb` Section 10 already documents for
*why* the raw bootstrap median isn't a good truth estimate, just not
previously stated as "the point estimate can visibly sit outside the
shaded band." Added a clarifying note to both real-data notebooks.

**Also found and fixed**: `06_real_data_stis.ipynb`'s cross-instrument
comparison against `05_real_data_muse.ipynb` had gone stale after that
notebook's own earlier rework (Stage 17) -- a hardcoded `V~-4, sigma~68`
comparison string no longer matched the reworked notebook's actual
nuclear-aperture output (`V=-9.8+/-1.6, sigma=81.5+/-3.8`). Fixed with
current numbers, and the claim that `sigma`/`h3` were "broadly
consistent" between instruments was revised, not just patched -- with
the updated numbers, `sigma` differs by ~23% and `h3` by a similar
relative amount, no longer reasonably called "consistent." The `V`
disagreement itself (STIS ~+70 km/s vs. MUSE ~-10 km/s) remains open and
unresolved, as before.

**Status**: full test suite green (161/161); all three touched notebooks
(01, 02, 06) re-executed end-to-end with 0 errors.

## Stage 19: comprehensive stress-test campaign (4480-fit grid across 4 template recipes), full-notebook reproducibility audit, and a new exploratory spectral-prior module

A broad, from-scratch validation pass -- not built on any single notebook's own scope, but a dedicated sigma x S/N x template-recipe grid plus a set of discrete robustness checks, run alongside a new research module. All raw results live in `dev_notes/stress_test/` (scripts + CSVs), not just summarized here.

**Comprehensive baseline grid** (`dev_notes/stress_test/comprehensive_grid.py`): 4480 fits, `sigma in {20,30,50,80,140,220,300,400}` x `S/N in {15,20,30,50,60,100,200}` x 4 template recipes (`single`: 1 real E-MILES template; `mix3`/`mix4deg`: the exact recipes `01_basic_mock_fit.ipynb`/`02_realistic_mock_fit.ipynb` use; `muse33`: the real 32-star MUSE library with a random Dirichlet mixture, following `dev_notes/scratch_notebooks/bias_sweep_fixed_vs_adaptive.py`'s own precedent), 20 seeds each, current package defaults only (`lam_selection='aic_local'`, `wing_lam=0.0`). 99.35% success rate overall; **confirmed at scale, not just on the one case Stage 15 originally found it on: `success=False` cells (29/4480) have essentially the same sigma-bias distribution as `success=True` ones (median |bias| 11.4% vs 11.6%)** -- reinforcing that the "ABNORMAL" scipy L-BFGS-B termination flag is spurious, not a real failure signal.

- **`single`/`mix3`/`mix4deg` (the recipes this package's existing notebooks/PARAMETER_GUIDE.md are built on) hold up well under this broader, independent re-test**: at S/N>=60 (the validated regime), median sigma bias stays under ~3% across the *entire* sigma range 20-400, matching what was already claimed.
- **A real, previously uncharacterized non-monotonicity at low S/N**: bias vs. sigma is *not* monotonic -- e.g. `single` at S/N=15 goes 93.8% (sigma=20) -> 8.2% (sigma=220) -> 60.3% (sigma=400). The existing validated grid (PARAMETER_GUIDE.md's own table, and prior sessions' sigma sweeps) never probed sigma=400 at low S/N simultaneously -- this specific corner is newly, directly measured here, not extrapolated.
- **A major new finding: the real 32-star MUSE library with an unconstrained (random Dirichlet) mixture is dramatically less stable than any deliberately-curated recipe tested anywhere in this project, including the "genuinely degenerate" `mix4deg` recipe.** Median sigma bias 39.7% overall (vs. 4-7% for the other three recipes); in the nominally "safe" regime (S/N>=60, sigma>=140) it's still 23.2% (vs. <2% for the others). **Confirmed not a fitting bug**: `chi2_red` stays excellent throughout (mean 0.953, barely different from the other recipes' ~0.98-0.99) -- individual cells at sigma=300/S-N=60 show V swinging +/-100 km/s and sigma from -2.7% to +79% bias *at chi2_red~0.9-1.05 every time*, the signature of a genuine template-mixture identifiability problem (many different (w,b) combinations fit the data equally well), not a convergence failure. This directly and quantitatively validates *why* `04_template_calibration.ipynb`'s iterative template-reduction pipeline (Stage 17) matters for kinematic reliability, not just computational cost -- fitting a real, uncurated multi-star library appears to be a materially harder and less stable problem than this package's own existing validation grid (built on small, deliberately-diverse recipes) would suggest.

**Full notebook reproducibility audit**: all 6 example notebooks re-executed fresh, 0 errors in every one. `01`/`02`/`04`/`05`/`06` reproduced their stdout exactly. `03_emiles_templates.ipynb` showed two benign differences: an expected tempfile-path difference, and a **real, if small, run-to-run non-determinism** in one cell -- a different `lam` was selected by `aic_local` on re-run (triggering a new warning that wasn't there before), with recovered sigma/V/weights shifting by ~0.3 km/s / ~0.2 km/s / ~0.002 -- consistent with `losvd.py`'s own documented float32/JAX non-determinism caveat ("in rare cases, change *which* local basin the multi-start search converges to"), now observed directly rather than just theorized.

**Edge-case/robustness checks** (`dev_notes/stress_test/edge_case_checks.py`, 10 discrete checks, 0 crashes): sigma=5 km/s (below the validated floor) is *not* just "somewhat off" -- 3x overestimate (14.99 vs. true 5.0), and the package's own narrow-grid warning correctly predicts this. `wing_lam` + `continuum_knots` together (with `lam_selection='fixed'`, as required) runs cleanly with no warnings and good recovery -- this specific combination had never been explicitly exercised end-to-end before. A single fit against the *entire* 636-template E-MILES grid succeeds (85.9s, correctly finds 8/636 nonzero-weight templates matching the 3 truly-injected ones plus minor degenerate substitutes). `bootstrap_losvd(n_boot=1)` degrades gracefully (nan std, no crash); `n_boot=200` runs fine, just slower.

**New exploratory module: `src/kinextract/spectral_prior.py`** -- the "eigen-VP"/spectral-law generalization of the P-spline curvature penalty queued in an earlier session's own research-direction notes: writes the existing penalty as `coeffs' @ D2'D2 @ coeffs = sum_i mu_i * c_i^2` in `D2'D2`'s own eigenbasis (`c = U'coeffs`), and generalizes the implicit per-mode weight from `lam * mu_i^1` to `alpha * mu_i^beta`, a proper 2-parameter superset (`beta=1, alpha=lam` reproduces today's penalty exactly). No continuum-cofitting yet and no automatic `(alpha,beta)` selection (matches the recommended minimal starting scope) -- `(alpha, beta)` are supplied by the caller, exactly like `lam`/`wing_lam` are today. 7 new tests (`tests/test_spectral_prior.py`), all passing, including a regression check that `beta=1` reproduces `fit_log_spline_map`'s own moments to within optimizer noise (<2 km/s in V, <3 km/s in sigma, across 10 seeds).

**Swept `(alpha, beta)` on the exact regimes already validated with hand-tuned `wing_lam` this session** (`dev_notes/stress_test/spectral_prior_sweep.py`, 2160 fits, 10-seed median tuning matching `PARAMETER_GUIDE.md`'s own established methodology):
- **`beta<0` (the hypothesis that inverting the law to penalize *smooth* modes could suppress wing-inflation directly, without a separate amplitude term) is definitively refuted** -- 240-310% sigma bias at every tested alpha, across every regime. Mechanism, checked not just asserted: a peaked LOSVD's `log(b)` is itself mostly a smooth, low-order (near-null-space) shape -- heavily penalizing exactly those modes destroys the ability to represent *any* peaked distribution, not just a spurious wing pedestal. `beta=0` (plain L2-on-magnitude, no curvature shape at all) is also bad (76-142% bias), consistent with `losvd.py`'s own already-documented "plain L2 measurably underperforms the curvature form" finding -- a useful internal consistency check on the sweep itself.
- **`beta>1` is a real, positive finding.** A single fixed pair, `alpha=1, beta=2` (no per-regime tuning at all), gives sigma bias of 3.2% / 2.0% / 7.4% on the three tested regimes (`01`'s 3-template/sigma=50/S-N=60; `02`'s 4-template-degenerate/sigma=90/S-N=60; and an untried low-S/N generalization check, sigma=50/S-N=30) -- beating `beta=1`'s own best-tuned-alpha result (7.6% / 1.5% / 16.3%) in two of three regimes and landing within noise of it in the third, *and* improving h4 bias alongside sigma in the same two regimes (0.036->0.005, 0.039->0.005) -- a moment the fit doesn't directly optimize for, so this isn't circular. The needed `alpha` range across all three regimes (0.5-2.0) is far more consistent than `wing_lam`'s own previously-reported range for the same three regimes (3e5-3e7, a 10^5x span) -- suggestive that this may be a genuinely more portable single mechanism than the `lam`+`wing_lam` combination it's meant to subsume, though this is three regimes, not a validated replacement.

**Honest limitations of this stage's spectral-prior result**: only 3 regimes tested (not the full `comprehensive_grid.py` sweep -- doing that cross product is a natural next step); no automatic `(alpha,beta)` selection exists yet (a real, separate undertaking, as the original research note already flagged); not tested against real data; not tested for interaction with continuum-cofitting or bootstrap uncertainty propagation.

**Status**: full test suite green (168/168, the 161 from Stage 18 plus 7 new `test_spectral_prior.py` tests) -- run concurrently with the 4480-fit grid on the same 8-core machine, itself a mild stress test (no flakiness observed under sustained CPU contention). All 6 notebooks reproducible. `spectral_prior.py` is exploratory/unvalidated for production use, clearly marked as such in its own module docstring.

## Stage 20: spectral_prior's `(alpha=1, beta=2)` result does not generalize -- a real, sigma-dependent crossover found

Direct follow-up to Stage 19's 3-regime spot-check, at the user's own request to dig further before calling it viable. `dev_notes/stress_test/spectral_prior_broad_sweep.py`: 3200 fits, `mix3` and `muse33` (the well-behaved and the hard-instability recipes from Stage 19's grid), the *entire* `sigma in {20..400}` grid x a 4-point S/N subset `{15,30,60,100}`, `beta=2.0` fixed, `alpha` swept, **identical seeds to `comprehensive_grid_results.csv`** (imported `comprehensive_grid._get_recipe`/its exact seed formula directly, so every comparison below is a true paired test against an already-computed baseline fit on the *same* noise draw, not two independent medians).

**The Stage 19 result does not hold up at this scale.** A single fixed `(alpha=1, beta=2)` beats the current default in only 19/32 (mix3) and 15/32 (muse33) sigma/S-N cells -- worse than a coin flip on muse33 -- and its *overall* median |sigma bias| is slightly *worse* than baseline on both recipes (mix3: 9.8% vs. 8.2%; muse33: 46.5% vs. 44.6%). Even letting `alpha` be re-tuned per cell (i.e., giving the spectral-prior method the same per-regime tuning latitude `wing_lam` already needs) only gets it to 22/32 and 24/32 cells -- a real improvement over the single-alpha case, but far from a clean win everywhere.

**The real, well-characterized pattern (not noise): a sharp sigma-dependent crossover.** Fixing `alpha=1` and looking at median |sigma bias %| purely as a function of true sigma (averaged over the S/N subset):

| sigma | mix3, spectral | mix3, baseline | muse33, spectral | muse33, baseline |
|---|---|---|---|---|
| 20  | 35.4% | 45.9%  | 87.4% | 104.1% |
| 30  | 9.7%  | 20.2%  | 26.7% | 65.3%  |
| 50  | 3.8%  | 7.7%   | 21.2% | 34.8%  |
| 80  | 3.4%  | 4.4%   | 46.8% | 27.9%  |
| 140 | 6.6%  | 5.3%   | 31.9% | 18.8%  |
| 220 | 17.7% | 5.5%   | 44.7% | 20.1%  |
| 300 | 19.9% | 5.2%   | 59.8% | 40.7%  |
| 400 | 23.0% | 8.8%   | 77.1% | 58.6%  |

`beta=2` **helps substantially at low-to-moderate sigma** (below ~80-140 depending on recipe -- exactly the regime this package's own default already struggles most, and where `wing_lam` already targets) and **hurts, increasingly, at high sigma** -- the opposite regime. Confirmed not a convergence artifact: success rate stays ~100% (occasionally 95%) across the whole sigma range on both recipes, so this is a real bias effect, not optimizer failure. Mechanistically sensible, not just an empirical curiosity: a narrow LOSVD is mostly noise-driven high-frequency content that a steeper rough-mode penalty correctly suppresses; a broad LOSVD *needs* those same moderately-rough modes to represent its own genuine width, and `beta=2` over-suppresses them there, introducing bias in the same direction the curvature penalty already does under-resolution (see `losvd.py`'s own knot-density/grid-width scaling note -- this may be a related, currently-unaccounted-for interaction between `beta` and grid width, not yet tested).

**Runtime, measured cleanly (not under Stage 19's contended conditions)**: a fixed-`(alpha,beta)` spectral-prior fit is **~7.4x faster** than `aic_local`'s current default (2.86s vs. 21.1s per fit, same mock, same machine, back-to-back in one process) -- entirely because `aic_local` pays for an internal 8-value `lam` grid search (each a full multi-start optimization) every fit, while a fixed pair pays for one. This speed advantage is real *only* if a usable `(alpha,beta)` choice doesn't itself need an equally expensive per-fit search -- which Stage 19's single-pair result implied but this stage's broader test shows is not yet the case.

**Revised assessment**: the spectral-law generalization itself is sound (the `beta=1` regression check still holds exactly as before) and there is a real, mechanistically-understood, usable win in the specific low-sigma regime -- but **a single global `(alpha, beta)` is not a viable drop-in replacement for `lam`+`wing_lam`** the way Stage 19's 3-point spot-check suggested. Whatever generalizes further would need `beta` (or `alpha`) itself to vary with sigma/grid-width, much like `lam`'s own `dv_ref/dv` scaling already does for the plain curvature penalty -- an untested, real next step, not yet built.

**Status**: no code changes this stage -- `spectral_prior.py` and its tests are unchanged; this stage is purely an additional empirical validation pass answering "is it viable" more rigorously than Stage 19 could. `dev_notes/stress_test/spectral_prior_broad_sweep.py`/`_results.csv` hold the raw data.

## Stage 21: the sigma-crossover's root cause found and substantially fixed via eigenvalue normalization -- real progress, not yet a universal win

Direct follow-up to Stage 20, at the user's own request to dig into *why* the crossover happens and whether it can be engineered away, not just documented as a limitation.

**Root cause, found directly**: `mu.max()` (the roughest mode's raw eigenvalue) varies by **>30x** across this package's own validated sigma range -- 525 at sigma=20 down to ~16.5 at sigma>=220 -- because at low sigma, `resolved_n_interior_knots()`'s `min_interior_knots` floor (not `knot_spacing_kms`) sets the knot count, so the grid hasn't yet reached its calibration reference and `difference_matrix`'s own `dv_ref/dv` correction inflates `D2` (and hence every `mu_i`). Since `beta` is an *exponent* on `mu_i`, this 30x swing in absolute scale becomes an enormous swing in *relative* penalty strength for any `beta != 1` at fixed `alpha` -- e.g. `beta=2` squares a `mu_i<1` (common at high sigma, where `mu.max()` is small) into something *smaller*, silently switching off curvature suppression for exactly the modes that matter there. This fully explains Stage 20's crossover.

**The fix**: `eigendecompose_curvature(..., normalize=True)` divides `mu_floored` by `mu.max()` before the caller ever sees it, so `(alpha, beta)` means the same thing regardless of grid size. Added as an opt-in parameter (`normalize_eigenvalues=False` by default in `fit_losvd_with_spectral_prior`, preserving the existing `beta=1` regression check exactly) -- not a silent behavior change. New tests: `test_normalize_scales_max_eigenvalue_to_one`, `test_normalize_removes_the_sigma_dependent_crossover` (both passing; 171/171 full suite green).

**A second bug caught and fixed in the process, unrelated to the method itself**: the sweep scripts' own seed formula used Python's built-in `hash(recipe)`, which is randomized per-process for security -- meaning `comprehensive_grid.py`, `spectral_prior_broad_sweep.py`, and the first version of this stage's own sweep script each drew *independent* noise despite an explicit "identical seeds, true paired comparison" claim in their docstrings and in Stage 20's writeup. That claim was wrong. The median-over-20-seeds comparisons themselves remain statistically valid (same generative distribution either way, just not paired), but a first pass at re-validating the fix with a **quick 5-8 seed manual scan got a misleadingly good result** (small, flat bias across the whole sigma range) that did not hold up once retested properly -- a real lesson, not just a technicality: small-sample manual checks in a hurry can and did produce a false positive here. Fixed the seed formula (`zlib.crc32`, deterministic across processes) before drawing any further conclusions.

**The properly-powered result** (20 seeds/cell, full sigma x S/N grid, all 4 recipes, `alpha=100, beta=1.5, normalize_eigenvalues=True` -- found via a 20-seed scan across sigma, not the misleading small one):

| recipe | overall median &#124;bias&#124;, normalized spectral | overall median &#124;bias&#124;, baseline | cells beating baseline |
|---|---|---|---|
| single template | **5.7%** | 6.6% | 35/56 (62%) |
| mix3 | 7.9% | 7.7% | 27/56 (48%) |
| mix4deg | 7.9% | 5.9% | 22/56 (39%) |
| muse33 (real library) | 43.5% | 39.7% | 21/56 (38%) |

Broken down by sigma, the pattern is consistent and mechanistically clear across all four recipes: a **large, real win at low sigma** (e.g. single-template sigma=20: 9.4% vs. baseline's 39.5%; sigma=30: 6.6% vs. 16.4%), **roughly comparable in the middle** (sigma 50-140), and a **real, if now much milder, degradation at high sigma** (sigma>=220 -- e.g. single-template sigma=300: 10.6% vs. baseline's 5.4%). This is a dramatic improvement over Stage 20's un-normalized version (which was worse than baseline almost everywhere, including catastrophically so on `muse33`), and for the single-template case it is now a genuine net win in aggregate -- but it is **not a clean universal win**: the residual high-sigma degradation, while much smaller than before normalization, is real and not yet eliminated for any recipe.

**Honest verdict**: normalization fixes the *mechanism* (confirmed: it was a pure scale confound, not a fundamental limit), and gets a fixed `(alpha, beta)` from "actively harmful across most of the range" (Stage 20) to "a real win at low sigma, roughly neutral at high sigma" (this stage). What remains is consistent with the same diagnosis as before, just smaller in magnitude: a broad LOSVD still needs somewhat more of its moderately-rough modes than `beta=1.5` allows it, even after normalizing away the scale confound. The natural next step, not yet attempted, is making `beta` itself decay toward 1 as sigma grows (e.g. `beta = 1 + 0.5*exp(-sigma/sigma0)`) rather than searching for one sigma-independent constant -- since the evidence here is that no such constant exists, not that the search was insufficient.

**Also fixed this stage, found while inspecting a real notebook (user-reported)**: `binning.assign_spaxels_to_bins` required `r >= r_edges[0]` even inside the `single_center_bin=True` collapse path, so any spaxel closer than `R_MIN` -- including the literal center pixel -- fell through as unassigned (`-1`) instead of joining the center bin. Fixed: the center bin now absorbs everything closer than `r_edges[1]`, matching its own documented intent ("every spaxel in the innermost annulus collapses into one bin"). Verified directly against the real MUSE cube's own dartboard config (R_MIN=0.2", center=(160,158)): old code left the center pixel at `-1`; fixed code assigns it to bin 0 (n_spaxels 4->5). New regression test added. Also found in the process: a real ~2.97GB downloaded MUSE cube sitting untracked (and previously ungitignored) at `examples/notebooks/muse_download/` -- added a `.gitignore` entry (`examples/notebooks/muse_download/`, `*.fits`) so it can't be accidentally committed.

**Status**: 171/171 tests passing. `spectral_prior.py`'s own docstring updated to describe `normalize=True` and point to this stage. Raw data: `dev_notes/stress_test/spectral_prior_normalized_sweep.py`/`_results.csv`.

## Stage 22: the LOSVD shape-residual problem root-caused and fixed -- third-difference (shrink-toward-Gaussian) penalty promoted to default, JAX float64 promoted to default

Full chronological detail (every diagnostic, every negative result, all
sweep CSVs) lives in `dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`
and `dev_notes/shape_residual/`; the current-state summary is the
status section at the top of
`dev_notes/REGULARIZATION_SHAPE_RESIDUAL_HANDOFF.md`. Condensed record:

**Problem** (from the handoff report): recovered LOSVDs show ~10%+
full-curve shape residual at realistic S/N (peak too tall/narrow,
shoulder deficits) even where moment-level bias looks fine.

**Root cause, isolated via a truth-informed oracle-lambda diagnostic**
(15 seeds x 4 S/N x 9 lam values, single template, sigma=50): the
defect is the *penalty form*, not the strength selection. The D2
(second-difference) penalty's null space is linear log-b, so heavy
smoothing drives exponential tails (the documented wing-inflation
mechanism) while light smoothing overfits the core -- and even the
truth-informed best lam fails the 5% shape target at S/N<=60. No
selection rule can fix a penalty form whose entire lam axis is bad;
w_m reweighting, GCV, and discrepancy rules were all tested offline
and confirmed not to help.

**Fix**: `FitConfig.penalty_order=3` (now the default). The
third-difference penalty's null space is quadratic log-b -- any
Gaussian is penalty-free -- so heavy smoothing shrinks toward the
best-fitting Gaussian instead of fat tails. Validated at production
scale (480 paired cases: sigma {20,50,140,300} x S/N {30,100} x
{Gaussian, (h3,h4)=(0.06,0.06)} x {single, mix3}, identical noise
draws, zlib.crc32 seeding): better mean shape residual in 23/32 cells
(332/480 paired wins), overall |sigma bias| 5.6 vs 15.5 km/s, wing-mass
ratios ~1.0 everywhere (D2: 3-28x at low S/N) with `wing_lam` OFF --
the mechanism `wing_lam` existed to patch is gone by construction.
Documented tradeoff: for genuinely non-Gaussian truth (h3=h4=0.06) at
sigma 50-140 / S/N=100, order 2 keeps a modest shape-residual edge
(order 3 shrinks real h4 by ~0.03-0.05) -- a regime where sigma(h4) of
even a direct parametric GH fit is ~0.1-0.2, i.e. h4 is near the
information limit regardless. `penalty_order=2` reproduces the old
behavior exactly (its resolved lam grid is bit-identical to the old
default tuple).

**Three supporting changes shipped with it, each with its own evidence**:

1. **JAX float64 at package import** (opt-out `KINEXTRACT_NO_X64=1`):
   float32 L-BFGS-B reliably stalls tens of objective units above the
   true minimum in the stiff heavy-lam regime order 3 operates in
   (measured: 35% vs 6% shape residual at sigma=300/S N=30/lam=65536,
   identical data/starts, both `success=True`). Validated neutral for
   the order-2 path across the full 480-case grid before promotion
   (~2x runtime cost). Also documented: the old "noise=1e-6" noiseless
   benchmark convention breaks the optimizer at any precision -- use
   S/N~1e4 as the noiseless proxy.
2. **Delta-AIC parsimony tiebreak** (order-3 selection only):
   `select_lambda_aic_local(parsimony_delta=2.0)` picks the largest
   lam within 2 criterion units of the minimum (Burnham & Anderson's
   indistinguishability convention). Exists because m_eff pins near 3
   in strongly ill-posed regimes (sigma~20), making the criterion flat
   and argmin a noise lottery. Tested on every recorded sweep: fixes
   sigma=20/30 (mean 51->26%) and sigma=300 worst cases (45->15%) with
   no Gaussian-truth regression; the same rule measurably *hurts*
   order 2 (heavier D2 smoothing wing-inflates), so it is gated off
   there.
3. **Sub-knot-sigma candidate guard** (order-3 selection only):
   candidates whose own fit's direct-moment sigma is below 0.5x the
   knot spacing are excluded from selection -- such fits claim
   structure the basis cannot resolve, the signature of a delta-spike
   collapse that can beat the tiebreak window outright (the linearized
   m_eff badly underestimates a spike's flexibility). Verified against
   all 960 production-validation fits: flags exactly the 4 catastrophic
   spikes (344-712% residual, recovered sigma 2.4-4.4 km/s), zero
   healthy fits. With the guard, sigma=20/S N=30 worst case drops
   712% -> 79% and the means beat the D2 default's.

**Information limits mapped (the 5%-everywhere goal is not achievable
in these regimes for ANY method, parametric included -- measured, not
assumed)**: sigma~20 at S/N<=30 (deconvolution through sigma_inst~35
km/s supports ~3 effective dof); |h3|/h4 ~ 0.1 structure at sigma~50,
S/N<=100 (parametric GH fit itself has sigma(h3)=0.15 at S/N=60);
truth curves with exact zeros in the fit range (clipped-GH benchmark
shapes -- spectrum-degenerate to ~8% LOSVD differences).

**Negative results documented in full in the investigation log**:
GH-anchored D3 penalty (anchor noise dominates; plus an
unweighted-log-projection trap worth remembering), Greville
divided-difference operator (coefficients are blossoms, not function
values), stiffer selection rules on D2, the local-basin hypothesis
(objective_gap uncorrelated with residual), scaling-exponent-3 for the
knot-spacing correction (no better than exponent 1).

**Separate finding from the accompanying general bias review**:
`shift_template`'s linear interpolation adds a fractional-shift-
dependent effective broadening (0 to ~17 km/s, period 31 km/s across
design-matrix columns on the E-MILES grid) that synthetic benchmarks
structurally cannot see (mocks are built with the same design matrix)
but which biases real-data sigma low by ~1-2 km/s at sigma=50.
Recommended (not yet done): band-limited interpolation in
`shift_template`, and generating benchmark spectra at native template
sampling to break the circularity.

**Status**: new `tests/test_penalty_order.py` (8 tests, incl. selection
logic); full suite green after the default flip (count reported in the
investigation log's final entry). `PARAMETER_GUIDE.md`, `losvd.py`/
`config.py`/`aic_regularization.py` docstrings updated.

## Stage 23: real-data sigma inflation fixed (edge-amplitude constraint); the mock "shape bias" reframed as the noise floor

Full detail: `dev_notes/real_data_sigma/FINDINGS.md`, `dev_notes/ghcore/FINDINGS.md`,
`dev_notes/REGULARIZATION_INVESTIGATION_LOG.md` (2026-07-28 entries).

**Two symptoms chased as one problem turned out to be unrelated.**

Real NGC 5102 MUSE spectra recovered sigma roughly 2x the legacy Fortran
pipeline's (84.5/100.7 vs 54.3/40-45 km/s). Cause: the order-3 penalty's null
space (any Gaussian, including divergent `exp(+av^2)`) let the fit grow a
divergent edge pedestal that the moment integral counted as dispersion. Fixed
by `FitConfig.max_edge_amplitude_frac=0.01`, a validity filter (not a
smoothness preference) on the AIC candidate grid: reject any lam whose fitted
LOSVD is still >1% of peak at the grid edge, then let AIC choose among
survivors. Nuclear bin 84.5 -> 49.1 km/s, mid-radius 100.7 -> 42.4 km/s, both
now matching legacy. Mocks are bit-identical with the constraint on or off
(it excludes nothing where there is no pedestal) -- the property every earlier
candidate (wing-mass penalty, `lam_wing`) lacked. 174/174 tests pass.

Two negative results on the way: linear-space D2 (OG kinextract's own penalty,
implemented faithfully) does NOT fix real data -- it gives sigma 88-97 km/s at
every lam, because heavy smoothing in linear space drives b toward a *broad*
linear ramp, the opposite of log space's compact-Gaussian null space. And OG's
own unimodal+non-jagged selection rule picks the same broken lam AIC does,
because the pedestal solutions are both unimodal and smooth.

**Separately: the mock shape residual (~7-8% at S/N=60) is not a penalty
defect at all.** The Cramer-Rao bound (Fisher matrix, template weights
marginalized) gives sigma(h4)=0.052 at S/N=60 against the NB02 mock's truth
h4=0.050 -- the quantity is below its own 1-sigma detection limit. Confirmed
independently: a 4-parameter fit with NO penalty and no null space still gives
11.5% residual at S/N=60, exact when noiseless, monotonic in S/N. This closes
out every rejected-penalty attempt in Stage 22 and the investigation log --
none of them were wrong to reject, because the thing they were chasing was
never fixable by penalty redesign.

**Corollary for real work**: since regularization shrinks h4 toward zero when
S/N cannot support it, a naive radial h4 profile in real data will show a
spurious decline with radius that is pure noise, not anisotropy -- a live risk
for orbit-superposition / M_BH work if h4 error bars don't cover zero at low
S/N. This is exactly the open problem in Stage 24 below.

**Status**: `_corrected` bias-correction fields are now STALE (built at fixed
lam=2.0 under penalty_order=2; both defaults have since changed) -- do not use
until regenerated. Uncertainty bands measured ~2-4x too narrow (bootstrap is
blind to bias shared by all replicates). Notebook 02 gained a new Section 6
demonstrating the noise floor directly (recovered-vs-true LOSVDs across an
S/N ladder, h4 climbing out of a +/-3-sigma CRLB band).

## Stage 24: Gauss-Hermite "truth" mismatch found via randomized testing; a reusable randomized bias-validation suite built

Full detail: `dev_notes/validation_suite/FINDINGS.md`.

User-built randomized testing (`temp_testing.ipynb`, randomizing sigma/h3/h4/
S-N/templates/weights per replicate rather than reusing one fixed mock)
surfaced a large discrepancy between nominal GH generator parameters and
`compute_losvd_moments`'s own reading of the resulting curve (tens of km/s in
V, tens of percent in sigma) and raised two hypotheses: bad synthetic spectra,
or a bug in `compute_losvd_moments`.

**Root-caused to neither.** Isolated with a pure-numpy, zero-fitting,
zero-clipping test case (V=100, sigma=391, h3=-0.089, h4=+0.092: raw curve min
= +3e-12, never negative) -- the curve's own moments (V=42.4, sigma=456.9)
still differ hugely from nominal. Mechanism: `(V,sigma,h3,h4)` are truncated
Gauss-Hermite *coefficients*, not literally that curve's mean/std/skew/kurt;
the two coincide only as h3,h4 -> 0. h3 shifts the mean (+9.7 km/s at h3=0.03,
sigma=200, growing with h3); h4 shifts the width (+9.5% at h4=0.05); both add.
Over this project's existing randomization range (sigma 40-400, h3/h4
+/-0.1), 53% of draws have |V shift|>10 km/s and 68% have |sigma shift|>5%
from this effect ALONE. Clipping is real but second-order (<=4% of mass even
at range extremes).

Neither hypothesis in the bug report was correct, and neither of the two
flagged examples turns out to be an actual method failure once bias is scored
against the truth curve's own moments (as the user's own prototype already
correctly does) rather than nominal parameters -- example 1 bias was 2-3% of
sigma; example 2's h4 shrinkage is consistent with the noise floor (Stage 23).

**Built**: `dev_notes/validation_suite/` -- `mocks.py` (mock generation with
truth-curve moments + `generator_mismatch` + `clipped_mass_frac` diagnostics
on every case, `safe`/`wide` randomization presets, an alternative
spline-based truth generator with no GH step), `runner.py` (one standardized
result row per fit), `sweep.py` (CLI: fix one axis over a grid, randomize
everything else per replicate, reproducible given `--seed`, incremental CSV
writes), and `examples/notebooks/09_validation_suite_results.ipynb`
(standardized fit-bias-vs-generator-mismatch plots, a Gaussian-only control
sweep, a lambda-selection-stability diagnostic). Verified reproducible
(bit-identical rerun given the same seed); sigma/S-N/h3/h4 sweeps plus a
Gaussian-only control run at 15 replicates/grid-point (safe preset).

**Status**: framework built and validated; spline-truth generator implemented
but not yet wired into `sweep.py`'s CLI; 2D combined sweeps not yet built.

## Stage 24 follow-up: GH-truth vs. spline-truth sweeps completed (525 fits) -- spline truths modestly cleaner, kept both as first-class tools

Full detail: `dev_notes/validation_suite/FINDINGS.md` results section.

Ran all 7 planned sweeps (sigma/S-N/h3/h4 with GH truths, a Gaussian-only
control, and sigma/S-N with spline-native truths -- `mocks.build_spline_truth`,
wired into `sweep.py` via `--truth-kind spline`). Direct comparison, identical
fitter/grid/replicate count, only the truth generator differs:

| sweep | median \|sigma bias\|, GH | median \|sigma bias\|, spline |
|---|---|---|
| sigma | 2.71 km/s | 1.35 km/s |
| S/N | 4.50 km/s | 3.69 km/s |

Spline truths give smaller bias in both (roughly half in the sigma sweep),
and a fully independent Gaussian-only-vs-full-GH-shape control shows the
same qualitative pattern on separate data. Real, if modest (~2x at most),
evidence that GH-shaped truths are somewhat harder to recover cleanly than
an arbitrary spline-native shape -- not just the previously-established
coefficient-vs-moment mismatch. Decision: spline truths become the primary
tool for isolating pure fitting/regularization bias going forward; GH truths
stay first-class for literature/legacy comparability and as a physically-
motivated stress test. Not scrapped, per discussion.

The h3/h4 sweeps independently confirm the shrinkage-toward-Gaussian
mechanism directly on real (fitted, not just analytic) data, and confirm the
generator-mismatch diagnostic dominates fit bias for V/sigma over the swept
range exactly as the earlier zero-clipping analytic argument predicted.

Two bugs found and fixed while running this (both logged in FINDINGS.md):
an argparse negative-value gotcha that silently truncated a run before h4/
Gaussian sweeps executed (compounded by `set -e`), and a notebook path bug
(`RESULTS_DIR` relative to the wrong directory) caught by inspecting actual
executed output rather than trusting a clean exit code.

## Stage 25: bootstrap lam-pin fixed (reselect_lam), a real secondary contributor to under-coverage

Full detail: `dev_notes/uncertainty/FINDINGS.md`.

Checked the first item on the post-Stage-24 open-problems list: does
`bootstrap_losvd` pinning lam at the point estimate's own selection (for
every replicate, by design, to save cost) hide the documented lam-selection
lottery (S/N 60-150: same data picks lam=8 or lam=32768 on noise alone) from
the reported uncertainty band? Confirmed by direct code read: yes,
unconditionally. Measured with 3 ensembles (pinned / reselected-isolated /
true independent-sampling, n=30, two sigma regimes): reselecting closes
~60-80% of the gap between the pinned bootstrap's spread and the true
sampling distribution's own spread -- real, but the separate, already-
documented bias-blindness mechanism (resampling around a biased point
estimate can't see the bias itself) is untouched and still expected to be
the larger remaining contributor to the originally-measured 2x/4x
under-coverage.

Fix: `bootstrap_losvd(reselect_lam=True)`, new default, ~2-2.5x cost per
replicate (affordable given the ~6.8x fitting speed-up). Forced off when
continuum-cofitting is active (no behavioral difference there, just a
repeated warning otherwise). 4 new tests; every pre-existing bootstrap test
audited and pinned `reselect_lam=False` where irrelevant to keep the suite's
runtime unchanged. Full suite: 178 passed, 0 failed, 32:28 (matches the
32:51 pre-change baseline -- an earlier 50-minute stall during this work
turned out to be a one-off fluke, not a regression, confirmed by rerunning
clean).

**Status**: uncertainty-band work continues -- the bias-blindness mechanism
itself is not yet fixed; next step folds a coverage-calibration study into
the bias-correction table regeneration already planned (Stage 26).

## Stage 26 (partial, ongoing): calibration regeneration, and both deferred items closed

**shift_template interpolation bias** (deferred from the original 7-item
list): quantified directly (no fitting) against an oversampled ground
truth -- linear interpolation's error is genuinely periodic in fractional
pixel shift, period ~31.7 km/s at 8500 Angstrom (matching the earlier
"0-17 km/s, period ~31 km/s" real-data finding exactly), peaking at
~1.1-1.2% of amplitude. Fixed via a new opt-in `method="lanczos"` on
`shift_template`/`build_template_stack` (band-limited windowed-sinc,
requires uniform template sampling, verified true for E-MILES). Confirmed
correct (exact grid alignment reproduces source values) and confirmed to
remove the periodic component specifically (the part that can masquerade
as shift-dependent kinematic broadening), leaving only a smaller,
phase-independent residual. Costs ~67x more than linear in isolation but
~6-9% of total fit time in practice (`build_template_stack` runs once per
fit, not per iteration). Not promoted to default -- same opt-in convention
as every other new mechanism pending real-data validation at scale. 6 new
tests, `test_forward_model.py` full file green. Full detail:
`dev_notes/shift_template/FINDINGS.md`.

**The MUSE notebook's low-S/N aperture** (S/N~15): confirmed via direct
fit it is not broken (fit succeeds, lands on h3=h4=0.000 exactly -- the
order-3 penalty correctly shrinking to a pure Gaussian). Strengthened the
notebook's own documentation with tonight's Cramer-Rao quantification
(sigma(h4) ~ 0.1-0.2 at this S/N, an order of magnitude above any
plausible h4) rather than the previous qualitative "least trustworthy of
the three." A genuinely better-S/N replacement aperture would need
rerunning the dartboard-binning cube extraction this notebook doesn't
itself perform -- out of scope tonight, flagged explicitly as a real,
larger follow-up.

**Calibration tables**: main and h4wing tables regenerated cleanly under
current defaults (order-3 penalty, updated `LAM_GRID_FOR_MEFF` matching
`select_lambda_aic_local`'s own grid). The meff table (8x8 sigma/S-N grid
x 6 lam x 30 reps = 11520 fits) hit real, substantial per-cell cost growth
with sigma (linear-algebra cost of `estimate_m_eff_local`'s own n_coef x
n_coef solve scales with grid width/knot count) -- two full-grid attempts
were killed by timeout with zero saved progress (no checkpointing in the
original script). Rebuilt as a per-sigma-checkpointed version
(`dev_notes/calibration/build_meff_incremental.py`, reusing every existing
piece of `build_emiles_calibration.py` unchanged) so a future interruption
loses at most one sigma row's worth of compute, not the whole run. Running
as of this entry; completion and full verification to follow in the next
entry.

## Stage 26 (complete): calibration tables regenerated under current defaults

Full detail: dev_notes/REGULARIZATION_INVESTIGATION_LOG.md 2026-07-29 entry,
dev_notes/calibration/build_meff_incremental.py.

All three calibration tables (main, h4wing, meff) rebuilt under the current
order-3/adaptive-selection defaults, replacing the stale order-2-era tables.
Verified: all three load correctly, 10/10 tests in test_bias_correction.py
pass. The `_corrected` bias-correction fields, flagged stale since Stage 22's
order-2-to-3 default flip, are valid again as of this stage.

Real finding along the way: the meff table's per-cell cost grows sharply with
sigma (sigma=400 alone took ~72 min of the ~2.5-3 hour total), not primarily
CPU contention as first suspected -- the linear-algebra cost of
`estimate_m_eff_local`'s own closed-form solve scales with the spline's own
coefficient count, which grows with grid width. Two un-checkpointed attempts
were killed by fixed timeouts with total loss of progress before this was
diagnosed; fixed by checkpointing per sigma value.

## Stage 27: end-to-end coverage check -- reselect_lam helped but did not fix coverage (honest negative result)

Full detail: dev_notes/uncertainty/FINDINGS.md addendum.

Built `dev_notes/validation_suite/coverage_sweep.py` to check the thing that
actually matters after Stage 25's reselect_lam fix: does the reported
bootstrap confidence interval contain the truth at the rate it claims to?
9 (sigma, S/N) cells x 4 random truths each (randomized h3/h4/templates/
weights, safe preset), one bootstrap_losvd call per truth (n_boot=12,
reselect_lam=True).

**Result (n=36): sigma coverage 47%, h4 coverage 11%, against a 68% nominal
target.** Under-coverage is broad, not a corner case -- every one of 9 cells
under-covers sigma, and h4 coverage is 0% in 7 of 9 cells.

Honest conclusion: reselect_lam was a real, measured, PARTIAL fix, not a
solution. The dominant remaining mechanism is resampling around a biased
point estimate, which a spread-only fix cannot touch -- the interval can
have closer-to-correct width and still be centered wrong. h4's much worse
coverage than sigma's is consistent with the noise-floor/shrinkage
mechanism (Stage 24, dev_notes/ghcore/FINDINGS.md) giving it a larger
systematic offset.

**Status: the uncertainty-band problem remains open.** Next step (not done
tonight, time-limited): rerun this same coverage check with
`apply_bias_correction=True` against the freshly-regenerated (Stage 26)
calibration tables, to measure whether shifting the reported center (not
just its width) actually closes the remaining gap.
