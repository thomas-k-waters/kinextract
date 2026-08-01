# The Gauss-Hermite "truth" mismatch, and the randomized bias-validation suite

**Date: 2026-07-28**

## The finding

The user built a prototype (`examples/notebooks/temp_testing.ipynb`) that
randomizes sigma, h3, h4, S/N, templates and weights per replicate, to check
whether earlier bias measurements (all done on one or two fixed mocks) were
representative or were an artifact of testing the same dataset every time.
Two examples surfaced apparently large errors:

```
Input true moments: v=100.00, sig=328.19, h3=0.080, h4=-0.081
GH fit true moments: v=120.85, sig=303.41, h3=0.054, h4=-0.061
Recovered moments:   v=113.49, sig=309.92, h3=0.052, h4=-0.050

Input true moments: v=100.00, sig=391.30, h3=-0.089, h4=0.092
GH fit true moments: v=42.32,  sig=456.82, h3=-0.110, h4=0.070
Recovered moments:   v=41.37,  sig=438.47, h3=-0.087, h4=0.017
```

The concern: "GH fit true moments" (i.e. `compute_losvd_moments` applied to
the actual, discretized, clipped, normalized truth curve) differs by tens of
km/s and tens of percent from "Input true moments" (the nominal generator
parameters) -- with no fitting involved at all. Two hypotheses were raised:
either the synthetic spectra don't actually reflect the injected LOSVD, or
`compute_losvd_moments` is buggy.

## Root cause (verified directly, no fitting involved)

Neither. Isolated the effect with a pure numpy experiment: build
`gauss_hermite_model(v, V, sigma, h3, h4)`, check whether it ever goes
negative, and compare its own measured moments to its defining parameters.

```
V=100, sigma=391.30, h3=-0.089, h4=+0.092:
    raw curve minimum = +3.18e-12  (NEVER GOES NEGATIVE -- zero clipping)
    curve's own moments: V=+42.41, sigma=456.87, h3=-0.1096, h4=+0.0704
```

**Zero clipping, and the moments still differ from nominal by 58 km/s in V and
17% in sigma.** This rules out both hypotheses in the bug report: the
spectrum is an exact convolution of the actual curve (nothing lost there),
and `compute_losvd_moments` correctly measured the actual curve's actual
moments (nothing wrong there either).

**The real mechanism:** `(V, sigma, h3, h4)` are the *coefficients* of a
truncated Gauss-Hermite series, not literally the resulting density's mean,
standard deviation, skewness, and kurtosis. Those two descriptions coincide
only as `h3, h4 -> 0`. Measured on a clean grid (V=100, sigma=200, no
clipping in any of these rows):

| h3 | h4 | dV (km/s) | dsigma (%) |
|---|---|---|---|
| 0.00 | 0.05 | +0.0 | +9.5 |
| 0.03 | 0.00 | +9.7 | -1.0 |
| 0.05 | 0.00 | +15.7 | -0.6 |
| 0.08 | 0.00 | +23.8 | +0.5 |
| 0.10 | 0.10 | +32.9 | +18.0 |

h3 (the odd/antisymmetric term) shifts the mean; h4 (the even/symmetric term)
shifts the width; both keep growing with no sign of saturating over this
range, and the two effects add roughly independently. This is a real,
well-known property of the Gauss-Hermite expansion (van der Marel & Franx
1993) -- not a defect in this package -- but its *magnitude* over the
parameter ranges convenient for randomized testing was not previously
quantified here, and it is large enough to dominate almost any test that
uses it carelessly.

**Clipping is a real, separate, smaller effect on top of this.** It only
matters when the raw curve goes negative (large negative h4 combined with
h3, mostly), and in the ranges checked below it stays under ~4% of mass even
at the extremes -- second-order compared to the parameterization mismatch
above.

## How big a problem this is for the project's existing randomized ranges

500 random draws, sigma 40-400 uniform, h3/h4 +/-0.1 uniform (the "wide"
preset, matching `temp_testing.ipynb`'s own ranges):

| | median | p90 | max |
|---|---|---|---|
| \|V shift\| (km/s) | 10.6 | 34.4 | 59.2 |
| \|sigma shift\| (%) | 7.2 | 15.4 | 19.2 |
| clipped mass (%) | 0.12 | 2.74 | 4.32 |

**53% of draws have \|V shift\| > 10 km/s; 68% have \|sigma shift\| > 5%,
from the generator alone.** Even the tighter "safe" preset (h3/h4 +/-0.05,
sigma 40-300) still has 22%/46% respectively -- smaller, but still not a
small-perturbation regime.

## What this does NOT invalidate

The user's own prototype already computes bias the correct way:
`bias = recovered_moments - moments_truth`, where `moments_truth` comes from
`compute_losvd_moments` on the actual truth curve, not from the nominal
generator parameters. Re-reading the two flagged examples through that lens:

```
example 1: truth (v=120.85, sig=303.41, h3=+0.054, h4=-0.061)
           recovered (v=113.49, sig=309.92, h3=+0.052, h4=-0.050)
           bias: dV=-7.4 (2.4% of sigma), dsigma=+6.5 (2.1%), dh3=-0.002, dh4=+0.011
           -> a genuinely good fit.

example 2: truth (v=42.32, sig=456.82, h3=-0.110, h4=+0.070)
           recovered (v=41.37, sig=438.47, h3=-0.087, h4=+0.017)
           bias: dV=-1.0, dsigma=-18.4 (-4%), dh3=+0.023, dh4=-0.053
           -> h4 shrinkage consistent with the noise-floor mechanism
              already established in dev_notes/ghcore/FINDINGS.md; not
              obviously anomalous once S/N (not shown in the printed
              summary) is accounted for.
```

Neither example is evidence of a bug once scored correctly. What the finding
*does* mean: this project's earlier fixed-mock bias measurements (h3<=0.03,
h4<=0.05, sigma<=90-140) sat in a regime where the generator-mismatch effect
was modest (a few km/s / a few percent) and easy to miss. Extending
randomized testing into h3/h4 up to +/-0.1 and sigma up to 400 makes the
effect dominant, and any bias number quoted in that regime **must** be
computed against the truth curve's own moments, never the nominal
parameters, or the result is meaningless.

## What was built in response

A reusable framework, `dev_notes/validation_suite/`:

- `mocks.py` -- mock generation with the correct truth (truth-curve moments,
  never nominal), a `generator_mismatch` diagnostic on every case, a
  `clipped_mass_frac` diagnostic, two labeled randomization presets (`safe`:
  h3/h4 +/-0.05; `wide`: +/-0.1, for deliberate stress-testing), and an
  alternative spline-based truth generator (`build_spline_truth`) that skips
  the Gauss-Hermite step entirely, for a future clean test of translation
  loss through the fitting basis alone.
- `runner.py` -- fits one randomized case, returns one standardized,
  flat result row (nominal params, truth-curve moments, generator mismatch,
  recovered moments, bias vs. truth, lam_used, success, timing).
- `sweep.py` -- a single reusable CLI: fix one named axis over a grid,
  randomize every other axis fresh per replicate, write incrementally to
  CSV. Reproducible given `--seed` (independent draws across replicates,
  not lack of reproducibility).
- `examples/notebooks/09_validation_suite_results.ipynb` -- standardized
  plots for any sweep CSV: fit bias vs. swept axis (median + IQR band)
  stacked directly above the generator-mismatch diagnostic on comparable
  axes, plus a Gaussian-only control sweep (h3=h4=0 by construction, so
  generator mismatch is exactly zero) to isolate pure fitting bias from the
  GH-parameterization effect, plus a lambda-selection-stability panel.

## Verified

- `sweep.py` end-to-end smoke-tested (4-row run) and confirmed
  bit-for-bit reproducible given the same `--seed`.
- Real sweeps run: sigma, S/N, h3, h4 (`safe` preset, 15 randomized
  replicates per grid point) plus a Gaussian-only sigma control (10
  replicates). Results and plots in the notebook above.

## Open follow-up (not yet built)

- `build_spline_truth` is implemented but not wired into `sweep.py`'s CLI --
  the natural next test is a spline-truth sweep to see whether the
  fit-vs-truth bias measured above is specific to LOSVDs generated in GH
  form, or identical for a truth that was never a GH curve at all.
- 2D combined sweeps (e.g. sigma x S/N jointly) to look for interaction
  effects, rather than one axis at a time.

---

## Addendum, same day: spline-truth generator wired into the CLI

Per discussion: rather than scrap GH truths, keep both as first-class,
differently-purposed tools. `sweep.py` now takes `--truth-kind {gh,spline}`.
`spline` builds truths directly via `mocks.build_spline_truth` (no GH step at
all -- see that function's docstring), sweeping a `wiggle_scale` axis in place
of h3/h4 (`--axis wiggle_scale`, invalid to combine with `--truth-kind gh` and
vice versa for `h3`/`h4`, both enforced with a clear CLI error rather than a
silent no-op). `generator_mismatch` means something different for spline
truths (incidental h3/h4 introduced by the random log-space wiggle relative to
a nominal (0,0) Gaussian core, not a GH-coefficient-vs-moment gap) -- every row
carries `truth_kind` explicitly so the two are never conflated downstream.

Smoke-tested: reproducible, `clipped_mass_frac` correctly always 0 (exp(.) is
never negative by construction), incidental h3/h4 from the wiggle are small
at the default `wiggle_scale` range (0.05-0.25) with an occasional bimodal
draw at the high end (flagged via `n_local_maxima`).

Comparison sweeps (sigma, S/N; GH vs. spline truth, identical fitter/grid/
replicate count) launched to answer directly: does the truth-generator family
itself change measured fitting bias, once bias is always scored against the
truth curve's own moments? Recommended framing going forward regardless of
the answer: GH truths for literature/legacy comparability and as a physically
motivated (anisotropy-shaped) stress test; spline truths as the primary tool
for isolating pure fitting/regularization bias with no generator confound.
## Results (2026-07-28, 525 total randomized fits across 7 sweeps)

All sweeps completed at 15 (GH) / 10 (spline, gaussian-control) randomized
replicates per grid point: sigma-safe (75), S/N-safe (105), h3-safe (75),
h4-safe (75), Gaussian-only sigma control (50), sigma-spline (50), S/N-spline
(70).

**Direct GH-truth vs. spline-truth comparison, identical fitter/grid/reps,
only the truth generator differs:**

| sweep | median \|sigma bias\|, GH truth | median \|sigma bias\|, spline truth |
|---|---|---|
| sigma (40-300 km/s) | 2.71 km/s | **1.35 km/s** |
| S/N (20-300) | 4.50 km/s | **3.69 km/s** |

Spline truths give smaller median bias in both sweeps -- roughly half in the
sigma sweep, modestly smaller in the S/N sweep. Visually (see notebook 09's
"GH truth vs. spline truth" section): in the sigma sweep the spline-truth
median stays close to flat near zero across the whole range, while the
GH-truth median drifts to -3 to -4 km/s at sigma >= 220. In the S/N sweep the
two track each other closely -- there, the noise floor dominates regardless
of which family generated the truth, exactly as expected (S/N sets the
information limit; the generator's own shape is second-order next to that).

**Independent confirmation from a completely separate comparison**: the
Gaussian-only control (h3=h4=0 by construction) vs. the full-GH-shape sweep
(same axis, different random seeds/dataset entirely) shows the identical
qualitative pattern -- the Gaussian-only median stays flatter, the full-shape
median trends negative at high sigma. Two independent datasets agreeing on
the same shape-dependent trend is real signal, not a sampling fluke of one
sweep.

**Reading on the original question ("should we scrap GH entirely"):** this is
modest, genuine evidence that GH-shaped truths are somewhat harder for this
fitter to recover cleanly than an arbitrary spline-native shape of comparable
"structuredness" -- not just the previously-established coefficient-vs-moment
bookkeeping issue, but a real (if modest, ~2x at most) fitting-difficulty
gap. That supports the agreed framing: spline truths as the *primary* tool
for isolating pure fitting/regularization bias going forward, GH truths kept
as a first-class option for literature/legacy comparability and as a
physically-motivated (anisotropy-shaped) stress test -- not scrapped, but no
longer the default lens for "how good is the fitter itself."

**h3/h4 sweeps directly confirm the shrinkage-toward-Gaussian mechanism**
documented elsewhere in this project (`dev_notes/ghcore/FINDINGS.md`,
notebook 02 Sections 5-6): both `h3 bias` and `h4 bias` (recovered minus
truth curve) cross zero near nominal 0 and grow in the direction that pulls
the recovered curve toward zero shape -- e.g. h4 bias goes from about +0.04
at nominal h4=-0.05 to about -0.017 at nominal h4=+0.05, monotonically. The
generator-mismatch panels (V/sigma mismatch vs. nominal h3/h4) grow with
|h3|/|h4| exactly as the earlier analytic calculation predicted, and are
larger in magnitude than the fit-bias panels for V and sigma across most of
the swept range -- a direct, real-data confirmation (not just the earlier
zero-clipping analytic argument) that nominal-vs-truth mismatch dominates
fit bias over this parameter range.

**Two bugs found and fixed while running this** (logged per standing rule):
a classic argparse gotcha (`--values -0.05,...` reads the leading `-` as an
unrecognized flag; fixed by requiring `--values=...` for negative-leading
grids, now documented in `sweep.py`'s own docstring) which, combined with
`set -e` in the driving shell script, silently aborted a full run before the
h4 and Gaussian-control sweeps ever executed; and a path bug in the analysis
notebook (`RESULTS_DIR` was relative to the wrong directory once the
notebook was placed in `examples/notebooks/` rather than
`dev_notes/validation_suite/`), caught by inspecting the notebook's actual
executed output (an empty file list) before reporting anything, not by
assuming a clean `nbconvert` exit code meant correct results.

---

## Addendum, 2026-07-29: the generator-mismatch trends are exact, closed-form, not "cube root"/empirical fits

The user asked (looking at notebook 09's h3/h4-sweep plots) whether the
generator-mismatch trends follow a known mathematical relationship, and
specifically guessed a cube-root shape for V/sigma/h3 and a
parabolic/quartic shape for h4. Verified directly (`gauss_hermite_model`'s
own algebra, no fitting, no clipping) rather than fit empirically to the
plot:

**V-shift from h3 is EXACT, not leading-order, for the raw (unclipped)
curve**: because `gauss_hermite_model` is exactly linear in h3, and this
package's `_fh3(w) = (2w^3-3w)/sqrt(3)` decomposes in the probabilists'
Hermite basis as `[2*He_3(w) + 3*He_1(w)]/sqrt(3)`, the raw first moment is

    dV_raw = sqrt(3) * sigma * h3         (exact, any h3, zero clipping)

Verified numerically to 6 significant figures against the raw trapezoidal
integral. The analogous exact result for h4 (using
`_fh4(w) = [4*He_4(w)+12*He_2(w)+3*He_0(w)]/sqrt(24)`):

    sigma_raw = sigma * sqrt[(1 + h4*27/sqrt(24)) / (1 + h4*3/sqrt(24))]

also verified exact (not approximate) against the raw integral across
h4 in [-0.05, 0.1].

**Why the plotted trend looks concave/"cube-root-like" instead of a clean
line**: clipping (`b_truth = max(raw, 0)`, needed once the skew term drives
the curve negative) pulls the shift back down, more so as |h3| grows and
more mass needs clipping. Decomposed at V=0, sigma=140, h3=0.1 (the exact
case this module's docstring already quotes "~20.6" for):

| stage                                  | dV (km/s) |
|-----------------------------------------|-----------|
| exact raw formula (`sqrt(3)*sigma*h3`)  | 24.249    |
| + clipping (`max(raw,0)`), no taper     | 20.637    |
| + `compute_direct_moments` smooth taper (the actual pipeline) | 20.006 |

So the module docstring's "~20.6" was the clipped-but-untapered number, not
the full pipeline output (20.0) -- both are real, clipping is the dominant
second-order correction, the taper only a small further one. The observed
sub-linear bending at large |h3| is this clipping correction, not a
fractional power law -- confirmed by fitting `dV = a1*h3 + a3*h3^3`, which
removes >98% of the residual (vs. plain linear leaving ~9%), and by the
local log-log power-law index of dV vs h3 -> 1.000 as h3 -> 0 (exactly
linear at the origin, the opposite of a cube-root's infinite slope there).

**The h4-in-the-h3-sweep "parabolic, ~-x^4" trend is a real, exact,
independent effect, not a fitting artifact**: even a purely-skewed raw
curve (h4=0 exactly) has genuine nonzero excess kurtosis when measured
around its own (h3-shifted) true mean -- an unavoidable consequence of
computing a 4th central moment around a shifted 1st moment, nothing to do
with clipping, the taper, or `fit_h3_h4`. Measured directly on the raw
curve (V=0, sigma=140, h4=0 exactly):

| h3   | raw excess kurtosis around raw mean |
|------|--------------------------------------|
| 0.02 | -0.0193 |
| 0.05 | -0.1222 |
| 0.08 | -0.3216 |
| 0.10 | -0.5159 |

Scales cleanly as `h3^2` (ratio to `h3=0.02`'s value: 6.4x at 2.5x h3,
16.9x at 4x h3, close to the 6.25x/16x a pure quadratic predicts) --
matching the user's own "parabolic, ~-x^4" read of the plotted `dh4` panel
in the h3 sweep (the pipeline's actual `dh4` through `fit_h3_h4` is smaller
in magnitude than this raw excess-kurtosis number, since that estimator
isn't a literal normalized 4th moment, but carries the same sign and shape).
By parity, no analogous h4->h3 or h4->V cross-term exists (h4's basis terms
are all even in `w`; confirmed numerically, `dh3`/`dV` stay at floating-
point zero across the whole h4 sweep at h3=0).

**The Gaussian-only control sweep's near-zero-scatter panels are correct,
not suspicious**: with h3=h4=0 exactly, `dV` is a deterministic floating-
point zero (~1e-14) because the curve is exactly symmetric and there is no
stochastic ingredient in this quantity at all (computed before noise is
even added) -- zero replicate-to-replicate scatter is expected, not an
error-bar-collapse bug. The `dsigma` "perfect negative linear trend with no
error bars" the user flagged is real and deterministic too: it is
`compute_direct_moments`'s own smooth-taper truncation acting on an exactly
Gaussian curve, and is an EXACT, constant *fraction* of sigma
(-1.2209% at every sigma in {40,80,140,200,220,300} tested, to 4 significant
figures) -- because the taper's `core_threshold=1%`-of-peak cutoff always
falls at the same number of sigmas (~3.03 sigma) regardless of the
absolute scale, a fixed fractional-variance truncation is exactly what a
self-similar (Gaussian) shape produces. Not a bug; a clean, fully
characterized property of the default taper settings.

---

## Addendum, 2026-07-29: `build_spline_truth` promoted to the package, now the default mock-truth generator everywhere

Following directly from the finding above (and the exact/closed-form
mismatch derivation), `build_spline_truth` (previously only this suite's
own `mocks.py`) is now `kinextract_lite.build_spline_truth` (lives in
`losvd.py`, next to `build_bspline_basis`), and `build_mock_cube` (the
production synthetic-cube generator used by `04_template_calibration.ipynb`
and any downstream notebook) defaults to it via a new `truth_kind` param
(`"spline"` default, `"gh"` explicit opt-in). `wiggle_scale=0.0` gives a
pure Gaussian core -- the spline-truth analog of `true_h3=true_h4=0` --
so h3/h4 recovery stays meaningful to validate against a near-zero-shape
truth, per the user's explicit ask.

**A real bug found and fixed while wiring this up**: unifying the two
`truth_kind`s' normalization (both computing `truth_moments` from a
trapz/unit-area curve, the correct convention for reporting moments)
initially also changed the curve used to build the noiseless *flux*
spectrum to unit-area -- silently rescaling the mock's absolute flux
amplitude by roughly `1/dv` (the velocity grid's bin spacing, ~7x for the
default grid) relative to the historical `.sum()`-normalized convention.
Since `noise_floor` is an absolute (not S/N-relative) constant, this
tanked the default synthetic cube's S/N enough that `voronoi_2d_binning`
outright failed ("not enough S/N in the whole set of pixels") on the
pPXF-comparison notebook built earlier this session. Fixed by keeping a
separate unit-SUM curve (`b_true_flux`) for the flux-amplitude
calculation specifically, matching this function's original,
long-standing behavior exactly regardless of `truth_kind` -- `truth_moments`
and `truth["b_true"]` stay properly unit-area (trapz) normalized. Caught
by noticing the downstream notebook's S/N collapsed, not by inspection --
a reminder that a normalization choice that looks like a harmless
cleanup can silently move an absolute-scale quantity three steps away.

**Left deliberately on `gauss_hermite_model` truth, not converted**: any
test or notebook section whose actual subject under test is the GH
parameterization or a mechanism calibrated specifically against it --
`tests/test_gh_moments.py` (tests `gauss_hermite_model`/`fit_gauss_hermite`
themselves), `tests/test_bias_correction.py` (its calibration table is
itself GH-parameterized), `02_realistic_mock_fit.ipynb` Sections 1-8
(the whole point is characterizing the GH-mismatch and its downstream
consequences -- Section 9 is the spline-truth complement, not a
replacement), and `temp_testing.ipynb` (the original ad-hoc notebook that
discovered this whole mismatch, via randomized GH parameters
specifically). Converting these would remove the thing they demonstrate,
not just change how their mock is built. All other general-purpose
fit-mechanics mocks across 12 test files and the `01`/`02`(Section 9)
example notebooks were switched.
