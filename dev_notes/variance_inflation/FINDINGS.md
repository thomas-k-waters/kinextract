# Variance inflation: implementing the never-built "rescale before clip" hierarchy, and promoting it to the default

**Date: 2026-08-05**

## The question

`PARAMETER_GUIDE.md` has said for a while that variance rescaling should
come before clipping, with `fit_losvd_clean`'s hard sigma-clip reserved
for genuine, localized per-pixel outliers -- but nothing in the package
actually implemented the "rescale" half of that hierarchy. On real data,
`fit_losvd_clean(robust_scale=True)` was the only defense against
unmodeled contamination (sky-subtraction residuals in particular), and it
is a binary keep/reject decision per pixel: real chi2_red on real MUSE
data commonly runs 11-28 (`real_data_sigma/FINDINGS.md`), meaning the
input noise vector is *structurally* under-scaled at many wavelengths,
not just at a handful of genuinely bad pixels -- exactly the regime where
outlier rejection can compound a model bias rather than fix contamination
(the pixels disagreeing most with an already-biased fit are often exactly
the ones correctly arguing against it).

Separately: this session's own ZAP investigation (`zap/FINDINGS.md`)
found a real, hard limit for at least one target -- NGC 4751 (MUSE NFM):
masked-source ZAP self-calibration still destroys real signal at the
galactic center on a galaxy-filling field with no dedicated sky exposure
or blank-sky region for the SVD eigenbasis to train on, confirmed across
several parameter combinations, consistent with Soto et al. (2016)'s own
stated requirement for one or the other. ZAP is not a fix for this
target; a fit-level (not cube-level) defense against sky-residual
contamination was still needed.

## The method: per-spectrum, residual-based variance inflation

Not a blanket wavelength list shared across every spectrum -- sky-residual
amplitude scales with a bin's own S/N and falls off with radius, and the
previous approach some real-data notebooks used (`SKY_RESIDUAL_WINDOWS`,
a small fixed window list found empirically on one target, NGC 5102,
never cross-checked against a real airglow atlas -- `VALIDATION_CAMPAIGN.md`
Stage 4's own open TODO) is applied identically regardless of a given
spectrum's own contamination level.

`kinextract.variance_inflation.fit_losvd_inflate`, per bin:

1. Preliminary fit with a loose, cleanup-only clip (catches genuine
   single-pixel spikes without doing the heavy lifting).
2. Whitened residual at *every* pixel (not just the surviving subset)
   from that fit's own effective model.
3. A median-filtered local scale of `|residual|` in wavelength -- smooths
   past single-pixel shot noise while preserving sky-line-width features
   (OH residuals are a few pixels wide at MUSE's resolution, not
   single-pixel).
4. That local scale becomes a per-wavelength inflation factor on the
   *input* error array (floored at 1 -- never deflate; capped, so a
   badly-contaminated window is downweighted, not zeroed out entirely).
5. Re-whiten with the inflated errors, refit -- this final fit is the one
   returned. A second loose clip pass on top is still applied (cleanup,
   not primary defense).

No change to the fitting engine itself: this wraps `whiten` +
`fit_losvd_clean` in an outer loop, the same architectural pattern
`fit_losvd_clean` itself already uses, so it composes with continuum-
cofitting, AR(1) whitening, and parallel per-bin fitting unchanged.
`fit_losvd_inflate_parallel` mirrors `fit_losvd_clean_parallel` exactly
(same `ProcessPoolExecutor`/`spawn`/memory-aware-worker-count pattern,
same per-bin failure isolation).

## Real-data validation (NGC 4751, MUSE NFM, raw non-ZAP cube)

Full record: `EHT_Project/notebooks/N4751_notebooks/kinematics/
08_variance_inflation_development.ipynb` (6-bin subset development +
18-trial injection-recovery test) and `09_sky_correction_full_galaxy_
comparison.ipynb` (full 113-bin comparison against the previous default,
`fit_losvd_clean(clip_sigma=3.0, robust_scale=True)`, identical binning/
templates/masking otherwise).

Full-field numbers (113 bins):

- **Median whitened residual RMS: 1.41 -> 0.96** (ideal is ~1) -- the
  inflated method lands almost exactly at the ideal even though the
  baseline already benefits from continuum-cofitting, emission masking,
  and correlated-noise whitening on top of its own hard clip.
- **Bins tripping `fit_losvd_clean`'s own >10%-clipped sanity warning:
  22/113 -> 0/113.**
- **RMS-vs-radius**: baseline climbs from ~1 near the nucleus to ~10-20
  at the largest radii (matching the original chi2_red-explodes-with-
  radius finding this investigation started from); the inflated method
  stays mostly under ~2-3 even at r~3", a real, radius-dependent
  improvement.
- **Recovered kinematics (V, sigma, h3, h4) track the previous default
  closely at essentially every radius** -- this method changes how
  well-calibrated the fit's own residuals are and how much of the
  spectrum survives to inform it, not the recovered physics.

## What is NOT (yet) shown

- **No demonstrated reduction in absolute kinematic bias.** An 18-trial
  injection-recovery test (known LOSVD injected onto a real bin's own
  leftover contamination+noise background, real template mixture/
  continuum reused from that bin's own baseline fit) found comparable
  bias magnitude between the two methods -- both carried a real,
  tens-of-km/s offset from the injected truth, attributed to factors
  other than sky-residual handling (regularization choice, template
  mismatch, both already open elsewhere in this package's own dev notes).
  Treat this as a calibration/robustness improvement, not a demonstrated
  accuracy improvement, until a larger injection-test sample says
  otherwise.
- **Not a pure sky-line isolator.** At low S/N/high radius, the
  per-spectrum inflation profile saturates the cap across large,
  contiguous stretches of the fit window, not just at a few discrete
  known-bad wavelengths -- it behaves as a general adaptive variance
  recalibration in that regime (consistent with real chi2_red there
  reflecting more than just sky lines: template mismatch, an
  under-scaled base noise vector, continuum-shape residual the
  multiplicative spline doesn't fully absorb). At high S/N/low radius,
  the profile is close to flat with a few narrow bumps, some of which do
  coincide with the old NGC 5102 `SKY_RESIDUAL_WINDOWS` list -- a real,
  if partial, cross-target/cross-instrument-mode validation of that list,
  found without telling the new method about it.
- **Single target, single instrument mode.** Validated on NGC 4751 (MUSE
  NFM) only. `examples/notebooks/04_template_calibration.ipynb`/
  `07_full_galaxy_analysis.ipynb` (NGC 5102, MUSE WFM) now use this as
  their own default too (see below), which is additional real-data
  exposure but not yet a from-scratch validation on that target the way
  NGC 4751 got one.

## Post-promotion correctness check: wrong-cube scare, and the re-verification it triggered

**2026-08-05, later the same day.** While wiring `fit_losvd_inflate` into
`04_template_calibration.ipynb`/`07_full_galaxy_analysis.ipynb` (NGC 4751)
as the default, found that both notebooks' *data-source* cells (not
touched by the earlier promotion work, which only changed the *fitting*
logic) still pointed at `NGC4751_zap_repaired.fits` -- `04` auto-preferred
it whenever present, `07` had it hardcoded. Every N4751 run up to that
point in this promotion effort was silently using the ZAP-cleaned cube,
despite ZAP already being ruled out for this target (see the ZAP section
above). Both fixed to use the raw cube unconditionally; `04`/`07` re-run
in dependency order.

This raised a real question: did the `08`/`09` validation notebooks
(the ones that actually produced this document's own headline numbers)
have the same problem? Checked directly: **no** -- both `08` and `09`
always fit their per-bin spectra from the raw cube. But both load
`ngc4751_template_subset.npz`/`ngc4751_radial_kinematics.npz` (`04`'s own
output), which they never regenerated themselves -- and per this same
`FINDINGS.md`'s own zap section, ZAP+repair was already "wired into
`04_template_calibration.ipynb`... for NGC 4751" *before* this session
started, so the template subset (V) `08`/`09` had been using was almost
certainly conditioned on the ZAP cube, not the raw one -- a second-order
contamination pathway through template selection and grid-centering, not
through the fitted spectra themselves.

Given the baseline/inflated comparison in `08`/`09` is paired (both arms
share the same template subset and grid centers, differing only in
cleaning method), the relative conclusion was expected to be robust to
this -- but "expected" isn't "verified," and this document's own
headline numbers are the entire evidence base for the promotion decision
above. Re-ran both `08` and `09` against the now-corrected (raw-cube,
5-template) subset:

- **`09`'s full 113-bin headline numbers were unchanged to two decimal
  places**: median whitened RMS 1.41 -> 0.96 (was 1.41 -> 0.96), bins
  tripping the 10%-clipped sanity warning 22/113 -> 0/113 (unchanged).
  Per-bin RMS numbers in `08`'s own 6-bin subset were within 0.01 of the
  original run.
- **`08`'s own small-sample (18-trial) injection-recovery bias numbers
  did change materially**: sigma bias MAD shrank from 12.3/9.1 km/s
  (baseline/inflated) to 6.3/2.5 km/s -- both methods' bias against known
  truth is *smaller* with the corrected template subset, consistent with
  template-conditioning quality mattering for absolute bias independent
  of which sky-correction method is used. The qualitative conclusion
  (neither method clearly better on absolute bias, in this small sample)
  is unchanged; `08`'s own conclusion cell was updated with the new
  numbers.

**Takeaway**: the promotion decision itself was not built on the wrong
cube -- the paired-comparison design in `08`/`09` made the headline
finding insensitive to the template-subset issue, confirmed rather than
assumed. The lesson for next time: when fixing a data-source bug in one
notebook, check every *other* notebook that consumes that notebook's
saved output, not just the ones that load the raw cube directly.

## Promotion

Promoted from a `kinextract_test_suite` scratch prototype
(`kinextract_test_suite/notebooks/kinextract_scratch_notebooks/
variance_inflation.py`, left in place -- it underpins the N4751 `08`/`09`
notebooks' own reproducibility, unchanged) into
`src/kinextract/variance_inflation.py` as a first-class module, exported
at the top level (`fit_losvd_inflate`, `fit_losvd_inflate_parallel`,
`compute_local_inflation`, `reconstruct_continuum`, `full_pixel_model`).
`tests/test_variance_inflation.py` added, mirroring `test_cleaning.py`'s
own conventions. `PARAMETER_GUIDE.md` updated: `fit_losvd_inflate` is now
the recommended default for contamination whose wavelength/severity
varies per spectrum; `fit_losvd_clean` alone remains reasonable only when
there's a specific, independent reason to suspect a handful of genuine
per-pixel outliers and nothing suggesting the noise vector itself is
structurally miscalibrated. `examples/notebooks/07_full_galaxy_analysis.
ipynb` (NGC 5102, the capstone real-data notebook) switched to the new
default and re-run end to end. `04_template_calibration.ipynb` was
deliberately left on `fit_losvd_clean` -- its own per-bin fits are
against the full, un-conditioned E-MILES grid (hundreds of templates,
vs. `07`'s already-conditioned handful), so re-running it under the new
method costs far more than `07` did for a step whose actual job (coarse
NNLS-weight-survival template selection) is not obviously sensitive to
the same calibration issue the final kinematic extraction is. Revisit if
a specific reason to suspect template selection itself is affected shows
up.
