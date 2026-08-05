# ZAP sky-residual cleaning: root cause, fix, and real-data validation

**Date: 2026-08-04**

## The question

Get ZAP (Zurich Atmosphere Purge, Soto et al. 2016) genuinely working as
a real preprocessing step for both NGC 5102 (WFM, low sigma) and NGC
4751 (NFM, high sigma) -- replacing/supplementing the fixed
`SKY_RESIDUAL_WINDOWS` list every real-data notebook currently hardcodes.
Explicit requirement: check that it does not remove real signal from the
galaxy center, and confirm it actually reduces sky-residual contamination
in the windows that matter.

This was previously an "OPEN" item in `HISTORY.md`: ZAP's automatic
self-calibration was found to visibly destroy real spectral structure on
NGC 5102 (roughness 4-56x worse, worse at higher source brightness),
attributed to the same-cube "sky" training region containing real (faint,
extended) galaxy light. That attribution turns out to be only half the
story.

## Masking the SVD/eigenbasis is real, and not sufficient

Confirmed directly from `zap.py`'s own source (not assumed): `mask`
(sky=0, source>=1) protects **both** steps a same-cube self-calibration
needs protecting -- the zeroth-order `zlevel` subtraction and the SVD
eigenbasis. When `mask` is given, `zap.process` computes a separate
`SVDoutput(..., mask=mask)` object (source pixels NaN'd out before the
SVD), then the real, full-cube `Zap` object's own `_run(...,
extSVD=extSVD)` call uses `extzlevel=extSVD` too -- so the final zlevel
step is *also* taken from the masked object, not recomputed on the
contaminated full cube.

A source mask was built via `photutils.segmentation.detect_threshold` +
`detect_sources` on the white-light image, dilated (`nsigma=2.0,
npixels=30, grow=15`) to include faint wings the raw detection would
miss (`kinextract_test_suite/notebooks/kinextract_scratch_notebooks/
zap_cleaning.py`'s `build_source_mask`). Real sky-fraction check before
committing to a full run: 76% for NGC 5102 (WFM, 60"x60" field), 73% for
NGC 4751 (NFM) -- the NFM field is tiny (~8.3"x8.5", confirmed via
`CD1_1`/`NAXIS` header math) but was **not** the "filled field" blocker
initially worried about; plenty of genuinely source-free sky survives
even a fairly aggressive dilation.

**Masking closes the eigenbasis-contamination pathway, but a second,
separate corruption survives it anyway.** Confirmed directly on a real,
masked full-cube NGC 5102 run: a single bright-nucleus pixel's flux
swings by up to several tens of thousands between adjacent wavelength
pixels post-ZAP, versus a real measured per-pixel noise of ~600-700
there (peak deviation ~140-540 sigma across two independent runs,
median ~23-85 sigma in a 200-pixel diagnostic window) -- an obvious
numerical artifact, not real structure or plausible noise.

## Root-causing the residual corruption

Ruled out (checked directly, not assumed):

- **Not a data-quality issue.** No NaN or non-positive `STAT` values at
  the affected pixels; flux/error scaling is completely ordinary
  (S/N ~90-120 there, smoothly continuous with the rest of the field).
- **Not primarily eigenvector-count-driven.** A 4-way crop-based
  parameter sweep on a 121x121-spaxel region around the NGC 5102
  nucleus: fixing `nevals=3` barely changed the corruption (roughness
  ratio 23.87 vs baseline's 25.91) -- matching `HISTORY.md`'s own
  earlier, *unmasked*-case finding that eigenvector count alone doesn't
  explain it.
- **Direction that made it worse, not better:** `cfwidthSP=300`
  (less aggressive continuum removal before the per-spaxel eigenvalue
  fit) made the corruption *far* worse (median deviation 85 sigma vs the
  baseline `cfwidthSP=40`'s 23 sigma) -- the doc-recommended smaller
  `cfwidthSP` ("better traces sources," 20-50 pixel range) is the
  better of the two, but does not eliminate the effect.

**The corruption is real, but spatially compact and brightness-linked --
on NGC 5102.** A brightness-percentile sweep across the same crop (5th
through 99.9th percentile) found ZAP roughness ratio *improves* (0.67-0.99,
i.e. genuinely better than the original) at every percentile from 5th
through 99th, then jumps to 5.45x *worse* specifically at the 99.9th
percentile. On the real full-cube run, the corrupted-spaxel set (finite
roughness ratio > 1.5, see "the fix" below) is 96% within 1 arcsec of the
true brightest pixel, with a median flux ~125x the field median -- tightly
localized to the extreme nuclear core, not diffuse. Working hypothesis,
not confirmed at the ZAP-internals level: per-spaxel eigenvector-weight
least-squares overfitting at very high S/N, independent of whether the
eigenbasis itself is clean.

**On NGC 4751 NFM, the failure mode is different.** The true brightest
pixel there came out essentially untouched (roughness ratio ~1.01,
*not* flagged). The ~36 spaxels that *did* get flagged sit 2.6-5 arcsec
from the peak, at only modestly-above-median brightness (~1.5x the field
median, nowhere near NGC 5102's ~125x) -- consistent with a distinct
mechanism specific to this target: NGC 4751 has a real dust lane
(differential extinction across the field, already handled elsewhere in
this project via an explicit exclusion mask,
`masked_coordinates.txt`), and the corrupted spaxels are plausible
candidates for differential-extinction-driven spectral-*shape* outliers
rather than raw brightness. Not confirmed further this session -- the
practical fix below doesn't need to know the mechanism to work.

## The fix: detect and revert, don't fight ZAP's internals further

`repair_corrupted_spaxels` (`zap_cleaning.py`): a cheap, vectorized,
per-spaxel roughness diagnostic (std of first differences / median flux,
in a rest-frame line-free continuum window -- `8600-8800` Angstrom for
both targets here, chosen to avoid the flagged `SKY_RESIDUAL_WINDOWS`
themselves, since real flux *should* change there) computed for both the
original and ZAP output, normalized by the original's own median so the
ratio is apples-to-apples. Spaxels with a finite ratio above 1.5 get
reverted to the original (pre-ZAP) data; the threshold was not finely
tuned -- the crop sweep's own percentile check showed a clean separation
(genuinely-fine spaxels sit at 0.4-1.2, corrupted ones at several to tens).

**One real bug caught and fixed mid-investigation**: near-zero-flux
background spaxels give an ill-defined (NaN) roughness ratio -- that
means "not enough signal to check," not "ZAP corrupted this spaxel." An
early version of the revert criterion treated NaN as corrupted and
inflated the revert count from the true 48-53 spaxels to ~4900 (nearly
5% of the field) by sweeping in irrelevant background pixels. Fixed:
only revert on a *finite* ratio exceeding the threshold.

## Real-data validation results

Both targets: real full-cube ZAP run (`nsigma=2.0, npixels=30, grow=15`
mask; `cfwidthSVD=300, cfwidthSP=40`, automatic `nevals`; `ncpu=1` --
see [[feedback_memory_heavy_jobs]] for why not more), then
`repair_corrupted_spaxels`, then S/N and sky-residual-window checks
using `kinextract`'s own `extract_aperture_spectrum` at several
apertures.

**NGC 5102 (WFM)**: 73 auto-selected eigenspectra; 53/100806 spaxels
(0.053%) reverted. Nuclear r=2px aperture: S/N ratio exactly 1.0000
(byte-identical to original there, as intended). r=5/15/40px apertures:
S/N ratio 0.99/0.97/0.91 (mild, monotonic decline with radius --
present even in unreverted regions, likely a real, modest zlevel/
background correction rather than corruption). Sky-residual-window RMS,
repaired vs original: 5 of 8 (window x aperture) combinations improved,
some substantially (67-92%); 2 of 8 got mildly worse (nuclear-aperture
`8414-8419` -47%, outer-aperture `8711-8714` -59%) -- a real,
reproducible pattern (confirmed across two independent full-cube runs
with matching parameters), not a fluke. **One hypothesis checked and
ruled out**: multi-segment wavelength splitting. `zap.py`'s own `SKYSEG`
(the fixed list of segment boundaries) defaults to an empty list, so
both the small test crop and the real full-cube runs use exactly one
segment covering the whole spectral range (confirmed directly in both
runs' own logs: `Calculating SVD on 1 segments`) -- not a plausible
explanation after all. Real cause of the crop-vs-full-cube discrepancy
not identified this session; most likely candidate not yet checked: the
full field's much larger and more spatially diverse sky-pixel
population (vs. the crop's tight ~12" box) changes what the auto-selected
eigenspectra (73 vs. 39 components) actually encode, and a specific
narrow window's reconstruction quality can genuinely shift between
field sizes without either being "wrong."

**NGC 4751 (NFM)**: 28 auto-selected eigenspectra; 36/111540 spaxels
(0.032%) reverted, none within 5px of the fitted kinematic center (that
region was never corrupted here -- see above). S/N ratio at the fitted
center: 0.99 (r=2px) down to 0.90 (r=40px) -- same mild, radius-
dependent pattern as NGC 5102. Sky-residual-window RMS: nuclear
aperture 3 of 4 windows *worse* (-7% to -57%), only `8748.5-8753` improved
(35%); outer aperture 3 of 4 improved (33-86%), one worse (-25%). A
more mixed result than NGC 5102's -- reported honestly rather than
smoothed over. Given this target's own dust lane and the different
corrupted-spaxel character found above, this may be a genuinely harder
case for ZAP than NGC 5102, not a bug in the repair step itself (which
correctly left the fitted-center aperture provably unmodified, since no
spaxels there were ever flagged).

## Real payoff, checked directly in the production pipeline (not just a proxy)

Wired the repaired NGC 4751 NFM cube into `04_template_calibration.ipynb`
(`ZAP_CLEANED_PATH` override, falls back to the raw download with a clear
message if the repaired cube doesn't exist) and re-ran it end-to-end,
same config as an earlier pre-ZAP run in this same session (identical
`FitConfig`, bins, cleaning settings -- only the input cube differs).
Real, apples-to-apples `chi2_red` comparison: nuclear bin (r=0.00")
1.36 -> 1.52 (noise-level, expected -- confirmed above that no spaxels
near this target's fitted center ever needed reverting, so ZAP+repair
leaves this bin's own data untouched); outer bin (r=3.13") **1309.71 ->
158.71, an 88% reduction**. This is the real acid test the earlier
sky-residual-window RMS checks were only a proxy for, and it landed
decisively positive -- the whole point of running ZAP in the first
place (04's own Section 7 markdown already documented this exact
"chi2_red climbs by orders of magnitude with radius" pattern as the
motivating problem).

Also re-ran `07_full_galaxy_analysis.ipynb` (same `CUBE_PATH` override,
all 113 real legacy dartboard bins, identical config to an earlier
pre-ZAP run this session) end-to-end, 0 errors. Real, apples-to-apples
cleaning-step comparison: overall clipped-pixel fraction **4.3% ->
0.5%** (8.6x reduction), median per-bin fraction 1.2% -> 0.5%, max
per-bin fraction 19.7% -> 4.0%. The pre-ZAP run had ~20-25 of 113 bins
trip `fit_losvd_clean`'s own ">10% clipped, consistent with a large
systematic mismatch" sanity warning; **zero bins trip it post-ZAP**
(confirmed by grepping the run log). Both notebooks now default to the
ZAP-cleaned cube (`ZAP_CLEANED_PATH`/`CUBE_PATH` overrides, falling back
to the raw download with a clear message if the repaired cube doesn't
exist).

## Status and what's not yet done

**Done and validated end-to-end, including in the real production
pipeline (not just synthetic proxies):** `build_source_mask` ->
`zap.process(mask=...)` -> `repair_corrupted_spaxels`, for both targets,
wired into `04_template_calibration.ipynb`/`07_full_galaxy_analysis.ipynb`
(NGC 4751) as the new default `CUBE_PATH`, with a real chi2_red/clip-
fraction win to show for it (see above).

Not yet done:
- Root-cause the 2 (NGC 5102) / 3 (NGC 4751) still-worse sky-residual
  windows further (segment-boundary hypothesis checked and ruled out
  above; real cause still open).
- Wire the repaired cube into the equivalent NGC 5102 notebooks (only
  NGC 4751's real analysis notebooks were updated this session, since
  that's the live science target -- NGC 5102's own real-data notebooks
  are lower-priority package examples/validation, not blocked on
  anything, just not done yet).
- `repair_corrupted_spaxels` has not been promoted out of
  `kinextract_test_suite` into `kinextract` itself yet.

See `[[project_ngc4751_ngc5102_zap_sky_correction]]` (Claude memory) for
the live status/next-steps pointer, and
`kinextract_test_suite/notebooks/kinextract_scratch_notebooks/
zap_cleaning.py` for the actual, documented, reusable implementation.
