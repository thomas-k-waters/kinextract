# Legacy dartboard binning: fidelity check, and two modernization features

**Date: 2026-08-06**

## The question

`kinextract.binning.assign_legacy_dartboard_bins` has been used throughout
this project's own real-data notebooks (`07_full_galaxy_analysis.ipynb`,
the GreatLakes `kinematicextraction` script) as "a faithful port of the SCO
Fortran framework's own `plotbin3.f` binning" -- but that claim had never
been checked line-by-line against the actual Fortran source, only against
the port's own docstring description of itself. Separately, two real
modernization gaps were requested: (1) binning should work directly off a
`MuseCube`, not the legacy `bin/binspectra` script's own per-spaxel `.fits`
file convention; (2) folding a bin's north/south spaxels together across
the kinematic major axis (the legacy scheme's own default, and until now
this port's *only* behavior) should be optional, not forced -- real galaxy
velocity fields are not always point/axis-symmetric, and the folded scheme
discards that information by construction.

## Fidelity check: line-by-line against `plotbin3.f`

Read `sco_framework/modprogs/plotbin3.f`'s own main loop and `writebin`
subroutine directly (not `bin/binspectra`'s own wrapper, which only drives
`plotbin3` -- see below for the *combination* step, a separate program).
Every step of `assign_legacy_dartboard_bins`'s own docstring checked
against the actual Fortran:

| Step | `plotbin3.f` | `assign_legacy_dartboard_bins` | Match |
| --- | --- | --- | --- |
| Rotate by PA | `x=xp*ca-yp*sa; y=xp*sa+yp*ca` | `x = dx*cos(pa) - dy*sin(pa)`, same for `y` | Yes |
| Side split | `if(x.lt.0) x=-x, ivel=0 (bnn)` | `side = where(x>=0, 1, -1)` | Yes |
| Major-axis fold | `y=abs(y)` | `y_fold = abs(y)` | Yes (now optional -- see below) |
| Radius lookup | `do j=2,irmax: if r in [r(j-1),r(j)): ir=j`, `if r<r(1): ir=1` | `searchsorted(r_edges, r_clipped)` | Equivalent (verified the clip-then-searchsorted vectorization reproduces the Fortran's explicit-loop-plus-shortcut exactly, including the `r<r(1)` absorbed-into-innermost-ring case) |
| Angle lookup | Same `do j=2,ivmax` pattern on `vp` | Same `searchsorted` pattern | Yes |
| Bin name | `"bin"`/`"bnn"` + 2-digit `ir` + 2-digit `iv`, zero-padded | `f"{prefix}{irc:02d}{irv:02d}"` | Yes, byte-identical |

**Verdict: confirmed faithful.** No discrepancy found in the core geometry.
`load_legacy_bin_edges`'s own `r_edges`/`v_edges_deg` construction
(`[rlow[0], rup...]` / `[0, vup...]`) was also checked against how
`plotbin3.f` reads `bin_r.out`/`bin_v.out` and uses its own `r(1)`/`v(1)`
as an implicit lower-bound shortcut -- confirmed equivalent, not just
superficially similar.

## Also read: `rcomb`/`imcmb.f` -- the actual per-bin spectrum *combination* step

`binspectra` (the legacy driver script) calls `plotbin3` for the geometric
assignment above, then `rcomb` (a thin wrapper) for each populated bin,
which calls the real combiner: `imcmb.f`, a per-pixel (per-wavelength)
biweight-mean-plus-two-error-terms combine across every spaxel `.fits`
file listed for that bin. This is the actual precedent for "how should the
modernized binning propagate errors when combining spaxels" -- not
something the geometric port (`assign_legacy_dartboard_bins`) touches at
all, since it only assigns spaxels to bins, it never combines their
spectra.

`imcmb.f`'s own design, per pixel:

1. **Biweight mean** (not a plain mean) of the per-spaxel flux values --
   robust to a single bad/cosmic-ray spaxel.
2. **Formal error**: standard inverse-variance-weighted combine,
   `1/sum(1/var_i)` -- correct under independent per-spaxel noise, which
   real MUSE data only approximately satisfies (adjacent spaxels are
   spatially correlated from pipeline resampling/drizzling).
3. **Empirical error**: the biweight *scale* of the actual per-spaxel flux
   values at that pixel -- a direct measurement of how much the spaxels
   disagree, independent of what their individually-reported variances
   claim. If spaxels are positively correlated, this term is systematically
   larger than the formal term alone would predict.
4. **Total error**: `sqrt(formal**2 + empirical**2)` -- effectively
   dominated by whichever term is larger, so this self-corrects for an
   under-estimated formal term without needing the true spatial
   correlation length measured in advance.

This is a genuinely well-designed, if old, error budget -- worth
preserving, not just modernizing past.

## What was built

### 1. `fold_major_axis` parameter on `assign_legacy_dartboard_bins`

Defaults to `True` (the legacy behavior, exactly) so every existing caller
(`07_full_galaxy_analysis.ipynb`, `kinematicextraction`, ...) keeps its
current behavior unchanged -- this is additive capability, not a behavior
change with a silent default flip. `fold_major_axis=False` keeps a bin's
north (`y>=0`) and south (`y<0`) halves as separate bins instead of merging
them (each still individually folded to `|y|` for the radius/angle
*lookup*, matching the legacy scheme's own angular-edge convention which
only covers `[0, 90)` -- only the post-lookup *merge* becomes optional).
New `hemisphere` field (`+1`/`-1`) on every `bin_table` row. Bin naming in
unfolded mode: the north half keeps the exact legacy `bin####`/`bnn####`
stem; the south half appends `s` (`bin####s`/`bnn####s`) -- **not**
legacy-pipeline-compatible (`fitlovw`/`pallmc`/... expect exactly the
7-character stem), so this mode is for kinextract-native workflows only.

Verified: total spaxel count and valid-footprint (`bin_id_map >= 0`
pattern) are identical folded vs. unfolded; every folded bin's own spaxel
count exactly equals its unfolded north+south split's sum. 4 new tests in
`tests/test_binning.py`; all 22 pre-existing tests unaffected (default
behavior unchanged).

### 2. `combine_bin_spectrum_robust` -- a modern, `MuseCube`-native port of `imcmb.f`'s own error budget

`kinextract.muse_cube.combine_bin_spectrum_robust(cube, spaxel_mask,
variance_correction_factor=1.0, min_spaxels_for_empirical=3)`. Same
formal-plus-empirical-in-quadrature design as `imcmb.f` (biweight
location/scale via `astropy.stats`, not a from-scratch reimplementation of
`biwgt`), operating directly on `cube.flux[:, spaxel_mask]`/
`cube.error[:, spaxel_mask]` -- no per-spaxel `.fits` files, no `rcomb`
wrapper, no `imcmb.def`/temp-file dance. Adds one thing `imcmb.f` didn't
have: an explicit, opt-in `variance_correction_factor` for the Bacon et al.
(2017) MUSE `STAT`-extension variance-underestimate correction (`(1/0.6)**2
~= 2.78`, their own recommended value -- already *demonstrated* in this
project's `05_noise_correlation.ipynb` but never wired into a reusable
function before now). Defaults to `1.0` (no correction) since the factor
is pipeline/instrument-version specific and should be measured for a given
cube, not silently assumed.

**Validated directly against a synthetic cube with real spatially-correlated
noise** (a shared common-mode component injected across every spaxel in an
aperture, on top of independent per-spaxel noise each spaxel's own
`error` array has no knowledge of -- the same structural blind spot real
MUSE STAT-extension errors have relative to true spatial correlation):

| | reported error | vs. true empirical scatter of the combined mean |
| --- | --- | --- |
| naive (`extract_aperture_spectrum`, independent-noise formula) | 0.333 | 4.5x too small |
| `combine_bin_spectrum_robust` (formal + empirical in quadrature) | 1.211 | within ~19% |

The naive formula reproduces the *independent*-noise prediction exactly
(0.333, matching `sigma/sqrt(n_spax)`) -- structurally blind to the
correlation by construction. The robust combiner's empirical term catches
most of the gap without needing the true correlation length measured in
advance. 6 new tests in `tests/test_muse_cube.py` (correlated-noise error
inflation, biweight outlier resistance, mean-not-sum flux convention,
`variance_correction_factor` scaling, all-bad-spaxel `NaN` handling).

## A confirmed, honest limitation: the empirical term can't separate correlated noise from a real gradient

2026-08-06, later the same day. After wiring `combine_bin_spectrum_
robust` into `04_template_calibration.ipynb` (both the N4751 project copy
and kinextract's own package example) and re-running, the *synthetic*
mock case (`build_mock_cube`, no noise correlation injected at all)
showed reported S/N dropping far more than expected -- e.g. one 8-spaxel
radial-annulus bin went from S/N=48.8 (`extract_aperture_spectrum`) to
S/N=5.3 (`combine_bin_spectrum_robust`), an ~10x drop with zero real
correlation present to explain it.

Checked directly (not assumed): the 8 spaxels' own true flux values at a
representative wavelength range 26.9-43.5 (std 6.57), against a per-
spaxel noise level of 2.0 -- i.e. the spaxels genuinely disagree by ~3x
more than their own noise level, because they sit at different points
within a real Sersic-profile gradient the coarse radial-only binning
scheme (`04_template_calibration.ipynb`'s own annuli, one bin per whole
ring regardless of azimuth) spans. The empirical (biweight-scale) term
has no way to distinguish "these spaxels disagree because of correlated
noise" from "these spaxels disagree because they have genuinely different
true flux" -- both look identical to it.

**Not a bug, but a real, now-documented limitation** (`PARAMETER_GUIDE.
md`'s own new section): a coarse/spatially-extended bin will report a
lower S/N under this combiner than the naive one regardless of whether
real noise correlation is present, simply because more of the bin's own
real astrophysical variation gets folded into the same "how much do
these spaxels disagree" measurement noise correlation was meant to catch.
Finer/more homogeneous bins (smaller radial annuli, or the fully
angularly-subdivided kinematic bins `07_full_galaxy_analysis.ipynb` uses,
rather than `04`'s own radial-only annuli) should be far less affected,
since less real gradient is spanned per bin -- not yet checked directly
at that finer scale.

## Deployment (2026-08-06)

`combine_bin_spectrum_robust` is now wired into every real-data notebook
in both this project and kinextract's own examples, and each has been
re-run end to end on real data with it:

- N4751 project: `n4751_template_calibration.ipynb` (real cube, verified:
  S/N at the largest bin dropped 1177.8 -> 2.9, confirming the fix is
  actually exercised) and `n4751_full_galaxy_analysis.ipynb` (real cube,
  116 bins, ~104 min fit -- see "A confirmed, honest limitation" above
  for why this took ~1.8x longer than the pre-robust-combiner baseline;
  two unrelated pre-existing notebook bugs, `plot_legacy_bin_map`'s
  `central_bin_id` kwarg and a `WAVEFITMIN`/`WAVEFITMAX` scoping issue on
  the cache-hit path, were found and fixed along the way, not introduced
  by this change).
- kinextract's own `04_template_calibration.ipynb` (both synthetic mode
  and real `DATA_MODE="download"`, the latter reusing the already-
  downloaded local NGC 5102 cube rather than re-triggering a multi-GB
  ESO archive pull -- this exposed a real gap in `eso_archive.download_
  dataset` itself, since fixed: it had no local-file check at all, so
  *any* caller re-running that cell would re-download unconditionally.
  Now checks `{dest_dir}/{dp_id}*.fits` first and reuses it if present;
  `force=True` bypasses this. 2 new tests in `test_eso_archive.py`)
  and `07_full_galaxy_analysis.ipynb`
  (`FORCE_RECOMPUTE=True`, 219 bins, ~91 min fit+Laplace-uncertainty,
  shipped `ngc5102_full_galaxy_results.npz` refreshed to reflect the new
  extraction -- 8/219 bins' Laplace uncertainty came back `NaN`, the
  documented zero-valid-posterior-predictive-sample fallback, not a new
  failure mode).

Both real-data capstone notebooks (`n4751_full_galaxy_analysis.ipynb`,
kinextract's `07_full_galaxy_analysis.ipynb`) needed the same fix once
deployed at real scale: their per-bin fit (and, for `07`, Laplace
uncertainty) calls were restructured from one unbatched parallel call
over every bin to batches of 20, each with a progress line printed *and*
appended to a flushed log file -- `jupyter nbconvert --execute` does not
stream a cell's own `print()` output live, so an unbatched version of
this exact cell blew through a 90-minute timeout with zero visibility
into how far it had gotten before this was added.

**Not yet wired in**: the GreatLakes `kinematicextraction` script (`sco_
framework_updated/bin/`) still uses `extract_aperture_spectrum`'s own
naive independent-noise combine -- out of scope for this deployment pass,
since that script runs on a SLURM cluster, not interactively, and wasn't
part of what was re-run this session.

## A real fidelity gap the line-by-line check missed: the exact center spaxel (2026-08-07)

The 2026-08-06 fidelity check above verified `assign_legacy_dartboard_bins`
against `plotbin3.f`'s general (non-degenerate) mechanics, but missed a
real edge case: `plotbin3.f`'s own angle branch is literally `if(abs(x)
.ne.0) vp=atan(y/x) else vp=89.99`, evaluated *after* `x`/`y` are already
folded non-negative. For any `x>0` this is identical to `atan2(y, x)` --
and for `x=0, y>0` (a point exactly on the minor axis, not the center)
`np.arctan2` already returns `pi/2` exactly, matching the Fortran's own
`89.99` fallback with no discrepancy either. **Only the fully-degenerate
origin itself (`x=0` *and* `y=0` simultaneously -- the literal single-
spaxel center bin) diverges**: `np.arctan2(0, 0)` silently returns `0.0`,
not the Fortran's forced `89.99`. That 0 vs. 89.99 difference lands the
center spaxel in the *first* angular sector (`irv=1`) here, vs. the
*last* one (`irv=n_v`) in the legacy pipeline.

**Confirmed directly, both directions, not assumed**: read `plotbin3.f`'s
actual branch (not just its docstring paraphrase) to confirm the Fortran
side; checked `assign_legacy_dartboard_bins`'s actual `np.arctan2` call
against it to confirm the Python side. On a real N4751 run (5-sector
`bin_v.out`), this function reported the center bin as `"bin0101"`; the
legacy pipeline's own convention is `"bin0105"` for the identical spaxel
-- a real naming incompatibility with the downstream Fortran modeling
chain (`fitlovw`/`pallmc`/... read spectrum files by this exact 7-
character name), not just an internal labeling choice. User-reported,
independently re-derived from the Fortran source before trusting it.

**Fixed**: `vp` is now computed as `np.where(x_side != 0, atan2(...),
89.99)`, replicating the Fortran's own branch exactly rather than relying
on `arctan2`'s own (different) origin convention. New regression test,
`test_assign_legacy_dartboard_bins_exact_center_lands_in_last_irv_not_
first` -- the existing `..._center_clips_into_innermost_ring` test only
ever asserted `irc==1`, never checking `irv`, which is exactly why this
slipped through originally.

**Blast radius**: only the single-spaxel innermost bin's own *name*
changes (kinematic content for that bin is computed identically either
way) -- but every real N4751 notebook run already completed this session
before this fix landed used the old, wrong `"bin0101"` name for that bin.
Not yet re-run to pick up the corrected name as of this note (see
session's own todo list) -- re-running the full ~100-minute fit again
purely for a single bin's filename is wasteful; a targeted rename of
that one bin's already-written output files is the planned fix instead.

## Not done / open
- The empirical-term validation above used one synthetic correlation
  strength/pattern (a single shared common-mode component); a real MUSE
  cube's own spatial correlation structure (length scale, whether it's
  uniform across the field) has not been measured and compared against
  what this function's own empirical term actually recovers on real data.
- `fold_major_axis=False`'s own real-data usefulness (e.g. does it surface
  real north/south asymmetry on NGC 4751, matching the point-symmetry
  failure `10_kinematic_center_pa_development.ipynb` already found from a
  completely different angle) has not yet been exercised on real data,
  only validated geometrically on a synthetic grid.
