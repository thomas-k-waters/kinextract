# Porting OG kinextract's safeguards into kinextract_lite -- status

## Why

`dev_notes/real_data_sigma/FINDINGS.md` root-caused the real-data sigma
failure (nucleus 81 km/s vs the legacy pipeline's ~54): the recovered
LOSVD grows a divergent pedestal at the velocity-grid edges, which the
moment estimator integrates. Inspecting the original `kinextract` showed
it already carries four safeguards `kinextract_lite` dropped when it
reinvented the regularization:

| defect measured in lite | OG kinextract's safeguard |
|---|---|
| divergent wing pedestal | wing-tapered roughness penalty, always on, onset in units of the *recovered* width (`sigl0`), self-consistent by fixed-point iteration |
| multi-modal/U-shaped LOSVDs accepted | `xlam_max_peaks=1` -- reject lam candidates whose LOSVD is not unimodal |
| AIC collapses to the least-regularized lam when residuals are systematics-dominated | `xlam_criterion='discrepancy'` (pPXF-style chi2-rise target), `nsigma=0.3` empirically recalibrated for a LOSVD histogram |
| 19 spline coefficients on a 149-point grid | `n_losvd_bins=29`, a coarse histogram |

## Done

* **`losvd.tapered_lam_vector`** -- OG's law ported exactly
  (`lam_j = lam` inside `sfac*sigl0`, `lam * (d/sigl0/sfac)**power`
  beyond; `sfac=1.8`, `power=4`). Verified: at `lam=100, sigl0=50` the
  per-row strength runs 100 in the core to 15137 at the grid edge (151x).
* **`losvd.greville_abscissae` / `penalty_row_velocities`** -- each
  penalty row now knows where it lives on the velocity axis, which is
  what makes a per-row strength meaningful for a *spline* penalty (OG's
  is per-histogram-bin, so this part had to be derived, not copied).
* **`losvd.count_losvd_peaks`** -- OG's prominence-based peak counter
  ported (verified: 1 for a Gaussian, 2 for a bimodal LOSVD).
* **`fit_log_spline_map`** now takes `lam` as scalar *or* per-row vector;
  the two-variant (`no_wing`/`with_wing`) objective is gone, replaced by
  one objective.
* **`fit_losvd`** runs the `sigl0` fixed point (fit -> measure width ->
  rebuild taper -> refit), exposed as `taper_sfac` / `taper_power` /
  `n_sigl0_iter` on `FitConfig`, and returns `sigl0`/`v_center`.
* **`wing_lam` / `wing_onset_frac` / `build_wing_taper` deleted** --
  superseded by the taper, which does the same job scale-free.

## Measured effect, and the blocker

Real NGC 5102, taper ON vs OFF at fixed lam:

| bin | lam | taper OFF | taper ON | legacy |
|---|---|---|---|---|
| nucleus | 512 | 68.2 | **49.9** | ~54.3 |
| nucleus | 64  | 75.4 | 74.6 | ~54.3 |
| nucleus | 1   | 79.7 | 80.6 | ~54.3 |
| mid-radius | 64 | 40.9 | 42.3 | ~40-45 |
| mid-radius | 512 | 41.8 | 43.5 | ~40-45 |

**The taper works, and does not help at the lam actually in use.** It
*multiplies* the base `lam`; `aic_local` selects `lam=1` on this data, so
the wing strength tops out around 58 -- far too weak. OG's own base
`xlam` grid spans 100 to 1e7, where the same taper is enormous.

**The two safeguards are coupled and only work together.** Porting the
taper without porting the selection rule leaves the fix inert. This is
the remaining work, and it is the last piece:

1. Replace `aic_local` with OG's **discrepancy criterion** (largest lam
   whose chi2 stays within `nsigma*sqrt(2*ngood)` of the minimum), on a
   grid spanning ~1e2-1e7 to match the taper's scale.
2. Add **peak rejection** (`count_losvd_peaks(b) > 1` disqualifies a lam
   candidate) -- already implemented, not yet wired into selection.
3. Then re-validate on both the real bins and the mock grid, and delete
   `aic_regularization.py` if the discrepancy rule supersedes it.

## Final state of this session

* `lam_wing` **defaults to 0.0 (off)**. The taper is ported, unit-tested
  and mechanically verified, but with it engaged -- and its onset seeded
  from the pedestal-immune core width -- the real NGC 5102 nucleus still
  returned 79 km/s against ~54. The wing ramp demonstrably fires (checked
  per-row), so in that configuration the excess width is not coming from
  where the ramp reaches. Shipping it on by default would mean shipping
  an unvalidated change; off keeps the default path exactly what the
  480-case grid covers.
* Deleted outright: `spectral_prior.py`, `tests/test_spectral_prior.py`,
  `examples/notebooks/07_spectral_prior_exploration.ipynb` (superseded by
  Stage 22), `wing_lam`/`wing_onset_frac`/`build_wing_taper`, the
  dual-objective `no_wing`/`with_wing` code path, and the obsolete tests
  that covered them.
* Added `tests/test_wing_taper.py` (8 tests) covering the ported
  machinery, with an explicit scope note that it tests mechanism, not
  the real-data fix.

## What is still open

**Update (2026-07-30): item 1 below is now CLOSED**, and item 2's
peak-rejection half turned out to be the wrong tool, not just
not-yet-wired-in -- see `real_data_sigma/FINDINGS.md`'s full resolution
and `config.py`'s `max_edge_amplitude_frac` docstring, which states the
mechanism directly: the divergent edge pedestal that inflated real-data
sigma ~2x is **smooth and single-peaked**, so a unimodality/peak-count
rule cannot distinguish it from a good fit no matter how it's tuned --
it needed a different kind of check (an edge-amplitude validity
condition on the fitted LOSVD itself), not the taper+peak-rejection
combination this section originally proposed. `count_losvd_peaks` was
removed from the package during 2026-07-30 spring cleaning as a result
(its own unimodality assumption doesn't hold for this failure mode); the
taper machinery (`lam_wing` et al.) remains ported, tested, and
available, just still off by default (`lam_wing=0.0`) since it wasn't
the fix that actually closed this.

1. ~~The real-data sigma inflation is NOT fixed.~~ **Fixed** via
   `max_edge_amplitude_frac`, `VALIDATION_CAMPAIGN.md` Stage 23.
2. ~~The remaining OG safeguards are still unported: the discrepancy lam
   criterion... and peak rejection during selection...~~ Peak rejection
   is superseded (see update above, not a gap to fill). The discrepancy
   lam criterion (an alternative to `aic_local` for detecting
   systematics-dominated residuals) remains genuinely unported, but
   `aic_local`'s own failure mode on real data is now covered by
   `max_edge_amplitude_frac` from a different angle, so this is lower
   priority than it reads here.
3. `n_losvd_bins`-equivalent (OG's coarse 29-bin histogram vs lite's
   19-coefficient spline on a 149-point grid) is unexamined.
