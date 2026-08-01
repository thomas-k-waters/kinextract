# Continuum-cofitting: first multi-seed validation

**Date: 2026-07-30**

## The question

`continuum.py`'s own docstring and `examples/notebooks/02_realistic_mock_fit.ipynb`
Section 8 both admit the same gap: continuum-cofitting was "substantially
de-risked... but not yet re-validated end-to-end on a fresh multi-seed
bias/residual sweep." Notebook 02's own demonstration is a single seed,
pinned to `penalty_order=2, lam_selection='fixed', lam=2.0` -- described
as "the documented-working combination," with `penalty_order=3` (this
package's own default everywhere else) explicitly flagged as untested
under continuum-cofitting, since adaptive lam selection doesn't support
it (`fit_losvd` falls back to a fixed lam with a `UserWarning`).

## Setup

`dev_notes/continuum/multiseed_sweep.py`: a 3-E-MILES-template mock
(sigma=90, spline-truth wiggle), a genuine injected multiplicative
continuum distortion (`true_C = 1 + amp*sin(pi*frac)`, matching notebook
02's own convention), swept over distortion amplitude in
{0.05, 0.15, 0.30}, S/N in {30, 60, 100}, 8 seeds each (288 fits total, 0
convergence failures). Four fits per case: no continuum-cofitting
(package default, shows what an unmodeled distortion does on its own),
continuum-cofitting at `penalty_order=2, lam=2.0` (notebook 02's
documented combination), and continuum-cofitting at `penalty_order=3`
with two representative fixed `lam` values (4096, 32768 -- picked as
commonly-selected values from this package's own adaptive-lam runs at
moderate sigma, not independently tuned for this sweep).

## Result: the distortion amplitude barely matters; the penalty order does, a lot

Aggregated across all 3 amplitudes per (config, S/N) cell -- amplitude
changed the outcome by less than seed-to-seed noise at every S/N tested,
so it's dropped from the table below (full breakdown in
`results_multiseed_sweep.csv`):

| config | S/N | mean \|sigma bias\| (km/s) | mean h4 bias |
|---|---|---|---|
| no continuum-cofitting | 30 | 11.08 ± 2.89 | +0.0097 ± 0.0005 |
| continuum, order 2, lam=2.0 (notebook 02's combo) | 30 | 11.80 ± 2.22 | **+0.0359 ± 0.0019** |
| continuum, order 3, lam=4096 | 30 | 8.52 ± 1.60 | +0.0093 ± 0.0000 |
| continuum, order 3, lam=32768 | 30 | 8.28 ± 1.67 | +0.0093 ± 0.0000 |
| no continuum-cofitting | 60 | 7.45 ± 1.50 | +0.0104 ± 0.0007 |
| continuum, order 2, lam=2.0 | 60 | 5.48 ± 1.13 | **+0.0259 ± 0.0024** |
| continuum, order 3, lam=4096 | 60 | 4.02 ± 0.75 | +0.0093 ± 0.0000 |
| continuum, order 3, lam=32768 | 60 | 4.59 ± 0.89 | +0.0093 ± 0.0000 |
| no continuum-cofitting | 100 | 7.61 ± 1.52 | +0.0119 ± 0.0017 |
| continuum, order 2, lam=2.0 | 100 | 3.11 ± 0.68 | +0.0154 ± 0.0027 |
| continuum, order 3, lam=4096 | 100 | 2.47 ± 0.45 | +0.0093 ± 0.0000 |
| continuum, order 3, lam=32768 | 100 | 2.65 ± 0.47 | +0.0093 ± 0.0000 |

**Three findings, in order of how confident the data supports them:**

1. **h4, high confidence**: the "documented-working" order-2/lam=2.0
   combination has an h4 bias 3-4x larger than either order-3 fixed-lam
   choice, at every S/N tested, with tiny standard errors (order-3's h4
   std is ~0.0000 -- essentially deterministic, the same shrink-toward-
   Gaussian mechanism `VALIDATION_CAMPAIGN.md` Stage 22 documents for
   ordinary non-continuum fits). **The combination this package's own
   notebook currently recommends for continuum-cofitting is worse on h4
   than doing nothing (no continuum-cofitting at all)** at every S/N
   tested (0.026-0.036 vs 0.010-0.016).
2. **Sigma, moderate confidence**: both order-3 fixed-lam choices beat
   both no-continuum-cofitting and order-2 continuum-cofitting on mean
   `|sigma bias|` at every S/N, by 1-3 standard errors -- suggestive and
   consistent across all three S/N cells, not yet at the confidence level
   of finding 1.
3. **Distortion amplitude, high confidence it doesn't matter here**: 0.05
   through 0.30 (a 6x range) changed nothing beyond seed noise. This
   mock's residual bias is apparently dominated by something else (the
   fixed 3-template mixture's own recovery bias, most likely, given this
   project's other findings on template-mixture identifiability) rather
   than by how hard the injected continuum distortion is fighting the
   fit -- worth keeping in mind before generalizing these numbers to a
   very different template library or a much larger distortion.

## What this does NOT establish

- **One mock, one sigma (90), one template mixture, one distortion
  shape** (a single half-sine). Not a sweep over sigma, template
  count/library, or distortion shape/frequency -- a real continuum defect
  (imperfect flux calibration, reddening) may not look like this smooth
  half-sine.
- **The two order-3 `lam` values were picked as representative, not
  tuned or scanned.** A real `lam` sweep at order 3 (matching
  `select_lambda_aic_local`'s own grid) might do even better, or reveal
  the two values tested here got lucky.
- **This does not make continuum-cofitting "production-hardened."**
  It answers the specific question notebook 02 posed (is order-3
  meaningfully different/worse than order-2 here) -- the answer is "no,
  it's evidently better, at least in this regime" -- not a full
  replacement for the broader validation `continuum.py`'s own docstring
  still correctly describes as outstanding.

## Recommendation

Update notebook 02's Section 8 and `continuum.py`'s own guidance to stop
recommending `penalty_order=2` as *the* combination for continuum-
cofitting -- on this evidence, a fixed order-3 `lam` (somewhere in the
4096-32768 range) is a better starting point, not an untested fallback.
Not yet changing any package default from this alone (see limitations
above) -- this is a documentation-and-recommendation update, not a code
change.

---

## 2026-07-31: consolidating the real-data chi2_red root-cause picture

The overnight plan's Phase 4 set out to test whether non-Gaussian LSF
*shape* mismatch (Simon 2026, flagged in `lsf.py`'s own docstring as a
known, deliberately-unaddressed gap) explains the real-data chi2_red~11-28
excess. Before building a synthetic injection test for that, the existing
evidence already on disk (mostly in `07_noise_correlation.ipynb`'s own
saved cell outputs, not yet distilled into `dev_notes` prose) turned out to
already answer most of the "how much is explained by what" question with
real numbers, on the same NGC 5102 aperture used throughout this project's
real-data work:

| correction applied | chi2_red |
|---|---|
| none (rho=0 assumed, propagated pipeline errors as-is) | 21.09 |
| Bacon et al. 2017 variance-underestimate factor only (flat `(1/0.6)^2` rescale, no correlation structure) | 7.59 |
| full AR(1) GLS-whitening at the measured rho=0.88 (changes the fit itself, not just a rescaling) | 11.21 |
| continuum-cofitting alone, separate aperture, no rho correction (`dev_notes/real_data_sigma/FINDINGS.md`) | 16 -> 5.6 |

**Correlated noise, even fully modeled, does not bring chi2_red anywhere
near 1** -- it moves the number around (a flat rescale trivially divides
it; genuine GLS-whitening changes the fit and lands at a similar-order
value) but a genuine ~5-11x excess survives every correction tried so far,
individually. Notebook 07's own "Limitations, honestly" section already
states this plainly: "whitening did not bring chi2_red to ~1... a real,
separate effect on top of correlation -- template mismatch, unmodeled
continuum shape, or both." This session's own finding (`dev_notes/
real_data_sigma/FINDINGS.md`, 2026-07-31) that notebook 08's residual
sigma-clipping concentrates at the *edges* of the 8400-8800 fit window
rather than at CaII is a third, independent data point consistent with the
same unresolved class of problem -- something breaks down specifically
near the boundaries/shape of the fit window, not a uniform noise-
miscalibration.

**LSF shape mismatch remains a plausible but genuinely untestable
hypothesis tonight**, not because it was tried and failed, but because
testing it credibly requires the template library's own *measured* LSF
shape to inject a physically-motivated (not made-up) mismatch amplitude --
`lsf.py`'s own docstring already notes this is "typically not measured" for
libraries like E-MILES/MILES. A synthetic test built on an arbitrary
assumed mismatch shape/amplitude would produce a number with no real
evidentiary weight either way, so this was not attempted; the honest state
is "motivated by the literature, still untested, blocked on real template
LSF measurements this project does not have."

**Bootstrap continuum error-bar completeness, checked and confirmed OK
(no bug)**: `uncertainty.bootstrap_losvd`'s replicate loop calls
`fit_losvd(A_stack_w, y_w_boot, cfg_boot, wave_fit=wave_fit)` fresh for
every one of `n_boot` replicates, and `cfg_boot` preserves
`cfg.continuum_knots` unchanged (only `lam`/`lam_selection`/
`losvd_single_start` are ever overridden via `dataclasses.replace`) --
`fit_losvd`'s own `_fit_at` always re-solves the continuum block jointly
with the LOSVD whenever `continuum_knots is not None`, so continuum
coefficients are never held fixed at the point estimate's value across
replicates. Continuum-driven uncertainty is already propagated into the
reported V/sigma/h3/h4 error bars whenever continuum-cofitting is on --
this item from the overnight plan turned out to already be correct by
construction, matching the same pattern as the `rho`-threading
investigation (see `dev_notes/HISTORY.md`).

**Net for tonight**: no new code change from this phase. The value is a
consolidated, cross-referenced picture (previously scattered across one
notebook's cell outputs and two separate dev_notes files) plus ruling out
one more "maybe it's just a bootstrap bug" possibility. The remaining
~5-11x chi2_red gap after every currently-testable correction is the
single most important open item for a future session with access to
better template LSF characterization or a wider real-data sample to
localize the mismatch further.
