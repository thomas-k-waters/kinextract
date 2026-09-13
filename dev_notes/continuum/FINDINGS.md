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

---

## 2026-08-01: the notebook 02 vertical-offset bug is real, root-caused, and NOT a simple implementation bug -- it's a genuine unidentifiability in the current (b, w, C) parametrization

`dev_notes/V1_HARDENING_ROADMAP.md` Phase 2a's working hypothesis (`ridge_c`
softly penalizes `C`'s own DC mode, `w`'s NNLS scale has zero penalty, so a
nonzero-mean true continuum gets partly absorbed into `w` instead of `C`)
was tested directly and needed real revision.

**Step 1 -- isolate the continuum block alone.** `fit_continuum_block`
called directly (known, fixed `M`, no `b`/`w` freedom at all -- the same
setup as the existing passing `test_fit_continuum_block_recovers_a_known_curve_in_isolation`),
comparing the existing test's full-period sine (`1+0.15*sin(2*pi*frac)`,
mean exactly 1.0) against a half-period sine (`1+0.15*sin(pi*frac)`, mean
~1.095 -- notebook 02's actual injected distortion): **both recover
accurately** (offset +0.0025 and +0.0059 respectively, RMS 0.003-0.006).
This refutes the "ridge_c biases the mean" half of the original
hypothesis in isolation -- `ridge_c`'s shrinkage-toward-flat exists but is
too weak to explain the notebook's own symptom by itself.

**Step 2 -- same comparison through the full 3-block alternation**
(`fit_losvd_with_continuum`, `test_3block_recovers_a_known_injected_continuum_distortion`'s
own setup, changing only `true_C`): full-period recovers essentially
perfectly (offset +0.0025, RMS 0.003, matching the existing passing
test); **half-period shows the bug directly**: recovered `C` mean lands
at 1.0017 (should be ~1.095), offset -0.0935, RMS 0.094 -- **30x worse**,
and `sherm`/`w_norm` (mixture fractions) both still recover correctly.
Confirms: the bug is real, specific to the 3-block alternation (not the
continuum block alone), and specific to a nonzero-mean true continuum
(the existing test suite's zero-mean sine/linear-trend cases never
exercise it).

**Step 3 -- find where the missing ~9.5% actually goes.** Not into `w`'s
raw NNLS scale as guessed (`w.sum()` is ~2.00 in *both* cases, unchanged).
It's `b`'s own amplitude: `fit_log_spline_map` never normalizes `b` to
unit area (nothing in the base 2-block path requires it either --
`compute_losvd_moments` is scale-invariant, so this is invisible there),
and `trapz(b, v_grid)` measured 0.498 in the full-period (working) case
vs 0.546 in the half-period (buggy) case -- a +9.7% inflation that,
multiplied through, exactly reproduces the data's true total flux
(`w_sum * C_mean * b_area` matches between both cases to 4 significant
figures). A uniform shift in the LOSVD's own spline coefficients (`b ->
b/area`) costs only this block's own `ridge_lam` penalty and is exactly
invisible to chi2 (block 2's very next NNLS re-solve for `w` absorbs any
rescaling of `b` perfectly, so it's free there too) -- a second, cheaper
escape valve than moving `C`'s own `ridge_c`-penalized coefficients away
from 0.

**Step 4 -- tried the direct fix, it didn't work, and that's the real
finding.** Pinned `b` to unit sum immediately after each iteration's
block-1 solve (`b -> b/b.sum()`, `coeffs -> coeffs - log(b.sum())` --
exact, not approximate, since `build_bspline_basis`'s rows are a verified
exact partition of unity). Confirmed the pin took effect (`b.sum()`
exactly 1.0 post-fix, `w.sum()` dropped from ~2.0 to ~0.136 to
compensate) -- but **the recovered `C` mean offset was completely
unchanged** (-0.0935, bit-for-bit similar to before). The escape valve
didn't close; it just moved. With `b`'s freedom removed, `w`'s
*already-zero-cost* NNLS scale absorbed 100% of what `b` used to absorb,
since nothing was ever regularizing `w`'s absolute magnitude in the first
place (the original hypothesis wasn't wrong, just incomplete -- both
`b`'s amplitude *and* `w`'s scale are equally free, and the fit uses
whichever is available).

**Why this isn't simply "add one more pin."** `w`'s absolute scale and
`C`'s mean are the *same* degree of freedom mathematically -- both
multiply the entire model uniformly across wavelength
(`C(lambda) * sum_k w_k * (T_k conv b)(lambda)`), and nothing in the
current design says which one *should* carry a given galaxy's overall
flux-normalization offset from the templates' own assumed continuum
level. Forcing `C`'s mean to always equal 1 (the plan's original 2b
proposal) doesn't fix this -- it hides it, by definition displaying a
mean-1 continuum regardless of the truth and silently pushing a
scientifically meaningful quantity (the galaxy's real broadband continuum
level relative to the normalized templates) into `w`'s absolute scale,
which is reported to users as normalized mixture *fractions*, not a flux
level. Constraining `w`'s own scale (e.g. a ridge penalty on its
magnitude, or fixing it from a principled reference point) is the
alternative, but iteration 0's own `w` solve happens against a
not-yet-correct `C` (starts flat), so naively anchoring to that reference
would just bake in the same bias it's supposed to prevent.

**Status: open, and bigger than originally scoped.** This is not a
one-line implementation bug -- it's a genuine parameter unidentifiability
in the current `(b_area, w_sum, C_mean)` parametrization (2 real degrees
of freedom competing for 3 nominal ones, resolved arbitrarily by whichever
regularization term happens to be weakest). Options for a future session,
not yet weighed against each other: (a) document as an honest, permanent
limitation -- report `C`'s *shape* as reliable, its absolute *mean* as
not identifiable without an external anchor, and teach users accordingly
in the revamped notebook 02; (b) add explicit regularization on `w`'s
overall scale (a real new hyperparameter, needs its own validation, and
changes what `w` means well beyond continuum-cofitting); (c) if templates
are known to be continuum-normalized to a specific convention, use that
as the external anchor instead of an internal one. No code was changed
this session as a result (the ineffective `b`-pin was implemented,
confirmed not to work, and reverted) -- `penalty_order=3`/fixed
`lam` in 4096-32768 (2026-07-30's finding, unaffected by this) remains
the right default regardless of how this identifiability question is
eventually resolved.

---

## 2026-08-11: `lam_selection="aic_local"` now supports continuum-cofitting

**Motivation.** Working through real N4751 full-galaxy outlier bins
(`dev_notes/real_data_sigma/FINDINGS.md`'s edge-pedestal entries), a
single population-wide fixed `lam` is a real compromise: strong enough
to suppress the worst bins' divergent edge pedestals, but too strong for
well-conditioned bins, which get over-smoothed (h3/h4 pulled toward or
past zero, sigma biased narrow). Adaptive per-bin `lam` selection is the
obvious fix for that tension, but was blocked entirely under
continuum-cofitting -- `fit_losvd` silently fell back to a fixed `lam`
with a `UserWarning` whenever `cfg.lam_selection="aic_local"` and
`cfg.continuum_knots` were both set, since `estimate_m_eff_local` (the
closed-form effective-df estimator the adaptive scan's own AIC criterion
needs) only ever linearized the LOSVD block, ignoring the continuum's own
contribution to the model and its own regularization.

**The fix.** `C(lambda) = exp(continuum_basis @ continuum_coeffs)` is the
same `exp`-link reparametrization as `b(v)` itself (this module's own
docstring), so it linearizes the same way. `estimate_m_eff_local` now
builds a *joint* Jacobian, stacking the LOSVD block (`diag(C) @ A_eff @
diag(b) @ basis`) and the continuum block (`diag(A_eff @ b) @
continuum_basis`, using the converged model = `C * (A_eff @ b)`), with a
block-diagonal penalty (`lam*D2'D2` for the LOSVD block, `lam_c*D2_c'D2_c
+ ridge_c*I` for the continuum block -- both terms, matching this
module's own "regularization has two independent terms" note above).
`chi2` is now computed against the continuum-multiplied model too,
matching what the fit itself actually minimizes (previously ignored `C`
entirely, though this branch was never reachable before since the
fallback always intercepted it first). The no-continuum path is
untouched -- same code, same numbers, confirmed by a regression test
(`test_m_eff_local_continuum_off_matches_pre_continuum_behavior`).
`select_lambda_aic_local`/`fit_losvd` both thread `wave_fit` through to
this new path; `bootstrap_losvd`'s own `reselect_lam` (previously forced
off under continuum for the same reason) now works there too.
`aic_regularization._edge_amplitude_frac` -- the specific diagnostic
that actually flags a divergent edge pedestal -- needed no changes at
all: it operates purely on `b`, never on the continuum.

**Validation: modest mock sweep, N4751's own real sigma regime
(400/550/700 km/s, S/N=60, `wiggle_scale=0.15`, 5 seeds each, true
sinusoidal continuum distortion), fixed `penalty_order=3`/`lam=4096.0`
vs. adaptive (mean absolute bias vs. the truth curve's own moments,
not the nominal generator parameters):**

| sigma | fixed \|bias_sigma\| | adaptive \|bias_sigma\| | fixed \|bias_h4\| | adaptive \|bias_h4\| | mean wall-time (fixed -> adaptive) |
|---|---|---|---|---|---|
| 400 | 13.44 km/s | 4.55 km/s | 0.0052 | 0.0100 | 6.4s -> 23.8s |
| 550 | 50.03 km/s | 12.77 km/s | 0.0141 | 0.0109 | 11.1s -> 53.1s |
| 700 | 153.45 km/s | 122.97 km/s | 0.0351 | 0.0341 | 21.6s -> 102.2s |

Adaptive meaningfully reduces mean `|bias_sigma|` at every sigma tested
(the exact regime N4751's own bins live in) -- 3-4x at 400/550, a more
modest ~20% at 700 (consistent with 2026-08-XX's earlier finding, see
`dev_notes/regularization/FINDINGS.md`, that fixed `lam=4096` is already
close to a local optimum specifically at sigma~700). `h4` bias is a wash,
not a clean win either way. Cost is real: consistently ~4-5x wall-clock
per fit (each scanned `lam` candidate now costs a full 3-block
alternation instead of 2-block).

**Not a clean win, though -- the documented "lam-selection lottery" (this
module's own module docstring, S/N 60-150) is real here too, and can make
a fit *worse*, not just fail to help**: one sigma=700 seed picked `lam=8`
(the adaptive scan's own noise-driven low-lam pick) and nearly doubled
`bias_sigma` (238 -> 440) with `bias_h4` growing from 0.049 to 0.144.
Fixed `lam=4096` never does this -- it's a known quantity, not a lottery.

**Recommendation, not yet implemented in this package**: don't blanket-
switch a whole population to adaptive-plus-continuum. Fit fixed first
(cheap, predictable), flag bins whose fixed-lam fit shows a real,
diagnosable problem (`edge_amplitude_frac(result["b"]) > `
some threshold, e.g. `FitConfig.max_edge_amplitude_frac`'s own 0.01), and
only pay adaptive's real cost -- and accept its real lottery risk -- on
that flagged subset. See `EHT_Project/notebooks/N4751_notebooks/
kinematics/n4751_full_galaxy_analysis.ipynb` for where this gets wired
into an actual per-target run.

## 2026-08-11 (later): the closed-form continuum m_eff's own designated ground-truth check was itself broken under continuum

**Trigger**: while doing a final "did today's new code introduce bias"
pass (user asked for the package to survive an expert statistician's
scrutiny before public release), tried to actually run the validation
`estimate_m_eff_local`'s own docstring calls for under continuum:
comparing it against `estimate_m_eff` (the bootstrap reference
implementation) at a shared `lam`. It could not run at all.

**Two bugs, both silent, both in code that predates today's continuum
extension** (so not a regression from today's work -- a pre-existing
gap this session's own more thorough validation habit surfaced):

1. `estimate_m_eff` had no `wave_fit` parameter -- its internal
   `fit_losvd` calls raised `ValueError` the instant `cfg.continuum_knots`
   was set, so this function (and `select_lambda_aic`, which calls it)
   simply could not run under continuum at all, ever.
2. `_model_prediction` (the model `estimate_m_eff`'s bootstrap resamples
   around, and what its own `chi2` is computed against) was `sum(w[k] *
   (A_stack_w[k] @ b))` -- the LOSVD-only model, never multiplied by the
   fitted continuum `C`. Even with (1) fixed, this would have made
   `estimate_m_eff`'s own `chi2`/bootstrap noise wrong under continuum,
   silently (no crash, just a wrong number) -- the two estimators would
   never have agreed even in a case where they should.

**Fix**: `_model_prediction` now multiplies by `result["C"]` when
present (`result.get("C") is not None` -- absent, not `None`, for a
non-continuum fit, matching `fit_losvd`'s own convention);
`estimate_m_eff`/`select_lambda_aic` both gained `wave_fit=None`,
threaded to every internal `fit_losvd` call. New regression tests
(`test_model_prediction_includes_continuum_factor`,
`test_m_eff_local_chi2_matches_bootstrap_chi2_at_the_same_lam_under_
continuum` -- the latter mirroring the no-continuum case's own existing
exact-agreement check) -- both would have failed outright against the
pre-fix code (the second couldn't even run). Full `test_aic_
regularization.py`: 22 passed.

**Real-world impact: none.** `estimate_m_eff`/`select_lambda_aic` are
standalone validation utilities, never reachable through
`FitConfig.lam_selection` (`"aic_local"`/`"fixed"` only) -- no real fit,
in any notebook or test, was ever affected. The actual, concerning gap
this closed was epistemic, not operational: the closed-form continuum
extension (this file's own 2026-08-11 entry above) shipped without ever
having been checked against the one reference its own docstring says is
"its whole reason to exist" for the no-continuum case. Now that the
check actually runs, it passes exactly, at the level of "the same
deterministic problem's chi2 must agree bit-for-bit" -- not just "close
enough."
