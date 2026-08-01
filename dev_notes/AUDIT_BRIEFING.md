# kinextract_lite — audit briefing

Read this file first. It exists so you don't have to rediscover, by exploration, things
that are already known and empirically settled — every claim below is backed by a real
test run in this repo, not assumed. Re-deriving these from scratch would waste your own
budget; the actual open questions are listed separately below, and that's where your time
is most useful.

> **2026-07-29 update: this file predates `VALIDATION_CAMPAIGN.md` Stages 22-27 and is
> stale in specific, important ways** -- most notably, the "known open issue" below about
> the `m_eff`-axis calibration table being unvalidated is now **outdated**: that table was
> rebuilt against the current `penalty_order=3` default's own lam grid (Stage 26,
> 2026-07-28/29) and its filename changed
> (`regularization_bias_emiles_multitemplate_meff.txt`, not the `_muse_` name below). This
> file also predates the `penalty_order=3`/JAX-float64 default flip itself (Stage 22), the
> spline-truth mock default, and the reflected dartboard binning work. Read
> `dev_notes/FULL_REPO_AUDIT_2026-07-29.md` and `VALIDATION_CAMPAIGN.md` Stages 22-27 for
> current state before trusting specific claims below; the architecture paragraph and the
> "already tried, measured, resolved" list are still accurate.

## What this package is

A from-scratch, non-parametric line-of-sight velocity distribution (LOSVD) extractor for
galaxy spectroscopy, used for black-hole-mass work via stellar dynamics. Full mathematical
model: `dev_notes/model_reference/model_reference.html` (open directly in a
browser — self-contained, no server needed). Full narrative of the methodology and current
status: `dev_notes/model_reference/current_state.html`. Read both before touching
code; they are more current and more complete than any docstring-level summary could be.

Core architecture in one paragraph: the LOSVD `b(v)` and a shared multiplicative continuum
`C(lambda)` are both parameterized as `exp(B-spline)` (structurally positive, no boundary at
zero). Template mixture weights `w` are non-negative (NNLS). The three are recovered by
alternating minimization (`multi_template._alternate_once`): fix two, solve the third, repeat.
Noise covariance is whitened once at construction time (`noise_model.py`) so the fitting
engines themselves only ever see an ordinary sum-of-squares — correct regardless of whether
the covariance is diagonal or AR(1)-correlated.

## Already tried, measured, and resolved — do not re-suggest these

- **Switching the forward model to log-wavelength for speed.** Directly profiled:
  `build_template_stack` (the part log-lambda + FFT convolution would speed up) is 23.6 ms
  vs. 6985 ms for the optimization itself — 0.3% of total fit time. Would not deliver a
  measurable speedup at current problem sizes, and would be a large, invasive rewrite
  (LSF matching, the continuum spline, and the entire test suite's mock generation all
  assume a linear wavelength grid). The current exact per-pixel Doppler resampling
  (`forward_model.shift_template`) is mathematically correct regardless of grid spacing —
  there is no accuracy defect from staying in linear-lambda, only a small, already-quantified
  effective-resolution variation (~4-5%) across a wide fit window. This would become the
  right call if template counts grow much larger (E-MILES-scale, hundreds of SSPs) — not
  currently.
- **Variable projection (VARPRO)** as a replacement for the alternating-minimization scheme
  (profile template weights out via NNLS at every trial LOSVD, single outer L-BFGS-B
  instead of alternating blocks). A 3-replicate smoke test looked excellent (9x faster, same
  bias). The full grid test told a different story: consistently *worse* bias than the
  current scheme in most cells, and one badly broken cell (S/N=60, sigma=150: 8 of 15 fits
  failed vs. 0 for the current scheme). Likely cause: the multi-start pool was seeded from a
  single arbitrary template rather than a sensible initial mixture — a fixable
  implementation gap, not necessarily proof the technique is unsound. **This is a legitimate
  place for a second opinion** — if you can identify a better multi-start/initialization
  strategy for the profiled objective, that would be genuinely new value, not a rediscovery.
- **E-MILES template integration.** Investigated at length (see `dev_notes/emiles/`).
  Verdict: viable but only with continuum-cofitting on and confined to the already-validated
  narrow (CaT, 8400-8800A) fit window — a wide window (1400A) causes catastrophic mismatch
  bias in *either* direction (whichever template set is mismatched pays severely). Not a
  "just wire it in" situation; not yet production-integrated.
- **Fixed vs. adaptive regularization.** The fixed `lam=2.0` default is a real, measured bias
  source, worse at low sigma. Adaptive regularization (`aic_regularization.
  select_lambda_aic_local`, a closed-form generalized-AIC criterion after Lipka & Thomas
  2021 / Thomas & Lipka 2022) has been validated at n=25 replicates/cell across both plain-
  Gaussian and genuine-wing (h4=0.15) truth, and beats the fixed default on sigma-bias in
  every cell tested (e.g. sigma=220: +50.0% -> +15.3%). This is settled; the open question is
  the calibration-table coupling below, not whether adaptive regularization helps.

## Known open issues — genuine, unresolved, good targets for fresh eyes

- **h3 recovery is severely suppressed** at low-to-moderate sigma for genuinely nonzero true
  h3 (near-total suppression across sigma=20-80, only starting to recover above ~140). No
  calibration data exists yet to correct this. Not investigated deeply this session.
- **sigma~300 km/s has an unresolved "sloppy manifold" over-coverage issue** — bootstrap
  intervals become measurably *too wide* as S/N improves, the opposite of the naive
  expectation. Not root-caused.
- **[STALE as of 2026-07-29, see banner above]** The bias-correction calibration table's
  `m_eff` axis (extending the Pinkney-style correction so it can be validly combined with
  adaptive regularization, not just the fixed `lam=2.0` it was originally calibrated at)
  was rebuilt against the `penalty_order=3` default's own lam grid
  (`dev_notes/calibration/build_emiles_calibration.py`,
  `src/kinextract_lite/calibration_data/regularization_bias_emiles_multitemplate_meff.txt`).
  The known severe-overcorrection failure this table mismatch caused
  (`examples/notebooks/01_basic_mock_fit.ipynb`'s lam=0.5 case) no longer reproduces
  (regression test: `tests/test_bias_correction.py::
  test_bootstrap_correction_does_not_severely_overcorrect_the_01_notebook_mock`). Full
  4-gate acceptance per `dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md` (per-cell
  bias, held-out coverage, worst-cell regression) is the subject of an in-progress overnight
  coverage sweep -- check `dev_notes/uncertainty/FINDINGS.md` for its outcome before treating
  this as fully closed.
- **Continuum-cofitting's own optimizer frequently reports `success=False`** ("ABNORMAL"
  L-BFGS-B termination). Traced this session to the continuum sub-fit's own optimizer (not,
  as first suspected, the LOSVD sub-fit), and a direct 60+ fit bias comparison found no
  statistically significant difference between flagged-success and flagged-failure fits —
  conclusion was that the flag is a scipy line-search technicality on an already-converged,
  low-dimensional problem, not a real accuracy signal. **Worth an independent check** if you
  have a different read on why L-BFGS-B would report this.

## The three things actually worth your budget

1. **Correctness audit.** Adversarial, line-by-line read of the core modules:
   `src/kinextract_lite/losvd.py`, `multi_template.py`, `continuum.py`, `noise_model.py`,
   `aic_regularization.py`, `bias_correction.py`, `uncertainty.py`, `gh_moments.py`. Look for
   actual bugs: sign errors, off-by-one, inconsistent conventions between modules (e.g. does
   every module agree on what "whitened" means, on array shapes, on which axis is which),
   unhandled degenerate inputs. Cross-reference against `tests/` to see what's already
   covered vs. not.
2. **Statistical-fidelity check.** Do the implementations actually match their cited papers?
   - Pinkney et al. (2003)'s bias correction: `bias_correction.py`, `calibrate_bias.py`.
   - Lipka & Thomas (2021, MNRAS 504, 4599) / Thomas & Lipka (2022)'s generalized AIC:
     `aic_regularization.py` (the criterion is `chi2 + w_m*m_eff`, `w_m=2.0`).
   - Eilers & Marx (1996) P-splines: `losvd.py`'s `difference_matrix`.
   This is "is this a faithful implementation," not "does it work empirically" (the latter
   is already measured).
3. **Speed audit of the actual bottleneck.** 95%+ of fit time is the alternating multi-start
   optimization (`multi_template._alternate_once`, `losvd.fit_log_spline_map`), not the
   forward model. Is there redundant computation in the alternation loop? Could JAX JIT
   compilation be cached/reused across calls more effectively? Is the multi-start pool
   (currently 3 starting points) over- or under-provisioned? And specifically: can the
   VARPRO attempt above be fixed with a better multi-start/initialization strategy?

## What NOT to spend time on

Do not re-derive or re-litigate: log-wavelength for speed, VARPRO-as-a-wholesale-replacement
(only "can the multi-start seeding be fixed" is open), whether E-MILES is production-ready
(it isn't, known), whether adaptive regularization helps (it does, measured). If your
analysis leads you back to one of these, say so explicitly and explain what's different
about your reasoning rather than silently re-proposing the same thing.
