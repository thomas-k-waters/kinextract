# kinextract_lite dev-history audit — full report (2026-07-29)

Full-repo audit of every `dev_notes/` subdirectory and root tracking doc, run
to build a complete, ranked picture of what's fixed and what's still open.
Scope note: this audit deliberately did not re-derive six items already
fresh at the time it was run (shape-residual -> noise-floor closure;
`max_edge_amplitude_frac` real-data fix; GH truth-vs-nominal mismatch and its
validation suite; bootstrap `reselect_lam` and the 47%/11% coverage finding;
calibration table regeneration; `shift_template` Lanczos fix) — those are
summarized only where needed for context and are covered in full elsewhere
(`dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`,
`dev_notes/validation_suite/FINDINGS.md`, `dev_notes/uncertainty/FINDINGS.md`,
`dev_notes/shift_template/FINDINGS.md`). Everything below is the older/
broader material, organized by section.

---

## (A) Full chronological arc — root tracking docs

**Files read in full**: `README.md`, `PARAMETER_GUIDE.md`,
`VALIDATION_CAMPAIGN.md` (all 1898 lines, Stages 0-27), `dev_notes/
AUDIT_BRIEFING.md`, `dev_notes/REGULARIZATION_INVESTIGATION_LOG.md` (760
lines), `dev_notes/REGULARIZATION_SHAPE_RESIDUAL_HANDOFF.md` (563 lines),
`dev_notes/PORT_STATUS.md`.

`VALIDATION_CAMPAIGN.md` is the master log; note the root also has a
companion memory file referenced repeatedly (`project_kinextract_lite_deep_
validation.md`, outside this repo) that was not accessible during this audit
— some "earlier session" claims cite it as the source and could not be
independently re-verified here.

**Stage 0-2 (pre-existing capability + broad sigma/S-N sweep + bootstrap
coverage, synthetic-only)**
- Stage 1: extended the sigma grid down to 20 km/s and confirmed **sigma=20
  km/s is essentially unresolvable regardless of S/N** (even S/N=200 gives
  +50% bias under MUSE sampling, +34% under a finer "high-res" convention —
  "resolution helps only modestly, this is an S/N effect not a sampling
  effect"). Also found the wing-inflation mechanism **reappears at high
  sigma (220-400) when S/N drops to 15-30** — previously undetected because
  the standard grid always paired high sigma with S/N=100.
- Stage 1c: **h3 recovery is severely suppressed** at sigma 20-80 for any
  true h3 (near-total suppression, not just bias) — never previously
  characterized. The low-sigma h4 bias is an **additive floor** (~+0.03
  regardless of true h4, even at true h4=-0.1).
- Stage 2: the capstone synthetic finding of the whole campaign — coverage
  failure is **two distinct, opposite-signed mechanisms that happen to
  cross near sigma=140/S-N=100** (why that cell looked historically
  well-calibrated): low sigma under-covers and improves with S/N; high sigma
  over-covers and gets *worse* with more S/N (e.g. sigma=300: 57%->65%->
  85%->95% as S/N climbs 15->100). sigma=20 is 0% coverage at every S/N
  tested.

**Stage 3-8 (feature composition, sky-residual masking, cleaning, continuum
— all real-data-focused)**
- Stage 4/6/7: built fixed-wavelength masking and iterative sigma-clipping;
  both individually validated on synthetic contaminants; **both made real
  NGC 5102 data measurably worse, not better** (e.g. bnn1605:
  legacy-relative bias +13.6->+25.8 km/s after cleaning).
- Stage 8: built the 3-block continuum co-fitting architecture; the single
  most important open design question (does continuum-fitting let template
  weights "cheat" via an adversarial continuum-vs-weight-mimicry setup)
  **resolved cleanly in favor of the architecture** (plain undamped
  alternation recovered the true 50/50 split, mean max weight error < 0.05).
  Applied to real data with raw (non-pre-normalized) templates: **made
  agreement with legacy worse, not better**.
- **The unifying finding across Stages 4/7/8**: three independent,
  synthetically-validated fixes (masking, cleaning, continuum) each failed
  or backfired on real NGC 5102 data — strong evidence the real driver was
  none of the three. Leading untested hypotheses at the time: imperfect
  upstream pre-normalization, LSF/resolution mismatch, or per-bin `wing_lam`
  tuning. (Direct ancestor of the real-data-sigma investigation already
  closed.)

**Stage 9 (Pinkney bias correction, first calibration attempts, ablation
framework)**
- First two calibration attempts (idealized single-template; fixed
  real-star mixture) **both failed hard against legacy** (~90-96% of bins
  worse). Root cause: **legacy (`pallmc.out`) is not ground truth** — this
  reframed the whole project's evaluation methodology from "match legacy" to
  "match known synthetic truth," a standing lesson repeated in later stages.
- Built the ablation-ladder framework (`kinextract_lite.ablation`).
  **Headline finding: the unregularized fit (`lam=0`) beats every
  regularized configuration including the shipped default in aggregate** —
  a real crossover by sigma (at sigma<=~80 the curvature penalty is a net
  bias *source*; only becomes net-helpful at sigma>=~140). Empirical
  trigger for the later sigma-adaptive-regularization work (Stage 12) and,
  more distantly, for the whole shape-residual/penalty-form investigation
  (Stage 22).

**Stage 10 (E-MILES calibration rebuild, 4 bugs found)**
Rebuilt the bias-correction calibration against E-MILES (replacing a
MUSE-star-specific table) after discovering the correction was calibrated
for one real-star library and **actively made things worse (24/25 draws) on
a toy Gaussian-template mock** — the mismatch was about template *shape
realism*, not which specific real-spectrum library. Found and fixed: (1)
`b_p16/p84_corrected` band-width not actually inflated, only recentered; (2)
`estimate_m_eff_local` crashes at `lam=0` on narrow grids (`solve`->
`lstsq`); (3) `save_calibration_table` always writing 13 columns, causing
false m_eff-axis auto-detection; (4) two notebooks built mocks at E-MILES'
native ~0.9 A/pix grid instead of MUSE's 1.25 A/pix, a silent sampling
mismatch.

**Stage 11 (continuum amplification bug found)**
Continuum-cofitting **amplifies a small pre-existing LOSVD bias into a
large, misleading spectral-fit residual** (~0.005 amplitude / r=0.06 without
continuum vs ~0.04 / r=0.999 with it, on an otherwise identical mock) — root
mechanism not found until Stage 15.

**Stage 12 (adaptive AIC lam promoted to default)**
`lam_selection="aic_local"` became default. A real regression found and
fixed in the same stage: `bootstrap_losvd` was auto-switching to the
thinner-sampled `m_eff`-axis calibration table *whenever* adaptive selection
ran, even when the picked `lam` matched the plain table's own calibration
point — fixed via a 25%-difference-from-table threshold.

**Stage 13 (head-to-head kinextract_lite vs OG kinextract harness)**
Built (in the sibling `kinextract_test_suite` repo) a harness driving OG
kinextract's file-based API directly on identical E-MILES mocks. Found and
fixed two config traps (OG's `.norm` reader silently ignoring the error
column by default; a missing `cfg.step` requirement). Flagged, never
resolved: **OG's `fit_losvd_gauss_hermite` may use a different h3/h4
Hermite convention** — not verified comparable.

**Stage 14 (two overcorrection bugs; `apply_bias_correction` flips to
default False)**
First ruled out a "skew bug" hypothesis empirically (V/h3 not statistically
distinguishable from zero, sigma/h4 are — matches everything already
known). Found a **second overcorrection bug**: even when the m_eff-table
switch is warranted, its correction can overcorrect badly (one seed: raw
sigma 52.9 km/s already-good -> corrected 35.5 km/s, ~5x worse). **Decision:
bias correction defaults off.** Produced `dev_notes/calibration/
CALIBRATION_TRUST_STRATEGY.md` (still marked "not started" as of this
audit — see §B).

**Stage 15 (speed/crash fixes; continuum bug root-caused)**
JIT-cache hoisting (~1.2x/fit) + `replicate_single_start=True` bootstrap
default (~2.5x). Crash bug fixed (NNLS driving all weights to zero on
iteration 1). **Continuum amplification bug (Stage 11) root cause found**:
the continuum spline's D2 penalty has an exact zero-cost null space
(constant+linear), so an arithmetic-sequence coefficient vector defeats any
`lam_c`; fixed via a new `ridge_c` L2 term. Also settled (after two prior
misdiagnoses): the frequent continuum-block `success=False` is a
**spurious scipy line-search technicality**, not a real accuracy signal
(confirmed at ~80-rep scale: bias statistically indistinguishable between
flagged-success/failure groups).

**Stage 16 (cleanup; three failed `wing_lam` auto-selection approaches
documented)**
Removed `plotting.py` (dead code). Documented in detail (so it's not
re-attempted) **three rejected `wing_lam` auto-calibration schemes**:
reference-model wing-mass matching (every reference estimator carried its
own S/N-dependent bias), bootstrap bias-correction of the point estimate
(corrupted already-good cells, e.g. 100%->58% coverage), and a
lam-sensitivity diagnostic (worked at sigma=50, wrong-signed/noise-dominated
at sigma=140/300).

**Stage 17 (real-data pipeline rebuilt self-contained)**
New modules `muse_cube.py`, `binning.py` (dartboard scheme), `mock_cube.py`,
`template_reduction.py`, `eso_archive.py`. Real NGC 5102 MUSE-DEEP cube:
636->9 templates in 3 iterations. Found that STAT-extension per-pixel
errors give **chi2_red ~15.7/11.3/27.5** at the three representative
apertures — confirmed a genuine systematic model-data residual (not
masking/regularization), root cause (LSF mismatch? continuum? template
incompleteness?) **left open, not resolved**.

**Stage 18 (examples cleanup; "skew" root-caused to wing-amplitude
inflation)**
The visible LOSVD "skew" in notebooks 01/02 = ordinary wing-amplitude
inflation, simply never turned on there. Also documented directly: **the
bootstrap 16th-84th percentile band is not guaranteed to contain its own
point estimate** — checked on real data (roughly half the velocity grid
falls outside the band in both real-data notebooks) — "expected, not a
bug," per the mechanism that every replicate is refit at the point
estimate's fixed lam/wing_lam.

**Stage 19-21 (spectral_prior exploration — built, explored at length,
ultimately not adopted)**
A 2-parameter generalization of the curvature penalty (`alpha, mu_i^beta`).
Stage 19's 3-regime spot-check looked like a clean win; Stage 20's
properly-powered 3200-fit rerun **refuted it** (worse than baseline on
`muse33`, no better in aggregate) and found a real sigma-dependent
crossover. Stage 21 root-caused the crossover to a >30x scale confound in
`mu.max()` across the sigma range and fixed it via eigenvalue normalization
— got to "real win at low sigma, roughly neutral at high sigma, not a clean
universal win." **This entire module (`spectral_prior.py`) and its notebook
were later deleted** (per `PORT_STATUS.md`, "superseded by Stage 22" —
confirmed still absent from the current tree).

**Stage 22-27**: the six items excluded from this audit's scope (see the
intro note above) — penalty_order=3 default flip; the real-data sigma fix
(`max_edge_amplitude_frac`); GH-truth-vs-nominal mismatch + validation
suite; bootstrap lam-pin fix + honest 47%/11% coverage; calibration
regeneration; `shift_template` Lanczos fix. The campaign file's account of
these was cross-checked and found consistent, no contradictions.

---

## (B) Per-dev_notes-subdirectory findings

**`dev_notes/shape_residual/`** (scripts/CSVs only, no FINDINGS.md —
narrated in `REGULARIZATION_INVESTIGATION_LOG.md`). One item worth flagging
beyond the already-known closure: a **"SHASH core" (sinh-arcsinh) tangent
tested 2026-07-28**, after D2/D3 — solves the GH-core's divergence but
**does not beat the D3 spline** (12.77% vs 7.73%/6.19% on the NB02
benchmark) and collapses via gradient underflow at heavy lam/small sigma.
It was this test — a 4-parameter SHASH fit with *no* spline and *no*
penalty still giving 11.5% residual at S/N=60 — that finally closed the
shape-residual investigation as a noise floor, not a penalty defect.

**`dev_notes/skew/FINDINGS.md`** — diagnosed the notebook-02 "skewed LOSVD"
appearance (2026-07-27), the direct precursor to the GH truth-vs-nominal
mismatch. Distinct additional finding: a direct curve-level quantification
that **order-3 erases h4 entirely** on this mock (h4=+0.000 at every usable
lam vs order-2's +0.033, true h4=+0.050) — "order 3 is robust but erases
h4; order 2 recovers h4 but needs a lam selection does not find." Also
flags, as still-open: a penalty whose null space is the *Gauss-Hermite
family* rather than the Gaussian family (would keep order-3's robustness
without erasing h4) — proposed, not attempted.

**`dev_notes/speed/RESULTS.md`** — profiled a user-reported 38s/fit slowdown
to `losvd.fit_log_spline_map` (114 L-BFGS-B calls per default fit). Two
structural fixes: warm-starting the LOSVD sub-fit across alternating
iterations, and reusing the AIC-selection scan's own winning-lam fit
instead of refitting. **6.8x faster (34.1s -> 5.0s)**, verified
quality-neutral via paired A/B (12 seeds x 3 sigma values). Resolved,
current default behavior.

**`dev_notes/noise/`** (3 scripts, no FINDINGS.md) — established real MUSE
pixel noise carries **AR(1) lag-1 correlation ~0.3-0.5** (surviving even
after continuum-detrending; the raw, non-detrended ACF ~0.99 was mostly
continuum-mismatch trend, not real correlation). This produced real package
infrastructure (`noise_model.build_ar1_covariance`, `whiten(rho=...)`,
`estimate_correlation_length`) — but **`FitConfig` has no `rho` field at
all**, and grepping every example notebook (including
`05_real_data_muse.ipynb`) confirms **no notebook or default code path ever
passes a nonzero `rho`** — every fit, including on real data, silently uses
independent-noise whitening (`rho=0.0`). This means chi2/AIC/bootstrap
variance on real data are plausibly all mildly miscalibrated by an
unaddressed, already-measured correlation. Not fixed, not wired in.

**`dev_notes/continuum/`** — `stage6_compare_masked.py`/
`stage8_real_data_continuum_check.py` reproduce Stage 6/8 numbers exactly
(h4 sign flips positive vs legacy's negative in every bin — visible in the
raw log but not called out explicitly in `VALIDATION_CAMPAIGN.md`'s own
Stage 8 text). Fully consistent with, and superseded in narrative terms by,
Stage 8/11/15's resolution.

**`dev_notes/coverage/`** — pre-existing (order-2-era) coverage-calibration
exploration, e.g. `coverage_calibration_results.txt` (sigma=50/h4=0/S-N=40:
sig_cov68=46.7%; h4=0.15/sigma=50/S-N=40: h4_cov68=0.0%) and
`fix_low_sigma_wing_coverage.log`/`rerun_coverage_with_h4wing_calibration.
log` sweeping `lam` in {0,0.5,1.0} against the old bias-correction tables.
All superseded by the much more decisive, current-defaults 47%/11%
coverage_sweep numbers. No contradiction, just an earlier, smaller-scale,
order-2-era version of the same question — worth noting only because none
of these files carry a "superseded" pointer forward the way other dirs'
READMEs do.

**`dev_notes/emiles/`** — E-MILES/MUSE-star template-mismatch stress tests
(order-2 era, `lam` in {0,2}). Confirms `AUDIT_BRIEFING.md`'s "viable but
only with continuum-cofitting on and confined to the narrow CaT window"
claim precisely: e.g. `emiles_mismatch_test_widewindow_results.txt` shows
the wide (1400A) window giving **+170% to +208% sigma bias** for template
mismatch in either direction, vs the narrow-window numbers (still large,
e.g. +204%/+128% at sigma 30/50, S/N=15, but the qualitative "confined to
narrow window" recommendation holds). Historical/superseded in exact
numbers (pre-adaptive-lam, pre-order-3) but the architectural conclusion
(E-MILES not production-integrated, confined scope) matches
`AUDIT_BRIEFING.md`.

**`dev_notes/regularization/`** — `varpro_test_results.txt` confirms
`AUDIT_BRIEFING.md`'s VARPRO claim to the exact number: `alternating`/
`varpro` rows at S/N=60, sigma=150 show `nfail=0` vs `nfail=8` (of 15) — the
literal "8 of 15 fits failed" cited in the briefing.
`aic_local_firming_up_results.txt`/`ablation_sweep_results.txt` confirm the
20-57%-reduction and ablation-ladder claims. All superseded-but-consistent
background for Stage 9/12; no contradictions.

**`dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md`** (2026-07, marked
**"Status: proposed, not started"**) — a significant, still-unactioned
document: it defines a 4-gap acceptance strategy (dense per-lam grids;
h3/h4 truth-domain coverage; template-conditioning axis; a per-query
confidence signal) and an explicit **acceptance gate** ("no calibration
table changes the default until it passes: per-cell not aggregate bias
check, independent held-out *coverage* check, worst-cell regression check,
a regression test per bug found") before `apply_bias_correction` can
default back to `True`. **The most recent calibration table regeneration
does not appear to have run this gate** — it verified load-correctness and
`test_bias_correction.py` passing, not per-cell/coverage criteria — and the
honest 47%/11% coverage number was measured WITHOUT bias correction applied
at all.

**`dev_notes/model_reference/`** (`model_reference.html`,
`current_state.html`) — `AUDIT_BRIEFING.md` explicitly tells a fresh reader
"read both before touching code; they are more current and more complete
than any docstring-level summary could be." **These files are dated
2026-07-23** (only date string found in either file) and contain **zero
occurrences** of `penalty_order`, `max_edge_amplitude_frac`,
`reselect_lam`, or `sigl0` — i.e. they predate the entire Stage 22-27 arc.
A genuine staleness finding, not a contradiction of fact — see §C/D.

**`dev_notes/scratch_notebooks/`** — `bias_sweep_fixed_vs_adaptive.py` is
order-2-era and is already self-documented (per the investigation log's own
note) as "kept as a historical order-2-era record" — not misleading, just
old. `fit_all_bins.py`, `ngc5102_check.ipynb`, `real_data_all_bins*.npz` are
working artifacts from the Stage 9/17 real-data pipeline, superseded by the
current self-contained notebooks 04/05.

**`dev_notes/og_tuning/`** — **the directory exists but is completely
empty** (0 files). Whatever this was meant to hold (per-bin OG-kinextract-
style tuning, presumably related to the PORT_STATUS.md taper effort or the
head-to-head harness) was never populated in this repo.

---

## (C) Conflicts / staleness found between older material and current state

The highest-value section. Four real findings:

1. **`README.md` and `PARAMETER_GUIDE.md` are substantially stale relative
   to current source and current status**, and this is not cosmetic. Both
   files were last touched 2026-07-27 (`README.md` 17:02,
   `PARAMETER_GUIDE.md` 06:21) — **before** `dev_notes/PORT_STATUS.md`
   (17:13 the same day) and well before the `max_edge_amplitude_frac` fix
   (2026-07-28), the bootstrap `reselect_lam` fix, and the calibration
   regeneration (all 2026-07-29). Confirmed by grep: **neither file
   mentions** `max_edge_amplitude_frac`, `reselect_lam`, `taper_sfac`/
   `sigl0`, `spectral_prior`, "Cramer-Rao"/"noise floor," `validation_suite`,
   or `coverage_sweep` at all. Concretely:
   - Both files' entire `wing_lam` guidance (including
     `PARAMETER_GUIDE.md`'s specific per-sigma tuning table: `wing_lam=3e6`
     at sigma=20, `3e5` at sigma=50/80/140, etc.) describes the **old,
     pre-port mechanism** (an amplitude ramp keyed to `wing_onset_frac`, a
     grid-position fraction). That field (`wing_onset_frac`) **no longer
     exists in `config.py`** (confirmed by grep, zero hits). The current
     `lam_wing`/`taper_sfac`/`taper_power`/`n_sigl0_iter` mechanism (from
     the OG-kinextract taper port) is a **different formula** (onset
     relative to the fit's own recovered width `sigl0`, not grid position)
     — so those specific tuned numeric values in `PARAMETER_GUIDE.md` are
     very likely no longer meaningful for the field they're attached to.
   - Neither doc mentions that the actual, *successful* fix for real-data
     sigma inflation was `max_edge_amplitude_frac` rather than any
     `wing_lam` value — both docs still frame `wing_lam` as "the"
     real-data-inflation remedy ("**Recommendation: use `lam_wing` for
     sigma inflation at low S/N, not outlier rejection**" — README, Known
     Limitations), now superseded advice.
   - `README.md`'s "Known limitations" bullets on bootstrap coverage still
     describe the pre-`reselect_lam`, pre-coverage_sweep picture (an
     "unreconciled discrepancy between two measurements of the same
     std-inflation number, 1.47x vs ~1.0x... flagged as needing a clean
     re-measurement, not yet done") — superseded by the much more decisive
     47%/11% coverage numbers.
   - `PARAMETER_GUIDE.md`'s `wing_lam` field name is written **backwards
     throughout the document** (18 occurrences of `wing_lam`) vs. the
     actual `FitConfig` field, which is named `lam_wing`. `README.md` gets
     the name right (`FitConfig.lam_wing`).

2. **`dev_notes/AUDIT_BRIEFING.md`** (2026-07-25 20:40) **predates the
   entire penalty_order=3 investigation and everything after it**. It is
   internally consistent and accurate for what it covers (VARPRO, E-MILES,
   fixed-vs-adaptive lam, the m_eff-axis-untested caveat, the continuum
   success=False question — all still true as far as this audit could
   tell), but a reader following its explicit instruction to treat it as
   "empirically settled... so you don't have to rediscover" would be
   missing the single largest architectural change in the project's history
   (the default penalty order flip and its consequences).

3. **`dev_notes/model_reference/*.html`** (2026-07-23) — same issue, more
   severe, since `AUDIT_BRIEFING.md` explicitly tells a reader these are
   "more current and more complete than any docstring-level summary could
   be." They are the single most out-of-date piece of documentation in the
   repo relative to the codebase's actual current state.

4. **A concrete, in-source stale claim** (not just doc drift): `uncertainty.
   py`'s own module docstring still says applying bias correction under
   `penalty_order=3` is unsafe because "neither calibration table has been
   rebuilt for it... this is a genuinely open recalibration task, not yet
   started" — factually superseded by the table regeneration. The clearest
   direct contradiction between an old claim and newer work found anywhere
   in the repo.

No contradictions of substance were found in the six items excluded from
this audit's scope — everything cross-checked (README's older wing_lam
story, the ablation/VARPRO numbers, the coverage-calibration scripts) is
consistent with, just chronologically earlier than, current understanding.

---

## (D) Source-code docstring/comment staleness (grep-based sweep)

- **`src/kinextract_lite/uncertainty.py`** (module docstring, ~line 89):
  explicitly states penalty_order=3 + bias correction is untested/unsafe and
  recalibration "not yet started" — **stale**, contradicted by the table
  regeneration. Needs an update pointer (currently has none — grepping for
  "Stage 26"/"Stage 25"/"Stage 27"/"regenerat" across all of `src/` returns
  zero hits anywhere in the package).
- **`src/kinextract_lite/config.py`**, `lam_wing` field help text (line
  ~298): says **"Default 512."** immediately after the field is declared
  `_f(0.0, ...)` — the docstring text disagrees with the actual default
  value in the same statement. Reads like a copy-paste leftover from an
  earlier draft where 512 actually was the default (plausible, since
  `PORT_STATUS.md`'s own measured-effect table uses `lam=512` as an
  experiment value) — should just say "Default 0.0 (off)."
- All other `dev_notes`-referencing docstrings checked (`losvd.py`,
  `aic_regularization.py`, `bias_correction.py`, `calibrate_bias.py`,
  `continuum.py`, `emiles.py`, `forward_model.py`, `config.py` elsewhere,
  `__init__.py`) correctly and currently point at the newest relevant
  findings docs — in good shape, not stale.
- Neither `PARAMETER_GUIDE.md` nor `AUDIT_BRIEFING.md` is referenced from
  any source docstring at all (zero grep hits) — consistent with them
  having drifted out of the maintained cross-reference web while
  `VALIDATION_CAMPAIGN.md` and the `dev_notes/*/FINDINGS.md` files were kept
  current.

---

## (E) Open items not covered by this audit's excluded six

1. **Real MUSE pixel noise has measured AR(1) correlation (~0.3-0.5 lag-1)
   that is never actually applied.** The mechanism exists
   (`noise_model.whiten(rho=...)`, `estimate_correlation_length`) but
   `FitConfig` has no `rho` field and no notebook — including the
   real-data ones — ever passes a nonzero `rho`. Every chi2/AIC/
   bootstrap-variance number computed on real data is thus computed under
   an assumption already measured to be wrong.
2. **`CALIBRATION_TRUST_STRATEGY.md`'s acceptance gate was never
   implemented.** Its four proposed gates (per-cell not aggregate bias
   check; independent held-out coverage check; worst-cell-regression
   check; a per-query confidence/domain-distance signal) are the explicit
   precondition the document sets for turning `apply_bias_correction` back
   on by default, and none appear to have been run even after the table
   regeneration.
3. **h3 recovery is severely/near-totally suppressed at sigma 20-80** for
   any nonzero true h3 (Stage 1c, and independently in `skew/FINDINGS.md`).
   No calibration or correction exists for this at all (unlike h4, which at
   least has the additive-floor characterization).
4. **sigma~300 km/s "sloppy manifold" over-coverage** (bootstrap intervals
   get measurably *too wide*, i.e. worse, as S/N improves) is flagged
   repeatedly (Stage 2, `AUDIT_BRIEFING.md`) as unresolved and not
   mechanistically root-caused — distinct from, and possibly but not
   proven related to, the low-sigma coverage problem.
5. **The continuum-cofitting real-data residual (Stage 17's chi2_red
   ~15.7/11.3/27.5)** is confirmed to be a genuine systematic
   model-data mismatch, root cause (LSF mismatch vs template-set
   incompleteness vs un-fit continuum shape) never isolated.
6. **A penalty whose null space is the Gauss-Hermite family rather than the
   plain-Gaussian family** — proposed in `skew/FINDINGS.md` as the way to
   get order-3's robustness without erasing real h4 — is a stated,
   concrete, not-yet-attempted research direction.
7. **VARPRO's multi-start/initialization gap is explicitly still an open
   invitation** in `AUDIT_BRIEFING.md` ("a legitimate place for a second
   opinion") — the technique itself wasn't proven unsound, just badly
   seeded in the one implementation tried.
8. **OG kinextract's h3/h4 Hermite-polynomial convention vs kinextract_lite's
   own was never verified to be numerically comparable** (Stage 13) — a
   live risk for any future head-to-head h3/h4 comparison.
9. **Multi-template weight-recovery reliability under a real, uncurated
   template library is far worse than any curated recipe tested** (Stage
   19: median sigma bias 39.7% for `muse33` vs 4-7% for curated recipes,
   even in the "safe" S/N>=60/sigma>=140 regime, at chi2_red essentially
   unchanged — a genuine identifiability problem, not a fit failure). This
   is the empirical justification for the template-reduction pipeline, but
   the underlying instability itself is not "fixed," only worked around by
   curation.
10. **`dev_notes/og_tuning/` is an empty stub** — whatever per-bin OG-style
    tuning work this was meant to hold was never done or was moved
    elsewhere without a pointer.
11. **The template-continuum-trust hierarchy** (library-keyed presets for
    how much of a raw template's own continuum to trust) is explicitly "not
    yet implemented" per `README.md`'s Features section and was never
    picked back up after the continuum architecture's real-data failure.
12. **A wavelength-dependent LSF-matching capability and a dedicated
    real-data I/O/orchestration layer** are both listed as not-yet-
    implemented in `README.md` and never appear again in any later stage.

---

## Priority ranking — everything open, most to least urgent

This ranks every currently-open item across this audit plus the six
excluded-but-still-open items from prior work, in the order they should
probably be picked up.

1. **Uncertainty-band coverage is 47%/11% against a 68% target** (item
   excluded from this audit's scope, still the single most referee-visible
   problem). The dominant remaining mechanism is resampling around a
   biased point estimate; `reselect_lam` fixed the spread half of this but
   not the center. This is priority 1 because it is the number a reviewer
   checks first and it fails outright.
2. **The real-MUSE-data AR(1) noise correlation is never applied** (item
   E1). This isn't a subtle secondary effect — it means every chi2/AIC/
   bootstrap-variance number ever reported on real data (including in the
   published notebooks) rests on a noise model already measured to be
   wrong. High priority because it's cheap to characterize further and
   plausibly touches everything downstream of it.
3. **`CALIBRATION_TRUST_STRATEGY.md`'s acceptance gate has never been run**
   (item E2). Until it is, `apply_bias_correction` cannot be trusted back
   to default-on, and the coverage work in (1) can't cleanly separate
   "spread is wrong" from "center is wrong." This blocks real progress on
   (1).
4. **Documentation staleness** (README.md, PARAMETER_GUIDE.md,
   AUDIT_BRIEFING.md, model_reference/*.html, and the uncertainty.py
   docstring — item C1-C4/D). High priority not because it's technically
   hard, but because a stale doc actively misleads (wrong field names,
   wrong claimed fix, wrong claimed status) rather than just being
   incomplete — this is a correctness risk for anyone (including a future
   session) trusting these docs at face value.
5. **h3 recovery is severely suppressed at sigma 20-80** (item E3 /
   Stage 1c). No correction exists at all, unlike h4's characterized
   additive floor. Directly limits what science can be claimed from any
   low-sigma real-data h3 measurement.
6. **sigma~300 "sloppy manifold" over-coverage** (item E4 / Stage 2). Not
   mechanistically understood; the fact that it gets *worse* with more S/N
   is counterintuitive enough to need a real explanation before it can be
   worked around.
7. **The continuum-cofitting real-data chi2_red excess** (item E5 /
   Stage 17). Root cause unisolated among three plausible mechanisms (LSF,
   templates, continuum shape) — needs a dedicated ablation, not further
   real-data-only iteration.
8. **Multi-template identifiability under uncurated libraries** (item E9 /
   Stage 19). Currently worked around by curation (the template-reduction
   pipeline), which is a real mitigation but not a fix — worth knowing the
   workaround's limits before leaning on it harder.
9. **A Gauss-Hermite-null-space penalty** (item E6 / `skew/FINDINGS.md`) —
   a concrete, well-motivated, not-yet-attempted idea that could let
   order-3's robustness coexist with real h4 recovery. Medium priority:
   promising but speculative until someone actually tries it.
10. **VARPRO's seeding gap** (item E7) and **OG-vs-lite h3/h4 convention
    parity** (item E8) — both flagged as open technical questions with no
    evidence either way; worth resolving before either technique is relied
    on for a real comparison or as a fitting alternative.
11. **Missing features already scoped but not built**: template-continuum-
    trust hierarchy, wavelength-dependent LSF matching, real-data I/O/
    orchestration layer (items E11-E12). Lower priority — these are
    roadmap items, not correctness bugs.
12. **`dev_notes/og_tuning/` empty stub** (item E10) — lowest priority,
    purely a cleanup/bookkeeping item (delete the stub or note why it's
    reserved).
