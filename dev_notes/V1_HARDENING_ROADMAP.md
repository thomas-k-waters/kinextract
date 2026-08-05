# kinextract v1.0 hardening: uncertainty overhaul, continuum integration, noise correlation, and a full notebook revamp

## Status as of 2026-08-04

This doc reads below as a forward-looking plan (all phases framed as
not-yet-done) but most of it has since landed, just never reflected here
-- treat the phase text below as historical design rationale, and this
status block as the actual current state.

**Done**: Phase 0 (housekeeping -- verified item-by-item 2026-08-04, all
clear); Phase 1a-1d (Laplace uncertainty engine promoted to
`laplace_uncertainty.py`, production-scale coverage sweep validated it as
the new default across sigma 50-600 km/s -- see
`dev_notes/uncertainty/FINDINGS.md`'s 2026-08-01/08-04 entries -- NUTS
reference mode built as `nuts_uncertainty.py`); Phase 2 (continuum
DC-offset root-caused as a real, structural `(w_sum, C_mean)`
non-identifiability rather than a bug -- see
`dev_notes/continuum/FINDINGS.md` -- `penalty_order=3`/fixed-`lam`
promoted to the automatic default, multi-seed sweep rebuilt); Phase 3
(noise-correlation convenience function added to `noise_model.py`);
Phase 5 (notebook revamp -- 01-08 renumbered/repurposed to 01-07, old
notebooks deleted, dev-narration/stale-path audit came back clean).

**Not done**: Phase 1a's continuum-coefficient Hessian extension (Laplace
uncertainty doesn't yet report continuum-coefficient uncertainty jointly
with LOSVD uncertainty when `continuum_knots` is set); Phase 4 (h3
calibration table build started, `build_h3_table_incremental.py` exists,
not confirmed complete/wired in); the residual low/moderate-sigma h4
coverage gap (real, quantified, root-cause hypotheses documented, no fix
attempted -- see `dev_notes/uncertainty/FINDINGS.md`'s final entries).

## Context

`kinextract` (renamed from `kinextract_lite`, now v1.0.0) went through an enormous validation campaign (27 stages in `dev_notes/VALIDATION_CAMPAIGN.md`) that root-caused and fixed the core LOSVD shape-residual problem and the real-data sigma-inflation bug. That work surfaced several things that are now the actual remaining gaps between "the fitting engine works" and "this is a trustworthy, teachable scientific tool":

1. **Uncertainty bands are not trustworthy**, especially for h4: bootstrap coverage sits at ~50-80% for sigma and **~3-11% for h4** against a 68% target, even after two rounds of real fixes (`reselect_lam`, `PerLamCalibration`). The mechanism is structural — bootstrap resamples around a fixed point estimate and cannot see that estimate's own bias/shrinkage.
2. **Continuum-cofitting is real but not production-hardened**: it silently falls back to a *worse* configuration (`penalty_order=2, lam=2.0`) whenever adaptive lam selection is on, and notebook 02's own demo shows a visible vertical offset between the recovered and true continuum that was never root-caused.
3. **Noise-correlation handling exists and works** (`noise_model.py`, demonstrated end-to-end in `07_noise_correlation.ipynb`) but requires the caller to hand-orchestrate it — not integrated into the real-data pipeline.
4. **The example notebooks accumulated a lot of development-process narration** ("we fixed X", stale cross-references, a hardcoded personal machine path in notebook 06, mismatched binning schemes between the calibration notebook and the capstone notebook, warning spam) rather than reading as a clean, honest, pedagogical guide to the package.

This plan covers both the engineering (Main Task 1) and the notebook revamp (Main Task 2) the user asked for, explicitly designed to survive across multiple sessions — treat this file as the durable checklist. Once approved, the first action is to also save a copy into `dev_notes/` (e.g. `dev_notes/V1_HARDENING_ROADMAP.md`) so it's git-tracked alongside everything else this project uses `dev_notes/` for, not just in the Claude Code plan-file location.

## Decisions locked in (asked directly, all confirmed)

- **Uncertainty**: the Laplace + AIC-weighted-over-lam method becomes the new **primary/default** uncertainty engine once validated at production scale. Bootstrap stays available as an alternative/cross-check (real ensemble properties Laplace can't give, e.g. empirical skew/multimodality) — stop investing further in bootstrap-specific coverage fixes.
- **Full Bayes**: build NUTS (via NumPyro, new optional dependency) as an **optional reference/validation mode** — used for simulation-based-calibration cross-checks of the Laplace approximation, and offered as an opt-in slow "gold standard" for end users. Not the default, not optimized for speed.
- **Continuum + adaptive lam**: "fully integrated" means a validated, automatic fixed-lam default (no more silent fallback to the worse order-2/lam=2.0 combo) with honest error propagation. Adaptive lam selection jointly with continuum-cofitting is a tracked future item, not a blocker.
- **Binning naming**: keep `DartboardBins`/`binning.py` as internal implementation names (no API churn). Stop using "dartboard" in user-facing notebook prose; use precise descriptive language instead.

## Dependency map (why the phases are ordered this way)

Notebook 02 (regularization + uncertainty + continuum) and notebook 05 (noise correlation) are explicitly supposed to *teach* the new uncertainty engine, the fixed continuum integration, and the noise-correlation workflow — they cannot be honestly rewritten before that underlying work lands. Notebooks 01, 03, 04, 06 are largely independent of Phases 1-3 and can proceed in parallel. Notebook 08 (capstone) depends on 04's realigned template subset and should be polished last.

---

## Phase 0 — Housekeeping (fast, do first, unblocks nothing but prevents confusion downstream)

- `04_template_calibration.ipynb`: reset checked-in `DATA_MODE` from `"download"`/`CONFIRM_DOWNLOAD=True` back to `"synthetic"` — as currently saved, re-running it top-to-bottom would trigger a real ~4.5GB ESO download.
- Fix the `"dartboad_bins"` colormap-name typo in `04_template_calibration.ipynb` cell 8.
- Fix the stale `04_real_data_muse.ipynb` reference in `src/kinextract/mock_cube.py:30` (renamed to `05_real_data_muse.ipynb` long ago).
- Fix stale `06_template_mismatch_diagnostic.ipynb` references in `src/kinextract/template_reduction.py:79` and `dev_notes/calibration/build_emiles_calibration.py:83` / `CALIBRATION_TRUST_STRATEGY.md:90` (that notebook was rebuilt into today's `04_template_calibration.ipynb` long ago).
- Remove dangling `dev_notes/scratch_notebooks/` cross-references in `08_full_galaxy_analysis.ipynb` (cells 0, 9, 12, 18) — that directory doesn't exist in this repo.
- Resolve the template-subset path disconnect: `04`'s current save cell writes `../data/template_subset_{DATA_MODE}.npz`, but `05`/`08` both hardcode-read `../data/muse/ngc5102_template_subset.npz` (an older, numerically different file — 9 templates vs. what `04` would produce today). Make `04` save directly to the path `05`/`08` actually consume.
- Add a plain `b_median` field (uncorrected, `np.median(b_boot, axis=0)`) to `bootstrap_losvd`'s return dict in `src/kinextract/uncertainty.py` — currently only `b_median_corrected` exists, and every notebook's final LOSVD plot needs to present the median-bootstrap curve, not the raw point estimate, per your explicit instruction (see Phase 5 cross-cutting note).
- Vendor the real STIS data into the repo: copy `bin0105sp.norm`, `bin0105sp.rms` (currently only at `/Users/waterstk/Documents/Gultekin_Astrophysics/normalizing_flows/data/N5102/kinematics/stis/kinematic_extraction/`) and the 50-star template library + `Tlist` (currently only at `.../sco_framework_updated/sco_workdir_example/analysis/stis_templates/`) into `examples/data/stis/`, so notebook 06 is portable rather than machine-specific — directly serves its own stated teaching goal ("adapt this code to other instruments").

## Phase 1 — Uncertainty engine overhaul (Main Task 1.1)

**1a. Promote Laplace + AIC-weighted-over-lam from prototype to package function.**
Starting point: `dev_notes/uncertainty/laplace_prototype.py`'s `laplace_aic_uncertainty`/`laplace_covariance_at_lam` (reuses `aic_regularization.estimate_m_eff_local`'s own linearization and fit, takes `jax.hessian` of `losvd._neg_log_post` at the converged coefficients — already the negative log*posterior*, i.e. curvature penalty included, matching the ChatGPT recommendation exactly). Gaps to close before it's a real `bootstrap_losvd` sibling, not just a moments-only screening tool:
- Add LOSVD-curve-level uncertainty: sample `coeffs ~ N(coeffs_MAP, cov_coeffs)` analytically (cheap — no new optimization, just draws from the already-computed Laplace covariance) at each lam, push through `exp(basis @ coeffs)`, combine across the AIC-weighted lam grid the same way the moments are combined — gives `b_p16`/`b_p84`/`b_median` without new fits.
- Add template-weight uncertainty: `w` is NNLS-constrained (non-Gaussian); follow the prototype's own documented simplification (weights held fixed at MAP for the covariance calculation) and state this limitation plainly, matching this project's own "verify before building" and honest-limitations culture — don't invent an ad hoc weight-uncertainty scheme without testing it first.
- Add continuum-coefficient handling when `cfg.continuum_knots` is set (depends on Phase 2 landing — extend the Hessian to the joint `(LOSVD coeffs, continuum coeffs)` block).
- Match `bootstrap_losvd`'s output-dict shape closely enough that notebooks can swap between them with minimal code change (same field names where the concept is shared: `point0`, `sherm_std`, `b_p16`, `b_p84`, `b_median`, etc.).
- Keep a **separate, even-cheaper "quick Laplace" mode**: Laplace at the single winning lam only (no AIC-weighting across the grid) — this is what gets exposed in notebooks as the fast sanity-check option per your request.

**1b. Validate at production scale, not n=18.**
Extend `dev_notes/validation_suite/coverage_sweep.py` (currently hardcoded to `bootstrap_losvd` at `run_one_coverage_case`, line ~62) to accept a pluggable uncertainty method. Re-run the full 9-cell `(sigma, S/N)` grid with a real number of independent truths/cell (not 2) for both bootstrap and the new Laplace engine, so the h4-coverage comparison is a trustworthy per-cell rate, not a pooled n=18 estimate. This is the acceptance gate before promoting it to default.

**1c. Full-Bayes reference mode (NumPyro/NUTS).**
Add `numpyro` as an optional dependency (`pyproject.toml` extras, matching the existing `eso` extras pattern). Build a NUTS sampler over the same model (spline coefficients + order-N curvature prior, ideally template weights and continuum too) reusing the existing JAX-jitted objective directly as the potential function — no new forward-model code needed. Use it for: (i) a one-time simulation-based-calibration cross-check of the Laplace approximation's own coverage on a handful of representative cases (this is what actually earns the "robust" label, not just faster-than-bootstrap), (ii) an opt-in slow mode end users can reach for.

**1d. Decide the new default.**
Once 1b passes, `bootstrap_losvd`'s role in notebooks/docs shifts to "alternative/cross-check"; the new Laplace engine becomes what gets called by default in the revamped notebooks (Phase 5).

**Files**: new `src/kinextract/laplace_uncertainty.py` (or a clearly-separated section of `uncertainty.py`), `dev_notes/validation_suite/coverage_sweep.py`, `pyproject.toml`, `tests/test_uncertainty.py` (new tests mirroring the existing bootstrap test patterns).

## Phase 2 — Continuum full integration (Main Task 1.2)

**2a. Confirm the vertical-offset root cause before fixing anything (verify-before-building).**
Working hypothesis, grounded in code reading, not yet tested directly: notebook 02's continuum demo injects a **half-period** sine (`1 + 0.15*sin(pi*frac)`, mean ≈1.0955 over the window — a real nonzero "DC" component), whereas `tests/test_continuum.py`'s passing tests use a **full-period** sine (mean exactly 1.0). `ridge_c` only *softly* penalizes the continuum's own constant/DC coefficient; the template weights `w` carry **zero** penalty on their overall scale (`multi_template.py`'s own documented `(w,b)->(cw,b/c)` invariance). So a true continuum with a nonzero mean has a cheaper path available: push the mean level into `w`'s free overall scale rather than into `C`'s ridge-penalized DC mode — leaving `C`'s recovered *shape* right but its *level* low, exactly the symptom observed. Build a small, targeted test isolating this (a half-sine, non-zero-mean `true_C` in the same single/small-template setting the existing passing tests use, with/without `ridge_c`) to confirm or refute before touching production code.

**2b. If confirmed, fix structurally, not by tuning.**
Anchor `C`'s overall scale by construction (e.g., after each continuum-block solve, rescale `C` to its own mean = 1 and reciprocally rescale `w` — exact, since `M(lambda)*C(lambda)` is invariant under this — forcing all free overall-scale into `w`, where the model already expects it to live). Not a new tunable knob; a fix to an existing, previously-unexamined degeneracy.

**2c. Promote the validated default.**
Per `dev_notes/continuum/FINDINGS.md`'s multi-seed sweep, `penalty_order=3` with fixed `lam` in 4096-32768 beats the current silent order-2/lam=2.0 fallback on both sigma and h4 bias, at every S/N tested. Make this the automatic default whenever `continuum_knots` is set and `lam_selection` is left at `'aic_local'`, replacing today's fallback in `multi_template.py`/`config.py`.

**2d. Re-validate at a broader grid after the offset fix.**
The original 288-fit sweep script (`multiseed_sweep.py`) no longer exists on disk — only its `FINDINGS.md` prose survives. Rebuild it (same design: injected continuum distortion × S/N × seeds), extend beyond the single sigma=90 case tested before, and keep the script this time.

**2e. Honest error propagation.**
Bootstrap already correctly re-solves continuum per replicate and folds `C0` into the noise-injection center (confirmed correct, no bug there) — but discards each replicate's own `C` (no `C_boot`/`C_std` reported). Neither engine currently reports continuum-coefficient uncertainty. Add it to both, as part of Phase 1's Hessian extension (2a/1a are coupled — the Laplace engine's continuum support and this item are the same piece of work).

**Files**: `src/kinextract/continuum.py`, `src/kinextract/multi_template.py`, `src/kinextract/config.py` (fallback logic), `tests/test_continuum.py`, new `dev_notes/continuum/multiseed_sweep.py`.

## Phase 3 — Noise-correlation integration (Main Task 1.3)

Goal: reduce the manual boilerplate `07_noise_correlation.ipynb` currently requires (measure `rho` via `estimate_correlation_length`, then hand-call `whiten(..., rho=...)` before every fit) into something the real-data pipeline can do with one call, ready for the revamped notebook 05 and consistent with notebook 08's existing `MEASURED_RHO` usage. Deliberately **not** auto-detecting `rho` inside `fit_losvd` itself — `noise_model.py`'s own design keeps this a caller-side decision, and that's still the right call. Add a convenience function that measures + whitens in one step, and confirm both uncertainty engines (bootstrap, and the new Laplace engine from Phase 1) correctly consume pre-whitened `L`/`A_w`/`y_w` — should already be structurally true for both since they operate on whatever's passed in, but verify explicitly rather than assume.

**Files**: `src/kinextract/noise_model.py` (new convenience function), consumed by the Phase 5 notebook 05 rewrite and cross-checked against `08`'s existing usage.

## Phase 4 — h3 calibration table (background, runs alongside Phases 1-3)

`calibrate_bias.run_calibration_sweep` already supports `h3_true` (added, unused — see `dev_notes/calibration/CALIBRATION_TRUST_STRATEGY.md`'s 2026-08-01 entry). Launch a real h3 table build using the same per-axis-checkpointed pattern proven necessary for the h4wing/meff tables (`dev_notes/calibration/build_meff_incremental.py`'s own approach — hours of compute, must survive interruption). Start this early as a background job; it doesn't block or get blocked by Phases 1-3.

**Files**: new `dev_notes/calibration/build_h3_table_incremental.py` (mirroring `build_meff_incremental.py`), `bias_correction.py` (wire the new table in once built, matching `h4wing`'s existing pattern).

## Phase 5 — Notebook revamp (Main Task 2)

Cross-cutting requirements for every notebook touched: no development-history/"we fixed X"/"legacy" narration; honest limitations sections written for end-users; final LOSVD plots present the **median bootstrap (or new Laplace-engine median) curve** as the central estimate with its 16th/84th-percentile band, never the raw point estimate (needs Phase 0's `b_median` field, or the Laplace engine's equivalent once Phase 1 lands).

- **01 (foundations rewrite)** — independent, can start anytime. Keep the current notebook's sections 1-6 (setup, template recipe, mock construction, single `fit_losvd` call, moments read off the truth curve, mixture recovery) as the spine; lead with the spectral model equation and template-mixture explanation. Move everything from the current sections 7-10 (shape-residual diagnostic, `penalty_order` deep-dive, bootstrap/uncertainty/bias-correction) to notebook 02 — that's real, substantial content (10 of 22 current cells), not a light trim.
- **03 (repurpose to template-mismatch demo)** — independent. No existing reusable content (checked directly — the old `06_template_mismatch_diagnostic.ipynb` this package's docstrings still reference doesn't exist anymore). Build fresh: fit a mock built from one stellar population using a template library that's missing/wrong for it, show the resulting LOSVD bias directly (curve comparison, not just moments), leading naturally into notebook 04.
- **04 (template calibration: realign + rename)** — independent of Phases 1-3, but should land before 08's final polish. Rebuild its binning to numerically match `08`'s actual production scheme exactly (`08`'s hardcoded `REFERENCE_R_EDGES_ARCSEC` × `build_reflected_sin_theta_edges(N_THETA_PER_QUADRANT=5)` × `pa_deg`, not `04`'s current `log_radial_edges × uniform_theta_edges(1)` with no PA) so the calibrated template subset is actually the right one for `08` to consume. Match `08`'s `plot_bin_map` wedge-based plotting style (explicit grid-line boundaries, consistent colormap) instead of `04`'s current discrete-`imshow` style. Drop "dartboard" language in prose (per the locked-in decision). Fix the save path (Phase 0).
- **06 (STIS extension notebook)** — needs Phase 0's vendored STIS data. Structure as 01→02→(real fit) progression specifically for STIS: basic mock with STIS instrument parameters, a more complex mock with continuum, then the real `bin0105sp` fit — teaching by direct analogy to notebooks 01/02/04's own progression, explicitly framed as "how to adapt this to your own instrument."
- **02 (realistic mock + regularization + uncertainty + continuum)** — **blocked on Phase 1 and Phase 2 landing.** Absorb 01's old sections 7-10 as the starting spine; explain regularization and the *new* default uncertainty engine (with bootstrap positioned as the cross-check alternative); explore which regimes show real bias and why (this project has extensive, already-quantified answers — low sigma, low S/N, near-Gaussian shrinkage of h4 — reuse those numbers rather than re-deriving); explain continuum-cofitting post-fix, including what the vertical-offset investigation found and how it was resolved.
- **05 (repurpose to noise-correlation)** — **blocked on Phase 3 landing.** Absorb and expand `07_noise_correlation.ipynb`'s existing strong content (real ACF measurement, Bacon et al. 2017 connection, GLS-whitening LOSVD comparison, bootstrap-uncertainty-widening consequence — all already built and working) plus new pieces the current notebook is missing: actual variance-cube plots (not just an ACF bar chart), and a synthetic mock spectrum with deliberately-injected correlated noise so the user can see the true LOSVD recovered with vs. without correlation correction on ground truth, not just real-data before/after.
- **Delete 07** (its content is now inside the rebuilt 05).
- **08 → renumbered 07 (capstone)** — do last, after 04's realignment. Fix the warning spam: root cause confirmed (NSIG=7 is genuinely used; the warning fires from a real interaction between the `min_interior_knots=16` floor binding for narrow-sigma nuclear bins and the adaptive selector's own top-of-grid `lam=32768` candidate) — decide between (a) collecting and printing one combined end-of-run summary instead of letting `warnings.warn` fire ~700 raw times inside the per-bin/per-bootstrap-replicate loop, or (b) revisiting whether `min_interior_knots=16` is really the right floor for this notebook's own narrow bins — investigate both, don't just suppress blindly. Consume the realigned `04` template subset. Reframe as "putting it all together." Remove dev-history commentary. Present median-uncertainty-engine LOSVD, not point estimate, per the cross-cutting requirement above.

## Phase 6 — Final verification

- Full test suite green (`pytest tests/ -q`).
- All notebooks (final 01-07 numbering) re-executed end-to-end, 0 errors — including confirming notebook 07 (ex-08)'s warning count is now a clean, bounded summary, not hundreds of raw prints.
- Cross-reference audit: no notebook points at a deleted/renumbered notebook by stale number (`grep` for old numbers/names across all notebooks and `README.md`).
- `README.md`'s notebook-list section updated to match the final 01-07 structure and what each one now actually teaches.

## Verification approach throughout

Each phase gets its own before/after evidence, matching this project's own established rigor (paired comparisons on identical noise draws, `zlib.crc32`-seeded reproducibility, coverage checked via `coverage_sweep.py`-style simulation calibration, not just point-bias). No phase is marked done on "looks right" — Phase 1 needs the extended `coverage_sweep.py` run before promotion to default; Phase 2 needs the targeted offset test before the structural fix, then the rebuilt multi-seed sweep after; every touched notebook gets a fresh end-to-end execution with 0 errors before being called complete. Nothing gets committed without your explicit go-ahead, per standing project rule.
