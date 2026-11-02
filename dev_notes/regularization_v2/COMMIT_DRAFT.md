# kinextract 2.0.0: draft commit message and release notes (not committed)

## Files

Stage every modification and deletion (`git add -u` also stages the removed modules, tests and calibration tables),
plus these untracked paths:

    git add -u
    git add tests/test_lam_rule.py tests/test_newton_optimizer.py \
            examples/data/muse/ngc4751_bin_r.out examples/data/muse/ngc4751_bin_v.out \
            examples/data/muse/ngc4751_nfm_full_galaxy_results.npz examples/data/muse/ngc4751_nfm_kma_mask.txt \
            examples/data/muse/ngc4751_radial_kinematics.npz examples/data/muse/ngc4751_representative_bins.npz \
            examples/data/muse/ngc4751_template_subset.npz \
            dev_notes/regularization_v2/FINDINGS.md dev_notes/regularization_v2/*.py \
            dev_notes/regularization_v2/*.npy dev_notes/regularization_v2/results

Leave out:
- this file;
- `examples/notebooks/scratch_ngc1399_investigation.ipynb` and `examples/data/muse/ngc1399_*` (scratch work);
- `dev_notes/regularization_v2/__pycache__/`;
- the downloaded cube and sky-line catalog in `examples/notebooks/muse_download/` (git-ignored).

Then:

    git commit -F <message file>
    git tag -a v2.0.0 -m "kinextract 2.0.0"

## Commit message

```
Release 2.0.0: the NGC 4751 recipe as the defaults; remove inferior methods

The defaults are now the recipe validated on 480 NGC 4751-like MUSE mocks
and the real NGC 4751 bins (dev_notes/regularization_v2/FINDINGS.md): a
bare FitConfig() with fit_losvd_robust reproduces that extraction bit for
bit, and losvd_error_bands its errors.

Breaking: new defaults
- FitConfig.lam 2.0 -> 4096 with FitConfig.lam_rule="sigma_scaled"
  (replaces lam_selection): refit at sigma_scaled_lam(sigma) =
  lam*(sigma/350)^6 (cap 131072) for broad LOSVDs, and at 2x then a further
  4x lam while the LOSVD is still at the grid edge. multi_template.
  with_lam_rule implements it for fit_losvd and fit_losvd_robust.
- FitConfig.continuum_knots None -> 4 (fits need wave_fit).
- fit_losvd_robust: smooth_window_A 3, inflation_cap 100,
  min_inflation_to_apply 1.1, CAII_PROTECTED_WINDOWS protected.
- Third-difference penalty only (PENALTY_ORDER = 3); exact-Hessian
  trust-region Newton is the only solver.
- assign_legacy_dartboard_bins folds with -pa_deg, as plotbin3.f (it was
  mirrored in x); legacy-dartboard bins change.

Removed (inferior in the validation):
- Adaptive lam selection: aic_regularization (select_lambda_aic[_local],
  estimate_m_eff[_local], AICSelectionResult), FitConfig.lam_selection,
  lam_grid, self_normalize_lam_grid. edge_amplitude_frac moves to losvd.
- AIC-weighted Laplace; quick_laplace_uncertainty and
  laplace_continuum_uncertainty fold into laplace_losvd_uncertainty (one
  lam, continuum-aware).
- Bootstrap: uncertainty module (bootstrap_losvd[_parallel],
  calibrate_noise_scale), FitConfig.losvd_single_start.
- Bias correction: bias_correction module and calibration tables,
  shape_aware_taper_correction, and their compute_losvd_moments options.
- FitConfig.optimizer (L-BFGS-B), FitConfig.penalty_order,
  fit_losvd_robust(cleaning_method="clip"), fit_losvd_basis(
  losvd_warm_start=).

New
- losvd_error_bands / losvd_error_bands_parallel: the real-data error
  bands in one call (noise model, Laplace at the fit's lam and rho without
  clipped pixels, re-centered, moment errors).
- Noise model: noise_model_errors, fit_noise_model, sky_line_profile,
  load_muse_sky_lines, recenter_samples.
- sigma_scaled_lam, with_lam_rule, PENALTY_ORDER, FitConfig.lam_rule,
  FitConfig.edge_retry_factors.
- legacy_io.write_spmc_in_laplace / transvd_bounds_laplace /
  spmc_side_inverts, with b_samples= and model_bins= (model-bin errors);
  muse_cube.trim_cube_wavelength_fits.
- Optional extra .[examples] (cmap) for the example moment maps.

Fixed
- Parallel workers run single-threaded (BLAS/OpenMP/XLA pinned in the
  spawn environment).
- laplace_losvd_uncertainty_parallel(..., clipped_mask=) centers the
  posterior on the robust fit; _fit_losvd_clip's clipped_mask is exactly
  the excluded pixel set.
- Gaussian start added to the LOSVD multi-start pool.

Docs and examples
- api.rst, quickstart, configuration, README and PARAMETER_GUIDE describe
  the new defaults; removed methods are gone from them.
- All seven example notebooks use the defaults with no comparisons with
  removed methods, and are re-run. 04 and 07 explain the half-cube dust
  mask; 07 is a complete NGC 4751 NFM extraction with regenerated shipped
  results; 04's regenerated template subset and radial profile feed it.

Validation: dev_notes/regularization_v2. Defaults checked bit-identical
against the cached NGC 4751 NFM fits and bands. Tests: 275 passed.
```

## Release notes (GitHub, v2.0.0)

### kinextract 2.0.0: the NGC 4751 recipe is now the default

kinextract 2.0 makes the extraction recipe validated on NGC 4751 MUSE data the package default. It also removes the
methods that validation found inferior. Real MUSE spectra carry residuals that per-pixel noise does not describe,
mainly sky-line residuals and template mismatch: on NGC 4751 the reduced χ² before any error inflation grew from about
2 at S/N 25 to more than 40 at S/N 250. Under those conditions adaptive regularization and plain Laplace bands both
misbehave. The recipe was validated on 480 NGC 4751-like MUSE mocks and the real NGC 4751 bins
(`dev_notes/regularization_v2/FINDINGS.md`). A bare `FitConfig()` with `fit_losvd_robust` now reproduces that
extraction bit for bit, and `losvd_error_bands` its error bands.

**Upgrading from 1.x**

This is a breaking release. Fitted values change if you relied on the old defaults, and several functions are gone.
- **Configs:** drop `lam_selection`, `lam_grid`, `penalty_order`, `optimizer` and `losvd_single_start` from your
  `FitConfig`s. `FitConfig(lam=..., lam_rule="fixed")` pins a `lam`.
- **Fits:** pass `wave_fit` to every fit, now that the continuum is co-fitted by default. Use
  `FitConfig(continuum_knots=None)` for spectra exactly normalized to the templates.
- **Uncertainties:** replace `bootstrap_losvd` and `quick_laplace_uncertainty` with `laplace_losvd_uncertainty`, or
  with `losvd_error_bands` for real data.
- **Bias correction:** remove the calls; there is no replacement. It never passed its validation.
- **Cleaning:** replace `fit_losvd_robust(cleaning_method="clip")` with the default inflate cleaning.

**New defaults**
- **Regularization:** `lam = 4096` on the third-difference penalty, with `FitConfig.lam_rule="sigma_scaled"`:
  - LOSVDs broader than 350 km/s are refitted at `sigma_scaled_lam(sigma) = 4096 (σ/350)⁶`, capped at 131072;
  - fits still at the velocity-grid edge are refitted at 2× and then a further 4× their `lam`.

  On the mocks no `lam` selector beat this rule by more than a few per cent.
- **Continuum:** a smooth multiplicative continuum (4 knots) is co-fitted.
- **Cleaning:** `fit_losvd_robust` inflates errors where the residuals call for it, smoothed over 3 Å and capped at
  100×. The Ca II triplet cores are protected from clipping.
- **Solver:** exact-Hessian trust-region Newton only, 20-90× faster than the L-BFGS-B solver it replaces and reaching a
  lower objective.
- **`assign_legacy_dartboard_bins` now folds correctly** (`-pa_deg`, as `plotbin3.f`); it was mirrored in x.

**New**
- **`losvd_error_bands` / `losvd_error_bands_parallel`:** error bands for real IFU data in one call:
  1. a per-pixel variance `err² + (κ·s)² + (f·model)²`, with `s` the LSF-blended sky-line spectrum, is fitted to a
     preliminary fit's residuals;
  2. the Laplace posterior is computed with those errors at the fit's `lam` and `rho`, without its clipped pixels;
  3. the draws are moved onto the fit;
  4. the moment errors are taken from the draws.

  On the mocks, the shapes were unchanged and the coverage of the model-binned LOSVD under sky contamination rose from
  0.48 to 0.78. On the real NGC 4751 bins, the mirrored bin/bnn pair χ² per model bin went from 11.7 to 0.74 (NFM) and
  from 11.3 to 0.59 (WFM).
- **`with_lam_rule`, `sigma_scaled_lam`, `PENALTY_ORDER`**, and the building blocks of the bands: `noise_model_errors`,
  `fit_noise_model`, `sky_line_profile`, `load_muse_sky_lines` and `recenter_samples`.
- **Inputs for SCO Schwarzschild models:** `write_spmc_in_laplace` / `transvd_bounds_laplace` write `spmc_in` files
  from the bands. With `b_samples=` and `model_bins=`, each model velocity bin carries at least its binned posterior
  scatter; `model.x`'s linear error sum understated it by about 2×.
- `trim_cube_wavelength_fits`, and an optional `.[examples]` extra for the example notebooks' colormaps.

**Removed**

| Removed | Why | Use instead |
|---|---|---|
| Adaptive `lam` selection (`aic_regularization`, `lam_selection`, `lam_grid`) | Failed on real data in both directions (Gaussian LOSVDs at low S/N; a broad pedestal at high S/N, e.g. σ 648 against 230 km/s for the mirror bin); no better on mocks | `lam_rule="sigma_scaled"` (default) |
| `lam`-marginalized Laplace, `quick_laplace_uncertainty`, `laplace_continuum_uncertainty` | Marginalizing over `lam` barely changed the errors | `laplace_losvd_uncertainty` (one `lam`, continuum-aware) |
| Bootstrap (`bootstrap_losvd[_parallel]`, `calibrate_noise_scale`) | About 20× slower than Laplace with worse coverage everywhere (h4: 0-53 % against 50-83 %) | `laplace_losvd_uncertainty`, `losvd_error_bands` |
| Bias correction (`bias_correction`, `correct_moments`, ...), `shape_aware_taper_correction` | Never passed its validation gates; worse at σ ≳ 300 km/s | none |
| L-BFGS-B (`optimizer`), the order-2 penalty (`penalty_order`) | Slower and stops short of the minimum; 3-4× larger h4 bias | Newton, order 3 (only options) |
| `cleaning_method="clip"` | Worse calibrated on real data (113 NGC 4751 bins: whitened RMS 1.41 against 0.96) | the default inflate cleaning |

Kept as cheaper checks: `cleaning_method=None`, `laplace_losvd_uncertainty` with the pipeline errors,
`fit_gauss_hermite`, `extract_aperture_spectrum`, `detect_emission_from_residual` and `compute_direct_moments`. Kept as
the slower, exact reference: `nuts_uncertainty`.

**Fixes**
- **Parallel workers now really run single-threaded.** Four workers used to start about 50 threads each.
- **`laplace_losvd_uncertainty_parallel(..., clipped_mask=)`** centers the posterior on the robust fit. Without it,
  some NGC 4751 WFM bins reported std=0 or moved by hundreds of km/s.
- **A Gaussian start was added to the LOSVD multi-start pool.**

**Docs and examples**
- The API reference, quickstart, configuration page, README and parameter guide describe the new defaults.
- **All seven example notebooks** use the defaults, with no comparisons with removed methods, and are re-run.
- **Example 07** is a complete extraction of the NGC 4751 MUSE NFM cube: the dust mask (explained), defective spaxels,
  a legacy dartboard, `losvd_error_bands_parallel`, hand tuning, maps, profiles, and a mirror-symmetry check of the
  errors. It runs from shipped results by default.
- **Examples 04, 05 and 07** use NGC 4751 and explain why the dusty half of its cube is masked.

**Known limits**
- On clean, high-S/N mocks built from different templates than the fit, model-bin coverage is still about 0.42. The
  remaining error is template-mismatch bias in the point estimate, which no band width fixes.
- Under the new defaults the direct moments and a joint Gauss-Hermite fit agree to about 0.5 km/s on the validation
  grid. The direct moments carry a 1-2 % σ taper bias.
