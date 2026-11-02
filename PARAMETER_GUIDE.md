# Parameter guide: what to set for your situation

`FitConfig.describe()` documents every field in isolation. This file is
the companion question: given a physical regime (true sigma, S/N, one
template vs. a mixture, known vs. unknown contamination), which knobs
actually matter, and what do they do to each other?

## The knobs, in one place

The defaults reproduce the NGC 4751 MUSE extraction validated in `dev_notes/regularization_v2/FINDINGS.md`; most
fits need only `vmin`/`vmax` (about V +- 5-7 sigma) and, for whitening, `rho`.

| Parameter | Default | Job | Interacts with |
|---|---|---|---|
| `lam` | 4096 | Strength of the roughness penalty on the third differences of the log-LOSVD spline coefficients. The third-difference penalty's null space is any *Gaussian* log-density, so heavy smoothing shrinks toward the best-fitting Gaussian instead of toward exponential tails (a second-difference penalty inflated wing mass and sigma at low S/N and gave 3-4x larger h4 bias with the continuum; it has been removed). With `lam_rule='sigma_scaled'`, the starting value. | `vmin`/`vmax`/`min_interior_knots` (a narrow grid can make `lam` behave far stronger than nominal -- `FitConfig` warns); requires the package's JAX-float64 default (on automatically) |
| `lam_rule` | `"sigma_scaled"` | Refits LOSVDs broader than 350 km/s at `sigma_scaled_lam(sigma) = lam (sigma/350)^6` (capped at 131072): the penalty scales as sigma^-6 for a given relative shape. `"fixed"`: one fit at exactly `lam`. On 480 NGC 4751-like mocks no `lam` selector beat the rule by more than a few per cent; on the real NGC 4751 data adaptive selection failed in both directions (Gaussian LOSVDs at low S/N, broad pedestals at high S/N). | `edge_retry_factors`, `max_edge_amplitude_frac` |
| `max_edge_amplitude_frac` / `edge_retry_factors` | 0.01 / (2, 4) | A LOSVD still holding more than this fraction of its peak within 5 points of either grid edge (`edge_amplitude_frac`) has a divergent edge pedestal -- a roughness penalty alone can't tell a compact LOSVD from one on a broad, low pedestal, and the pedestal inflates every moment. The `lam` rule refits such fits at 2x, then a further 4x, their `lam`; the Laplace sampler rejects draws above it. | `vmin`/`vmax`: a grid too narrow for the LOSVD looks like a pedestal too -- widen the grid first |
| `continuum_knots` / `lam_c` / `ridge_c` | 4 / 2.0 / 0.1 | The co-fitted multiplicative continuum (fits need `wave_fit`). `None` switches it off, for spectra exactly continuum-normalized against the templates (e.g. mocks built from the fit's own templates). | Its absolute level is degenerate with the template weights' overall scale -- read only its shape |
| `rho` | 0.0 | AR(1) pixel-noise correlation used for whitening. `fit_losvd_robust(..., measure_rho=True)` measures it per bin (reliable from about 20 spaxels per bin on MUSE; use a fixed value below that). | `combine_bin_spectrum_robust`'s `spatial_rho` is the separate spaxel-to-spaxel correlation |
| `smooth_window_A` / `inflation_cap` / `min_inflation_to_apply` (`fit_losvd_robust`) | 3.0 / 100 / 1.1 | Per-spectrum, per-wavelength error inflation from a preliminary fit's own residual: smoothed over a few MUSE sky-line widths, floored at 1, capped so a badly contaminated window is down-weighted, not zeroed; skipped when nothing anywhere exceeds 1.1. Real full-field validation (113 MUSE bins, NGC 4751): median whitened RMS 1.41 -> 0.96, bins with more than 10 % of pixels clipped 22 -> 0, kinematics essentially unchanged. | A loose cleanup clip runs underneath (`prelim_clip_sigma`/`final_clip_sigma`) -- do not tighten it to do the inflation's job |
| `excluded_wavelength_windows` (`fit_losvd_robust`) | `CAII_PROTECTED_WINDOWS` | Pixels never clipped, however large their residual: the CaII triplet cores, where a slightly mis-modeled absorption core otherwise looks like a cosmic ray to the cleanup clip. Inert for spectra that do not cover them. | Pass your own windows for other strong lines, or `None` |
| `mask_wavelength_windows` (`windows=`) | -- | Fixed-wavelength exclusion for *known* bad-pixel ranges. | Complementary to the robust fit's inflation: use this when the bad wavelengths are known and fixed across every spectrum |
| `n_templates` (shape of `A_stack`) | -- | Not a `FitConfig` field -- the shape of the template stack you build and pass in. Every fit function accepts any `n_templates >= 1`. | More templates costs identifiability: the same LOSVD now competes against more free template weights -- condition the library first (`04_template_calibration.ipynb`) |

## Recommended starting points by regime

- **Most spectra: the defaults.** `FitConfig(vmin=..., vmax=...)` and
  `fit_losvd_robust(A_stack, flux, err, cfg, wave_fit=wave)`, then
  `losvd_error_bands` for the uncertainties.
- **Real IFU spectra whose residuals are systematics-dominated** (reduced
  chi2 before inflation well above 1 and growing with S/N -- sky-line
  residuals, template mismatch):
  - Take the error bands from **`losvd_error_bands`**, which fits the
    sky-line + template-mismatch noise model and runs Laplace with it, not
    from the pipeline errors. Do not *fit* with those errors: at a fixed
    `lam`, near-uniformly larger errors act as a stronger penalty and
    over-smooth.
  - If you write SCO `spmc_in` files from these bands, pass `b_samples=`
    and `model_bins=` to `write_spmc_in_laplace`.
  - Details: `dev_notes/regularization_v2/FINDINGS.md`.
- **Known, fixed bad-pixel wavelengths** (e.g. sky-subtraction residuals
  recurring at the same wavelengths across every spectrum in a survey):
  `mask_wavelength_windows`, applied before fitting.
- **A quick look**: `fit_losvd_robust(..., cleaning_method=None)` (one
  plain fit) or `laplace_losvd_uncertainty` with the pipeline errors.
- **A recovered LOSVD looks visually skewed/asymmetric in a way that
  doesn't match physical expectations**: check `edge_amplitude_frac(b)`
  first -- if it's ~0, the shape is noise-driven or a genuine physical
  signal near its own Cramer-Rao detection floor (see
  `examples/notebooks/02_realistic_mock_fit.ipynb`'s Section 5), not a
  divergent edge pedestal.

## Combining spaxels into one bin spectrum

`extract_aperture_spectrum` (flux sum, independent-noise error combine)
is fine for a synthetic mock or a quick look, but **for real MUSE data,
prefer `combine_bin_spectrum_robust`**: a biweight (not plain) mean for
the flux -- resistant to a single bad/cosmic-ray-contaminated spaxel
dominating the combined spectrum the way a plain mean would let it.

Error is the formal, independence-assuming inverse-variance combine
alone -- correct for spatially combining different spaxels, though (like
any such formula) it can under-report the true error if per-spaxel noise
is positively correlated (e.g. from resampling/drizzling during pipeline
reduction). That correlation is `noise_model.whiten`'s own AR(1) `rho`
job at fit time (on the wavelength axis), not this function's -- see
`05_noise_correlation.ipynb` for measuring a real target's own `rho`.
An earlier version of this function also added an "empirical" error term
(biweight scale of the raw per-spaxel flux, ported from the legacy SCO
Fortran framework's own repeat-exposure combiner, `imcmb.f`) meant to
catch that same correlation -- removed (`dev_notes/real_data_sigma/
FINDINGS.md`, 2026-08-09 entry): `imcmb.f` combines *repeat exposures of
the same sky position* (where flux scatter genuinely is noise), not
*spatially different spaxels* (where flux differences are real
surface-brightness structure) -- applied to the latter, that term ran
120-470x the actual formal noise on real data, i.e. it was measuring the
bin's own spatial gradient, not noise.

Two things worth knowing about the current, formal-only error:

- **Flux convention differs from `extract_aperture_spectrum`.** This
  function returns a *mean* (one spaxel's worth of surface brightness),
  not that function's own *sum* (total flux across the aperture) -- not
  interchangeable without rescaling by the spaxel count. This doesn't
  change the recovered LOSVD (the fit is scale-invariant -- template
  weights absorb a uniform rescale for free), but it does change any
  S/N number computed downstream from the combined flux/error.
- **The MUSE `STAT`-extension variance-underestimate correction is
  opt-in**, via `variance_correction_factor` (Bacon et al. 2017's own
  recommended value, also available as the package constant
  `BACON2017_VARIANCE_CORRECTION_FACTOR`, `1/0.6**2 ~= 2.78`) -- left at
  the default `1.0` (no correction) since the true factor is
  instrument/pipeline-version specific and should be measured for your
  own cube, not assumed. Do not apply it to a synthetic mock (its own
  noise model has no such resampling artifact to correct for).

Related: `assign_legacy_dartboard_bins`'s `fold_major_axis` (default
`True`, matching the legacy Fortran scheme exactly) controls whether a
bin's north/south halves (across the assumed kinematic major axis) get
merged or kept separate -- set `False` to preserve real north/south
asymmetry (e.g. from a warp, or a genuine failure of point-symmetry) the
folded scheme discards by construction.
