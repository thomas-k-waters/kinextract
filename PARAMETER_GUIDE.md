# Parameter guide: what to set for your situation

`FitConfig.describe()` documents every field in isolation. This file is
the companion question: given a physical regime (true sigma, S/N, one
template vs. a mixture, known vs. unknown contamination), which knobs
actually matter, and what do they do to each other?

## The knobs, in one place

| Parameter | Default | Job | Interacts with |
|---|---|---|---|
| `penalty_order` | 3 (as of 2026-07; was 2) | Which difference order the roughness penalty acts on. Order 3's null space is any *Gaussian* log-density, so heavy smoothing shrinks toward the best-fitting Gaussian instead of toward exponential tails -- this eliminates the wing-inflation mechanism `lam_wing` exists to patch, and won the 480-case paired validation over order 2 (23/32 cells better shape residual, overall \|sigma bias\| 5.6 vs 15.5 km/s). Set 2 to reproduce pre-2026-07 behavior exactly; order 2 also keeps a modest shape edge for genuinely non-Gaussian LOSVDs (\|h3\|/h4 ~ 0.06+) at sigma 50-140, S/N~100, where order 3 shrinks real h4 by ~0.03-0.05 (a regime where h4 is near the measurement noise floor regardless -- see `dev_notes/REGULARIZATION_INVESTIGATION_LOG.md`). | `lam`/`lam_grid` (order-3 lam values sit orders of magnitude higher); activates two selection safeguards in `fit_losvd` (delta-AIC parsimony tiebreak, sub-knot-sigma candidate guard); requires the package's JAX-float64 default (on automatically) |
| `lam` | 2.0 (fallback value; see `lam_selection`; calibrated for `penalty_order=2` only) | Roughness penalty on the LOSVD spline. Always on. | `lam_wing` (see below); `vmin`/`vmax`/`min_interior_knots` (a narrow grid can make `lam` behave far stronger than calibrated -- `FitConfig` warns if this happens); `penalty_order` (order 3 needs far larger values -- `FitConfig` warns on the stale combination) |
| `lam_selection` | `"aic_local"` | Whether `lam` is picked per fit (`"aic_local"`, minimizing chi2+2·m_eff -- measured 20-57% lower sigma bias than fixed `lam=2.0`) or held at the fixed `lam` value above (`"fixed"`). | Falls back to `"fixed"` with a warning if `continuum_knots` is set (not supported together yet); `bootstrap_losvd` selects once at the point estimate and holds that `lam` fixed for every replicate, and auto-switches its default bias-correction table to the `m_eff`-axis one when this is `"aic_local"` |
| `lam_wing` | 0.0 (off) | Absolute strength of an amplitude penalty on the LOSVD's far wings, ramped on beyond `taper_sfac` (see below). Corrects a real failure mode the *order-2* penalty cannot: it has no amplitude term, so nothing stops noise-driven wing inflation, and more order-2 smoothing makes it worse, not better. **Largely obsolete under the order-3 default** -- the wing-inflation mechanism it patches is eliminated by construction there (validated wing-mass ratios ~1.0 across the full grid without it); it remains relevant only for `penalty_order=2` fits. | Needs its own tuning per sigma, per S/N, and per template realism (see table below) -- not portable across any of the three |
| `taper_sfac` / `taper_power` | 1.8 / 4.0 | Where the wing ramp turns on (in units of the LOSVD's own recovered width, re-seeded each of `n_sigl0_iter` rounds -- not a fixed velocity offset) and how steeply it grows beyond that point. | Only relevant if `lam_wing > 0` |
| `shape_aware_correction` | False | Post-fit correction for a moment-estimator windowing bias. | Helps below ~sigma=200 km/s, measurably *hurts* above it -- `compute_losvd_moments` warns if active and the recovered sigma lands above that boundary |
| `noise_scale` (via `calibrate_noise_scale`) | 1.0 (off) | Inflates bootstrap-replicate noise injection when the assumed per-pixel noise is wrong. | Only affects bootstrap replicates, never the point estimate or `lam`/`lam_wing`'s own behavior -- deliberately decoupled |
| `clip_sigma` / `max_clip_fraction` (`fit_losvd_clean`) | 3.0 / 0.1 | Iterative sigma-clipping for contamination whose wavelength isn't known in advance. | Do not reach for this to fix sigma inflation -- that is `lam_wing`'s job. Outlier rejection can *compound* a model bias, since the pixels disagreeing most with an already-biased fit are often exactly the ones correctly arguing against it |
| `mask_wavelength_windows` (`windows=`) | -- | Fixed-wavelength exclusion for *known* bad-pixel ranges (e.g. sky-subtraction residuals at specific wavelengths). | Complementary to `fit_losvd_clean`, not a substitute -- use this when the bad wavelengths are known and fixed across every spectrum, cleaning when they aren't |
| `n_templates` (shape of `A_stack`) | -- | Not a `FitConfig` field -- the shape of the template stack you build and pass in. `fit_losvd`/`bootstrap_losvd`/`fit_losvd_clean` all accept any `n_templates >= 1`. | More templates costs identifiability: the same fixed-size LOSVD histogram now has to compete against more free template weights, which measurably worsens sigma/V bias at some sigma ranges relative to a single known template (see `lam_wing` table below) |

## `lam_wing`: what value, for what regime

`lam_wing=0.0` is the safe default for the S/N~100, sigma>=140,
single-known-template regime this package's own validation grid centers
on. Outside that regime, tune it -- a value that helps at one sigma/S-N/
template-realism combination can measurably hurt at another. A concrete,
validated example (single real stellar template, S/N=100):

| true sigma (km/s) | tuned `lam_wing` | sigma bias, untuned | sigma bias, tuned |
|---|---|---|---|
| 20 | 3e6 | +98% | -2.4% |
| 30 | 1e6 | +84% | +1.0% |
| 50 | 3e5 | +55% | +3.9% |
| 80 | 3e5 | +65% | +1.4% |
| 140 | 3e5 | +7% | +2.7% |
| 220 | 3e6 | +11% | -4.8% |
| 300 | 3e6 | +12% | -0.4% |
| 400 | 3e6 | +11% | +3.4% |

Two things this table does NOT tell you, confirmed directly rather than
assumed:

- **These values do not transfer cleanly to a multi-template mixture.**
  Applying this same single-template-tuned table to a genuine 32-template
  mixture helps substantially at low-to-mid sigma (e.g. sigma=50: 17%
  bias vs. ~35% untuned) but is measurably too strong at sigma>=220
  (worse than leaving `lam_wing` off entirely at some points in that
  range). If you're fitting a real stellar population mixture, re-tune
  for that library specifically -- don't reuse a single-template table.
- **S/N matters too, independent of sigma.** The table above is at
  S/N=100 only; a value tuned for one S/N can measurably degrade shape
  residual at another (see `FitConfig.describe("lam_wing")`).

Tuning approach that produced the table above: a small grid search (10
seeds/cell is enough to see the trend) over `lam_wing` at fixed
sigma/S-N, picking the value that minimizes `|median(recovered sigma) -
true sigma|`. Cheap to redo for a new template library or S/N regime --
this is not a one-time calibration to memorize, it's a recipe to re-run.

> **2026-07 note**: the `lam_wing` table and tuning advice below predate
> the `penalty_order=3` default, which eliminates the wing-inflation
> mechanism `lam_wing` corrects. Under current defaults, start with
> `lam_wing=0.0` at every sigma/S-N; this section remains the reference
> for `penalty_order=2` fits.

## Recommended starting points by regime

- **Current defaults (`penalty_order=3`, `lam_selection="aic_local"`,
  `lam_wing=0.0`) are the validated starting point everywhere** the
  480-case 2026-07 grid covers (sigma 20-300, S/N 30-100, single
  template and mixtures). The bullets below are the pre-2026-07
  guidance, still applicable when running `penalty_order=2`:
- **Single known template, S/N>=60, sigma in [140, 400]**: order-2
  defaults (`lam_selection="aic_local"`, `lam_wing=0.0`) are fine.
- **Single known template, sigma < 140 or S/N < 60**: with
  `penalty_order=2`, turn on `lam_wing` and tune it for your specific
  sigma/S-N (see table above as a starting guess, not a substitute for
  checking your own case).
- **Real stellar template mixture (unknown population)**: expect worse
  bias than the single-template numbers above at every sigma -- `lam_wing`
  still helps, but needs its own tuning pass for your specific template
  library, not a borrowed table.
- **Known, fixed bad-pixel wavelengths** (e.g. sky-subtraction residuals
  recurring at the same wavelengths across every spectrum in a survey):
  `mask_wavelength_windows`, applied before fitting.
- **Suspected localized contamination at an unknown, per-spectrum-varying
  location** (a cosmic ray, a bad detector column): `fit_losvd_clean`.
  Do not reach for this to fix a fit that just looks biased -- check
  whether the bias is instead the known low-S/N wing-inflation mechanism
  first (`lam_wing`'s job), since outlier rejection can make that specific
  kind of bias worse, not better.
- **Suspected wrong per-pixel noise level**: `calibrate_noise_scale`,
  validated only for a uniformly-wrong noise level -- do not apply to
  data whose residuals show a handful of localized outlier pixels rather
  than a uniform mis-specification (check your own residuals first).
