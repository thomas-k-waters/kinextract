# Regularization v2: minimally regularized, honest non-parametric LOSVDs across S/N (started 2026-10-31)

**Goal (user, 2026-10-31):** keep the non-parametric LOSVD model; get minimally regularized LOSVDs and excellent
spectral fits across the full S/N range. Work through (1) the error / spectral model, (2) templates, (3) the
regularization itself, testing recovery on synthetic spectra at every step. Recovery must stay at least as good as
now. Nothing is committed or pushed.

**Benchmark:** `bench.py` (see its docstring), results in `results/`, one CSV per method on identical mocks.

**kinextract 2.0 (2026-11-02):** the recipe adopted here is now the package default (`FitConfig()`, `fit_losvd_robust`,
`losvd_error_bands`; checked bit-identical on cached NGC 4751 NFM fits and bands), and the methods these benchmarks
found inferior (AIC `lam` selection, bootstrap, the bias correction, the order-2 penalty, L-BFGS-B, clip cleaning)
were removed. `bench.py` and `bench_dev.py` ran against the 1.1 development API and are kept as the record of what was
run; they no longer import under 2.0.

## Why (NGC 4751 MUSE, 2026-10-31)

- Adaptive AIC λ failed both ways on the real data:
  - Low S/N or central bins: the AIC curve is flat across 1024–131072, so the parsimony rule picks λ_max and the
    LOSVD comes out Gaussian (h4 = 0.000).
  - High-S/N NFM bnn bins: AIC picks λ = 1024 and the LOSVDs are wrong (bnn1203 σ 648, h4 +0.25, V +37 vs about
    −160).
- The reduced χ² before error inflation rises with S/N: NFM 1.8 / 2.9 / 7 / 44 at S/N 25 / 75 / 140 / 250; WFM
  33–170.

## Findings so far

1. **The dominant high-S/N "systematic" is sky residuals in the cubes, not template mismatch.**
   - The fractional residuals of the 12 highest-S/N NFM fits are narrow positive spikes of 10–60 % at the OH
     sky-line positions, across 8300–8470 and 8690–8850 Å. The CaT region (8480–8680) is clean to about 1 %.
   - They pass both `detect_emission_lines` and the error inflation.
   - Masking them brings the reduced χ² of those fits from 18 to 2.6, and to 2.3 in the CaT region alone.
   - The library is saved in `n4751_nfm_sky_residuals_rest.npy`.
2. **The sky mask comes from the MUSE pipeline's sky-line catalog,** not from the residuals.
   - `MU_GSKL_140101A_sky_lines.fits`, blended with the LSF (2.5 Å FWHM). Saved in
     `muse_sky_profile_rest_z0.006985.npy`.
   - At a blended flux ≥ 40, it masks 34 % of the window (30 % of 8480–8680).
   - Lower thresholds catch more spikes but mask 40–60 %.
   - The WFM reference mask (`test_rawcube_ringgrid_skymask.pkl`) misses lines at 8305, 8324, 8393, 8434, 8489,
     8730 and 8788 Å (rest).
3. **How a non-parametric LOSVD goes wrong: an additive pedestal.**
   - bnn1203 at λ = 1024 is the correct core (V −160, σ 240) plus a faint flat pedestal at +600…+2000 km/s.
   - The moments (σ 501, V +22) come from the pedestal.
   - The χ² gain is spread over the whole spectrum, not the CaT lines. The pedestal acts as an additive continuum
     (a faint smoothed copy of the spectrum) that the model otherwise lacks.
   - The NFM's ±3000 km/s grid, the same for every bin, leaves room for it.
   - This is the artificial-wing failure Mehrgan et al. (2023) warn about.
4. **Prototype additive continuum (3 featureless positive pseudo-templates):**
   - It takes 5–22 % of the light and lowers the reduced χ² by 20–30 %.
   - It repairs bnn1103 and improves bnn0904.
   - It moves σ in good bins (bnn1201 231 → 178, bin0105 554 → 531).
   - bnn1203 stays broken.
   - **Not yet validated on mocks**, where the possible σ bias has to be measured.
5. **The spmc_in error transform understates model-bin errors by about 2×.**
   - Median E_spmc / E_post is 0.41–0.52 on the first mocks. E_post is the scatter of the Laplace posterior samples
     integrated over each 300 km/s model bin.
   - The writer (`transvd_bounds_laplace`, `model_bin_kms` = 133.3) and model.x's linear summation give errors about
     half the true binned uncertainty, so the χ² the models see is about 4× too large from this alone.
   - The fix belongs in the writer: give each model bin its posterior binned error. Still to be done.
6. **The posterior itself is too narrow under contamination.** χ²_post (recovered vs true, with the binned
   posterior scatter) is 15–58 on contaminated S/N 300 mocks, against about 1 for honest errors.

## Mock realism check

The contaminated mocks (`contam=skyadd`) reproduce the real behavior:

| | Reduced χ² before inflation |
|---|---|
| S/N 300 mock, no mask | 108 |
| S/N 300 mock, sky mask | 13 |
| Real high-S/N NFM bins, no mask | 50–370 |
| Real high-S/N NFM bins, sky mask | 13–21 |

## Baseline (screen grid, 480 mocks per method, finished 2026-10-31)

Methods: `lam=fixed4096|aic` × `sky=none|mask40`. Medians over the mocks; coverage is the fraction of model bins
inside ±1σ (0.68 for honest errors).

| contam | method | shape_rms | χ²_spmc | χ²_post | cover_post | dσ/σ | failed |
|---|---|---|---|---|---|---|---|
| clean | aic, none | 0.0096 | 14.6 | 3.6 | 0.43 | −2.8 % | 0 |
| clean | fixed4096, none | 0.0103 | 13.8 | 2.7 | 0.45 | −2.6 % | 11 (both contam.) |
| clean | aic, mask40 | 0.0109 | 13.7 | 3.3 | 0.42 | −2.8 % | 0 |
| clean | fixed4096, mask40 | 0.0105 | 12.9 | 2.6 | 0.46 | −2.7 % | 13 (both contam.) |
| skyadd | aic, none | 0.0150 | 16.3 | 4.6 | 0.37 | −5.5 % | |
| skyadd | fixed4096, none | 0.0152 | 15.7 | 3.5 | 0.39 | −5.0 % | |
| skyadd | aic, mask40 | 0.0166 | 19.3 | 5.7 | 0.35 | −2.4 % | |
| skyadd | fixed4096, mask40 | 0.0166 | 18.4 | 5.0 | 0.35 | −2.5 % | |

7. **The sky mask does not improve recovery on mocks.**
   - Masking 34 % of the window loses more information than the sky residuals cost: shape_rms gets worse
     (0.015 → 0.017 under contamination) and χ²_post rises.
   - It only removes the σ bias from the contamination (−5 % → −2.5 %).
   - Masking is not the way forward. The sky has to be modeled in the noise instead (`noise=sky`).
8. **AIC and fixed λ = 4096 recover shapes equally well;** AIC's errors are slightly worse.
   - AIC picks λ_max (131072) for most mocks below S/N 300 and 2048 at S/N 300.
   - On mocks this costs little shape accuracy. On the real data it gave Gaussian LOSVDs (finding in "Why").
9. **The errors are too small everywhere, even on clean mocks.**
   - Coverage is 0.25–0.55 instead of 0.68.
   - χ²_post is 2–9 and grows with S/N (clean S/N 300: 7–9).
   - The Laplace errors at fixed λ leave out the regularization bias, which matters most at high S/N.
   - χ²_spmc (13–50) adds the writer's 2× understatement (finding 5).
10. **Wing-shaped LOSVDs are biased low in σ by 8–17 %, with h4 low by about 0.02–0.03,** for every method.
    χ²_post reaches 23–106 under contamination. The order-3 penalty (Gaussian null space) suppresses real
    wings. This is the target for step 3.
11. **Fixed λ = 4096 fails on 2–3 % of mocks.**
    - All are σ = 550 on the ±3000 km/s grid. The fit grows a broad pedestal to the grid edge.
    - Every Laplace draw is then rejected by the edge filter. The degenerate fallback gives zero-width errors, and
      the spmc writer refuses them.
    - AIC (λ_max) never fails.
    - These are logged in `results/*.failed.txt` and count against the method. `bench.py` now writes these
      failure logs itself.

## Step 1 screen (480 mocks per method, finished 2026-11-01)

Each step-1 method is compared with the baseline `lam=fixed4096` and `lam=aic` (no sky mask). Medians over the mocks.

| contam | method | shape_rms | χ²_post | cover_post | χ²_tot | cover_tot | failed |
|---|---|---|---|---|---|---|---|
| clean | fixed4096 (baseline) | 0.0103 | 2.74 | 0.45 | – | – | 11 (both contam.) |
| clean | fixed4096, noise=sky, terr | 0.0103 | 2.62 | 0.46 | 2.10 | 0.51 | 6 |
| clean | + add=feat3 | 0.0101 | 2.66 | 0.48 | 1.93 | 0.53 | 3 |
| clean | + grid=ring | 0.0103 | 2.71 | 0.47 | 2.16 | 0.51 | 4 |
| clean | aic, noise=sky, terr | 0.0097 | 3.42 | 0.44 | 2.52 | 0.50 | 0 |
| skyadd | fixed4096 (baseline) | **0.0152** | 3.51 | 0.39 | – | – | |
| skyadd | fixed4096, noise=sky, terr | 0.0180 | 1.24 | 0.58 | 1.22 | 0.59 | |
| skyadd | + add=feat3 | 0.0187 | 1.26 | 0.56 | 1.18 | 0.58 | |
| skyadd | + grid=ring | 0.0180 | 1.27 | 0.59 | 1.21 | 0.60 | |
| skyadd | aic, noise=sky, terr | 0.0171 | 1.50 | 0.56 | 1.39 | 0.57 | |

At skyadd S/N 300: shape 0.0110 → 0.0145, χ²_post 7.7 → 1.03, coverage 0.25 → 0.61.

16. **The noise model makes the errors honest under contamination, but over-smooths.**
    - The contaminated mocks get χ²_post ≈ 1 and coverage 0.6.
    - The shape gets worse, by 18 % overall and 32 % at S/N 300. That breaks the "no worse recovery" rule.
    - Cause: the mismatch term f·m is 0.5×, 1.8× and 4.9× the photon error at S/N 40 / 120 / 300 (f = 1.3–1.6 %;
      κ ≈ 0.12–0.14 of the median). The errors grow nearly uniformly, which at a fixed λ is a penalty that is
      inflation² ≈ 25× stronger.
    - AIC under the noise model goes to λ_max as well.
    - On clean mocks f ≈ 0 except 0.17 % at S/N 300, so it changes nothing there.
17. **The additive term (`feat3`) and the per-ring grid make no clear difference.**
    - feat3: slightly better errors on clean mocks. It worsens shape and σ under contamination (dσ −6 % at S/N
      300).
    - grid=ring: neutral, and it does not stop the σ = 550 pedestal failures (4 vs 6).
    - Neither is adopted.
18. **Fix: decouple the point estimate from the error bands (`noise=skysplit`, in `bench_dev.py`).**
    - The LOSVD is the plain-error fit, i.e. the baseline point estimate.
    - The bands are the `noise=sky` posterior at the same λ, moved onto that fit (a shift in log b).
    - Two other variants failed on single mocks:
      - `noise=skyrel` (λ divided by the inflation²): λ goes to 23–92, which helps some shapes and hurts others.
      - Splitting at λ/inflation²: the bands come out 3–4× too wide (χ² 0.1).
    - On single mocks skysplit keeps the baseline shape (0.0108 vs 0.0105, 0.0164 vs 0.0163) with χ²_mb about 0.5
      and coverage 0.83–0.86.
    - **Screen result (480 mocks, `results/dev_*.csv`).** χ²_mb / cover_mb are the spmc_in errors with the
      model-bin fix plus the template term, i.e. what model.x sees.

| contam | method | shape_rms | χ²_mb | cover_mb | χ²_post | cover_post | failed |
|---|---|---|---|---|---|---|---|
| clean | fixed4096 (baseline) | 0.0103 | 2.14 | 0.52 | 2.74 | 0.45 | 11 (both contam.) |
| clean | fixed4096, skysplit, terr | 0.0102 | 1.61 | 0.58 | 2.59 | 0.47 | 6 |
| clean | aic, skysplit, terr | 0.0095 | 1.65 | 0.58 | 3.24 | 0.46 | 0 |
| skyadd | fixed4096 (baseline) | 0.0152 | 2.55 | 0.48 | 3.51 | 0.39 | |
| skyadd | fixed4096, skysplit, terr | 0.0153 | 0.50 | 0.78 | 0.75 | 0.68 | |
| skyadd | aic, skysplit, terr | 0.0150 | 0.52 | 0.77 | 0.90 | 0.67 | |

    - **Shape is preserved exactly.** The paired split/base shape_rms ratio is 1.001 (median). 4 % of mocks are
      >10 % worse and 3 % are >10 % better. Every (contam, S/N) cell is within ±1 %.
    - **The errors are honest to conservative under contamination.** At skyadd S/N 300, χ²_mb goes 5.8 → 0.29 and
      cover_mb 0.32 → 0.86.
    - **Remaining gap: clean, high S/N.** At clean S/N 300, χ²_mb is 7.2 → 2.9 and cover_mb 0.31 → 0.42. The noise
      model sees nothing to add there (f ≈ 0). What is left is regularization bias at fixed λ, which is the target
      for step 3, together with the σ = 550 pedestal failures of fixed λ.
    - **AIC + skysplit** is about equal on errors and 3–8 % better in shape on clean mocks. It never fails, but its
      scatter per mock is larger (16 % of mocks >10 % worse, 18 % >10 % better).
    - **Adopted for step 1:** `noise=skysplit` plus `terr=1`, with the model-bin error fix in the spmc writer.

## Real data: NGC 4751 NFM, 110 bins (`real_check.py`, 2026-11-01)

These are the simplified extraction's own spectra, masks, templates and grids. Three versions are compared:
- `aic`: the cached adaptive-λ run, with λ and template spreads in its bands;
- `fixed`: λ = 4096, retried at 2λ / 4λ if a fit reaches the grid edge (4 bins at ×2, none needed ×4), plain
  Laplace bands;
- `split`: the fixed fits with noise=skysplit bands, the template spread (emiles_old, xsl_old, xsl_all) and the
  model-bin errors.

Output is in `notebook_data/muse/simple_20261031/regv2_check/`.

| | χ²/model bin, 0–0.55″ | 0.55–1.6″ | 1.6–5″ | all (51 pairs) | pair shape rms, all |
|---|---|---|---|---|---|
| aic | 8.94 | 11.26 | 22.60 | 11.70 | 0.0136 |
| fixed (plain bands) | 10.33 | 13.00 | 36.73 | 15.06 | 0.0144 |
| split | 0.93 | 0.62 | 0.41 | **0.74** | (same fits as fixed) |

22. **Fixed λ repairs the bnn bins that AIC broke.** Each pair below is bnn | its mirror bin:

    | Pair | AIC: V, σ | Fixed λ: V, σ | Mirror bin: V, σ |
    |---|---|---|---|
    | bnn1203 | +37, 648 | −147, 232 | +233, 230 |
    | bnn1103 | −39, 453 | −153, 242 | +219, 262 |
    | bnn1201 | −114, 422 | −243, 234 | +291, 208 |

    - With the fixed fits, the bnn σ matches the mirror bin. The AIC h3/h4 of 0.2–0.34 were pedestals.
    - V_sys from the fixed pairs is 51.34 km/s.
23. **The split errors bring the real pair χ² to 0.74 per model bin, from 11.7 with AIC.**
    - The pair test cannot see template mismatch, which both sides share, so < 1 is expected with the template term
      in.
    - The components are being separated (`split_notmpl`, `split_nomb`).
24. **Remaining real-data problems at λ = 4096: broad bins.**
    - In the center (< 0.55″) AIC's pairs are slightly more consistent: shape rms 0.0136 vs 0.0147.
    - bnn0402 goes wrong at 4096: σ 589, V −28, h3 +0.12, against bin0402 at 422 and AIC at 131072 giving −160 / 444.
    - bnn0904 is off in both versions (σ 406–478 vs 319).
    - The oracle's σ trend predicts this. On the mock oracle, λ = 4096 × max(1, (σ_fit/350)⁶), capped at 131072, is
      never worse in any cell (worst 1.001) and improves σ = 550 by 4–6 %.
    - Applied to the real bins with σ_fit > 350 (`--sig6`); results in finding 25.
25. **The σ-scaled λ (`sig6`) fixes the broad bins** (58 of 110 NFM bins have σ_fit > 350).

    | Bin | Fixed λ: V, σ | sig6: V, σ (λ) | Mirror bin / AIC |
    |---|---|---|---|
    | bnn0402 | −28, 589 | −169, 450 (λ 92826) | bin0402: +245, 416 |
    | bnn0904 | +12, 406 | −44, 322 (λ 10024) | bin0904: +169, 319 |
    | bin0105 | V 116, σ 654 | V 74, σ 616 (λ 131072) | AIC: 74, 619 |

    The pair shape rms is the best of the three versions in every range:

    | Version | 0–0.55″ | 0.55–1.6″ | 1.6–5″ | all |
    |---|---|---|---|---|
    | sig6 | 0.0130 | 0.0108 | 0.0177 | 0.0126 |
    | aic | 0.0136 | 0.0105 | 0.0216 | 0.0136 |
    | fixed | 0.0147 | 0.0111 | 0.0180 | 0.0144 |

    - sig6 is slightly behind aic at 0.55–1.6″ (0.0108 vs 0.0105).
    - The pair χ² with split errors is 0.74.
    - **Adopted rule:** λ = 4096 × max(1, (σ_fit / 350)⁶), capped at 131072, with σ_fit from the λ = 4096 fit, then
      the 2λ / 4λ edge retry.
26. **What sets the pair χ² of 0.74 (NFM):**

    | Error model | χ² per model bin |
    |---|---|
    | plain bands | 15.1 |
    | noise-model bands | 4.95 |
    | + model-bin fix | 0.75 |
    | + template spread | 0.74 |

    - The template term is invisible to the pair test, as expected: both sides share the population.
    - The model-bin fix is the largest single factor (4.95 → 0.75).
    - 0.75 means the errors are about 15 % conservative in their random part. That agrees with the mocks
      (contaminated cover_mb 0.78, χ²_mb 0.5).
    - Conservative is the safe side, so the errors are left unscaled. If needed, a per-range downscale could be
      calibrated from the pairs.

## Real data: NGC 4751 WFM, 76 bins (rings 11–18, 1.6–37″)

| | χ²/model bin, 1.6–4.5″ | 4.5–13.5″ | 13.5–50″ | all (38 pairs) | pair shape rms, all |
|---|---|---|---|---|---|
| aic | 10.93 | 14.89 | 7.13 | 11.30 | 0.0233 |
| fixed (plain) | 15.62 | 54.21 | 24.16 | 33.70 | 0.0234 |
| split | 0.55 | 0.75 | 0.41 | **0.59** | (same fits) |

27. **The WFM confirms the recipe, with stronger systematics.**
    - No WFM bin has σ_fit > 350, so sig6 changes nothing. The retry ladder: 6 bins at ×2, then 1 at ×4.
    - Fixed and AIC shapes are equal (pair rms 0.0234 vs 0.0233). Fixed is better at 1.6–4.5″ (0.0122 vs
      0.0141).
    - The noise model inflates the WFM errors much more than the NFM's: median 4.6 (16–84: 2.9–9.6), f = 1.3 %,
      up to about 10× in the outer rings (χ²_prelim 170–800).
    - The model-bin fix again does most of the work for the pair χ²: 2.76 → 0.60.
    - V_sys from the WFM fixed pairs is 42.94 km/s; from the NFM it is 51.34.
28. **The outer WFM rings (25–37″) have a data problem that predates this work and is the same with every
    method.**
    - **bin1805:** V 657, σ 28. Broken in AIC too; it should be rejected in REVIEW.
    - **σ asymmetry:** bin180x has σ 101–152 against 49–83 for bnn180x.
    - **Near the instrument limit:** σ of 30–80 km/s is close to MUSE's instrumental σ at the CaT, about 40 km/s.
    - These bins carry 4–10× noise-model inflation, so they are strongly down-weighted. The pair χ² of 0.41 says
      the errors cover the asymmetry.

29. **V_sys: the NFM pairs give 51.3 km/s, the WFM pairs 42.9. There is no wavelength-scale offset between the
    cubes.**
    - **Same bins, both cubes** (11 pairs at 2.3–5.2″, the shared dartboard): the median NFM − WFM midpoint
      difference is −0.1 km/s. The mean is +11 ± 10, pulled up by one outlier pair at +102. With the same template sets
      in both, the median is +0.3 to +7.
    - **The global difference comes from which radii each cube samples.** The pair midpoint velocity varies with radius:

      | Radius | NFM | WFM |
      |---|---|---|
      | < 0.55″ | 61 (23 of 51 pairs) | – |
      | 0.55–1.6″ | 47 | – |
      | 1.6–5″ | 43 | 37 |
      | 5–13.5″ | 52 (3 pairs) | 39 |
      | 13.5–25″ | – | 49 |

    - Beyond 0.55″: NFM 46.4, WFM 42.9, combined 44.4 ± 1.9 (66 pairs).
    - **The central NFM excess** (+15–20 km/s inside 0.55″) is not wavelength: it is the same cube as the outer NFM
      bins. The likely cause is a binning-center offset of about 0.03–0.05″ along the major axis, since a midpoint picks
      up the gradient × the offset. Dust or a real offset are the alternatives. Not yet tested.
    - Within one cube the template set moves V_sys by up to 8 km/s (NFM emiles_old 56.8 vs xsl 48.7).
    - **Decision (user, 2026-11-01):** one V_sys for both cubes, the NFM notebook's value (all NFM pairs).

30. **NFM central "wings": an NFM-cube systematic on the bnn side, not PSF scattering and not the galaxy**
    (2026-11-01, λ = 4096 fits).
    - **bnn0403 is not bad data.** With the LOSVD held fixed and kinextract's own (w, C) blocks, the λ = 4096 free fit
      (core at the mirror's V ≈ −200, σ ≈ 450, plus a red wing to +2500 km/s) has χ² 216.8.

      | LOSVD | χ² |
      |---|---|
      | λ = 4096 free fit | 216.8 |
      | bin0403 mirrored | 229.0 |
      | adopted (λ cap) | 237.1 |

      `sigma_scaled_lam` read the wing-inflated σ (700) and smoothed core and wing into one broad Gaussian. The NFM
      notebook now overrides bnn0403 to λ = 4096.
    - **The wings are one-sided.** Rings 2–9: mean h3 is +0.064 on the bnn side and −0.023 on the bin side. Light beyond
      ±1500 km/s: 3.1 % on the bnn side, 1.2 % on the bin side. PSF scattering would give equal and opposite values.
    - **Physical mixtures fail.** Mirror core + f × (nucleus LOSVD) or + f × (the opposite-side LOSVD) never reaches the
      free fit (bnn0403: +8.9 / +10.0). For bnn0501, 0802, 0904, 1102 and 0601 the mirror stays Δχ² 21–65 worse with
      any mixture. bnn0403's free fit needs 22 % of its light beyond ±1500 km/s, more than any stellar component here
      provides.
    - **Absent in the WFM.** For the same sky bins (2.3–5.2″), mean bnn h3 is +0.036 in the NFM and +0.006 in the WFM;
      bnn1102 is +0.283 in the NFM and +0.008 in the WFM. The WFM PSF is several times broader, so scattering would be
      stronger there, and anything in the galaxy would appear in both cubes.
    - **Suspect: processing of the NFM cube.** Its primary header is empty, so the file was rewritten. The code's
      comments mention ZAP + `repair_corrupted_spaxels`. ZAP can remove galaxy absorption where the source mask
      under-covers the galaxy. No raw ADP copy or ZAP script was found on disk.
    - **The test:** re-extract a few winged bins (bnn0403, 0501, 0802, 1102) from the raw ESO cube.
31. **The wings are in the ESO product itself; two processing suspects are ruled out** (raw cube downloaded from
    the ESO archive into `data/N4751/MUSE/NFM/raw/`; `raw_vs_processed.py`).
    - **Not ZAP or spaxel repair.** The cube on disk is the ESO MUSE-DEEP product with its primary header stripped:
      zero spaxels differ in 8300–8850 Å, and the 16 winged and mirror bins refit identically.
    - **Not galaxy light in the pipeline's sky model.** `muse_scipost` used `skymethod = subtract-model` with
      `skymodel_fraction = 0.1` (the darkest 10 % of the 7.5″ field, here at 2.9–5.4″, which is galaxy). But those
      spaxels still show galaxy CaT at the expected depth (CaII 8542: 0.102 against 0.084 in the WFM at the same radii).
      The NFM/WFM surface-brightness ratio is 0.90–1.13 over 0.5–4.5″, with no deficit growing outward.
    - **Remaining suspect: instrument layout.** The cube was taken at `INS DROT POSANG = 0` (12 exposures, Jan + Jun/Jul
      2022, `genericoffsetLGS`). MUSE stacks its 24 IFU sub-fields as horizontal strips, and the galaxy's minor axis
      (84.7°) is 5° from horizontal. So the bin (north) and bnn (south) halves of the NFM field are observed by
      different sets of IFUs and spectrographs.
    - That would explain one-sided wings that appear in the NFM but not in the WFM, whose ~2.5″ strips put the inner
      2″ in one or two IFUs. It cannot be tested further with the combined cube; it needs the 12 raw exposures.

32. **The NFM "wings" come from defective spaxels** (2026-11-02; scan of every kept spaxel inside 0.35″ at λ = 4096,
    `notebook_data/muse/simple_20261031/raw_check/nfm_spaxel_scan.pkl`, `figures/nfm_spaxel_scan.png`).
    - **bnn0403:** one of its 3 spaxels, (y 165, x 177), gives V +236, σ 837, h3 +0.27, with 28 % of the light beyond
      1500 km/s. The other two spaxels give −181 / 452 and −170 / 496, like its mirror.
    - **Across the center:** 11 spaxels are off their 8 neighbors' median V by more than 300 km/s, almost all
      red-skewed. They cluster in a stripe at cube column x = 169 (y 160–178; median |dV| 112 km/s against 40–60 for
      the other columns), plus isolated spaxels.
    - **Nothing in the cube flags them.** The product has no DQ extension. Its reduction used no `BADPIX_TABLE` (it is
      absent from the calibration list), so bad detector columns were never flagged. Every spaxel has the same 214–215
      NaN pixels (the AO laser notch), and the flagged spaxels' STAT/flux is normal (0.94 against 0.99).
    - **Likely cause:** unflagged bad CCD columns. All 12 exposures are at POSANG 0 with small offsets, so they land at
      fixed cube positions. This supersedes the IFU-half reading in finding 31.
    - **Fix:** `simple_extraction.NFM_BAD_SPAXELS` masks the x = 169 stripe and 6 isolated spaxels. The bins
      containing them are re-extracted (`reextract_changed`) and refitted.
    - Also adopted: the template-set refits are now optional (`TEMPLATE_ERROR = False`). That term was ~5–7 km/s in σ
      against 16–28 km/s for the Laplace bands.

## Step 3 probes (single mocks, `bench_dev.py`, 2026-10-31)

19. **AIC λ is bimodal, and the edge pedestal sets it, not the smoothness of the data** (aic + skysplit screen,
    2026-11-01).
    - S/N 40: λ_max (131072) on 78 of 80 mocks.
    - S/N 300: the grid floor (1024) on 33–38 of 80, and λ_max on 7–18.
    - On a clean S/N 300 spline mock with the grid extended to 128, the AIC criterion falls monotonically all the
      way to λ = 128: Δ = 256 between λ_max and 128. The data want less smoothing than any λ allowed.
    - But every λ ≤ 8192 is rejected as "invalid" (edge amplitude > 1 %), so AIC lands on 16384.
    - On the standard grid the same mock took λ = 1024 as valid. The warm start runs up from the lowest λ and carries
      the low-λ pedestal into neighboring λ, so the result depends on the path.
    - The pedestal (finding 3) is what forces the heavy regularization. With no additive contamination it comes
      from template mismatch.
    - The AIC-weighted λ spread (`lerr=1`) is about 0, because the curve is sharp. The λ/4 bias proxy (`lerr=2`) is
      also 0 here, since every lower λ was invalid. Neither helps until the pedestal is dealt with.
20. **The velocity-support prior (`ref=supp5`) does not help.**
    - It is an amplitude taper (not a roughness taper) beyond 5σ₀ of the plain fit, via the column-scaling trick,
      with the λ grid down to 128.
    - Single mocks (clean S/N 300 spline, GH and wings; skyadd S/N 40 Gaussian): shape 0.0095 → 0.0134 for the
      spline, and about the same or slightly worse for the rest.
    - The earlier roughness tapers in `PORT_STATUS.md` could not work in principle: a flat pedestal has no roughness.
      This amplitude prior did not work in practice.
    - Next: an oracle test. The shape error is scored for every λ of the scan (`shape_l<λ>` columns,
      `results/dev_oracle_aic_low.csv`) to see whether any λ beats the adopted one before trying more λ machinery.
21. **λ selection is a weak lever** (oracle screen: 480 mocks, `lam=aic;noise=skysplit;lgrid=low`, the shape error
    of every λ from 128 to 131072, point fits).
    - **The median shape error is almost flat in λ.** Within each (contam, S/N) cell it varies by ≤ 5 % over
      1024–131072. The best single fixed λ is 4096 in 4 of 6 cells; the other two are 8192 and 131072, and both
      are within 1 % of 4096.
    - **AIC loses to fixed 4096 at S/N 300,** by 7 % on clean mocks and 4 % with skyadd. It is equal elsewhere.
    - **The per-mock oracle,** the best valid λ in hindsight, beats fixed 4096 by 2 / 8 / 13 % (clean, S/N
      40 / 120 / 300) and 5 / 7 / 8 % (skyadd).
    - **Most of that gain is noise-realization luck.** The best rule that uses only what the fit measures is
      λ = 8192 (σ_fit / 350)^7. It recovers just 1–3.5 % (median ratio 0.989); with the true σ it is 0.994. So the
      oracle's 2–13 % is out of reach of any selector.
    - The σ dependence itself is physical. The order-3 penalty on log b scales as σ⁻⁶, so broad LOSVDs want a
      larger λ: the oracle median is about 500–5800 at σ = 250 and 16k–130k at σ = 550. But it is only worth about
      1 %.
    - **Conclusion:** fixed λ = 4096, with the retry ladder (2λ, 4λ) for the σ = 550 edge-pedestal failures, is as
      good as any λ selector on these mocks. What is left of step 3 is the errors (clean high-S/N coverage), not λ.

12. **The wing σ bias is not the penalty and not template mismatch.**
    - Wings at σ = 250 (5–15 % of the light at 2–3σ) come out σ −13 % even with the mismatch-free `emiles`
      truth.
    - It is the same at λ = 1024 as at 4096.
    - At S/N 3000 it drops to −6 % with the 4-knot continuum. Without the continuum it overshoots, because the
      mock's ±3 % continuum then goes unmodelled.
    - So at S/N ≲ 300 a faint broad wing is mostly below what the data constrain, and the multiplicative continuum
      absorbs part of it.
13. **A Gauss-Hermite reference penalty (`ref=gh2`) does not help.**
    - Shrinking towards r(v) × exp(quadratic), with r the GH fit to the previous solution, leaves the wing bias
      unchanged on 2 of 3 wing mocks.
    - It makes Gauss-shaped mocks worse (shape_rms 0.0018 → 0.0034).
    - The posterior collapses where r is floored (χ²_post up to 10⁴).
    - Dropped.
14. **λ marginalization (`post=pool`) barely changes the errors.**
    - It pools Laplace draws over the λ grid with AIC weights.
    - Wing mocks: χ²_post 1.3×10⁴ → 980 and 1.1×10⁴ → 6,700. Coverage stays 0–0.1.
    - Gaussian mock: χ²_post 0.2 → 0.12.
    - Double S/N 300 skyadd: 6.1 → 5.4. Spline S/N 300: 27 → 19, with coverage 0.43 → 0.29.
    - Shapes are 2–4 % better. It costs 3–4× the fit time.
    - The failure is structural instead: the posterior lives in log b, so its absolute scatter is about zero
      wherever the median has about zero light, and an undetected 1 % wing has no error bar.
    - The spmc writer's peak floor is what protects model.x from this.
15. **Fix to the new `model_bins` option in `legacy_io`.**
    - It rescaled faint wing bins (0.1–1 % of the light) to their near-zero posterior scatter, which undid the peak
      floor.
    - It now only widens: scale factor ≥ 1.
    - The test checks that.
    - `bench_dev.py` now scores `chi2_mb` / `cover_mb`: the spmc_in errors with this fix (plus the template term
      when `terr=1`), which is what model.x would see.

## Runs

- [done] Baseline screen: see above.
- [done] Step-1 screen: `lam=fixed4096;noise=sky;terr=1`, `lam=aic;noise=sky;terr=1`,
  `lam=fixed4096;noise=sky;terr=1;add=feat3`, `lam=fixed4096;noise=sky;terr=1;grid=ring` (see above).
- [done] `bench_dev.py` screen, written to `results/dev_*.csv`, with the model.x error metric (`chi2_mb`):
  `lam=fixed4096`, `lam=fixed4096;noise=skysplit;terr=1`, `lam=aic;noise=skysplit;terr=1`.
