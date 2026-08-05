"""Parametric-bootstrap uncertainty quantification for a recovered LOSVD.

Fits once for the point estimate, then refits `n_boot` times against that
point estimate's own model spectrum plus fresh noise (the design matrix is
already noise-whitened, so i.i.d. `N(0, 1)` per pixel is the assumed noise
model). Each replicate is a full fit, multi-start included -- not a
cheaper linearized shortcut -- so a replicate that lands in a different
local optimum shows up as spread in the resulting distribution rather than
being invisible to a curvature-based (Laplace/Hessian) estimate.

Works for any `n_templates >= 1` (single template included) via
`kinextract.multi_template.fit_losvd`.

Every bootstrap replicate curve is retained, not just summary statistics
-- `b_boot` (and its covariance, `b_boot_cov`) supports uncertainty bands
on the LOSVD itself, bin-bin covariance, or a confidence interval on any
quantity computed from `b(v)` later, without rerunning the bootstrap. See
`save_bootstrap`/`load_bootstrap` to persist a result to disk.

**The full ensemble (`b_boot`/`w_boot`/`vherm_boot`/`sherm_boot`/`h3_boot`/
`h4_boot`), not `point0_corrected` plus a symmetric `*_std`, is the
recommended product for anything propagating uncertainty *forward* into
another analysis** (e.g. an external orbit-superposition/dynamical model).
A single point estimate and a Gaussian-shaped error bar necessarily throw
away whatever the ensemble's actual shape says -- skew, multimodality, or
correlation between, say, sigma and h4 within the same replicate set --
and a downstream analysis that only ever sees a mean and a std is
implicitly assuming that thrown-away structure doesn't matter. It might;
this package has no way to know that from here. Prefer feeding the
ensemble itself (or its covariance, `b_boot_cov`/`np.cov(np.column_stack(
[vherm_boot, sherm_boot, h3_boot, h4_boot]), rowvar=False)`) forward where
the consuming analysis can use it, and treat `point0_corrected`/
`sherm_std_corrected`/etc. as a convenient summary for reporting a single
number, not as the actual, most information-preserving output of this
function.

`calibrate_noise_scale`/`noise_scale` corrects for a mis-specified assumed
per-pixel noise level: it changes only how much synthetic scatter gets
injected into each bootstrap replicate, never the point estimate or the
fit's own regularization. Validated for a uniformly mis-specified noise
level; not yet validated for data whose residuals have a handful of
localized outlier pixels (e.g. sky-subtraction residuals) rather than a
uniformly-wrong noise level -- inspect your own data's residuals before
trusting it in that regime.

**Regularization-bias correction (`apply_bias_correction`, default
False as of 2026-07 -- see below for why this flipped).** The raw
parametric-bootstrap median is NOT, in general, a good estimate of the
true LOSVD: refitting on the point estimate's own exact, noiseless model
recovers a measurably different (more smoothed/shrunk) `b`, because the
roughness penalty pulls every replicate toward the same regularized
optimum regardless of the specific noise draw, while the point estimate
itself benefits from whatever its own particular noise realization
happened to support. This is real regularization/shrinkage bias (Pinkney
et al. 2003's own documented behavior for this MPL lineage -- see
`kinextract.bias_correction`'s module docstring), not a bug, and
naive bootstrap statistics don't reveal or correct it on their own.

Optionally, `bootstrap_losvd` can apply `bias_correction.correct_moments`
to the point estimate (using an S/N estimated from the point estimate's
own best-fit model, `bias_correction.estimate_snr_whitened`) and inflate
`sherm_std`/`h4_std` in quadrature with the calibration's own bias-scatter
term -- Pinkney's own two-part prescription (correct the point estimate,
inflate the interval to account for the correction's own uncertainty).
**All existing keys (`point0`, `sherm_std`, `h4_std`, `b_p16`, `b_p84`,
`b_boot`, ...) are left exactly as before** regardless of this setting --
the correction is exposed entirely through additively-named keys
(`point0_corrected`, `sherm_std_corrected`, `h4_std_corrected`,
`b_p16_corrected`, `b_p84_corrected`, `b_median_corrected`,
`snr_estimated`); nothing about existing output changes shape or meaning.

**The default is off** because a real user applying this correction to
real data has no ground truth to catch a bad correction the way synthetic
testing can -- see `dev_notes/VALIDATION_CAMPAIGN.md` Stage 14 for the two
overcorrection failure modes that motivated this, and
`kinextract.bias_correction`'s own module docstring for the
calibration-strategy work needed before this could default back on.

Set `apply_bias_correction=True` to opt in anyway (e.g. for research/
comparison purposes) -- the `_corrected` keys are then real corrections,
not just copies of the raw ones; the reverse (leaving it `False`) makes
every `_corrected` key an exact copy of its raw counterpart, so code
reading them either way still works. See `bias_correction`'s own module
docstring for this correction's scope (sigma/h4 only, not V/h3;
calibrated at V=0, single-template, MUSE sampling -- treat results
outside that regime as provisional) and its warn-not-extrapolate behavior
outside the calibrated sigma/S-N grid.

**`FitConfig.penalty_order=3` (the default as of 2026-07, see
`dev_notes/VALIDATION_CAMPAIGN.md` Stage 22): a per-`lam` calibration set
(`PerLamCalibration`) has been built for it (Stage 26, 2026-07-28/29;
redesigned 2026-07-30 -- see below) -- the plain 2-D table has NOT.**
Order 3's adaptively-selected `lam` typically lands far outside the
*plain* table's calibrated regime (often near the top of its own grid),
which is exactly the case `apply_bias_correction`'s table-switch logic
exists for: it auto-selects the per-`lam` set whenever `lam` is far from
the plain table's own reference point (see `_get_default_calibration`/
`get_default_lam_axis_calibration` above), and that set's own lam grid
(`dev_notes/calibration/build_emiles_calibration.py`'s
`LAM_GRID_FOR_MEFF`) matches `FitConfig`'s order-3 default grid exactly.
The severe-overcorrection failure this mismatch used to cause
(`examples/notebooks/01_basic_mock_fit.ipynb` Section 10: an
already-good order-3 point estimate, sigma=50.9 km/s vs. truth 50,
corrected down to ~22 km/s) no longer reproduces on that same mock --
see the regression test `tests/test_bias_correction.py::
test_bootstrap_correction_does_not_severely_overcorrect_the_01_notebook_mock`.

**2026-07-30: the per-`lam` calibration was redesigned, not just
rebuilt.** The original version of this table stored `m_eff` (effective
degrees of freedom) as a continuous third interpolation axis alongside
`(sigma, S/N)`, queried via 3-D `scipy.interpolate.griddata`. Running
`CALIBRATION_TRUST_STRATEGY.md`'s 4-gate acceptance test found this
failed gate 1 (per-cell bias) even against data dense enough to satisfy
the document's own "Fix #1" -- the actual problem was that `m_eff`
depends on both `lam` and the local `(sigma, S/N)` regime, so two
calibration points with similar `m_eff` for unrelated reasons could get
interpolated together. `PerLamCalibration` replaces this with the
originally-proposed design: one 2-D `(sigma, S/N)` table per canonical
`lam` value, interpolated between the two nearest `lam` values at query
time -- see that class's own docstring in `bias_correction.py`. Reuses
the exact same underlying calibration data (no rebuild), just partitions
and queries it differently. Gate 1 has not yet been rerun against this
redesign; treat `_corrected` values under order 3 as no longer
known-broken from the old mismatch, but not yet as fully trusted -- check
`dev_notes/uncertainty/FINDINGS.md` for the latest result before relying
on this for unsupervised use.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from .bias_correction import calibration_has_lam_axis as _calibration_has_lam_axis
from .bias_correction import correct_moments as _correct_moments
from .bias_correction import estimate_snr_whitened as _estimate_snr_whitened
from .bias_correction import _get_default_calibration
from .bias_correction import get_default_lam_axis_calibration as _get_default_lam_axis_calibration
from .config import FitConfig
from .gh_moments import compute_losvd_moments, gauss_hermite_model
from .multi_template import fit_losvd

_NOISE_SCALE_MAX = 3.0
_NOISE_SCALE_MIN = 1.0 / 3.0


def calibrate_noise_scale(A_w: np.ndarray, y_w: np.ndarray, b: np.ndarray) -> float:
    """Estimate how wrong the assumed per-pixel noise level is, from the
    point estimate's own whitened residuals -- for passing to
    `bootstrap_losvd`'s `noise_scale`, **not** for rescaling `A_w`/`y_w`
    themselves before fitting.

    `A_w`/`y_w` are already whitened by whatever noise level the caller
    assumed; if that assumed level is wrong, the whitened residuals
    `y_w - A_w@b` won't have the unit variance the whitening assumed.
    Their actual scale (a MAD-based robust estimate, not `sqrt(chi2_red)`
    -- a plain sum-of-squares statistic is sensitive to localized
    systematic residuals that aren't representative of the general
    per-pixel noise level) directly measures the assumed-vs-true noise
    mismatch, in the same whitened units `bootstrap_losvd` works in.

    Single-pass (call once against one fitted `b`; don't refit-and-
    recalibrate in a loop). Passing the returned scale into
    `bootstrap_losvd(..., noise_scale=...)` only changes how much
    synthetic scatter gets injected into each bootstrap replicate -- it
    does not rescale `A_w`/`y_w`, does not change the point estimate, and
    does not change any replicate's own regularization behavior.

    For a multi-template fit, pass the effective single-template form
    (`np.tensordot(result0['w'], A_stack_w, axes=([0],[0]))`) as `A_w`.

    Clipped to `[1/3, 3]` with a `UserWarning` if the raw estimate falls
    outside that range -- an estimate that extreme more likely indicates a
    bad fit or a masking issue than an honest noise-level error.
    """
    resid_w = np.asarray(y_w, float) - A_w @ np.asarray(b, float)
    mad = np.median(np.abs(resid_w - np.median(resid_w)))
    scale = 1.4826 * float(mad)
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    if scale > _NOISE_SCALE_MAX or scale < _NOISE_SCALE_MIN:
        warnings.warn(
            f"calibrate_noise_scale: raw estimate {scale:.2f} is outside the sane "
            f"[{_NOISE_SCALE_MIN:.2f}, {_NOISE_SCALE_MAX:.2f}] range and has been clipped -- "
            "this large a mismatch more likely indicates a bad fit, a masking issue, or "
            "genuinely non-Gaussian residuals than an honest noise-level error; inspect "
            "before trusting it.",
            UserWarning, stacklevel=2,
        )
        scale = float(np.clip(scale, _NOISE_SCALE_MIN, _NOISE_SCALE_MAX))
    return scale


def bootstrap_losvd(A_stack_w: np.ndarray, y_w: np.ndarray, cfg: FitConfig,
                     n_boot: int = 30, seed: int | None = None,
                     fit_h3h4: bool = True, noise_scale: float = 1.0,
                     apply_bias_correction: bool = False, bias_calibration=None,
                     wave_fit: np.ndarray | None = None, L: np.ndarray | None = None,
                     replicate_single_start: bool = True,
                     reselect_lam: bool = True) -> dict:
    """Parametric bootstrap uncertainty for a LOSVD + template-mixture-
    weight fit, for any `n_templates >= 1`.

    Parameters
    ----------
    A_stack_w : ndarray, shape (n_templates, n_pix, n_bins), as passed to
        `kinextract.multi_template.fit_losvd`.
    y_w : the observed, noise-whitened flux.
    cfg : `FitConfig`.
    L : **not needed for the mainline correlated-noise case.** If
        `A_stack_w`/`y_w` were whitened correctly via
        `noise_model.whiten(..., rho=rho)` using the *true* `rho`, the
        residuals are already i.i.d. N(0,1) by construction (GLS pre-
        whitening removes correlation, it doesn't just rescale it --
        confirmed directly in `tests/test_noise_model.py`), so the plain
        `rng.normal(0, noise_scale, n_pix)` injection below is already
        correct with no changes, regardless of `rho`. `L` exists only to
        deliberately inject noise from a *different* covariance than
        whatever `A_stack_w`/`y_w` were actually whitened against (testing
        an assumed-vs-true `rho` mismatch, the realistic case since `rho`
        is itself only ever an estimate). Default `None` (no deliberate
        mismatch): `noise_scale * L @ rng.standard_normal(n_pix)` replaces
        `rng.normal(0, noise_scale, n_pix)` when given.
    wave_fit : the observed wavelength grid, only required if
        `cfg.continuum_knots is not None` (see `fit_losvd`'s own
        docstring) -- every bootstrap replicate then re-fits with
        continuum-cofitting active, not just the point estimate. Measured
        cost: ~1.17x a plain fit per replicate (see
        `kinextract_test_suite`/this session's own continuum-cofit timing
        check), not the "materially slower" risk this was flagged as
        before being measured.
    n_boot : number of bootstrap replicates. 30 is a reasonable default;
        more tightens the uncertainty-on-the-uncertainty at proportionally
        higher cost.
    seed : for reproducibility; `None` draws fresh entropy each call.
    replicate_single_start : default True. Fits every bootstrap replicate
        with `cfg.losvd_single_start` forced True (single deterministic
        start), regardless of what `cfg` itself says -- the point
        estimate (`result0` above) always keeps `cfg`'s own multi-start
        setting untouched. `FitConfig.losvd_single_start`'s own docstring
        flags single-start as "not recommended for real fits" because an
        occasional (~1-in-150) bad local minimum corrupts a point
        estimate outright; that risk is far smaller here, since a bad
        replicate just contributes one outlier to an n_boot-sized
        distribution instead of being the reported answer. Measured
        directly (this package's own dev notes, 3 seeds x n_boot=50 on a
        sigma=50/S-N=60 mock): ~2.5x faster per bootstrap call, with
        replicate-to-replicate scatter (`sherm_std`/`h4_std`) equal to or
        slightly *larger* than the multi-start band, never smaller -- i.e.
        it does not silently understate the uncertainty. Set False to
        match every replicate's start strategy to `cfg`'s own
        `losvd_single_start` instead (the previous, always-multi-start-
        unless-`cfg`-says-otherwise behavior).

        **This 2.5x number predates a 2026-07-30 bug fix and is now a
        floor, not the whole story.** `FitConfig.losvd_single_start` used
        to force its deterministic start on *every* alternating iteration,
        not just the first -- disabling the separate iteration-to-
        iteration warm-start mechanism entirely and, on a real-templates/
        order-3-default case, making single-start replicates ~3-5x
        *slower* than multi-start ones (the opposite of this flag's whole
        purpose). Fixed (see that field's own docstring); replicates
        should now be faster than multi-start in every regime, not just
        the specific mock this 2.5x figure was measured on -- not yet
        re-measured end-to-end on that same mock to get a fresh number.
    reselect_lam : default True (as of 2026-07-28). If `cfg.lam_selection ==
        "aic_local"`, every bootstrap replicate re-runs that same adaptive
        selection independently instead of being pinned at the point
        estimate's own `lam_used` -- a real, measured contributor to the
        previously under-reported uncertainty band (see
        `dev_notes/uncertainty/FINDINGS.md`). Lambda selection is a
        documented lottery at S/N 60-150 (the same real data can pick
        lam=8 or lam=32768 depending only on the noise draw -- see
        `aic_regularization`'s own module docstring); pinning it hides that
        instability from the reported band entirely. Measured directly
        (isolated from the separate single-start-vs-multi-start question,
        `dev_notes/uncertainty/diag01b_reselect_isolated.py`): reselecting
        moves `sherm_std` from 4.12 -> 3.38 (true independent-noise-draw
        sampling distribution: 3.18) at sigma=50/S-N=60, and 4.76 -> 4.01
        (true: 3.49) at sigma=90/S-N=60 -- closing roughly 60-80% of the
        gap to the true sampling distribution's own spread in both regimes
        tested. This is real but NOT the whole under-coverage story: the
        separate, larger effect of resampling around a *biased* point
        estimate (see `apply_bias_correction` above) is untouched by this
        parameter and is still expected to be the dominant remaining
        contributor. Costs roughly 2-2.5x per replicate (measured: ~7.3s
        vs ~3.2s for a single-start fit on a 4-template mock) -- real, but
        affordable given this package's own ~6.8x fitting speed-up. When
        `cfg.lam_selection == "fixed"` already (a user-supplied fixed
        `lam`), this parameter has no effect either way -- there is
        nothing to reselect, and behavior is identical to `False`.
        Likewise forced off (regardless of this argument's value) when
        `cfg.continuum_knots` is set: adaptive selection doesn't support
        continuum-cofitting yet and would just re-trigger `fit_losvd`'s own
        fixed-lam fallback (and its `UserWarning`) on every replicate for
        no behavioral difference. Set `False` to restore the original
        (cheaper, pinned) behavior.
    fit_h3h4 : passed through to `compute_losvd_moments` for every
        replicate.
    noise_scale : default 1.0 (every replicate's injected noise is unit-
        variance, matching what the whitened `A_stack_w`/`y_w` assume).
        Set to the output of `calibrate_noise_scale` to inject
        `N(0, noise_scale)` instead.
    apply_bias_correction : default False (as of 2026-07 -- two distinct
        overcorrection failure modes found under real testing; see this
        module's own docstring for both, and for how to opt back in).
        When True, applies the Pinkney-style regularization-bias
        correction to the point estimate and inflates the reported
        sigma/h4 uncertainty accordingly. Adds new `_corrected`-suffixed
        keys either way, does not change any existing key's meaning --
        when False, those keys are just copies of the raw ones.
    bias_calibration : optional `BiasCalibration` or `PerLamCalibration`
        (from `bias_correction.load_calibration_table`/
        `load_per_lam_calibration_table`) to use instead of the default
        table -- always wins if given, whatever it is. If left `None`,
        the table is chosen automatically from the fit's own `lam`
        (`result0["lam_used"]`, whether it got there via adaptive
        selection or a caller-supplied fixed value): the plain (sigma, S/N) table
        (built at a single fixed `lam`) unless that `lam` is more than 25%
        away from the table's own -- only then does the per-`lam` set
        (`bias_correction.get_default_lam_axis_calibration`) get used
        instead. Checked against the actual `lam` value itself, not
        against whether adaptive selection happened to run -- see this
        module's own body comment at the call site for why.

    Returns
    -------
    dict with:
      result0 -- the point-estimate `fit_losvd` result, including
        `result0['lam_used']` (the `lam` actually used, after adaptive
        selection if `cfg.lam_selection == 'aic_local'`) and
        `result0['lam_selection_result']` (the full `AICSelectionResult`,
        or `None` if selection didn't run). Every bootstrap replicate
        below is fit at this same `lam_used`, not re-selected per
        replicate -- see this function's own body comment for why.
      point0 -- `compute_losvd_moments` applied to `result0['b']`
        (uncorrected -- see `point0_corrected`).
      w_norm0 -- `result0['w']` normalized to mixture fractions
        (`w / w.sum()`), the scientifically meaningful quantity given the
        weight/LOSVD scale non-identifiability (see `multi_template.py`'s
        docstring); trivially `[1.0]` for a single template.
      b_boot -- (n_boot, n_bins) array of bootstrap-recovered LOSVD
        curves, each unit-area normalized (`trapz(b, v_grid) == 1`) to
        remove the `(w, b) -> (c*w, b/c)` scale degeneracy before
        comparing replicates -- compare against a similarly unit-area-
        normalized point-estimate curve, not `result0['b']` directly.
      b_boot_cov -- (n_bins, n_bins) covariance matrix of `b_boot`.
      b_median, b_std, b_p16, b_p84 -- `np.median(b_boot, axis=0)` and the
        per-bin std/16th/84th percentile envelope of `b_boot`. Plot this
        (not `result0['b']`/`point0`'s own curve) as the reported LOSVD --
        the bootstrap band is centered on `b_median`, not on the point
        estimate, so presenting the point estimate as "the answer" next to
        a band that isn't centered on it is misleading.
      w_boot -- (n_boot, n_templates) array of normalized mixture
        fractions per replicate.
      w_std, w_p16, w_p84 -- per-template std and 16th/84th percentile of
        `w_boot`.
      vherm_boot, sherm_boot, h3_boot, h4_boot -- (n_boot,) arrays of the
        recovered moments for each replicate.
      vherm_std, sherm_std, h3_std, h4_std -- std of each moment array --
        raw bootstrap spread, uncorrected (see `sherm_std_corrected`/
        `h4_std_corrected`).
      snr_estimated -- `bias_correction.estimate_snr_whitened(model_w)`,
        the S/N used to look up the correction.
      point0_corrected -- same shape as `point0`, with `sherm`/`h4`
        bias-corrected (`vherm`/`h3`/`amp` unchanged, `model` rebuilt from
        the corrected moments). **The recommended central estimate.**
      sherm_std_corrected, h4_std_corrected -- `sherm_std`/`h4_std`
        inflated in quadrature with the calibration's own bias-scatter
        term at this (sherm, S/N) -- **the recommended reported
        uncertainty.** Equal to the raw `*_std` if `apply_bias_correction`
        is False.
      sig_uncertainty_inflation, h4_uncertainty_inflation -- the
        calibration bias-scatter terms actually added in quadrature above
        (0.0 if `apply_bias_correction` is False).
      b_median_corrected, b_p16_corrected, b_p84_corrected -- the
        nonparametric percentile band (`b_p16`/`b_p84`), recentered on the
        bias-corrected curve (`bias_correction.correct_losvd`'s parametric
        reconstruction) and its width stretched by the same ratio
        `sherm_std_corrected/sherm_std` was inflated by -- an approximation
        (one scalar ratio, not a per-quantity/per-bin recalibration), but
        keeps the plotted band's width consistent with the recommended
        reported uncertainty rather than silently showing the raw,
        uninflated bootstrap width next to a much larger reported number.
        **The recommended LOSVD curve + band to plot.** Equal to
        `np.median(b_boot, axis=0)`/`b_p16`/`b_p84` if
        `apply_bias_correction` is False.
    """
    rng = np.random.default_rng(seed)
    result0 = fit_losvd(A_stack_w, y_w, cfg, wave_fit=wave_fit)
    v_grid = cfg.build_velocity_grid()
    point0 = compute_losvd_moments(v_grid, result0["b"], fit_h3h4=fit_h3h4)
    w_norm0 = result0["w"] / result0["w"].sum()
    A_eff0 = np.tensordot(result0["w"], A_stack_w, axes=([0], [0]))
    # With continuum active, the model each bootstrap replicate's noise is
    # injected around must include it too -- omitting C0 here would inject
    # noise around the wrong (continuum-free) prediction, silently biasing
    # every replicate low/high wherever C0 deviates from 1.
    C0 = result0.get("C")
    model_w = (C0 * (A_eff0 @ result0["b"])) if C0 is not None else (A_eff0 @ result0["b"])
    n_templates, n_pix, _ = A_stack_w.shape

    # reselect_lam=True (default, 2026-07-28): every one of the n_boot replicates
    # below re-runs cfg's own lam_selection independently (adaptive, if that's what
    # cfg says) instead of being pinned at the point estimate's single selection --
    # see this function's own reselect_lam docstring for the measured evidence this
    # closes a real fraction of the previously under-reported uncertainty band.
    # reselect_lam=False restores the original behavior: lam pinned at the point
    # estimate's own result0["lam_used"] for every replicate (cheaper, at the cost
    # of not propagating lam-selection uncertainty into the reported band).
    #
    # Forced off when cfg.continuum_knots is set, regardless of the reselect_lam
    # argument: fit_losvd's own adaptive selection doesn't support continuum-cofitting
    # yet and silently falls back to a fixed lam with a UserWarning every time it's
    # asked to try (see fit_losvd's own docstring) -- reselecting per replicate here
    # would just re-trigger that fallback (and its warning) n_boot times for no
    # behavioral difference at all, since every replicate would fall back to the same
    # fixed lam anyway.
    if reselect_lam and cfg.continuum_knots is None:
        cfg_boot = dataclasses.replace(
            cfg, losvd_single_start=(True if replicate_single_start else cfg.losvd_single_start),
        )
    else:
        cfg_boot = dataclasses.replace(
            cfg, lam=result0["lam_used"], lam_selection="fixed",
            losvd_single_start=(True if replicate_single_start else cfg.losvd_single_start),
        )

    b_boot = np.empty((n_boot, len(v_grid)))
    w_boot = np.empty((n_boot, n_templates))
    vherm = np.empty(n_boot)
    sherm = np.empty(n_boot)
    h3 = np.empty(n_boot)
    h4 = np.empty(n_boot)
    for i in range(n_boot):
        injected = (L @ rng.standard_normal(n_pix)) if L is not None else rng.standard_normal(n_pix)
        y_w_boot = model_w + noise_scale * injected
        result_i = fit_losvd(A_stack_w, y_w_boot, cfg_boot, wave_fit=wave_fit)
        b_i = result_i["b"]
        b_area = np.trapezoid(np.maximum(b_i, 0.0), v_grid)
        b_boot[i] = b_i / b_area if b_area > 0 else b_i
        w_boot[i] = result_i["w"] / result_i["w"].sum()
        m = compute_losvd_moments(v_grid, result_i["b"], fit_h3h4=fit_h3h4)
        vherm[i], sherm[i], h3[i], h4[i] = m["vherm"], m["sherm"], m["h3"], m["h4"]

    b_median = np.median(b_boot, axis=0)
    b_p16 = np.percentile(b_boot, 16, axis=0)
    b_p84 = np.percentile(b_boot, 84, axis=0)
    sherm_std = float(sherm.std(ddof=1))
    h4_std = float(h4.std(ddof=1))

    snr_estimated = _estimate_snr_whitened(model_w)
    if apply_bias_correction:
        # Default table choice: only switch away from the plain (sigma, S/N) table (built
        # at a single fixed lam) to the per-lam set if the fit's own lam
        # (result0["lam_used"], however it got there -- adaptive selection or a
        # caller-supplied fixed value) is meaningfully different from that table's own.
        # The plain table has far more (sigma, S/N) reps concentrated at its one lam
        # than any one table inside the per-lam set has at its own lam, so
        # switching tables when the lam values actually match makes the correction
        # worse, not better (see VALIDATION_CAMPAIGN.md Stage 12). Never overrides a
        # caller-supplied bias_calibration. Uses result0["lam_used"] (the POINT
        # ESTIMATE's own lam), not cfg_boot.lam -- the two are no longer always equal
        # now that reselect_lam can leave cfg_boot's lam_selection adaptive, in which
        # case cfg_boot.lam is just whatever cfg.lam happened to default to, unused.
        lam_used = result0["lam_used"]
        if bias_calibration is None:
            plain_table = _get_default_calibration()
            lam_matches_plain_table = (
                plain_table.lam is not None
                and abs(lam_used - plain_table.lam) <= 0.25 * plain_table.lam
            )
            bias_calibration = plain_table if lam_matches_plain_table else _get_default_lam_axis_calibration()
        # Only pass lam if the resolved calibration actually has that axis -- passing
        # it unconditionally would raise ValueError against a plain single-lam table.
        # No extra fit/computation needed for this (unlike the old m_eff-axis design):
        # result0["lam_used"] is already known from the point-estimate fit itself.
        lam_for_correction = lam_used if _calibration_has_lam_axis(bias_calibration) else None
        corr = _correct_moments(point0["vherm"], point0["sherm"], point0["h4"],
                                 snr_estimated, calibration=bias_calibration, lam=lam_for_correction)
        sherm_corrected = corr["sherm_corrected"]
        h4_corrected = corr["h4_corrected"]
        sig_inflation = corr["sig_uncertainty_inflation"]
        h4_inflation = corr["h4_uncertainty_inflation"]
        point0_corrected = dict(
            point0,
            sherm=sherm_corrected, h4=h4_corrected,
            # clipped non-negative: the calibrated correction shifts h4 by
            # an offset with no constraint keeping the truncated Gauss-
            # Hermite reconstruction inside its non-negative region -- see
            # `bias_correction.correct_losvd`'s docstring for why this
            # matters and why clipping (not renormalizing) is the fix.
            model=np.maximum(gauss_hermite_model(v_grid, point0["amp"], point0["vherm"],
                                                  sherm_corrected, point0["h3"], h4_corrected), 0.0),
        )
        sherm_std_corrected = float(np.hypot(sherm_std, sig_inflation))
        h4_std_corrected = float(np.hypot(h4_std, h4_inflation))
        # Renormalize to unit area, matching b_boot's own convention --
        # NOT a division by point0['amp'], since the corrected curve's
        # own integral isn't guaranteed to equal that unchanged amplitude
        # parameter exactly (sherm/h4 changed, amp did not).
        corrected_curve = np.maximum(point0_corrected["model"], 0.0)
        corrected_area = np.trapezoid(corrected_curve, v_grid)
        b_median_corrected = corrected_curve / corrected_area if corrected_area > 0 else corrected_curve
        b_median_raw = b_median
        # Band WIDTH must reflect sherm_std_corrected's own inflation, not just
        # recenter the raw (uninflated) bootstrap band on the corrected median.
        # A previous version did only the recenter step -- b_p16_corrected/
        # b_p84_corrected were the raw band shifted, silently leaving the plotted
        # width at the raw, uninflated value even though sherm_std_corrected (the
        # recommended reported number) is inflated in quadrature with the
        # calibration's own bias-scatter term and can be several times larger.
        # Confirmed directly: this made the plotted band look far tighter than
        # the number next to it. Stretching the raw band's own deviation from its
        # median by the same ratio sherm_std was inflated by is an approximation
        # (one scalar ratio can't separately capture sigma's and h4's own
        # inflation at every v-bin), but is far more honest than no inflation.
        width_inflation = sherm_std_corrected / sherm_std if sherm_std > 0 else 1.0
        b_p16_corrected = np.maximum(b_median_corrected + (b_p16 - b_median_raw) * width_inflation, 0.0)
        b_p84_corrected = np.maximum(b_median_corrected + (b_p84 - b_median_raw) * width_inflation, 0.0)
    else:
        sig_inflation = 0.0
        h4_inflation = 0.0
        point0_corrected = point0
        sherm_std_corrected = sherm_std
        h4_std_corrected = h4_std
        b_median_corrected = b_median
        b_p16_corrected = b_p16
        b_p84_corrected = b_p84

    return dict(
        result0=result0,
        point0=point0,
        w_norm0=w_norm0,
        b_boot=b_boot,
        b_boot_cov=np.cov(b_boot, rowvar=False),
        b_std=b_boot.std(axis=0, ddof=1),
        b_median=b_median,
        b_p16=b_p16,
        b_p84=b_p84,
        w_boot=w_boot,
        w_std=w_boot.std(axis=0, ddof=1),
        w_p16=np.percentile(w_boot, 16, axis=0),
        w_p84=np.percentile(w_boot, 84, axis=0),
        vherm_boot=vherm, sherm_boot=sherm, h3_boot=h3, h4_boot=h4,
        vherm_std=float(vherm.std(ddof=1)), sherm_std=sherm_std,
        h3_std=float(h3.std(ddof=1)), h4_std=h4_std,
        snr_estimated=snr_estimated,
        point0_corrected=point0_corrected,
        sherm_std_corrected=sherm_std_corrected,
        h4_std_corrected=h4_std_corrected,
        sig_uncertainty_inflation=sig_inflation,
        h4_uncertainty_inflation=h4_inflation,
        b_median_corrected=b_median_corrected,
        b_p16_corrected=b_p16_corrected,
        b_p84_corrected=b_p84_corrected,
    )


def _bootstrap_losvd_worker(args):
    """Top-level (picklable) per-bin worker for `bootstrap_losvd_parallel`
    -- process pools need a module-level function, not a closure, since
    workers unpickle it by import path."""
    (A_stack_w, y_w, cfg, seed, wave_fit, L, n_boot, fit_h3h4, noise_scale,
     apply_bias_correction, bias_calibration, replicate_single_start, reselect_lam) = args
    return bootstrap_losvd(
        A_stack_w, y_w, cfg, n_boot=n_boot, seed=seed, fit_h3h4=fit_h3h4,
        noise_scale=noise_scale, apply_bias_correction=apply_bias_correction,
        bias_calibration=bias_calibration, wave_fit=wave_fit, L=L,
        replicate_single_start=replicate_single_start, reselect_lam=reselect_lam,
    )


def bootstrap_losvd_parallel(A_stack_w_list, y_w_list, cfg_list, n_boot: int = 30,
                              seed: int | None = None, wave_fit=None, L_list=None,
                              fit_h3h4: bool = True, noise_scale: float = 1.0,
                              apply_bias_correction: bool = False, bias_calibration=None,
                              replicate_single_start: bool = True, reselect_lam: bool = True,
                              n_jobs: int | None = None) -> list[dict]:
    """Bootstrap uncertainty for many independent bins in parallel via a
    process pool -- the bootstrap analogue of
    `multi_template.fit_losvd_parallel`, and usually the bigger win of the
    two for a full multi-bin analysis: `fit_losvd_parallel` parallelizes
    each bin's *point-estimate* fit (one fit per bin), while this
    parallelizes each bin's full bootstrap (`n_boot` refits per bin) --
    bootstrap is typically the dominant per-bin cost precisely because
    it's many refits, not one, and `fit_losvd_parallel`'s own docstring
    flags this exact gap ("parallelizes across bins, not across bootstrap
    replicates within one bin"). This closes it at the bin level: every
    bin's *entire* bootstrap call runs in one worker, so the speedup
    scales with bin count the same way `fit_losvd_parallel`'s does.

    Every bin's bootstrap is completely independent (no shared state
    between bins), so this is pure wall-clock speedup with zero accuracy
    cost -- identical per-bin numerics to calling `bootstrap_losvd`
    serially in a loop with the same per-bin seed (see `seed` below), just
    distributed across worker processes.

    Parameters
    ----------
    A_stack_w_list, y_w_list : sequences of the same length, one entry per
        bin, each matching `bootstrap_losvd`'s own `A_stack_w`/`y_w`.
    cfg_list : a single `FitConfig` (reused for every bin) or a sequence
        of one per bin -- see `fit_losvd_parallel`'s own docstring for why
        (the common two-stage per-bin refit pattern).
    seed : base seed. Bin `i` is bootstrapped with `seed + i` (or `None`
        for every bin, drawing fresh entropy each time, if `seed` itself
        is `None`) -- **not** the same `seed` repeated for every bin,
        which would make every bin's replicate noise draws identical in a
        way that has nothing to do with the data and everything to do
        with sharing one RNG stream. This is also exactly what a serial
        `for i, ...: bootstrap_losvd(..., seed=seed+i)` loop would do, so
        results are directly comparable to (and, at `n_jobs=1`,
        bit-identical to) that serial convention.
    wave_fit, L_list : `wave_fit` (only needed if `cfg.continuum_knots` is
        set) and `L` (only needed to deliberately inject a mismatched
        covariance -- see `bootstrap_losvd`'s own docstring) each accept
        either a single value (shared across every bin) or a sequence of
        one per bin.
    n_boot, fit_h3h4, noise_scale, apply_bias_correction, bias_calibration,
    replicate_single_start, reselect_lam : shared across every bin,
        forwarded directly to each bin's own `bootstrap_losvd` call -- see
        that function's own docstring for what each controls.
    n_jobs : worker process count. `None` (default) uses `os.cpu_count()`,
        memory-permitting -- see `multi_template.resolve_n_jobs`'s own
        docstring. `1` disables the process pool and runs serially
        in-process. See `fit_losvd_parallel`'s own docstring for the same
        heterogeneous-core-hardware caveat (measure your own machine;
        more workers is not always faster on Apple Silicon's performance/
        efficiency mix).

    Returns
    -------
    list of `bootstrap_losvd`'s own result dicts, one per bin, in the same
    order as the inputs -- not a generator, so a failure partway through
    surfaces before any results are used.

    Uses the same `ProcessPoolExecutor`/`spawn`-context/BLAS-thread-pinning
    machinery as `fit_losvd_parallel` (`multi_template._limit_worker_threads`)
    -- see that function's own docstring for why (JAX + multiprocessing
    safety, avoiding BLAS oversubscription across workers).
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from .multi_template import _limit_worker_threads, resolve_n_jobs

    n_bins = len(A_stack_w_list)
    if len(y_w_list) != n_bins:
        raise ValueError(f"A_stack_w_list and y_w_list must have the same length "
                          f"(got {n_bins} and {len(y_w_list)}).")
    cfgs = cfg_list if isinstance(cfg_list, (list, tuple)) else [cfg_list] * n_bins
    if len(cfgs) != n_bins:
        raise ValueError(f"cfg_list, given as a sequence, must have one entry per bin "
                          f"(got {len(cfgs)} for {n_bins} bins).")
    waves = wave_fit if isinstance(wave_fit, (list, tuple)) else [wave_fit] * n_bins
    if len(waves) != n_bins:
        raise ValueError(f"wave_fit, given as a sequence, must have one entry per bin "
                          f"(got {len(waves)} for {n_bins} bins).")
    Ls = L_list if isinstance(L_list, (list, tuple)) else [L_list] * n_bins
    if len(Ls) != n_bins:
        raise ValueError(f"L_list, given as a sequence, must have one entry per bin "
                          f"(got {len(Ls)} for {n_bins} bins).")
    seeds = [None] * n_bins if seed is None else [seed + i for i in range(n_bins)]

    jobs = [
        (A_stack_w_list[i], y_w_list[i], cfgs[i], seeds[i], waves[i], Ls[i],
         n_boot, fit_h3h4, noise_scale, apply_bias_correction, bias_calibration,
         replicate_single_start, reselect_lam)
        for i in range(n_bins)
    ]

    if n_jobs == 1:
        return [_bootstrap_losvd_worker(job) for job in jobs]

    # 4x the largest single design matrix -- each of the n_boot replicates reuses the
    # same A_stack_w by reference (bootstrap_losvd's own loop, not a per-replicate copy),
    # so the base design matrix still dominates a job's own memory footprint (see
    # resolve_n_jobs's own docstring for why this cap is memory-aware, not just cpu_count()).
    per_job_bytes = 4 * max((a.nbytes for a in A_stack_w_list), default=0)
    n_jobs = resolve_n_jobs(n_jobs, per_job_bytes)
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=get_context("spawn"),
                              initializer=_limit_worker_threads) as pool:
        return list(pool.map(_bootstrap_losvd_worker, jobs))


def save_bootstrap(boot: dict, filepath: str) -> None:
    """Save a `bootstrap_losvd` result to an `.npz` file -- everything
    needed for uncertainty bands, bin-bin covariance, or a new derived
    quantity invented later, without rerunning the bootstrap.

    `result0` and `point0` are nested dicts (not directly npz-able); the
    fields actually needed for reuse are flattened in with `result0_`/
    `point0_` prefixes.
    """
    arrays = {k: v for k, v in boot.items() if isinstance(v, np.ndarray)}
    scalars = {
        k: v for k, v in boot.items()
        if isinstance(v, float) or (isinstance(v, int) and not isinstance(v, bool))
    }
    for prefix, sub in (("result0_", boot["result0"]), ("point0_", boot["point0"])):
        for k, v in sub.items():
            if isinstance(v, np.ndarray):
                arrays[f"{prefix}{k}"] = v
            elif isinstance(v, float) or (isinstance(v, int) and not isinstance(v, bool)):
                scalars[f"{prefix}{k}"] = v
            else:
                # e.g. objective_gap/losvd_objective_gap (None, when only one
                # starting point was used) or lam_selection_result (an
                # AICSelectionResult object, or None) -- not a savable .npz
                # scalar/array, and not needed for reuse (a diagnostic, not
                # data the caller needs back from disk). Would otherwise get
                # pickled into an object array that load_bootstrap's own
                # allow_pickle=False np.load can't read back.
                continue
    np.savez(filepath, **arrays, **scalars)


def load_bootstrap(filepath: str) -> dict:
    """Load a bootstrap result saved by `save_bootstrap`. Returns a flat
    dict of arrays/scalars (not byte-identical to the original --
    `result0`/`point0`'s own dict structure isn't reconstructed, just the
    arrays that matter for reuse), keyed the same way as the original
    result otherwise.
    """
    data = np.load(filepath)
    return {k: data[k] for k in data.files}
