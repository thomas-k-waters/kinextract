"""Summarizing a recovered non-parametric LOSVD `b(v)` as (V, sigma, h3, h4).

**`compute_losvd_moments` is the recommended entry point.** V and sigma
come from a direct moment of `b(v)` itself (`compute_direct_moments`),
not from fitting a second parametric model -- the recovered curve already
*is* the scientific product, so its own moments are the more direct
estimator. h3/h4 still come from a Gauss-Hermite fit (`fit_h3_h4`), with
V/sigma held fixed at their direct-moment values rather than jointly
fit: raising `v` to the 3rd/4th power amplifies tail noise faster than
direct-moment windowing can discount, so a parametric shape fit does
better there, while fixing V/sigma first (rather than fitting all five
GH parameters jointly) loses nothing relative to the joint fit.
`fit_gauss_hermite` (the full joint 5-parameter fit) is kept as a
standalone, general-purpose alternative -- primarily for comparison
against the wider dynamics literature, which almost universally reports
jointly-fit Gauss-Hermite moments this way -- but is no longer the
recommended way to get V/sigma; see `compute_direct_moments`'s own
docstring for the benchmark this is based on.

**Windowing detail that matters: the direct-moment window must be
smooth, not a hard cutoff.** A raw full-domain moment gets dragged around
by noise in the far wings, so `compute_direct_moments` downweights them
via a smooth logistic taper on `b`'s own value. A *hard* threshold
(`b > 1%` of peak) is not equivalent and is measurably worse for skewed
LOSVDs specifically: a hard window on an asymmetric density is itself
asymmetric about the true center (the heavy-tail side extends further
before dropping below threshold), which biases the windowed moment
toward the heavy tail in direct proportion to the skewness. The smooth
taper removes this bias; see `compute_direct_moments`'s docstring for
the numbers.

**A related subtlety worth knowing when interpreting either estimator:
the Gauss-Hermite `vel`/`sig` parameters are not, in general, the same
thing as the distribution's true first/second moment once h3/h4 are
non-negligible.** E.g. `gauss_hermite_model` with `vel=0, sig=140,
h3=0.1` has a numerically-integrated true mean of about +20.6 km/s, not
0 -- `vel`/`sig` are constructed (van der Marel & Franx 1993) to decouple
from h3/h4 only at leading order, not exactly. Any comparison against a
skewed synthetic truth should use the numerically-integrated moment, not
the raw input `vel`/`sig` parameter (`test_gh_moments.py` does this via a
`_true_moments_numeric` helper). This also means `fit_gauss_hermite`'s
own output isn't guaranteed to equal a skewed LOSVD's actual physical
moments either -- a direct moment measures the physical quantity itself,
independent of any model family's parameter convention.

Uses the van der Marel & Franx (1993) H3/H4 basis-polynomial convention
throughout, so these numbers are directly comparable to the wider
dynamics literature.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares

from .bias_correction import correct_moments as _correct_moments


def _fh3(w: np.ndarray) -> np.ndarray:
    """Third-order Gauss-Hermite basis polynomial (van der Marel & Franx 1993 convention)."""
    return (2 * w**3 - 3 * w) / np.sqrt(3.0)


def _fh4(w: np.ndarray) -> np.ndarray:
    """Fourth-order Gauss-Hermite basis polynomial (van der Marel & Franx 1993 convention)."""
    return (4 * w**4 - 12 * w**2 + 3) / np.sqrt(24.0)


def gauss_hermite_model(v: np.ndarray, amp: float, vel: float, sig: float, h3: float, h4: float) -> np.ndarray:
    """Evaluate the 4-moment Gauss-Hermite LOSVD model at velocities `v`."""
    if not (np.isfinite(sig) and sig > 0):
        return np.full_like(v, np.nan)
    w = (v - vel) / sig
    return amp / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * w**2) * (1.0 + h3 * _fh3(w) + h4 * _fh4(w))


def fit_gauss_hermite(v: np.ndarray, b: np.ndarray, fit_h3h4: bool = True) -> dict:
    """Jointly fit all 5 Gauss-Hermite parameters (amp, V, sigma, h3, h4)
    to a non-parametric LOSVD `b(v)` in one nonlinear least-squares solve.

    **No longer the recommended way to get V/sigma -- see
    `compute_losvd_moments`, which beats this on both, and this module's
    own docstring for the full benchmark.** Still needed for h3/h4
    (`compute_losvd_moments` calls this internally via `fit_h3_h4`, with
    V/sigma held fixed rather than jointly fit), and useful directly for
    comparison against the wider dynamics literature, which almost
    universally reports jointly-fit Gauss-Hermite moments this way. Has a
    real V bias driven
    by its uniform per-point weighting (`sigy=0.01` below is a flat
    weight regardless of how much LOSVD mass is actually there --
    over-sensitive to wing-region residuals relative to a flux-weighted
    estimate); kept uniformly weighted since a flux-weighted alternative
    measurably underperforms it on a broad validation grid.

    Returns a dict with keys: vherm, sherm, h3, h4, amp, model, success.
    """
    v = np.asarray(v, float)
    b = np.asarray(b, float)
    ok = np.isfinite(v) & np.isfinite(b)
    vfit, bfit = v[ok], b[ok]

    nan_out = dict(vherm=np.nan, sherm=np.nan, h3=np.nan, h4=np.nan, amp=np.nan,
                    model=np.full_like(v, np.nan), success=False)
    if vfit.size < 5:
        return nan_out

    bpos = np.maximum(bfit, 0.0)
    total = np.trapezoid(bpos, vfit) if bpos.sum() > 0 else 0.0
    if total <= 0:
        vel0 = float(vfit[np.argmax(bfit)])
        sig0 = max(float(np.std(vfit)), 1.0)
        amp0 = max(float(np.sum(bfit)), 1e-6)
    else:
        vel0 = float(np.trapezoid(bpos * vfit, vfit) / total)
        sig0 = max(float(np.sqrt(np.trapezoid(bpos * (vfit - vel0) ** 2, vfit) / total)), 1.0)
        amp0 = total

    vspan = max(float(np.ptp(vfit)), float(np.std(vfit)), 1.0)
    sigy = 0.01  # uniform weight in the moment fit

    if fit_h3h4:
        p0 = [amp0, vel0, np.log(sig0), 0.0, 0.0]
        lo = [0.0, vfit.min() - 2 * vspan, np.log(1e-6), -1.0, -1.0]
        hi = [np.inf, vfit.max() + 2 * vspan, np.log(1e6), 1.0, 1.0]

        def resid(p):
            return (gauss_hermite_model(vfit, p[0], p[1], np.exp(p[2]), p[3], p[4]) - bfit) / sigy

        res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=10000)
        amp, vel, log_sig, h3, h4 = res.x
        sig = np.exp(log_sig)
    else:
        p0 = [amp0, vel0, np.log(sig0)]
        lo = [0.0, vfit.min() - 2 * vspan, np.log(1e-6)]
        hi = [np.inf, vfit.max() + 2 * vspan, np.log(1e6)]

        def resid(p):
            return (gauss_hermite_model(vfit, p[0], p[1], np.exp(p[2]), 0.0, 0.0) - bfit) / sigy

        res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=10000)
        amp, vel, log_sig = res.x
        sig = np.exp(log_sig)
        h3 = h4 = 0.0

    model = gauss_hermite_model(v, amp, vel, sig, h3, h4)
    return dict(vherm=float(vel), sherm=float(sig), h3=float(h3), h4=float(h4),
                amp=float(amp), model=model, success=bool(res.success))


def compute_direct_moments(v: np.ndarray, b: np.ndarray, core_threshold: float = 0.01,
                            taper_width: float = 0.004) -> dict:
    """V and sigma computed directly from `b(v)`'s own first and second
    moments -- no model fit. Downweights the noise-dominated far wings via
    a smooth logistic taper on `b`'s own value (a sigmoid centered at
    `core_threshold` of `b`'s peak, with transition width `taper_width` x
    peak), not a hard cutoff.

    **The taper must be smooth, not a hard threshold.** A hard
    `b > core_threshold*peak` cutoff is not equivalent: for a *skewed*
    LOSVD (h3!=0), a hard window is itself asymmetric about the true
    center (the heavy-tail side extends further before dropping below
    threshold than the light-tail side), which biases the windowed moment
    toward the heavy tail (by ~20 km/s at h3=+-0.1 on this package's
    standard validation grid, vs. ~1 km/s for the smooth taper on the same
    cases). Benchmarked across the full grid (V in {-200,0,200}, sigma in
    {50,140,300}, plus moderate h3/h4, S/N~100), the smooth-taper moment
    beats `fit_gauss_hermite` for both V (std 2.5 vs 8.7 km/s, max 11.0 vs
    25.5 km/s) and sigma (std 2.9 vs 7.4 km/s, max 8.8 vs 28.3 km/s).
    Result is not sensitive to the exact taper width (0.001-0.01 x peak
    all perform similarly); this is what `taper_width`'s default is drawn
    from.

    Returns a dict with keys: vherm, sherm, n_eff (the effective sample
    size under the taper weights, `sum(w)^2/sum(w^2)` -- a cheap
    diagnostic; very small values mean the window or grid resolution
    needs a second look).
    """
    v = np.asarray(v, float)
    b = np.asarray(b, float)
    bpos = np.maximum(b, 0.0)
    peak = bpos.max()
    if peak <= 0:
        return dict(vherm=np.nan, sherm=np.nan, n_eff=0.0)
    taper = 1.0 / (1.0 + np.exp(-(bpos - core_threshold * peak) / (taper_width * peak)))
    w = bpos * taper
    total = w.sum()
    if total <= 0:
        return dict(vherm=np.nan, sherm=np.nan, n_eff=0.0)
    vel = float(np.sum(v * w) / total)
    sig = float(np.sqrt(np.sum(w * (v - vel) ** 2) / total))
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    return dict(vherm=vel, sherm=sig, n_eff=n_eff)


def fit_h3_h4(v: np.ndarray, b: np.ndarray, vel: float, sig: float) -> dict:
    """Fit only `amp`, `h3`, `h4` to `b(v)`, with `vel`/`sig` held fixed
    (e.g. at `compute_direct_moments`'s output) rather than jointly
    optimized. Loses nothing relative to `fit_gauss_hermite`'s joint
    5-parameter fit (see this module's own docstring for the benchmark)
    while being simpler and removing any V/sigma-h3/h4 coupling from the
    h3/h4 estimate.

    Returns a dict with keys: h3, h4, amp, success.
    """
    v = np.asarray(v, float)
    b = np.asarray(b, float)
    ok = np.isfinite(v) & np.isfinite(b)
    vfit, bfit = v[ok], b[ok]
    if vfit.size < 3 or not (np.isfinite(sig) and sig > 0):
        return dict(h3=np.nan, h4=np.nan, amp=np.nan, success=False)

    amp0 = max(float(np.trapezoid(np.maximum(bfit, 0.0), vfit)), 1e-6)

    def resid(p):
        return gauss_hermite_model(vfit, p[0], vel, sig, p[1], p[2]) - bfit

    res = least_squares(resid, [amp0, 0.0, 0.0], bounds=([0.0, -1.0, -1.0], [np.inf, 1.0, 1.0]), max_nfev=10000)
    amp, h3, h4 = res.x
    return dict(h3=float(h3), h4=float(h4), amp=float(amp), success=bool(res.success))


def shape_aware_taper_correction(vel_est: float, sig_est: float, h3_est: float, h4_est: float,
                                  core_threshold: float = 0.01, taper_width: float = 0.004) -> tuple[float, float]:
    """Correction factor/shift for `compute_direct_moments`'s taper-
    windowing bias (see `compute_direct_moments`'s own docstring), tuned
    to the *specific* recovered shape rather than a universal (Gaussian)
    constant.

    **Mechanism.** The windowing bias is a purely geometric property of
    the taper applied to a given shape -- confirmed shape- and noise-
    independent (identical at S/N=100 and S/N=100000 on the same true
    shape). Given a current `(vel, sig, h3, h4)` estimate, build a fine-
    grid *reference* Gauss-Hermite curve with exactly that shape, run it
    through the identical taper windowing, and measure the bias that
    shape produces -- then return the factor/shift needed to undo it.
    Exact (zero residual bias) when applied to the noiseless truth its
    own shape estimate came from, by construction.

    **On noisy fits, a real net improvement, with one known exception.**
    Tested across the standard 13-case validation grid (20 seeds/case,
    S/N~100): excluding sigma=300, mean |bias| dropped from 1.95 to 1.41
    km/s (~28%), and the case this was built for (h4=+0.1) improved
    substantially (-3.88 -> -1.36 km/s, ~65%). **At sigma=300 specifically,
    this correction makes bias worse, not better** (e.g. V=0/sigma=300:
    +0.8 -> +4.8 km/s) -- that regime already has a separate, not-yet-
    understood issue (a "sloppy," near-degenerate parameter manifold at
    high coefficient counts, see `kinextract.uncertainty`'s module
    docstring), and the noisier h3/h4 *estimate* this correction depends
    on there means it's applying a real correction mechanism to an
    unreliable shape estimate, compounding rather than fixing the
    existing problem. Not resolved -- until it is, don't use this
    correction on sigma ~300+ (broad, well-detected) targets specifically.

    Returns
    -------
    (factor, v_shift) -- multiply the raw `sherm` by `factor`, add
    `v_shift` to the raw `vherm`.
    """
    v_ref = np.linspace(vel_est - 20 * sig_est, vel_est + 20 * sig_est, 50000)
    b_ref = np.maximum(gauss_hermite_model(v_ref, 1.0, vel_est, sig_est, h3_est, h4_est), 0.0)
    tot = np.trapezoid(b_ref, v_ref)
    if tot <= 0:
        return 1.0, 0.0
    true_v_ref = float(np.trapezoid(b_ref * v_ref, v_ref) / tot)
    true_s_ref = float(np.sqrt(np.trapezoid(b_ref * (v_ref - true_v_ref) ** 2, v_ref) / tot))
    dm_ref = compute_direct_moments(v_ref, b_ref, core_threshold=core_threshold, taper_width=taper_width)
    if not np.isfinite(dm_ref["sherm"]) or dm_ref["sherm"] <= 0:
        return 1.0, 0.0
    factor = true_s_ref / dm_ref["sherm"]
    v_shift = true_v_ref - dm_ref["vherm"]
    return factor, v_shift


_SHAPE_AWARE_UNSAFE_SIGMA = 200.0


def compute_losvd_moments(v: np.ndarray, b: np.ndarray, core_threshold: float = 0.01,
                           fit_h3h4: bool = True, shape_aware_correction: bool = False,
                           regularization_bias_correction: bool = False, snr: float | None = None,
                           bias_calibration=None, lam: float | None = None) -> dict:
    """**The recommended way to summarize a recovered LOSVD as (V, sigma,
    h3, h4).** V and sigma come from a smooth-taper direct moment
    (`compute_direct_moments`) -- confirmed to beat `fit_gauss_hermite`'s
    joint fit across this project's full validation grid, see this
    module's own docstring for the benchmark. h3/h4, if requested, come
    from a Gauss-Hermite fit with V/sigma held fixed at those direct
    values (`fit_h3_h4`) rather than a full joint fit -- confirmed to
    lose nothing relative to jointly fitting all 5 parameters.

    `shape_aware_correction` (default False, opt-in): apply
    `shape_aware_taper_correction` to `vherm`/`sherm` using the h3/h4
    just estimated. A real, tested improvement for skewed/kurtotic
    LOSVDs *except* at high sigma (~300+ km/s) where it currently makes
    things worse -- see that function's own docstring for the numbers
    and the (not yet understood) reason. Off by default because of that
    exception, not because the underlying correction is unvalidated. Has
    no effect on `h3`/`h4` themselves. Ignored if `fit_h3h4=False`
    (h3=h4=0 has no windowing bias for V/sigma to correct). If enabled
    and the *recovered* sigma comes back above `_SHAPE_AWARE_UNSAFE_SIGMA`
    (200 km/s -- conservative, interpolated between the validated-good
    140 km/s case and the validated-bad 300 km/s case, not itself a
    directly-tested boundary), this function emits a `UserWarning`
    recommending the caller re-fit with `shape_aware_correction=False`
    -- the fit's own output, not just the input configuration, is what
    determines whether this opt-in helped or hurt, so the check has to
    happen post-fit, here, where sigma is actually known.

    `regularization_bias_correction` (default False, opt-in): apply
    `kinextract.bias_correction.correct_moments` to `sherm`/`h4`
    (applied after `shape_aware_correction`, if both are requested --
    the two target different, independent mechanisms -- windowing-taper
    geometry vs. Tikhonov roughness-penalty shrinkage -- but running both
    together is not itself separately validated; each is individually
    validated on its own). Requires `snr` (see
    `kinextract.bias_correction.estimate_snr_whitened`) -- raises
    `ValueError` if not supplied, since there's no way to look up the
    right calibration point without it. Corrects `sherm`/`h4` only; `V`
    and `h3` pass through unchanged (see `bias_correction`'s own module
    docstring for why). Warns, via `bias_correction`'s own machinery, if
    `snr`/the recovered `sherm` falls outside the calibration grid's
    range rather than silently extrapolating. `bias_calibration` lets you
    pass a specific `BiasCalibration` (e.g. from
    `load_calibration_table(other_path)`) instead of the default table.

    `lam` (default `None`): this function has no fit context of its own
    (only the already-recovered curve `b`), so unlike `uncertainty.
    bootstrap_losvd` it cannot look this up itself -- pass the fit's own
    `lam` (e.g. `result["lam_used"]`) only if correcting a fit made with
    *adaptive* regularization against a `PerLamCalibration` (`lam_grid`
    varied in `calibrate_bias.run_calibration_sweep`). Omit it for this
    package's fixed `lam=2.0` default -- see `bias_correction.
    correct_moments`'s own docstring for what happens either way.

    Returns a dict with the same keys as `fit_gauss_hermite` (vherm,
    sherm, h3, h4, amp, model, success) so the two are interchangeable,
    including for the diagnostic case of comparing them directly against
    each other -- a large disagreement is itself informative (it flags a
    recovered shape neither estimator is fully resolving cleanly).
    `success` reflects the h3/h4 fit's own convergence when `fit_h3h4` is
    True (V/sigma are a direct calculation, not an optimization, so they
    can't fail to converge -- only degenerate input, e.g. b<=0
    everywhere, produces NaN, in which case `success` is False). If
    `regularization_bias_correction=True`, the dict additionally has
    `sherm_raw`/`h4_raw` (the pre-correction values) and
    `sig_uncertainty_inflation`/`h4_uncertainty_inflation` (the
    calibration's own bias scatter at this point -- add in quadrature to
    a bootstrap-derived std; `bootstrap_losvd` does this automatically
    when its own `apply_bias_correction=True`).
    """
    direct = compute_direct_moments(v, b, core_threshold=core_threshold)
    vel, sig = direct["vherm"], direct["sherm"]
    if not (np.isfinite(vel) and np.isfinite(sig) and sig > 0):
        return dict(vherm=np.nan, sherm=np.nan, h3=np.nan, h4=np.nan, amp=np.nan,
                    model=np.full_like(np.asarray(v, float), np.nan), success=False)

    if fit_h3h4:
        h34 = fit_h3_h4(v, b, vel, sig)
        h3, h4, amp, success = h34["h3"], h34["h4"], h34["amp"], h34["success"]
        if shape_aware_correction:
            factor, v_shift = shape_aware_taper_correction(vel, sig, h3, h4, core_threshold=core_threshold)
            sig = sig * factor
            vel = vel + v_shift
            if sig > _SHAPE_AWARE_UNSAFE_SIGMA:
                warnings.warn(
                    f"compute_losvd_moments: shape_aware_correction=True produced sherm={sig:.1f} "
                    f"km/s, above the {_SHAPE_AWARE_UNSAFE_SIGMA:.0f} km/s regime where this "
                    "correction is documented to make sigma *worse*, not better (validated bad at "
                    "sigma=300; validated good at sigma=140) -- see shape_aware_taper_correction's "
                    "own docstring. Re-run with shape_aware_correction=False for this fit.",
                    UserWarning, stacklevel=2,
                )
    else:
        bpos = np.maximum(np.asarray(b, float), 0.0)
        amp = float(np.trapezoid(bpos, np.asarray(v, float)))
        h3 = h4 = 0.0
        success = True

    extra = {}
    corrected = False
    if regularization_bias_correction:
        if snr is None:
            raise ValueError(
                "compute_losvd_moments: regularization_bias_correction=True requires snr "
                "(see kinextract.bias_correction.estimate_snr_whitened)."
            )
        corr = _correct_moments(vel, sig, h4, snr, calibration=bias_calibration, lam=lam)
        extra = dict(
            sherm_raw=sig, h4_raw=h4,
            sig_uncertainty_inflation=corr["sig_uncertainty_inflation"],
            h4_uncertainty_inflation=corr["h4_uncertainty_inflation"],
        )
        sig = corr["sherm_corrected"]
        h4 = corr["h4_corrected"]
        corrected = True

    model = gauss_hermite_model(np.asarray(v, float), amp, vel, sig, h3, h4)
    if corrected:
        # The calibrated correction shifts h4 by an offset chosen to cancel
        # a population-level bias -- nothing constrains the *shifted* h4 to
        # stay inside the (h3, h4) region where the truncated Gauss-Hermite
        # series stays non-negative everywhere (a known property of this
        # expansion even uncorrected, see `losvd.py`'s own docstring on
        # h4=-0.1). Clip rather than leave a "density" that goes negative
        # near the core edges -- matches the convention already used
        # everywhere else in this package a Gauss-Hermite curve is treated
        # as a final density (`ablation.py`, `calibrate_bias.py`,
        # `shape_aware_taper_correction`'s own `b_ref`), just not
        # previously applied to this specific corrected-model reconstruction.
        # Not applied to the *uncorrected* branch above -- that pre-existing
        # behavior is unchanged, out of scope for this fix.
        model = np.maximum(model, 0.0)
    return dict(vherm=vel, sherm=sig, h3=h3, h4=h4, amp=amp, model=model, success=success, **extra)
