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
docstring for the benchmark this is based on. That benchmark used
adaptively-regularized fits, whose broad wing pedestals hurt a joint GH
fit; on fits at the default fixed `lam` (order-3 penalty, smooth
near-Gaussian wings) the two agree to about 0.5 km/s on the same grid,
with the direct moment's taper costing ~1-2 % in sigma
(`tests/test_gh_moments.py`).

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


import numpy as np
from scipy.optimize import least_squares


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


def compute_losvd_moments(v: np.ndarray, b: np.ndarray, core_threshold: float = 0.01,
                           fit_h3h4: bool = True) -> dict:
    """**The recommended way to summarize a recovered LOSVD as (V, sigma,
    h3, h4).** V and sigma come from a smooth-taper direct moment
    (`compute_direct_moments`) -- confirmed to beat `fit_gauss_hermite`'s
    joint fit across this project's full validation grid, see this
    module's own docstring for the benchmark. h3/h4, if requested, come
    from a Gauss-Hermite fit with V/sigma held fixed at those direct
    values (`fit_h3_h4`) rather than a full joint fit -- confirmed to
    lose nothing relative to jointly fitting all 5 parameters.

    Returns a dict with the same keys as `fit_gauss_hermite` (vherm,
    sherm, h3, h4, amp, model, success) so the two are interchangeable,
    including for the diagnostic case of comparing them directly against
    each other -- a large disagreement is itself informative (it flags a
    recovered shape neither estimator is fully resolving cleanly).
    `success` reflects the h3/h4 fit's own convergence when `fit_h3h4` is
    True (V/sigma are a direct calculation, not an optimization, so they
    can't fail to converge -- only degenerate input, e.g. b<=0
    everywhere, produces NaN, in which case `success` is False).
    """
    direct = compute_direct_moments(v, b, core_threshold=core_threshold)
    vel, sig = direct["vherm"], direct["sherm"]
    if not (np.isfinite(vel) and np.isfinite(sig) and sig > 0):
        return dict(vherm=np.nan, sherm=np.nan, h3=np.nan, h4=np.nan, amp=np.nan,
                    model=np.full_like(np.asarray(v, float), np.nan), success=False)

    if fit_h3h4:
        h34 = fit_h3_h4(v, b, vel, sig)
        h3, h4, amp, success = h34["h3"], h34["h4"], h34["amp"], h34["success"]
    else:
        bpos = np.maximum(np.asarray(b, float), 0.0)
        amp = float(np.trapezoid(bpos, np.asarray(v, float)))
        h3 = h4 = 0.0
        success = True

    model = gauss_hermite_model(np.asarray(v, float), amp, vel, sig, h3, h4)
    return dict(vherm=vel, sherm=sig, h3=h3, h4=h4, amp=amp, model=model, success=success)
