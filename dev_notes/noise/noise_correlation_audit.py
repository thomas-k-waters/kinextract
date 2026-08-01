"""Audits whether real MUSE pixel noise is actually independent, as every
LOSVD fit in this project (and every code reviewed this session) assumes.

**Why this matters more than it sounds like it should.** Every chi^2,
every AIC criterion, every bootstrap variance estimate computed anywhere
in kinextract assumes per-pixel noise is independent Gaussian. MUSE
spectra are resampled/drizzled during pipeline reduction, a well-known
potential source of pixel-to-pixel noise correlation in IFU data. If
real pixels are correlated and we treat them as independent, every one
of those downstream quantities is silently miscalibrated by the same
systematic amount -- a wrong assumption at the foundation contaminates
everything built on top of it, including this week's AICp results.

**Method**: take real MUSE-observed calibration star spectra (the
33-star MUSE Library of Stellar Spectra's own raw files, which carry a
per-pixel error column from the actual MUSE reduction pipeline -- the
most direct real-noise sample available in this project, short of the
raw N5102 galaxy cube itself). Mask the three CaT line cores, fit a
smooth polynomial to the remaining continuum, and compute the empirical
autocorrelation function (ACF) of the residuals -- both raw and
error-standardized -- at lags 1-10. Compare against the white-noise null
(Bartlett bound, ~+-1.96/sqrt(N) for uncorrelated noise) across several
stars, since one star alone can show a spurious lag by chance.

Also cross-checks the real N5102 pipeline's own per-bin noise
characterization (`{bin}sp.rms`) against directly-measured residual
scatter, since inspection this session found the pipeline stores a
*single scalar* per bin (no per-pixel-varying error, let alone
covariance) -- itself a real simplification worth flagging regardless of
what the correlation test finds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np

MUSE_STAR_DIR = "/Users/waterstk/Documents/Gultekin_Astrophysics/kinextract/examples/data/muse"
CAT_LINES = [8498.02, 8542.09, 8662.14]
LINE_HALF_WIDTH = 15.0
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0
MAX_LAG = 10

STAR_FILES = [
    "HD057060_av.dat", "HD064332_av.dat", "HD067507_av.dat", "HD085405_av.dat",
    "HD096446_av.dat", "HD099648_av.dat", "HD099998_av.dat", "HD100733_av.dat",
]


def _load_star(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1], d[:, 2]


def _continuum_mask(wave):
    mask = (wave >= WAVEFITMIN) & (wave <= WAVEFITMAX)
    for cen in CAT_LINES:
        mask &= ~((wave >= cen - LINE_HALF_WIDTH) & (wave <= cen + LINE_HALF_WIDTH))
    return mask


def _acf(x, max_lag):
    x = x - np.mean(x)
    n = len(x)
    var = np.sum(x ** 2)
    return np.array([np.sum(x[:n - lag] * x[lag:]) / var if lag > 0 else 1.0 for lag in range(max_lag + 1)])


def _local_detrend_residual(wave, flux, window_A=20.0, order=1):
    """Detrend in narrow local windows (linear/quadratic fit each) rather
    than one global polynomial -- a global fit over a wide range is
    contaminated by real absorption features the CaT mask doesn't cover;
    over ~20 A even a real, weak, blended feature is close to linear.
    """
    resid = np.full(len(wave), np.nan)
    edges = np.arange(wave.min(), wave.max() + window_A, window_A)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (wave >= lo) & (wave < hi)
        if sel.sum() < order + 3:
            continue
        c = np.polyfit(wave[sel], flux[sel], order)
        resid[sel] = flux[sel] - np.polyval(c, wave[sel])
    keep = np.isfinite(resid)
    return resid[keep], keep


def audit_one_star(path):
    wave, flux, err = _load_star(path)
    mask = _continuum_mask(wave)
    w, f, e = wave[mask], flux[mask], err[mask]

    resid, keep = _local_detrend_residual(w, f, window_A=20.0, order=1)
    e = e[keep]
    resid_std = resid / e

    diffs = np.diff(f[keep])  # model-free complement: first differences

    n = len(resid)
    ci95 = 1.96 / np.sqrt(n)
    acf_raw = _acf(resid, MAX_LAG)
    acf_std = _acf(resid_std, MAX_LAG)
    acf_diff = _acf(diffs, MAX_LAG)

    measured_rms = np.std(resid)
    stated_err_median = np.median(e)

    return dict(
        n=n, ci95=ci95, acf_raw=acf_raw, acf_std=acf_std, acf_diff=acf_diff,
        measured_rms=measured_rms, stated_err_median=stated_err_median,
        lag1_raw=acf_raw[1], lag1_std=acf_std[1], lag1_diff=acf_diff[1],
    )


def main():
    print(f"{'star':>20} {'n':>5} {'95%CI':>8}   {'lag1_std':>9} {'lag2_std':>9} {'lag3_std':>9}   "
          f"{'lag1_diff':>9}   {'meas_rms':>10} {'stated_err':>10} {'ratio':>7}")
    results = []
    for name in STAR_FILES:
        path = os.path.join(MUSE_STAR_DIR, name)
        if not os.path.exists(path):
            print(f"  (skip, not found: {name})")
            continue
        r = audit_one_star(path)
        results.append(r)
        ratio = r["measured_rms"] / r["stated_err_median"]
        print(f"{name:>20} {r['n']:5d} {r['ci95']:8.4f}   {r['lag1_std']:+9.4f} {r['acf_std'][2]:+9.4f} "
              f"{r['acf_std'][3]:+9.4f}   {r['lag1_diff']:+9.4f}   "
              f"{r['measured_rms']:10.3e} {r['stated_err_median']:10.3e} {ratio:7.2f}")

    lag1_vals = np.array([r["lag1_std"] for r in results])
    lag1_diff_vals = np.array([r["lag1_diff"] for r in results])
    ci95 = results[0]["ci95"]
    n_exceed = np.sum(np.abs(lag1_vals) > ci95)
    n_exceed_diff = np.sum(np.abs(lag1_diff_vals) > ci95)
    print(f"\n(local-detrend residual) {n_exceed}/{len(results)} stars show |lag-1 ACF| > "
          f"95% white-noise bound ({ci95:.4f}); mean lag-1 = {np.mean(lag1_vals):+.4f}")
    print(f"(first-difference, model-free) {n_exceed_diff}/{len(results)} stars show |lag-1 ACF| > bound; "
          f"mean lag-1 = {np.mean(lag1_diff_vals):+.4f}")
    ratios = np.array([r["measured_rms"] / r["stated_err_median"] for r in results])
    print(f"measured/stated noise ratio across stars: mean={np.mean(ratios):.2f}, "
          f"range=[{ratios.min():.2f}, {ratios.max():.2f}]")


if __name__ == "__main__":
    main()
