"""Noise-covariance architecture: replaces the implicit "independent
per-pixel Gaussian" assumption every fit in this package has made so far
with an explicit, pluggable covariance model, defaulting to exactly the
old behavior.

**Why this exists.** Every fitting function in this package
(`fit_log_spline_map`, `fit_losvd_basis`, `fit_continuum_block`)
receives already-"whitened" `A_w`/`y_w` and just
computes `0.5*sum(resid**2)` -- correct only if per-pixel noise is
independent. Two real-data audits this project has now run (real MUSE
star spectra, two independent diagnostics) found genuine, non-negligible
lag-1 pixel correlation (~0.3-0.5) that survives even after the model is
allowed to detrend for template-incompleteness via a continuum term (see
`continuum.py` and `examples/notebooks/nima_detrend_vs_noise_correlation.py`).
Every chi^2 and posterior width computed anywhere in this package
inherits whatever's wrong with this assumption.

**The fix requires no changes to the fitting engines themselves.** If
`Sigma = L @ L.T` (Cholesky) and whitening is done as `A_w = L^-1 @ A`,
`y_w = L^-1 @ y` (proper GLS pre-whitening) instead of the old `A/errors`,
`y/errors`, then `sum(resid_w**2) == resid^T @ Sigma^-1 @ resid` exactly,
for free -- `losvd.py`'s spline fit, `multi_template.py`'s NNLS weight
solve, and `continuum.py`'s objective all stay unchanged; they were
already only ever consuming whatever whitened inputs they were handed.
The entire noise-model upgrade lives in *how the whitened inputs are
built*.

**Scope, deliberately narrow for now.** Only a stationary AR(1)-type
correlation (`rho` at lag 1, decaying geometrically) is implemented --
not a general banded or empirically-measured kernel. This matches what's
actually been measured so far (a single lag-1 ACF number from real
star-spectrum residuals) and keeps a dense Cholesky factorization cheap
at MUSE fit-window sizes (~300-500 pixels, milliseconds). Don't build a
more general (banded/sparse/empirical-kernel) covariance until a specific,
verified need for one appears -- this is deliberately the simplest thing
that could be right, not a placeholder for something fancier assumed to
be needed later.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def build_ar1_covariance(errors: np.ndarray, rho: float) -> np.ndarray:
    """Stationary AR(1)-type noise covariance: `Sigma_ij = errors_i *
    errors_j * rho**|i-j|`. At `rho=0` this is exactly `diag(errors**2)`
    (independent noise); `rho` close to 1 means strong pixel-to-pixel
    correlation (the real-data measurement so far: ~0.3-0.5 at lag 1 on
    real MUSE star spectra, surviving continuum-detrending).

    `rho` is a single scalar correlation strength, not a measured kernel
    shape -- see module docstring for why this is deliberately the
    simplest model that could be right.
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"build_ar1_covariance: rho must be in [0, 1), got {rho}.")
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.outer(errors, errors) * (rho ** lag)


def whiten(A_stack: np.ndarray, y: np.ndarray, errors: np.ndarray, rho: float = 0.0):
    """Whiten a template design-matrix stack and observed flux against an
    AR(1) noise covariance (`rho=0`, the default, is the independent-
    pixel case this package has always assumed).

    Parameters
    ----------
    A_stack : ndarray, shape (n_templates, n_pix, n_bins) or (n_pix, n_bins)
        Unwhitened per-template design matrix stack (or a single
        template's design matrix).
    y : ndarray, shape (n_pix,)
        Unwhitened observed flux.
    errors : ndarray, shape (n_pix,)
        Per-pixel noise standard deviation (the diagonal scale -- `rho`
        controls correlation on top of this, not the per-pixel amplitude).
    rho : float
        AR(1) lag-1 correlation, default 0.0 (independent pixels).
        **At `rho=0.0` this reproduces `A_stack/errors[..., None, :]`-style
        division bit-for-bit** (an explicit special case, not relying on
        the general Cholesky path to coincide numerically) -- a
        regression safety net for every existing caller of the old
        `A_stack/errors`, `y/errors` convention.

    Returns
    -------
    (A_stack_w, y_w, L) -- whitened design matrix/matrices (same shape as
    input), whitened flux, and the Cholesky factor `L` (`Sigma = L @ L.T`,
    shape `(n_pix, n_pix)`) -- callers that inject *synthetic* noise
    (e.g. mock spectra) need `L` to draw correlated noise consistent
    with this same covariance (`L @ rng.standard_normal(n_pix)`
    has covariance `Sigma`, vs. plain `rng.standard_normal(n_pix)` which
    only does at `rho=0`).
    """
    errors = np.asarray(errors, dtype=float)
    y = np.asarray(y, dtype=float)
    single_template = A_stack.ndim == 2
    stack = A_stack[None, ...] if single_template else A_stack

    if rho == 0.0:
        A_stack_w = stack / errors[None, :, None]
        y_w = y / errors
        L = np.diag(errors)
    else:
        Sigma = build_ar1_covariance(errors, rho)
        L = cholesky(Sigma, lower=True)
        y_w = solve_triangular(L, y, lower=True)
        A_stack_w = np.stack(
            [solve_triangular(L, stack[k], lower=True) for k in range(stack.shape[0])],
            axis=0,
        )

    if single_template:
        A_stack_w = A_stack_w[0]
    return A_stack_w, y_w, L


def estimate_correlation_length(residuals: np.ndarray, method: str = "lag1_acf") -> float:
    """Estimate the AR(1) `rho` from a real (already continuum-detrended,
    per the module docstring's ordering) residual array -- the reusable
    version of the ad hoc logic in
    `examples/notebooks/noise_correlation_audit.py`.

    `method="lag1_acf"` (the only one implemented): the lag-1 sample
    autocorrelation of `residuals` (mean-subtracted), directly usable as
    an AR(1) `rho` estimate. Returns 0.0 (not negative) if the measured
    lag-1 ACF is negative or non-finite -- a negative estimate isn't a
    meaningful AR(1) correlation strength for `build_ar1_covariance`
    (which assumes `rho >= 0`), and small negative fluctuations are
    expected from pure white noise by chance.
    """
    if method != "lag1_acf":
        raise ValueError(f"estimate_correlation_length: unknown method {method!r}.")
    x = np.asarray(residuals, dtype=float)
    x = x - np.mean(x)
    var = np.sum(x ** 2)
    if var <= 0 or len(x) < 3:
        return 0.0
    lag1 = np.sum(x[:-1] * x[1:]) / var
    return float(lag1) if np.isfinite(lag1) and lag1 > 0 else 0.0


def measure_and_whiten(A_stack: np.ndarray, y: np.ndarray, errors: np.ndarray,
                        residuals: np.ndarray, method: str = "lag1_acf"):
    """`estimate_correlation_length` + `whiten` in one call -- the common
    real-data pattern (fit once, measure `rho` from that fit's own
    residuals, refit against the resulting AR(1)-whitened inputs) that
    callers have so far hand-rolled themselves (e.g.
    `07_full_galaxy_analysis.ipynb` currently hardcodes a `MEASURED_RHO`
    value copied over from a separate notebook run, rather than measuring
    it live from the target's own data).

    Parameters
    ----------
    A_stack, y, errors : as in `whiten`.
    residuals : ndarray
        Continuum-detrended residuals to measure `rho` from -- typically
        from a preliminary `rho=0` fit on the same data (see
        `estimate_correlation_length`'s own docstring for the detrending
        requirement; an un-detrended residual array's ACF reflects
        template mismatch/continuum shape, not per-pixel noise
        correlation).
    method : passed straight through to `estimate_correlation_length`.

    Returns
    -------
    (A_stack_w, y_w, L, rho) -- `whiten`'s own return values, plus the
    `rho` that was measured and used.
    """
    rho = estimate_correlation_length(residuals, method=method)
    A_stack_w, y_w, L = whiten(A_stack, y, errors, rho=rho)
    return A_stack_w, y_w, L, rho


# ------------------------------------------------------------------------------------------------ systematics
# A per-pixel variance model for the residual systematics of real IFU spectra -- sky-line residuals and template
# mismatch -- validated on NGC 4751-like MUSE mocks and the real NGC 4751 bins
# (dev_notes/regularization_v2/FINDINGS.md, 2026-11-01). Use its errors for the uncertainty bands, not the fit: at a
# fixed lam, errors inflated near-uniformly act as a correspondingly stronger penalty and over-smooth the LOSVD (the
# same FINDINGS, finding 16). `laplace_uncertainty.recenter_samples` moves the bands onto the plain-error fit.

def sky_line_profile(wave: np.ndarray, line_wave: np.ndarray, line_flux: np.ndarray, z: float = 0.0,
                     fwhm: float = 2.5) -> np.ndarray:
    """Sky-line emission blended by the instrument's LSF, on the fit's wavelength grid.

    `wave` is the fit's grid in the galaxy rest frame; `line_wave`/`line_flux` are an observed-frame sky-line list in
    the same air/vacuum convention (e.g. `load_muse_sky_lines`). Each line is a Gaussian of `fwhm` (observed-frame
    Angstrom, divided by 1+z here) at `line_wave/(1+z)`. Normalized so its 99th percentile over `wave` is 1."""
    wave = np.asarray(wave, float)
    lw = np.asarray(line_wave, float) / (1.0 + z)
    lf = np.asarray(line_flux, float)
    s = fwhm / (1.0 + z) / 2.3548
    near = (lw > wave.min() - 5 * s) & (lw < wave.max() + 5 * s)
    prof = np.zeros_like(wave)
    for l, f in zip(lw[near], lf[near]):
        prof += f * np.exp(-0.5 * ((wave - l) / s) ** 2)
    norm = np.percentile(prof, 99)
    return prof / norm if norm > 0 else prof


def load_muse_sky_lines(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(wavelength, flux) of the MUSE pipeline's sky-line catalog (`MU_GSKL_*_sky_lines.fits`, air wavelengths)."""
    from astropy.io import fits
    d = fits.getdata(path, 1)
    return np.asarray(d["lambda"], float), np.asarray(d["flux"], float)


def fit_noise_model(residual: np.ndarray, model: np.ndarray, errors: np.ndarray, sky_profile: np.ndarray,
                    outlier_nsigma: float = 10.0) -> dict:
    """Maximum-likelihood per-pixel variance `errors^2 + (kappa*s)^2 + (f*model)^2` for the residuals of a fit.

    `s` is `sky_profile` (sky-line residuals scale with the sky, not the galaxy); `f*model` is a mismatch term
    proportional to the model. `kappa` and `f` maximize the Gaussian likelihood of `residual` (two starts,
    Nelder-Mead on their logs); pixels beyond `outlier_nsigma` of a generous provisional variance are excluded.
    Returns `errors` (the new per-pixel errors), `kappa` (in units of the median model), `f`, and `inflation` (median
    new/old error ratio). On spectra without these systematics both terms go to ~0 and the errors are unchanged."""
    from scipy.optimize import minimize
    res, m = np.asarray(residual, float), np.asarray(model, float)
    err, s = np.asarray(errors, float), np.asarray(sky_profile, float)
    mm = float(np.median(m))
    ok = np.abs(res) < outlier_nsigma * np.sqrt(err ** 2 + (0.3 * s * mm) ** 2 + (0.03 * m) ** 2)

    def nll(p):
        V = err[ok] ** 2 + (np.exp(p[0]) * s[ok]) ** 2 + (np.exp(p[1]) * m[ok]) ** 2
        return np.sum(res[ok] ** 2 / V + np.log(V))

    best = min((minimize(nll, x0, method="Nelder-Mead") for x0 in ([np.log(1e-3 * mm), np.log(1e-3)],
                                                                    [np.log(0.05 * mm), np.log(0.01)])),
               key=lambda r: r.fun)
    kappa, f = np.exp(best.x)
    new = np.sqrt(err ** 2 + (kappa * s) ** 2 + (f * m) ** 2)
    return dict(errors=new, kappa=float(kappa / mm), f=float(f), inflation=float(np.median(new / err)))


def noise_model_errors(A_stack: np.ndarray, y: np.ndarray, errors: np.ndarray, wave_fit: np.ndarray, cfg,
                       sky_profile: np.ndarray, lam: float = 4096.0) -> dict:
    """`fit_noise_model` on the residuals of a preliminary fit: `fit_losvd_robust` at fixed `lam`, `rho=0`, no
    cleaning (outlier pixels are excluded inside `fit_noise_model` instead). `cfg` supplies the grid, penalty and
    continuum settings. Returns `fit_noise_model`'s dict plus `model` (the preliminary fit at every pixel)."""
    import dataclasses
    from .robust_fit import fit_losvd_robust, full_pixel_model
    c0 = dataclasses.replace(cfg, lam_rule="fixed", lam=float(lam), rho=0.0)
    r0 = fit_losvd_robust(A_stack, y, errors, c0, wave_fit=wave_fit, cleaning_method=None)
    m = full_pixel_model(r0, A_stack, wave_fit, c0)
    out = fit_noise_model(np.asarray(y, float) - m, m, errors, sky_profile)
    out["model"] = m
    return out
