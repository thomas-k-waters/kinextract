"""Noise-covariance architecture: replaces the implicit "independent
per-pixel Gaussian" assumption every fit in this package has made so far
with an explicit, pluggable covariance model, defaulting to exactly the
old behavior.

**Why this exists.** Every fitting function in this package
(`fit_log_spline_map`, `fit_losvd_basis`, `fit_continuum_block`,
`estimate_m_eff`) receives already-"whitened" `A_w`/`y_w` and just
computes `0.5*sum(resid**2)` -- correct only if per-pixel noise is
independent. Two real-data audits this project has now run (real MUSE
star spectra, two independent diagnostics) found genuine, non-negligible
lag-1 pixel correlation (~0.3-0.5) that survives even after the model is
allowed to detrend for template-incompleteness via a continuum term (see
`continuum.py` and `examples/notebooks/nima_detrend_vs_noise_correlation.py`).
Every chi^2, AIC criterion, and bootstrap variance computed anywhere in
this package inherits whatever's wrong with this assumption.

**The fix requires no changes to the fitting engines themselves.** If
`Sigma = L @ L.T` (Cholesky) and whitening is done as `A_w = L^-1 @ A`,
`y_w = L^-1 @ y` (proper GLS pre-whitening) instead of the old `A/errors`,
`y/errors`, then `sum(resid_w**2) == resid^T @ Sigma^-1 @ resid` exactly,
for free -- `losvd.py`'s spline fit, `multi_template.py`'s NNLS weight
solve, and `continuum.py`'s objective all stay unchanged; they were
already only ever consuming whatever whitened inputs they were handed.
The entire noise-model upgrade lives in *how the whitened inputs are
built*, plus fixing the two places that inject *synthetic* noise
(`bootstrap_losvd`, `aic_regularization.estimate_m_eff`) to respect the
same covariance instead of drawing i.i.d. noise -- see those modules for
that half of the fix.

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
    (`bootstrap_losvd`, `estimate_m_eff`) need `L` to draw correlated
    noise consistent with this same covariance (`L @ rng.standard_normal(n_pix)`
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
