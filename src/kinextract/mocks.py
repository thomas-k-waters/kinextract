"""Matched mock-spectrum generation for recovery-bias validation.

Given an already-completed fit (the output of
:func:`kinextract.fitting.run_spectral_fit`), :func:`build_matched_mock`
injects a known LOSVD truth in place of the recovered one and adds noise
consistent with the real target's own per-pixel error array, producing a
synthetic spectrum with the same instrument resolution, template mixture,
continuum, and noise level as the real target -- but a known ground
truth. This is the building block for
:func:`kinextract.validation.assess_recovery_bias`: rather than
reconstructing an instrument/template/continuum model from scratch (as
``benchmarks/mock_spectrum.py`` does for controlled stress-testing, which
is dev/test infrastructure not shipped with the installed package), it
reuses the real fit's own :class:`~kinextract.state.FitState`, so "matched
to this target" is automatic rather than something the caller has to
reconstruct by hand.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .state import FitState
from .losvd import gauss_hermite_losvd_model
from .numerics import evaluate_model_gp


def true_losvd_on_grid(
    xl: np.ndarray, v_true: float, sigma_true: float, h3: float = 0.0, h4: float = 0.0,
) -> np.ndarray:
    """Ground-truth LOSVD histogram on `xl`, normalized so ``sum(b) == 1``.

    Uses kinextract's own :func:`kinextract.losvd.gauss_hermite_losvd_model`
    (legacy Fortran H3/H4 convention) so an injected truth is directly
    comparable to :func:`kinextract.losvd.fit_losvd_gauss_hermite`'s
    recovered moments -- a synthetic truth built from an independent GH
    polynomial basis would make any measured "bias" partly an artifact of
    mismatched normalizations, not a real recovery error.

    Normalizing by ``sum(b)`` (a Riemann-sum/histogram-weight convention),
    not ``trapz(b, xl)`` (a continuous-density convention), matters here:
    the fit's own LOSVD parameter vector ``b`` is a set of per-bin weights
    constrained by the objective's own normalization penalty
    (``0.1 * |sum(b) - 1|``, see :func:`kinextract.numerics.objective_map`)
    to sum to 1, not to integrate to 1 -- on a `xl` grid that isn't unit
    spacing (the typical case), the two conventions differ by a factor of
    the bin spacing and feeding a trapz-normalized truth in as if it were
    an actual `b` vector silently injects a spectrum with the wrong overall
    flux level.
    """
    b = gauss_hermite_losvd_model(xl, amp=1.0, vel=v_true, sig=sigma_true, h3=h3, h4=h4)
    b = np.clip(b, 0.0, None)
    s = float(np.sum(b))
    return b / s if s > 0 else b


def build_matched_mock(
    st: FitState,
    a_fit: np.ndarray,
    v_true: float,
    sigma_true: float,
    h3: float = 0.0,
    h4: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Synthetic spectrum matched to `st`'s instrument/template/continuum/noise.

    Evaluates the forward model (:func:`kinextract.numerics.evaluate_model_gp`)
    at `a_fit` but with the LOSVD entries replaced by a known ground truth
    (`v_true`/`sigma_true`/`h3`/`h4`), so template weights, continuum, and
    amplitude all match the real target being validated. Per-pixel
    Gaussian noise is drawn at the level of `st.gerr` (the real target's
    own error array), reproducing its actual per-pixel S/N structure
    rather than assuming a single flat scalar S/N.

    Parameters
    ----------
    st : FitState
        Reference fit state (instrument grid, template matrix, continuum,
        per-pixel errors) to match. Not modified.
    a_fit : ndarray
        Flat parameter vector from a real fit (e.g. ``fit["a_map"]``);
        only its non-LOSVD entries (template weights, continuum,
        amplitude) are used -- the LOSVD entries are overridden by the
        injected truth.
    v_true, sigma_true, h3, h4 : float
        Ground-truth LOSVD parameters to inject.
    rng : numpy.random.Generator, optional
        Random generator for the noise draw. A fresh, non-reproducible
        default generator is used if not given; pass an explicit seeded
        generator for reproducible mocks.

    Returns
    -------
    ndarray
        Synthetic noisy spectrum, same length and units as `st.g`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    a = np.asarray(a_fit, float).copy()
    a[: st.nl] = true_losvd_on_grid(st.xl, v_true, sigma_true, h3, h4)
    gp_true, *_ = evaluate_model_gp(a, st)
    good = np.isfinite(st.gerr) & (st.gerr > 0.0) & (st.gerr < 1e9)
    noise_sigma = np.where(good, st.gerr, 0.0)
    return gp_true + rng.normal(0.0, 1.0, size=gp_true.shape) * noise_sigma
