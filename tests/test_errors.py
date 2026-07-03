"""Tests for the Laplace covariance active-set conditioning and convergence
diagnostic in errors.py.

These guard against a real bug found in review: `laplace_covariance` used
to silently clip negative posterior variances to exactly 0.0
(np.maximum(diag, 0.0)), which is indistinguishable from a genuinely
well-determined parameter with tiny uncertainty. In practice this happened
whenever the joint Hessian (LOSVD bins + template weights + continuum
terms) was not positive-definite -- most commonly because some LOSVD bins
were pinned at their optimizer lower bound (1e-6), whose one-sided local
curvature has no valid symmetric-Gaussian interpretation and corrupts the
marginal variance of nearby *free* bins when the whole matrix is inverted
jointly. The fix conditions the Laplace covariance on the optimizer's
active set (excluding pinned parameters from the matrix that gets
inverted) and reports genuine numerical anomalies as NaN instead of a
falsely precise zero, plus a convergence diagnostic based on the
KKT-projected gradient at the reported MAP solution.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kinextract.errors import _active_bound_mask, _projected_gradient


def test_active_bound_mask_detects_lower_and_upper():
    a = np.array([1e-6, 0.5, 1.0, 0.3])
    bounds = [(1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0), (1e-6, 1.0)]
    mask = _active_bound_mask(a, bounds)
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_active_bound_mask_no_false_positive_near_but_not_at_bound():
    a = np.array([0.1, 0.9])
    bounds = [(1e-6, 1.0), (1e-6, 1.0)]
    mask = _active_bound_mask(a, bounds)
    np.testing.assert_array_equal(mask, [False, False])


def test_projected_gradient_zeros_infeasible_direction_at_lower_bound():
    # At the lower bound, a positive gradient (wants to decrease further,
    # into the infeasible region) is a valid constrained optimum and should
    # be zeroed by the projection.
    a = np.array([1e-6, 0.5])
    grad = np.array([5.0, -0.5])
    bounds = [(1e-6, 1.0), (1e-6, 1.0)]
    proj = _projected_gradient(grad, a, bounds)
    assert proj[0] == 0.0
    assert proj[1] == -0.5


def test_projected_gradient_keeps_wrong_signed_gradient_at_bound():
    # A NEGATIVE gradient at the lower bound wants to decrease further,
    # which is fine, but a bound "pinned" with the WRONG sign gradient
    # (pointing back into the feasible region) means the point is not
    # actually a valid constrained optimum there -- it should be kept.
    a = np.array([1e-6])
    grad = np.array([-3.0])  # wants to increase b, but b is pinned at lower bound
    bounds = [(1e-6, 1.0)]
    proj = _projected_gradient(grad, a, bounds)
    assert proj[0] == -3.0


def test_projected_gradient_zeros_infeasible_direction_at_upper_bound():
    a = np.array([1.0])
    grad = np.array([-2.0])  # wants to increase further past the upper bound
    bounds = [(1e-6, 1.0)]
    proj = _projected_gradient(grad, a, bounds)
    assert proj[0] == 0.0


@pytest.mark.slow
def test_laplace_covariance_no_silent_zero_for_free_bins(real_muse_fit):
    """End-to-end: pinned bins get 0, free bins get smooth nonzero errors,
    and negative-variance anomalies (if any) are NaN, never a bare zero
    indistinguishable from a well-determined parameter.

    Uses the real_muse_fit fixture (see conftest.py) if bundled example
    data is available; skipped otherwise.
    """
    from kinextract import LOSVDErrorEstimator

    fit, cfg = real_muse_fit
    st = fit["state"]

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        est = LOSVDErrorEstimator(fit, cfg)
        laplace = est.laplace_covariance()

    b_err = laplace["b_err"]
    pinned = laplace["pinned_mask_b"]

    # Pinned bins are exactly zero (the documented "fixed, no variance"
    # convention), never NaN.
    assert np.all(b_err[pinned] == 0.0)

    # Free bins must not silently collapse to exactly zero -- either a
    # real positive error, or an honestly-reported NaN, but not 0.0.
    free_err = b_err[~pinned]
    assert not np.any(free_err == 0.0), (
        "a free (non-pinned) LOSVD bin reported exactly zero Laplace "
        "error -- this is the bug being regression-tested; it should be "
        "a positive value or NaN, never a bare 0.0"
    )
