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

from kinextract.errors import _active_bound_mask, _losvd_moments, _projected_gradient


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


def test_summarize_gh_center_always_equals_map_not_a_bootstrap_derived_value(real_muse_fit):
    """summarize()'s "recommended center" (gh_center_recommended,
    b_center_recommended, moments_center_recommended) must always be the
    MAP fit's own values, never a bootstrap-derived alternative -- matching
    the original Fortran pipeline's convention (see _print_summary's
    "Gauss-Hermite moments (MAP, consistent with pallmc.f)") and avoiding a
    second, different "correct" central estimate for the same quantity.
    bootstrap_result should only ever affect the *_err/_lo/_hi (spread)
    fields, never the center.

    Two earlier versions of this code got this wrong in different ways:
    (1) fitting GH to the pointwise-averaged LOSVD *curve* across replicates
    (a real bug -- averaging smears out each replicate's own peak
    position/width jitter, inflating the reported sigma), and (2) using the
    median of each replicate's own GH fit (not a bug, but still a second
    central estimate that can legitimately drift away from the MAP fit if
    the bootstrap ensemble itself is shifted -- confirmed happening on real
    data, and confusing/inconsistent with the "MAP fit looked excellent"
    diagnostic plot right next to it). This test constructs a synthetic
    bootstrap_result whose replicate-derived values are deliberately very
    different from the MAP fit, and checks summarize() ignores them for the
    center and only uses the MAP fit.
    """
    from kinextract.errors import LOSVDErrorEstimator

    fit, cfg = real_muse_fit
    est = LOSVDErrorEstimator(fit, cfg)

    xl = est.xl
    n = len(xl)
    # Bootstrap replicates deliberately centered far from the MAP LOSVD, so
    # any leakage of a bootstrap-derived value into the "center" fields
    # would be caught.
    rng = np.random.default_rng(0)
    b_samples = np.abs(rng.normal(loc=est.b_map * 3.0 + 1.0, scale=0.1, size=(5, n)))

    bootstrap_result = {
        "b_samples": b_samples,
        "b_err": np.std(b_samples, axis=0, ddof=1),
        "b_lo": np.percentile(b_samples, 16, axis=0),
        "b_hi": np.percentile(b_samples, 84, axis=0),
        "gh_err": {"gh_vherm": 1.0, "gh_sherm": 1.0, "gh_h3": 0.01, "gh_h4": 0.01},
        "gh_lo": {}, "gh_hi": {},
        "gh_med": {"gh_vherm": 9999.0, "gh_sherm": 9999.0, "gh_h3": 9.0, "gh_h4": 9.0},
        "moments_samples": {
            "v": np.full(5, 9999.0), "sigma": np.full(5, 9999.0),
        },
        "moments_err": {"v": 1.0, "sigma": 1.0},
        "moments_lo": {}, "moments_hi": {},
        "n_success": 5, "n_failed": 0, "confidence": 0.68,
    }

    summary = est.summarize(bootstrap_result=bootstrap_result)

    assert np.array_equal(summary["b_center_recommended"], est.b_map)
    assert summary["gh_center_recommended"] is summary["gh_map"]
    mv_map, ms_map = _losvd_moments(est.xl, est.b_map)
    assert summary["moments_center_recommended"]["v"] == pytest.approx(mv_map)
    assert summary["moments_center_recommended"]["sigma"] == pytest.approx(ms_map)
    # The spread fields, in contrast, SHOULD come straight from the
    # (deliberately weird) bootstrap_result -- summarize() isn't supposed
    # to touch those.
    assert summary["gh_err_recommended"]["gh_sherm"] == 1.0


def test_bias_correction_warns_when_lsf_makes_it_actively_harmful(real_muse_fit):
    """bias_correction()'s own docstring documents it as actively harmful
    when recovered sigma is comparable to or below the instrument's LSF
    width (amplifies noise catastrophically; can produce a LARGER bias than
    the uncorrected MAP estimate). This must never be a silent trap: when
    data_fwhm_A/template_fwhm_A place the fit in that regime, a
    RuntimeWarning must fire before the (otherwise unguarded) correction runs.
    """
    import copy

    from kinextract import LOSVDErrorEstimator

    fit, cfg = real_muse_fit
    cfg = copy.deepcopy(cfg)
    # Deliberately implausible, huge LSF FWHM to force the "recovered sigma
    # << LSF sigma" harmful regime regardless of this fixture's own sigma.
    cfg.data_fwhm_A = 500.0
    cfg.template_fwhm_A = 500.0

    est = LOSVDErrorEstimator(fit, cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.bias_correction()

    messages = [str(w.message) for w in caught]
    assert any("actively harmful" in m for m in messages), (
        f"expected a RuntimeWarning citing the documented actively-harmful regime; got: {messages}"
    )


def test_bias_correction_warns_when_lsf_unknown(real_muse_fit):
    """When data_fwhm_A/template_fwhm_A are not set (the common default),
    the actively-harmful regime from bias_correction()'s docstring can't be
    checked quantitatively -- but the caller must still be warned that the
    check couldn't be performed, rather than staying silent about a
    documented risk.
    """
    from kinextract import LOSVDErrorEstimator

    fit, cfg = real_muse_fit
    assert cfg.data_fwhm_A is None and cfg.template_fwhm_A is None

    est = LOSVDErrorEstimator(fit, cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.bias_correction()

    messages = [str(w.message) for w in caught]
    assert any("can't be checked automatically" in m for m in messages), (
        f"expected a RuntimeWarning about the unmeasurable LSF regime; got: {messages}"
    )
