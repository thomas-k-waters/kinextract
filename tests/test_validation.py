"""Tests for kinextract.mocks and kinextract.validation.

These guard against a real bug found via a comprehensive stress test of
`assess_recovery_bias` shortly after it was added: `true_losvd_on_grid`
originally normalised the injected LOSVD via ``trapz(b, xl) == 1`` (a
continuous-density convention), but the flat parameter vector `b` that
kinextract's own MAP objective actually optimises is constrained by
``objective_map``'s own normalisation penalty to satisfy ``sum(b) == 1``
(a Riemann-sum/histogram-weight convention). On a non-unit-spacing
velocity grid (the typical case), these differ by a factor of the bin
spacing -- feeding a trapz-normalised truth in as if it were a real `b`
vector silently injected a mock with the wrong overall flux level,
corrupting the whole bias measurement.

Also covers the `cfg` mutation side-effect fix (xlam_auto overwrites
cfg.xlam as a side effect; assess_recovery_bias must not leak that back
to the caller's own config object across dozens of replicates).

Note: an earlier version of this module also recomputed a per-mock
``v_center`` to recentre the MAP objective's wing-taper smoothness prior
on a data-driven velocity estimate. That recentering was later found (via
real notebook usage, not just this stress test) to introduce worse bias
than the original zero-centered convention whenever the velocity estimate
itself was imprecise, and was reverted package-wide -- the MAP path now
always uses v_center=0.0, matching the original Fortran implementation.
The corresponding regression test for stale per-mock v_center reuse was
removed along with the feature it was guarding.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kinextract.mocks import true_losvd_on_grid, build_matched_mock
from kinextract.validation import assess_recovery_bias, correct_recovered_losvd


def test_true_losvd_on_grid_normalizes_by_sum_not_trapz():
    # Non-uniform-spacing-equivalent check: use a grid whose spacing is
    # far from 1 km/s/bin, so the sum vs. trapz distinction is not
    # accidentally masked by dx ~ 1.
    xl = np.linspace(-300.0, 300.0, 29)  # dx ~ 21.4 km/s/bin
    b = true_losvd_on_grid(xl, v_true=50.0, sigma_true=80.0, h3=0.05, h4=0.03)
    assert np.sum(b) == pytest.approx(1.0, abs=1e-9)
    # A trapz-normalised version would differ from a sum-normalised one by
    # roughly the bin spacing whenever dx != 1; assert they really do differ
    # here so this test would have caught the original bug.
    trapz_norm = float(np.trapezoid(b, xl))
    assert trapz_norm != pytest.approx(1.0, abs=1e-6)


def test_build_matched_mock_flux_scale_matches_reference_state(real_muse_fit):
    """A matched mock injected at the fit's own recovered (V, sigma) should
    land on the same overall flux scale as the real data -- not off by the
    ~bin-spacing factor the sum-vs-trapz bug introduced.
    """
    fit, cfg = real_muse_fit
    st = fit["state"]
    a_fit = np.asarray(fit["a_map"], float)
    from kinextract.losvd import fit_losvd_gauss_hermite

    gh = fit_losvd_gauss_hermite(st.xl, fit["outputs"]["b"], fit_h3h4=True)
    rng = np.random.default_rng(0)
    g_mock = build_matched_mock(st, a_fit, gh["vherm"], gh["sherm"], rng=rng)

    good = np.isfinite(st.gerr) & (st.gerr > 0.0) & (st.gerr < 1e9)
    real_scale = float(np.median(np.abs(st.g[good])))
    mock_scale = float(np.median(np.abs(g_mock[good])))
    # Same order of magnitude (within a factor of 3) -- the bug produced a
    # mismatch of order the LOSVD bin spacing (tens of km/s/bin), which is
    # far larger than this tolerance.
    ratio = mock_scale / real_scale
    assert 0.3 < ratio < 3.0, f"mock flux scale off by {ratio:.2f}x from the real target's own scale"


def test_assess_recovery_bias_does_not_mutate_caller_cfg(real_muse_fit):
    fit, cfg = real_muse_fit
    cfg = dataclasses.replace(cfg)  # isolate from the session-scoped fixture
    xlam_before = cfg.xlam
    clean_before = cfg.clean
    assess_recovery_bias(
        fit, cfg, v_true_grid=[0.0, 80.0], sigma_true_grid=[60.0, 120.0],
        n_seeds=1, seed0=12345,
    )
    assert cfg.xlam == xlam_before
    assert cfg.clean == clean_before


def test_assess_recovery_bias_uses_zero_centered_map_objective(real_muse_fit):
    """The MAP path's wing-taper smoothness prior is always zero-centered
    (matching the original Fortran convention -- see numerics.py's
    _compute_smoothness); assess_recovery_bias's replicate FitStates should
    inherit that (v_center left at 0.0), not reintroduce a data-driven
    recentering that was found to introduce worse bias than not recentering
    at all.
    """
    fit, cfg = real_muse_fit
    assert fit["state"].v_center == 0.0
    bias_table = assess_recovery_bias(
        fit, cfg, v_true_grid=[80.0], sigma_true_grid=[90.0], n_seeds=1, seed0=777,
    )
    assert (80.0, 90.0) in bias_table


def test_correct_recovered_losvd_single_grid_point():
    bias_table = {(80.0, 90.0): dict(bias_v=2.0, bias_v_std=1.0, bias_sigma=-3.0, bias_sigma_std=2.0)}
    corrected = correct_recovered_losvd(82.0, 88.0, bias_table)
    assert corrected["v_corrected"] == pytest.approx(80.0)
    assert corrected["sigma_corrected"] == pytest.approx(91.0)
    assert corrected["v_uncertainty_inflation"] == pytest.approx(1.0)


def test_correct_recovered_losvd_extrapolation_is_finite():
    bias_table = {
        (0.0, 45.0): dict(bias_v=1.0, bias_v_std=0.5, bias_sigma=2.0, bias_sigma_std=1.0),
        (80.0, 90.0): dict(bias_v=-1.0, bias_v_std=0.5, bias_sigma=-2.0, bias_sigma_std=1.0),
    }
    corrected = correct_recovered_losvd(5000.0, 5000.0, bias_table)
    assert all(np.isfinite(v) for v in corrected.values())


def test_correct_recovered_losvd_empty_table_raises():
    with pytest.raises(ValueError):
        correct_recovered_losvd(80.0, 90.0, {})
