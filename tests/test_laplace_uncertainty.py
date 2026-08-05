"""Tests for kinextract.laplace_uncertainty (the new primary uncertainty
engine, Phase 1 of dev_notes/V1_HARDENING_ROADMAP.md)."""
from __future__ import annotations

import numpy as np
import pytest

from kinextract import FitConfig, build_template_stack, gauss_hermite_model, load_emiles_templates
from kinextract.laplace_uncertainty import (
    laplace_covariance_at_lam, laplace_losvd_uncertainty, quick_laplace_uncertainty,
    laplace_losvd_uncertainty_parallel, laplace_covariance_at_lam_with_continuum,
    laplace_continuum_uncertainty, _sample_curves_and_moments,
)

EMILES_GRID = "src/kinextract/template_data/emiles_muse_grid.npz"
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0


def _mock_case(seed=0, true_v=20.0, true_sigma=90.0, snr=60.0):
    templates, template_wave, names, ages, metals = load_emiles_templates(
        EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)])
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    cfg = FitConfig(vmin=true_v - 7 * true_sigma, vmax=true_v + 7 * true_sigma)
    v_grid = cfg.build_velocity_grid()
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, true_v, true_sigma, 0.02, 0.03), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    w_true = np.array([0.3, 0.4, 0.3])
    galaxy_noiseless = sum(w_true[k] * (A_stack[k] @ b_truth) for k in range(3))
    noise = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise, len(wave_fit))
    errors = np.full(len(wave_fit), noise)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return A_stack_w, y_w, cfg, v_grid


def _mock_continuum_case(seed=0, true_v=0.0, true_sigma=90.0, snr=80.0, amp=0.15):
    """Matches 02_realistic_mock_fit.ipynb's own continuum-cofitting demo
    convention: a half-period sine distortion (nonzero mean), the regime
    that surfaced the (b-area, C-mean) gauge degeneracy this module's
    joint sampler has to handle correctly."""
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=seed, true_v=true_v, true_sigma=true_sigma, snr=snr)
    templates, template_wave, *_ = load_emiles_templates(
        EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)])
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, true_v, true_sigma, 0.02, 0.03), 0.0)
    b_truth /= b_truth.sum()
    A_stack = build_template_stack(wave_fit, template_wave, templates, v_grid)
    w_true = np.array([0.3, 0.4, 0.3])
    galaxy_noiseless = sum(w_true[k] * (A_stack[k] @ b_truth) for k in range(3))
    frac = (wave_fit - wave_fit.min()) / (wave_fit.max() - wave_fit.min())
    true_C = 1.0 + amp * np.sin(np.pi * frac)
    galaxy_distorted = galaxy_noiseless * true_C
    noise = np.median(galaxy_distorted) / snr
    rng = np.random.default_rng(seed + 1000)
    galaxy = galaxy_distorted + rng.normal(0.0, noise, len(wave_fit))
    errors = np.full(len(wave_fit), noise)
    A_stack_w_c = A_stack / errors[None, :, None]
    y_w_c = galaxy / errors
    return A_stack_w_c, y_w_c, wave_fit, cfg, v_grid, true_C


def test_laplace_covariance_at_lam_returns_expected_shapes():
    A_stack_w, y_w, cfg, v_grid = _mock_case()
    lc = laplace_covariance_at_lam(A_stack_w, y_w, cfg, lam=4096.0)
    n_coef = len(lc["coeffs"])
    assert lc["cov_coeffs"].shape == (n_coef, n_coef)
    assert lc["basis"].shape[1] == n_coef
    assert lc["chi2"] > 0
    assert lc["m_eff"] > 0


def test_sample_curves_and_moments_rejects_divergent_edge_draws():
    """At a weakly-regularized lam, naive sampling produces some
    divergent-edge draws (a real, measured failure mode -- see the
    function's own docstring). The rejection filter must return exactly
    n_samples valid draws, all below the edge-amplitude threshold.
    """
    A_stack_w, y_w, cfg, v_grid = _mock_case()
    lc = laplace_covariance_at_lam(A_stack_w, y_w, cfg, lam=1.0)  # weakly regularized
    rng = np.random.default_rng(0)
    b_s, vherm_s, sherm_s, h3_s, h4_s = _sample_curves_and_moments(
        lc["coeffs"], lc["cov_coeffs"], lc["basis"], v_grid, n_samples=30, rng=rng,
        max_edge_amplitude_frac=0.01)
    assert b_s.shape[0] == 30
    edge_frac = np.maximum(b_s[:, 0], b_s[:, -1]) / b_s.max(axis=1)
    assert np.all(edge_frac <= 0.01 + 1e-9)
    # sigma from these filtered samples should be sane, not the wild 300+ km/s
    # values unfiltered draws at this lam produce.
    assert sherm_s.max() < 200.0


def test_quick_laplace_uncertainty_runs_and_is_sane():
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=2)
    q = quick_laplace_uncertainty(A_stack_w, y_w, cfg, n_samples=50, seed=1)
    assert q["method"] == "quick_laplace"
    assert q["sherm_std"] > 0
    assert q["h4_std"] >= 0
    assert q["b_median"].shape == v_grid.shape
    assert q["b_p16"].shape == v_grid.shape
    assert np.all(q["b_p16"] <= q["b_p84"] + 1e-9)


def test_laplace_losvd_uncertainty_runs_and_is_sane():
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=3)
    lap = laplace_losvd_uncertainty(A_stack_w, y_w, cfg, n_samples=60, seed=1)
    assert lap["method"] == "laplace"
    assert lap["lam_used"] in cfg.resolved_lam_grid()
    assert np.isclose(lap["lam_weights"].sum(), 1.0)
    assert lap["sherm_std"] > 0
    assert lap["b_median"].shape == v_grid.shape
    # Should be in the right ballpark, not wildly inflated by divergent-edge
    # samples (the bug this whole rejection mechanism exists to fix).
    assert lap["sherm_std"] < 50.0
    assert lap["vherm_std"] < 50.0


def test_laplace_losvd_uncertainty_agrees_roughly_with_bootstrap():
    """Not a tight numerical match (different methods), but should be the
    same order of magnitude on a well-behaved mock -- this is the basic
    sanity bar before trusting either engine's own coverage claims.
    """
    from kinextract.uncertainty import bootstrap_losvd
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=4)
    lap = laplace_losvd_uncertainty(A_stack_w, y_w, cfg, n_samples=100, seed=0)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=15, seed=0, reselect_lam=False)
    assert 0.2 * boot["sherm_std"] < lap["sherm_std"] < 5.0 * boot["sherm_std"]


def test_laplace_functions_reject_continuum_cofitting():
    A_stack_w, y_w, cfg, v_grid = _mock_case()
    cfg_c = FitConfig(vmin=cfg.vmin, vmax=cfg.vmax, continuum_knots=4)
    with pytest.raises(ValueError, match="continuum"):
        laplace_losvd_uncertainty(A_stack_w, y_w, cfg_c)
    with pytest.raises(ValueError, match="continuum"):
        quick_laplace_uncertainty(A_stack_w, y_w, cfg_c)


def test_laplace_losvd_uncertainty_parallel_matches_serial_at_n_jobs_1():
    """At n_jobs=1 (no process pool), per-bin results must be bit-identical
    to calling laplace_losvd_uncertainty in a serial loop with the same
    per-bin seed convention (seed + i) -- the same guarantee
    bootstrap_losvd_parallel's own docstring makes for its serial path.
    """
    A_stack_w0, y_w0, cfg, v_grid = _mock_case(seed=5)
    A_stack_w1, y_w1, _, _ = _mock_case(seed=6)

    results = laplace_losvd_uncertainty_parallel(
        [A_stack_w0, A_stack_w1], [y_w0, y_w1], cfg, n_samples=40, seed=0, n_jobs=1)
    assert len(results) == 2
    expected0 = laplace_losvd_uncertainty(A_stack_w0, y_w0, cfg, n_samples=40, seed=0)
    expected1 = laplace_losvd_uncertainty(A_stack_w1, y_w1, cfg, n_samples=40, seed=1)
    np.testing.assert_allclose(results[0]["sherm_std"], expected0["sherm_std"])
    np.testing.assert_allclose(results[1]["sherm_std"], expected1["sherm_std"])


def test_laplace_losvd_uncertainty_parallel_requires_wave_fit_for_continuum_bins():
    """The parallel wrapper dispatches continuum-configured bins to
    laplace_continuum_uncertainty (not a blanket rejection like the plain
    functions) -- but that dispatch needs wave_fit, so omitting it for a
    continuum bin should still raise, just for a different reason.
    """
    A_stack_w, y_w, cfg, v_grid = _mock_case()
    from dataclasses import replace
    cfg_c = replace(cfg, continuum_knots=4)
    with pytest.raises(ValueError, match="continuum"):
        laplace_losvd_uncertainty_parallel([A_stack_w], [y_w], cfg_c, n_jobs=1)


def test_laplace_covariance_at_lam_with_continuum_returns_expected_shapes():
    A_stack_w_c, y_w_c, wave_fit, cfg, v_grid, true_C = _mock_continuum_case()
    lc = laplace_covariance_at_lam_with_continuum(A_stack_w_c, y_w_c, wave_fit, cfg, lam=4096.0)
    n_coef = len(lc["coeffs"])
    n_coef_c = len(lc["continuum_coeffs"])
    assert lc["cov_joint"].shape == (n_coef + n_coef_c, n_coef + n_coef_c)
    assert lc["basis"].shape[1] == n_coef
    assert lc["basis_c"].shape[1] == n_coef_c


def test_laplace_continuum_uncertainty_runs_and_is_sane():
    A_stack_w_c, y_w_c, wave_fit, cfg, v_grid, true_C = _mock_continuum_case(seed=2)
    lap = laplace_continuum_uncertainty(A_stack_w_c, y_w_c, wave_fit, cfg, lam=4096.0,
                                         n_samples=150, seed=1)
    assert lap["method"] == "laplace_continuum"
    assert lap["lam_used"] == 4096.0
    assert lap["b_median"].shape == v_grid.shape
    assert lap["C_median"].shape == wave_fit.shape
    assert np.all(lap["b_p16"] <= lap["b_p84"] + 1e-9)
    assert np.all(lap["C_p16"] <= lap["C_p84"] + 1e-9)
    # The (b-area, C-mean) gauge fix should keep C's spread physically sane --
    # not the wild (mean up to ~15x the point estimate) values measured before
    # compensating C by the same per-draw area factor used to normalize b.
    assert lap["C_samples"].mean(axis=1).max() < 3.0
    assert lap["sherm_std"] > 0
    assert lap["sherm_std"] < 50.0


def test_laplace_continuum_uncertainty_shape_matches_truth_reasonably():
    """Mean-subtracted shape RMS (the metric continuum.py's own docstring
    recommends -- C's absolute level is not identifiable, see that
    module) should be small on a well-behaved mock, even though C's own
    absolute level need not match the truth's."""
    A_stack_w_c, y_w_c, wave_fit, cfg, v_grid, true_C = _mock_continuum_case(seed=3)
    lap = laplace_continuum_uncertainty(A_stack_w_c, y_w_c, wave_fit, cfg, lam=4096.0,
                                         n_samples=200, seed=0)
    shape_resid = (lap["C_median"] - lap["C_median"].mean()) - (true_C - true_C.mean())
    shape_rms = np.sqrt(np.mean(shape_resid ** 2))
    assert shape_rms < 0.1


def test_laplace_losvd_uncertainty_parallel_dispatches_continuum_bins_correctly():
    """A mixed batch (one plain bin, one continuum bin) should route each
    to its own method and return the right shape for each."""
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=5)
    A_stack_w_c, y_w_c, wave_fit, cfg_c_base, _, true_C = _mock_continuum_case(seed=5)
    from dataclasses import replace
    cfg_c = replace(cfg_c_base, continuum_knots=4, lam_selection="fixed", lam=4096.0)

    results = laplace_losvd_uncertainty_parallel(
        [A_stack_w, A_stack_w_c], [y_w, y_w_c], [cfg, cfg_c],
        n_samples=80, seed=0, n_jobs=1, wave_fit=[None, wave_fit])
    assert results[0]["method"] == "laplace"
    assert "C_median" not in results[0]
    assert results[1]["method"] == "laplace_continuum"
    assert results[1]["C_median"].shape == wave_fit.shape


def test_quick_laplace_uncertainty_with_fixed_lam_skips_selection():
    """lam_selection='fixed' should use cfg.lam directly, no AIC scan."""
    A_stack_w, y_w, cfg, v_grid = _mock_case()
    from dataclasses import replace
    cfg_fixed = replace(cfg, lam_selection="fixed", lam=4096.0)
    q = quick_laplace_uncertainty(A_stack_w, y_w, cfg_fixed, n_samples=30, seed=0)
    assert q["lam_used"] == 4096.0


def test_zero_valid_samples_falls_back_to_point_estimate_instead_of_crashing(monkeypatch):
    """Found running the capstone notebook at real 219-bin scale
    (2026-08-03): a bin so poorly constrained that its winning lam's own
    posterior-predictive sampling produced literally zero valid draws
    (every draw hit the divergent-edge rejection filter even after the
    full retry budget) -- np.percentile on the resulting empty array
    raised IndexError, crashing the entire parallel batch over one bad
    bin. Forces that exact condition via monkeypatch (reproducing it
    "naturally" needs a specific pathological real spectrum, not
    reliably constructible from a mock) and confirms all three entry
    points degrade to a point-estimate fallback instead of crashing.
    """
    import kinextract.laplace_uncertainty as lu

    def _always_empty(*args, **kwargs):
        n_bins = args[2].shape[0]  # basis
        return np.zeros((0, n_bins)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    def _always_empty_joint(*args, **kwargs):
        n_bins_b = args[4].shape[0]  # basis
        n_bins_c = args[5].shape[0]  # basis_c
        return (np.zeros((0, n_bins_b)), np.zeros((0, n_bins_c)),
                np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

    monkeypatch.setattr(lu, "_sample_curves_and_moments", _always_empty)
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=6)
    q = quick_laplace_uncertainty(A_stack_w, y_w, cfg, n_samples=30, seed=0)
    assert q["sherm_std"] == 0.0
    assert q["b_median"].shape == v_grid.shape

    lap = laplace_losvd_uncertainty(A_stack_w, y_w, cfg, n_samples=30, seed=0)
    assert lap["sherm_std"] == 0.0
    assert lap["b_median"].shape == v_grid.shape

    monkeypatch.setattr(lu, "_sample_joint_curves_and_moments", _always_empty_joint)
    A_stack_w_c, y_w_c, wave_fit, cfg_c, v_grid_c, true_C = _mock_continuum_case(seed=6)
    lapc = laplace_continuum_uncertainty(A_stack_w_c, y_w_c, wave_fit, cfg_c, lam=4096.0,
                                          n_samples=30, seed=0)
    assert lapc["sherm_std"] == 0.0
    assert lapc["C_median"].shape == wave_fit.shape
