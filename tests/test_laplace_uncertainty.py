"""Tests for kinextract.laplace_uncertainty: the Laplace posterior at one lam, its parallel driver, and the
`losvd_error_bands` recipe."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kinextract import FitConfig, build_template_stack, gauss_hermite_model, load_emiles_templates
from kinextract.laplace_uncertainty import (
    laplace_covariance_at_lam, laplace_losvd_uncertainty, laplace_losvd_uncertainty_parallel,
    laplace_covariance_at_lam_with_continuum, _sample_curves_and_moments,
)

EMILES_GRID = "src/kinextract/template_data/emiles_muse_grid.npz"
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0


def _mock_case(seed=0, true_v=20.0, true_sigma=90.0, snr=60.0):
    templates, template_wave, names, ages, metals = load_emiles_templates(
        EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06), (14.0, 0.40)])
    mask = (template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)
    wave_fit = template_wave[mask]
    cfg = FitConfig(vmin=true_v - 7 * true_sigma, vmax=true_v + 7 * true_sigma, continuum_knots=None)
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
    return A_stack_w_c, y_w_c, wave_fit, replace(cfg, continuum_knots=4), v_grid, true_C


def test_laplace_covariance_at_lam_returns_expected_shapes():
    A_stack_w, y_w, cfg, v_grid = _mock_case()
    lc = laplace_covariance_at_lam(A_stack_w, y_w, cfg, lam=4096.0)
    n_coef = len(lc["coeffs"])
    assert lc["cov_coeffs"].shape == (n_coef, n_coef)
    assert lc["basis"].shape[1] == n_coef
    assert lc["chi2"] > 0


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


def test_laplace_losvd_uncertainty_runs_and_is_sane():
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=3)
    lap = laplace_losvd_uncertainty(A_stack_w, y_w, cfg, n_samples=60, seed=1)
    assert lap["method"] == "laplace"
    assert lap["lam_used"] == cfg.lam
    assert "C_median" not in lap
    assert lap["b_median"].shape == v_grid.shape
    assert np.all(lap["b_p16"] <= lap["b_p84"] + 1e-9)
    # Should be in the right ballpark, not wildly inflated by divergent-edge
    # samples (the bug this whole rejection mechanism exists to fix).
    assert 0 < lap["sherm_std"] < 50.0
    assert lap["vherm_std"] < 50.0
    assert laplace_losvd_uncertainty(A_stack_w, y_w, cfg, lam=777.0, n_samples=20, seed=1)["lam_used"] == 777.0


def test_laplace_losvd_uncertainty_requires_wave_fit_with_continuum():
    A_stack_w, y_w, cfg, _ = _mock_case()
    with pytest.raises(ValueError, match="wave_fit"):
        laplace_losvd_uncertainty(A_stack_w, y_w, replace(cfg, continuum_knots=4))
    with pytest.raises(ValueError, match="wave_fit"):
        laplace_losvd_uncertainty_parallel([A_stack_w], [y_w], replace(cfg, continuum_knots=4), n_jobs=1)


def test_laplace_losvd_uncertainty_parallel_matches_serial_at_n_jobs_1():
    """At n_jobs=1 (no process pool), per-bin results must be bit-identical
    to calling laplace_losvd_uncertainty in a serial loop with the same
    per-bin seed convention (seed + i), and per-bin `lam` must be honored."""
    A_stack_w0, y_w0, cfg, v_grid = _mock_case(seed=5)
    A_stack_w1, y_w1, _, _ = _mock_case(seed=6)

    results = laplace_losvd_uncertainty_parallel(
        [A_stack_w0, A_stack_w1], [y_w0, y_w1], cfg, lam=[4096.0, 8192.0], n_samples=40, seed=0, n_jobs=1)
    assert len(results) == 2
    expected0 = laplace_losvd_uncertainty(A_stack_w0, y_w0, cfg, n_samples=40, seed=0)
    expected1 = laplace_losvd_uncertainty(A_stack_w1, y_w1, cfg, lam=8192.0, n_samples=40, seed=1)
    np.testing.assert_allclose(results[0]["sherm_std"], expected0["sherm_std"])
    np.testing.assert_allclose(results[1]["sherm_std"], expected1["sherm_std"])
    assert results[1]["lam_used"] == 8192.0


def test_laplace_covariance_at_lam_with_continuum_returns_expected_shapes():
    A_stack_w_c, y_w_c, wave_fit, cfg, v_grid, true_C = _mock_continuum_case()
    lc = laplace_covariance_at_lam_with_continuum(A_stack_w_c, y_w_c, wave_fit, cfg, lam=4096.0)
    n_coef = len(lc["coeffs"])
    n_coef_c = len(lc["continuum_coeffs"])
    assert lc["cov_joint"].shape == (n_coef + n_coef_c, n_coef + n_coef_c)
    assert lc["basis"].shape[1] == n_coef
    assert lc["basis_c"].shape[1] == n_coef_c


def test_laplace_with_continuum_runs_and_is_sane():
    A_stack_w_c, y_w_c, wave_fit, cfg, v_grid, true_C = _mock_continuum_case(seed=2)
    lap = laplace_losvd_uncertainty(A_stack_w_c, y_w_c, cfg, wave_fit=wave_fit, n_samples=150, seed=1)
    assert lap["lam_used"] == 4096.0
    assert lap["b_median"].shape == v_grid.shape
    assert lap["C_median"].shape == wave_fit.shape
    assert np.all(lap["b_p16"] <= lap["b_p84"] + 1e-9)
    assert np.all(lap["C_p16"] <= lap["C_p84"] + 1e-9)
    # The (b-area, C-mean) gauge fix should keep C's spread physically sane --
    # not the wild (mean up to ~15x the point estimate) values measured before
    # compensating C by the same per-draw area factor used to normalize b.
    assert lap["C_samples"].mean(axis=1).max() < 3.0
    assert 0 < lap["sherm_std"] < 50.0


def test_laplace_with_continuum_shape_matches_truth_reasonably():
    """Mean-subtracted shape RMS (the metric continuum.py's own docstring
    recommends -- C's absolute level is not identifiable, see that
    module) should be small on a well-behaved mock, even though C's own
    absolute level need not match the truth's."""
    A_stack_w_c, y_w_c, wave_fit, cfg, v_grid, true_C = _mock_continuum_case(seed=3)
    lap = laplace_losvd_uncertainty(A_stack_w_c, y_w_c, cfg, wave_fit=wave_fit, n_samples=200, seed=0)
    shape_resid = (lap["C_median"] - lap["C_median"].mean()) - (true_C - true_C.mean())
    shape_rms = np.sqrt(np.mean(shape_resid ** 2))
    assert shape_rms < 0.1


def test_laplace_losvd_uncertainty_parallel_handles_mixed_continuum_bins():
    """A mixed batch (one plain bin, one continuum bin) returns the right shape for each."""
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=5)
    A_stack_w_c, y_w_c, wave_fit, cfg_c, _, true_C = _mock_continuum_case(seed=5)
    results = laplace_losvd_uncertainty_parallel(
        [A_stack_w, A_stack_w_c], [y_w, y_w_c], [cfg, cfg_c],
        n_samples=80, seed=0, n_jobs=1, wave_fit=[None, wave_fit])
    assert "C_median" not in results[0]
    assert results[1]["C_median"].shape == wave_fit.shape


def test_laplace_losvd_uncertainty_parallel_clipped_mask_drops_pixels_like_the_fit():
    """`clipped_mask` must drop exactly those whitened rows (and the matching
    wave_fit entries), giving the same answer as slicing by hand -- and a
    corrupted pixel that the robust fit clipped must not pull the Laplace MAP."""
    A_stack_w_c, y_w_c, wave_fit, cfg_c, _, _ = _mock_continuum_case(seed=6)
    n_pix = len(y_w_c)
    bad = np.zeros(n_pix, dtype=bool)
    bad[[40, 41, 200]] = True
    y_bad = y_w_c.copy()
    y_bad[bad] -= 50.0 * np.abs(y_w_c).max()  # outliers far beyond any real feature
    waves_in = [wave_fit]

    masked = laplace_losvd_uncertainty_parallel(
        [A_stack_w_c], [y_bad], [cfg_c], n_samples=60, seed=0, n_jobs=1,
        wave_fit=waves_in, clipped_mask=[bad])[0]
    by_hand = laplace_losvd_uncertainty_parallel(
        [A_stack_w_c[:, ~bad, :]], [y_bad[~bad]], [cfg_c], n_samples=60, seed=0, n_jobs=1,
        wave_fit=[wave_fit[~bad]])[0]
    clean = laplace_losvd_uncertainty_parallel(
        [A_stack_w_c], [y_w_c], [cfg_c], n_samples=60, seed=0, n_jobs=1, wave_fit=[wave_fit])[0]

    np.testing.assert_allclose(masked["b_median"], by_hand["b_median"])
    assert masked["sherm_std"] == by_hand["sherm_std"]
    assert abs(masked["point0"]["sherm"] - clean["point0"]["sherm"]) < 10.0
    assert waves_in[0] is wave_fit and len(waves_in[0]) == n_pix  # caller's list untouched


def test_laplace_losvd_uncertainty_parallel_clipped_mask_rejects_wrong_shape():
    A_stack_w, y_w, cfg, _ = _mock_case(seed=7)
    with pytest.raises(ValueError, match="clipped_mask"):
        laplace_losvd_uncertainty_parallel([A_stack_w], [y_w], cfg, n_jobs=1,
                                            clipped_mask=[np.zeros(3, dtype=bool)])


def test_zero_valid_samples_falls_back_to_point_estimate_instead_of_crashing(monkeypatch):
    """Found running the capstone notebook at real 219-bin scale
    (2026-08-03): a bin so poorly constrained that its posterior-predictive
    sampling produced literally zero valid draws (every draw hit the
    divergent-edge rejection filter even after the full retry budget) --
    np.percentile on the resulting empty array raised IndexError, crashing
    the entire parallel batch over one bad bin. Forces that exact condition
    via monkeypatch and confirms both paths degrade to a point-estimate
    fallback instead of crashing.
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
    lap = laplace_losvd_uncertainty(A_stack_w, y_w, cfg, n_samples=30, seed=0)
    assert lap["sherm_std"] == 0.0
    assert lap["b_median"].shape == v_grid.shape

    monkeypatch.setattr(lu, "_sample_joint_curves_and_moments", _always_empty_joint)
    A_stack_w_c, y_w_c, wave_fit, cfg_c, v_grid_c, true_C = _mock_continuum_case(seed=6)
    lapc = laplace_losvd_uncertainty(A_stack_w_c, y_w_c, cfg_c, wave_fit=wave_fit, n_samples=30, seed=0)
    assert lapc["sherm_std"] == 0.0
    assert lapc["C_median"].shape == wave_fit.shape


def test_losvd_error_bands_brackets_the_fit_and_widens_with_mismatch():
    """The real-data recipe end to end on a mock with a continuum distortion: the bands are re-centered on the
    robust fit, unit area, with finite positive moment errors; a model-proportional systematic in the data
    raises the fitted mismatch term and widens the bands."""
    from kinextract import fit_losvd_robust, losvd_error_bands, losvd_error_bands_parallel
    templates, template_wave, *_ = load_emiles_templates(EMILES_GRID, select=[(1.0, -0.66), (5.0, 0.06),
                                                                              (14.0, 0.40)])
    wave = template_wave[(template_wave >= WAVEFITMIN) & (template_wave <= WAVEFITMAX)]
    cfg = FitConfig(vmin=-630.0, vmax=630.0)
    v = cfg.build_velocity_grid()
    b_true = np.exp(-0.5 * (v / 90.0) ** 2)
    A = build_template_stack(wave, template_wave, templates, v)
    model = sum(wk * (A[k] @ b_true) for k, wk in enumerate((0.3, 0.4, 0.3)))
    rng = np.random.default_rng(3)
    err = np.full(len(wave), np.median(model) / 80.0)
    out = {}
    for name, f in (("clean", 0.0), ("mismatch", 0.02)):
        flux = model * (1 + f * np.sin(wave / 3.0)) + rng.normal(0.0, err)
        fit = fit_losvd_robust(A, flux, err, cfg, wave_fit=wave)
        out[name] = (fit, losvd_error_bands(A, flux, err, wave, fit, cfg, n_samples=80, n_moment_draws=40))
    fit, bands = out["clean"]
    unit = fit["b"] / np.trapezoid(fit["b"], v)
    core = unit > 0.1 * unit.max()
    assert np.allclose(np.trapezoid(bands["b_samples"], v, axis=1), 1.0)
    assert np.allclose(bands["b_median"][core], unit[core], rtol=0.05)
    assert np.all(bands["b_p16"][core] <= unit[core] * 1.01) and np.all(bands["b_p84"][core] >= unit[core] * 0.99)
    assert all(np.isfinite(bands[k]) and bands[k] > 0 for k in ("vherm_err", "sherm_err", "h3_err", "h4_err"))
    assert bands["lam_used"] == fit["lam_used"]
    assert out["mismatch"][1]["f"] > bands["f"]
    assert out["mismatch"][1]["sherm_err"] > bands["sherm_err"]
    par = losvd_error_bands_parallel([A], [model], [err], [wave], [fit], cfg, n_jobs=1, n_samples=20,
                                     n_moment_draws=10)
    assert len(par) == 1 and par[0]["b_samples"].shape == (20, len(v))


def test_recenter_samples_moves_the_median_onto_the_point_estimate_and_keeps_unit_area():
    from kinextract import recenter_samples
    rng = np.random.default_rng(0)
    v = np.linspace(-1000, 1000, 401)
    point = np.exp(-0.5 * ((v - 50) / 200) ** 2)
    other = np.exp(-0.5 * ((v + 20) / 250) ** 2)
    draws = other[None, :] * np.exp(0.05 * rng.standard_normal((300, 1)) + 0.02 * rng.standard_normal((300, v.size)))
    out = recenter_samples(draws, v, point)
    unit = point / np.trapezoid(point, v)
    core = unit > 0.05 * unit.max()
    assert np.allclose(out["b_median"][core], unit[core], rtol=0.03)
    assert np.allclose(np.trapezoid(out["b_samples"], v, axis=1), 1.0)
    unit_draws = draws / np.trapezoid(draws, v, axis=1)[:, None]     # unit-area LOSVDs carry no overall scale
    rel_before = np.std(unit_draws / np.median(unit_draws, 0), axis=0)[core]
    rel_after = np.std(out["b_samples"] / out["b_median"], axis=0)[core]
    assert np.allclose(rel_after, rel_before, rtol=0.2)
