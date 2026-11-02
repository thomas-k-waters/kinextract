"""Tests for the order-3 penalty and `FitConfig.lam_rule` (`multi_template.with_lam_rule`): a fit at `lam`, a refit
at `sigma_scaled_lam(sigma)` for broad LOSVDs, and the edge retry. The statistical validation lives in
`dev_notes/regularization_v2/FINDINGS.md`; these are cheap smoke/behavior tests on a single-template mock with no
continuum (`continuum_knots=None`).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kinextract import FitConfig, build_spline_truth, build_template_stack, sigma_scaled_lam, with_lam_rule
from kinextract.losvd import difference_matrix, edge_amplitude_frac
from kinextract.multi_template import fit_losvd


def _mock(sigma=100.0, snr=80.0, seed=0, nsig=7.0):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= 8400.0) & (wavelength <= 8800.0)
    wave_fit = wavelength[mask]
    template = np.ones(n_pix)
    for cen, depth in zip([8498.02, 8542.09, 8662.14], [0.55, 0.70, 0.65]):
        template -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)
    cfg_kwargs = dict(vmin=-nsig * sigma, vmax=nsig * sigma, continuum_knots=None)
    v_grid = FitConfig(**cfg_kwargs).build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sigma, wiggle_scale=0.0, seed=seed)
    A_stack = build_template_stack(wave_fit, wavelength, [template], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    return A_stack / errors[None, :, None], galaxy / errors, cfg_kwargs, b_truth, v_grid


def test_third_difference_operator_annihilates_quadratic_interior():
    # D3's null space (interior rows) contains quadratic coefficient
    # sequences -- the property the shrink-toward-Gaussian behavior rests on.
    D3 = difference_matrix(12, order=3)
    i = np.arange(12, dtype=float)
    quad = 1.7 - 0.3 * i - 0.05 * i ** 2
    assert np.allclose(D3 @ quad, 0.0, atol=1e-12)


def test_config_validates_lam_rule_and_retry_factors():
    with pytest.raises(ValueError, match="lam_rule"):
        FitConfig(lam_rule="aic_local")
    with pytest.raises(ValueError, match="edge_retry_factors"):
        FitConfig(edge_retry_factors=(0.5,))
    cfg = FitConfig()
    assert (cfg.lam, cfg.lam_rule, cfg.continuum_knots) == (4096.0, "sigma_scaled", 4)


def test_narrow_losvd_stays_at_the_base_lam():
    A_w, y_w, kw, b_truth, v_grid = _mock(sigma=100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_losvd(A_w, y_w, FitConfig(**kw))
    assert result["lam_used"] == 4096.0 and result["lam_tried"] == [4096.0]
    core = b_truth / np.trapezoid(b_truth, v_grid)
    b_n = result["b"] / np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    resid = np.abs(b_n - core) / core.max()
    assert resid[core > 0.01 * core.max()].max() < 0.30  # loose smoke bound


def test_broad_losvd_is_refit_at_the_sigma_scaled_lam():
    A_w, y_w, kw, _, v_grid = _mock(sigma=500.0, snr=150.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_losvd(A_w, y_w, FitConfig(**kw))
        first = fit_losvd(A_w, y_w, FitConfig(**kw, lam_rule="fixed"))
    from kinextract import compute_losvd_moments
    sigma0 = compute_losvd_moments(v_grid, first["b"], fit_h3h4=True)["sherm"]
    assert sigma0 > 350.0
    assert result["lam_tried"][:2] == [4096.0, sigma_scaled_lam(sigma0)]
    assert result["lam_used"] >= sigma_scaled_lam(sigma0) > 4096.0


def test_fixed_rule_fits_once_at_exactly_lam():
    A_w, y_w, kw, _, _ = _mock(sigma=500.0, snr=150.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_losvd(A_w, y_w, FitConfig(**kw, lam_rule="fixed", lam=777.0))
    assert result["lam_used"] == 777.0 and result["lam_tried"] == [777.0]


def test_edge_retry_raises_lam_until_the_fit_leaves_the_grid_edge():
    """`with_lam_rule` with a stand-in fit: at the edge below lam 3e4, contained above it."""
    v = FitConfig(continuum_knots=None).build_velocity_grid()
    edge_b, good_b = np.ones_like(v), np.exp(-0.5 * (v / 100.0) ** 2)
    calls = []

    def fit_at(c):
        assert c.lam_rule == "fixed"
        calls.append(c.lam)
        return dict(b=good_b if c.lam > 3e4 else edge_b)

    out = with_lam_rule(fit_at, FitConfig(continuum_knots=None, edge_retry_factors=(2.0, 4.0, 4.0)))
    assert calls == [4096.0, 8192.0, 32768.0]
    assert out["lam_used"] == 32768.0 and edge_amplitude_frac(out["b"]) < 0.01
    calls.clear()
    out = with_lam_rule(fit_at, FitConfig(continuum_knots=None, edge_retry_factors=()))
    assert calls == [4096.0] and out["lam_used"] == 4096.0
