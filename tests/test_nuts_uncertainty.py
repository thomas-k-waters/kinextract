"""Tests for kinextract.nuts_uncertainty -- the opt-in NUTS reference mode
(Phase 1c of dev_notes/V1_HARDENING_ROADMAP.md). Skipped entirely if the
optional `numpyro` dependency (the `nuts` extra) isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")

from kinextract import FitConfig, build_template_stack, gauss_hermite_model, load_emiles_templates
from kinextract.laplace_uncertainty import quick_laplace_uncertainty
from kinextract.nuts_uncertainty import nuts_losvd_uncertainty

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


def test_nuts_losvd_uncertainty_runs_and_is_sane():
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=1)
    nuts = nuts_losvd_uncertainty(A_stack_w, y_w, cfg, n_samples=150, n_warmup=100, seed=0)
    assert nuts["method"] == "nuts"
    assert nuts["b_samples"].shape[0] == 150
    assert nuts["b_median"].shape == v_grid.shape
    assert nuts["sherm_std"] > 0
    assert np.all(nuts["b_p16"] <= nuts["b_p84"] + 1e-9)
    # NUTS should mostly not hit divergences on this well-behaved mock.
    assert nuts["n_divergences"] < 0.05 * 150


def test_nuts_losvd_uncertainty_agrees_roughly_with_quick_laplace_at_the_same_lam():
    """Not a tight numerical match (Gaussian approximation vs. the true
    posterior), but should be the same order of magnitude on a
    well-behaved mock -- a basic cross-check that both engines are
    describing the same underlying posterior, per this project's own
    verify-before-trusting culture.
    """
    A_stack_w, y_w, cfg, v_grid = _mock_case(seed=2)
    quick = quick_laplace_uncertainty(A_stack_w, y_w, cfg, n_samples=200, seed=0)
    nuts = nuts_losvd_uncertainty(A_stack_w, y_w, cfg, lam=quick["lam_used"],
                                   n_samples=300, n_warmup=200, seed=0)
    assert 0.2 * quick["sherm_std"] < nuts["sherm_std"] < 5.0 * quick["sherm_std"]


def test_nuts_losvd_uncertainty_rejects_continuum_cofitting():
    A_stack_w, y_w, cfg, v_grid = _mock_case()
    from dataclasses import replace
    cfg_c = replace(cfg, continuum_knots=4)
    with pytest.raises(ValueError, match="continuum"):
        nuts_losvd_uncertainty(A_stack_w, y_w, cfg_c, n_samples=20, n_warmup=20)
