"""Tests for `FitConfig.penalty_order` (third-difference / shrink-toward-
Gaussian penalty option) -- see dev_notes/REGULARIZATION_SHAPE_RESIDUAL_
HANDOFF.md (2026-07-26 status update) for the investigation that
motivated it.

Cheap smoke/sanity tests only; the statistical validation lives in
`dev_notes/shape_residual/` (scripts 01-06 + the investigation log).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kinextract import FitConfig, build_template_stack, build_spline_truth
from kinextract.losvd import build_bspline_basis, difference_matrix
from kinextract.multi_template import fit_losvd


def _mock(sigma=100.0, snr=80.0, seed=0):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= 8400.0) & (wavelength <= 8800.0)
    wave_fit = wavelength[mask]
    template = np.ones(n_pix)
    for cen, depth in zip([8498.02, 8542.09, 8662.14], [0.55, 0.70, 0.65]):
        template -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)
    cfg_kwargs = dict(vmin=-7 * sigma, vmax=7 * sigma)
    cfg = FitConfig(**cfg_kwargs)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sigma, wiggle_scale=0.0, seed=seed)
    A_stack = build_template_stack(wave_fit, wavelength, [template], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    return A_stack / errors[None, :, None], galaxy / errors, cfg_kwargs, b_truth, v_grid


def test_penalty_order_validation():
    with pytest.raises(ValueError, match="penalty_order"):
        FitConfig(penalty_order=4)


def test_fixed_lam2_with_order3_warns():
    with pytest.warns(UserWarning, match="penalty_order=3"):
        FitConfig(penalty_order=3, lam_selection="fixed", lam=2.0)


def test_resolved_lam_grid_per_order():
    assert FitConfig(penalty_order=2).resolved_lam_grid() == (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert FitConfig().resolved_lam_grid() == (1.0, 8.0, 64.0, 512.0, 4096.0, 32768.0)
    # explicit grid always wins, either order
    assert FitConfig(lam_grid=(3.0, 7.0)).resolved_lam_grid() == (3.0, 7.0)


def test_default_is_order3_and_order2_resolves_to_old_default_grid():
    # the 2026-07 default flip: order 3 is the default; the order-2 path
    # resolves to exactly the tuple that used to be the field default, so
    # penalty_order=2 reproduces pre-flip aic_local behavior.
    assert FitConfig().penalty_order == 3
    assert FitConfig().lam_grid is None
    assert FitConfig(penalty_order=2).resolved_lam_grid() == (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def test_third_difference_operator_annihilates_quadratic_interior():
    # D3's null space (interior rows) contains quadratic coefficient
    # sequences -- the property the shrink-toward-Gaussian behavior rests on.
    D3 = difference_matrix(12, order=3)
    i = np.arange(12, dtype=float)
    quad = 1.7 - 0.3 * i - 0.05 * i ** 2
    assert np.allclose(D3 @ quad, 0.0, atol=1e-12)


def test_parsimony_delta_prefers_largest_indistinguishable_lam(monkeypatch):
    # pure selection-logic test: stub out the per-lam fits with a flat
    # criterion (the sigma~20 ill-posed regime this rule exists for) and
    # check the tiebreak resolves to the largest lam within delta.
    from kinextract import aic_regularization as ar

    fake = {0.5: (5.0, 100.0), 2.0: (4.5, 100.4), 8.0: (4.0, 101.0), 32.0: (3.5, 103.5)}

    # Fake fit has no real "coeffs"/"losvd_objective_gap" keys, so this stub can't
    # support warm_start_lam_grid's own coeffs-chaining or winner-confirmation logic
    # (unrelated to what this test checks -- pure tiebreak-selection logic) --
    # disable it explicitly rather than let the fake silently misrepresent it.
    def fake_m_eff_local(A, y, cfg, lam, initial_coeffs=None):
        m, chi2 = fake[lam]
        return m, chi2, None

    monkeypatch.setattr(ar, "estimate_m_eff_local", fake_m_eff_local)
    # criteria: 110.0, 109.4, 109.0 (min), 110.5
    res_argmin = ar.select_lambda_aic_local(None, None, None, [0.5, 2.0, 8.0, 32.0],
                                            warm_start_lam_grid=False)
    assert res_argmin.best_lam == 8.0
    res_tb = ar.select_lambda_aic_local(None, None, None, [0.5, 2.0, 8.0, 32.0],
                                        parsimony_delta=2.0, warm_start_lam_grid=False)
    assert res_tb.best_lam == 32.0  # within 2.0 of the min -> most regularized wins
    res_tb_tight = ar.select_lambda_aic_local(None, None, None, [0.5, 2.0, 8.0, 32.0],
                                              parsimony_delta=1.0, warm_start_lam_grid=False)
    assert res_tb_tight.best_lam == 8.0  # 32.0 is 1.5 above the min -> excluded


def test_min_resolvable_sigma_guard_excludes_spike_candidates(monkeypatch):
    # selection-logic test: candidates whose fit claims sub-knot-scale
    # sigma are excluded even when their criterion "wins" outright.
    from kinextract import aic_regularization as ar

    cfg = FitConfig(vmin=-140.0, vmax=140.0)  # narrow grid, floor-bound knots
    v_grid = cfg.build_velocity_grid()
    dv_knot = (cfg.vmax - cfg.vmin) / (cfg.resolved_n_interior_knots() + 1)

    def make_b(sigma):
        b = np.exp(-0.5 * (v_grid / sigma) ** 2)
        return b / b.sum()

    # lam=1: a spike (sigma ~ dv/4) with a MUCH better criterion;
    # lam=8/64: healthy fits with slightly worse chi2.
    fake = {1.0: (5.0, 80.0, make_b(dv_knot / 4)),
            8.0: (4.0, 100.0, make_b(50.0)),
            64.0: (3.5, 100.8, make_b(50.0))}

    # Fake fit has no real "coeffs"/"losvd_objective_gap" keys, so this stub can't
    # support warm_start_lam_grid's own coeffs-chaining or winner-confirmation logic
    # (unrelated to what this test checks -- pure candidate-exclusion logic) --
    # disable it explicitly rather than let the fake silently misrepresent it.
    def fake_m_eff_local(A, y, cfg_, lam, initial_coeffs=None):
        m, chi2, b = fake[lam]
        return m, chi2, {"b": b}

    monkeypatch.setattr(ar, "estimate_m_eff_local", fake_m_eff_local)
    # without the guard, the spike wins outright (criterion 90 vs 108/107.8)
    assert ar.select_lambda_aic_local(None, None, cfg, [1.0, 8.0, 64.0],
                                      warm_start_lam_grid=False).best_lam == 1.0
    # with the guard, the spike is excluded; tiebreak then picks the
    # largest lam among the healthy near-tie (108.0 vs 107.8)
    res = ar.select_lambda_aic_local(None, None, cfg, [1.0, 8.0, 64.0],
                                     parsimony_delta=2.0, min_resolvable_sigma_frac=0.5,
                                     warm_start_lam_grid=False)
    assert res.best_lam == 64.0


def test_fit_losvd_penalty_order3_smoke():
    A_stack_w, y_w, cfg_kwargs, b_truth, v_grid = _mock()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg3 = FitConfig(**cfg_kwargs)  # penalty_order=3 is now the default
    result = fit_losvd(A_stack_w, y_w, cfg3)
    assert result["b"].shape == v_grid.shape
    assert np.all(np.isfinite(result["b"]))
    assert result["lam_used"] in cfg3.resolved_lam_grid()
    # recovered curve should be a broadly sane LOSVD (unit-normalizable,
    # non-degenerate width)
    b_n = result["b"] / np.trapezoid(np.maximum(result["b"], 0.0), v_grid)
    peak = b_n.max()
    core = b_truth / np.trapezoid(b_truth, v_grid)
    resid = np.abs(b_n - core) / core.max()
    assert resid[core > 0.01 * core.max()].max() < 0.30  # loose smoke bound
