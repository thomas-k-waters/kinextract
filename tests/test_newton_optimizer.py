"""The exact-Hessian Newton solver: its closed-form value/gradient/Hessian
have to match JAX autodiff of the objectives the package defines, and on
fits across the stiff order-3 regime it has to converge (gradient at the
minimum near zero) to the right LOSVD.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kinextract import FitConfig, build_spline_truth, build_template_stack, fit_gauss_hermite
from kinextract.continuum import _continuum_fgh, _neg_log_post_continuum
from kinextract.losvd import (_losvd_fgh, _neg_log_post, build_bspline_basis, difference_matrix,
                              fit_log_spline_map)
from kinextract.multi_template import fit_losvd


def _ca_mock(wavemin=4750.0, step=1.25, n_pix=3681, wavefitmin=8400.0, wavefitmax=8800.0):
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    template_flux = np.ones(n_pix)
    for cen, depth in zip([8498.02, 8542.09, 8662.14], [0.55, 0.70, 0.65]):
        template_flux -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)
    return wavelength, wave_fit, template_flux


def _mock_fit_inputs(true_v, true_sig, snr, seed, cfg, continuum_slope=0.0):
    wavelength, wave_fit, template_flux = _ca_mock()
    template_2 = 1.0 - 0.3 * (1.0 - template_flux)  # a weaker-lined second template
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=true_v, sigma=true_sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, [template_flux, template_2], v_grid)
    galaxy = 0.7 * (A_stack[0] @ b_truth) + 0.3 * (A_stack[1] @ b_truth)
    galaxy = galaxy * (1.0 + continuum_slope * (wave_fit - wave_fit.mean()) / np.ptp(wave_fit))
    noise = np.median(galaxy) / snr
    galaxy = galaxy + np.random.default_rng(seed).normal(0.0, noise, len(wave_fit))
    errors = np.full(len(wave_fit), noise)
    return A_stack / errors[None, :, None], galaxy / errors, wave_fit, v_grid


def test_losvd_closed_form_value_gradient_hessian_match_jax():
    rng = np.random.default_rng(0)
    cfg = FitConfig(vmin=-1500.0, vmax=1500.0, lam_rule="fixed", lam=4096.0, continuum_knots=None)
    A_w, y_w, _, v_grid = _mock_fit_inputs(0.0, 250.0, 50.0, 0, cfg)
    A_w = A_w[0]
    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=3)
    D = difference_matrix(n_coef, order=3, dv=3000.0 / (n_knots + 1), dv_ref=cfg.knot_spacing_kms)
    lam_vec = 4096.0 * (1.0 + rng.uniform(0.0, 1.0, D.shape[0]))  # per-row lam, not just a scalar
    c = np.log(1e-3) + 0.3 * rng.normal(size=n_coef)

    f, g, H = _losvd_fgh(c, basis, D, lam_vec, A_w, y_w, A_w.T @ A_w)
    f2, g2, H2 = _losvd_fgh(c, basis, D, lam_vec, A_w, y_w)  # the Gram-free Hessian route
    np.testing.assert_allclose(H2, H, rtol=1e-10, atol=1e-12 * np.abs(H).max())
    args = (jnp.array(basis), jnp.array(D), jnp.array(A_w), jnp.array(y_w), jnp.array(lam_vec))
    f_j = float(_neg_log_post(jnp.array(c), *args))
    g_j = np.asarray(jax.grad(_neg_log_post)(jnp.array(c), *args))
    H_j = np.asarray(jax.hessian(_neg_log_post)(jnp.array(c), *args))
    np.testing.assert_allclose(f, f_j, rtol=1e-10)
    np.testing.assert_allclose(g, g_j, rtol=1e-8, atol=1e-8 * np.abs(g_j).max())
    np.testing.assert_allclose(H, H_j, rtol=1e-8, atol=1e-8 * np.abs(H_j).max())


def test_continuum_closed_form_value_gradient_hessian_match_jax():
    rng = np.random.default_rng(1)
    _, wave_fit, _ = _ca_mock()
    basis, n_coef = build_bspline_basis(wave_fit, n_interior_knots=4, degree=3)
    D2 = difference_matrix(n_coef, order=2)
    M = 50.0 * (1.0 + 0.1 * rng.normal(size=len(wave_fit)))
    y_w = M * 1.05 + rng.normal(size=len(wave_fit))
    c = 0.05 * rng.normal(size=n_coef)

    f, g, H = _continuum_fgh(c, M, y_w, basis, D2, 2.0, 0.1)
    args = (jnp.array(M), jnp.array(y_w), jnp.array(basis), jnp.array(D2), 2.0, 0.1)
    f_j = float(_neg_log_post_continuum(jnp.array(c), *args))
    g_j = np.asarray(jax.grad(_neg_log_post_continuum)(jnp.array(c), *args))
    H_j = np.asarray(jax.hessian(_neg_log_post_continuum)(jnp.array(c), *args))
    np.testing.assert_allclose(f, f_j, rtol=1e-10)
    np.testing.assert_allclose(g, g_j, rtol=1e-8, atol=1e-8 * np.abs(g_j).max())
    np.testing.assert_allclose(H, H_j, rtol=1e-8, atol=1e-8 * np.abs(H_j).max())


@pytest.mark.parametrize("true_sig,lam,snr,seed,continuum_knots", [
    (150.0, 4096.0, 100.0, 0, None),
    (300.0, 65536.0, 30.0, 1, None),   # the stiff regime where float32 objectives used to stall the solver
    (450.0, 4096.0, 40.0, 2, 4),       # NGC 4751-like: wide grid, continuum co-fit
    (250.0, 16384.0, 15.0, 3, 4),
])
def test_newton_converges_to_a_stationary_point_with_sane_moments(true_sig, lam, snr, seed, continuum_knots):
    v_half = 5.0 * true_sig + 100.0
    cfg = FitConfig(vmin=-v_half, vmax=v_half, lam_rule="fixed", lam=lam, continuum_knots=continuum_knots)
    A_w, y_w, wave_fit, v_grid = _mock_fit_inputs(40.0, true_sig, snr, seed, cfg,
                                                  continuum_slope=0.1 if continuum_knots else 0.0)
    res = fit_losvd(A_w, y_w, cfg, wave_fit=wave_fit if continuum_knots else None)
    gh = fit_gauss_hermite(v_grid, res["b"], fit_h3h4=True)
    assert abs(gh["vherm"] - 40.0) < 0.15 * true_sig
    assert abs(gh["sherm"] - true_sig) < 0.25 * true_sig
    # the core solver itself, weights fixed: the returned coefficients are a stationary point
    n_knots = cfg.resolved_n_interior_knots()
    basis, n_coef = build_bspline_basis(v_grid, n_interior_knots=n_knots, degree=3)
    D = difference_matrix(n_coef, order=3, dv=2 * v_half / (n_knots + 1), dv_ref=cfg.knot_spacing_kms)
    A_eff = 0.7 * A_w[0] + 0.3 * A_w[1]
    core = fit_log_spline_map(A_eff, y_w, basis, D, lam)
    f, g, _ = _losvd_fgh(np.asarray(core["coeffs"]), basis, D, np.full(D.shape[0], lam), A_eff, y_w)
    assert core["success"] and np.linalg.norm(g) < 1e-4


def test_newton_keeps_the_multi_start_escape_from_a_known_bad_local_minimum():
    # Same counter-example as test_losvd.test_multi_start_escapes_a_known_bad_local_minimum.
    wavelength, wave_fit, template_flux = _ca_mock()
    true_v, true_sig = 80.0, 240.0
    v_half = abs(true_v) + 5 * true_sig + 50.0
    cfg = FitConfig(vmin=-v_half, vmax=v_half, lam=10.0, lam_rule="fixed", continuum_knots=None)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=true_v, sigma=true_sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / 100.0
    galaxy = galaxy_noiseless + np.random.default_rng(1).normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    result = fit_losvd(A_stack / errors[None, :, None], galaxy / errors, cfg)
    gh = fit_gauss_hermite(v_grid, result["b"], fit_h3h4=True)
    assert abs(gh["vherm"] - true_v) < 10.0
    assert abs(gh["sherm"] - true_sig) < 15.0


def test_capped_trust_exact_subproblem_falls_back_to_the_cauchy_point():
    """Before scipy 1.17 the trust-exact subproblem can loop forever (scipy issue 12513). `_CappedSubproblem` stops
    it after `MAX_CHOLESKY` factorizations and returns the Cauchy point: inside the trust region, along -g, and
    lowering the quadratic model."""
    from kinextract.losvd import _CappedSubproblem

    class _Forced(_CappedSubproblem):
        MAX_CHOLESKY = 0          # fire on the first factorization

    rng = np.random.default_rng(0)
    M = rng.normal(size=(6, 6))
    H = M @ M.T + 0.1 * np.eye(6)
    g = rng.normal(size=6)
    x0 = np.zeros(6)
    sub = _Forced(x0, lambda x: 0.0, lambda x: g, lambda x: H)
    p, hits_boundary = sub.solve(0.5)
    assert np.linalg.norm(p) <= 0.5 + 1e-12
    np.testing.assert_allclose(p / np.linalg.norm(p), -g / np.linalg.norm(g))
    assert g @ p + 0.5 * p @ H @ p < 0
    assert hits_boundary == np.isclose(np.linalg.norm(p), 0.5)
    # the normal path is untouched: the uncapped subclass agrees with scipy's own subproblem
    from scipy.optimize._trustregion_exact import IterativeSubproblem
    p_ref, _ = IterativeSubproblem(x0, lambda x: 0.0, lambda x: g, lambda x: H).solve(0.5)
    p_cap, _ = _CappedSubproblem(x0, lambda x: 0.0, lambda x: g, lambda x: H).solve(0.5)
    np.testing.assert_array_equal(p_cap, p_ref)


def test_capped_trust_exact_subproblem_survives_a_nan_damping_factor():
    """A NaN damping factor (the square root of a negative bound in scipy's bracketing) must give the Cauchy step on
    every LAPACK build -- OpenBLAS factorizes a NaN matrix 'successfully' and scipy's cho_solve then raises."""
    from kinextract.losvd import _CappedSubproblem

    class _NaNStart(_CappedSubproblem):
        def _initial_values(self, tr_radius):
            return np.nan, 0.0, 1.0

    rng = np.random.default_rng(1)
    M = rng.normal(size=(5, 5))
    H = M @ M.T + 0.1 * np.eye(5)
    g = rng.normal(size=5)
    p, _ = _NaNStart(np.zeros(5), lambda x: 0.0, lambda x: g, lambda x: H).solve(0.3)
    assert np.all(np.isfinite(p)) and np.linalg.norm(p) <= 0.3 + 1e-12
    np.testing.assert_allclose(p / np.linalg.norm(p), -g / np.linalg.norm(g))
