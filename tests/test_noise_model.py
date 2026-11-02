"""Isolated verification for kinextract.noise_model, BEFORE this is
wired into any fitting engine (per this project's own standing rule:
verify each new piece in isolation before combining it with anything
else). See noise_model.py's own docstring for why a covariance-based
whitening requires no changes to the fitting engines themselves -- these
tests validate that claim directly, on synthetic data with a known,
controlled covariance.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from kinextract.noise_model import build_ar1_covariance, whiten, estimate_correlation_length, measure_and_whiten


def test_ar1_covariance_at_rho_zero_is_diagonal():
    errors = np.array([1.0, 2.0, 3.0, 0.5])
    Sigma = build_ar1_covariance(errors, rho=0.0)
    np.testing.assert_allclose(Sigma, np.diag(errors ** 2))


def test_ar1_covariance_rejects_out_of_range_rho():
    errors = np.ones(5)
    for bad_rho in (-0.1, 1.0, 1.5):
        try:
            build_ar1_covariance(errors, rho=bad_rho)
            assert False, f"expected ValueError for rho={bad_rho}"
        except ValueError:
            pass


def test_whiten_at_rho_zero_reproduces_old_convention_bit_for_bit():
    rng = np.random.default_rng(0)
    n_pix, n_templates, n_bins = 200, 3, 15
    A_stack = rng.normal(size=(n_templates, n_pix, n_bins))
    y = rng.normal(size=n_pix)
    errors = rng.uniform(0.5, 2.0, n_pix)

    A_w, y_w, L = whiten(A_stack, y, errors, rho=0.0)
    np.testing.assert_array_equal(A_w, A_stack / errors[None, :, None])
    np.testing.assert_array_equal(y_w, y / errors)
    np.testing.assert_array_equal(L, np.diag(errors))


def test_whiten_single_template_2d_input():
    rng = np.random.default_rng(1)
    n_pix, n_bins = 150, 10
    A = rng.normal(size=(n_pix, n_bins))
    y = rng.normal(size=n_pix)
    errors = np.full(n_pix, 0.7)
    A_w, y_w, L = whiten(A, y, errors, rho=0.0)
    assert A_w.shape == (n_pix, n_bins)
    np.testing.assert_allclose(A_w, A / 0.7)


def test_whitening_at_the_true_rho_decorrelates_residuals():
    # Generate many independent draws of AR(1)-correlated noise with a
    # KNOWN Sigma, whiten each with that same true rho, and confirm the
    # whitened residuals are consistent with i.i.d. N(0,1) -- i.e. the
    # Cholesky-whitening trick actually decorrelates, not just rescales.
    n_pix = 300
    errors = np.full(n_pix, 1.0)
    true_rho = 0.4
    Sigma = build_ar1_covariance(errors, true_rho)
    L_true = np.linalg.cholesky(Sigma)

    rng = np.random.default_rng(42)
    n_draws = 400
    lag1_acfs = []
    for _ in range(n_draws):
        noise = L_true @ rng.standard_normal(n_pix)  # true correlated draw
        # whiten against the SAME true rho (not fitting anything -- just
        # checking the whitening operation itself is correct)
        _, y_w, _ = whiten(np.zeros((1, n_pix, 1)), noise, errors, rho=true_rho)
        x = y_w - y_w.mean()
        lag1_acfs.append(np.sum(x[:-1] * x[1:]) / np.sum(x ** 2))

    mean_lag1 = np.mean(lag1_acfs)
    # should be centered on ~0 (residual correlation removed), well within
    # the sampling noise of a lag-1 ACF at n_pix=300 (~1/sqrt(300)=0.058)
    assert abs(mean_lag1) < 0.03, f"whitened residuals still show lag-1 correlation: {mean_lag1:.4f}"


def test_whitened_quadratic_form_is_a_true_chi2():
    # sum(resid_w**2) should behave like a real chi2(n_pix) statistic when
    # residuals are actually drawn from the covariance being whitened
    # against -- validates the Cholesky-whitening trick end to end, not
    # just its decorrelation property.
    n_pix = 100
    errors = np.full(n_pix, 1.0)
    true_rho = 0.3
    Sigma = build_ar1_covariance(errors, true_rho)
    L_true = np.linalg.cholesky(Sigma)

    rng = np.random.default_rng(7)
    n_draws = 2000
    chi2_vals = np.empty(n_draws)
    for i in range(n_draws):
        noise = L_true @ rng.standard_normal(n_pix)
        _, y_w, _ = whiten(np.zeros((1, n_pix, 1)), noise, errors, rho=true_rho)
        chi2_vals[i] = np.sum(y_w ** 2)

    expected_mean, expected_var = n_pix, 2 * n_pix
    assert abs(chi2_vals.mean() - expected_mean) < 3 * np.sqrt(expected_var / n_draws)
    # KS test against the true chi2(n_pix) distribution as a stronger check
    ks_stat, ks_p = stats.kstest(chi2_vals, "chi2", args=(n_pix,))
    assert ks_p > 0.01, f"whitened quadratic form doesn't look like chi2({n_pix}): KS p={ks_p:.4f}"


def test_rho_mismatch_stress_leaves_measurable_residual_correlation():
    # If the ASSUMED rho doesn't match the TRUE rho (a real scenario --
    # rho itself is only ever an estimate), whitening should measurably
    # fail to fully decorrelate -- confirms the test methodology above
    # actually has power to detect a wrong rho, not just always passing.
    n_pix = 300
    errors = np.full(n_pix, 1.0)
    true_rho = 0.5
    assumed_rho = 0.0  # deliberately wrong (the old, pre-this-module assumption)
    Sigma = build_ar1_covariance(errors, true_rho)
    L_true = np.linalg.cholesky(Sigma)

    rng = np.random.default_rng(99)
    n_draws = 300
    lag1_acfs = []
    for _ in range(n_draws):
        noise = L_true @ rng.standard_normal(n_pix)
        _, y_w, _ = whiten(np.zeros((1, n_pix, 1)), noise, errors, rho=assumed_rho)
        x = y_w - y_w.mean()
        lag1_acfs.append(np.sum(x[:-1] * x[1:]) / np.sum(x ** 2))

    mean_lag1 = np.mean(lag1_acfs)
    assert mean_lag1 > 0.3, (
        f"expected a mismatched (too-low) assumed rho to leave clear residual correlation, "
        f"got mean lag-1 ACF={mean_lag1:.4f}"
    )


def test_estimate_correlation_length_recovers_a_known_rho():
    n_pix = 2000
    errors = np.full(n_pix, 1.0)
    true_rho = 0.35
    Sigma = build_ar1_covariance(errors, true_rho)
    L_true = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(5)
    noise = L_true @ rng.standard_normal(n_pix)
    est = estimate_correlation_length(noise)
    assert abs(est - true_rho) < 0.05, f"expected ~{true_rho}, got {est:.3f}"


def test_estimate_correlation_length_on_white_noise_is_near_zero():
    rng = np.random.default_rng(11)
    white = rng.standard_normal(2000)
    est = estimate_correlation_length(white)
    assert est < 0.1


def test_measure_and_whiten_matches_manual_estimate_then_whiten():
    n_pix = 2000
    errors = np.full(n_pix, 1.0)
    true_rho = 0.35
    Sigma = build_ar1_covariance(errors, true_rho)
    L_true = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(5)
    noise = L_true @ rng.standard_normal(n_pix)

    A_stack = np.zeros((1, n_pix, 1))
    y = noise
    A_w, y_w, L, rho = measure_and_whiten(A_stack, y, errors, residuals=noise)

    expected_rho = estimate_correlation_length(noise)
    A_w_expected, y_w_expected, L_expected = whiten(A_stack, y, errors, rho=expected_rho)
    assert rho == expected_rho
    np.testing.assert_array_equal(A_w, A_w_expected)
    np.testing.assert_array_equal(y_w, y_w_expected)
    np.testing.assert_array_equal(L, L_expected)


def test_measure_and_whiten_on_white_noise_measures_a_small_rho():
    rng = np.random.default_rng(11)
    n_pix = 2000
    white = rng.standard_normal(n_pix)
    errors = np.ones(n_pix)
    A_stack = np.zeros((1, n_pix, 1))
    A_w, y_w, L, rho = measure_and_whiten(A_stack, white, errors, residuals=white)
    assert rho < 0.1


def test_fitconfig_rho_defaults_to_zero_and_round_trips():
    """FitConfig.rho is a documentation/carrier field only (see its own
    docstring) -- fit_losvd never reads it directly, the caller must
    still call whiten(..., rho=cfg.rho) themselves. This just confirms
    the field exists, defaults to the independent-pixel case, and a
    measured rho can be stored on the same config a fit already uses.
    """
    from kinextract import FitConfig

    assert FitConfig(continuum_knots=None).rho == 0.0
    cfg = FitConfig(rho=0.42, continuum_knots=None)
    assert cfg.rho == 0.42


# ------------------------------------------------------------------------------------------------ systematics model
from kinextract.noise_model import fit_noise_model, sky_line_profile


def test_sky_line_profile_peaks_at_the_rest_frame_lines_and_is_normalized():
    wave = np.arange(8300.0, 8850.0, 1.25)
    z = 0.01
    prof = sky_line_profile(wave, np.array([8400.0, 8600.0]) * (1 + z), np.array([1.0, 3.0]), z=z, fwhm=2.5 * (1 + z))
    assert abs(wave[np.argmax(prof)] - 8600.0) <= 1.25
    assert np.isclose(np.percentile(prof, 99), 1.0)
    assert prof[np.argmin(np.abs(wave - 8500.0))] < 1e-6


def test_fit_noise_model_recovers_the_injected_sky_and_mismatch_terms():
    rng = np.random.default_rng(0)
    n = 4000
    model = 1.0 + 0.3 * np.sin(np.linspace(0, 40, n))
    err = np.full(n, 0.002)
    s = np.zeros(n); s[rng.choice(n, 400, replace=False)] = rng.uniform(0.2, 1.0, 400)
    kappa, f = 0.05, 0.01
    res = rng.standard_normal(n) * np.sqrt(err ** 2 + (kappa * s) ** 2 + (f * model) ** 2)
    out = fit_noise_model(res, model, err, s)
    assert abs(out["kappa"] * np.median(model) / kappa - 1) < 0.15
    assert abs(out["f"] / f - 1) < 0.1
    assert out["inflation"] > 4


def test_fit_noise_model_leaves_pure_noise_alone():
    rng = np.random.default_rng(1)
    n = 3000
    model = np.ones(n); err = np.full(n, 0.01)
    s = np.zeros(n); s[::30] = 1.0
    out = fit_noise_model(err * rng.standard_normal(n), model, err, s)
    assert out["inflation"] < 1.02
    assert np.all(out["errors"] >= err)
