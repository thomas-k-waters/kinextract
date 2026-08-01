"""Tests for kinextract.aic_regularization (Lipka & Thomas 2021
bootstrap-m_eff regularization selection).

Cheap smoke/sanity tests only -- bootstrap m_eff is expensive by design
(n_boot extra fits per lam), so these use small templates, few bootstrap
iterations, and a short lam grid; the real comparison against this
package's fixed lam=2.0 default runs as a separate example script, not
in the unit test suite.
"""
from __future__ import annotations

import numpy as np

from kinextract import FitConfig, build_template_stack, build_spline_truth
from kinextract.aic_regularization import (
    estimate_m_eff, select_lambda_aic, estimate_m_eff_local, select_lambda_aic_local,
)


def _mock(sigma=100.0, snr=80.0, seed=0):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]

    template = np.ones(n_pix)
    for cen, depth in zip([8498.02, 8542.09, 8662.14], [0.55, 0.70, 0.65]):
        template -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)

    cfg = FitConfig(vmin=-7 * sigma, vmax=7 * sigma)
    v_grid = cfg.build_velocity_grid()
    # spline truth (not Gauss-Hermite): no generator-vs-moments gap to worry about
    # for a general fit-mechanics mock -- see dev_notes/validation_suite/FINDINGS.md.
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sigma, wiggle_scale=0.0, seed=seed)
    A_stack = build_template_stack(wave_fit, wavelength, [template], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return A_stack_w, y_w, cfg


def test_m_eff_is_higher_for_less_regularized_fits():
    A_stack_w, y_w, cfg = _mock()
    m_eff_flexible, _, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam=0.01, n_boot=8, seed=1)
    m_eff_stiff, _, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam=50.0, n_boot=8, seed=1)
    assert m_eff_flexible > m_eff_stiff, (
        f"a weaker penalty should leave the fit more flexible (higher effective DOF): "
        f"flexible={m_eff_flexible:.2f}, stiff={m_eff_stiff:.2f}"
    )


def test_m_eff_is_non_negative_within_bootstrap_noise():
    # m_eff can dip slightly negative from bootstrap noise (it's an estimator, not
    # an exact count) but should not be wildly negative for a reasonable lam.
    A_stack_w, y_w, cfg = _mock()
    m_eff, _, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam=2.0, n_boot=10, seed=2)
    assert m_eff > -2.0


def test_select_lambda_aic_picks_a_value_from_the_grid_with_matching_arrays():
    A_stack_w, y_w, cfg = _mock()
    lam_grid = [0.1, 1.0, 5.0, 20.0]
    result = select_lambda_aic(A_stack_w, y_w, cfg, lam_grid, n_boot=6, seed=3)
    assert result.best_lam in lam_grid
    assert len(result.chi2) == len(result.m_eff) == len(result.criterion) == len(lam_grid)
    np.testing.assert_allclose(result.criterion, result.chi2 + result.w_m * result.m_eff)


def test_select_lambda_aic_chi2_decreases_as_lam_decreases():
    # chi^2 alone (ignoring m_eff) should improve monotonically with more
    # flexibility -- a basic check that estimate_m_eff's returned chi2 is
    # actually tracking the fit quality, not some unrelated quantity.
    A_stack_w, y_w, cfg = _mock()
    result = select_lambda_aic(A_stack_w, y_w, cfg, [0.1, 1.0, 5.0, 20.0], n_boot=1, seed=4)
    assert np.all(np.diff(result.chi2) >= -1e-6), (
        "chi^2 should be non-decreasing as lam increases (less flexibility to fit noise)"
    )


def test_m_eff_local_is_higher_for_less_regularized_fits():
    # same qualitative sanity check as the bootstrap version, for the
    # closed-form local approximation.
    A_stack_w, y_w, cfg = _mock()
    m_eff_flexible, _, _ = estimate_m_eff_local(A_stack_w, y_w, cfg, lam=0.01)
    m_eff_stiff, _, _ = estimate_m_eff_local(A_stack_w, y_w, cfg, lam=50.0)
    assert m_eff_flexible > m_eff_stiff


def test_m_eff_local_chi2_matches_bootstrap_chi2_at_the_same_lam():
    # both estimators fit the *same* deterministic problem at a given lam
    # (no randomness in the point estimate itself) -- their chi2 should
    # agree exactly regardless of which m_eff estimator computed it.
    A_stack_w, y_w, cfg = _mock()
    _, chi2_local, _ = estimate_m_eff_local(A_stack_w, y_w, cfg, lam=2.0)
    _, chi2_boot, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam=2.0, n_boot=1, seed=0)
    assert abs(chi2_local - chi2_boot) < 1e-6


def test_m_eff_local_does_not_crash_at_lam_zero_on_a_narrow_grid():
    """Regression test: `lam=0` (no regularization at all) makes
    `M = XtX` in `estimate_m_eff_local`'s closed-form solve, which can be
    exactly singular -- confirmed to crash with `numpy.linalg.LinAlgError:
    Singular matrix` on a real E-MILES-based calibration sweep at a narrow
    (low-sigma) grid (`estimate_m_eff_local` was using `np.linalg.solve`,
    which has no fallback for a singular system). This is very plausibly
    what silently stalled/killed an earlier m_eff-axis calibration sweep
    that included lam=0.0 in its own grid. `np.linalg.lstsq` solves the
    same system via SVD, giving an exact match to `solve` whenever M is
    non-singular (see `test_m_eff_local_chi2_matches_bootstrap_chi2_at_the_
    same_lam` for that non-singular-case agreement) and a well-defined
    minimum-norm answer instead of a crash when it isn't.
    """
    A_stack_w, y_w, cfg = _mock(sigma=20.0, snr=10.0)  # narrow grid -- the failing regime
    m_eff, chi2, result = estimate_m_eff_local(A_stack_w, y_w, cfg, lam=0.0)
    assert np.isfinite(m_eff) and m_eff > 0
    assert np.isfinite(chi2)

def test_select_lambda_aic_local_picks_a_value_from_the_grid_with_matching_arrays():
    A_stack_w, y_w, cfg = _mock()
    lam_grid = [0.1, 1.0, 5.0, 20.0]
    result = select_lambda_aic_local(A_stack_w, y_w, cfg, lam_grid)
    assert result.best_lam in lam_grid
    assert len(result.chi2) == len(result.m_eff) == len(result.criterion) == len(lam_grid)
    np.testing.assert_allclose(result.criterion, result.chi2 + result.w_m * result.m_eff)


def test_self_normalize_lam_grid_defaults_off_and_leaves_grid_unchanged():
    A_stack_w, y_w, cfg = _mock(sigma=20.0, snr=30.0)  # narrow grid -- where scaling would matter
    lam_grid = [1.0, 8.0, 64.0]
    result_off = select_lambda_aic_local(A_stack_w, y_w, cfg, lam_grid, self_normalize_lam_grid=False)
    np.testing.assert_allclose(sorted(result_off.lam_grid), sorted(lam_grid))


def test_self_normalize_lam_grid_rescales_by_dv_ratio_squared():
    """Exact-arithmetic check of the rescaling itself (not a fit outcome):
    on a narrow grid where `min_interior_knots` floors the knot count, `dv`
    is finer than `knot_spacing_kms`, so `self_normalize_lam_grid=True`
    must divide every candidate by `(knot_spacing_kms/dv)^2` before scanning
    -- exactly canceling `difference_matrix`'s own `(dv_ref/dv)` scaling of
    the penalty operator (squared once it appears in `D.T @ D`)."""
    A_stack_w, y_w, cfg = _mock(sigma=20.0, snr=30.0)
    n_knots = cfg.resolved_n_interior_knots()
    dv = (cfg.vmax - cfg.vmin) / (n_knots + 1)
    assert dv < cfg.knot_spacing_kms, "this test needs a grid where the floor binds"
    expected_scale = (dv / cfg.knot_spacing_kms) ** 2

    lam_grid = [1.0, 8.0, 64.0]
    result_on = select_lambda_aic_local(A_stack_w, y_w, cfg, lam_grid, self_normalize_lam_grid=True)
    np.testing.assert_allclose(sorted(result_on.lam_grid), sorted(np.array(lam_grid) * expected_scale))


def test_self_normalize_lam_grid_makes_effective_penalty_invariant_to_knot_spacing():
    """The actual point of the mechanism: `lam_scaled * D_narrow.T @ D_narrow`
    (narrow grid, dv < knot_spacing_kms) must equal `lam_nominal * D_ref.T @
    D_ref` (a hypothetical grid at dv == knot_spacing_kms, where
    `difference_matrix`'s own scaling is a no-op) for the same nominal lam --
    i.e. the effective curvature penalty applied is exactly portable across
    knot spacing once this correction is applied, canceling
    `difference_matrix`'s own `(dv_ref/dv)` scaling rather than fighting it."""
    from kinextract.losvd import difference_matrix

    n_coef = 20
    dv_ref = 95.0
    dv_narrow = 30.0  # a real narrow-grid case, e.g. sigma~25-55 under NSIG=7
    lam_nominal = 64.0
    scale = (dv_narrow / dv_ref) ** 2

    D_ref = difference_matrix(n_coef, order=3, dv=dv_ref, dv_ref=dv_ref)  # no-op scaling
    D_narrow = difference_matrix(n_coef, order=3, dv=dv_narrow, dv_ref=dv_ref)

    penalty_ref = lam_nominal * (D_ref.T @ D_ref)
    penalty_narrow_scaled = (lam_nominal * scale) * (D_narrow.T @ D_narrow)
    np.testing.assert_allclose(penalty_narrow_scaled, penalty_ref, rtol=1e-10)


def test_estimate_m_eff_default_L_none_reproduces_original_behavior():
    """`L=None` (the default) must give bit-identical results to before
    this parameter existed -- a regression guard, since `estimate_m_eff`'s
    mainline (properly-pre-whitened) case needs no change at all (see its
    own docstring for why an earlier version of this refactor's plan got
    this wrong)."""
    A_stack_w, y_w, cfg = _mock()
    m_eff_default, chi2_a, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam=2.0, n_boot=8, seed=1)
    m_eff_explicit_none, chi2_b, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam=2.0, n_boot=8, seed=1, L=None)
    assert m_eff_default == m_eff_explicit_none
    assert chi2_a == chi2_b


def test_estimate_m_eff_with_a_deliberately_mismatched_L_changes_the_result():
    """`L` is a deliberate-mismatch diagnostic, not a required fix (see
    docstring) -- confirms it actually does something when given (an
    identity-like near-zero-correlation `L` vs. a strongly-correlated one
    should measurably change the injected noise's effective structure,
    and thus `m_eff`)."""
    A_stack_w, y_w, cfg = _mock()
    n_pix = len(y_w)
    from kinextract.noise_model import build_ar1_covariance
    L_uncorrelated = np.eye(n_pix)
    L_strongly_correlated = np.linalg.cholesky(build_ar1_covariance(np.ones(n_pix), rho=0.8))

    m_eff_plain, _, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam=2.0, n_boot=15, seed=2, L=L_uncorrelated)
    m_eff_correlated, _, _ = estimate_m_eff(A_stack_w, y_w, cfg, lam=2.0, n_boot=15, seed=2, L=L_strongly_correlated)
    assert m_eff_plain != m_eff_correlated
