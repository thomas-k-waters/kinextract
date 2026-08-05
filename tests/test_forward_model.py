"""Tests for kinextract.forward_model, in particular
`mask_wavelength_windows` -- added for dev_notes/VALIDATION_CAMPAIGN.md Stage 4,
motivated by a real finding on NGC 5102 MUSE data (fixed, recurring
severely-deviant pixels at specific wavelengths in every bin, most
consistent with sky-subtraction residuals -- see that function's own
docstring).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth, compute_losvd_moments,
    mask_wavelength_windows,
)
from kinextract.forward_model import shift_template
from kinextract.multi_template import fit_losvd


def test_mask_wavelength_windows_drops_the_right_pixels():
    wave = np.linspace(8400.0, 8800.0, 321)
    flux = np.sin(wave / 50.0)
    errors = np.full_like(wave, 0.01)

    wave_m, flux_m, err_m = mask_wavelength_windows(
        wave, flux, errors, windows=[(8415.0, 8418.0), (8450.0, 8453.0)]
    )
    assert len(wave_m) < len(wave)
    assert not np.any((wave_m >= 8415.0) & (wave_m <= 8418.0))
    assert not np.any((wave_m >= 8450.0) & (wave_m <= 8453.0))
    # everything outside the windows survives, in order
    outside = ~(((wave >= 8415.0) & (wave <= 8418.0)) | ((wave >= 8450.0) & (wave <= 8453.0)))
    np.testing.assert_array_equal(wave_m, wave[outside])
    np.testing.assert_array_equal(flux_m, flux[outside])
    np.testing.assert_array_equal(err_m, errors[outside])


def test_mask_wavelength_windows_with_no_windows_is_a_no_op():
    wave = np.linspace(8400.0, 8800.0, 50)
    flux = np.arange(50, dtype=float)
    wave_m, flux_m = mask_wavelength_windows(wave, flux, windows=[])
    np.testing.assert_array_equal(wave_m, wave)
    np.testing.assert_array_equal(flux_m, flux)


def test_mask_wavelength_windows_drops_nonfinite_pixels_by_default():
    """Real cube data can carry NaN flux/error at unobserved/flagged
    pixels (extract_aperture_spectrum's own docstring) -- drop_nonfinite
    (default True) removes these automatically, the same mechanism as an
    explicit window, so real-data callers don't have to remember a
    separate step."""
    wave = np.linspace(8400.0, 8800.0, 20)
    flux = np.ones(20)
    errors = np.full(20, 0.1)
    flux[5] = np.nan
    errors[10] = np.inf
    errors[15] = 0.0  # not itself dropped by drop_nonfinite (0.0 IS finite) -- a
                       # caller-side data-quality check, not this function's job

    wave_m, flux_m, err_m = mask_wavelength_windows(wave, flux, errors, windows=[])
    assert len(wave_m) == 18
    assert not np.any(np.isnan(flux_m))
    assert not np.any(np.isinf(err_m))
    assert 0.0 in err_m  # confirms the zero-error pixel was NOT dropped by this path


def test_mask_wavelength_windows_drop_nonfinite_false_preserves_old_behavior():
    wave = np.linspace(8400.0, 8800.0, 10)
    flux = np.ones(10)
    flux[3] = np.nan
    wave_m, flux_m = mask_wavelength_windows(wave, flux, windows=[], drop_nonfinite=False)
    assert len(wave_m) == 10
    assert np.isnan(flux_m[3])


def test_masking_a_localized_contaminant_recovers_the_clean_fit():
    """The actual motivating scenario: a handful of severely deviant
    pixels at a fixed wavelength (mimicking a sky-subtraction residual)
    injected into an otherwise clean mock. Confirms (a) the contaminant
    measurably degrades the recovered moments relative to the true clean
    fit, and (b) masking those exact pixels before fitting recovers
    accuracy close to the truly-clean case -- not just "runs without
    crashing," a real before/after comparison against known ground truth.
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    ca_centers = [8498.02, 8542.09, 8662.14]
    ca_depths = [0.55, 0.70, 0.65]
    template_flux = np.ones(n_pix)
    for cen, depth in zip(ca_centers, ca_depths):
        template_flux -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)

    vel, sig, nsig, snr = 0.0, 140.0, 7, 100.0
    # penalty_order=2 pinned deliberately (2026-07, Stage 22): this test
    # characterizes the old order-2 fit's sensitivity to a localized
    # contaminant (and masking's remedy for it) at a specific severity/seed
    # count. The order-3 default's strong shrink-toward-Gaussian prior is
    # measurably MORE robust to a bounded contaminant -- checked directly
    # (15 seeds, same setup): the sign of (contaminated_err - clean_err)
    # flips seed to seed and the median case shows no reliable degradation
    # at this contaminant severity, so this specific demonstration (and its
    # threshold/seed count) no longer applies to the new default. The
    # increased real-data robustness itself is a genuine improvement, not a
    # regression -- it just isn't what this test's premise is about.
    cfg = FitConfig(vmin=vel - nsig * sig, vmax=vel + nsig * sig, penalty_order=2)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr

    # inject a localized contaminant: a handful of pixels near 8416 A
    # with a huge, fixed flux excess -- mimicking an unsubtracted sky line.
    contaminant_window = (8415.0, 8418.0)
    contaminated_idx = (wave_fit >= contaminant_window[0]) & (wave_fit <= contaminant_window[1])
    assert contaminated_idx.sum() >= 1, "test wavelength window must actually hit some pixels"
    errors = np.full(len(wave_fit), noise_level)

    def fit_and_score(wave_, flux_, err_):
        A_stack_ = build_template_stack(wave_, wavelength, [template_flux], v_grid)
        A_stack_w, y_w = A_stack_ / err_[None, :, None], flux_ / err_
        result = fit_losvd(A_stack_w, y_w, cfg)
        m = compute_losvd_moments(v_grid, result["b"])
        return abs(m["sherm"] - sig)

    # A single-seed comparison is too noisy to reliably show the causal
    # direction (a curvature-penalized fit is somewhat robust to a few
    # pixels, so one lucky/unlucky clean-noise draw can dominate) --
    # average over several seeds, and use a severe enough contaminant
    # (100x the noise level) that its effect isn't marginal.
    clean_errs, contaminated_errs, masked_errs = [], [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        galaxy_clean = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
        galaxy_contaminated = galaxy_clean.copy()
        galaxy_contaminated[contaminated_idx] += 100.0 * noise_level

        clean_errs.append(fit_and_score(wave_fit, galaxy_clean, errors))
        contaminated_errs.append(fit_and_score(wave_fit, galaxy_contaminated, errors))

        wave_masked, flux_masked, err_masked = mask_wavelength_windows(
            wave_fit, galaxy_contaminated, errors, windows=[contaminant_window]
        )
        masked_errs.append(fit_and_score(wave_masked, flux_masked, err_masked))

    mean_clean, mean_contaminated, mean_masked = np.mean(clean_errs), np.mean(contaminated_errs), np.mean(masked_errs)
    assert mean_contaminated > mean_clean, (
        f"the injected contaminant should measurably degrade the fit on average, "
        f"got clean={mean_clean:.2f} contaminated={mean_contaminated:.2f}"
    )
    assert mean_masked < mean_contaminated, (
        f"masking the contaminant should measurably improve the fit, "
        f"got contaminated={mean_contaminated:.2f} masked={mean_masked:.2f}"
    )


def test_shift_template_default_method_is_linear_and_unchanged():
    """method='linear' (the default) must be bit-identical to np.interp,
    matching every version of this package before `method` existed --
    same discipline as every other opt-in feature in this project."""
    wave = np.linspace(8400.0, 8500.0, 50)
    template_wave = np.arange(8300.0, 8700.0, 0.9)
    template_flux = np.sin((template_wave - 8300.0) / 20.0) + 2.0
    explicit = shift_template(wave, template_wave, template_flux, 100.0, method="linear")
    implicit = shift_template(wave, template_wave, template_flux, 100.0)
    np.testing.assert_array_equal(explicit, implicit)
    direct = np.interp(wave / (1.0 + 100.0 / 299792.458), template_wave, template_flux,
                       left=0.0, right=0.0)
    np.testing.assert_array_equal(implicit, direct)


def test_shift_template_lanczos_reproduces_exact_grid_alignment():
    """At a shift where the query wavelengths land exactly on native
    template grid points, Lanczos (like any reasonable interpolation
    kernel) must reproduce those points' own values to numerical
    precision -- sinc(0)=1, sinc(nonzero integer)=0."""
    template_wave = np.arange(8300.0, 8700.0, 0.9)
    template_flux = np.sin((template_wave - 8300.0) / 20.0) + 2.0
    wave = template_wave[100:300]  # v=0 query exactly on the native grid
    lz = shift_template(wave, template_wave, template_flux, 0.0, method="lanczos")
    np.testing.assert_allclose(lz[3:-3], template_flux[100:300][3:-3], atol=1e-9)


def test_shift_template_lanczos_removes_the_periodic_component_of_linear_error():
    """The defect this exists to fix: linear interpolation's error is
    periodic in the fractional pixel shift (one period per native
    template pixel). Confirms Lanczos does not show that same periodic
    swing -- checked via the coefficient of variation of the max-error
    envelope across one full period, which should be much smaller for
    Lanczos than for linear (see dev_notes/shift_template/FINDINGS.md for
    the full quantification against an independent ground truth).
    """
    from scipy.interpolate import CubicSpline

    template_wave = np.arange(8300.0, 8700.0, 0.9)
    rng = np.random.default_rng(0)
    # a few sharp, absorption-line-like features, not a pure sinusoid --
    # closer to a real spectrum than the smooth case above.
    template_flux = np.ones_like(template_wave)
    for center in (8420.0, 8480.0, 8550.0):
        template_flux -= 0.4 * np.exp(-0.5 * ((template_wave - center) / 3.0) ** 2)

    wave_obs = np.arange(8400.0, 8600.0, 1.25)
    dx = 0.9
    one_pixel_kms = 299792.458 * dx / 8500.0

    cs = CubicSpline(template_wave, template_flux)
    wave_fine = np.arange(template_wave[0], template_wave[-1], dx / 20)
    flux_fine = cs(wave_fine)

    def true_shift(v):
        lam_rest = wave_obs / (1.0 + v / 299792.458)
        return np.interp(lam_rest, wave_fine, flux_fine, left=0.0, right=0.0)

    velocities = np.linspace(0.0, one_pixel_kms, 11)
    lin_max_errs, lz_max_errs = [], []
    for v in velocities:
        true = true_shift(v)
        denom = np.abs(true).max()
        lin = shift_template(wave_obs, template_wave, template_flux, v, method="linear")
        lz = shift_template(wave_obs, template_wave, template_flux, v, method="lanczos")
        lin_max_errs.append(np.abs(lin - true).max() / denom)
        lz_max_errs.append(np.abs(lz - true).max() / denom)

    lin_max_errs, lz_max_errs = np.array(lin_max_errs), np.array(lz_max_errs)
    # coefficient of variation across the phase scan: how much the peak error
    # itself swings as a function of velocity phase, relative to its own mean.
    lin_cv = lin_max_errs.std() / lin_max_errs.mean()
    lz_cv = lz_max_errs.std() / lz_max_errs.mean()
    assert lz_cv < 0.5 * lin_cv, (
        f"expected lanczos to substantially reduce the phase-periodic swing in "
        f"interpolation error, got linear CV={lin_cv:.3f} lanczos CV={lz_cv:.3f}"
    )


def test_shift_template_lanczos_rejects_nonuniform_template_wave():
    template_wave = np.concatenate([np.arange(8300.0, 8500.0, 0.9),
                                    np.arange(8500.0, 8700.0, 1.3)])  # spacing changes
    template_flux = np.ones_like(template_wave)
    wave = np.linspace(8400.0, 8600.0, 50)
    try:
        shift_template(wave, template_wave, template_flux, 0.0, method="lanczos")
        assert False, "expected ValueError for non-uniform template_wave"
    except ValueError as e:
        assert "uniformly-sampled" in str(e)


def test_shift_template_unknown_method_raises():
    template_wave = np.arange(8300.0, 8700.0, 0.9)
    template_flux = np.ones_like(template_wave)
    wave = np.linspace(8400.0, 8600.0, 50)
    try:
        shift_template(wave, template_wave, template_flux, 0.0, method="cubic")
        assert False, "expected ValueError for an unknown method"
    except ValueError as e:
        assert "unknown method" in str(e)


def test_build_template_stack_default_method_matches_shift_template_directly():
    """build_template_stack's own method/lanczos_a plumbing must actually
    reach shift_template -- confirmed by comparing a hand-built stack
    against shift_template called directly, for both methods."""
    template_wave = np.arange(8300.0, 8700.0, 0.9)
    template_flux = np.sin((template_wave - 8300.0) / 15.0) + 2.0
    wave_obs = np.linspace(8400.0, 8600.0, 80)
    v_grid = np.array([-50.0, 0.0, 73.0])

    for method in ("linear", "lanczos"):
        stack = build_template_stack(wave_obs, template_wave, [template_flux], v_grid, method=method)
        assert stack.shape == (1, len(wave_obs), len(v_grid))
        for j, v in enumerate(v_grid):
            expected = shift_template(wave_obs, template_wave, template_flux, v, method=method)
            np.testing.assert_array_equal(stack[0, :, j], expected)
