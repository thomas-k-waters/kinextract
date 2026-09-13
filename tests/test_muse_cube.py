"""Tests for kinextract.muse_cube -- the shared MuseCube container
and FITS I/O. Round-trip tested against mock_cube.py's own synthetic
output (see that module's docstring for why one shared container is
what lets binning/reduction code stay identical for real and synthetic
cubes).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from kinextract.muse_cube import (
    MuseCube, white_light_image, extract_aperture_spectrum, combine_bin_spectrum_robust,
    load_muse_cube, save_muse_cube_fits, spatial_correlation_inflation,
)

EMILES_GRID = "src/kinextract/template_data/emiles_muse_grid.npz"


def _tiny_cube(seed=0):
    rng = np.random.default_rng(seed)
    n_wave, ny, nx = 20, 6, 6
    flux = rng.normal(10.0, 1.0, size=(n_wave, ny, nx))
    error = np.full((n_wave, ny, nx), 1.0)
    wave = 8400.0 + np.arange(n_wave) * 1.25
    return MuseCube(flux=flux, error=error, wave=wave, pixel_scale=0.2)


def test_shape_property_returns_spatial_dims():
    cube = _tiny_cube()
    assert cube.shape == (6, 6)


def test_white_light_image_is_wavelength_sum():
    cube = _tiny_cube()
    wl = white_light_image(cube)
    assert wl.shape == (6, 6)
    np.testing.assert_allclose(wl, np.nansum(cube.flux, axis=0))


def test_extract_aperture_spectrum_sums_flux_and_combines_errors_in_quadrature():
    cube = _tiny_cube()
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True  # 4 spaxels
    wave, flux, error = extract_aperture_spectrum(cube, mask)
    np.testing.assert_allclose(wave, cube.wave)
    np.testing.assert_allclose(flux, np.nansum(cube.flux[:, 0:2, 0:2], axis=(1, 2)))
    expected_error = np.sqrt(np.nansum(cube.error[:, 0:2, 0:2] ** 2, axis=(1, 2)))
    np.testing.assert_allclose(error, expected_error)


def test_extract_aperture_spectrum_single_spaxel_matches_that_spaxels_own_values():
    cube = _tiny_cube()
    mask = np.zeros((6, 6), dtype=bool)
    mask[3, 4] = True
    wave, flux, error = extract_aperture_spectrum(cube, mask)
    np.testing.assert_allclose(flux, cube.flux[:, 3, 4])
    np.testing.assert_allclose(error, cube.error[:, 3, 4])


def test_extract_aperture_spectrum_excludes_bad_spaxels_from_the_sum_not_zero_fills():
    """A spaxel with NaN flux/error, or zero/negative error, at a given
    wavelength must be excluded from that wavelength's combination, not
    silently summed as a zero-flux/zero-error contribution (which would
    either lose real signal without renormalizing, or look like a
    perfect, zero-uncertainty non-detection)."""
    cube = _tiny_cube()
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True  # 4 spaxels
    bad_i, bad_j = 0, 0
    cube.flux[5, bad_i, bad_j] = np.nan
    cube.error[7, bad_i, bad_j] = 0.0
    cube.error[9, bad_i, bad_j] = np.nan

    wave, flux, error = extract_aperture_spectrum(cube, mask)
    good_mask = np.ones((2, 2), dtype=bool)
    good_mask[bad_i, bad_j] = False

    for w in (5, 7, 9):
        expected_flux = cube.flux[w, 0:2, 0:2][good_mask].sum()
        expected_var = (cube.error[w, 0:2, 0:2][good_mask] ** 2).sum()
        np.testing.assert_allclose(flux[w], expected_flux)
        np.testing.assert_allclose(error[w] ** 2, expected_var)
    # untouched wavelengths still include all 4 spaxels
    np.testing.assert_allclose(flux[0], cube.flux[0, 0:2, 0:2].sum())


def test_extract_aperture_spectrum_returns_nan_when_every_spaxel_is_bad_at_a_wavelength():
    cube = _tiny_cube()
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True
    cube.error[3, 0:2, 0:2] = 0.0  # every spaxel in the aperture bad at this one wavelength

    wave, flux, error = extract_aperture_spectrum(cube, mask)
    assert np.isnan(flux[3])
    assert np.isnan(error[3])
    assert np.isfinite(flux[0]) and np.isfinite(error[0])


def test_load_muse_cube_nans_out_zero_and_negative_variance_pixels(tmp_path):
    cube = _tiny_cube()
    cube.error[2, 1, 1] = 0.0  # zero variance -- a real ESO STAT-extension occurrence
    path = str(tmp_path / "test_cube.fits")
    save_muse_cube_fits(cube, path)
    loaded = load_muse_cube(path)
    assert not np.isfinite(loaded.error[2, 1, 1])
    assert not np.isfinite(loaded.flux[2, 1, 1])
    # every other pixel is untouched
    assert np.isfinite(loaded.error[0, 0, 0])


def test_fits_round_trip_preserves_flux_wave_and_pixel_scale():
    cube = _tiny_cube(seed=1)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        save_muse_cube_fits(cube, path)
        loaded = load_muse_cube(path)
        assert loaded.flux.shape == cube.flux.shape
        np.testing.assert_allclose(loaded.flux, cube.flux, atol=1e-3)
        np.testing.assert_allclose(loaded.wave, cube.wave, atol=1e-3)
        assert loaded.pixel_scale == pytest.approx(cube.pixel_scale, rel=1e-3)
        # STAT extension stores variance -- round-tripped error should recover the original
        np.testing.assert_allclose(loaded.error, cube.error, atol=1e-3)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_round_trip_of_a_real_mock_cube():
    """End-to-end: mock_cube.build_mock_cube's own output round-trips
    through save/load cleanly -- this is what Stage 3 of the template-
    calibration pipeline actually relies on (no real downloaded cube
    needed to test the FITS reader).
    """
    from kinextract.mock_cube import build_mock_cube
    cube = build_mock_cube(EMILES_GRID, ny=12, nx=12, seed=0)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        path = f.name
    try:
        save_muse_cube_fits(cube, path)
        loaded = load_muse_cube(path)
        assert loaded.shape == cube.shape
        np.testing.assert_allclose(loaded.wave, cube.wave, atol=1e-3)
        np.testing.assert_allclose(loaded.flux, cube.flux, atol=1e-2)
    finally:
        if os.path.exists(path):
            os.remove(path)


def _correlated_cube(seed=0, n_wave=40, ny=6, nx=6, per_spaxel_sigma=2.0):
    """A mock cube where every spaxel shares a common-mode noise component
    (a crude stand-in for MUSE's real spatially-correlated resampling
    noise) on top of its own independent noise -- each spaxel's own
    reported `error` only reflects the independent part, exactly like a
    real MUSE STAT extension that doesn't know about the correlation."""
    rng = np.random.default_rng(seed)
    true_flux = 100.0 + 5 * np.sin(np.linspace(0, 10, n_wave))
    common = rng.normal(0, per_spaxel_sigma * 0.8, size=n_wave)
    flux = np.empty((n_wave, ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            indep = rng.normal(0, per_spaxel_sigma * 0.6, size=n_wave)
            flux[:, iy, ix] = true_flux + common + indep
    error = np.full((n_wave, ny, nx), per_spaxel_sigma)
    wave = 8400.0 + np.arange(n_wave) * 1.25
    return MuseCube(flux=flux, error=error, wave=wave, pixel_scale=0.2), true_flux


def test_combine_bin_spectrum_robust_error_is_pure_formal_inverse_variance():
    """No empirical (biweight-scale-of-raw-flux) term anymore -- see
    combine_bin_spectrum_robust's own docstring, 2026-08-09: that term was a
    misapplied port (valid for combining repeat exposures of one sky
    position, not spatially different spaxels within one bin -- confirmed on
    real data to run 120-470x the actual noise level, from measuring real
    surface-brightness structure, not noise). Error here should now exactly
    match a hand-computed inverse-variance combine, nothing more. Catching
    correlated noise the formal term alone underestimates is
    noise_model.whiten's job now, not this function's."""
    cube, true_flux = _correlated_cube()
    mask = np.ones(cube.shape, dtype=bool)

    _, flux_rob, err_rob = combine_bin_spectrum_robust(cube, mask)

    variance_pix = cube.error[:, mask] ** 2
    expected_error = np.sqrt(1.0 / np.sum(1.0 / variance_pix, axis=1))
    np.testing.assert_allclose(err_rob, expected_error, rtol=1e-10)

    # still recovers the right mean flux level, within the real per-wavelength noise
    # scatter this mock injects (per_spaxel_sigma=2.0 combined over n_spax=36 spaxels,
    # plus the shared common-mode term) -- biweight_location is not literally the plain
    # mean, so this is a ballpark check, not an exact-equality one.
    np.testing.assert_allclose(flux_rob, true_flux, atol=5.0)


def test_combine_bin_spectrum_robust_flux_is_mean_not_sum():
    """flux_rob should be a per-spaxel-level quantity (~10, this fixture's
    own flux level), not an aperture-total (~n_spax*10=360) -- a ballpark
    check, not exact equality, since biweight_location is a robust
    location estimator, not literally the plain mean."""
    cube = _tiny_cube()
    mask = np.ones(cube.shape, dtype=bool)
    n_spax = mask.sum()
    _, flux_rob, _ = combine_bin_spectrum_robust(cube, mask)
    _, flux_sum, _ = extract_aperture_spectrum(cube, mask)
    np.testing.assert_allclose(flux_rob, flux_sum / n_spax, atol=0.5)
    assert np.all(flux_rob < flux_sum / 2)  # nowhere close to the aperture total


def test_combine_bin_spectrum_robust_is_resistant_to_one_outlier_spaxel():
    cube = _tiny_cube()
    contaminated = cube.flux.copy()
    contaminated[:, 0, 0] = 1e6  # one badly cosmic-ray-like spaxel, still "finite"
    bad_cube = MuseCube(flux=contaminated, error=cube.error, wave=cube.wave,
                         pixel_scale=cube.pixel_scale)
    mask = np.ones(cube.shape, dtype=bool)

    _, flux_rob, _ = combine_bin_spectrum_robust(bad_cube, mask)
    _, flux_sum, _ = extract_aperture_spectrum(bad_cube, mask)
    naive_mean = flux_sum / mask.sum()

    # the naive (non-robust) mean is dragged way up by the single bad spaxel;
    # the biweight combine should stay close to the other 35 spaxels' own level (~10.0)
    assert naive_mean[0] > 1000.0
    assert abs(flux_rob[0] - 10.0) < 2.0


def test_combine_bin_spectrum_robust_variance_correction_factor_scales_formal_error():
    cube = _tiny_cube()
    mask = np.ones(cube.shape, dtype=bool)
    _, _, err_uncorrected = combine_bin_spectrum_robust(cube, mask, variance_correction_factor=1.0)
    _, _, err_corrected = combine_bin_spectrum_robust(cube, mask, variance_correction_factor=4.0)
    # variance x4 -> error x2 exactly, now that error is pure formal inverse-variance
    # (no empirical term left to dilute the scaling).
    np.testing.assert_allclose(err_corrected, 2.0 * err_uncorrected, rtol=1e-10)


def test_combine_bin_spectrum_robust_all_bad_spaxels_gives_nan():
    cube = _tiny_cube()
    bad_error = np.where(np.ones_like(cube.error, dtype=bool), -1.0, cube.error)
    bad_cube = MuseCube(flux=cube.flux, error=bad_error, wave=cube.wave, pixel_scale=cube.pixel_scale)
    mask = np.ones(cube.shape, dtype=bool)
    _, flux_rob, err_rob = combine_bin_spectrum_robust(bad_cube, mask)
    assert np.all(np.isnan(err_rob))


def test_spatial_correlation_inflation_is_one_at_rho_zero():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True  # 36 spaxels, tightly packed -- would show a big effect if rho > 0
    assert spatial_correlation_inflation(mask, 0.0) == 1.0


def test_spatial_correlation_inflation_is_one_for_a_single_spaxel():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert spatial_correlation_inflation(mask, 0.5) == 1.0


def test_spatial_correlation_inflation_matches_hand_computation_for_two_adjacent_spaxels():
    # Var(mean of 2 samples, each variance `var`, correlation rho) = (var/4)*(1+1+2*rho)
    # = var*(1+rho)/2 -- vs. the independent-noise var/2 -- so the inflation ratio is
    # exactly (1+rho), a simple enough case to check by hand rather than trust the
    # general N-spaxel implementation on faith.
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    mask[2, 3] = True  # adjacent, dist=1
    for rho in (0.15, 0.5, 0.8):
        np.testing.assert_allclose(spatial_correlation_inflation(mask, rho), 1.0 + rho, rtol=1e-10)


def test_spatial_correlation_inflation_grows_with_rho_and_saturates_with_bin_size():
    mask_small = np.zeros((20, 20), dtype=bool)
    mask_small[9:11, 9:11] = True  # 4 tightly-packed spaxels
    mask_large = np.zeros((20, 20), dtype=bool)
    mask_large[2:18, 2:18] = True  # 256 spaxels, same density, larger footprint

    infl_lo = spatial_correlation_inflation(mask_small, 0.1)
    infl_hi = spatial_correlation_inflation(mask_small, 0.5)
    assert infl_hi > infl_lo > 1.0  # bigger rho -> bigger correction

    # A bigger, compact bin shows *more* per-spaxel inflation than a small one, not
    # less -- every interior spaxel's own local neighborhood (out to where rho**dist
    # is negligible) is fully populated only once the bin is big enough not to run
    # into its own edge, so inflation grows with bin size and *saturates* toward a
    # size-independent ceiling (each spaxel's own local sum), rather than shrinking
    # toward 1 -- confirmed directly: at rho=0.15, inflation keeps climbing from 4
    # spaxels (~1.37) through 10000 spaxels (~2.13) with diminishing increments, not
    # unbounded growth. Checked here only that big > small and both stay finite/sane.
    infl_big_bin = spatial_correlation_inflation(mask_large, 0.15)
    infl_small_bin = spatial_correlation_inflation(mask_small, 0.15)
    assert 1.0 < infl_small_bin < infl_big_bin < 5.0


def test_spatial_correlation_inflation_max_dist_truncation_matches_untruncated_for_a_small_case():
    # For a case small enough that every real pair is well within max_dist anyway,
    # truncating shouldn't change anything -- confirms the truncation is a performance
    # optimization, not a silent approximation, for the regime it's meant to cover.
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True  # 9 spaxels, max real separation ~2*sqrt(2) < 6
    infl_default = spatial_correlation_inflation(mask, 0.15, max_dist=6.0)
    infl_generous = spatial_correlation_inflation(mask, 0.15, max_dist=50.0)
    np.testing.assert_allclose(infl_default, infl_generous, rtol=1e-10)


def test_combine_bin_spectrum_robust_spatial_rho_zero_matches_no_spatial_rho_at_all():
    cube = _tiny_cube()
    mask = np.ones(cube.shape, dtype=bool)
    _, _, err_default = combine_bin_spectrum_robust(cube, mask)
    _, _, err_explicit_zero = combine_bin_spectrum_robust(cube, mask, spatial_rho=0.0)
    np.testing.assert_allclose(err_default, err_explicit_zero, rtol=1e-10)


def test_combine_bin_spectrum_robust_spatial_rho_inflates_error_by_the_expected_factor():
    cube = _tiny_cube()
    mask = np.ones(cube.shape, dtype=bool)  # 6x6 = 36 spaxels
    _, _, err_uncorrected = combine_bin_spectrum_robust(cube, mask, spatial_rho=0.0)
    _, _, err_corrected = combine_bin_spectrum_robust(cube, mask, spatial_rho=0.15)
    expected_factor = np.sqrt(spatial_correlation_inflation(mask, 0.15))
    assert expected_factor > 1.0  # sanity: this test is only meaningful if there's a real effect
    np.testing.assert_allclose(err_corrected, err_uncorrected * expected_factor, rtol=1e-10)
