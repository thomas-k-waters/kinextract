"""Tests for kinextract.lsf -- Gaussian LSF matching."""
from __future__ import annotations

import numpy as np

from kinextract.lsf import (
    fwhm_to_sigma, muse_lsf_fwhm, variable_gaussian_convolve, match_gaussian_lsf,
)


def test_fwhm_to_sigma_matches_known_constant():
    # sigma = FWHM / 2.3548...
    np.testing.assert_allclose(fwhm_to_sigma(2.3548200450309493), 1.0, atol=1e-6)


def test_muse_lsf_fwhm_matches_published_reference_points():
    # Bacon et al.-derived polynomial: ~3.0 A at the blue end (4800 A),
    # a minimum of ~2.47 A near the CaT triplet (8500-8700 A).
    assert abs(muse_lsf_fwhm(4800.0) - 2.98) < 0.05
    assert 2.45 < muse_lsf_fwhm(8600.0) < 2.50


def test_matching_equal_fwhm_is_a_no_op():
    wavelength = np.linspace(8400.0, 8800.0, 321)
    rng = np.random.default_rng(0)
    flux = 1.0 + 0.1 * rng.standard_normal(321)
    matched, cannot_sharpen = match_gaussian_lsf(wavelength, flux, template_fwhm=2.51, target_fwhm=2.51)
    np.testing.assert_allclose(matched, flux)
    assert not np.any(cannot_sharpen)


def test_broadening_a_narrow_line_recovers_the_target_fwhm():
    wavelength = np.linspace(8400.0, 8800.0, 2001)
    step = wavelength[1] - wavelength[0]
    center = 8600.0
    narrow_sigma = 0.3  # much narrower than either FWHM below
    flux = np.exp(-0.5 * ((wavelength - center) / narrow_sigma) ** 2)

    target_fwhm = 3.0
    matched, cannot_sharpen = match_gaussian_lsf(wavelength, flux, template_fwhm=1.0, target_fwhm=target_fwhm)
    assert not np.any(cannot_sharpen)

    # measure output FWHM directly: effective sigma^2 = narrow_sigma^2 + sigma_diff^2
    half_max = matched.max() / 2.0
    above = matched >= half_max
    measured_fwhm = wavelength[above][-1] - wavelength[above][0] + step
    expected_fwhm = np.sqrt(narrow_sigma ** 2 * (2.0 * np.sqrt(2 * np.log(2))) ** 2 + target_fwhm ** 2)
    assert abs(measured_fwhm - expected_fwhm) < 0.15


def test_cannot_sharpen_flagged_when_target_is_narrower_than_template():
    wavelength = np.linspace(8400.0, 8800.0, 321)
    flux = np.ones(321)
    matched, cannot_sharpen = match_gaussian_lsf(wavelength, flux, template_fwhm=3.0, target_fwhm=2.0)
    assert np.all(cannot_sharpen)
    np.testing.assert_allclose(matched, flux)  # passthrough, not a fabricated sharpening


def test_wavelength_dependent_target_only_sharpens_where_valid():
    wavelength = np.linspace(4750.0, 9350.0, 4601)
    flux = np.ones(len(wavelength))
    matched, cannot_sharpen = match_gaussian_lsf(
        wavelength, flux, template_fwhm=2.51, target_fwhm=muse_lsf_fwhm,
    )
    # near the CaT triplet MUSE's own FWHM is close to or below E-MILES' 2.51 A,
    # so at least part of that range should be flagged as un-sharpenable
    cat_region = (wavelength > 8400) & (wavelength < 8700)
    assert np.any(cannot_sharpen[cat_region])
    # far bluer, MUSE is clearly coarser (~3.0 A), so that part should convolve fine
    blue_region = wavelength < 5000
    assert not np.any(cannot_sharpen[blue_region])


def test_variable_gaussian_convolve_zero_sigma_is_identity():
    wavelength = np.linspace(8400.0, 8800.0, 101)
    rng = np.random.default_rng(1)
    flux = rng.standard_normal(101)
    out = variable_gaussian_convolve(wavelength, flux, sigma=np.zeros(101))
    np.testing.assert_allclose(out, flux)
