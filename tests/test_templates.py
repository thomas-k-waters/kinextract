"""Tests for the instrumental LSF matching helpers in templates.py."""
from __future__ import annotations

import numpy as np
import pytest

from kinextract.templates import resolution_mismatch_sigma_A, convolve_gaussian_pixels
from kinextract.config import FitConfig

_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def test_resolution_mismatch_templates_sharper():
    # sigma_data > sigma_tpl -> templates must be convolved
    data_fwhm = 2.0
    tpl_fwhm = 1.0
    sigma_diff, direction = resolution_mismatch_sigma_A(data_fwhm, tpl_fwhm)
    assert direction == "convolve_templates"
    sigma_data = data_fwhm / _SIGMA_TO_FWHM
    sigma_tpl = tpl_fwhm / _SIGMA_TO_FWHM
    expected = np.sqrt(sigma_data ** 2 - sigma_tpl ** 2)
    np.testing.assert_allclose(sigma_diff, expected, rtol=1e-10)


def test_resolution_mismatch_data_sharper():
    sigma_diff, direction = resolution_mismatch_sigma_A(1.0, 2.0)
    assert direction == "convolve_data"
    assert sigma_diff > 0


def test_resolution_mismatch_equal_is_noop():
    sigma_diff, direction = resolution_mismatch_sigma_A(1.5, 1.5)
    assert direction == "none"
    assert sigma_diff == 0.0


def test_resolution_mismatch_rejects_nonpositive():
    with pytest.raises(ValueError):
        resolution_mismatch_sigma_A(0.0, 1.0)
    with pytest.raises(ValueError):
        resolution_mismatch_sigma_A(1.0, -1.0)
    with pytest.raises(ValueError):
        resolution_mismatch_sigma_A(np.nan, 1.0)


def test_convolve_gaussian_pixels_zero_sigma_is_noop():
    flux = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    out = convolve_gaussian_pixels(flux, 0.0)
    np.testing.assert_array_equal(out, flux)


def test_convolve_gaussian_pixels_widens_a_narrow_spike():
    n = 201
    flux = np.zeros(n)
    flux[n // 2] = 1.0
    sigma_pix = 5.0
    out = convolve_gaussian_pixels(flux, sigma_pix)

    # Recovered width (second moment) should match the input sigma.
    x = np.arange(n) - n // 2
    mean = np.sum(x * out) / np.sum(out)
    var = np.sum((x - mean) ** 2 * out) / np.sum(out)
    np.testing.assert_allclose(np.sqrt(var), sigma_pix, rtol=0.05)
    # Flux-conserving.
    np.testing.assert_allclose(np.sum(out), np.sum(flux), rtol=1e-8)


def test_convolve_gaussian_pixels_2d_along_axis0():
    n, ntempl = 101, 3
    T = np.zeros((n, ntempl))
    T[n // 2, :] = 1.0
    out = convolve_gaussian_pixels(T, 4.0, axis=0)
    assert out.shape == T.shape
    # Every column should be smoothed identically (same spike, same kernel).
    np.testing.assert_allclose(out[:, 0], out[:, 1])
    np.testing.assert_allclose(out[:, 0], out[:, 2])
    assert out[n // 2, 0] < T[n // 2, 0]  # peak spread out, no longer a spike


def test_fitconfig_requires_both_fwhm_fields_together():
    with pytest.raises(ValueError):
        FitConfig(data_fwhm_A=2.0, template_fwhm_A=None)
    with pytest.raises(ValueError):
        FitConfig(data_fwhm_A=None, template_fwhm_A=1.0)
    # Both None (default): fine.
    FitConfig()
    # Both set: fine.
    FitConfig(data_fwhm_A=2.0, template_fwhm_A=1.0)


def test_fitconfig_data_fwhm_frame_validated():
    with pytest.raises(ValueError):
        FitConfig(data_fwhm_A=2.0, template_fwhm_A=1.0, data_fwhm_frame="bogus")
    FitConfig(data_fwhm_A=2.0, template_fwhm_A=1.0, data_fwhm_frame="rest")
    FitConfig(data_fwhm_A=2.0, template_fwhm_A=1.0, data_fwhm_frame="observed")


@pytest.mark.slow
def test_lsf_matching_end_to_end_changes_the_fit(real_muse_fit):
    """Both LSF-matching directions should run cleanly and measurably change
    the recovered LOSVD relative to an unmatched fit (sanity check that the
    convolution wired into make_fit_state actually takes effect), using the
    same bundled MUSE spectrum as the real_muse_fit fixture.
    """
    from kinextract import FitConfig, run_spectral_fit

    fit_baseline, cfg_baseline = real_muse_fit
    b_baseline = fit_baseline["outputs"]["b"]
    b_baseline = b_baseline / b_baseline.sum()

    import copy
    gal_file = fit_baseline["gal_file"]
    import numpy as np
    data = np.loadtxt(gal_file)
    ferr = data[:, 1] / 50.0

    # templates sharper -> convolve_templates branch
    cfg_a = copy.deepcopy(cfg_baseline)
    cfg_a.data_fwhm_A = 3.0
    cfg_a.template_fwhm_A = 1.0
    fit_a = run_spectral_fit(cfg_a, gal_file=gal_file, gal_errors=ferr, write_outputs=False)
    b_a = fit_a["outputs"]["b"]
    b_a = b_a / b_a.sum()
    assert np.max(np.abs(b_a - b_baseline)) > 1e-4

    # data sharper -> convolve_data branch
    cfg_b = copy.deepcopy(cfg_baseline)
    cfg_b.data_fwhm_A = 1.0
    cfg_b.template_fwhm_A = 3.0
    fit_b = run_spectral_fit(cfg_b, gal_file=gal_file, gal_errors=ferr, write_outputs=False)
    b_b = fit_b["outputs"]["b"]
    b_b = b_b / b_b.sum()
    assert np.max(np.abs(b_b - b_baseline)) > 1e-4
