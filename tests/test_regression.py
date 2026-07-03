"""Regression tests to catch silent changes to mathematical results.

These use only standalone utility functions (no full fit) so they run in <1 s
but will catch refactoring bugs that accidentally change the math.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract.fitting import compute_losvd_roughness, compute_losvd_n_peaks
from kinextract.losvd import fit_losvd_gauss_hermite, getfwhm_fortran_like
from kinextract.continuum import robust_sigma, grow_boolean_mask


# Fixed reference inputs — never change these.
_V = np.linspace(-500.0, 500.0, 101)
_SIGMA_100 = np.exp(-0.5 * (_V / 100.0) ** 2)
_SIGMA_100 /= _SIGMA_100.sum()


def test_regression_roughness_sigma100():
    """Roughness of a sigma=100 km/s Gaussian must not change between versions."""
    r = compute_losvd_roughness(_SIGMA_100)
    # Reference value computed at v0.1.0 on 101-bin grid, v_max=500 km/s
    np.testing.assert_allclose(r, 0.009975, rtol=0.05, err_msg="roughness regressed")


def test_regression_n_peaks_sigma100():
    assert compute_losvd_n_peaks(_SIGMA_100) == 1


def test_regression_gh_fit_sigma100():
    """GH fit of a pure Gaussian should recover vherm~0, sherm~100."""
    result = fit_losvd_gauss_hermite(_V, _SIGMA_100, fit_h3h4=True)
    # fit_losvd_gauss_hermite uses vherm/sherm for GH velocity/dispersion
    np.testing.assert_allclose(result["vherm"], 0.0, atol=2.0, err_msg="vherm regressed")
    np.testing.assert_allclose(result["sherm"], 100.0, rtol=0.03, err_msg="sherm regressed")
    np.testing.assert_allclose(result["h3"], 0.0, atol=0.02, err_msg="h3 regressed")
    np.testing.assert_allclose(result["h4"], 0.0, atol=0.02, err_msg="h4 regressed")


def test_regression_fwhm_gaussian():
    v = np.linspace(-500, 500, 1001)
    y = np.exp(-0.5 * (v / 100.0) ** 2)
    result = getfwhm_fortran_like(v, y)
    expected = 2 * np.sqrt(2 * np.log(2)) * 100.0  # ≈ 235.48 km/s
    np.testing.assert_allclose(result["fwhm"], expected, rtol=0.01, err_msg="FWHM regressed")


def test_regression_robust_sigma():
    rng = np.random.default_rng(999)
    x = rng.normal(0.0, 3.0, 5000)
    rs = robust_sigma(x)
    np.testing.assert_allclose(rs, 3.0, rtol=0.05, err_msg="robust_sigma regressed")


def test_regression_grow_mask():
    m = np.zeros(20, dtype=bool)
    m[10] = True
    grown = grow_boolean_mask(m, grow_pix=2)
    expected = np.zeros(20, dtype=bool)
    expected[8:13] = True
    np.testing.assert_array_equal(grown, expected, err_msg="grow_boolean_mask regressed")
