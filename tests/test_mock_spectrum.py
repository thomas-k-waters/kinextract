"""Correctness tests for benchmarks/mock_spectrum.py.

Unlike the rest of the benchmarks/ validation suite (a diagnostic/reporting
tool, not run under pytest), the mock generator itself makes concrete
mathematical claims (the general LOSVD-convolution path agrees with the
simpler closed-form Gaussian path; degenerate parametrizations reduce to
expected special cases) that are legitimate, fast, pytest-style correctness
questions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.mock_spectrum import (
    apply_gaussian_losvd,
    convolve_template_with_losvd,
    gauss_hermite_losvd_on_grid,
    double_peak_losvd_on_grid,
    inject_emission_lines,
    DEFAULT_EMISSION_LINES_A,
)


def test_general_convolution_matches_gaussian_fast_path():
    """The general kernel-convolution path (used for GH-skewed/double-peak
    LOSVDs) must agree with the closed-form gaussian_filter+shift path (used
    for the plain-Gaussian fast path) to within a small tolerance, for a
    pure Gaussian LOSVD where both are defined.
    """
    wavelength = 4750.0 + np.arange(3681) * 1.25
    template = np.ones_like(wavelength)
    for cen, depth in zip((8498.02, 8542.09, 8662.14), (0.55, 0.70, 0.65)):
        template -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)

    v_true, sigma_true, lam_center, step = 80.0, 140.0, 8580.0, 1.25
    fast = apply_gaussian_losvd(template, v_true, sigma_true, lam_center, step)

    # Shape kernel is built centered at v=0 (not v_true) -- np.convolve's
    # "same" mode aligns the kernel's array-index center with zero lag, with
    # no notion of what velocity that index represents, so building the
    # shape off-center would silently discard the intended shift (see the
    # equivalent note in benchmarks.mock_spectrum.build_mock_spectrum). The
    # v_true offset is applied explicitly afterward, exactly mirroring
    # apply_gaussian_losvd's own gaussian_filter-then-shift decomposition.
    from scipy.ndimage import shift as ndimage_shift
    v_grid = np.linspace(-8 * sigma_true, 8 * sigma_true, 4001)
    losvd = gauss_hermite_losvd_on_grid(v_grid, 0.0, sigma_true, h3=0.0, h4=0.0)
    shape_general = convolve_template_with_losvd(template, step, v_grid, losvd, lam_center)
    shift_pix = v_true * lam_center / (299792.458 * step)
    general = ndimage_shift(shape_general, shift=+shift_pix)

    mask = (wavelength > 8420) & (wavelength < 8740)  # avoid edge effects
    rel_diff = np.abs(fast[mask] - general[mask]) / np.abs(fast[mask]).max()
    assert np.max(rel_diff) < 0.02, f"general convolution disagrees with fast Gaussian path by {np.max(rel_diff):.3%}"


def test_gauss_hermite_reduces_to_gaussian_when_h3_h4_zero():
    v_grid = np.linspace(-500, 500, 2001)
    losvd = gauss_hermite_losvd_on_grid(v_grid, v_true=20.0, sigma_true=100.0, h3=0.0, h4=0.0)
    expected = np.exp(-0.5 * ((v_grid - 20.0) / 100.0) ** 2)
    expected /= expected.sum()
    np.testing.assert_allclose(losvd, expected, atol=1e-10)


def test_double_peak_reduces_to_single_gaussian_when_degenerate():
    v_grid = np.linspace(-500, 500, 2001)
    losvd = double_peak_losvd_on_grid(v_grid, v_center=0.0, sigma_true=100.0,
                                       separation_frac=0.0, width_frac=1.0, weight_ratio=1.0)
    expected = np.exp(-0.5 * (v_grid / 100.0) ** 2)
    expected /= expected.sum()
    np.testing.assert_allclose(losvd, expected, atol=1e-10)


def test_double_peak_actually_has_two_local_maxima_when_well_separated():
    v_grid = np.linspace(-500, 500, 4001)
    losvd = double_peak_losvd_on_grid(v_grid, v_center=0.0, sigma_true=200.0,
                                       separation_frac=0.8, width_frac=0.3, weight_ratio=1.0)
    d = np.diff(losvd)
    sign_changes = np.diff(np.sign(d))
    n_maxima = int(np.sum(sign_changes < 0))
    assert n_maxima >= 2, f"expected >=2 local maxima for a well-separated double peak, got {n_maxima}"


def test_inject_emission_lines_is_detectable_by_the_real_masking_logic():
    """Round-trip through kinextract.masking's actual emission detection to
    confirm injected contamination is representative of what the real
    pipeline looks for, not just an arbitrary bump.
    """
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from kinextract.masking import _segment_emission_mask
    from kinextract.config import FitConfig

    wavelength = 8400.0 + np.arange(320) * 1.25
    continuum = 1000.0
    flux = np.full_like(wavelength, continuum)
    rng = np.random.default_rng(0)
    flux += rng.normal(0, 5.0, len(flux))  # small photon-like noise
    gerr = np.full_like(wavelength, 5.0)

    strong_line = DEFAULT_EMISSION_LINES_A[3]  # an "em"-tagged line within this window
    contaminated = inject_emission_lines(
        wavelength, flux, [strong_line], continuum, amplitude_range=(2.0, 2.0), fwhm_A=3.0, rng=rng,
    )
    cfg = FitConfig(template_list_file="unused", segment_emission_n_sigma=3.0, segment_emission_win_A=50.0)
    mask = _segment_emission_mask(wavelength, contaminated, gerr, cfg)
    near_line = np.abs(wavelength - strong_line) < 3.0
    assert mask[near_line].any(), "injected emission line was not detected by the real segment-emission-mask logic"
