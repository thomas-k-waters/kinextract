"""Tests for LOSVD analysis utilities (losvd.py).

These are pure-numpy functions with no FitState dependency, so they run fast
and cover the most fundamental scientific invariants.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract.losvd import (
    fit_losvd_gauss_hermite,
    gauss_hermite_losvd_model,
    getfwhm_fortran_like,
)
from kinextract.fitting import compute_losvd_roughness, compute_losvd_n_peaks


# ── Roughness ───────────────────────────────────────────────────────────────

def test_roughness_flat_losvd_is_zero():
    b = np.ones(51)
    assert compute_losvd_roughness(b) == 0.0


def test_roughness_spike_is_large():
    b = np.zeros(51)
    b[25] = 1.0  # single spike → maximum curvature
    r_spike = compute_losvd_roughness(b)
    b_smooth = np.exp(-0.5 * ((np.arange(51) - 25) / 8.0) ** 2)
    r_smooth = compute_losvd_roughness(b_smooth)
    assert r_spike > r_smooth * 5


def test_roughness_increases_with_regularization():
    """More regularization → smoother LOSVD → lower roughness."""
    v = np.linspace(-500, 500, 101)
    # Narrow Gaussian is rougher than a wide one (same peak height)
    b_narrow = np.exp(-0.5 * (v / 30.0) ** 2)
    b_wide = np.exp(-0.5 * (v / 200.0) ** 2)
    assert compute_losvd_roughness(b_narrow) > compute_losvd_roughness(b_wide)


def test_roughness_degenerate_near_zero_returns_inf():
    """A near-zero LOSVD must return inf so _auto_select_xlam keeps searching."""
    b = np.full(51, 1e-10)
    assert compute_losvd_roughness(b) == np.inf


# ── Peak counting ────────────────────────────────────────────────────────────

def test_n_peaks_single_gaussian():
    v = np.linspace(-500, 500, 101)
    b = np.exp(-0.5 * (v / 100.0) ** 2)
    assert compute_losvd_n_peaks(b) == 1


def test_n_peaks_two_separated_gaussians():
    v = np.linspace(-500, 500, 201)
    b = np.exp(-0.5 * ((v + 200) / 60.0) ** 2) + np.exp(-0.5 * ((v - 200) / 60.0) ** 2)
    assert compute_losvd_n_peaks(b) == 2


def test_n_peaks_shoulder_not_counted():
    """A shallow shoulder (low prominence) should NOT be counted as a peak."""
    v = np.linspace(-500, 500, 201)
    # Main peak + very small bump = shoulder, not a genuine secondary lobe
    b = np.exp(-0.5 * (v / 100.0) ** 2) + 0.05 * np.exp(-0.5 * ((v - 200) / 60.0) ** 2)
    assert compute_losvd_n_peaks(b, min_prominence=0.1) == 1


def test_n_peaks_all_zeros_returns_one():
    b = np.zeros(51)
    assert compute_losvd_n_peaks(b) == 1


# ── Gauss-Hermite fitting ────────────────────────────────────────────────────

def _make_gh_losvd(v, V, sigma, h3=0.0, h4=0.0):
    """Reference GH model for generating test data."""
    w = (v - V) / sigma
    L = np.exp(-0.5 * w ** 2) / (np.sqrt(2 * np.pi) * sigma)
    H3 = (2 * np.sqrt(2) * w ** 3 - 3 * np.sqrt(2) * w) / np.sqrt(6)
    H4 = (4 * w ** 4 - 12 * w ** 2 + 3) / np.sqrt(24)
    return L * (1 + h3 * H3 + h4 * H4)


def test_gh_fit_recovers_velocity():
    v = np.linspace(-500, 500, 201)
    true_V, true_sigma = 80.0, 130.0
    b = _make_gh_losvd(v, true_V, true_sigma)
    b = np.maximum(b, 0.0)
    result = fit_losvd_gauss_hermite(v, b, fit_h3h4=False)
    # fit_losvd_gauss_hermite returns vherm/sherm (GH parameterisation), v1/v2 (moments)
    assert abs(result["vherm"] - true_V) < 3.0, f"vherm={result['vherm']:.1f} != {true_V}"
    assert abs(result["sherm"] - true_sigma) < 5.0, f"sherm={result['sherm']:.1f} != {true_sigma}"


def test_gh_fit_h3h4_nonzero():
    v = np.linspace(-500, 500, 201)
    b = _make_gh_losvd(v, 0.0, 100.0, h3=0.1, h4=0.05)
    b = np.maximum(b, 0.0)
    result = fit_losvd_gauss_hermite(v, b, fit_h3h4=True)
    assert abs(result["h3"] - 0.1) < 0.05
    assert abs(result["h4"] - 0.05) < 0.05


def test_gh_fit_symmetric_losvd_gives_zero_h3():
    v = np.linspace(-500, 500, 201)
    b = np.exp(-0.5 * (v / 120.0) ** 2)
    result = fit_losvd_gauss_hermite(v, b, fit_h3h4=True)
    # Symmetric LOSVD → h3 ≈ 0, vherm ≈ 0
    assert abs(result["h3"]) < 0.02, f"h3={result['h3']:.4f} should be ~0 for symmetric LOSVD"
    assert abs(result["vherm"]) < 3.0, f"vherm={result['vherm']:.2f} should be ~0 for symmetric LOSVD"


# ── FWHM ────────────────────────────────────────────────────────────────────

def test_getfwhm_gaussian():
    v = np.linspace(-500, 500, 1001)
    sigma = 100.0
    y = np.exp(-0.5 * (v / sigma) ** 2)
    result = getfwhm_fortran_like(v, y)
    expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert abs(result["fwhm"] - expected_fwhm) / expected_fwhm < 0.01
