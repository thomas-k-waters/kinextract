"""Tests for core numerics (numerics.py) and metamorphic fitting properties.

These tests check mathematical properties that must hold regardless of the
specific spectrum, without needing a full end-to-end fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract.fitting import compute_losvd_roughness, compute_losvd_n_peaks


# ── Metamorphic: roughness monotonicity ─────────────────────────────────────

def _losvd_with_sigma(sigma_kms: float, n: int = 101, v_max: float = 600.0) -> np.ndarray:
    v = np.linspace(-v_max, v_max, n)
    b = np.exp(-0.5 * (v / sigma_kms) ** 2)
    return b / b.sum()


@pytest.mark.parametrize("sigma_narrow,sigma_wide", [
    (30, 80),
    (50, 150),
    (80, 300),
])
def test_roughness_decreases_as_losvd_widens(sigma_narrow, sigma_wide):
    """Wider Gaussian → smoother → lower roughness."""
    r_narrow = compute_losvd_roughness(_losvd_with_sigma(sigma_narrow))
    r_wide = compute_losvd_roughness(_losvd_with_sigma(sigma_wide))
    assert r_narrow > r_wide, (
        f"Expected roughness(sigma={sigma_narrow}) > roughness(sigma={sigma_wide}), "
        f"got {r_narrow:.4g} <= {r_wide:.4g}"
    )


# ── Metamorphic: flux scaling invariance ─────────────────────────────────────

def test_roughness_scale_invariant():
    """Scaling the LOSVD by a constant should not change roughness
    (roughness is normalised by peak)."""
    b = _losvd_with_sigma(100.0)
    r1 = compute_losvd_roughness(b)
    r2 = compute_losvd_roughness(b * 42.0)
    assert abs(r1 - r2) < 1e-10


def test_n_peaks_scale_invariant():
    """Scaling the LOSVD amplitude should not change peak count."""
    b = _losvd_with_sigma(100.0)
    assert compute_losvd_n_peaks(b) == compute_losvd_n_peaks(b * 1000.0)


# ── Peak detection edge cases ────────────────────────────────────────────────

def test_n_peaks_negative_values_handled():
    """Negative LOSVD values (numerical noise) must not cause crashes."""
    b = _losvd_with_sigma(100.0) - 0.001  # some bins go negative
    result = compute_losvd_n_peaks(b)
    assert result >= 1


def test_n_peaks_very_short_array_returns_one():
    assert compute_losvd_n_peaks(np.array([0.5, 1.0, 0.5])) >= 1
    assert compute_losvd_n_peaks(np.array([1.0])) == 1


def test_n_peaks_two_equal_lobes():
    v = np.linspace(-500, 500, 201)
    b = (np.exp(-0.5 * ((v + 250) / 50.0) ** 2)
         + np.exp(-0.5 * ((v - 250) / 50.0) ** 2))
    assert compute_losvd_n_peaks(b) == 2


# ── Roughness boundary conditions ────────────────────────────────────────────

def test_roughness_finite_for_smooth_gaussian():
    b = _losvd_with_sigma(150.0)
    r = compute_losvd_roughness(b)
    assert np.isfinite(r)
    assert r >= 0.0


def test_roughness_returns_inf_for_all_zero():
    b = np.zeros(51)
    assert compute_losvd_roughness(b) == np.inf


def test_roughness_returns_inf_for_near_zero():
    b = np.full(51, 1e-12)
    assert compute_losvd_roughness(b) == np.inf
