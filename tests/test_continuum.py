"""Tests for standalone continuum utilities (continuum.py).

Covers standalone mathematical properties — no FitState needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract.continuum import (
    asymmetric_least_squares_continuum,
    grow_boolean_mask,
    grow_boolean_mask_A,
    robust_sigma,
)


# ── robust_sigma ─────────────────────────────────────────────────────────────

def test_robust_sigma_gaussian():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 5.0, 10_000)
    assert abs(robust_sigma(x) - 5.0) < 0.2


def test_robust_sigma_outlier_resistant():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 1000)
    x_with_outliers = np.concatenate([x, [1000.0, -1000.0]])
    assert abs(robust_sigma(x_with_outliers) - 1.0) < 0.1


def test_robust_sigma_constant_array():
    x = np.ones(100) * 3.0
    # All values identical → MAD = 0 → sigma = 0
    assert robust_sigma(x) == 0.0


# ── grow_boolean_mask ────────────────────────────────────────────────────────

def test_grow_boolean_mask_grows_by_n():
    m = np.zeros(20, dtype=bool)
    m[10] = True
    grown = grow_boolean_mask(m, grow_pix=3)
    assert grown[7:14].all()
    assert not grown[6]
    assert not grown[14]


def test_grow_boolean_mask_zero_pixels_unchanged():
    m = np.array([False, True, False, False, True, False])
    grown = grow_boolean_mask(m, grow_pix=0)
    np.testing.assert_array_equal(grown, m)


def test_grow_boolean_mask_all_true_stays_all_true():
    m = np.ones(10, dtype=bool)
    grown = grow_boolean_mask(m, grow_pix=5)
    assert grown.all()


# ── ALS continuum ────────────────────────────────────────────────────────────

def _make_spectrum_with_lines(n=500, sigma_noise=0.01):
    x = np.linspace(8400, 8800, n)
    cont = 1.0 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, n))
    # Add absorption lines
    for cen in [8500, 8600, 8700]:
        cont -= 0.3 * np.exp(-0.5 * ((x - cen) / 4.0) ** 2)
    rng = np.random.default_rng(42)
    y = cont + rng.normal(0.0, sigma_noise, n)
    return x, y, cont


def test_als_continuum_above_absorption():
    """ALS continuum must lie above absorption lines."""
    x, y, true_cont = _make_spectrum_with_lines()
    weights = np.ones(len(x))
    cont = asymmetric_least_squares_continuum(y, weights, lam=1e4, p=0.01, niter=20)
    # At the line centers the continuum should be close to the true continuum
    for cen in [8500, 8600, 8700]:
        idx = np.argmin(np.abs(x - cen))
        assert cont[idx] > y[idx], f"Continuum below absorption at x={cen}"


def test_als_continuum_same_length_as_input():
    y = np.ones(300) + 0.1 * np.sin(np.linspace(0, 6, 300))
    w = np.ones(300)
    cont = asymmetric_least_squares_continuum(y, w, lam=1e3, p=0.05, niter=10)
    assert cont.shape == y.shape


def test_als_large_lam_gives_flat_continuum():
    """Very large lambda → maximally smooth (nearly flat) continuum."""
    y = np.ones(200)
    y[100] -= 0.5  # absorption dip
    w = np.ones(200)
    cont_strong = asymmetric_least_squares_continuum(y, w, lam=1e8, p=0.01, niter=15)
    cont_weak = asymmetric_least_squares_continuum(y, w, lam=1.0, p=0.01, niter=15)
    # Strong regularization → flatter (less variation)
    assert np.std(cont_strong) < np.std(cont_weak)
