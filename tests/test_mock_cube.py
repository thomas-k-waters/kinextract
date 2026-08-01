"""Tests for kinextract.mock_cube -- the synthetic Sersic-profile
MUSE-like cube generator. See that module's own docstring for why the
kinematics are spatially constant and the noise is independent Gaussian
at a fixed floor (S-N falls with radius purely from the Sersic profile's
own brightness decline).
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract.mock_cube import build_mock_cube
from kinextract.muse_cube import white_light_image, extract_aperture_spectrum

EMILES_GRID = "src/kinextract/template_data/emiles_muse_grid.npz"


def test_build_mock_cube_shape_and_wavelength_range():
    cube = build_mock_cube(EMILES_GRID, ny=16, nx=20, seed=0)
    assert cube.shape == (16, 20)
    assert cube.flux.shape == (len(cube.wave), 16, 20)
    assert cube.error.shape == cube.flux.shape
    assert cube.wave.min() >= 8400.0 and cube.wave.max() <= 8800.0


def test_truth_dict_has_expected_keys_and_normalized_weights():
    cube = build_mock_cube(EMILES_GRID, ny=12, nx=12, seed=0)
    assert cube.truth is not None
    for key in ("w_true", "b_true", "v_grid", "template_select", "true_v", "true_sigma"):
        assert key in cube.truth
    assert cube.truth["w_true"].sum() == pytest.approx(1.0)


def test_surface_brightness_peaks_at_center_and_falls_off():
    # single-pixel comparisons are noise-dominated far from center (fixed noise floor,
    # vanishing signal), so this compares small *aggregated* apertures at increasing
    # radii instead of raw adjacent-pixel diffs, which averages the noise down enough
    # to see the underlying monotonic Sersic falloff clearly.
    cube = build_mock_cube(EMILES_GRID, ny=40, nx=40, seed=0)
    wl = white_light_image(cube)
    cy, cx = 20, 20
    apertures = [wl[cy - 1:cy + 2, cx - 1:cx + 2].sum(),   # r ~ 0-1 px
                 wl[cy - 1:cy + 2, cx + 8:cx + 11].sum(),  # r ~ 9 px
                 wl[cy - 1:cy + 2, cx + 18:cx + 21].sum()] # r ~ 19 px (near the edge)
    assert apertures[0] > apertures[1] > apertures[2]


def test_snr_declines_with_radius():
    cube = build_mock_cube(EMILES_GRID, ny=50, nx=50, seed=0)
    center_mask = np.zeros((50, 50), dtype=bool)
    center_mask[24:27, 24:27] = True
    outer_mask = np.zeros((50, 50), dtype=bool)
    outer_mask[0:4, 0:4] = True

    _, flux_c, err_c = extract_aperture_spectrum(cube, center_mask)
    _, flux_o, err_o = extract_aperture_spectrum(cube, outer_mask)
    snr_center = np.median(flux_c) / np.median(err_c)
    snr_outer = np.median(flux_o) / np.median(err_o)
    assert snr_center > 10 * snr_outer  # should be a large, clear difference, not a marginal one


def test_deterministic_given_same_seed():
    cube1 = build_mock_cube(EMILES_GRID, ny=10, nx=10, seed=42)
    cube2 = build_mock_cube(EMILES_GRID, ny=10, nx=10, seed=42)
    np.testing.assert_array_equal(cube1.flux, cube2.flux)


def test_different_seeds_give_different_noise_realizations():
    cube1 = build_mock_cube(EMILES_GRID, ny=10, nx=10, seed=1)
    cube2 = build_mock_cube(EMILES_GRID, ny=10, nx=10, seed=2)
    assert not np.allclose(cube1.flux, cube2.flux)


def test_custom_template_select_and_weights_are_recorded_in_truth():
    select = [(2.0, 0.06), (11.0, -0.66)]
    w = np.array([0.4, 0.6])
    cube = build_mock_cube(EMILES_GRID, ny=8, nx=8, template_select=select, true_w=w, seed=0)
    assert cube.truth["template_select"] == select
    np.testing.assert_allclose(cube.truth["w_true"], w)
