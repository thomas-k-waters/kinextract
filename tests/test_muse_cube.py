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
    MuseCube, white_light_image, extract_aperture_spectrum, load_muse_cube, save_muse_cube_fits,
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
