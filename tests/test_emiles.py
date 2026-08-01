"""Tests for kinextract.emiles -- uses a small, fabricated set of
MILES-convention FITS files (not the real ~1.4GB E-MILES release) so
these run fast and need no external data. Mirrors the OG (non-lite)
kinextract package's own `_write_emiles_fits` synthetic-fixture pattern
for the same reason: round-trip testing the packing/loading logic
doesn't need real stellar spectra, just the right file format and
metadata.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from kinextract.emiles import (
    parse_emiles_filename, pack_emiles_templates, load_emiles_templates,
)


def _write_fake_emiles_fits(directory, imf_type="bi", imf_slope=1.30, metal=0.06, age=1.0,
                             wave_min=3000.0, cdelt=0.9, n_pix=2000):
    """Writes one minimal, MILES-convention FITS file with a smooth,
    strictly-positive synthetic "spectrum" (a shallow absorption-like dip,
    not real stellar physics -- irrelevant for testing packing/loading).
    """
    sign = "m" if metal < 0 else "p"
    name = f"E{imf_type}{imf_slope:.2f}Z{sign}{abs(metal):.2f}T{age:09.4f}_iTp0.00_baseFe.fits"
    wave = wave_min + np.arange(n_pix) * cdelt
    flux = 1.0 - 0.3 * np.exp(-0.5 * ((wave - (wave_min + wave[-1]) / 2) / 50.0) ** 2)
    hdu = fits.PrimaryHDU(flux.astype(np.float32))
    hdu.header["CRVAL1"] = wave_min
    hdu.header["CDELT1"] = cdelt
    path = os.path.join(directory, name)
    hdu.writeto(path, overwrite=True)
    return name


def test_parse_emiles_filename_extracts_correct_metadata():
    meta = parse_emiles_filename("Ebi1.30Zp0.06T01.0000_iTp0.00_baseFe.fits")
    assert meta == dict(imf_type="bi", imf_slope=1.30, metal=0.06, age=1.0)


def test_parse_emiles_filename_handles_negative_metallicity():
    meta = parse_emiles_filename("Ebi1.30Zm0.66T14.0000_iTp0.00_baseFe.fits")
    assert meta["metal"] == pytest.approx(-0.66)


def test_parse_emiles_filename_returns_none_for_non_conforming_name():
    assert parse_emiles_filename("not_an_emiles_file.fits") is None


def test_pack_and_load_roundtrip(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    grid = [(z, t) for z in (-0.66, 0.06, 0.40) for t in (1.0, 5.0, 14.0)]
    for metal, age in grid:
        _write_fake_emiles_fits(str(fits_dir), metal=metal, age=age)

    out = str(tmp_path / "packed.npz")
    pack_emiles_templates(str(fits_dir), out, target_lsf_fwhm=3.0, template_fwhm=2.51, verbose=False)

    templates, wavelength, names, ages, metals = load_emiles_templates(out)
    assert len(templates) == len(grid) == 9
    assert len(wavelength) == len(templates[0])
    assert set(zip(metals.tolist(), ages.tolist())) == set(grid)
    for t in templates:
        assert np.all(np.isfinite(t))
        assert np.all(t > 0)  # LSF-matched + normalized, should stay strictly positive


def test_pack_filters_by_imf_and_skips_non_conforming_files(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    _write_fake_emiles_fits(str(fits_dir), imf_slope=1.30, metal=0.0, age=1.0)
    _write_fake_emiles_fits(str(fits_dir), imf_slope=1.80, metal=0.0, age=1.0)  # different IMF, should be skipped
    with open(fits_dir / "readme.txt", "w") as f:
        f.write("not a fits file")

    out = str(tmp_path / "packed.npz")
    pack_emiles_templates(str(fits_dir), out, imf_slope="1.30", target_lsf_fwhm=3.0, verbose=False)
    templates, _, _, _, _ = load_emiles_templates(out)
    assert len(templates) == 1


def test_load_with_select_restricts_to_requested_pairs(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    grid = [(z, t) for z in (-0.66, 0.06, 0.40) for t in (1.0, 5.0, 14.0)]
    for metal, age in grid:
        _write_fake_emiles_fits(str(fits_dir), metal=metal, age=age)
    out = str(tmp_path / "packed.npz")
    pack_emiles_templates(str(fits_dir), out, target_lsf_fwhm=3.0, verbose=False)

    templates, _, names, ages, metals = load_emiles_templates(out, select=[(1.0, 0.06), (14.0, 0.40)])
    assert len(templates) == 2
    assert set(zip(metals.tolist(), ages.tolist())) == {(0.06, 1.0), (0.40, 14.0)}


def test_load_with_select_raises_clear_error_for_missing_pair(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    _write_fake_emiles_fits(str(fits_dir), metal=0.06, age=1.0)
    out = str(tmp_path / "packed.npz")
    pack_emiles_templates(str(fits_dir), out, target_lsf_fwhm=3.0, verbose=False)

    with pytest.raises(ValueError, match="no packed template matches"):
        load_emiles_templates(out, select=[(99.0, 99.0)])


def test_pack_raises_on_wavelength_grid_mismatch(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    _write_fake_emiles_fits(str(fits_dir), metal=0.06, age=1.0, cdelt=0.9)
    _write_fake_emiles_fits(str(fits_dir), metal=0.40, age=1.0, cdelt=1.1)  # different grid

    out = str(tmp_path / "packed.npz")
    with pytest.raises(ValueError, match="different wavelength grid"):
        pack_emiles_templates(str(fits_dir), out, target_lsf_fwhm=3.0, verbose=False)


def test_pack_respects_ages_and_metals_filters(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    grid = [(z, t) for z in (-0.66, 0.06, 0.40) for t in (1.0, 5.0, 14.0)]
    for metal, age in grid:
        _write_fake_emiles_fits(str(fits_dir), metal=metal, age=age)
    out = str(tmp_path / "packed.npz")
    pack_emiles_templates(str(fits_dir), out, ages_gyr=[1.0, 14.0], metals=[0.06],
                           target_lsf_fwhm=3.0, verbose=False)
    templates, _, _, ages, metals = load_emiles_templates(out)
    assert len(templates) == 2
    assert set(ages.tolist()) == {1.0, 14.0}
    assert set(metals.tolist()) == {0.06}


def test_pack_builds_ready_to_use_template_stack(tmp_path):
    """The whole point: output should plug directly into
    build_template_stack alongside any other template source."""
    from kinextract import build_template_stack, FitConfig

    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    for metal, age in [(0.06, 1.0), (0.40, 5.0)]:
        _write_fake_emiles_fits(str(fits_dir), metal=metal, age=age, wave_min=8000.0, n_pix=3000, cdelt=0.5)
    out = str(tmp_path / "packed.npz")
    pack_emiles_templates(str(fits_dir), out, target_lsf_fwhm=2.6, template_fwhm=2.51, verbose=False)
    templates, wavelength, names, ages, metals = load_emiles_templates(out)

    wave_fit = np.linspace(8400.0, 8800.0, 200)
    cfg = FitConfig(vmin=-700.0, vmax=700.0)
    v_grid = cfg.build_velocity_grid()
    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    assert A_stack.shape == (2, 200, len(v_grid))
    assert np.all(np.isfinite(A_stack))
