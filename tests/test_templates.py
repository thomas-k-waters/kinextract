"""Tests for the instrumental LSF matching helpers in templates.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinextract.templates import (
    resolution_mismatch_sigma_A, convolve_gaussian_pixels, build_template_matrix_fortran,
    parse_emiles_filename, pack_templates_to_npz, load_packed_templates,
    build_template_matrix_from_npz,
)
from kinextract.io import read_emiles_fits
from kinextract.config import FitConfig

MUSE_DIR = Path(__file__).parent.parent / "examples" / "data" / "muse"


def test_template_coverage_mask_is_per_template_not_any(tmp_path):
    # Two templates covering different (overlapping-but-not-identical) ranges:
    # template A covers [0, 9), template B covers [2, 13) (coverage is
    # half-open -- see interp_template_tp_with_outside). Pixel x=1 is only
    # covered by A, x=10 only by B, and x=5 by both.
    xg = np.arange(13, dtype=float)

    def write_template(path, wave, flux):
        np.savetxt(path, np.column_stack([wave, flux]))

    wave_a = np.arange(0, 9, dtype=float)
    write_template(tmp_path / "a.dat", wave_a, np.full_like(wave_a, 2.0))
    wave_b = np.arange(2, 14, dtype=float)
    write_template(tmp_path / "b.dat", wave_b, np.full_like(wave_b, 3.0))

    T, T_err, outside_each, outside_all = build_template_matrix_fortran(
        xg, [str(tmp_path / "a.dat"), str(tmp_path / "b.dat")],
    )

    assert outside_each.shape == (13, 2)
    # Template A (column 0) has no coverage past x=8.
    assert outside_each[10, 0] and outside_each[11, 0]
    assert not outside_each[0, 0]
    # Template B (column 1) has no coverage before x=2.
    assert outside_each[0, 1] and outside_each[1, 1]
    assert not outside_each[10, 1]
    # A pixel covered by at least one template must NOT be flagged in outside_all,
    # even though it's outside one specific template's own range.
    assert not outside_all[1]    # covered by A only
    assert not outside_all[10]   # covered by B only
    assert not outside_all[5]    # covered by both
    # No pixel in [0, 12] is outside every template's range here.
    assert not outside_all.any()

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


# ── Packed .npz template grid: parsing, packing, loading ───────────────────

def _write_emiles_fits(path, imf_type, imf_slope, metallicity, age_gyr,
                        crval1=1680.2, cdelt1=0.9, npix=200):
    """Write a tiny synthetic MILES/E-MILES-convention FITS template file."""
    from astropy.io import fits
    sign = "m" if metallicity < 0 else "p"
    name = (f"E{imf_type}{imf_slope:.2f}Z{sign}{abs(metallicity):.2f}"
            f"T{age_gyr:07.4f}_iTp0.00_baseFe.fits")
    rng = np.random.default_rng(int(abs(metallicity * 100) + age_gyr * 10))
    flux = (1.0 + 0.1 * np.sin(np.linspace(0, 6 * np.pi, npix))
            + 0.01 * rng.standard_normal(npix)).astype(np.float32)
    hdu = fits.PrimaryHDU(flux)
    hdu.header["CRVAL1"] = crval1
    hdu.header["CDELT1"] = cdelt1
    hdu.writeto(Path(path) / name, overwrite=True)
    return name


def test_parse_emiles_filename_matches_standard_convention():
    meta = parse_emiles_filename("Ebi1.30Zp0.06T01.0000_iTp0.00_baseFe.fits")
    assert meta == {"imf_type": "bi", "imf_slope": 1.30, "metallicity": 0.06, "age_gyr": 1.0}

    meta_neg = parse_emiles_filename("Ebi0.30Zm1.79T05.0000_iTp0.00_baseFe.fits")
    assert meta_neg["metallicity"] == -1.79
    assert meta_neg["age_gyr"] == 5.0
    assert meta_neg["imf_slope"] == 0.30


def test_parse_emiles_filename_returns_none_for_nonconforming_name():
    assert parse_emiles_filename("B86133_av.dat") is None
    assert parse_emiles_filename("not_a_miles_file.fits") is None


def test_read_emiles_fits_reconstructs_wavelength_grid(tmp_path):
    _write_emiles_fits(tmp_path, "bi", 1.30, 0.06, 1.0, crval1=1680.2, cdelt1=0.9, npix=50)
    fits_path = next(tmp_path.glob("*.fits"))
    wave, flux = read_emiles_fits(str(fits_path))
    assert len(wave) == len(flux) == 50
    np.testing.assert_allclose(wave[0], 1680.2)
    np.testing.assert_allclose(wave[1] - wave[0], 0.9)


def test_pack_templates_to_npz_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError):
        pack_templates_to_npz(str(tmp_path / "out.npz"))
    with pytest.raises(ValueError):
        pack_templates_to_npz(
            str(tmp_path / "out.npz"),
            fits_dir=str(tmp_path), template_list_file="Tlist", template_dir=str(tmp_path),
        )


def test_pack_templates_to_npz_from_fits_round_trip(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    _write_emiles_fits(fits_dir, "bi", 1.30, -0.25, 1.0)
    _write_emiles_fits(fits_dir, "bi", 1.30, -0.25, 5.0)
    _write_emiles_fits(fits_dir, "bi", 1.30, 0.06, 1.0)

    out = pack_templates_to_npz(str(tmp_path / "grid.npz"), fits_dir=str(fits_dir))
    wave, flux, meta = load_packed_templates(out)

    assert flux.shape == (200, 3)
    assert sorted(meta["ages"].tolist()) == [1.0, 1.0, 5.0]
    assert sorted(meta["metals"].tolist()) == [-0.25, -0.25, 0.06]
    assert np.all(meta["imf_slope"] == 1.30)
    assert "MILES/E-MILES" in meta["source"]


def test_pack_templates_to_npz_from_dat_round_trip(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    wave = np.linspace(8400.0, 8800.0, 100)
    for i, name in enumerate(("star_a.dat", "star_b.dat")):
        flux = 1.0 + 0.1 * (i + 1) * np.sin(np.linspace(0, 4 * np.pi, 100))
        np.savetxt(lib_dir / name, np.column_stack([wave, flux]))
    tlist = lib_dir / "Tlist"
    tlist.write_text("star_a.dat\nstar_b.dat\n")

    out = pack_templates_to_npz(
        str(tmp_path / "grid.npz"), template_list_file=str(tlist), template_dir=str(lib_dir),
    )
    wave_out, flux, meta = load_packed_templates(out)
    # Same input grid for both templates here -> packing still oversamples
    # (see pack_templates_to_npz's .dat branch), so don't assume the exact
    # original point count, just that both templates share one grid.
    assert flux.shape[1] == 2
    assert len(wave_out) == flux.shape[0]
    assert np.all(np.isnan(meta["ages"]))
    assert np.all(np.isnan(meta["metals"]))
    assert set(meta["names"].tolist()) == {"star_a", "star_b"}
    assert "user template library" in meta["source"]


def test_load_packed_templates_select_filters_and_raises_on_missing_pair(tmp_path):
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    _write_emiles_fits(fits_dir, "bi", 1.30, -0.25, 1.0)
    _write_emiles_fits(fits_dir, "bi", 1.30, -0.25, 5.0)
    _write_emiles_fits(fits_dir, "bi", 1.30, 0.06, 1.0)
    out = pack_templates_to_npz(str(tmp_path / "grid.npz"), fits_dir=str(fits_dir))

    wave, flux, meta = load_packed_templates(out, select=[(1.0, -0.25)])
    assert flux.shape == (200, 1)
    assert meta["ages"][0] == 1.0 and meta["metals"][0] == -0.25

    with pytest.raises(ValueError, match=r"not found"):
        load_packed_templates(out, select=[(99.0, -0.25)])


def test_build_template_matrix_from_npz_matches_file_based(tmp_path):
    """The packed-npz path should closely reproduce the file-per-template
    path for the same underlying spectra (same median-normalization/
    interpolation core, see _interpolate_normalize_templates) -- but not
    bit-exactly for a real, heterogeneous .dat library like this one:
    individual MUSE stellar templates share the same nominal step but not
    the same pixel *phase* (see pack_templates_to_npz's .dat branch), so
    packing onto one common (16x oversampled) grid and then interpolating
    onto the fit grid composes two piecewise-linear interpolations at
    different phases -- not identical to one direct interpolation, though
    small (well under 1%) at this oversampling factor. This is an inherent
    property of packing a heterogeneous library into one shared-grid
    format, not a bug; the primary E-MILES/FITS packing path never hits
    this since those files already share one native grid exactly."""
    if not MUSE_DIR.exists():
        pytest.skip("bundled MUSE example data not found")

    tlist = MUSE_DIR / "Tlist"
    from kinextract.io import read_template_list
    paths = read_template_list(str(tlist), str(MUSE_DIR))[:3]

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    names = []
    for p in paths:
        import shutil
        shutil.copy(p, lib_dir / Path(p).name)
        names.append(Path(p).name)
    (lib_dir / "Tlist").write_text("\n".join(names) + "\n")

    npz_path = pack_templates_to_npz(
        str(tmp_path / "grid.npz"), template_list_file=str(lib_dir / "Tlist"), template_dir=str(lib_dir),
    )

    xg = np.linspace(8400.0, 8750.0, 300)
    T_file, T_err_file, outside_each_file, outside_all_file = build_template_matrix_fortran(xg, paths)
    T_npz, T_err_npz, outside_each_npz, outside_all_npz = build_template_matrix_from_npz(xg, npz_path)

    np.testing.assert_allclose(T_npz, T_file, rtol=1e-2, atol=1e-3)
    np.testing.assert_array_equal(outside_each_npz, outside_each_file)
    np.testing.assert_array_equal(outside_all_npz, outside_all_file)


def test_fitconfig_template_npz_file_and_select_default_to_none():
    cfg = FitConfig()
    assert cfg.template_npz_file is None
    assert cfg.template_npz_select is None


def test_make_fit_state_with_template_npz_file_end_to_end(tmp_path):
    """A real make_fit_state() call using template_npz_file= (instead of
    template_list_file/template_dir) should closely reproduce the template
    matrix from the file-based path, on the bundled real MUSE spectrum --
    see test_build_template_matrix_from_npz_matches_file_based's docstring
    for why this is not bit-exact for a real, phase-heterogeneous library."""
    if not MUSE_DIR.exists():
        pytest.skip("bundled MUSE example data not found")
    from kinextract.spectrum import make_fit_state

    spec_file = MUSE_DIR / "bin0105sp.spec"
    data = np.loadtxt(spec_file)
    ferr = data[:, 1] / 50.0

    npz_path = pack_templates_to_npz(
        str(tmp_path / "grid.npz"),
        template_list_file=str(MUSE_DIR / "Tlist"), template_dir=str(MUSE_DIR),
    )

    base_kwargs = dict(
        wavemin_full=4750.0, step=1.25,
        wavefitmin=8400.0, wavefitmax=8750.0,
        zgal=0.001556,
        losvd_vmin=-300.0, losvd_vmax=300.0,
        fit_continuum=False,
        use_spectrum_errors=False,
        sigl=100.0, clean=False,
    )
    cfg_file = FitConfig(template_list_file=str(MUSE_DIR / "Tlist"), template_dir=str(MUSE_DIR), **base_kwargs)
    cfg_npz = FitConfig(template_npz_file=str(npz_path), **base_kwargs)

    st_file, _ = make_fit_state(cfg_file, gal_file=str(spec_file), gal_errors=ferr)
    st_npz, _ = make_fit_state(cfg_npz, gal_file=str(spec_file), gal_errors=ferr)

    np.testing.assert_allclose(st_npz.t, st_file.t, rtol=1e-2, atol=1e-3)
