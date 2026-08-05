"""Tests for kinextract.legacy_io -- the SCO Fortran framework's own
per-bin output file writers (.fit/.ascii/.rms/.mcfit). Checks format
(row/column counts, header structure), not physical correctness of the
fit itself (that's covered elsewhere, e.g. test_multi_template.py)."""
from __future__ import annotations

import numpy as np

from kinextract.legacy_io import write_fit_outputs


def _mock_result(n_bins=5, n_pix=20, clipped=None):
    rng = np.random.default_rng(0)
    b = np.abs(rng.normal(size=n_bins))
    result = dict(b=b, w=np.array([1.0]))
    if clipped is not None:
        result["clipped_mask"] = clipped
    return result


def test_write_fit_outputs_creates_all_four_files(tmp_path):
    v_grid = np.linspace(-100.0, 100.0, 5)
    wave_fit = np.linspace(8400.0, 8420.0, 20)
    flux_fit = np.ones(20)
    model = np.ones(20) * 0.99
    result = _mock_result(n_bins=5, n_pix=20)

    paths = write_fit_outputs("bin0301", result, v_grid, wave_fit, flux_fit, model, str(tmp_path))

    assert set(paths.keys()) == {"fit", "ascii", "rms", "mcfit"}
    for p in paths.values():
        assert (tmp_path / __import__("os").path.basename(p)).exists()
    assert paths["fit"].endswith("bin0301sp.fit")
    assert paths["ascii"].endswith("bin0301sp.ascii")


def test_mcfit_uses_spmc_stem_not_sp_stem(tmp_path):
    """pallmc.f opens a bin's velocity-grid reference via
    file1(1:nfile-3)//'mcfit' where file1 is that bin's own <bin>spmc.sim
    pallmc_inlist entry -- i.e. it looks for <bin>spmc.mcfit specifically,
    never <bin>sp.mcfit. Locks in the naming fix."""
    v_grid = np.linspace(-10.0, 10.0, 3)
    wave_fit = np.linspace(8400.0, 8410.0, 4)
    flux_fit = np.ones(4)
    model = np.ones(4)
    result = _mock_result(n_bins=3, n_pix=4)

    paths = write_fit_outputs("bin0901", result, v_grid, wave_fit, flux_fit, model, str(tmp_path))
    assert paths["mcfit"].endswith("bin0901spmc.mcfit")
    assert not paths["mcfit"].endswith("bin0901sp.mcfit")


def test_fit_file_has_header_line_then_one_row_per_velocity_point(tmp_path):
    v_grid = np.linspace(-100.0, 100.0, 7)
    wave_fit = np.linspace(8400.0, 8420.0, 10)
    flux_fit = np.ones(10)
    model = np.ones(10)
    result = _mock_result(n_bins=7, n_pix=10)

    paths = write_fit_outputs("bin0101", result, v_grid, wave_fit, flux_fit, model, str(tmp_path))
    lines = open(paths["fit"]).read().splitlines()
    assert len(lines) == 1 + 7  # header + one row per velocity point

    header = lines[0].split()
    assert len(header) == 4
    assert float(header[2]) == 666.0  # the dead sentinel, verified from Fortran source
    assert int(header[3]) == 10  # n_good pixels

    row1 = lines[1].split()
    assert int(row1[0]) == 1  # 1-based index
    assert float(row1[1]) == v_grid[0]


def test_ascii_file_has_five_columns_one_row_per_pixel(tmp_path):
    v_grid = np.linspace(-50.0, 50.0, 5)
    wave_fit = np.linspace(8400.0, 8420.0, 12)
    flux_fit = np.linspace(0.9, 1.1, 12)
    model = np.linspace(0.91, 1.09, 12)
    result = _mock_result(n_bins=5, n_pix=12)

    paths = write_fit_outputs("bin0501", result, v_grid, wave_fit, flux_fit, model, str(tmp_path))
    lines = open(paths["ascii"]).read().splitlines()
    assert len(lines) == 12
    for line in lines:
        assert len(line.split()) == 5


def test_ascii_file_bad_pixel_flag_reflects_clipped_mask(tmp_path):
    v_grid = np.linspace(-50.0, 50.0, 5)
    wave_fit = np.linspace(8400.0, 8420.0, 6)
    flux_fit = np.ones(6)
    model = np.ones(6)
    clipped = np.array([False, False, True, False, True, False])
    result = _mock_result(n_bins=5, n_pix=6, clipped=clipped)

    paths = write_fit_outputs("bin0601", result, v_grid, wave_fit, flux_fit, model, str(tmp_path))
    flags = [int(line.split()[-1]) for line in open(paths["ascii"]).read().splitlines()]
    assert flags == [0, 0, 1, 0, 1, 0]


def test_rms_file_is_one_line_four_fields(tmp_path):
    v_grid = np.linspace(-50.0, 50.0, 5)
    wave_fit = np.linspace(8400.0, 8420.0, 8)
    flux_fit = np.ones(8)
    model = np.full(8, 0.98)
    result = _mock_result(n_bins=5, n_pix=8)

    paths = write_fit_outputs("bin0701", result, v_grid, wave_fit, flux_fit, model, str(tmp_path))
    lines = open(paths["rms"]).read().splitlines()
    assert len(lines) == 1
    fields = lines[0].split()
    assert len(fields) == 4
    assert int(fields[1]) == 8
    assert float(fields[2]) == 0.0 and float(fields[3]) == 0.0  # coff/coff2 -- no kinextract equivalent


def test_mcfit_file_header_and_velocity_column(tmp_path):
    v_grid = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    wave_fit = np.linspace(8400.0, 8420.0, 8)
    flux_fit = np.ones(8)
    model = np.ones(8)
    result = _mock_result(n_bins=5, n_pix=8)

    paths = write_fit_outputs("bin0801", result, v_grid, wave_fit, flux_fit, model,
                               str(tmp_path), n_templates=3)
    lines = open(paths["mcfit"]).read().splitlines()
    assert len(lines) == 1 + 5
    nl, ns = (int(x) for x in lines[0].split())
    assert nl == 5 and ns == 3
    for line, v in zip(lines[1:], v_grid):
        fields = line.split()
        assert len(fields) == 6
        assert float(fields[0]) == v


def test_write_fit_outputs_creates_outdir_if_missing(tmp_path):
    outdir = tmp_path / "nested" / "does" / "not" / "exist"
    v_grid = np.linspace(-10.0, 10.0, 3)
    wave_fit = np.linspace(8400.0, 8410.0, 4)
    flux_fit = np.ones(4)
    model = np.ones(4)
    result = _mock_result(n_bins=3, n_pix=4)

    paths = write_fit_outputs("bin0201", result, v_grid, wave_fit, flux_fit, model, str(outdir))
    assert outdir.exists()
    assert (outdir / "bin0201sp.fit").exists()
