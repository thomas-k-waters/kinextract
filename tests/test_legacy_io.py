"""Tests for kinextract.legacy_io -- the SCO Fortran framework's own
per-bin output file writers (.fit/.ascii/.rms/.mcfit, and now
<bin>spmc_in). Checks format (row/column counts, header structure), not
physical correctness of the fit itself (that's covered elsewhere, e.g.
test_multi_template.py) -- except for write_spmc_in's own regression
tests below, which check numerical fidelity against a real Fortran run."""
from __future__ import annotations

import os

import numpy as np
import pytest

from kinextract.legacy_io import write_fit_outputs, write_spmc_in, legacy_transvd_curve


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


def test_ascii_file_has_six_columns_one_row_per_pixel(tmp_path):
    """pspecfit.f's own source (read(2,*) x1,x2,x3,x4,x5,i6) wants 6 columns:
    wave, data, model, continuum, continuum (diagnostic copy), flag."""
    v_grid = np.linspace(-50.0, 50.0, 5)
    wave_fit = np.linspace(8400.0, 8420.0, 12)
    flux_fit = np.linspace(0.9, 1.1, 12)
    model = np.linspace(0.91, 1.09, 12)
    result = _mock_result(n_bins=5, n_pix=12)

    paths = write_fit_outputs("bin0501", result, v_grid, wave_fit, flux_fit, model, str(tmp_path))
    lines = open(paths["ascii"]).read().splitlines()
    assert len(lines) == 12
    for line in lines:
        assert len(line.split()) == 6


def test_ascii_file_continuum_columns_are_zero_when_no_continuum_given(tmp_path):
    v_grid = np.linspace(-50.0, 50.0, 5)
    wave_fit = np.linspace(8400.0, 8420.0, 6)
    flux_fit = np.ones(6)
    model = np.ones(6)
    result = _mock_result(n_bins=5, n_pix=6)

    paths = write_fit_outputs("bin0502", result, v_grid, wave_fit, flux_fit, model, str(tmp_path))
    for line in open(paths["ascii"]).read().splitlines():
        fields = line.split()
        assert float(fields[3]) == 0.0
        assert float(fields[4]) == 0.0


def test_ascii_file_continuum_columns_reflect_given_continuum(tmp_path):
    v_grid = np.linspace(-50.0, 50.0, 5)
    wave_fit = np.linspace(8400.0, 8420.0, 6)
    flux_fit = np.ones(6)
    model = np.ones(6)
    continuum = np.linspace(0.1, 0.2, 6)
    result = _mock_result(n_bins=5, n_pix=6)

    paths = write_fit_outputs("bin0503", result, v_grid, wave_fit, flux_fit, model,
                               str(tmp_path), continuum=continuum)
    lines = open(paths["ascii"]).read().splitlines()
    for line, c in zip(lines, continuum):
        fields = line.split()
        assert float(fields[3]) == pytest.approx(c, abs=1e-3)
        assert float(fields[4]) == pytest.approx(c, abs=1e-3)


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


# --- write_spmc_in / legacy_transvd_curve -----------------------------------------
#
# Fixtures under fixtures/legacy_transvd/ are real data: bin0105spmc.mcfit2 comes
# from kinextract_debug_suite (a real MUSE bin's actual .mcfit2). The two
# *.groundtruth files were generated by rebuilding the real transvd.f (+ the real
# numrec_f/spline.f, splint.f it links against) with gfortran and running it, then
# the real fix_losvd_errs.py, on that real .mcfit2 -- once with ivert=0 (the
# "noninvert" file) and once with ivert=1 (the "invert" file), isym=0 both times
# (this framework never sets isym=1). These lock in a byte-for-byte-verified port.

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "legacy_transvd")


def _load_mcfit2_curve():
    data = np.loadtxt(os.path.join(_FIXTURE_DIR, "bin0105spmc.mcfit2"))
    v, y, y16, y84 = data.T
    return v, y, y16, y84


def test_legacy_transvd_curve_matches_real_fortran_noninvert():
    v, y, y16, y84 = _load_mcfit2_curve()
    vel, losvd, lo, hi = legacy_transvd_curve(v, y, y16, y84, invert=False)

    gt = np.loadtxt(os.path.join(_FIXTURE_DIR, "bin0105spmc_in_noninvert.groundtruth"))
    gv, gy, glo, ghi = gt.T

    assert np.max(np.abs(vel - gv)) < 1e-4
    # one known point (the exact right-edge sample) diverges from a Fortran quirk --
    # xlinint's half-open interval test leaves that one yp stale; see
    # legacy_transvd_curve's own docstring. Every other point matches near machine
    # precision (float32 Fortran vs. float64 numpy).
    assert np.sum(np.abs(losvd - gy) > 1e-3) <= 1
    assert np.sum(np.abs(lo - glo) > 1e-3) <= 1
    assert np.sum(np.abs(hi - ghi) > 1e-3) <= 1


def test_legacy_transvd_curve_matches_real_fortran_invert():
    v, y, y16, y84 = _load_mcfit2_curve()
    vel, losvd, lo, hi = legacy_transvd_curve(v, y, y16, y84, invert=True)

    gt = np.loadtxt(os.path.join(_FIXTURE_DIR, "bin0105spmc_in_invert.groundtruth"))
    gv, gy, glo, ghi = gt.T

    assert np.max(np.abs(vel - gv)) < 1e-4
    assert np.max(np.abs(losvd - gy)) < 1e-3
    assert np.max(np.abs(lo - glo)) < 1e-3
    assert np.max(np.abs(hi - ghi)) < 1e-3


def test_write_spmc_in_infers_invert_from_bin_vs_bnn_prefix(tmp_path):
    """finalizeextraction calls `rtransvd <bin> 1` for non-bnn stems (invert=True)
    and `rtransvd <bin> -1` for bnn* stems (invert=False, since transvd.f's own
    check is `ivert.eq.1`). write_spmc_in must reproduce that split without the
    caller having to know about it."""
    v, y, y16, y84 = _load_mcfit2_curve()

    path_bin = write_spmc_in("bin0105", v, y, y16, y84, str(tmp_path))
    path_bnn = write_spmc_in("bnn0105", v, y, y16, y84, str(tmp_path))

    curve_bin = np.loadtxt(path_bin)
    curve_bnn = np.loadtxt(path_bnn)
    # different invert settings must give different LOSVD curves for an asymmetric input
    assert not np.allclose(curve_bin[:, 1], curve_bnn[:, 1])

    gt_invert = np.loadtxt(os.path.join(_FIXTURE_DIR, "bin0105spmc_in_invert.groundtruth"))
    gt_noninvert = np.loadtxt(os.path.join(_FIXTURE_DIR, "bin0105spmc_in_noninvert.groundtruth"))
    assert np.max(np.abs(curve_bin[:, 1] - gt_invert[:, 1])) < 1e-3
    # noninvert hits the one known Fortran stale-value edge point (see
    # legacy_transvd_curve's own docstring) -- allow that single point to differ.
    assert np.sum(np.abs(curve_bnn[:, 1] - gt_noninvert[:, 1]) > 1e-3) <= 1


def test_write_spmc_in_file_naming_and_shape(tmp_path):
    v, y, y16, y84 = _load_mcfit2_curve()
    path = write_spmc_in("bin0105", v, y, y16, y84, str(tmp_path))
    assert path.endswith("bin0105spmc_in")
    assert os.path.exists(path)

    data = np.loadtxt(path)
    assert data.shape == (1000, 4)
    # vel column spans [v_grid[0], -v_grid[0]] (transvd.f's own resampling range)
    assert np.isclose(data[0, 0], v[0], atol=1e-3)
    assert np.isclose(data[-1, 0], -v[0], atol=1e-3)


def test_write_spmc_in_creates_outdir_if_missing(tmp_path):
    v, y, y16, y84 = _load_mcfit2_curve()
    outdir = tmp_path / "nested" / "does" / "not" / "exist"
    path = write_spmc_in("bin0105", v, y, y16, y84, str(outdir))
    assert outdir.exists()
    assert os.path.exists(path)


# --- write_spmc_in_laplace / transvd_bounds_laplace ------------------------------------------------

from kinextract.legacy_io import spmc_side_inverts, transvd_bounds_laplace  # noqa: E402


def test_spmc_side_from_stem_ignores_source_prefix():
    assert spmc_side_inverts("bin0105") and spmc_side_inverts("mwbin0901") and spmc_side_inverts("mnbin1203")
    assert not spmc_side_inverts("bnn0704") and not spmc_side_inverts("mwbnn0704")
    with pytest.raises(ValueError):
        spmc_side_inverts("bin07")


def _laplace_like(sigma=150.0, rel=0.02):
    """A Gaussian LOSVD whose percentiles scale with its amplitude, as a log-parameterized Laplace posterior does."""
    v = np.arange(-1500.0, 1500.0 + 1e-9, 25.0)
    y = np.exp(-0.5 * (v / sigma) ** 2)
    return v, y, y * (1 - rel), y * (1 + rel)


def test_noise_floor_peak_bounds_the_wing_errors():
    v, y, y16, y84 = _laplace_like()
    vel, L, lo, hi = transvd_bounds_laplace(v, y, y16, y84, invert=False)
    ip = np.argmax(L)
    w_peak = hi[ip] - L[ip]
    assert np.all(hi - L >= w_peak * (1 - 1e-12))          # no width below the peak's
    assert np.all(lo >= 0) and np.all(lo <= L)
    assert np.isclose(w_peak, (y84 - y).max() / y.sum() / np.sqrt(133.3 / 25.0), rtol=0.05)
    vel0, L0, lo0, hi0 = transvd_bounds_laplace(v, y, y16, y84, invert=False, noise_floor=None)
    assert np.array_equal(L, L0)                            # the floor changes only the errors
    assert (hi0 - L0).min() < 1e-10 * w_peak               # without it the wing errors all but vanish


def test_invert_mirrors_about_zero():
    v, y, y16, y84 = _laplace_like()
    y = y * (1 + 0.3 * np.tanh(v / 300.0)); y84 = y * 1.02; y16 = y * 0.98
    _, Lf, _, _ = transvd_bounds_laplace(v, y, y16, y84, invert=False)
    _, Li, _, _ = transvd_bounds_laplace(v, y, y16, y84, invert=True)
    assert np.allclose(Li, Lf[::-1], atol=1e-12)


def _posterior_draws(n=400, sigma=250.0, seed=1):
    """Correlated posterior draws of a Gaussian LOSVD: random width and center per draw, as a smooth posterior has."""
    rng = np.random.default_rng(seed)
    v = np.arange(-2000.0, 2000.0 + 1e-9, 8.0)
    draws = np.array([np.exp(-0.5 * ((v - rng.normal(0, 15)) / (sigma * (1 + rng.normal(0, 0.04)))) ** 2)
                      for _ in range(n)])
    draws /= np.trapezoid(draws, v, axis=1)[:, None]
    return v, draws


def _model_binned(vel, L, lo, hi, centers, half):
    err = np.maximum(hi - L, (hi - lo) / 2)
    return (np.array([L[(vel >= c - half) & (vel < c + half)].sum() for c in centers]) / L.sum(),
            np.array([err[(vel >= c - half) & (vel < c + half)].sum() for c in centers]) / L.sum())


def test_model_bins_match_the_posterior_scatter_of_the_binned_losvd():
    v, draws = _posterior_draws()
    med, p16, p84 = np.median(draws, 0), np.percentile(draws, 16, 0), np.percentile(draws, 84, 0)
    centers, half = np.linspace(-1800, 1800, 13), 150.0
    vel, L, lo, hi = transvd_bounds_laplace(v, med, p16, p84, invert=False, vmax_out=1950.0,
                                            b_samples=draws, model_bins=(centers, half))
    S, E = _model_binned(vel, L, lo, hi, centers, half)
    resampled = np.array([np.interp(vel, v, d) for d in draws])
    target = np.array([resampled[:, (vel >= c - half) & (vel < c + half)].sum(1) / resampled.sum(1)
                       for c in centers]).std(axis=1, ddof=1)
    light = S > 0.01
    E0 = _model_binned(*transvd_bounds_laplace(v, med, p16, p84, invert=False, vmax_out=1950.0), centers, half)[1]
    assert np.allclose(E[light], np.maximum(target, E0)[light], rtol=0.15)
    assert np.all(E >= E0 * (1 - 1e-9))  # only ever widened, so the noise floor survives in faint bins
    # the default call is unchanged
    a = transvd_bounds_laplace(v, med, p16, p84, invert=False, vmax_out=1950.0)
    b = transvd_bounds_laplace(v, med, p16, p84, invert=False, vmax_out=1950.0, b_samples=draws, model_bins=None)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_model_bins_require_samples():
    v, draws = _posterior_draws(n=20)
    med = np.median(draws, 0)
    with pytest.raises(ValueError):
        transvd_bounds_laplace(v, med, med * 0.98, med * 1.02, invert=False, model_bins=(np.zeros(1), 150.0))
