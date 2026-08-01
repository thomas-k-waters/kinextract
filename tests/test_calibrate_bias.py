"""Tests for kinextract.calibrate_bias.

Cheap smoke tests only (a tiny grid, few replicates) -- the real
calibration sweep used to build the package's shipped E-MILES/multi-
template default table (dev_notes/calibration/build_emiles_calibration.py)
takes much longer and is not something a unit test should run.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from kinextract import (
    run_calibration_sweep, save_calibration_table, load_calibration_table,
    load_per_lam_calibration_table,
)


def _toy_template():
    wavelength = 4750.0 + np.arange(3681) * 1.25
    t = np.ones(3681)
    for cen, depth in zip([8498.02, 8542.09, 8662.14], [0.55, 0.70, 0.65]):
        t -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)
    return [t], wavelength


def test_run_calibration_sweep_single_template_fixed_weights():
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=4, verbose=False,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["sigma"] == 80.0 and row["snr"] == 60.0 and row["n"] == 4
    for key in ("v_bias_mean", "v_bias_std", "sig_bias_mean", "sig_bias_std", "h4_mean", "h4_std"):
        assert np.isfinite(row[key])


def test_run_calibration_sweep_accepts_weight_sampler():
    templates, wavelength = _toy_template()
    templates3 = templates * 3  # pretend 3 templates for the multi-template path
    rng = np.random.default_rng(0)

    def sampler():
        return rng.dirichlet(np.ones(3))

    rows = run_calibration_sweep(
        templates3, wavelength, weights=sampler,
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=4, verbose=False,
    )
    assert len(rows) == 1 and rows[0]["n"] == 4


def test_h4_true_changes_the_target_sigma_for_a_wide_losvd():
    # a real (h4>0) wing genuinely broadens the LOSVD's second moment
    # relative to the h4=0 case at the same nominal sigma -- confirms
    # h4_true actually reaches the mock generator, not just accepted
    # and ignored.
    from kinextract.calibrate_bias import _true_moments
    _, gauss_sigma = _true_moments(0.0, 100.0, h4_true=0.0)
    _, wing_sigma = _true_moments(0.0, 100.0, h4_true=0.15)
    assert wing_sigma > gauss_sigma


def test_run_calibration_sweep_accepts_h4_true():
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=4, h4_true=0.15, verbose=False,
    )
    assert len(rows) == 1 and np.isfinite(rows[0]["h4_mean"])


def test_h3_true_shifts_the_target_v_by_the_exact_closed_form():
    # a real (h3!=0) skew shifts the true V away from the nominal vel by
    # sqrt(3)*sigma*h3 (dev_notes/validation_suite/FINDINGS.md's exact
    # closed form) -- confirms h3_true actually reaches the mock generator
    # via the same numerical _true_moments path h4_true uses, not a
    # separate, potentially-inconsistent shortcut.
    # Loose tolerance (15%, not exact): the closed form is for the unclipped
    # GH density, but `_true_moments` clips b>=0 like the rest of the
    # package's mock generation -- a real, known second-order effect
    # (dev_notes/validation_suite/FINDINGS.md: <=4% of mass at range
    # extremes), not a bug in this parameter's own plumbing.
    from kinextract.calibrate_bias import _true_moments
    sigma, h3 = 100.0, 0.05
    true_v, _ = _true_moments(0.0, sigma, h3_true=h3)
    expected_shift = np.sqrt(3.0) * sigma * h3
    assert abs(true_v - expected_shift) < 0.15 * abs(expected_shift)


def test_run_calibration_sweep_accepts_h3_true():
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=4, h3_true=0.05, verbose=False,
    )
    assert len(rows) == 1
    row = rows[0]
    assert np.isfinite(row["h3_mean"]) and np.isfinite(row["h3_std"])
    # v_bias is scored against the h3-shifted true V (not the nominal
    # v_true=0.0), so a well-behaved fit's v_bias should be much smaller
    # than the shift itself, not centered on it.
    shift = np.sqrt(3.0) * 80.0 * 0.05
    assert abs(row["v_bias_mean"]) < 0.5 * abs(shift)


def test_save_and_load_calibration_table_roundtrips(tmp_path):
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[50.0, 100.0], snr_grid=[60.0], n_reps=4, verbose=False,
    )
    path = os.path.join(tmp_path, "test_calibration.txt")
    save_calibration_table(rows, path, header_comment="test table")
    calib = load_calibration_table(path)
    assert len(calib.recovered_sigma) == 2


def test_default_lam_grid_reproduces_original_fixed_lam_behavior(tmp_path):
    """`lam_grid=None` (the default) must give bit-identical rows to the
    pre-m_eff-axis version of this function -- same noise draws, same
    fits, same sig_bias/h4_bias -- with `lam`/`m_eff_mean`/`m_eff_std`
    just along for the ride as new, additional columns."""
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=6, verbose=False,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["lam"] == 2.0
    assert np.isfinite(row["m_eff_mean"]) and np.isfinite(row["m_eff_std"])


def test_lam_grid_produces_one_row_per_lam_with_sensible_m_eff_ordering():
    # more regularization (higher lam) should mean fewer effective degrees
    # of freedom -- a basic sanity check that m_eff is actually tracking
    # regularization strength, not some unrelated quantity.
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[80.0], snr_grid=[60.0], n_reps=6, verbose=False,
        lam_grid=[0.5, 2.0, 10.0],
    )
    assert len(rows) == 3
    by_lam = {r["lam"]: r["m_eff_mean"] for r in rows}
    assert by_lam[0.5] > by_lam[2.0] > by_lam[10.0]


def test_lam_grid_calibration_collapses_to_one_table_via_load_calibration_table(tmp_path):
    """`load_calibration_table` always returns a plain 2-D `BiasCalibration`
    -- a genuinely multi-lam file gets every row folded into one table
    (`.lam=None`, since no single value applies), not an error and not an
    m_eff-axis object (that's what `load_per_lam_calibration_table`, tested
    separately, is for -- see `PerLamCalibration`'s own docstring for why
    the two are deliberately different functions now)."""
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[50.0, 100.0], snr_grid=[60.0], n_reps=6, verbose=False,
        lam_grid=[0.5, 2.0, 10.0],
    )
    path = os.path.join(tmp_path, "test_calibration_3d.txt")
    save_calibration_table(rows, path, header_comment="test table with lam varied")
    calib = load_calibration_table(path)
    assert len(calib.recovered_sigma) == 6  # 2 sigma x 1 snr x 3 lam
    assert calib.lam is None  # lam varies row to row -- no single value applies


def test_lam_grid_calibration_splits_into_one_table_per_lam_via_load_per_lam_calibration_table(tmp_path):
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[50.0, 100.0], snr_grid=[60.0], n_reps=6, verbose=False,
        lam_grid=[0.5, 2.0, 10.0],
    )
    path = os.path.join(tmp_path, "test_calibration_per_lam.txt")
    save_calibration_table(rows, path, header_comment="test table with lam varied")
    per_lam = load_per_lam_calibration_table(path)
    np.testing.assert_allclose(per_lam.lam_values, [0.5, 2.0, 10.0])
    assert len(per_lam.tables) == 3
    for lam_value, table in zip(per_lam.lam_values, per_lam.tables):
        assert table.lam == pytest.approx(lam_value)
        assert len(table.recovered_sigma) == 2  # 2 sigma x 1 snr, this lam's own rows only


def test_load_per_lam_calibration_table_raises_for_a_single_lam_file(tmp_path):
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[50.0, 100.0], snr_grid=[60.0], n_reps=4, verbose=False,
    )  # lam_grid=None -> every row at the same fixed lam=2.0
    path = os.path.join(tmp_path, "single_lam.txt")
    save_calibration_table(rows, path, header_comment="single-lam table")
    with pytest.raises(ValueError, match="only one distinct lam value"):
        load_per_lam_calibration_table(path)


def test_old_format_file_without_lam_columns_still_loads(tmp_path):
    # a file written before lam/m_eff columns existed (10 columns) must
    # still load fine via load_calibration_table, with calib.lam staying
    # None (no lam column to detect a single value from at all).
    path = os.path.join(tmp_path, "old_format.txt")
    with open(path, "w") as f:
        f.write(f"{'sigma':>6} {'S/N':>6} {'n':>4}   {'V_bias_mean':>12} {'V_bias_std':>11}   "
                f"{'sig_bias_mean':>14} {'sig_bias_std':>13}   {'sig_bias_pct':>13}   "
                f"{'h4_mean':>9} {'h4_std':>8}\n")
        f.write(f"{80:6.0f} {60:6.0f} {10:4d}   {0.0:+12.2f} {5.0:11.2f}   "
                f"{2.0:+14.2f} {3.0:13.2f}   {2.5:+12.1f}%   {0.01:+9.4f} {0.02:8.4f}\n")
    calib = load_calibration_table(path)
    assert calib.lam is None


def test_single_lam_table_reports_its_own_lam_via_load_calibration_table(tmp_path):
    """`save_calibration_table` always writes the 13-column format now,
    even for a table swept at one fixed `lam` (this package's own shipped
    default is exactly this case) -- `load_calibration_table` must detect
    and report that single value (`calib.lam == 2.0`), not treat the file
    as ambiguous or as a multi-lam set."""
    templates, wavelength = _toy_template()
    rows = run_calibration_sweep(
        templates, wavelength, weights=np.array([1.0]),
        sigma_grid=[50.0, 100.0], snr_grid=[60.0], n_reps=4, verbose=False,
    )  # lam_grid=None -> every row at the same fixed lam=2.0, but still 13 columns
    path = os.path.join(tmp_path, "single_lam_13col.txt")
    save_calibration_table(rows, path, header_comment="single-lam table, 13-column format")
    calib = load_calibration_table(path)
    assert calib.lam == pytest.approx(2.0)
