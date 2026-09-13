"""Tests for kinextract.plotting -- the standalone (no cube/WCS needed)
moment-map plotting built for sco_framework_updated/bin/plotmoments."""
from __future__ import annotations

import os

import numpy as np
import pytest

from kinextract.plotting import (
    parse_bin_name, load_bin_kinematics_dir, load_pallmc_out, load_bin_kinematics,
    plot_moment_maps,
)


def test_parse_bin_name_bin_prefix_is_positive_side():
    assert parse_bin_name("bin0301") == (3, 1, 1)


def test_parse_bin_name_bnn_prefix_is_negative_side():
    assert parse_bin_name("bnn0705") == (7, 5, -1)


def test_parse_bin_name_ignores_trailing_suffix():
    assert parse_bin_name("bin0301spmc") == (3, 1, 1)


def test_parse_bin_name_rejects_bad_prefix():
    with pytest.raises(ValueError, match="bin.*bnn"):
        parse_bin_name("xyz0301")


def test_parse_bin_name_rejects_non_digit_stem():
    with pytest.raises(ValueError, match="digits"):
        parse_bin_name("binABCD")


def _write_kinematics_file(path, v50, s50, h3=None, h4=None):
    with open(path, "w") as f:
        f.write(f"v_16 {v50 - 8:.6f}\n")
        f.write(f"v_50 {v50:.6f}\n")
        f.write(f"v_84 {v50 + 8:.6f}\n")
        f.write(f"sigma_16 {s50 - 10:.6f}\n")
        f.write(f"sigma_50 {s50:.6f}\n")
        f.write(f"sigma_84 {s50 + 10:.6f}\n")
        if h3 is not None:
            f.write(f"h3_16 {h3 - 0.01:.6f}\n")
            f.write(f"h3_50 {h3:.6f}\n")
            f.write(f"h3_84 {h3 + 0.01:.6f}\n")
        if h4 is not None:
            f.write(f"h4_16 {h4 - 0.01:.6f}\n")
            f.write(f"h4_50 {h4:.6f}\n")
            f.write(f"h4_84 {h4 + 0.01:.6f}\n")


def test_load_bin_kinematics_dir_reads_six_line_file(tmp_path):
    _write_kinematics_file(tmp_path / "bin0301spmc.kinematics", v50=12.0, s50=150.0)
    result = load_bin_kinematics_dir(str(tmp_path))
    assert set(result.keys()) == {"bin0301"}
    entry = result["bin0301"]
    assert entry["V"] == 12.0
    assert entry["V_lo"] == 4.0 and entry["V_hi"] == 20.0
    assert entry["sigma"] == 150.0
    assert "h3" not in entry


def test_load_bin_kinematics_dir_reads_twelve_line_extension(tmp_path):
    _write_kinematics_file(tmp_path / "bin0301spmc.kinematics", v50=12.0, s50=150.0, h3=0.02, h4=-0.01)
    result = load_bin_kinematics_dir(str(tmp_path))
    entry = result["bin0301"]
    assert entry["h3"] == 0.02
    assert entry["h4"] == -0.01
    assert entry["h3_lo"] == pytest.approx(0.01)
    assert entry["h4_hi"] == pytest.approx(0.0)


def test_load_bin_kinematics_dir_multiple_bins(tmp_path):
    _write_kinematics_file(tmp_path / "bin0301spmc.kinematics", v50=12.0, s50=150.0)
    _write_kinematics_file(tmp_path / "bnn0705spmc.kinematics", v50=-30.0, s50=200.0)
    result = load_bin_kinematics_dir(str(tmp_path))
    assert set(result.keys()) == {"bin0301", "bnn0705"}


def test_load_pallmc_out_parses_summary_lines(tmp_path):
    path = tmp_path / "pallmc.out"
    path.write_text(
        "   0.50   12.300   8.000 145.600  10.000   0.020   0.010  -0.010   0.010  "
        "bin0105spmc.sim     \n"
        "   1.20  -30.000   5.000 200.000  12.000   0.000   0.005   0.000   0.005  "
        "bnn0704spmc.sim     \n"
    )
    result = load_pallmc_out(str(path))
    assert set(result.keys()) == {"bin0105", "bnn0704"}
    e = result["bin0105"]
    assert e["V"] == 12.3 and e["V_lo"] == pytest.approx(4.3) and e["V_hi"] == pytest.approx(20.3)
    assert e["sigma"] == 145.6
    assert e["h3"] == 0.02
    assert e["h4"] == -0.01


def test_load_bin_kinematics_prefers_kinematics_dir_over_pallmc_out(tmp_path):
    _write_kinematics_file(tmp_path / "bin0105spmc.kinematics", v50=99.0, s50=99.0)
    pallmc_path = tmp_path / "pallmc.out"
    pallmc_path.write_text(
        "   0.50   12.300   8.000 145.600  10.000   0.020   0.010  -0.010   0.010  "
        "bin0105spmc.sim     \n"
        "   1.20  -30.000   5.000 200.000  12.000   0.000   0.005   0.000   0.005  "
        "bnn0704spmc.sim     \n"
    )
    result = load_bin_kinematics(str(tmp_path), str(pallmc_path))
    assert result["bin0105"]["V"] == 99.0  # .kinematics wins on overlap
    assert result["bnn0704"]["V"] == -30.0  # pallmc.out fills the gap


def _synthetic_bin_values(bin_names):
    rng = np.random.default_rng(0)
    return {
        b: dict(V=float(rng.normal(0, 50)), sigma=float(abs(rng.normal(150, 30))),
                h3=float(rng.normal(0, 0.05)), h4=float(rng.normal(0, 0.05)))
        for b in bin_names
    }


def test_plot_moment_maps_smoke_runs_and_returns_figure_with_one_axis_per_moment():
    r_edges = np.array([0.1, 0.5, 1.0, 2.0])
    v_edges_deg = np.array([0.0, 30.0, 60.0, 90.0])
    bin_names = [f"bin0{ir}0{iv}" for ir in range(1, 4) for iv in range(1, 4)]
    bin_names += [f"bnn0{ir}0{iv}" for ir in range(1, 4) for iv in range(1, 4)]
    bin_values = _synthetic_bin_values(bin_names)

    fig = plot_moment_maps(bin_values, r_edges, v_edges_deg)
    assert len(fig.axes) == 4 * 2  # 4 moments, each with its own colorbar axis
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_moment_maps_saves_to_file(tmp_path):
    r_edges = np.array([0.1, 0.5, 1.0])
    v_edges_deg = np.array([0.0, 45.0, 90.0])
    bin_names = ["bin0101", "bin0102", "bnn0101", "bnn0102"]
    bin_values = _synthetic_bin_values(bin_names)
    out_path = str(tmp_path / "moments.png")

    plot_moment_maps(bin_values, r_edges, v_edges_deg, out_path=out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_moment_maps_handles_missing_moment_values_gracefully():
    """A bin with no h3/h4 (e.g. from load_pallmc_out's own symmetric-error
    fallback missing a key) should not crash the whole figure."""
    r_edges = np.array([0.1, 1.0])
    v_edges_deg = np.array([0.0, 90.0])
    bin_values = {"bin0101": dict(V=10.0, sigma=100.0)}  # no h3/h4 at all
    fig = plot_moment_maps(bin_values, r_edges, v_edges_deg)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_moment_maps_respects_explicit_vlims_and_cmaps():
    r_edges = np.array([0.1, 1.0])
    v_edges_deg = np.array([0.0, 90.0])
    bin_values = {"bin0101": dict(V=10.0, sigma=100.0, h3=0.01, h4=-0.01)}
    fig = plot_moment_maps(bin_values, r_edges, v_edges_deg,
                            vlims={"V": (-500, 500)}, cmaps={"V": "coolwarm"})
    import matplotlib.pyplot as plt
    plt.close(fig)
