"""Tests for kinextract.binning -- the dartboard (log/linear-radial
x equal-angle) spatial binning scheme. See binning.py's own module
docstring for why equal-angle (not equal-sin(angle)) and circular-only
(no PA/ellipticity) were chosen.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract.binning import (
    DartboardBins, log_radial_edges, linear_radial_edges, uniform_theta_edges,
    build_reflected_sin_theta_edges, assign_spaxels_to_bins,
    load_legacy_bin_edges, assign_legacy_dartboard_bins, pixel_aware_radial_edges,
)


def test_log_radial_edges_spans_range_and_is_increasing():
    edges = log_radial_edges(1.0, 100.0, 6)
    assert len(edges) == 7
    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(100.0)
    assert np.all(np.diff(edges) > 0)
    # log-spaced: consecutive ratios should be constant
    ratios = edges[1:] / edges[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)


def test_log_radial_edges_rejects_nonpositive_r_min():
    with pytest.raises(ValueError, match="r_min"):
        log_radial_edges(0.0, 10.0, 5)


def test_pixel_aware_radial_edges_second_edge_is_exactly_one_pixel():
    edges = pixel_aware_radial_edges(pixel_scale=0.025, r_max=4.0, n_bins=10)
    assert len(edges) == 11
    assert edges[1] == pytest.approx(0.025)


def test_pixel_aware_radial_edges_first_edge_is_a_formality_smaller_than_second():
    edges = pixel_aware_radial_edges(pixel_scale=0.2, r_max=30.0, n_bins=8)
    assert 0 < edges[0] < edges[1]


def test_pixel_aware_radial_edges_last_edge_is_r_max_and_strictly_increasing():
    edges = pixel_aware_radial_edges(pixel_scale=0.025, r_max=4.0, n_bins=10)
    assert edges[-1] == pytest.approx(4.0)
    assert np.all(np.diff(edges) > 0)


def test_pixel_aware_radial_edges_beyond_center_is_geometric():
    edges = pixel_aware_radial_edges(pixel_scale=1.0, r_max=100.0, n_bins=6)
    beyond_center = edges[1:]  # pixel_scale ... r_max, geometric
    ratios = beyond_center[1:] / beyond_center[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)


def test_pixel_aware_radial_edges_with_dartboard_bins_collapses_only_the_center_pixel():
    """The whole point: DartboardBins(single_center_bin=True) collapses everything
    below r_edges[1] into one bin -- with r_edges[1]=pixel_scale exactly, only the
    literal center pixel (r=0) should fall in that collapsed bin; the nearest ring
    of real neighbors (r=1 pixel) must land in a genuine annulus."""
    pixel_scale = 0.025
    edges = pixel_aware_radial_edges(pixel_scale=pixel_scale, r_max=4.0, n_bins=15)
    theta_edges = uniform_theta_edges(4)
    bins = DartboardBins(r_edges=edges, theta_edges=theta_edges, center=(50.0, 50.0),
                          single_center_bin=True)

    shape = (101, 101)
    bin_id_map, _ = assign_spaxels_to_bins(shape, bins, pixel_scale=pixel_scale)
    center_bin_id = bin_id_map[50, 50]
    # the literal center pixel and its immediate neighbor (1 pixel away) must NOT
    # share the same collapsed bin id
    neighbor_bin_id = bin_id_map[50, 51]
    assert neighbor_bin_id != center_bin_id


def test_pixel_aware_radial_edges_rejects_r_max_not_exceeding_pixel_scale():
    with pytest.raises(ValueError, match="r_max"):
        pixel_aware_radial_edges(pixel_scale=1.0, r_max=1.0, n_bins=5)


def test_pixel_aware_radial_edges_rejects_too_few_bins():
    with pytest.raises(ValueError, match="n_bins"):
        pixel_aware_radial_edges(pixel_scale=0.1, r_max=10.0, n_bins=1)


def test_linear_radial_edges_spans_range_and_is_uniform():
    edges = linear_radial_edges(0.0, 20.0, 4)
    assert len(edges) == 5
    np.testing.assert_allclose(np.diff(edges), 5.0)


def test_uniform_theta_edges_spans_full_turn_with_equal_steps():
    edges = uniform_theta_edges(8)
    assert len(edges) == 9
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(2 * np.pi)
    np.testing.assert_allclose(np.diff(edges), 2 * np.pi / 8)


def test_dartboard_bins_rejects_non_increasing_edges():
    with pytest.raises(ValueError, match="r_edges"):
        DartboardBins(r_edges=np.array([1.0, 1.0, 5.0]), theta_edges=uniform_theta_edges(4), center=(0, 0))
    with pytest.raises(ValueError, match="theta_edges"):
        DartboardBins(r_edges=log_radial_edges(1.0, 10.0, 3), theta_edges=np.array([0.0, 0.5, 0.4]), center=(0, 0))


def test_n_bins_accounts_for_single_center_bin_collapse():
    r_edges = log_radial_edges(1.0, 50.0, 5)
    theta_edges = uniform_theta_edges(4)
    collapsed = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(0, 0), single_center_bin=True)
    uncollapsed = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(0, 0), single_center_bin=False)
    assert collapsed.n_bins == 1 + 4 * 4  # center collapses to 1, remaining 4 annuli x 4 sectors
    assert uncollapsed.n_bins == 5 * 4    # every annulus sectored


def test_assign_spaxels_to_bins_every_valid_spaxel_accounted_for():
    r_edges = log_radial_edges(1.0, 50.0, 5)
    theta_edges = uniform_theta_edges(4)
    bins = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(50, 50))
    bin_id_map, bin_table = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0)

    assert bin_id_map.shape == (100, 100)
    valid_ids = bin_id_map[bin_id_map >= 0]
    assert valid_ids.min() >= 0
    assert valid_ids.max() < bins.n_bins
    assert sum(row["n_spaxels"] for row in bin_table) == int(np.sum(bin_id_map >= 0))
    # every populated bin_id in the table actually appears that many times in the map
    for row in bin_table:
        assert int(np.sum(bin_id_map == row["bin_id"])) == row["n_spaxels"]


def test_assign_spaxels_center_bin_collapses_regardless_of_theta():
    r_edges = log_radial_edges(1.0, 50.0, 5)
    theta_edges = uniform_theta_edges(4)
    bins = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(50, 50), single_center_bin=True)
    bin_id_map, bin_table = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0)

    center_rows = [row for row in bin_table if row["bin_id"] == 0]
    assert len(center_rows) == 1
    assert center_rows[0]["r_index"] == 0
    assert center_rows[0]["theta_index"] is None
    assert center_rows[0]["theta_mid"] is None


def test_assign_spaxels_center_pixel_is_binned_even_inside_r_min():
    """Regression test for a real bug: the literal center pixel (r=0) is
    closer to center than r_edges[0] (log_radial_edges requires r_min>0),
    and with single_center_bin=True must still land in the center bin,
    not fall through as unassigned (-1)."""
    r_edges = log_radial_edges(1.0, 50.0, 5)  # r_min=1.0 -- strictly > 0
    theta_edges = uniform_theta_edges(4)
    bins = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(50, 50), single_center_bin=True)
    bin_id_map, bin_table = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0)

    assert bin_id_map[50, 50] == 0  # the exact center pixel, r=0 < r_min=1.0
    center_row = next(row for row in bin_table if row["bin_id"] == 0)
    assert center_row["n_spaxels"] >= 1


def test_assign_spaxels_outside_outer_edge_is_unassigned():
    r_edges = log_radial_edges(1.0, 10.0, 3)
    theta_edges = uniform_theta_edges(4)
    bins = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(50, 50))
    # a 100x100 image extends well past r=10 from a (50,50) center in the corners
    bin_id_map, _ = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0)
    assert bin_id_map[0, 0] == -1  # corner, far outside r_max
    assert bin_id_map[50, 51] != -1  # one pixel from center, inside r_min..r_max


def test_pixel_scale_rescales_physical_radius():
    r_edges = log_radial_edges(1.0, 10.0, 3)
    theta_edges = uniform_theta_edges(4)
    bins = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(50, 50))
    # at pixel_scale=1.0, a pixel 5 away from center is at physical r=5 (inside r_max=10)
    map_scale1, _ = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0)
    assert map_scale1[50, 55] != -1
    # at pixel_scale=5.0, that same pixel is at physical r=25 (outside r_max=10)
    map_scale5, _ = assign_spaxels_to_bins((100, 100), bins, pixel_scale=5.0)
    assert map_scale5[50, 55] == -1


def test_build_reflected_sin_theta_edges_matches_the_legacy_pipeline_convention():
    """sco_framework's own bin_v.out uses exactly these 5 angular edges
    per quadrant: arcsin([0.2, 0.4, 0.6, 0.8, 1.0]) in degrees.
    """
    edges = build_reflected_sin_theta_edges(5)
    assert len(edges) == 4 * 5 + 1
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(2 * np.pi)
    first_quadrant_deg = np.degrees(edges[:6])
    expected = np.degrees(np.arcsin([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    np.testing.assert_allclose(first_quadrant_deg, expected, atol=1e-8)


def test_build_reflected_sin_theta_edges_is_finest_near_the_axes():
    """Equal-sin(theta) spacing: sectors near theta=0/180 (the assumed
    major axis) must be narrower than sectors near theta=90/270 (minor
    axis) -- the whole point of this convention.
    """
    edges = build_reflected_sin_theta_edges(5)
    widths = np.diff(edges)
    assert widths[0] < widths[-1]  # first sector (near 0 deg) vs last (near 360/minor-adjacent)


def test_assign_spaxels_to_bins_pa_deg_zero_matches_default_behavior():
    r_edges = log_radial_edges(1.0, 50.0, 5)
    theta_edges = uniform_theta_edges(8)
    bins = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(50, 50))
    map_default, table_default = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0)
    map_pa0, table_pa0 = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0, pa_deg=0.0)
    np.testing.assert_array_equal(map_default, map_pa0)
    assert table_default == table_pa0


def test_assign_spaxels_to_bins_pa_deg_rotates_theta_but_not_r():
    """A 90-degree PA rotation must leave radial membership (valid vs
    not) completely unaffected, and must assign a point due east of
    center at pa_deg=90 the same bin_id as the point due south of center
    gets at pa_deg=0 -- rotating the frame by +90 deg is equivalent to
    rotating the query point by -90 deg in the original frame.
    """
    r_edges = log_radial_edges(1.0, 50.0, 3)
    theta_edges = uniform_theta_edges(4)
    bins = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=(50, 50))

    map_0, _ = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0, pa_deg=0.0)
    map_90, _ = assign_spaxels_to_bins((100, 100), bins, pixel_scale=1.0, pa_deg=90.0)

    # r-only membership (valid vs not) must be identical regardless of pa_deg
    np.testing.assert_array_equal(map_0 >= 0, map_90 >= 0)

    east_of_center = (50, 55)   # (y, x): same row, +5 in x -> due east
    south_of_center = (45, 50)  # (y, x): -5 in y, same x -> due "south" (smaller y)
    assert map_90[east_of_center] != -1
    assert map_0[south_of_center] == map_90[east_of_center]


# -- assign_legacy_dartboard_bins / load_legacy_bin_edges -------------------------------

_LEGACY_R_OUT = """  irc    rlow      rmid      rup    bin_size
 ----   ------    ------    ------    ------
   1       0.500       0.750       1.000       0.500
   2       1.000       1.500       2.000       1.000
   3       2.000       2.500       3.000       1.000
"""

_LEGACY_V_OUT = """  irv    vlow      vmid      vup    bin_size
 ----   ------    ------    ------    ------
   1       0.000      10.000      20.000      20.000
   2      20.000      30.000      40.000      20.000
   3      40.000      65.000      90.000      50.000
"""


def _write_legacy_edges(tmp_path):
    r_path = tmp_path / "bin_r.out"
    v_path = tmp_path / "bin_v.out"
    r_path.write_text(_LEGACY_R_OUT)
    v_path.write_text(_LEGACY_V_OUT)
    return str(r_path), str(v_path)


def test_load_legacy_bin_edges_reads_rlow0_and_rup_column(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    np.testing.assert_allclose(r_edges, [0.5, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(v_edges_deg, [0.0, 20.0, 40.0, 90.0])


def test_assign_legacy_dartboard_bins_on_axis_point_lands_in_irv_1(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    # (dy=0, dx=+1.5) with pa_deg=0 -> x=1.5, y=0 -> on the major axis (vp=0) -> irv=1,
    # r=1.5 is in [1,2) -> irc=2, x>=0 -> "bin" side
    point = (50, 51)  # (y, x): 1 pixel east of center, pixel_scale=1.5 -> r=1.5 exactly
    bin_id_map, bin_table = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.5)
    entry = next(r for r in bin_table if r["bin_id"] == bin_id_map[point])
    assert entry["bin_name"] == "bin0201"
    assert entry["irc"] == 2 and entry["irv"] == 1 and entry["side"] == 1


def test_assign_legacy_dartboard_bins_bnn_side_is_negative_x(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    bin_id_map, bin_table = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.5)
    west_point = (50, 49)  # 1 pixel west of center -> dx=-1.5 -> x=-1.5 -> "bnn" side
    entry = next(r for r in bin_table if r["bin_id"] == bin_id_map[west_point])
    assert entry["bin_name"] == "bnn0201"
    assert entry["side"] == -1


def test_assign_legacy_dartboard_bins_folds_north_and_south_of_major_axis(tmp_path):
    """A point straight north and straight south of center (both on the
    kinematic minor axis, x=0) must land in the identical bin -- the
    abs(y) fold this module's docstring describes.
    """
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    bin_id_map, _ = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.5)
    north_point = (51, 50)  # dy=+1.5 -> y=+1.5 (pa_deg=0: x=dx, y=dy)
    south_point = (49, 50)  # dy=-1.5 -> y=-1.5
    assert bin_id_map[north_point] != -1
    assert bin_id_map[north_point] == bin_id_map[south_point]


def test_assign_legacy_dartboard_bins_center_clips_into_innermost_ring(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    bin_id_map, bin_table = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.0)
    entry = next(r for r in bin_table if r["bin_id"] == bin_id_map[50, 50])
    assert entry["irc"] == 1


def test_assign_legacy_dartboard_bins_exact_center_lands_in_last_irv_not_first(tmp_path):
    """plotbin3.f's own branch is `if(abs(x).ne.0) vp=atan(y/x) else vp=89.99` -- at the
    literal grid center (x=0 *and* y=0 simultaneously, not just x=0), `np.arctan2(0, 0)`
    silently returns 0.0 instead of the Fortran's forced 89.99, landing the center spaxel
    in the *first* angular sector (irv=1) instead of the *last* one (irv=n_v). Confirmed
    directly against a real N4751 run before this fix: this function reported the center
    bin as "bin0101", plotbin3.f's own writebin (for the same 5-sector bin_v.out) writes
    "bin0105" for the identical spaxel -- a real legacy-pipeline naming incompatibility,
    not just an internal labeling choice (fitlovw/pallmc/... read files by this exact
    name). test_..._perpendicular_point_lands_in_last_irv (above) does not cover this --
    that test's own point is one pixel *off* center (x=0, y=+1.5), which np.arctan2
    already handles correctly (arctan2(y>0, 0) = pi/2 exactly, no ambiguity); only the
    fully-degenerate (0, 0) origin itself diverges."""
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    n_v = len(v_edges_deg) - 1
    bin_id_map, bin_table = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.0)
    entry = next(r for r in bin_table if r["bin_id"] == bin_id_map[50, 50])
    assert entry["irv"] == n_v
    assert entry["bin_name"] == f"bin01{n_v:02d}"


def test_assign_legacy_dartboard_bins_outside_outer_edge_is_unassigned(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    bin_id_map, _ = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.0)
    assert bin_id_map[50, 99] == -1  # 49 pixels away, r_edges tops out at 3.0


def test_assign_legacy_dartboard_bins_perpendicular_point_lands_in_last_irv(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    bin_id_map, bin_table = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.5)
    minor_axis_point = (51, 50)  # x=0, y=+1.5 -> vp≈90deg, clipped to 89.99 -> last irv bin
    entry = next(r for r in bin_table if r["bin_id"] == bin_id_map[minor_axis_point])
    assert entry["irv"] == 3


def test_fold_major_axis_default_true_matches_pre_toggle_behavior(tmp_path):
    """fold_major_axis defaults to True specifically so every existing caller
    (built before this parameter existed) keeps its current behavior -- see
    assign_legacy_dartboard_bins's own docstring. Every hemisphere must be
    +1 (north) by convention when folded, and bin_name must have no suffix."""
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    bin_id_map, bin_table = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.0)
    assert all(r["hemisphere"] == 1 for r in bin_table)
    assert all(not r["bin_name"].endswith("s") for r in bin_table)


def test_fold_major_axis_false_conserves_total_spaxel_count_and_footprint(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    shape, center, pixel_scale = (100, 100), (50, 50), 1.0
    bmap_f, btab_f = assign_legacy_dartboard_bins(
        shape, center=center, pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=pixel_scale, fold_major_axis=True)
    bmap_u, btab_u = assign_legacy_dartboard_bins(
        shape, center=center, pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=pixel_scale, fold_major_axis=False)
    # Same spaxels get assigned either way -- folding only changes which bin they
    # land in, never whether they're inside r_edges[-1] at all.
    np.testing.assert_array_equal(bmap_f >= 0, bmap_u >= 0)
    assert sum(r["n_spaxels"] for r in btab_f) == sum(r["n_spaxels"] for r in btab_u)


def test_fold_major_axis_false_north_plus_south_equals_folded_count(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    shape, center, pixel_scale = (100, 100), (50, 50), 1.0
    _, btab_f = assign_legacy_dartboard_bins(
        shape, center=center, pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=pixel_scale, fold_major_axis=True)
    _, btab_u = assign_legacy_dartboard_bins(
        shape, center=center, pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=pixel_scale, fold_major_axis=False)
    for row in btab_f:
        n_folded = row["n_spaxels"]
        n_north = next((r["n_spaxels"] for r in btab_u
                         if r["bin_name"] == row["bin_name"] and r["hemisphere"] == 1), 0)
        south_name = row["bin_name"] + "s"
        n_south = next((r["n_spaxels"] for r in btab_u
                         if r["bin_name"] == south_name and r["hemisphere"] == -1), 0)
        assert n_north + n_south == n_folded


def test_fold_major_axis_false_south_bin_name_has_s_suffix(tmp_path):
    r_path, v_path = _write_legacy_edges(tmp_path)
    r_edges, v_edges_deg = load_legacy_bin_edges(r_path, v_path)
    bin_id_map, bin_table = assign_legacy_dartboard_bins(
        (100, 100), center=(50, 50), pa_deg=0.0, r_edges=r_edges, v_edges_deg=v_edges_deg,
        pixel_scale=1.5, fold_major_axis=False)
    south_point = (49, 51)  # y=-1.5 (south of major axis), x=+1.5 -> "bin" side, south hemisphere
    entry = next(r for r in bin_table if r["bin_id"] == bin_id_map[south_point])
    assert entry["hemisphere"] == -1
    assert entry["bin_name"].endswith("s")
    assert entry["bin_name"][:-1].startswith("bin")


def test_legacy_dartboard_major_axis_sector_lies_on_pa():
    """pa_deg is the major-axis PA counterclockwise from +x: the bin/bnn sector-1 wedges must be centered on pa_deg and
    pa_deg + 180 (before 2026-10-16 they were centered on 180 - pa_deg, the axis mirrored in x)."""
    r_edges = np.array([0.0, 5.0, 40.0])
    v_edges_deg = np.array([0.0, 11.537, 23.578, 36.870, 53.130, 90.0])
    for pa in (0.0, 30.0, 85.66, 120.0):
        id_map, table = assign_legacy_dartboard_bins((201, 201), center=(100, 100), pa_deg=pa, r_edges=r_edges,
                                                     v_edges_deg=v_edges_deg, pixel_scale=1.0)
        yy, xx = np.mgrid[0:201, 0:201]
        for name, axis in (("bin0201", pa), ("bnn0201", pa + 180.0)):
            row = next(t for t in table if t["bin_name"] == name)
            m = id_map == row["bin_id"]
            ang = np.degrees(np.arctan2(yy[m] - 100, xx[m] - 100))
            off = (ang - axis + 180.0) % 360.0 - 180.0          # signed offset from the expected axis direction
            assert abs(np.median(off)) < 0.5, (pa, name, np.median(off))
            assert np.all(np.abs(off) <= 11.537 + 1.0)
