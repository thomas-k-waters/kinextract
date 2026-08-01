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
