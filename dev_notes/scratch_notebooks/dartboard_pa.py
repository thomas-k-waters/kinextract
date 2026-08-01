"""Reflected sin(theta) dartboard binning, PA-aware.

Built for `ppxf_vs_kinextract_comparison.ipynb`: a fixed geometric binning
scheme -- radial annuli log-spaced in `r`, crossed with angular sectors
equally spaced in `sin(theta)` (not `theta` itself) measured from a given
kinematic position angle -- so sectors are narrowest near the major axis
and widest near the minor axis, matching the `sco_framework` pipeline's own
`bin_r.out`/`bin_v.out` (`binspectra`/`plotbin3`) convention. Verified
directly against `sco_framework_updated/sco_workdir_example/analysis/N5102/
kinematics/muse/spectral_binning/bin_v.out`: its 5 angular edges per
quadrant (11.54, 23.58, 36.87, 53.13, 90.00 deg) are exactly
`arcsin([0.2,0.4,0.6,0.8,1.0])`.

**The one deliberate change from the legacy scheme**: `plot_dartboard` in
`IMBH_Project/.../talk_figures_n5102.ipynb` builds this exact 4-quadrant
wedge pattern but then *combines* data across pairs of mirror-image
quadrants (the standard assumed-symmetry trick real dynamical-modeling
pipelines use to halve the number of independent spectra needed). Here
every one of the 4 reflected wedges is its own independent bin, built only
from the spaxels that actually fall inside it -- no combination across the
major or minor axis, so an asymmetric feature shows up as an asymmetry in
downstream maps instead of being averaged away.

**2026-07-30**: `build_reflected_sin_theta_edges` and the `pa_deg`-aware
assignment (`assign_spaxels_to_bins_pa`, formerly a local duplicate) are
now first-class package functions --
`kinextract.binning.build_reflected_sin_theta_edges` and
`kinextract.binning.assign_spaxels_to_bins`'s own `pa_deg` parameter.
This module now just re-exports/wraps them for the convenience helpers
below (`build_pa_dartboard` etc.) that are still specific to this
notebook's own workflow (auto-centering, bin renumbering, x_gen/y_gen/
bin_sn diagnostics), not general enough to belong in the package itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kinextract.binning import (
    DartboardBins, build_reflected_sin_theta_edges, assign_spaxels_to_bins,
)


def assign_spaxels_to_bins_pa(shape: tuple[int, int], bins: DartboardBins,
                               pixel_scale: float, pa_deg: float) -> tuple[np.ndarray, list[dict]]:
    """Thin wrapper kept for this notebook's own call sites -- forwards
    directly to `kinextract.binning.assign_spaxels_to_bins`'s own
    `pa_deg` parameter, which is where this logic now actually lives.
    """
    return assign_spaxels_to_bins(shape, bins, pixel_scale=pixel_scale, pa_deg=pa_deg)


def build_radial_edges(nx: int, ny: int, pixel_scale: float, rmax_arcsec: float,
                        reference_r_edges_arcsec: np.ndarray) -> np.ndarray:
    """Scale `reference_r_edges_arcsec` to this cube's field of view, then
    truncate at `rmax_arcsec` -- a smaller `rmax_arcsec` gives fewer radial
    bins (and a faster per-bin fit loop downstream), not a rescaled log
    spacing. Always returns at least 2 edges.
    """
    fov_half_extent = 0.5 * min(nx, ny) * pixel_scale
    r_edges_full = reference_r_edges_arcsec * (0.9 * fov_half_extent / reference_r_edges_arcsec[-1])
    r_edges = r_edges_full[r_edges_full < rmax_arcsec]
    if len(r_edges) == 0:
        r_edges = np.array([0.0, rmax_arcsec])
    elif r_edges[-1] < rmax_arcsec:
        r_edges = np.r_[r_edges, rmax_arcsec]
    if len(r_edges) < 2:
        r_edges = np.array([0.0, rmax_arcsec])
    return r_edges


@dataclass
class PADartboardResult:
    bin_map: np.ndarray
    nbins: int
    bin_number: np.ndarray
    x_gen: np.ndarray
    y_gen: np.ndarray
    bin_sn: np.ndarray
    good: np.ndarray
    bins: DartboardBins
    center_pix: tuple[int, int]
    r_index_map: np.ndarray
    theta_index_map: np.ndarray
    bin_table: list = field(repr=False)


def build_pa_dartboard(good: np.ndarray, signal_map: np.ndarray, noise_map: np.ndarray,
                        x_arcsec: np.ndarray, y_arcsec: np.ndarray, pixel_scale: float,
                        pa_deg: float, rmax_arcsec: float, reference_r_edges_arcsec: np.ndarray,
                        n_theta_per_quadrant: int = 5) -> PADartboardResult:
    """Do everything the comparison notebook's own binning cell used to do
    inline: scale/truncate the radial edges to `rmax_arcsec`, center on the
    brightest spaxel in `signal_map` (restricted to `good`), assign
    PA-aware reflected dartboard bins, renumber to contiguous ids, and
    compute the `x_gen`/`y_gen`/`bin_number`/`bin_sn` diagnostics every
    downstream cell needs (kept in the same shape/convention the notebook's
    original Voronoi-binning code produced, so nothing downstream needed to
    change).

    `good` is narrowed further to spaxels that actually landed in a
    populated bin (returned as `.good`) -- assign it back to the caller's
    own `good` so later cells only ever see spaxels that were actually fit.
    """
    ny, nx = signal_map.shape
    r_edges = build_radial_edges(nx, ny, pixel_scale, rmax_arcsec, reference_r_edges_arcsec)
    theta_edges = build_reflected_sin_theta_edges(n_theta_per_quadrant)

    valid_signal = np.where(good, signal_map, np.nan)
    brightest_j, brightest_i = np.unravel_index(np.nanargmax(valid_signal), valid_signal.shape)
    center_pix = (int(brightest_i), int(brightest_j))
    center_x_arcsec = x_arcsec[center_pix[1], center_pix[0]]
    center_y_arcsec = y_arcsec[center_pix[1], center_pix[0]]

    bins = DartboardBins(r_edges=r_edges, theta_edges=theta_edges, center=center_pix,
                          single_center_bin=True)
    raw_bin_map, raw_bin_table = assign_spaxels_to_bins_pa((ny, nx), bins, pixel_scale, pa_deg)

    populated_ids_raw = set(np.unique(raw_bin_map[good & (raw_bin_map >= 0)]))
    populated_ids = [d["bin_id"] for d in raw_bin_table if d["bin_id"] in populated_ids_raw]
    remap = {old: new for new, old in enumerate(populated_ids)}

    bin_map = np.full((ny, nx), -1, dtype=int)
    for old, new in remap.items():
        bin_map[(raw_bin_map == old) & good] = new
    good_out = good & (bin_map >= 0)
    nbins = len(populated_ids)

    pa_rad = np.radians(pa_deg)
    x_gen = np.zeros(nbins)
    y_gen = np.zeros(nbins)
    r_index_map = np.full((ny, nx), np.nan)
    theta_index_map = np.full((ny, nx), np.nan)
    for d in raw_bin_table:
        if d["bin_id"] not in remap:
            continue
        new_id = remap[d["bin_id"]]
        mask = bin_map == new_id
        r_index_map[mask] = d["r_index"]
        if d["theta_mid"] is None:
            x_gen[new_id], y_gen[new_id] = center_x_arcsec, center_y_arcsec
            continue
        theta_index_map[mask] = d["theta_index"]
        xr = d["r_mid"] * np.cos(d["theta_mid"])
        yr = d["r_mid"] * np.sin(d["theta_mid"])
        x_gen[new_id] = center_x_arcsec + xr * np.cos(pa_rad) - yr * np.sin(pa_rad)
        y_gen[new_id] = center_y_arcsec + xr * np.sin(pa_rad) + yr * np.cos(pa_rad)

    bin_number = bin_map.ravel()[np.flatnonzero(good_out.ravel())]
    bin_sn = np.zeros(nbins)
    for j in range(nbins):
        mask = bin_map == j
        if mask.any():
            bin_sn[j] = np.sum(signal_map[mask]) / np.sqrt(np.sum(noise_map[mask] ** 2))

    return PADartboardResult(
        bin_map=bin_map, nbins=nbins, bin_number=bin_number, x_gen=x_gen, y_gen=y_gen,
        bin_sn=bin_sn, good=good_out, bins=bins, center_pix=center_pix,
        r_index_map=r_index_map, theta_index_map=theta_index_map, bin_table=raw_bin_table,
    )
