"""Spatial binning for IFU data: a "dartboard" scheme -- radial annuli
(log- or linear-spaced) crossed with angular sectors, in either of two
angular conventions (see below).

**Two angular conventions, same underlying scheme.** The default,
`uniform_theta_edges`, bins equal in `theta` itself (every sector spans
the same angle and, at fixed radius, the same physical area) with no
assumed axis -- `theta=0` is just wherever the caller's own
pixel-coordinate convention puts it. This is the right default when
nothing about the target's orientation is known yet. The alternative,
`build_reflected_sin_theta_edges`, instead bins equal in `sin(theta)`
measured from an assumed kinematic major axis -- narrowest sectors on the
major axis, widest on the minor axis, concentrating angular resolution
where rotation actually shows up. Both are available here:
`build_reflected_sin_theta_edges` builds that edge pattern, and
`assign_spaxels_to_bins`'s `pa_deg` parameter (default `0.0`, i.e. off)
rotates `theta=0` onto a known position angle to pair with it. Use the
default when no axis is known; use `pa_deg` + `build_reflected_sin_
theta_edges` once one is (e.g. from a prior kinematic PA measurement).

**Bin-edge arrays, not just bin counts.** `DartboardBins` takes explicit
`r_edges`/`theta_edges` arrays rather than a single "number of bins"
integer. This is what lets a caller concentrate resolution wherever they
need it -- more, closer-together edges near the center for a nuclear
star cluster, or a denser wedge of `theta_edges` in one direction for a
known structural feature -- without any special-casing in this module.
`log_radial_edges`/`linear_radial_edges`/`uniform_theta_edges`/
`build_reflected_sin_theta_edges` below are convenience builders for the
common, uniform cases; nothing stops a caller building a nonuniform edge
array by hand instead.

**Circular only.** No position-angle/ellipticity *flattening* -- a bin's
`r` is always a plain Euclidean distance from `center`, never an
elliptical radius, regardless of `pa_deg` (which only rotates where
`theta=0` points, not the shape of the radial annuli themselves). This is
the right tradeoff for a roughly-round target and the wrong one for a
strongly flattened one, where equal-`r` annuli don't track the target's
own isophotes. Not implemented here -- see this package's own notebooks
for where this limitation is called out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def log_radial_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
    """`n_bins + 1` radial edges, logarithmically spaced from `r_min` to
    `r_max` (both must be > 0). The natural default for a galaxy, where
    surface brightness/S-N falls off over orders of magnitude in radius
    and most of the interesting structure (a nuclear component, a steep
    inner profile) is concentrated at small `r`.
    """
    if r_min <= 0:
        raise ValueError(f"log_radial_edges: r_min must be > 0 (got {r_min}) -- use "
                          "linear_radial_edges, or a nonzero inner radius, for a "
                          "grid that needs to start at r=0.")
    return np.geomspace(r_min, r_max, n_bins + 1)


def linear_radial_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
    """`n_bins + 1` radial edges, linearly spaced from `r_min` to `r_max`.
    `r_min` may be 0. Prefer `log_radial_edges` unless there's a specific
    reason to want uniform physical width per annulus regardless of
    radius (e.g. deliberately coarser sampling of the outskirts than a
    log grid would give).
    """
    return np.linspace(r_min, r_max, n_bins + 1)


def uniform_theta_edges(n_bins: int, theta0: float = 0.0) -> np.ndarray:
    """`n_bins + 1` angular edges, equally spaced in `theta` itself (not
    `sin(theta)`) over one full turn starting at `theta0` (radians). Every
    sector spans the same angle, `2*pi / n_bins`.
    """
    return theta0 + np.linspace(0.0, 2.0 * np.pi, n_bins + 1)


def build_reflected_sin_theta_edges(n_per_quadrant: int) -> np.ndarray:
    """Full-circle theta edges (radians, `[0, 2*pi]`), equally spaced in
    `sin(theta)` (not `theta` itself) within each quadrant -- finest near
    `theta=0`/`180 deg`, coarsest near `90`/`270 deg` -- then mirrored into
    all 4 quadrants so the pattern repeats smoothly with no jump at any
    quadrant boundary.

    See `binning.py`'s module docstring for why the package's own default,
    `uniform_theta_edges`, deliberately does *not* use this convention:
    when paired with `assign_spaxels_to_bins`'s `pa_deg` parameter to
    measure `theta` from a known kinematic major axis, sectors come out
    narrowest along the major axis and widest along the minor axis,
    concentrating angular resolution where rotation actually shows up in
    the LOSVD. The `n_per_quadrant` angular edges are exactly
    `arcsin(linspace(0, 1, n_per_quadrant + 1))` in degrees.

    Only the geometric edge pattern -- every one of the resulting
    `4 * n_per_quadrant` sectors is fully independent (no combination
    across mirror-image quadrants), so an asymmetric feature shows up as
    an asymmetry in downstream maps rather than being averaged away.
    """
    frac = np.linspace(0.0, 1.0, n_per_quadrant + 1)
    q1 = np.degrees(np.arcsin(frac))          # 0 -> 90
    q2 = 180.0 - q1[::-1]                     # 90 -> 180 (mirror)
    q3 = 180.0 + q1[1:]                       # 180 -> 270 (repeat)
    q4 = 360.0 - q1[::-1][1:]                 # 270 -> 360 (mirror)
    theta_edges_deg = np.concatenate([q1, q2[1:], q3, q4])
    return np.radians(theta_edges_deg)


@dataclass
class DartboardBins:
    """One dartboard binning scheme: `r_edges` (shape `(n_radial+1,)`,
    strictly increasing, same physical units as the pixel-coordinate
    distances `assign_spaxels_to_bins` is called with -- typically
    arcsec) crossed with `theta_edges` (shape `(n_angular+1,)`, radians,
    strictly increasing, normally spanning one full `2*pi` turn -- see
    `log_radial_edges`/`linear_radial_edges`/`uniform_theta_edges`).

    `center` : `(x0, y0)` in the same pixel-coordinate convention the
        data this scheme is applied to uses (e.g. 0-indexed spaxel
        column/row).
    `single_center_bin` : if True (default), every spaxel closer than
        `r_edges[1]` -- including closer than `r_edges[0]` itself, e.g.
        the literal center pixel -- collapses into one bin regardless of
        `theta`: angular sectors are usually too small at the smallest
        radii to be individually meaningful, and `r_edges[0]` (which
        `log_radial_edges` requires to be > 0) marks where the *sectored*
        annuli beyond the center start, not a hole that excludes the
        center itself. Set False to sector the innermost annulus like
        every other one instead (in that mode, spaxels closer than
        `r_edges[0]` are genuinely unassigned -- see
        `assign_spaxels_to_bins`).
    """
    r_edges: np.ndarray
    theta_edges: np.ndarray
    center: tuple[float, float]
    single_center_bin: bool = True

    def __post_init__(self):
        self.r_edges = np.asarray(self.r_edges, dtype=float)
        self.theta_edges = np.asarray(self.theta_edges, dtype=float)
        if self.r_edges.ndim != 1 or len(self.r_edges) < 2:
            raise ValueError("DartboardBins: r_edges must be a 1-D array of at least 2 edges "
                              f"(got shape {self.r_edges.shape})")
        if self.theta_edges.ndim != 1 or len(self.theta_edges) < 2:
            raise ValueError("DartboardBins: theta_edges must be a 1-D array of at least 2 edges "
                              f"(got shape {self.theta_edges.shape})")
        if np.any(np.diff(self.r_edges) <= 0):
            raise ValueError("DartboardBins: r_edges must be strictly increasing")
        if np.any(np.diff(self.theta_edges) <= 0):
            raise ValueError("DartboardBins: theta_edges must be strictly increasing")

    @property
    def n_radial(self) -> int:
        return len(self.r_edges) - 1

    @property
    def n_angular(self) -> int:
        return len(self.theta_edges) - 1

    @property
    def n_bins(self) -> int:
        """Total distinct bin count, accounting for `single_center_bin`
        collapsing the innermost annulus's angular sectors into one.
        """
        if self.single_center_bin:
            return 1 + (self.n_radial - 1) * self.n_angular
        return self.n_radial * self.n_angular


def assign_spaxels_to_bins(shape: tuple[int, int], bins: DartboardBins,
                            pixel_scale: float = 1.0, pa_deg: float = 0.0) -> tuple[np.ndarray, list[dict]]:
    """Assign every spaxel in a `shape = (ny, nx)` image to a dartboard
    bin.

    Parameters
    ----------
    shape : `(ny, nx)`, the spatial shape of the cube/image being binned.
    bins : a `DartboardBins` scheme. `bins.center` and `bins.r_edges` are
        both in physical units (e.g. arcsec) once scaled by
        `pixel_scale`; `bins.center` itself is in pixel coordinates.
    pixel_scale : physical size of one pixel (e.g. arcsec/pixel), used to
        convert pixel-grid distances into the same units as `r_edges`.
    pa_deg : rotate `theta=0` to point along this position angle (degrees,
        counterclockwise from the pixel-grid `+x` axis) instead of the raw
        `+x` axis itself. Default `0.0` reproduces the original
        axis-agnostic behavior exactly (this module's own docstring
        explains why that's the right default when no axis is assumed);
        pass a known kinematic major axis together with
        `build_reflected_sin_theta_edges`'s edge pattern to concentrate
        angular resolution where rotation actually shows up in the LOSVD.
        Purely a rotation of the `theta` coordinate -- `r` (a plain
        Euclidean distance) is unaffected either way.

    Returns
    -------
    `(bin_id_map, bin_table)` -- `bin_id_map` is an `(ny, nx)` int array,
    `-1` for any spaxel outside every bin (outside `r_edges[-1]`, always;
    also inside `r_edges[0]` when `single_center_bin` is False and
    `r_edges[0] > 0` -- with the default `single_center_bin=True`,
    nothing inside `r_edges[-1]` is ever unassigned on the inner side,
    since the center bin absorbs everything closer than `r_edges[1]`);
    valid bin ids run `0` to `bins.n_bins - 1`.
    `bin_table` is a list of dicts, one per populated bin id, with keys
    `bin_id`, `r_index`, `theta_index` (`None` for the collapsed center
    bin), `r_mid`, `theta_mid` (`None` for the collapsed center bin),
    `n_spaxels`.
    """
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    dx = (x - bins.center[0]) * pixel_scale
    dy = (y - bins.center[1]) * pixel_scale
    if pa_deg != 0.0:
        pa = np.radians(pa_deg)
        dx, dy = dx * np.cos(pa) + dy * np.sin(pa), -dx * np.sin(pa) + dy * np.cos(pa)
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.mod(np.arctan2(dy, dx) - bins.theta_edges[0], 2.0 * np.pi) + bins.theta_edges[0]

    n_r, n_theta = bins.n_radial, bins.n_angular
    r_idx = np.searchsorted(bins.r_edges, r, side="right") - 1
    theta_idx = np.searchsorted(bins.theta_edges, theta, side="right") - 1
    valid = (r_idx >= 0) & (r_idx < n_r) & (theta_idx >= 0) & (theta_idx < n_theta)

    bin_id_map = np.full((ny, nx), -1, dtype=int)
    bin_table = []

    if bins.single_center_bin:
        # Everything closer than the first *interior* radial boundary collapses into
        # the center bin -- including r < r_edges[0] (e.g. the literal center pixel,
        # r=0), not just r in [r_edges[0], r_edges[1]). `r_edges[0]` (`log_radial_edges`
        # requires it > 0) is a lower bound for the *sectored* annuli beyond the center,
        # not a hole excluding the center itself -- a pixel more central than r_edges[0]
        # is still, if anything, more clearly part of the center bin, not less.
        is_center = r < bins.r_edges[1]
        bin_id_map[is_center] = 0
        n_center = int(np.sum(is_center))
        if n_center > 0:
            bin_table.append(dict(bin_id=0, r_index=0, theta_index=None,
                                   r_mid=0.5 * bins.r_edges[1],
                                   theta_mid=None, n_spaxels=n_center))
        rest = valid & (r_idx > 0)
        rest_bin_id = 1 + (r_idx - 1) * n_theta + theta_idx
        bin_id_map[rest] = rest_bin_id[rest]
        r_range = range(1, n_r)
    else:
        bin_id_map[valid] = r_idx[valid] * n_theta + theta_idx[valid]
        r_range = range(0, n_r)

    for ir in r_range:
        r_mid = 0.5 * (bins.r_edges[ir] + bins.r_edges[ir + 1])
        for it in range(n_theta):
            if bins.single_center_bin:
                bin_id = 1 + (ir - 1) * n_theta + it
            else:
                bin_id = ir * n_theta + it
            n_spaxels = int(np.sum(bin_id_map == bin_id))
            if n_spaxels > 0:
                theta_mid = 0.5 * (bins.theta_edges[it] + bins.theta_edges[it + 1])
                bin_table.append(dict(bin_id=bin_id, r_index=ir, theta_index=it,
                                       r_mid=r_mid, theta_mid=theta_mid, n_spaxels=n_spaxels))

    bin_table.sort(key=lambda d: d["bin_id"])
    return bin_id_map, bin_table
