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

**A third convention: `assign_legacy_dartboard_bins`, a faithful port of
the SCO Fortran framework's own `plotbin3.f` binning** (`bin_r.out`/
`bin_v.out`-driven, `bin####`/`bnn####` naming), for real targets whose
downstream analysis is a legacy Schwarzschild-modeling pipeline that
expects that exact scheme. Unlike `build_reflected_sin_theta_edges`
(which *centers* its narrowest sector *on* `theta=0`, so a sector can
straddle an assumed symmetry axis), this convention folds `y` about the
major axis (`abs(y)`) before computing the angular index -- `theta=0`
becomes a bin *edge*, never a sector's own center, which is also what
makes it safe to use directly against data that's been masked on one
whole side of that axis (nothing can straddle the boundary to begin
with). See `assign_legacy_dartboard_bins`'s own docstring for the exact
geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def load_legacy_bin_edges(bin_r_path: str, bin_v_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the SCO Fortran framework's own `bin_r.out`/`bin_v.out`
    dartboard-geometry files (`irc rlow rmid rup bin_size` / `irv vlow
    vmid vup bin_size`, 2 header lines then one row per bin, produced by
    `gden.f`'s `getbinp` subroutine) and return `(r_edges, v_edges_deg)`:

    - `r_edges` : `n_radial + 1` radial edges (arcsec), `[rlow[0], rup[0],
      rup[1], ..., rup[-1]]` -- `rlow[0]` is not `0` in the real files
      (a small inner "core" bin is handled separately upstream), so this
      is the true innermost edge, not an assumed zero.
    - `v_edges_deg` : `n_angular + 1` angular edges (degrees, `0` =
      major axis, `90` = minor axis), `[0, vup[0], vup[1], ..., vup[-1]]`.

    Standard files have 20 radial and 5 angular bins (`Nrlib=20, Nvlib=5`
    in the Fortran framework's own `bothdefs.h`), but this reads whatever
    row count is actually present, not a hardcoded 20/5.

    These edges are a *relative pattern*, not literal final arcsec values
    for every target -- the framework's own `angleRmax` (from that
    target's `galaxy.params`) sets the overall physical scale, and
    real-target notebooks in this package's own examples rescale this
    pattern to their own cube's field of view rather than using it as-is
    (this is why the same `bin_r.out` numbers show up unchanged across
    different galaxies' working directories -- expected, not a bug).
    """
    r_rows = np.loadtxt(bin_r_path, skiprows=2)
    v_rows = np.loadtxt(bin_v_path, skiprows=2)
    r_edges = np.concatenate([[r_rows[0, 1]], r_rows[:, 3]])
    v_edges_deg = np.concatenate([[0.0], v_rows[:, 3]])
    return r_edges, v_edges_deg


def assign_legacy_dartboard_bins(shape: tuple[int, int], center: tuple[float, float], pa_deg: float,
                                  r_edges: np.ndarray, v_edges_deg: np.ndarray,
                                  pixel_scale: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Assign every spaxel in a `shape = (ny, nx)` image to a bin using
    the SCO Fortran framework's own dartboard convention (`plotbin3.f`'s
    `writebin`, ported exactly -- see this module's own docstring for why
    this differs from `assign_spaxels_to_bins`).

    For a spaxel at `(dx, dy)` = pixel offset from `center`, scaled to
    arcsec by `pixel_scale`:

    1. Rotate by `pa_deg` (the kinematic major-axis PA, same convention
       `assign_spaxels_to_bins`'s own `pa_deg` uses): `x = dx*cos(pa) -
       dy*sin(pa)`, `y = dx*sin(pa) + dy*cos(pa)`.
    2. Side: `x >= 0` -> `"bin"` (sign `+1`); `x < 0` -> `"bnn"` (sign
       `-1`, then `x` is mirrored positive for the lookup below). This
       split is across the *kinematic minor axis* (north/south of the
       rotation-axis line) -- a different cut from step 3's major-axis
       fold.
    3. Fold: `y = abs(y)` -- north and south of the *major* axis are
       merged into the same bin (the real, physical assumption the
       legacy scheme makes; see this module's own docstring for why this
       also means a bin can never straddle a major-axis exclusion mask).
    4. Radius `r = sqrt(x**2 + y**2)` -> radial index `irc` (1-based, 1 =
       innermost) via `r_edges` (from `load_legacy_bin_edges`). `r`
       smaller than `r_edges[0]` (e.g. the literal center spaxel) clips
       up to `r_edges[0]` -- absorbed into the innermost ring rather than
       left unassigned, the same philosophy `DartboardBins`'
       `single_center_bin=True` already uses.
    5. Angle `vp = degrees(atan2(y, x))`, clipped to `[0, 89.99]` ->
       angular index `irv` (1-based, 1 = nearest the major axis) via
       `v_edges_deg`.
    6. Bin name: `f"{'bin' if sign>0 else 'bnn'}{irc:02d}{irv:02d}"` --
       7 characters, e.g. `"bin0301"` = radial ring 3, angular sector 1,
       positive minor-axis side. This is the literal spectrum-file stem
       the rest of the legacy pipeline (`fitlovw`, `mcfitw`, `pallmc`,
       ...) expects.

    Returns
    -------
    `(bin_id_map, bin_table)` -- `bin_id_map` is an `(ny, nx)` int array,
    `-1` outside every bin (`r` beyond `r_edges[-1]`). Bin ids are unique
    integers (`irc`, `irv`, side packed together) so this return shape is
    a drop-in match for `assign_spaxels_to_bins`'s own (usable by the same
    `plot_bin_map`-style plotting code); `bin_table` is a list of dicts,
    one per populated bin, with keys `bin_id`, `bin_name` (the legacy
    `bin####`/`bnn####` string), `irc`, `irv`, `side` (`+1`/`-1`), `r_mid`
    (arcsec), `vp_mid` (degrees from the major axis), `theta_mid` (signed
    radians from the major axis, `+vp` on the `bin` side / `-vp`... no --
    see note below), `n_spaxels`.

    `theta_mid` here is provided purely for compatibility with plotting
    code written against `assign_spaxels_to_bins`'s own `theta_mid`
    convention (signed angle, radians) -- it is `radians(vp_mid)` on the
    `bin` (`x>=0`) side and `pi - radians(vp_mid)` on the `bnn` (`x<0`)
    side, i.e. the angle this bin's own representative point would have
    in the *unfolded*, unmirrored frame `assign_spaxels_to_bins` uses.
    """
    ny, nx = shape
    r_edges = np.asarray(r_edges, dtype=float)
    v_edges_deg = np.asarray(v_edges_deg, dtype=float)
    n_r, n_v = len(r_edges) - 1, len(v_edges_deg) - 1

    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = (xx - center[0]) * pixel_scale
    dy = (yy - center[1]) * pixel_scale
    pa = np.radians(pa_deg)
    x = dx * np.cos(pa) - dy * np.sin(pa)
    y = dx * np.sin(pa) + dy * np.cos(pa)

    side = np.where(x >= 0, 1, -1)
    x_side = np.abs(x)
    y_fold = np.abs(y)
    r = np.clip(np.sqrt(x_side ** 2 + y_fold ** 2), r_edges[0], None)
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = np.degrees(np.arctan2(y_fold, x_side))
    vp = np.nan_to_num(np.clip(vp, 0.0, 89.99), nan=89.99)

    irc = np.searchsorted(r_edges, r, side="right") - 1  # 0-based here, +1 -> legacy 1-based
    irv = np.searchsorted(v_edges_deg, vp, side="right") - 1
    valid = (r <= r_edges[-1]) & (irc >= 0) & (irc < n_r) & (irv >= 0) & (irv < n_v)

    # bin_id packs (side, irc, irv) into one unique int: bin side offset by 1000 keeps
    # "bin" (offset 0) and "bnn" (offset 1000) ids disjoint for any n_r*n_v < 1000
    # (the standard 20x5=100 legacy grid included) -- and is stable/deterministic across
    # calls, unlike an arbitrary enumeration order.
    side_offset = np.where(side > 0, 0, 1000)
    bin_id_raw = side_offset + irc * n_v + irv

    bin_id_map = np.full((ny, nx), -1, dtype=int)
    bin_id_map[valid] = bin_id_raw[valid]

    bin_table = []
    for s, prefix in ((1, "bin"), (-1, "bnn")):
        for ir in range(n_r):
            r_mid = 0.5 * (r_edges[ir] + r_edges[ir + 1])
            for iv in range(n_v):
                bid = (0 if s > 0 else 1000) + ir * n_v + iv
                n_spaxels = int(np.sum(bin_id_map == bid))
                if n_spaxels == 0:
                    continue
                vp_mid = 0.5 * (v_edges_deg[iv] + v_edges_deg[iv + 1])
                theta_mid = np.radians(vp_mid) if s > 0 else np.pi - np.radians(vp_mid)
                bin_table.append(dict(
                    bin_id=bid, bin_name=f"{prefix}{ir + 1:02d}{iv + 1:02d}",
                    irc=ir + 1, irv=iv + 1, side=s, r_mid=r_mid, vp_mid=vp_mid,
                    theta_mid=theta_mid, n_spaxels=n_spaxels,
                ))

    bin_table.sort(key=lambda d: d["bin_id"])
    return bin_id_map, bin_table
