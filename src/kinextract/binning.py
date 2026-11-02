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
from astropy.io import ascii


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


def pixel_aware_radial_edges(pixel_scale: float, r_max: float, n_bins: int) -> np.ndarray:
    """`n_bins + 1` radial edges for use with `DartboardBins(..., single_center_bin=True)`,
    built so the collapsed center bin contains *only* the literal center pixel --
    `r_edges[1] = pixel_scale` exactly, so any spaxel at 1 pixel or farther from
    center (the nearest ring of real neighbors) falls in a genuine annulus, not
    swept into the center bin along with it. The remaining `n_bins - 1` edges grow
    geometrically (`log_radial_edges`) from `pixel_scale` out to `r_max`.

    Exists because reusing a reference dataset's own radial bin edges (e.g. a
    legacy `bin_r.out`, generated for a specific target's own field of view) and
    rescaling them by an unrelated cube's own field-of-view fraction does not
    generally line up with *that* cube's own pixel scale -- concretely, uniformly
    shrinking a pattern built for a much larger field of view can push several of
    the innermost bins below one pixel's own width, so no real spaxel can ever
    populate them (see `dev_notes/legacy_binning/FINDINGS.md`'s 2026-08-07 entry
    for a real case: NGC 4751's own MUSE NFM data, 0.025"/pixel, left bins 2-7 of
    a rescaled 20-bin pattern entirely empty). Building `r_edges` directly from
    the cube's own `pixel_scale` instead makes that structurally impossible: the
    smallest non-center annulus always starts exactly at 1 pixel.

    `r_edges[0]` (the collapsed center bin's own nominal inner edge) is a
    formality `DartboardBins` doesn't use as a real boundary when
    `single_center_bin=True` (see that class's own docstring) -- set to
    `pixel_scale / 1000` here only so the array stays strictly positive and
    increasing, never interpreted as a real cutoff.
    """
    if pixel_scale <= 0:
        raise ValueError(f"pixel_aware_radial_edges: pixel_scale must be > 0 (got {pixel_scale})")
    if r_max <= pixel_scale:
        raise ValueError(f"pixel_aware_radial_edges: r_max ({r_max}) must be > pixel_scale "
                          f"({pixel_scale}) -- need at least one real annulus beyond the center bin.")
    if n_bins < 2:
        raise ValueError(f"pixel_aware_radial_edges: n_bins must be >= 2 (a center bin plus at "
                          f"least one real annulus), got {n_bins}")
    beyond_center = log_radial_edges(pixel_scale, r_max, n_bins - 1)
    return np.concatenate([[pixel_scale / 1000.0], beyond_center])


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
    bin_r = ascii.read(bin_r_path, data_start=2, names=["bin_id", "r_low", "r_mid", "r_high", "bin_size"], guess=False)
    bin_v = ascii.read(bin_v_path, data_start=2, names=["bin_id", "v_low", "v_mid", "v_high", "bin_size"], guess=False)

    r_edges = np.concatenate([[bin_r["r_low"][0]], np.asarray(bin_r["r_high"])])
    v_edges_deg = np.concatenate([[0.0], np.asarray(bin_v["v_high"])])
    return r_edges, v_edges_deg


def assign_legacy_dartboard_bins(shape: tuple[int, int], center: tuple[float, float], pa_deg: float,
                                  r_edges: np.ndarray, v_edges_deg: np.ndarray,
                                  pixel_scale: float = 1.0,
                                  fold_major_axis: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Assign every spaxel in a `shape = (ny, nx)` image to a bin using
    the SCO Fortran framework's own dartboard convention (`plotbin3.f`'s
    `writebin`, ported exactly -- see this module's own docstring for why
    this differs from `assign_spaxels_to_bins`). Verified line-by-line
    against `sco_framework/modprogs/plotbin3.f`'s own main loop
    (2026-08-06): the rotation, side split, fold, radius/angle binning,
    and `bin####`/`bnn####` naming below all reproduce that source
    exactly when `fold_major_axis=True` (the legacy behavior, and this
    function's default -- see that parameter's own docstring for why the
    default does not change existing callers' behavior).

    For a spaxel at `(dx, dy)` = pixel offset from `center`, scaled to
    arcsec by `pixel_scale`:

    1. Rotate so the major axis lies along `+x`: `pa_deg` is the
       kinematic major-axis PA, counterclockwise from the image `+x` axis
       (the convention of `assign_spaxels_to_bins`, of the bin-map
       plotting, and of the kinematic-PA fit), so `x = dx*cos(pa) +
       dy*sin(pa)`, `y = -dx*sin(pa) + dy*cos(pa)`. `plotbin3.f` instead
       reads a "rotation axis" `ang` and computes `x = xp*cos(ang) -
       yp*sin(ang)`, i.e. `ang = -pa_deg`. (Before 2026-10-16 this function
       used `+pa_deg` there, which folded the dartboard about the major
       axis mirrored in `x`: for NGC 4751's 85.66 deg, about a line at
       94.34 deg, 8.7 deg off.)
    2. Side: `x >= 0` -> `"bin"` (sign `+1`); `x < 0` -> `"bnn"` (sign
       `-1`, then `x` is mirrored positive for the lookup below). This
       split is across the *kinematic minor axis* (north/south of the
       rotation-axis line) -- a different cut from step 3's major-axis
       fold.
    3. Fold: `y = abs(y)` -- north and south of the *major* axis are
       merged into the same bin (the real, physical assumption the
       legacy scheme makes; see this module's own docstring for why this
       also means a bin can never straddle a major-axis exclusion mask).
       Only applied when `fold_major_axis=True` -- see that parameter.
    4. Radius `r = sqrt(x**2 + y**2)` -> radial index `irc` (1-based, 1 =
       innermost) via `r_edges` (from `load_legacy_bin_edges`). `r`
       smaller than `r_edges[0]` (e.g. the literal center spaxel) clips
       up to `r_edges[0]` -- absorbed into the innermost ring rather than
       left unassigned, the same philosophy `DartboardBins`'
       `single_center_bin=True` already uses.
    5. Angle `vp = degrees(atan2(y, x))`, clipped to `[0, 89.99]` ->
       angular index `irv` (1-based, 1 = nearest the major axis) via
       `v_edges_deg`. `y` here is always `abs(y)` for this lookup
       regardless of `fold_major_axis` (`v_edges_deg` only covers
       `[0, 90)` either way) -- what `fold_major_axis` controls is
       whether the north (`y>=0`) and south (`y<0`) halves sharing the
       same `(irc, irv, side)` are then merged into one bin or kept apart.
    6. Bin name: `f"{'bin' if sign>0 else 'bnn'}{irc:02d}{irv:02d}"` --
       7 characters, e.g. `"bin0301"` = radial ring 3, angular sector 1,
       positive minor-axis side. This is the literal spectrum-file stem
       the rest of the legacy pipeline (`fitlovw`, `mcfitw`, `pallmc`,
       ...) expects. Only exact when `fold_major_axis=True`; see that
       parameter for the non-folded naming.

    Parameters
    ----------
    fold_major_axis : if `True` (the default, and the legacy
        `plotbin3.f` behavior exactly), a bin's north (`y>=0`) and south
        (`y<0`) spaxels are merged, matching the original Fortran
        pipeline (whose own downstream tools -- `fitlovw`, `pallmc`,
        ... -- expect exactly this 7-character `bin####`/`bnn####`
        naming and cannot consume anything else). Defaults to `True`
        specifically so every existing caller of this function (the
        N4751 real-data notebooks, `kinematicextraction`, ...) keeps its
        current behavior unchanged -- this is new, additive capability,
        not a behavior change.

        If `False`, north and south are kept as separate bins instead
        (each still individually folded to `|y|` for the radius/angle
        *lookup* -- step 5 -- but no longer merged into one bin
        afterward): a real modernization, not just a config knob, since
        it recovers kinematic information (e.g. genuine north/south
        asymmetry from a warp, or from the galaxy's own real velocity
        field failing point-symmetry -- see `EHT_Project/notebooks/
        N4751_notebooks/kinematics/10_kinematic_center_pa_development.
        ipynb` for a real, current example of exactly that failure mode)
        the folded scheme discards by construction. Bin names in this
        mode are **not** legacy-pipeline-compatible: the same
        `bin####`/`bnn####` stem is used for the north (`y>=0`) half,
        and an `s` is appended (`bin####s`/`bnn####s`, 8 characters) for
        the south (`y<0`) half, so this mode should only feed a
        Python/kinextract-native workflow, never the legacy Fortran
        tools directly.

    Returns
    -------
    `(bin_id_map, bin_table)` -- `bin_id_map` is an `(ny, nx)` int array,
    `-1` outside every bin (`r` beyond `r_edges[-1]`). Bin ids are unique
    integers (`irc`, `irv`, side, and -- when `fold_major_axis=False` --
    hemisphere, all packed together) so this return shape is a drop-in
    match for `assign_spaxels_to_bins`'s own (usable by the same
    `plot_bin_map`-style plotting code); `bin_table` is a list of dicts,
    one per populated bin, with keys `bin_id`, `bin_name` (the legacy
    `bin####`/`bnn####` string, or its `s`-suffixed south-half variant
    when unfolded), `irc`, `irv`, `side` (`+1`/`-1`), `hemisphere`
    (`+1` = north/`y>=0`, `-1` = south/`y<0` -- always populated, even
    when `fold_major_axis=True`, in which case every bin's own
    `hemisphere` is `+1` by convention since north/south were merged into
    it), `r_mid` (arcsec), `vp_mid` (degrees from the major axis),
    `theta_mid` (signed radians from the major axis, `+vp` on the `bin`
    side / `-vp`... no -- see note below), `n_spaxels`.

    `theta_mid` here is provided purely for compatibility with plotting
    code written against `assign_spaxels_to_bins`'s own `theta_mid`
    convention (signed angle, radians) -- it is `radians(vp_mid)` on the
    `bin` (`x>=0`) side and `pi - radians(vp_mid)` on the `bnn` (`x<0`)
    side, i.e. the angle this bin's own representative point would have
    in the *unfolded*, unmirrored frame `assign_spaxels_to_bins` uses.
    When `fold_major_axis=False`, this is further negated for the south
    (`hemisphere=-1`) half, so a north/south pair's own `theta_mid`
    values are themselves reflections of each other, matching their real
    sky positions.
    """
    ny, nx = shape
    r_edges = np.asarray(r_edges, dtype=float)
    v_edges_deg = np.asarray(v_edges_deg, dtype=float)
    n_r, n_v = len(r_edges) - 1, len(v_edges_deg) - 1

    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = (xx - center[0]) * pixel_scale
    dy = (yy - center[1]) * pixel_scale
    pa = np.radians(pa_deg)
    x = dx * np.cos(pa) + dy * np.sin(pa)       # plotbin3.f's rotation with ang = -pa_deg (see step 1)
    y = -dx * np.sin(pa) + dy * np.cos(pa)

    side = np.where(x >= 0, 1, -1)
    hemisphere = np.where(y >= 0, 1, -1)
    x_side = np.abs(x)
    y_fold = np.abs(y)
    r = np.clip(np.sqrt(x_side ** 2 + y_fold ** 2), r_edges[0], None)
    # plotbin3.f's own branch is literally `if(abs(x).ne.0) vp=atan(y/x) else vp=89.99`,
    # evaluated *after* x/y are already folded positive (x_side/y_fold here) -- for any
    # x_side>0 this is identical to atan2(y_fold, x_side) (single-quadrant, matches
    # exactly), but at the exact grid center (x_side=0 *and* y_fold=0 -- the literal
    # single-spaxel innermost bin) `np.arctan2(0, 0)` silently returns 0.0, not the
    # Fortran's forced 89.99. That 0 vs. 89.99 difference lands the center spaxel in the
    # *first* angular sector (irv=1) here vs. the *last* one (irv=n_v) in the legacy
    # pipeline -- confirmed directly against a real run: this function reports the
    # center bin as "bin0101", `plotbin3.f` writes "bin0105" for the same target's own
    # 5-sector bin_v.out. Not caught by the earlier line-by-line fidelity check (this
    # module's own docstring) because that check exercised the general (non-degenerate)
    # mechanics, not this specific x==0-exactly edge case.
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = np.where(x_side != 0, np.degrees(np.arctan2(y_fold, x_side)), 89.99)
    vp = np.nan_to_num(np.clip(vp, 0.0, 89.99), nan=89.99)

    irc = np.searchsorted(r_edges, r, side="right") - 1  # 0-based here, +1 -> legacy 1-based
    irv = np.searchsorted(v_edges_deg, vp, side="right") - 1
    valid = (r <= r_edges[-1]) & (irc >= 0) & (irc < n_r) & (irv >= 0) & (irv < n_v)

    # bin_id packs (side, irc, irv) -- and, when not folding, hemisphere -- into one
    # unique int. side offset by 1000 keeps "bin" (offset 0) and "bnn" (offset 1000)
    # ids disjoint for any n_r*n_v < 1000 (the standard 20x5=100 legacy grid included);
    # a further hemisphere offset of 2000 (only added when fold_major_axis=False) keeps
    # the south half disjoint from both, rather than colliding with the folded (or
    # north-only) ids -- stable/deterministic across calls, unlike an arbitrary
    # enumeration order.
    side_offset = np.where(side > 0, 0, 1000)
    if fold_major_axis:
        hemi_offset = np.zeros_like(side_offset)
    else:
        hemi_offset = np.where(hemisphere > 0, 0, 2000)
    bin_id_raw = side_offset + hemi_offset + irc * n_v + irv

    bin_id_map = np.full((ny, nx), -1, dtype=int)
    bin_id_map[valid] = bin_id_raw[valid]

    hemispheres = (1,) if fold_major_axis else (1, -1)
    bin_table = []
    for s, prefix in ((1, "bin"), (-1, "bnn")):
        for h in hemispheres:
            hemi_offset_h = 0 if (fold_major_axis or h > 0) else 2000
            hemi_suffix = "" if (fold_major_axis or h > 0) else "s"
            for ir in range(n_r):
                r_mid = 0.5 * (r_edges[ir] + r_edges[ir + 1])
                for iv in range(n_v):
                    bid = (0 if s > 0 else 1000) + hemi_offset_h + ir * n_v + iv
                    n_spaxels = int(np.sum(bin_id_map == bid))
                    if n_spaxels == 0:
                        continue
                    vp_mid = 0.5 * (v_edges_deg[iv] + v_edges_deg[iv + 1])
                    theta_mid = np.radians(vp_mid) if s > 0 else np.pi - np.radians(vp_mid)
                    if h < 0:
                        theta_mid = -theta_mid
                    bin_table.append(dict(
                        bin_id=bid, bin_name=f"{prefix}{ir + 1:02d}{iv + 1:02d}{hemi_suffix}",
                        irc=ir + 1, irv=iv + 1, side=s, hemisphere=h, r_mid=r_mid, vp_mid=vp_mid,
                        theta_mid=theta_mid, n_spaxels=n_spaxels,
                    ))

    bin_table.sort(key=lambda d: d["bin_id"])
    return bin_id_map, bin_table
