"""Moment-map plotting for the legacy SCO dartboard binning scheme --
turns a directory of per-bin kinematic results into 2D V/sigma/h3/h4
maps, standalone (no cube, no WCS, no notebook needed): everything it
needs is `bin_r.out`/`bin_v.out` (the dartboard geometry, already used
elsewhere in this framework via `kinextract.binning.load_legacy_bin_edges`)
plus each bin's own `<bin>spmc.kinematics` file (or, for legacy `.sim`-based
bins with no bridge file, `pallmc.out`'s own summary line). Built for
`sco_framework_updated/bin/plotmoments`, a thin CLI wrapper around
`plot_moment_maps` meant to run standalone on Great Lakes right after a
`kinematicextraction`/`losvderrors` run -- no interactive display, just a
saved figure.

**Plotted in the kinematic-PA-aligned frame (major axis = x, minor axis =
y), not sky-aligned RA/Dec** -- `bin_r.out`/`bin_v.out` and a bin's own
`(irc, irv, side)` (recoverable from its name alone, see `parse_bin_name`)
carry no absolute sky orientation, only the dartboard's own relative
geometry, so there is nothing here to rotate by. This matches how this
kind of diagnostic map is conventionally shown in this pipeline (relative
to the kinematic major axis) and needs no extra metadata (cube WCS,
`pa_deg`) that may not be available in a bare GreatLakes run directory.

Every bin is drawn as up to 4 mirrored wedges (north/south of the major
axis, `bin`/`bnn` side of the minor axis) filled with that one bin's own
scalar value -- the direct visual consequence of `assign_legacy_dartboard_bins`'s
own `fold_major_axis=True` convention (north/south spaxels already merged
into one bin before any of this), not an approximation.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def parse_bin_name(bin_name: str) -> tuple[int, int, int]:
    """`"bin0301"` -> `(irc=3, irv=1, side=+1)`, `"bnn0705"` -> `(irc=7,
    irv=5, side=-1)` -- the inverse of `assign_legacy_dartboard_bins`'s own
    `f"{'bin' if sign>0 else 'bnn'}{irc:02d}{irv:02d}"` naming. Any
    trailing suffix after the 7-character `bin####`/`bnn####` stem (e.g.
    the `spmc` in `bin0301spmc`) is ignored, so this also accepts a raw
    `.kinematics`/`.mcfit` file's own stem directly.
    """
    if bin_name.startswith("bin"):
        side = 1
    elif bin_name.startswith("bnn"):
        side = -1
    else:
        raise ValueError(f"bin_name must start with 'bin' or 'bnn', got {bin_name!r}")
    digits = bin_name[3:7]
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"expected 4 digits after 'bin'/'bnn' in {bin_name!r}, got {digits!r}")
    return int(digits[:2]), int(digits[2:]), side


def load_bin_kinematics_dir(dirpath: str) -> dict[str, dict[str, float]]:
    """Read every `<bin>spmc.kinematics` file in `dirpath` (the bridge
    files `run_losvd_errors.py` writes -- see that script's own docstring
    for the exact `label value` line format, six lines always present:
    `v_16/50/84`, `sigma_16/50/84`; six more when the writer includes them:
    `h3_16/50/84`, `h4_16/50/84`) into `{bin_name: {"V":..., "V_lo":...,
    "V_hi":..., "sigma":..., "sigma_lo":..., "sigma_hi":..., "h3":...,
    "h4":..., ...}}` (h3/h4 keys only present when the file has them).
    `bin_name` is the filename with the trailing `spmc.kinematics` stripped.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(dirpath, "*spmc.kinematics"))):
        bin_name = os.path.basename(path)[: -len("spmc.kinematics")]
        values = {}
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                values[parts[0]] = float(parts[1])
        entry = {
            "V": values.get("v_50"), "V_lo": values.get("v_16"), "V_hi": values.get("v_84"),
            "sigma": values.get("sigma_50"), "sigma_lo": values.get("sigma_16"),
            "sigma_hi": values.get("sigma_84"),
        }
        if "h3_50" in values:
            entry.update(h3=values["h3_50"], h3_lo=values.get("h3_16"), h3_hi=values.get("h3_84"))
        if "h4_50" in values:
            entry.update(h4=values["h4_50"], h4_lo=values.get("h4_16"), h4_hi=values.get("h4_84"))
        out[bin_name] = entry
    return out


def load_pallmc_out(path: str) -> dict[str, dict[str, float]]:
    """Read `pallmc.out` (format `1101`, `pallmc.f`: `rad, vel, vel_err,
    sig, sig_err, h3, h3_err, h4, h4_err, filename`) into the same
    `{bin_name: {...}}` shape `load_bin_kinematics_dir` returns (symmetric
    errors here, since that is all `pallmc.out` itself carries -- `V_lo` =
    `V_hi` = `vel_err` on either side of `V`, etc.) -- the fallback for a
    bin with no `.kinematics` bridge file (a genuine legacy `.sim`-based
    bin, not a kinextract-produced one).
    """
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 10:
                continue
            _rad, vel, vel_err, sig, sig_err, h3, h3_err, h4, h4_err = (float(x) for x in parts[:9])
            filename = parts[9]
            bin_name = filename
            for suffix in ("spmc.sim", "sp.sim", ".sim"):
                if bin_name.endswith(suffix):
                    bin_name = bin_name[: -len(suffix)]
                    break
            out[bin_name] = dict(
                V=vel, V_lo=vel - vel_err, V_hi=vel + vel_err,
                sigma=sig, sigma_lo=sig - sig_err, sigma_hi=sig + sig_err,
                h3=h3, h3_lo=h3 - h3_err, h3_hi=h3 + h3_err,
                h4=h4, h4_lo=h4 - h4_err, h4_hi=h4 + h4_err,
            )
    return out


def load_bin_kinematics(kinematics_dir: str, pallmc_out_path: str | None = None) -> dict[str, dict]:
    """`load_bin_kinematics_dir(kinematics_dir)`, topped up with
    `load_pallmc_out(pallmc_out_path)` entries for any bin not already
    covered (i.e. every bin that went through the legacy `.sim`-based
    path instead of kinextract's own bridge files) -- the union a real
    mixed run (some bins refit with kinextract, some still legacy) needs.
    `.kinematics`-derived entries always win on overlap, since they carry
    kinextract's own real (possibly asymmetric) posterior, not a
    Monte-Carlo-ensemble approximation of it.
    """
    combined = load_bin_kinematics_dir(kinematics_dir)
    if pallmc_out_path and os.path.exists(pallmc_out_path):
        for bin_name, entry in load_pallmc_out(pallmc_out_path).items():
            combined.setdefault(bin_name, entry)
    return combined


def _wedge_polygon(r_lo: float, r_hi: float, theta1_deg: float, theta2_deg: float, n_arc: int = 8):
    """A filled annular wedge polygon from `theta1_deg` to `theta2_deg`
    (math convention: degrees CCW from +x), between radii `r_lo`/`r_hi` --
    built from explicit arc points rather than `matplotlib.patches.Wedge`
    so behavior at `theta2_deg > 360` (the `side=-1`/`bnn` mirrored wedges
    below routinely need this) is unambiguous."""
    from matplotlib.patches import Polygon
    thetas = np.radians(np.linspace(theta1_deg, theta2_deg, n_arc))
    outer = np.column_stack([r_hi * np.cos(thetas), r_hi * np.sin(thetas)])
    inner = np.column_stack([r_lo * np.cos(thetas[::-1]), r_lo * np.sin(thetas[::-1])])
    return Polygon(np.vstack([outer, inner]), closed=True)


def _bin_wedge_angles(t_lo: float, t_hi: float, side: int) -> list[tuple[float, float]]:
    """The (up to) 2 mirrored angle ranges (math convention, degrees) one
    `(irv, side)` sector covers in the fold_major_axis=True plotting frame:
    north and south of the major axis (x-axis) on the `side`'s own half."""
    if side > 0:
        return [(t_lo, t_hi), (-t_hi, -t_lo)]
    return [(180.0 - t_hi, 180.0 - t_lo), (180.0 + t_lo, 180.0 + t_hi)]


def plot_bin_map(ax, bin_table, bins, values_by_bin_id, center_pix, pa_deg, pixel_scale,
                  background=None, cmap="RdBu_r", vmin=None, vmax=None,
                  grid_color="black", grid_lw=0.5, grid_alpha=0.6, colorbar_label=None,
                  add_cbar=True):
    """Paints each populated dartboard bin as a filled wedge at its real
    sky position/extent, with explicit grid-line boundaries drawn on top --
    the *new*-scheme (`kinextract.binning.DartboardBins`/`assign_spaxels_to_bins`,
    `mpatches.Wedge`-based) counterpart to `plot_moment_maps` above, which
    is for the legacy string-keyed (`bin_name`/`r_edges`/`v_edges_deg`)
    scheme instead -- the two are not interchangeable, use whichever
    matches how your bins were assigned.

    Replaces `imshow(paint_map(...))`: a discrete per-pixel raster plus a
    continuous colormap gives adjacent bins with similar values no visual
    separation at all, and bin edges only show up as raster jaggedness,
    not a real line. `background` (e.g. the white-light image), if given,
    is drawn underneath at reduced alpha for spatial context -- the same
    idea as overlaying kinematic maps on a galaxy image in a real paper
    figure.

    A bin whose value is NaN (a failed fit -- see `cleaning._failed_fit_result`)
    renders as plain gray (`set_bad`), not a color implying a real
    measurement. Angular grid spokes and the kinematic-major-axis marker
    line are only drawn when there's more than one angular bin
    (`bins.n_angular > 1`) -- a single full-circle bin per annulus has no
    meaningful angular boundary to draw.

    Parameters
    ----------
    ax : the `matplotlib.axes.Axes` to draw on.
    bin_table : `assign_spaxels_to_bins`'s own per-bin row list (`bin_id`,
        `theta_mid`, `r_index`, `theta_index`, ...).
    bins : the `DartboardBins` `bin_table` was built from (`r_edges`,
        `theta_edges`, `n_angular`).
    values_by_bin_id : `{bin_id: scalar}` -- the quantity to color each
        bin by (e.g. a moment map value); a bin present in `bin_table` but
        missing here is left unfilled.
    center_pix, pa_deg, pixel_scale : the same geometry `bin_table` was
        assigned with -- `pa_deg` rotates the local (PA-relative) angle
        convention `assign_spaxels_to_bins` uses back to sky angle
        (`sky = R(+pa) @ local`, matching that function's own convention
        exactly).
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    bx, by = center_pix
    if background is not None:
        ax.imshow(background, origin="lower", cmap="gray", alpha=0.35, zorder=0)

    vals_present = np.array([v for v in values_by_bin_id.values() if np.isfinite(v)])
    if vmin is None:
        vmin = -np.nanmax(np.abs(vals_present)) if cmap == "RdBu_r" else np.nanmin(vals_present)
    if vmax is None:
        vmax = np.nanmax(np.abs(vals_present)) if cmap == "RdBu_r" else np.nanmax(vals_present)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj.set_bad("gray")

    for row in bin_table:
        bid = row["bin_id"]
        if bid not in values_by_bin_id:
            continue
        color = cmap_obj(norm(values_by_bin_id[bid]))
        if row["theta_mid"] is None:
            # collapsed center bin -- a full circle out to the first radial edge
            r_hi = bins.r_edges[1] / pixel_scale
            wedge = mpatches.Wedge((bx, by), r_hi, 0.0, 360.0, facecolor=color,
                                    edgecolor="none", zorder=1)
        else:
            r_lo = bins.r_edges[row["r_index"]] / pixel_scale
            r_hi = bins.r_edges[row["r_index"] + 1] / pixel_scale
            # local (PA-relative) angle -> sky angle, matching assign_spaxels_to_bins's
            # own rotation convention exactly (binning.py: sky = R(+pa) @ local)
            theta_lo = np.degrees(bins.theta_edges[row["theta_index"]]) + pa_deg
            theta_hi = np.degrees(bins.theta_edges[row["theta_index"] + 1]) + pa_deg
            wedge = mpatches.Wedge((bx, by), r_hi, theta_lo, theta_hi, width=r_hi - r_lo,
                                    facecolor=color, edgecolor="none", zorder=1)
        ax.add_patch(wedge)

    # grid lines drawn separately from the fill, so bin boundaries stay
    # crisp regardless of how similar two neighboring bins' colors are
    # r_edges[0] isn't a real bin boundary (the collapsed center bin's own
    # outer edge is r_edges[1] -- see the fill loop above), so skip it here
    # too, or it draws a small, meaningless circle inside the center bin
    t = np.linspace(0.0, 2 * np.pi, 400)
    for r_edge in bins.r_edges[1:]:
        rp = r_edge / pixel_scale
        ax.plot(bx + rp * np.cos(t), by + rp * np.sin(t), color=grid_color,
                lw=grid_lw, alpha=grid_alpha, zorder=2)
    # angular spokes start at the *center bin's own* outer edge (r_edges[1] --
    # matching the collapsed-center-bin fill above, not r_edges[0]/the true
    # center), so the innermost bin renders as a plain undivided circle
    # instead of looking cut into wedges
    if bins.n_angular > 1:
        r_inner_pix = bins.r_edges[1] / pixel_scale
        r_max_pix = bins.r_edges[-1] / pixel_scale
        for theta_edge in bins.theta_edges:
            th = theta_edge + np.radians(pa_deg)
            ax.plot([bx + r_inner_pix * np.cos(th), bx + r_max_pix * np.cos(th)],
                    [by + r_inner_pix * np.sin(th), by + r_max_pix * np.sin(th)],
                    color=grid_color, lw=grid_lw, alpha=grid_alpha, zorder=2)
        # kinematic major axis (local theta=0, the angular scheme's own narrowest-sector
        # direction -- see assign_spaxels_to_bins's own PA-rotation convention above) --
        # drawn as a full diameter through the center, not just one ray, so both the
        # receding and approaching sides are marked. White + dotted so it reads clearly
        # against both the dark grid lines and whatever colormap is in use.
        th_major = np.radians(pa_deg)
        ax.plot([bx - r_max_pix * np.cos(th_major), bx + r_max_pix * np.cos(th_major)],
                [by - r_max_pix * np.sin(th_major), by + r_max_pix * np.sin(th_major)],
                color="white", lw=1.5, ls=":", zorder=3)

    if add_cbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    ax.set_aspect("equal")


def plot_moment_maps(bin_values: dict[str, dict], r_edges, v_edges_deg,
                      moments: tuple[str, ...] = ("V", "sigma", "h3", "h4"),
                      cmaps: dict[str, str] | None = None,
                      vlims: dict[str, tuple[float, float]] | None = None,
                      out_path: str | None = None, title: str | None = None,
                      figsize: tuple[float, float] = (16, 4.5), dpi: int = 150):
    """Render a `1 x len(moments)` panel figure, one dartboard map per
    moment, from `bin_values` (`{bin_name: {"V":..., "sigma":..., "h3":...,
    "h4":...}}`, e.g. from `load_bin_kinematics`) and the dartboard's own
    `r_edges`/`v_edges_deg` (from `kinextract.binning.load_legacy_bin_edges`).

    Parameters
    ----------
    cmaps : per-moment matplotlib colormap name; defaults to
        `{"V": "RdBu_r", "sigma": "viridis", "h3": "PuOr_r", "h4": "PuOr_r"}`
        for any moment not given explicitly (a diverging map for the
        signed quantities V/h3/h4, sequential for sigma).
    vlims : per-moment `(vmin, vmax)` colorbar range; a moment not given
        here is scaled symmetrically about 0 to the run's own max
        `abs(value)` (V/h3/h4) or `(0, max)` (sigma) -- never per-panel
        independently rescaled in a way that would make two different
        bins' colors incomparable across a saved figure re-run with a
        different bin subset, since the caller can always pin `vlims`
        explicitly for that.
    out_path : if given, the figure is saved here (format from the
        extension, e.g. `.png`/`.pdf`) and closed; otherwise the caller
        gets the open `Figure` back.

    Returns
    -------
    The `matplotlib.figure.Figure` (already saved and closed if `out_path`
    was given -- in that case the returned object is a closed figure,
    useful only for tests that check panel count/titles before saving,
    not for further interactive use).
    """
    import matplotlib
    if out_path is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    default_cmaps = {"V": "RdBu_r", "sigma": "viridis", "h3": "PuOr_r", "h4": "PuOr_r"}
    cmaps = {**default_cmaps, **(cmaps or {})}
    vlims = dict(vlims or {})

    r_edges = np.asarray(r_edges, dtype=float)
    v_edges_deg = np.asarray(v_edges_deg, dtype=float)
    rmax = float(r_edges[-1])

    parsed = {}
    for bin_name in bin_values:
        try:
            parsed[bin_name] = parse_bin_name(bin_name)
        except ValueError:
            continue  # skip anything that isn't a real bin/bnn stem

    fig, axes = plt.subplots(1, len(moments), figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes)

    for ax, moment in zip(axes, moments):
        vals = [bin_values[b].get(moment) for b in parsed]
        vals = [v for v in vals if v is not None]
        if moment in vlims:
            vmin, vmax = vlims[moment]
        elif vals:
            amax = max(abs(v) for v in vals)
            vmin, vmax = (0.0, amax) if moment == "sigma" else (-amax, amax)
        else:
            vmin, vmax = (0.0, 1.0)

        cmap = plt.get_cmap(cmaps.get(moment, "viridis"))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for bin_name, (irc, irv, side) in parsed.items():
            value = bin_values[bin_name].get(moment)
            if value is None or irc < 1 or irc > len(r_edges) - 1 or irv < 1 or irv > len(v_edges_deg) - 1:
                continue
            r_lo, r_hi = r_edges[irc - 1], r_edges[irc]
            t_lo, t_hi = v_edges_deg[irv - 1], v_edges_deg[irv]
            color = cmap(norm(value))
            for theta1, theta2 in _bin_wedge_angles(t_lo, t_hi, side):
                patch = _wedge_polygon(r_lo, r_hi, theta1, theta2)
                patch.set_facecolor(color)
                patch.set_edgecolor("none")
                ax.add_patch(patch)

        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_aspect("equal")
        ax.set_xlabel("major axis [arcsec]")
        ax.set_ylabel("minor axis [arcsec]")
        ax.set_title(moment)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path)
        plt.close(fig)
    return fig
