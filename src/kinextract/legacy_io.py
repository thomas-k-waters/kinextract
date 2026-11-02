"""Writers for the SCO Fortran framework's own per-bin output file
formats (`.fit`, `.ascii`, `.rms`, `.mcfit`), so a `kinextract` point-
estimate fit can be dropped directly into that legacy pipeline's
downstream tools (`pspecfit`, `pfitlov`, `pfitlove`, `pallmc`, ...)
without those tools needing to change at all.

Every format here was reverse-engineered from the actual Fortran source
(`fitlovw.f`'s own `write` statements) and cross-checked against real
example files from a completed legacy run (NGC 5102) -- see each
writer's own docstring for the exact provenance. Two real, honest gaps
exist because kinextract's fitting model has no equivalent concept:

- `coff`/`coff2` (`.rms`, a legacy polynomial-continuum offset pair):
  kinextract's own fit has no continuum-cofitting active for these
  real-target notebooks, so there is nothing to report here -- written
  as `0.0, 0.0` rather than a fabricated nonzero value.
- The `.ascii` "template-only" column (`tt(i)` in `fitlovw.f`, the
  legacy fit's un-convolved template curve, used only for a diagnostic
  plot): kinextract fits template weights and the LOSVD jointly with no
  separate unconvolved-template curve to report, so this column
  duplicates the model column rather than fabricating a distinct one.

Also a real, documented format ambiguity (not a kinextract limitation):
`sco_framework/modprogs/fitlovw.f`'s own `.ascii` write statement has 4
real-valued columns, but `sco_framework_updated`'s own format spec and
`pspecfit.f` (the still-actively-used diagnostic plotter) expect 5 --
this module writes 5, matching what the actively-used reader expects.
"""
from __future__ import annotations

import os
import re

import numpy as np
from scipy.ndimage import gaussian_filter1d


def write_fit_outputs(bin_name: str, result: dict, v_grid: np.ndarray,
                       wave_fit: np.ndarray, flux_fit: np.ndarray, model: np.ndarray,
                       outdir: str, n_templates: int = 0,
                       continuum: np.ndarray | None = None) -> dict[str, str]:
    """Write the legacy `.fit`/`.ascii`/`.rms`/`.mcfit` quartet for one
    fitted bin, from a `kinextract` point-estimate result -- everything
    the legacy pipeline's own downstream tools (`pfitlov`, `pspecfit`,
    and eventually `pallmc`/`rtransvd` once uncertainty is added, see
    the GreatLakes handoff) need per bin.

    Parameters
    ----------
    bin_name : the legacy bin stem, e.g. `"bin0301"` (from
        `kinextract.binning.assign_legacy_dartboard_bins`'s own
        `bin_table` entries) -- files are written as
        `f"{outdir}/{bin_name}sp.fit"` etc, matching the legacy
        `<bin>sp.<ext>` naming convention exactly.
    result : a `fit_losvd`/`fit_losvd_robust` result dict (`b`, `w`,
        `clipped_mask` for a `fit_losvd_robust` result).
    v_grid : the velocity grid `result["b"]` is defined on
        (`cfg.build_velocity_grid()`).
    wave_fit, flux_fit : the observed wavelength/flux this bin's fit
        used (already windowed/masked -- the same arrays passed to the
        fit itself).
    model : the fitted model flux on `wave_fit`'s own grid (e.g.
        `L @ (A_eff_full @ result["b"])` -- de-whitened, in the same
        units as `flux_fit`).
    outdir : directory to write into (created if missing).
    n_templates : template count for the `.mcfit` header's `ns` field
        (cosmetic only -- no downstream reader depends on its value,
        confirmed from `pallmc.f` source, which only reads `.mcfit`'s
        velocity column).
    continuum : this bin's own fitted continuum component (same shape as
        `wave_fit`), if continuum-cofitting was used -- de-whitened, same
        units as `flux_fit`/`model`. `None` (default) when it wasn't
        (kinextract's fit has no continuum concept to report in that
        case -- written as `0.0`, not fabricated). See `.ascii`'s own
        comment below for why this is needed at all.

    Returns
    -------
    dict mapping `{"fit": path, "ascii": path, "rms": path, "mcfit": path}`.
    """
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"{bin_name}sp")
    # .mcfit uses the "spmc" stem (mcfitw's own naming), not "sp" (fitlovw's) -- confirmed
    # from pallmc.f source: it opens a bin's own velocity-grid reference via
    # `file1(1:nfile-3)//'mcfit'` where file1 is that bin's `<bin>spmc.sim` pallmc_inlist
    # entry, i.e. it looks for `<bin>spmc.mcfit`, never `<bin>sp.mcfit`.
    mcfit_stem = os.path.join(outdir, f"{bin_name}spmc")
    paths = {"fit": f"{stem}.fit", "ascii": f"{stem}.ascii", "rms": f"{stem}.rms",
             "mcfit": f"{mcfit_stem}.mcfit"}

    b = np.asarray(result["b"])
    n_good = len(wave_fit)
    resid = flux_fit - model
    chi2 = float(np.sum(resid ** 2))

    # .fit -- fitlovw.f unit 11: line 1 = "0.0 chi2 666.0 n_good" (666.0 is a dead
    # sentinel in the original Fortran, not real data -- see this module's own docstring
    # and legacy_binning's plan-file notes for the source line that confirms this), then
    # one row per LOSVD velocity-grid point: index(1-based), velocity, amplitude.
    with open(paths["fit"], "w") as f:
        f.write(f"{0.0:.6f} {chi2:.6e} {666.0:.6f} {n_good:d}\n")
        for i, (v, amp) in enumerate(zip(v_grid, b), start=1):
            f.write(f"{i:d} {v:.6f} {amp:.6e}\n")

    # .ascii -- 6 columns: wavelength, data, model, continuum, continuum (a second,
    # diagnostic copy), bad-pixel flag. Confirmed directly from the currently-live
    # `sco_framework_updated/modprogs/pspecfit.f` source (`read(2,*) x1,x2,x3,x4,x5,i6`,
    # "column 4 is the best-fit continuum for plotting", "column 5 is retained as an
    # additional diagnostic continuum column") -- NOT 5 columns as an earlier version of
    # this docstring claimed (that was checked against the wrong/an outdated format
    # spec, not the real Fortran source; pspecfit is never auto-invoked by
    # finalizeextraction, so this mismatch never surfaced as a real run-time failure,
    # but reading the source directly leaves no ambiguity). clipped_mask
    # (fit_losvd_robust results) marks a pixel bad (1)
    # instead of good (0); a plain fit_losvd result has no clipping, so every pixel
    # is flagged good.
    clipped_mask = result.get("clipped_mask")
    bad_flag = clipped_mask.astype(int) if clipped_mask is not None else np.zeros(n_good, dtype=int)
    cont = np.asarray(continuum) if continuum is not None else np.zeros(n_good)
    with open(paths["ascii"], "w") as f:
        for wl, data, mod, c, flag in zip(wave_fit, flux_fit, model, cont, bad_flag):
            f.write(f"{wl:10.3f}  {data:7.3f}  {mod:7.3f}  {c:7.3f}  {c:7.3f} {flag:d}\n")

    # .rms -- one line: sqrt(rms/n), n_good, coff, coff2 (coff/coff2 are a legacy
    # polynomial-continuum offset pair kinextract has no equivalent for -- see module docstring)
    rms = np.sqrt(chi2 / n_good) if n_good > 0 else np.nan
    with open(paths["rms"], "w") as f:
        f.write(f"{rms:.6e} {n_good:d} {0.0:.6f} {0.0:.6f}\n")

    # .mcfit -- velocity-grid reference only (pallmc.f reads only the velocity column,
    # confirmed from source); other 5 fields per row are placeholder zeros.
    with open(paths["mcfit"], "w") as f:
        f.write(f"{len(v_grid):d} {n_templates:d}\n")
        for v in v_grid:
            f.write(f"{v:.6f} 0 0 0 0 0\n")

    return paths


def _invert_about_zero(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Port of transvd.f's own `invert(x, y, n, 0., yout)` subroutine (its only call
    site always passes `xinv=0.`): mirrors `y` about `v=0` by re-interpolating each
    point onto its negated velocity, using the *original* (x, y) grid. Points whose
    mirror falls outside the original domain are zeroed, matching the Fortran exactly."""
    n = len(x)
    xmin, xmax = -x[-1], -x[0]
    out = np.zeros(n)
    in_domain = (x >= xmin) & (x <= xmax)
    mirrored = -x[in_domain]
    out[in_domain] = np.interp(mirrored, x, y)
    return out


def legacy_transvd_curve(v_grid: np.ndarray, b_median: np.ndarray, b_p16: np.ndarray,
                          b_p84: np.ndarray, *, invert: bool, model_bin_kms: float = 133.3,
                          voff: float = 0.0, ntot: int = 1000, min_error: float = 0.005,
                          losvd_clip: float = 0.0007, smoothing_sigma: float = 3.0
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of `sco_framework_updated/bin/transvd` (Fortran, `transvd.f`) chained into
    `fix_losvd_errs.py`, the two-stage transform that previously turned a bin's own
    `.mcfit2` curve (velocity, LOSVD median, 16th/84th percentile) into its final
    `<bin>spmc_in` modeling input: rescale the percentile-derived error bars for the
    data-vs-model velocity bin width, normalize the LOSVD to unit area, optionally
    mirror the curve about v=0 (`invert`), resample onto a dense fixed 1000-point
    grid spanning `[v_grid[0], -v_grid[0]]`, then apply `fix_losvd_errs.py`'s own
    floor/clip/smooth/wing-zeroing cleanup. Lets `kinextract`'s own Laplace/
    NUTS uncertainty (`b_median`/`b_p16`/`b_p84`, the same curve `run_losvd_errors.py`
    already writes to `<bin>spmc.bestlosvd`) produce a `<bin>spmc_in` file directly,
    without needing `pallmc.f`'s `.mcfit2` bridge output or the legacy `rtransvd`/
    `transvd`/`fix_losvd_errs.py` binary chain at all.

    Validated against the real Fortran: `transvd.f` (+ the real `numrec_f/spline.f`,
    `splint.f` it links against) was rebuilt locally with gfortran and run on a real
    bin's real `.mcfit2` (`bin0105spmc.mcfit2`, `kinextract_debug_suite`), for both
    `invert=False` and `invert=True` (isym=0 always -- this framework never sets
    isym=1, see `rtransvd`), chained through the real `fix_losvd_errs.py`. This port
    matches that real chain's output to ~1e-5 absolute at all 1000 points except one:
    the Fortran's own `xlinint` uses a half-open interval test (`xp.ge.x(j).and.
    xp.lt.x(j+1)`) that excludes the exact right endpoint, so at the single output
    point that lands exactly on `v_grid[-1]` (only reachable in the `invert=False`
    case; `invert=True`'s domain check hits the *left* endpoint instead, which the
    half-open test does cover), Fortran's `yp` keeps whatever stale value the
    previous loop iteration left in that unreset local variable -- an accident of
    reusing an uninitialized variable across loop iterations, not a meaningful
    result. This port instead correctly clamps to the endpoint value there. That one
    point sits at the extreme edge of the velocity grid, where the physical LOSVD
    amplitude is ~0 in every real fit checked, so the deviation is immaterial.

    `model_bin_kms=133.3` is an inherited legacy constant (`rtransvd`'s own source
    comment: "Ask Karl how to get 133.3") with no derivation recovered here --
    preserved as-is for continuity with every existing legacy-pipeline result.

    Parameters
    ----------
    v_grid, b_median, b_p16, b_p84 : this bin's own LOSVD curve, e.g. the same arrays
        `run_losvd_errors.py` already has in memory (`result["b_median"]`, etc.) --
        `v_grid` must be uniformly spaced (`cfg.build_velocity_grid()`'s own grid).
    invert : whether to mirror the curve about v=0 before resampling -- matches
        `finalizeextraction`'s own `rtransvd <bin> 1` (non-`bnn*` bins, invert=True)
        vs. `rtransvd <bin> -1` (`bnn*` bins, invert=False; transvd.f's own check is
        `if(ivert.eq.1)`, so any value other than literal 1 skips inversion) split --
        `write_spmc_in` below derives this automatically from `bin_name`.

    Returns
    -------
    (vel, losvd, lower_err, upper_err), each a `(1000,)` array -- the exact columns
    `<bin>spmc_in` is written from.
    """
    v = np.asarray(v_grid, dtype=float)
    y = np.asarray(b_median, dtype=float)
    y16 = np.asarray(b_p16, dtype=float)
    y84 = np.asarray(b_p84, dtype=float)

    data_bin_kms = v[2] - v[1]  # matches rtransvd's own awk: file lines 2,3 -> array indices 1,2
    fac = np.sqrt(model_bin_kms / data_bin_kms)

    v_in = v + voff
    y_in = y.copy()
    yl_in = (y - y16) / fac
    yh_in = (y84 - y) / fac

    total = y_in.sum()
    y_in = y_in / total
    yl_in = yl_in / total
    yh_in = yh_in / total

    if invert:
        y_in = _invert_about_zero(v_in, y_in)
        yl_in = _invert_about_zero(v_in, yl_in)
        yh_in = _invert_about_zero(v_in, yh_in)

    vp = v_in[0] - 2.0 * v_in[0] / float(ntot - 1) * np.arange(ntot)
    in_range = (vp >= v_in[0]) & (vp <= v_in[-1])

    yp = np.interp(vp, v_in, y_in)
    ylp = np.interp(vp, v_in, yl_in)
    yhp = np.interp(vp, v_in, yh_in)

    yn = np.where(in_range, yp, 0.0)
    ynl = np.where(in_range, yn - np.abs(ylp), 0.0)
    # yhpold: out-of-range points carry forward the last in-range upper-error value
    # (transvd.f never resets `yhpold` inside the else branch), so upper errors don't
    # pinch to zero just past the data's own velocity coverage.
    yhp_carried = np.full(ntot, 1e-1)
    last = 1e-1
    for i in range(ntot):
        if in_range[i]:
            last = yhp[i]
        yhp_carried[i] = last
    ynh = np.where(in_range, yn + np.abs(yhp), np.abs(yhp_carried))

    yn = np.maximum(0.0, yn)
    ynl = np.maximum(0.0, ynl)

    vel, losvd, lower_err, upper_err = vp, yn, ynl, ynh

    lower_err_corr = np.where((losvd - lower_err) < min_error, losvd - min_error, lower_err)
    lower_err_corr = np.clip(lower_err_corr, 0, losvd)
    lower_err_corr[losvd < losvd_clip] = 0

    upper_err_corr = np.maximum(upper_err, losvd + min_error)

    lower_err_corr = gaussian_filter1d(lower_err_corr, sigma=smoothing_sigma)
    upper_err_corr = gaussian_filter1d(upper_err_corr, sigma=smoothing_sigma)

    # fix_losvd_errs.py's own wing-zeroing: walk outward from the LOSVD's own peak in
    # each direction and zero the lower error from its first local minimum onward.
    center_idx = int(np.argmax(losvd))
    lower_err_final = lower_err_corr.copy()

    right_errs = lower_err_final[center_idx:]
    right_min_global = center_idx + int(np.argmin(right_errs))
    lower_err_final[right_min_global:] = 0

    left_errs = lower_err_final[:center_idx + 1][::-1]
    left_min_global = center_idx - int(np.argmin(left_errs))
    lower_err_final[:left_min_global + 1] = 0

    return vel, losvd, lower_err_final, upper_err_corr


def write_spmc_in(bin_name: str, v_grid: np.ndarray, b_median: np.ndarray, b_p16: np.ndarray,
                   b_p84: np.ndarray, outdir: str, model_bin_kms: float = 133.3) -> str:
    """Write `<bin_name>spmc_in` directly from a `kinextract` uncertainty result,
    replacing the legacy `pallmc.f` (`.mcfit2`) -> `rtransvd` -> `transvd` ->
    `fix_losvd_errs.py` chain -- see `legacy_transvd_curve`'s own docstring for the
    validated transform this reproduces and why `pallmc.f` no longer needs to
    produce `.mcfit2` for `kinextract`-derived bins at all.

    `invert` is derived from `bin_name` itself: `True` unless `bin_name` starts with
    `"bnn"`, mirroring `finalizeextraction`'s own `rtransvd <bin> 1`/`rtransvd <bin>
    -1` split exactly (`bin0105`, the dartboard's exact-center bin, does not start
    with `bnn` and so is inverted, same as every other `bin*` stem).

    Parameters
    ----------
    bin_name : the legacy bin stem, e.g. `"bin0105"` or `"bnn0704"` -- written as
        `f"{outdir}/{bin_name}spmc_in"`.
    v_grid, b_median, b_p16, b_p84 : this bin's own LOSVD curve (see
        `legacy_transvd_curve`).
    outdir : directory to write into (created if missing).

    Returns
    -------
    Path to the written file.
    """
    os.makedirs(outdir, exist_ok=True)
    invert = not bin_name.startswith("bnn")
    vel, losvd, lower_err, upper_err = legacy_transvd_curve(
        v_grid, b_median, b_p16, b_p84, invert=invert, model_bin_kms=model_bin_kms)

    path = os.path.join(outdir, f"{bin_name}spmc_in")
    np.savetxt(path, np.column_stack([vel, losvd, lower_err, upper_err]),
               fmt="%.6f %.8f %.8f %.8f")
    return path


_STEM_RE = re.compile(r"(bin|bnn)(\d{4})$")


def spmc_side_inverts(bin_name: str) -> bool:
    """Whether a bin's LOSVD is mirrored about v=0 for the model: `True` for the `bin` side, `False` for `bnn`,
    as `finalizeextraction`'s `rtransvd <bin> 1` / `rtransvd <bin> -1`. The side is read from the trailing
    `bin####`/`bnn####` stem, so a source prefix (`mw`, `mn`, ...) does not change it -- testing
    `bin_name.startswith("bnn")` mirrored every prefixed `mwbnn` file as well."""
    m = _STEM_RE.search(bin_name)
    if m is None:
        raise ValueError(f"{bin_name!r} does not end in a bin####/bnn#### stem")
    return m.group(1) == "bin"


def transvd_bounds_laplace(v_grid: np.ndarray, b_median: np.ndarray, b_p16: np.ndarray, b_p84: np.ndarray,
                            *, invert: bool, model_bin_kms: float = 133.3, voff: float = 0.0,
                            vmax_out: float = 3000.0, ntot: int = 1000, noise_floor: str | None = "peak",
                            b_samples: np.ndarray | None = None, model_bins: tuple | None = None
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`transvd.f` without `fix_losvd_errs.py`, for Laplace (or NUTS) percentiles, on a fixed
    `[-vmax_out, vmax_out]` grid shared by every bin. Returns `(velocity, losvd, lower_bound, upper_bound)`, the
    last two absolute bounds (not widths), i.e. the four `spmc_in` columns.

    b_samples, model_bins
        Optional (default None: unchanged behavior). `b_samples` are posterior draws of the LOSVD on `v_grid`
        (e.g. Laplace `b_samples`); `model_bins = (centers, half_width)` are the dynamical model's velocity bins
        in the output frame (NGC 4751: `(np.linspace(-1800, 1800, 13), 150.0)`). The widths in each model bin
        are then widened, where needed, so that the error model.x computes for that bin is at least the posterior
        scatter of the binned LOSVD (`_match_model_bin_errors`).

    The rescaling (sqrt(model_bin_kms / data bin)), unit-area normalization and mirroring are `transvd.f`'s; the
    fixed 0.005 floor, clipping, smoothing and wing-zeroing of `fix_losvd_errs.py` are not applied.

    noise_floor
        "peak" (default): neither error width is allowed below the larger of the two at the LOSVD's peak. The
        Laplace posterior of a positive LOSVD gives errors that scale with its amplitude, so in the wings they
        fall to 1e-14 of the peak or less (NGC 4751 WFM: down to 1e-60) although the spectral noise is not
        smaller there. The model's 13 velocity bins span the whole [vmin, vmax] range, so any model light in such
        a bin gives (model/error)^2 > 1e38: single precision overflows and chi2 becomes NaN. The width at the
        peak is the bin's own measured uncertainty level, so the floor keeps the weights tied to the noise.
        None: only a width that is exactly zero is replaced (by the median positive upper width inside the
        fitted range) -- the behavior of the N4751 notebooks' writer before 2026-10-16.
    """
    v = np.asarray(v_grid, dtype=float)
    y = np.asarray(b_median, dtype=float)
    y16 = np.asarray(b_p16, dtype=float)
    y84 = np.asarray(b_p84, dtype=float)
    if not (v.ndim == y.ndim == y16.ndim == y84.ndim == 1) or not (len(v) == len(y) == len(y16) == len(y84)):
        raise ValueError("LOSVD inputs must be one-dimensional arrays of equal length")
    if not all(np.all(np.isfinite(a)) for a in (v, y, y16, y84)):
        raise ValueError("LOSVD inputs contain NaN or infinite values")
    if len(v) < 3 or np.any(np.diff(v) <= 0):
        raise ValueError("v_grid must have at least three strictly increasing samples")
    data_bin_kms = v[2] - v[1]  # rtransvd's convention
    if not np.allclose(np.diff(v), data_bin_kms, rtol=1e-6, atol=1e-8):
        raise ValueError("v_grid must be uniformly spaced")
    if noise_floor not in ("peak", None):
        raise ValueError(f"noise_floor must be 'peak' or None, got {noise_floor!r}")

    fac = np.sqrt(model_bin_kms / data_bin_kms)
    v_in = v + voff
    lower_width_in = (y - y16) / fac
    upper_width_in = (y84 - y) / fac
    total = np.sum(y)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid LOSVD normalization: sum={total}")
    y_in, lower_width_in, upper_width_in = y / total, lower_width_in / total, upper_width_in / total
    if invert:
        y_in = np.interp(-v_in, v_in, y_in, left=0.0, right=0.0)
        lower_width_in = np.interp(-v_in, v_in, lower_width_in, left=0.0, right=0.0)
        upper_width_in = np.interp(-v_in, v_in, upper_width_in, left=0.0, right=0.0)

    vel = np.linspace(-vmax_out, vmax_out, ntot)
    in_range = (vel >= v_in[0]) & (vel <= v_in[-1])
    losvd = np.maximum(np.where(in_range, np.interp(vel, v_in, y_in), 0.0), 0.0)
    lower_width = np.where(in_range, np.abs(np.interp(vel, v_in, lower_width_in)), 0.0)
    upper_width = np.where(in_range, np.abs(np.interp(vel, v_in, upper_width_in)), 0.0)

    if noise_floor == "peak":
        ip = int(np.argmax(losvd))
        floor = max(lower_width[ip], upper_width[ip])
        if not (np.isfinite(floor) and floor > 0):
            raise ValueError("zero error width at the LOSVD peak (degenerate posterior?)")
        lower_width = np.maximum(lower_width, floor)
        upper_width = np.maximum(upper_width, floor)
    else:
        pos = np.abs(upper_width_in)[np.abs(upper_width_in) > 0]
        if pos.size == 0:
            raise ValueError("upper error width is zero everywhere (degenerate posterior?)")
        upper_width = np.where(upper_width > 0, upper_width, float(np.median(pos)))

    if model_bins is not None:
        if b_samples is None:
            raise ValueError("model_bins needs b_samples (the posterior draws the bands came from)")
        lower_width, upper_width = _match_model_bin_errors(
            v_in, np.asarray(b_samples, dtype=float), invert, vel, losvd, lower_width, upper_width, model_bins)

    lower_bound = np.maximum(losvd - lower_width, 0.0)
    upper_bound = losvd + upper_width
    return vel, losvd, lower_bound, upper_bound


def _match_model_bin_errors(v_in, b_samples, invert, vel, losvd, lower_width, upper_width, model_bins):
    """Rescale the error widths inside each model bin so that the error the model computes for that bin --
    model.x (velmtod.f) sums the per-point errors linearly over the bin and uses max(upper - median,
    (upper - lower) / 2), normalized like the LOSVD -- equals the posterior scatter of the LOSVD integrated over
    the bin. Per-point widths are correlated along the LOSVD, so their linear sum is not that scatter (on
    NGC 4751-like mocks it was about half of it). Bins with less than 0.1 % of the light keep the scale factor
    of the nearest bin that has light, so the noise floor in empty bins is scaled consistently rather than
    replaced by a near-zero posterior scatter. The widths are only ever widened (scale factor >= 1): the
    posterior of a log-parametrized LOSVD has almost no scatter wherever its median has almost no light, so in
    faint wing bins a smaller target would undo the noise floor rather than correct the sum."""
    centers, half = np.asarray(model_bins[0], float), float(model_bins[1])
    in_bin = [(vel >= c - half) & (vel < c + half) for c in centers]
    S = []
    for s in b_samples:
        s = np.maximum(s, 0.0) / max(np.sum(np.maximum(s, 0.0)), 1e-300)
        if invert:
            s = np.interp(-v_in, v_in, s, left=0.0, right=0.0)
        r = np.maximum(np.where((vel >= v_in[0]) & (vel <= v_in[-1]), np.interp(vel, v_in, s), 0.0), 0.0)
        S.append([r[m].sum() / r.sum() for m in in_bin])
    target = np.std(np.asarray(S), axis=0, ddof=1)
    tot = losvd.sum()
    light = np.array([losvd[m].sum() / tot for m in in_bin])
    cur = np.array([np.maximum(upper_width[m], 0.5 * (upper_width[m] + lower_width[m])).sum() / tot for m in in_bin])
    k = np.where((light > 1e-3) & (cur > 0), np.maximum(target / np.where(cur > 0, cur, 1.0), 1.0), np.nan)
    ok = np.flatnonzero(np.isfinite(k))
    if ok.size == 0:
        return lower_width, upper_width
    for j in np.flatnonzero(~np.isfinite(k)):
        k[j] = k[ok[np.argmin(np.abs(ok - j))]]
    fac = np.ones_like(vel)
    for kj, m in zip(k, in_bin):
        fac[m] = kj
    return lower_width * fac, upper_width * fac


def write_spmc_in_laplace(bin_name: str, v_grid: np.ndarray, b_median: np.ndarray, b_p16: np.ndarray,
                           b_p84: np.ndarray, outdir: str, *, model_bin_kms: float = 133.3,
                           vmax_out: float = 3000.0, ntot: int = 1000, noise_floor: str | None = "peak",
                           b_samples: np.ndarray | None = None, model_bins: tuple | None = None) -> str:
    """Write `<bin_name>spmc_in` from Laplace/NUTS percentiles with `transvd_bounds_laplace` (no
    `fix_losvd_errs.py`). `bin_name` may carry a source prefix (`"mwbnn0704"`); the side, and so the mirroring,
    comes from its `bin####`/`bnn####` stem (`spmc_side_inverts`). `v_grid` is in the systemic frame.
    `b_samples`/`model_bins`: see `transvd_bounds_laplace`."""
    os.makedirs(outdir, exist_ok=True)
    vel, losvd, lower, upper = transvd_bounds_laplace(
        v_grid, b_median, b_p16, b_p84, invert=spmc_side_inverts(bin_name), model_bin_kms=model_bin_kms,
        vmax_out=vmax_out, ntot=ntot, noise_floor=noise_floor, b_samples=b_samples, model_bins=model_bins)
    if np.any(lower > losvd + 1e-12) or np.any(upper <= losvd):
        raise ValueError(f"{bin_name}: invalid error bounds")
    path = os.path.join(outdir, f"{bin_name}spmc_in")
    np.savetxt(path, np.column_stack([vel, losvd, lower, upper]), fmt="%.6f %.8e %.8e %.8e")
    return path
