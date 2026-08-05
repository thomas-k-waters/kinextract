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

import numpy as np


def write_fit_outputs(bin_name: str, result: dict, v_grid: np.ndarray,
                       wave_fit: np.ndarray, flux_fit: np.ndarray, model: np.ndarray,
                       outdir: str, n_templates: int = 0) -> dict[str, str]:
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
    result : a `fit_losvd`/`fit_losvd_clean` result dict (`b`, `w`,
        `clipped_mask` if from the `_clean` variant).
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

    # .ascii -- format 1301, 5 columns: wavelength, data, model, template-only, bad-pixel
    # flag. clipped_mask (fit_losvd_clean only) marks a pixel bad (1) instead of good (0);
    # a plain fit_losvd result has no clipping, so every pixel is flagged good.
    clipped_mask = result.get("clipped_mask")
    bad_flag = clipped_mask.astype(int) if clipped_mask is not None else np.zeros(n_good, dtype=int)
    with open(paths["ascii"], "w") as f:
        for wl, data, mod, flag in zip(wave_fit, flux_fit, model, bad_flag):
            f.write(f"{wl:10.3f}  {data:7.3f}  {mod:7.3f}  {mod:7.3f} {flag:d}\n")

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
