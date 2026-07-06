"""Legacy-format spectral I/O for kinextract.

This module handles all reading and writing of the plain-text file formats
inherited from the original Fortran pipeline (Gebhardt & Waters): raw
index/wavelength-flux-error spectrum files, pre-normalized ``.norm``
spectra, stellar template files, ``galaxy.params`` metadata files, and the
``.fit``/``.temp``/``.ascii``/``.rms`` output files consumed by downstream
Schwarzschild modeling and plotting tools (e.g. pspecfit). It also provides
small wavelength-grid utilities and the Fortran-compatible ``NINT``
rounding function used to keep pixel-index arithmetic bit-for-bit
consistent with the legacy code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ._utils import BIG, log

# =============================================================================
# Section 4 - I/O helpers
# =============================================================================

def read_galaxy_index_flux_err(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a raw 3-column galaxy spectrum file.

    Parameters
    ----------
    path : str
        Path to a whitespace-delimited text file with at least 3 columns:
        pixel index or wavelength, flux, and flux error (or variance).

    Returns
    -------
    col0 : ndarray
        First column: pixel index or wavelength, depending on file
        convention (converted to a wavelength grid downstream via
        `build_wavelength_from_index` when it is a pixel index).
    flux : ndarray
        Second column: observed flux.
    flux_err : ndarray
        Third column: flux error or variance, depending on file convention.
    """
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"{path}: expected >= 3 columns (index-or-wave, flux, flux_err_or_variance)"
        )
    return arr[:, 0], arr[:, 1], arr[:, 2]


def read_norm_spectrum(path: str) -> dict:
    """Read a pre-normalized ``.norm`` spectrum file.

    ``.norm`` files store a continuum-normalized spectrum (flux divided by
    an independently-derived continuum estimate) alongside the original
    un-normalized flux and continuum, so the LOSVD fit can skip the
    ALS-continuum co-fit and instead work directly with normalized flux
    (``prenormalized=True`` in :class:`kinextract.state.FitState`).

    Parameters
    ----------
    path : str
        Path to the ``.norm`` file. Must have at least 3 whitespace-delimited
        columns; a 7-column extended format additionally carries the
        original flux/error and continuum/error.

    Returns
    -------
    dict
        Dictionary with keys ``"wavelength"``, ``"normflux"``,
        ``"normflux_err"`` (always present), and ``"orig_flux"``,
        ``"orig_flux_err"``, ``"continuum"``, ``"continuum_err"`` (present
        only when the file has 7+ columns; otherwise `None`).
    """
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path}: .norm file must have at least 3 columns")
    keys = ["wavelength", "normflux", "normflux_err",
            "orig_flux", "orig_flux_err", "continuum", "continuum_err"]
    data: dict = {k: None for k in keys}
    for i, k in enumerate(keys[:3]):
        data[k] = arr[:, i].astype(float)
    if arr.shape[1] >= 7:
        for i, k in enumerate(keys[3:], 3):
            data[k] = arr[:, i].astype(float)
    return data


def read_template_xy(path: str) -> tuple[np.ndarray, np.ndarray, "np.ndarray | None"]:
    """Load a single stellar/SSP template spectrum file.

    Parameters
    ----------
    path : str
        Path to a whitespace-delimited template file with 2 columns
        (wavelength, flux) or 3 columns (wavelength, flux, flux_err).
        Files may be ``.txt`` (already-normalized flux) or ``.dat``
        (physical flux units, e.g. MUSE spectral library files; these are
        median-normalized downstream by
        :func:`kinextract.templates.build_template_matrix_fortran`).

    Returns
    -------
    wave : ndarray
        Template wavelength grid, in Angstroms.
    flux : ndarray
        Template flux values.
    flux_err : ndarray or None
        Per-pixel template flux uncertainty, or `None` if the file has no
        third column.
    """
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Template {path} must have >= 2 columns")
    wave = arr[:, 0].astype(float)
    flux = arr[:, 1].astype(float)
    err = arr[:, 2].astype(float) if arr.shape[1] >= 3 else None
    return wave, flux, err


def read_template_list(list_file: str, template_dir: Optional[str] = None) -> list[str]:
    """Read a template-list file and resolve each entry to a full path.

    Parameters
    ----------
    list_file : str
        Path to a text file listing one template filename per line (blank
        lines and lines starting with ``#`` are ignored). This is the
        legacy "Tlist" file used to select which stellar/SSP templates
        participate in the fit.
    template_dir : str, optional
        Directory used to resolve relative filenames listed in `list_file`.
        Defaults to the directory containing `list_file` itself.

    Returns
    -------
    list of str
        Resolved (absolute or as-given) paths to each template file, in the
        order listed, suitable for passing to
        :func:`kinextract.templates.build_template_matrix_fortran`.
    """
    list_file = Path(list_file)
    base = Path(template_dir) if template_dir else list_file.parent
    files = []
    with open(list_file) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s)
            files.append(str(p if p.is_absolute() else base / p))
    if not files:
        raise ValueError(f"No templates found in {list_file}")
    return files


def infer_output_prefix(gal_file: str) -> str:
    """Derive the output file prefix from a galaxy spectrum filename.

    Strips a recognized spectrum-file suffix (``.norm``, ``.spec``,
    ``.txt``, ``.dat``) from the filename, falling back to the plain file
    stem otherwise. The resulting prefix is used to name the
    ``.fit``/``.temp``/``.ascii``/``.rms`` output files written by
    :func:`write_fitlov_outputs` so they can be matched back to their
    input spectrum.

    Parameters
    ----------
    gal_file : str
        Path to the input galaxy spectrum file.

    Returns
    -------
    str
        Output prefix (filename without its spectrum-file suffix).
    """
    p = Path(gal_file)
    for suf in [".norm", ".spec", ".txt", ".dat"]:
        if p.name.endswith(suf):
            return p.name[: -len(suf)]
    return p.stem


def nint_fortran(z: np.ndarray) -> np.ndarray:
    """Round to nearest integer using Fortran's NINT convention.

    Implements round-half-away-from-zero (e.g. 2.5 -> 3, -2.5 -> -3),
    which differs from NumPy's default round-half-to-even. This exact
    convention is required wherever pixel indices are computed (e.g. the
    LOSVD convolution index map in
    :func:`kinextract.state.precompute_ip_map`) so that outputs are
    bit-for-bit reproducible against the legacy Fortran pipeline that
    downstream Schwarzschild modeling code expects.

    Parameters
    ----------
    z : ndarray
        Array of values to round.

    Returns
    -------
    ndarray of int
        `z` rounded to the nearest integer, ties rounded away from zero.
    """
    z = np.asarray(z)
    return np.where(z >= 0, np.floor(z + 0.5), np.ceil(z - 0.5)).astype(int)


def setbadreg(x: np.ndarray, gerr: np.ndarray, iflip: int, regions_bad_path: str) -> np.ndarray:
    """Mask pixels inside user-specified bad wavelength regions.

    Sets `gerr` to the `BIG` sentinel for every pixel whose wavelength `x`
    falls within any ``[x1, x2]`` interval listed in the bad-region file,
    effectively removing those pixels from the chi-squared fit (e.g. to
    exclude sky-line residuals, cosmic rays, or other known-bad ranges).

    Parameters
    ----------
    x : ndarray
        Pixel wavelengths (or velocities), in the same convention as the
        bad-region file bounds.
    gerr : ndarray
        Per-pixel flux error array to modify in place (and return).
    iflip : int
        Sign multiplier applied to each region's bounds before comparison
        (+1 or -1), used to check a bad-region list defined in the opposite
        wavelength convention (e.g. observed vs. rest frame).
    regions_bad_path : str
        Path to a text file of two-column ``x1 x2`` bad-region bounds. If
        empty or the file does not exist, `gerr` is returned unchanged.

    Returns
    -------
    ndarray
        `gerr` with `BIG` inserted at masked pixels.
    """
    if not regions_bad_path:
        return gerr
    p = Path(regions_bad_path)
    if not p.exists() or not p.is_file():
        return gerr
    for x1, x2 in np.atleast_2d(np.loadtxt(p)):
        m = (x >= iflip * x1) & (x <= iflip * x2)
        gerr[m] = BIG
    return gerr


def build_wavelength_from_index(npts: int, wavemin_full: float, step: float) -> np.ndarray:
    """Build a uniform wavelength grid from a starting value and step.

    Parameters
    ----------
    npts : int
        Number of pixels (grid points) to generate.
    wavemin_full : float
        Wavelength of the first pixel, in Angstroms.
    step : float
        Wavelength spacing per pixel, in Angstroms.

    Returns
    -------
    ndarray
        Wavelength grid ``wavemin_full + step * arange(npts)``, used when
        the input spectrum is stored by pixel index rather than explicit
        wavelength.
    """
    return wavemin_full + step * np.arange(npts)


def select_region_with_errors(
    x: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
    xmin: float, xmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the open interval ``(xmin, xmax)`` of a spectrum for fitting.

    Parameters
    ----------
    x : ndarray
        Wavelength grid.
    flux : ndarray
        Observed flux, same length as `x`.
    flux_err : ndarray
        Flux error, same length as `x`.
    xmin, xmax : float
        Lower and upper wavelength bounds of the region to keep (exclusive
        on both ends), typically the fit window requested via
        ``FitConfig.wavefitmin``/``wavefitmax``.

    Returns
    -------
    x_sel : ndarray
        Wavelengths within the selected region (copy).
    flux_sel : ndarray
        Flux within the selected region (copy).
    flux_err_sel : ndarray
        Flux error within the selected region (copy).
    mask : ndarray of bool
        Boolean mask (same length as the input arrays) marking the selected
        pixels, so callers can index other associated arrays consistently.
    """
    m = (x > xmin) & (x < xmax)
    return x[m].copy(), flux[m].copy(), flux_err[m].copy(), m


def estimate_step_from_wavelength(x: np.ndarray) -> float:
    """Estimate the (assumed-uniform) wavelength step of a spectral grid.

    Parameters
    ----------
    x : ndarray
        Wavelength grid, ideally uniformly sampled.

    Returns
    -------
    float
        Median pixel-to-pixel wavelength spacing, in the same units as `x`
        (Angstroms). Using the median rather than the mean spacing makes
        the estimate robust to the occasional non-uniform or duplicated
        wavelength value.

    Raises
    ------
    ValueError
        If no finite pixel-to-pixel differences exist, or if the estimated
        step is non-positive (indicating an unsorted or degenerate grid).
    """
    dx = np.diff(np.asarray(x, float))
    dx = dx[np.isfinite(dx)]
    if dx.size == 0:
        raise ValueError("Could not estimate wavelength step: no finite differences.")
    step = float(np.median(dx))
    if step <= 0:
        raise ValueError(f"Non-positive wavelength step estimated: {step}")
    return step


def count_in_window(x: np.ndarray, xmin: float, xmax: float) -> int:
    """Count how many values of `x` fall strictly within ``(xmin, xmax)``.

    Parameters
    ----------
    x : ndarray
        Values to test (typically a wavelength grid).
    xmin, xmax : float
        Open interval bounds.

    Returns
    -------
    int
        Number of elements of `x` satisfying ``xmin < x < xmax``. Used to
        auto-detect which wavelength frame (rest vs. observed) puts more
        pixels inside the requested fit window.
    """
    return int(np.count_nonzero((np.asarray(x) > xmin) & (np.asarray(x) < xmax)))


def read_galaxy_params(path: str | Path) -> dict:
    """
    Read a legacy galaxy.params file.

    Expected format:
        First non-comment line = galaxy name
        Remaining lines = key value [optional comments]

    Example:
        NGC3258
        distanceMpc  31.9
        vmin        -1100.
        vmax         1100.

    Returns
    -------
    params : dict
        Parsed values. Numeric values are converted to int or float when
        possible. The first line is stored as params["galaxy_name"].
    """
    path = Path(path)

    # Accept a directory: look for galaxy.params inside it.
    if path.is_dir():
        path = path / "galaxy.params"

    if not path.exists():
        raise FileNotFoundError(f"galaxy.params file not found: {path}")

    params: dict = {}
    got_name = False

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            # First non-comment line is the galaxy name.
            if not got_name:
                params["galaxy_name"] = line.split()[0]
                got_name = True
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            key = parts[0]
            val_raw = parts[1]

            # Convert value to int/float where possible.
            try:
                val_float = float(val_raw)

                if np.isfinite(val_float) and val_float.is_integer():
                    val = int(val_float)
                else:
                    val = val_float

            except Exception:
                val = val_raw

            params[key] = val

    return params


# =============================================================================
# Section 13 - Output writing (part B)
# =============================================================================

def write_fitlov_outputs(
    st, a_best: np.ndarray,
    outdir: str = ".", prefix: Optional[str] = None,
) -> dict:
    """Write the final LOSVD fit results to legacy-format ASCII files.

    Evaluates the best-fit model (LOSVD amplitudes, template weights, and
    continuum) at the optimizer's solution `a_best` and writes it out using
    the exact file layout expected by the legacy Fortran pipeline's
    downstream tools (Schwarzschild orbit modeling, ``pspecfit`` plotting),
    so this package is a drop-in replacement for the original fitting code:

    - ``{prefix}.fit``   : recovered LOSVD, one row per velocity bin.
    - ``{prefix}.temp``  : fractional weight of each template star/SSP.
    - ``{prefix}.ascii`` : per-pixel wavelength, data, model, continuum,
      template-only spectrum, and a good/bad pixel flag (for plotting).
    - ``{prefix}.rms``   : reduced chi-squared and fit-quality summary.

    Parameters
    ----------
    st : FitState
        Fit state holding the spectrum, template matrix, and LOSVD grid
        used to evaluate the model.
    a_best : ndarray
        Best-fit parameter vector (LOSVD amplitudes, template weights, and
        any continuum-offset parameters) as returned by the optimizer.
    outdir : str, default "."
        Directory to write the output files into; created if it does not
        exist.
    prefix : str, optional
        Filename prefix for the four output files. Defaults to
        ``"fitlov"`` if not given; typically produced by
        :func:`infer_output_prefix`.

    Returns
    -------
    dict
        Dictionary with the evaluated model quantities (``"gp"``, ``"b"``,
        ``"w"``, ``"wfrac"``, ``"tt"``, ``"coff"``, ``"coff2"``, ``"A"``),
        fit-quality statistics (``"chi2_red"``, ``"chi2_total"``,
        ``"nrms"``), the continuum array, and a ``"paths"`` dict giving the
        four output file paths.
    """
    from .numerics import compute_weighted_template_spectrum, evaluate_model_gp

    outdir = Path("." if outdir is None else outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gp, b, w, coff, coff2, A = evaluate_model_gp(a_best, st)
    sw = float(np.sum(w))
    wfrac = w / sw if sw else w
    tt = compute_weighted_template_spectrum(st, w)
    good = st.gerr < 1e9
    ierr = (~good).astype(int)
    cont = getattr(st, "continuum_mult", np.ones(st.npix))

    # Use original naming scheme (compatible with downstream modeling)
    if prefix:
        fit_path    = outdir / f"{prefix}.fit"
        temp_path   = outdir / f"{prefix}.temp"
        ascii_path  = outdir / f"{prefix}.ascii"
        rms_path    = outdir / f"{prefix}.rms"
    else:
        fit_path    = outdir / "fitlov.fit"
        temp_path   = outdir / "fitlov.temp"
        ascii_path  = outdir / "fitlov.ascii"
        rms_path    = outdir / "fitlov.rms"

    # Write .fit file (LOSVD vector)
    # Header: coff, velocity_reference, nlosvd, n_templates
    with open(fit_path, "w") as f:
        f.write(f"{coff:13.8f} {st.xl[0]:16.9g} {st.nl:14d} {st.nt:12d}\n")
        for k in range(st.nl):
            f.write(f"{k + 1:12d} {st.xl[k]:12.6f} {b[k]:17.8E}\n")

    # Write .temp file (template weights)
    with open(temp_path, "w") as f:
        for k in range(st.nt):
            f.write(f"{wfrac[k]:17.8E} {k + 1:11d}\n")

    # Write .ascii file (per-pixel data for plotting with pspecfit).
    # Format: wavelength  data  model  continuum  template  ierr
    # Columns 1-3: standard spectrum/model vectors.
    # Column 4: ALS continuum (drawn in green by pspecfit as the baseline).
    # Column 5: template-only spectrum (model without continuum scaling).
    # Column 6: integer pixel mask (0=good, 1=masked).
    # pspecfit.f reads: read(2,*) x1,x2,x3,x4,x5,i6  (5 reals + 1 integer).
    with open(ascii_path, "w") as f:
        for i in range(st.iskip, st.npix - st.iskip):
            tt_out = tt[i] * cont[i] if st.fit_als_continuum else tt[i]
            f.write(
                f" {st.x[i]:12.5f}  {st.g[i]:15.8E}  {gp[i]:15.8E}"
                f"  {cont[i]:15.8E}  {tt_out:15.8E} {ierr[i]:1d}\n"
            )

    # Write .rms file (χ² and statistics)
    # Format: chi2_reduced  n_good  coff  coff2
    sl = slice(st.iskip, st.npix - st.iskip)
    gfr = good[sl]
    chi2_total = float(np.sum((st.g[sl][gfr] - gp[sl][gfr]) ** 2 / st.gerr[sl][gfr] ** 2))
    nrms = int(gfr.sum())
    chi2_red = chi2_total / max(nrms - 1, 1)
    with open(rms_path, "w") as f:
        f.write(f"{chi2_red:16.8E} {nrms:10d} {coff:14.8g} {coff2:14.8g}\n")

    log(f"Outputs written to {outdir}/ (prefix={prefix})")
    log("  .fit, .temp, .ascii, .rms files created")

    return {
        "gp": gp, "b": b, "w": w, "wfrac": wfrac, "tt": tt,
        "coff": coff, "coff2": coff2, "A": A,
        "chi2_red": chi2_red, "chi2_total": chi2_total, "nrms": nrms, "continuum": cont,
        "paths": {
            "fit": fit_path, "temp": temp_path, "ascii": ascii_path,
            "rms": rms_path,
        },
    }


def write_losvd_errors_file(
    summary: dict,
    prefix: str,
    outdir: str = ".",
) -> str:
    """
    Write LOSVD vector with error bars to a file for modeling pipeline.

    Parameters
    ----------
    summary : dict
        Output from losvd_errors.LOSVDErrorEstimator.summarize()
    prefix : str
        Output file prefix
    outdir : str
        Output directory

    Returns
    -------
    str
        Path to written file
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xl = summary["xl"]
    b_map = summary["b_map"]
    b_err = summary.get("b_err_recommended", np.zeros_like(b_map))
    b_lo = summary.get("b_lo_recommended", b_map - b_err)
    b_hi = summary.get("b_hi_recommended", b_map + b_err)
    method = summary.get("method_recommended", "unknown")

    outfile = outdir / f"{prefix}.losvd_errs.out"
    with open(outfile, "w") as f:
        f.write(f"# LOSVD error estimates (method: {method})\n")
        f.write("# bin_index  velocity_km/s  amplitude_map  amplitude_err  amplitude_lo  amplitude_hi\n")
        for i, (vel, amp, err, lo, hi) in enumerate(zip(xl, b_map, b_err, b_lo, b_hi)):
            f.write(f"{i+1:10d} {vel:15.6f} {amp:20.11E} {err:20.11E} {lo:20.11E} {hi:20.11E}\n")

    log(f"Wrote {outfile}")
    return str(outfile)


def write_gh_errors_file(
    summary: dict,
    prefix: str,
    outdir: str = ".",
) -> str:
    """
    Write Gauss-Hermite moment errors to a file.

    Parameters
    ----------
    summary : dict
        Output from losvd_errors.LOSVDErrorEstimator.summarize()
    prefix : str
        Output file prefix
    outdir : str
        Output directory

    Returns
    -------
    str
        Path to written file
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gh_map = summary["gh_map"]
    gh_err = summary.get("gh_err_recommended", {})
    gh_lo = summary.get("gh_lo_recommended", {})
    gh_hi = summary.get("gh_hi_recommended", {})
    method = summary.get("method_recommended", "unknown")

    params = ["V", "sigma", "h3", "h4"]
    keys = ["vherm", "sherm", "h3", "h4"]

    outfile = outdir / f"{prefix}.gh_errs.out"
    with open(outfile, "w") as f:
        f.write(f"# Gauss-Hermite moment errors (method: {method})\n")
        f.write("# parameter  value  error  error_lo  error_hi\n")
        for param, key in zip(params, keys):
            val = gh_map.get(key, np.nan)
            err_key = f"gh_{key}"
            err = gh_err.get(err_key, np.nan)
            lo = gh_lo.get(err_key, np.nan)
            hi = gh_hi.get(err_key, np.nan)
            f.write(f"{param:10s} {val:15.6f} {err:15.6f} {lo:15.6f} {hi:15.6f}\n")

    log(f"Wrote {outfile}")
    return str(outfile)
