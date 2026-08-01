"""ZAP (Zurich Atmosphere Purge, Soto et al. 2016) sky-subtraction-residual
cleaning for a real MUSE cube -- a cube-level preprocessing step upstream
of everything kinextract does, evaluated as a more principled
replacement for the fixed `SKY_RESIDUAL_WINDOWS` list every real-data
notebook in this package currently hardcodes (`VALIDATION_CAMPAIGN.md`
Stage 4's four windows, found once on NGC 5102 and reused everywhere).

ZAP removes residual OH-airglow/sky-subtraction structure via PCA/SVD on
designated sky pixels, then subtracts the reconstructed residual cube-wide.
It does **not** address real astrophysical emission-line contamination --
if anything the opposite risk applies (its own docs warn that too many
eigenvectors can subtract genuine emission signal along with sky
residuals) -- so this is complementary to, not a replacement for,
`kinextract.cleaning.detect_emission_lines`/`detect_emission_from_residual`.
It is also not expected to fully explain a fit whose chi2_red is far from
1 for other reasons (unmodeled continuum shape, template mismatch,
mis-scaled noise) -- see `dev_notes/` for that separate investigation.

Requires the optional `zap` and `photutils` packages (not kinextract
dependencies) -- install separately if not already present:
    pip install zap-muse photutils

Runtime: genuinely multi-minute on a full MUSE cube (direct benchmark on
this project's own 318x317x3681 NGC 5102 cube: the `cfwidthSVD=300`
continuum-filter pass alone extrapolates to ~9 min single-core, less with
more cores but not free) -- this is expected, not a hang. Run once, cache
the cleaned cube to disk, reuse it afterward.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation

from kinextract import load_muse_cube, white_light_image


def build_source_mask(cube, nsigma: float = 2.0, npixels: int = 30, grow: int = 15) -> np.ndarray:
    """Build a 2D boolean source mask (`True` = source, `False` = sky) for
    ZAP's own `mask` argument, from a `MuseCube`'s white-light image.

    Parameters
    ----------
    cube : `kinextract.MuseCube` (e.g. from `load_muse_cube`).
    nsigma : detection threshold above the background (passed to
        `photutils.segmentation.detect_threshold`). Default 2.0.
    npixels : minimum connected pixels for a detection (`detect_sources`).
        Default 30.
    grow : number of `scipy.ndimage.binary_dilation` iterations applied to
        the detected source mask, to include faint galaxy wings the
        detection threshold alone would miss. Default 15 (at MUSE's usual
        0.2"/pixel scale, ~3").

    Returns
    -------
    ndarray of bool, shape `(ny, nx)` -- `True` where a source was
    detected (to be excluded from ZAP's sky/SVD computation).
    """
    from photutils.segmentation import detect_sources, detect_threshold

    white = white_light_image(cube)
    threshold = detect_threshold(white, nsigma=nsigma)
    segm = detect_sources(white, threshold, npixels=npixels)
    if segm is None:
        return np.zeros_like(white, dtype=bool)
    mask = segm.data > 0
    if grow > 0:
        mask = binary_dilation(mask, iterations=grow)
    return mask


def run_zap_cleaning(cube_path: str, mask_path: str | None = None,
                      output_cube_path: str | None = None,
                      varcurve_path: str | None = None,
                      nsigma: float = 2.0, npixels: int = 30, grow: int = 15,
                      zlevel: str = "median", cftype: str = "median",
                      cfwidthSVD: float = 300, cfwidthSP: float = 40,
                      nevals=(), interactive: bool = False,
                      min_sky_fraction: float = 0.1,
                      overwrite: bool = True, ncpu: int | None = None) -> dict:
    """Run ZAP sky-residual cleaning on a real MUSE cube, end-to-end:
    build a source mask, write it, run `zap.process`, and (unless
    `interactive=True`) write the cleaned cube via `Zap.mergefits` --
    which preserves the original `STAT` (variance) extension untouched, so
    the output stays directly loadable by `kinextract.load_muse_cube`
    (unlike `Zap.writecube`, which writes flux only, no variance).

    With `interactive=True`, nothing is written (matches `zap.process`'s
    own semantics) -- the returned `zobj` is for inspecting/tuning before
    committing to a final write (see the returned dict's docstring below).

    Parameters
    ----------
    cube_path : path to the input MUSE cube (ESO Phase 3 `DATA`/`STAT`
        layout, as `load_muse_cube` expects).
    mask_path : path to write the 2D source mask FITS to. Defaults to
        `<cube_path stem>_zap_mask.fits` alongside the input cube.
    output_cube_path : path to write the cleaned cube to (ignored if
        `interactive=True`). Defaults to `<cube_path stem>_zap.fits`.
    varcurve_path : optional path to write the explained-variance curves
        to (ignored if `interactive=True`). `None` (default) skips it.
    nsigma, npixels, grow : passed to `build_source_mask`.
    zlevel, cftype, cfwidthSVD, cfwidthSP, nevals, ncpu : passed straight
        through to `zap.process` -- see that function's own docstring
        (https://zap.readthedocs.io/) for what each controls. Defaults
        here match the values recommended there for a galaxy field
        (`cfwidthSVD=300`, `cfwidthSP=40` for better source tracing) with
        automatic eigenvector selection (`nevals=()`).
    interactive : if `True`, returns the `Zap` object without writing
        anything -- inspect via `zobj.plotvarcurve()`, retune via
        `zobj.reprocess(nevals=...)`, then write explicitly (see the
        printed instructions). Default `False` (writes immediately using
        ZAP's own automatic eigenvector selection).
    min_sky_fraction : warn if fewer than this fraction of spaxels remain
        unmasked (available as "sky") after source detection -- ZAP's own
        "filled field" case, where the SVD eigenbasis is poorly
        constrained by too little genuine sky (see `zap.SVDoutput`/
        `extSVD` for the offset-sky-observation workaround if this
        triggers). Default 0.1.
    overwrite : overwrite existing output files. Default `True`.

    Returns
    -------
    dict with keys:
      zobj -- the `Zap` object if `interactive=True`, else `None` (already
        written to disk at that point).
      mask_path, output_cube_path, varcurve_path -- the paths actually
        used (`output_cube_path`/`varcurve_path` are `None` if
        `interactive=True`, since nothing was written).
      sky_fraction -- fraction of spaxels available as sky after masking.
    """
    import zap

    cube_path = str(cube_path)
    cube = load_muse_cube(cube_path)
    mask = build_source_mask(cube, nsigma=nsigma, npixels=npixels, grow=grow)
    sky_fraction = float(1.0 - mask.mean())

    print(f"source mask: {100*mask.mean():.1f}% of spaxels flagged as source, "
          f"{100*sky_fraction:.1f}% available as sky for the SVD")
    if sky_fraction < min_sky_fraction:
        warnings.warn(
            f"run_zap_cleaning: only {100*sky_fraction:.1f}% of spaxels are available as "
            f"sky, below the {100*min_sky_fraction:.0f}% sanity threshold -- close to ZAP's "
            "own 'filled field' case, where the SVD eigenbasis is poorly constrained by too "
            "little genuine sky. Consider a separate offset-sky exposure with "
            "zap.SVDoutput()/extSVD instead, or check the mask directly before trusting this "
            "result.",
            UserWarning, stacklevel=2,
        )

    if mask_path is None:
        mask_path = str(Path(cube_path).with_name(Path(cube_path).stem + "_zap_mask.fits"))
    fits.writeto(mask_path, mask.astype(np.uint8), overwrite=overwrite)
    print(f"source mask written to {mask_path}")

    if output_cube_path is None:
        output_cube_path = str(Path(cube_path).with_name(Path(cube_path).stem + "_zap.fits"))

    zobj = zap.process(
        cubefits=cube_path,
        outcubefits=output_cube_path,
        mask=mask_path,
        zlevel=zlevel, cftype=cftype,
        cfwidthSVD=cfwidthSVD, cfwidthSP=cfwidthSP,
        nevals=nevals,
        interactive=interactive,
        varcurvefits=varcurve_path,
        overwrite=overwrite,
        ncpu=ncpu,
    )

    if interactive:
        print(
            "interactive=True: nothing written yet. Inspect with zobj.plotvarcurve(), "
            "retune with zobj.reprocess(nevals=...), then write explicitly:\n"
            f"  zobj.mergefits({output_cube_path!r}, overwrite=True)  "
            "# preserves the STAT extension -- use this, not zobj.writecube()\n"
            + (f"  zobj.writevarcurve({varcurve_path!r}, overwrite=True)\n" if varcurve_path else "")
        )
        return dict(zobj=zobj, mask_path=mask_path, output_cube_path=None,
                    varcurve_path=None, sky_fraction=sky_fraction)

    print(f"cleaned cube written to {output_cube_path} (STAT extension preserved from the input)")
    if varcurve_path is not None:
        print(f"variance curves written to {varcurve_path}")
    return dict(zobj=None, mask_path=mask_path, output_cube_path=output_cube_path,
                varcurve_path=varcurve_path, sky_fraction=sky_fraction)
