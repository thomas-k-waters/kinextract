"""MUSE datacube I/O and spectrum extraction: a `MuseCube` container
that a real, downloaded FITS cube and `mock_cube`'s synthetic cube both
populate identically, so `binning.py`/`template_reduction.py` and any
notebook code built on top never need to know which kind of cube they're
looking at.

**Why one shared container instead of two code paths.** A real MUSE
Phase 3 cube and a synthetic mock differ in exactly one place -- how
`flux`/`error` got their values -- and nowhere else: both are a
`(n_wave, ny, nx)` flux cube plus a matching error cube, a 1-D wavelength
axis, and a spatial pixel scale. Keeping that the only shared surface
means the dartboard binning scheme and the template-reduction pipeline
are exercised identically on synthetic data (fast, ground truth known)
and real data (slow, no ground truth) with no separate mock-specific
code to keep in sync.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits


@dataclass
class MuseCube:
    """A MUSE-like IFU datacube, real or synthetic.

    flux, error : ndarray, shape `(n_wave, ny, nx)`.
    wave : ndarray, shape `(n_wave,)` -- observed-frame wavelength
        (Angstrom).
    pixel_scale : spatial pixel size (arcsec/pixel), constant and equal
        on both spatial axes.
    wcs : `astropy.wcs.WCS`, if built from a real FITS file with one;
        `None` for a synthetic cube (its spatial coordinates are already
        plain pixel indices, no on-sky projection to speak of).
    truth : only populated by `mock_cube.build_mock_cube` -- the known
        generating parameters (`w_true`, `b_true`, `v_grid`,
        `template_select`, `sersic_params`, ...) a synthetic cube's
        recovery can be checked against. Always `None` for a real cube,
        since there is no ground truth for real data.
    """
    flux: np.ndarray
    error: np.ndarray
    wave: np.ndarray
    pixel_scale: float
    wcs: object | None = None
    truth: dict | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape, `(ny, nx)`."""
        return self.flux.shape[1], self.flux.shape[2]


def white_light_image(cube: MuseCube) -> np.ndarray:
    """Collapse a cube along wavelength (`nansum`) into a `(ny, nx)`
    white-light image -- the usual "let me see the galaxy and pick a
    center" sanity check before binning.
    """
    return np.nansum(cube.flux, axis=0)


def extract_aperture_spectrum(cube: MuseCube, spaxel_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum every spaxel where `spaxel_mask` (`(ny, nx)` bool) is True into
    one spectrum. Flux sums directly; errors combine in quadrature
    (independent per-spaxel noise, matching this cube's own noise
    model -- see `mock_cube.py`'s module docstring for the synthetic
    case, and `binning.assign_spaxels_to_bins`'s `bin_id_map` for the
    usual way `spaxel_mask` is built, e.g. `bin_id_map == some_bin_id`).

    Returns `(wave, flux, error)`, each shape `(n_wave,)`.
    """
    flux = np.nansum(cube.flux[:, spaxel_mask], axis=1)
    error = np.sqrt(np.nansum(cube.error[:, spaxel_mask] ** 2, axis=1))
    return cube.wave, flux, error


def load_muse_cube(fits_path: str) -> MuseCube:
    """Load a real MUSE Phase 3 datacube from disk.

    Expects the standard ESO Phase 3 MUSE cube layout: a `DATA` extension
    (flux) and a `STAT` extension (variance), both `(n_wave, ny, nx)`,
    with a FITS WCS on the `DATA` header whose third axis is wavelength
    (`CRVAL3`/`CRPIX3`/`CD3_3` or `CDELT3`, Angstrom) and whose first two
    axes carry the spatial pixel scale (`CD1_1`/`CDELT1`, degrees/pixel,
    assumed square pixels). Falls back to the primary HDU if no `DATA`/
    `STAT` extension names are present (some cubes ship flux-only, no
    variance -- `error` is then `NaN` everywhere and must be supplied
    separately, e.g. from a known per-pixel noise level).

    **Validated so far only against `mock_cube.py`'s own synthetic FITS
    output** (see `tests/test_muse_cube.py`) -- real-cube header
    conventions can vary across ESO pipeline versions; treat this as a
    reasonable-effort reader to be hardened once tested against an actual
    downloaded cube, not as exhaustively real-data-validated yet.
    """
    with fits.open(fits_path) as hdul:
        names = [hdu.name for hdu in hdul]
        data_hdu = hdul["DATA"] if "DATA" in names else hdul[0]
        flux = np.asarray(data_hdu.data, dtype=np.float64)
        header = data_hdu.header

        if "STAT" in names:
            variance = np.asarray(hdul["STAT"].data, dtype=np.float64)
            error = np.sqrt(np.maximum(variance, 0.0))
        else:
            error = np.full_like(flux, np.nan)

        n_wave = flux.shape[0]
        crval3 = float(header["CRVAL3"])
        crpix3 = float(header.get("CRPIX3", 1.0))
        cd3_3 = float(header["CD3_3"]) if "CD3_3" in header else float(header["CDELT3"])
        wave = crval3 + (np.arange(n_wave) + 1.0 - crpix3) * cd3_3

        cd1_1 = header["CD1_1"] if "CD1_1" in header else header["CDELT1"]
        pixel_scale = abs(float(cd1_1)) * 3600.0  # degrees/pixel -> arcsec/pixel

        wcs = None
        try:
            from astropy.wcs import WCS
            wcs = WCS(header)
        except Exception:
            pass  # WCS is a convenience for display only; absence doesn't block fitting

        return MuseCube(flux=flux, error=error, wave=wave, pixel_scale=pixel_scale, wcs=wcs)


def save_muse_cube_fits(cube: MuseCube, fits_path: str) -> None:
    """Write a `MuseCube` to disk in the same `DATA`/`STAT`-extension
    layout `load_muse_cube` reads, so a synthetic cube built by
    `mock_cube.py` can round-trip through the same reader real cubes use
    (see `tests/test_muse_cube.py`) -- this is what lets Stage 3 of this
    package's own development test `load_muse_cube` without needing a
    real downloaded cube yet.
    """
    header = fits.Header()
    header["CRVAL3"] = float(cube.wave[0])
    header["CRPIX3"] = 1.0
    header["CD3_3"] = float(cube.wave[1] - cube.wave[0]) if len(cube.wave) > 1 else 1.0
    header["CD1_1"] = -cube.pixel_scale / 3600.0
    header["CD2_2"] = cube.pixel_scale / 3600.0

    data_hdu = fits.ImageHDU(data=cube.flux.astype(np.float32), header=header, name="DATA")
    stat_hdu = fits.ImageHDU(data=(cube.error.astype(np.float32) ** 2), header=header, name="STAT")
    hdul = fits.HDUList([fits.PrimaryHDU(), data_hdu, stat_hdu])
    hdul.writeto(fits_path, overwrite=True)
