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

import warnings
from dataclasses import dataclass

import numpy as np
from astropy.io import fits

# Bacon et al. (2017)'s own recommended correction for the MUSE pipeline's
# per-spaxel STAT-extension variance being biased low relative to the true
# single-spaxel noise -- a reasonable starting point for
# `combine_bin_spectrum_robust`'s own `variance_correction_factor`, not a
# universal constant (see that function's own docstring for why the
# default is `1.0`, not this value).
BACON2017_VARIANCE_CORRECTION_FACTOR = 1.0 / 0.6 ** 2


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


def spatial_correlation_inflation(spaxel_mask: np.ndarray, rho_spatial: float,
                                   max_dist: float = 6.0) -> float:
    """Variance-inflation factor for `combine_bin_spectrum_robust`'s formal
    inverse-variance combine, given real spaxel-to-spaxel correlation --
    see that function's own `spatial_rho` parameter for how this gets
    used, and `EHT_Project/notebooks/N4751_notebooks/kinematics/NFM/
    02_noise_correlation.ipynb`'s own Section 8 for how `rho_spatial` is
    measured on real data (wavelength-domain per-spaxel residuals,
    correlated between spatial neighbors -- NOT raw flux, which is
    dominated by real galaxy light gradient, not noise; see that
    section's own documented false start for why).

    **The model**: an isotropic 2-D generalization of the same AR(1)
    picture `noise_model.whiten` already uses along wavelength --
    `Corr(i, j) = rho_spatial ** dist(i, j)`, `dist` the real Euclidean
    spaxel separation (not just an integer lag), matching the close
    x/y-lag-1 agreement (~0.15 vs ~0.15) the notebook measured, i.e.
    treating the correlation as direction-independent.

    **What "inflation" means here.** For `N` equally-weighted, correlated
    samples, `Var(mean) = (var/N**2) * sum_i sum_j rho_spatial**dist(i,j)`
    instead of the independent-noise `var/N` -- the ratio of those two,
    `(1/N) * sum_i sum_j rho_spatial**dist(i,j)`, is what this function
    returns. It is exactly `1.0` at `rho_spatial=0` (no correction) and
    grows with both `rho_spatial` and how spatially compact the bin is
    (a bin's own many-spaxel `sqrt(N)` improvement is capped by this,
    not unbounded, exactly Section 8's own finding).

    **A real simplification, not silently swept under the rug**: this
    treats every spaxel as equally weighted, when `combine_bin_spectrum_
    robust`'s actual inverse-variance weights vary (mildly) with
    wavelength and per-spaxel noise level. Computing the exact weighted
    version at every wavelength for a bin with thousands of spaxels is
    not worth its own cost for a correction this size (`rho_spatial ~
    0.15`, an order of magnitude weaker than the wavelength-axis effect)
    -- this single geometry-only scalar, applied uniformly across
    wavelength, is treated as good enough; revisit only if `rho_spatial`
    on some future target turns out to be much larger.

    **Why `max_dist` matters for performance, not just correctness**: an
    exact `N x N` pairwise sum is `O(N**2)` -- prohibitive for this
    project's own largest real bins (~4800 spaxels, ~23 million pairs).
    `rho_spatial ** dist` is negligible well before `dist=6` for any
    `rho_spatial` this package has actually measured (`0.15 ** 6 ~
    1.1e-5`), so pairs farther apart than `max_dist` are dropped entirely
    rather than summed -- turning this into an effectively `O(N)`
    nearest-neighbor query (`scipy.spatial.cKDTree`) instead, since each
    spaxel only has a small, bounded number of neighbors within
    `max_dist` regardless of how large the bin itself is.

    Parameters
    ----------
    spaxel_mask : `(ny, nx)` bool, same convention as `combine_bin_
        spectrum_robust`'s own `spaxel_mask` -- spaxel positions are read
        directly from where this is `True`, in pixel units (not arcsec;
        `rho_spatial` was itself measured in pixel units, so this stays
        consistent without needing `cube.pixel_scale` at all).
    rho_spatial : lag-1 (nearest-neighbor) spatial correlation, e.g.
        `~0.15` for N4751's own NFM data (Section 8 above) -- measure
        fresh for a different target/instrument mode rather than reusing
        this number, the same way `FitConfig.rho`/`FALLBACK_RHO` are
        never silently reused for the wavelength axis either.
    max_dist : pixel separation beyond which `rho_spatial ** dist` is
        dropped from the sum (default `6.0`, negligible for any
        `rho_spatial` measured so far -- widen only if a future
        measurement finds a much larger `rho_spatial`, e.g. `> 0.5`,
        where the tail truncated here would no longer be negligible).

    Returns
    -------
    `variance_inflation` : multiply the naive (independent-noise) formal
        *variance* by this; equivalently multiply the formal *error* by
        `sqrt(variance_inflation)`. Always `>= 1.0`.
    """
    if rho_spatial <= 0.0:
        return 1.0
    ys, xs = np.nonzero(spaxel_mask)
    n = len(ys)
    if n <= 1:
        return 1.0
    from scipy.spatial import cKDTree
    points = np.column_stack([ys, xs]).astype(float)
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=max_dist, output_type="ndarray")
    total = float(n)  # the i == j diagonal: rho_spatial**0 == 1, summed N times
    if len(pairs) > 0:
        d = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        total += 2.0 * np.sum(rho_spatial ** d)  # unordered pairs -> count each (i,j) and (j,i)
    return total / n


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

    **A spaxel with non-finite flux/error, or non-positive error, at a
    given wavelength is excluded from that wavelength's sum** (not
    treated as a zero-flux, zero-error measurement -- silently summing a
    NaN-as-zero contribution would either drop real signal without
    adjusting for it, or worse, look like a perfectly-measured
    non-detection). If *every* spaxel in the aperture is bad at some
    wavelength, that output pixel is `NaN` (genuinely unmeasured, not
    zero) -- pass the result through `mask_wavelength_windows` (its own
    `drop_nonfinite=True` default) before fitting to remove these.

    Returns `(wave, flux, error)`, each shape `(n_wave,)`.
    """
    flux_pix = cube.flux[:, spaxel_mask]
    error_pix = cube.error[:, spaxel_mask]
    valid = np.isfinite(flux_pix) & np.isfinite(error_pix) & (error_pix > 0)
    n_valid = valid.sum(axis=1)
    flux = np.where(valid, flux_pix, 0.0).sum(axis=1)
    variance = np.where(valid, error_pix ** 2, 0.0).sum(axis=1)
    error = np.sqrt(variance)
    no_data = n_valid == 0
    flux = np.where(no_data, np.nan, flux)
    error = np.where(no_data, np.nan, error)
    return cube.wave, flux, error


def combine_bin_spectrum_robust(cube: MuseCube, spaxel_mask: np.ndarray,
                                 variance_correction_factor: float = 1.0,
                                 spatial_rho: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine every spaxel where `spaxel_mask` is True into one **mean**
    (not summed -- see below) bin spectrum: a robust (biweight) location
    across spaxels for the flux, and the standard inverse-variance-weighted
    combined error, `sqrt(1 / sum(1/var_i))`, for the noise.

    **No longer includes an "empirical" (biweight-scale-of-raw-flux)
    error term -- removed 2026-08-09, see `dev_notes/real_data_sigma/
    FINDINGS.md`'s own entry for the full diagnosis.** That term was
    ported from the legacy SCO Fortran framework's `imcmb.f`
    (`dvar`/`biwgt`), but `imcmb.f` combines **repeat exposures of the
    same already-spatially-binned spectrum** -- across exposures of one
    sky position, scatter in raw flux genuinely is noise. Applied here to
    **spatially different spaxels within one bin, from a single
    exposure**, the same formula instead measures real surface-brightness
    structure across the bin, not noise -- confirmed directly on real
    N4751 data: this term ran 120-470x the actual (formal) noise
    estimate, meaning the combined error was, for practical purposes,
    just this spatial-gradient term, unrelated to measurement
    uncertainty. This was the dominant driver of implausibly large
    reported errors (whitened chi2/dof as low as 0.01) on real data.

    Only the formal, independence-assuming inverse-variance term remains
    -- correct for spatial spaxel combination, though (like any such
    formula) it can under-report the true error if per-spaxel noise is
    positively correlated (e.g. from resampling/drizzling during pipeline
    reduction). That failure mode has its own, already-existing fix in
    this package: `noise_model.whiten`'s AR(1) `rho` machinery, applied
    at fit time on the wavelength axis, not folded into this function.
    Still uses `astropy.stats.biweight_location` (not a plain mean) for
    the flux itself -- resistant to a single bad/cosmic-ray-contaminated
    spaxel dominating the combined spectrum the way a plain mean would
    let it; that part of the port remains valid regardless of the above.

    `variance_correction_factor` : Bacon et al. (2017)'s own finding that
        the MUSE pipeline's per-spaxel `STAT`-extension variance is
        itself biased low relative to the true single-spaxel noise (their
        own recommended correction factor is `(1/0.6)**2 ~= 2.78`,
        already demonstrated in this project's own `05_noise_correlation.
        ipynb`). Multiplies every per-spaxel *variance* (not the flux) by
        this factor before combining -- default `1.0` (no correction)
        since this factor is instrument/pipeline-version specific and
        should be measured for a given cube, not silently assumed; pass a
        measured value explicitly (`1/0.6**2` is Bacon et al.'s own MUSE
        WFM/NFM estimate, a reasonable starting point, not a universal
        constant).

    `spatial_rho` : real spaxel-to-spaxel correlation (see
        `spatial_correlation_inflation`'s own docstring for the model and
        how to measure it -- `02_noise_correlation.ipynb`'s own Section 8
        measured `~0.15` for N4751's NFM data). Default `0.0` (no
        correction), same "measure it, don't silently assume it"
        convention as `variance_correction_factor`. Inflates `error`
        uniformly across wavelength by `sqrt(spatial_correlation_
        inflation(spaxel_mask, spatial_rho))` -- a real, if usually
        modest, floor under how tight a many-spaxel bin's error can get:
        the naive formula here promises unlimited `sqrt(N)` improvement,
        which real correlated spaxels don't actually deliver.

    Returns `(wave, flux, error)`, each shape `(n_wave,)` -- same shape
    contract as `extract_aperture_spectrum`, but `flux` here is a *mean*
    (one spaxel's worth of surface brightness), not that function's own
    *sum* (total flux across the aperture) -- pick whichever convention
    the caller actually needs; they are not interchangeable without
    rescaling by `n_valid`.
    """
    from astropy.stats import biweight_location

    flux_pix = cube.flux[:, spaxel_mask]
    error_pix = cube.error[:, spaxel_mask]
    valid = np.isfinite(flux_pix) & np.isfinite(error_pix) & (error_pix > 0)
    n_valid = valid.sum(axis=1)

    # Fully vectorized across wavelength (axis=1 = the spaxel axis) via astropy's own
    # ignore_nan support -- a per-wavelength Python loop calling biweight_location one
    # wavelength at a time does the identical computation but was measured to be
    # prohibitively slow for a bin with O(1e4) spaxels (e.g. the largest radial bins in
    # 04_template_calibration.ipynb's own real N4751 run, up to ~38000 spaxels) -- this
    # single-call-per-statistic form is the fix, not just a style preference.
    flux_masked = np.where(valid, flux_pix, np.nan)
    variance_pix = (error_pix ** 2) * variance_correction_factor
    variance_masked = np.where(valid & (variance_pix > 0), variance_pix, np.nan)

    with warnings.catch_warnings():
        # astropy's own biweight_location warns when a row is all-NaN (n_valid[iw]==0) --
        # handled explicitly below via `no_data`, not silently wrong; suppress the
        # otherwise-unavoidable per-row warning noise this triggers at scale.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*Input data contains invalid values.*")

        # n_valid==1 rows: biweight_location returns the lone finite value correctly on
        # its own (no special-casing needed).
        flux = biweight_location(flux_masked, axis=1, ignore_nan=True)
        formal_error = np.sqrt(1.0 / np.nansum(1.0 / variance_masked, axis=1))

    if spatial_rho > 0.0:
        formal_error = formal_error * np.sqrt(spatial_correlation_inflation(spaxel_mask, spatial_rho))

    no_data = n_valid == 0
    # formal_error is +inf where no_data (nansum over an all-NaN row -> 1/0); the actual
    # returned value is NaN there regardless (via the no_data mask applied below).
    error = np.where(no_data, 0.0, np.nan_to_num(formal_error, nan=0.0))
    flux = np.where(no_data, np.nan, flux)
    error = np.where(no_data, np.nan, error)
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
            # A zero or negative variance (a real occurrence in ESO Phase 3 STAT
            # extensions -- e.g. saturated/masked pixels) is not "perfectly measured,
            # zero noise" -- it means this pixel has no usable variance estimate at
            # all. Treating it as error=0.0 (np.sqrt(max(variance, 0)) alone would) is
            # actively worse than treating it as unmeasured: a zero-error pixel gets
            # effectively infinite statistical weight in any chi^2 fit, and dividing
            # by it directly (A_stack/errors, the whitening convention used
            # throughout this package) raises a divide-by-zero. NaN out both the
            # error AND the flux at these pixels, matching a genuinely bad pixel
            # (excluded downstream by mask_wavelength_windows's drop_nonfinite
            # default), not a suspiciously perfect one.
            bad = ~np.isfinite(variance) | (variance <= 0)
            error = np.sqrt(np.where(bad, np.nan, variance))
            flux = np.where(bad, np.nan, flux)
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


def trim_cube_wavelength_fits(in_path: str, out_path: str, wave_min: float, wave_max: float) -> None:
    """Write a new FITS cube containing only `[wave_min, wave_max]` (Angstrom,
    same frame as the input's own wavelength axis -- observed frame for a raw
    ESO Phase 3 download, not rest frame) of `in_path`'s `DATA`/`STAT`
    extensions, at the same spatial sampling and file layout `load_muse_cube`
    reads.

    A real MUSE-DEEP cube is tens of GB (the full instrument bandpass,
    ~4750-9350 Angstrom); most single-line science (e.g. the CaII triplet
    region for stellar kinematics) only ever fits a few hundred Angstrom of
    that. Trimming once, on disk, to just the window actually used means
    every downstream notebook cell loads and holds a cube an order of
    magnitude smaller, without changing anything about how that cube reads
    (same `DATA`/`STAT`/WCS-keyword layout as the untrimmed file).

    Reads only the requested wavelength slices via `ImageHDU.section`
    (memory-mapped -- never materializes the full input cube in RAM), so
    this runs safely on a machine too small to hold the untrimmed cube at
    all, which is the actual point of trimming it.
    """
    with fits.open(in_path, memmap=True) as hdul:
        names = [hdu.name for hdu in hdul]
        primary_header = hdul[0].header.copy()
        data_header = hdul["DATA" if "DATA" in names else 0].header.copy()

        crval3 = float(data_header["CRVAL3"])
        crpix3 = float(data_header.get("CRPIX3", 1.0))
        cd3_3 = float(data_header["CD3_3"]) if "CD3_3" in data_header else float(data_header["CDELT3"])
        n_wave = int(data_header["NAXIS3"])
        wave = crval3 + (np.arange(n_wave) + 1.0 - crpix3) * cd3_3

        i0 = int(np.searchsorted(wave, wave_min))
        i1 = int(np.searchsorted(wave, wave_max, side="right"))
        if i1 <= i0:
            raise ValueError(f"no wavelength slices in [{wave_min}, {wave_max}] "
                              f"-- cube covers {wave[0]:.1f}-{wave[-1]:.1f} A")
        new_crval3 = wave[i0]

        out_hdus = [fits.PrimaryHDU(header=primary_header)]
        for ext_name in ("DATA", "STAT"):
            if ext_name not in names:
                continue
            trimmed = hdul[ext_name].section[i0:i1, :, :]
            hdr = hdul[ext_name].header.copy()
            hdr["NAXIS3"] = i1 - i0
            hdr["CRVAL3"] = new_crval3
            hdr["CRPIX3"] = 1.0
            out_hdus.append(fits.ImageHDU(data=trimmed, header=hdr, name=ext_name))

        fits.HDUList(out_hdus).writeto(out_path, overwrite=True)
