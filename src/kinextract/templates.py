"""Stellar template loading and interpolation for kinextract.

This module builds the template matrix ``T`` (and its error matrix
``T_err``) used by the LOSVD fit: each column is a single-star or SSP
template spectrum, resampled onto the galaxy's observed wavelength grid so
it can be convolved with a trial LOSVD and compared pixel-by-pixel against
the observed spectrum. Templates supplied in physical flux units are
median-normalized to ~1 so their scale matches the (typically ALS- or
otherwise continuum-normalized) galaxy spectrum, with template weights fit
separately by the optimizer to reflect each star's contribution.

It also provides the instrumental line-spread-function (LSF) matching
machinery (:func:`resolution_mismatch_sigma_A`, :func:`convolve_gaussian_pixels`)
used by :func:`~kinextract.spectrum.make_fit_state` to remove a resolution
mismatch between the galaxy data and the template library *before* the
LOSVD fit, so that the recovered LOSVD width reflects genuine kinematic
broadening rather than a resolution difference baked in as spurious extra
"kinematics". This only activates when the caller supplies both
``cfg.data_fwhm_A`` and ``cfg.template_fwhm_A`` (see :class:`~kinextract.config.FitConfig`);
by default kinextract assumes the two are already matched.
"""
from __future__ import annotations
import numpy as np

# Gaussian FWHM = 2*sqrt(2*ln(2)) * sigma
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def resolution_mismatch_sigma_A(
    data_fwhm_A: float, template_fwhm_A: float, atol: float = 1e-6,
) -> tuple[float, str]:
    """Compute the Gaussian kernel needed to remove a data/template LSF mismatch.

    Following the standard convolution-matching convention (e.g. Cappellari
    2017, pPXF): if the two instrumental line-spread functions (LSFs) have
    different Gaussian FWHM, the *sharper* (narrower) one must be convolved
    with an additional Gaussian kernel of width ``sigma_diff = sqrt(sigma_broad**2
    - sigma_narrow**2)`` (quadrature subtraction) to bring it down to the
    coarser resolution -- you cannot sharpen the coarser side, only degrade
    the sharper one, so the fit is always carried out at the *worse* of the
    two resolutions.

    Parameters
    ----------
    data_fwhm_A : float
        Instrumental LSF FWHM of the galaxy spectrum, in Angstrom, in the
        *same rest-frame wavelength units used for the fit* (i.e. already
        deredshifted if it was measured in the observed frame -- see
        ``FitConfig.data_fwhm_frame``).
    template_fwhm_A : float
        Instrumental LSF FWHM of the stellar template library, in Angstrom,
        in the template's native (rest-frame) wavelength units.
    atol : float, optional
        Absolute tolerance (Angstrom, in sigma units) below which the two
        resolutions are treated as already matched and no convolution is
        applied.

    Returns
    -------
    sigma_diff_A : float
        The Gaussian sigma (Angstrom) of the kernel to convolve with the
        sharper spectrum. Zero if the two are already matched.
    direction : str
        ``"convolve_templates"`` if the templates are sharper and must be
        degraded to match the data; ``"convolve_data"`` if the data is
        sharper and must be degraded to match the templates (rarer, but
        supported since which side is sharper depends on the specific
        instrument/library pairing and should not be assumed); ``"none"``
        if the two already match to within ``atol``.

    Raises
    ------
    ValueError
        If either FWHM is not a finite positive number.
    """
    data_fwhm_A = float(data_fwhm_A)
    template_fwhm_A = float(template_fwhm_A)
    if not (np.isfinite(data_fwhm_A) and data_fwhm_A > 0):
        raise ValueError(f"data_fwhm_A must be a finite positive number, got {data_fwhm_A!r}")
    if not (np.isfinite(template_fwhm_A) and template_fwhm_A > 0):
        raise ValueError(f"template_fwhm_A must be a finite positive number, got {template_fwhm_A!r}")

    sigma_data = data_fwhm_A * _FWHM_TO_SIGMA
    sigma_tpl = template_fwhm_A * _FWHM_TO_SIGMA

    if abs(sigma_data - sigma_tpl) <= atol:
        return 0.0, "none"
    if sigma_tpl < sigma_data:
        return float(np.sqrt(sigma_data ** 2 - sigma_tpl ** 2)), "convolve_templates"
    return float(np.sqrt(sigma_tpl ** 2 - sigma_data ** 2)), "convolve_data"


def convolve_gaussian_pixels(flux: np.ndarray, sigma_pix: float, axis: int = -1) -> np.ndarray:
    """Convolve `flux` with a Gaussian kernel of a fixed width in pixels.

    Thin wrapper over `scipy.ndimage.gaussian_filter1d` used to apply the
    resolution-matching kernel from :func:`resolution_mismatch_sigma_A`
    (after converting its Angstrom sigma to pixels via the wavelength
    step). Works on a single 1-D spectrum or a 2-D template matrix (pass
    ``axis=0`` for a (npix, ntemplates) matrix, matching the convention
    used in :func:`build_template_matrix_fortran`).

    Parameters
    ----------
    flux : ndarray
        Spectrum or template matrix to convolve.
    sigma_pix : float
        Gaussian sigma, in pixels. If ``<= 0``, `flux` is returned
        unchanged (no-op) rather than raising, since a zero-width match is
        a valid (if unusual) outcome of :func:`resolution_mismatch_sigma_A`.
    axis : int, optional
        Axis along which to convolve (the wavelength axis).

    Returns
    -------
    ndarray
        Convolved array, same shape as `flux`.

    Notes
    -----
    Uses ``mode="nearest"`` boundary handling (edge values repeated) rather
    than wrapping or zero-padding, since spectra are not periodic and
    zero-padding would spuriously pull flux down near the edges.
    """
    if sigma_pix <= 0:
        return np.asarray(flux, float)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(flux, float), sigma_pix, axis=axis, mode="nearest")


# =============================================================================
# Section 5 - Template handling
# =============================================================================

def interp_template_tp_with_outside(
    xg: np.ndarray, xt: np.ndarray, ft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a single template spectrum onto the galaxy grid.

    Resamples a template's ``(xt, ft)`` tabulation onto the target
    wavelength array ``xg`` using piecewise-linear interpolation, which is
    sufficient because template libraries are typically sampled at much
    finer or comparable resolution to the galaxy spectrum.

    Parameters
    ----------
    xg : ndarray
        Target (galaxy) wavelength grid, in Angstroms.
    xt : ndarray
        Template wavelength grid, in Angstroms (must be increasing).
    ft : ndarray
        Template flux values sampled at `xt`.

    Returns
    -------
    tp : ndarray
        Interpolated template values on `xg`. Pixels outside the template's
        wavelength coverage are set to 1.0 (flat/neutral) rather than
        extrapolated, since the template carries no information there.
    outside : ndarray of bool
        True where `xg` falls outside ``[xt[0], xt[-1])``, i.e. where `tp`
        is the fill value rather than a genuine interpolated value.

    Notes
    -----
    The `outside` mask is returned as a dedicated boolean array instead of
    being inferred from ``tp == 1.0``, because that sentinel comparison
    could misidentify genuine in-range pixels whose interpolated value
    happens to equal 1.0 exactly.
    """
    xg, xt, ft = np.asarray(xg, float), np.asarray(xt, float), np.asarray(ft, float)
    tp = np.ones(len(xg), float)
    outside = (xg < xt[0]) | (xg >= xt[-1])

    j1 = np.searchsorted(xt, xg, side="right")
    j0 = j1 - 1
    inside = ~outside & (j0 >= 0) & (j1 < len(xt))

    jj = j0[inside]
    x0, x1 = xt[jj], xt[jj + 1]
    f0, f1 = ft[jj], ft[jj + 1]
    ok = (f0 > 0) & (f1 > 0) & (x1 != x0)
    val = np.ones(np.count_nonzero(inside), float)
    frac = np.zeros_like(val)
    frac[ok] = (xg[inside][ok] - x0[ok]) / (x1[ok] - x0[ok])
    val[ok] = f0[ok] + (f1[ok] - f0[ok]) * frac[ok]
    tp[inside] = val
    return tp, outside


def build_template_matrix_fortran(
    xg: np.ndarray, template_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the full (npix, ntemplates) stellar template matrix.

    Loads every template listed in `template_paths`, median-normalizes each
    to a flux level of approximately 1.0, and interpolates it onto the
    galaxy wavelength grid `xg`. The resulting matrix is what
    :mod:`kinextract.numerics` convolves with a trial LOSVD (one column at a
    time, weighted by a fitted per-template amplitude) to build the model
    spectrum for chi-squared minimization.

    Median normalization is required when templates are supplied in
    physical flux units (e.g. MUSE spectral library ``.dat`` files in
    erg cm⁻² s⁻¹ Å⁻¹) so their scale matches the galaxy spectrum, which is
    itself close to unity after ALS or pre-normalization.

    Parameters
    ----------
    xg : ndarray
        Galaxy wavelength grid, in Angstroms, onto which every template is
        resampled.
    template_paths : list of str
        Paths to individual template spectrum files (2- or 3-column
        wavelength/flux[/flux_err] text files), typically produced by
        :func:`kinextract.io.read_template_list`.

    Returns
    -------
    T : ndarray, shape (npix, ntemplates)
        Template matrix on the galaxy grid, each column normalized so its
        flux values are of order 1.
    T_err : ndarray, shape (npix, ntemplates)
        Per-pixel template flux uncertainty, normalized identically to `T`.
        Columns are all-zero for templates whose file had no error column.
    outside_each : ndarray of bool, shape (npix, ntemplates)
        True where a given template has no wavelength coverage at a given
        pixel -- used to exclude just that template (not the whole pixel)
        from the per-pixel mixture when template libraries have slightly
        different native wavelength ranges (the common case for real
        libraries assembled from more than one source/run).
    outside_all : ndarray of bool, shape (npix,)
        True at pixels that fall outside the wavelength coverage of *every*
        template in the list; such pixels carry no template information at
        all and are typically masked out of the fit (``gerr`` set to
        `BIG`).
    """
    from .io import read_template_xy
    T = np.empty((len(xg), len(template_paths)), float)
    T_err = np.zeros_like(T)
    outside_each = np.zeros((len(xg), len(template_paths)), dtype=bool)
    for k, p in enumerate(template_paths):
        wave, flux, err = read_template_xy(p)
        # Normalize to median positive flux so template ≈ 1.0 (shape only).
        pos = flux > 0
        med = float(np.nanmedian(flux[pos])) if pos.any() else 1.0
        if med > 0:
            flux = flux / med
            if err is not None:
                err = err / med
        tp, outside = interp_template_tp_with_outside(xg, wave, flux)
        T[:, k] = tp
        outside_each[:, k] = outside
        if err is not None:
            te, _ = interp_template_tp_with_outside(xg, wave, err)
            T_err[:, k] = te
    outside_all = outside_each.all(axis=1)
    return T, T_err, outside_each, outside_all
