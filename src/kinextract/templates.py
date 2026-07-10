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


def reduce_templates_svd(
    T: np.ndarray, n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a template library to ``n_components`` orthogonal eigen-templates.

    A large stellar library assembled from individual real stars (as
    opposed to a dense, physically-smooth grid of synthetic SSP/stellar
    models, e.g. E-MILES) has no smooth structure between templates --
    each one is an idiosyncratic individual spectrum. Fitting all of them
    simultaneously with independent, non-negative weights gives the
    optimizer far more freedom than the data can actually constrain:
    confirmed directly (see the notebooks/conversation this was built for)
    by refitting the identical synthetic spectrum at several different
    noise realizations and finding wildly different recovered velocities
    (over 100 km/s of scatter) even though every fit converged to a
    similar, plausible chi-squared -- a genuine, reproducible degeneracy,
    not an optimizer failure.

    Truncated SVD (no mean-subtraction) replaces the ``nt`` raw templates
    with the top ``n_components`` right-singular vectors of the
    (rescaled) template matrix, which span the same dominant subspace at
    a fraction of the free parameters. This is the standard remedy used
    for exactly this failure mode with large template libraries in the
    literature (e.g. Cappellari 2017, pPXF); it works best combined with
    a spectral-type-restricted starting library (physically implausible
    templates -- e.g. wildly wrong luminosity classes for the science
    target -- should be excluded *before* this reduction, not relied on
    to be down-weighted by it).

    **Fitting with these eigen-templates requires template weights to be
    allowed negative values** (unlike ordinary per-star templates, which
    are fit with a non-negative-weight convention): only the dominant,
    most positive-definite component behaves like an ordinary flux
    spectrum; higher-order components are orthogonal correction terms
    that generically have both positive- and negative-going regions, so
    reconstructing the true template mixture legitimately needs
    mixed-sign coefficients on them. Pass a FitConfig with
    ``template_w_bounds`` set to something like ``(-1.0, 1.0)`` when
    fitting with eigen-templates -- the ordinary default (non-negative)
    bounds will silently distort the fit by disallowing the sign
    corrections these templates need.

    Parameters
    ----------
    T : ndarray, shape (npix, ntemplates)
        Template matrix, e.g. from :func:`build_template_matrix_fortran`
        (same ``(npix, ntemplates)`` column convention).
    n_components : int
        Number of eigen-templates to keep, capped at ``min(npix, ntemplates)``.

    Returns
    -------
    eigen_templates : ndarray, shape (npix, n_components)
        Reduced template matrix, median-normalized the same way ordinary
        templates are (see :func:`build_template_matrix_fortran`), with
        each component's sign oriented so its dominant contribution to
        reconstructing the (positive) input templates is positive.
    explained_variance : ndarray, shape (n_components,)
        Fraction of the input matrix's total variance captured by each
        kept component, in decreasing order -- a diagnostic for how much
        of the library's real diversity survives the reduction (and,
        conversely, how much residual template-mismatch risk remains).
    """
    T = np.asarray(T, float)
    npix, nt = T.shape
    n_components = int(min(n_components, npix, nt))

    # Put every input template on a comparable scale first (matches this
    # module's median~1 convention) so no single star's overall brightness
    # dominates the SVD purely because of its normalization.
    scale_in = np.median(np.abs(T), axis=0, keepdims=True)
    scale_in = np.where(scale_in > 0, scale_in, 1.0)
    T_scaled = T / scale_in

    # No mean-subtraction: this keeps the dominant (first) component close
    # to a rescaled "typical" spectrum shape (mostly positive), which is
    # what lets it alone already explain most of the library's variance --
    # mean-centering would instead force every component, including the
    # first, to be a zero-mean correction term needing negative weights.
    #
    # T_scaled is (npix, nt), so U (npix, k) holds the wavelength-space
    # eigen-spectra we actually want as templates; Vt (k, nt) holds each
    # component's *template-space* loading (how much of each original star
    # it draws on), used only below for the sign convention.
    U, S, Vt = np.linalg.svd(T_scaled, full_matrices=False)
    eigen_templates = U[:, :n_components].copy()  # (npix, n_components)

    # Sign convention: orient each component so its dominant contribution to
    # reconstructing the (positive) input templates is positive.
    signs = np.sign(np.sum(Vt[:n_components], axis=1))
    signs[signs == 0] = 1.0
    eigen_templates *= signs[np.newaxis, :]

    # Re-apply the median~1 convention to the reduced templates themselves.
    scale_out = np.median(np.abs(eigen_templates), axis=0, keepdims=True)
    scale_out = np.where(scale_out > 0, scale_out, 1.0)
    eigen_templates = eigen_templates / scale_out

    total_var = float(np.sum(S ** 2))
    explained_variance = (S[:n_components] ** 2) / total_var if total_var > 0 else np.zeros(n_components)
    return eigen_templates, explained_variance


def write_svd_reduced_templates(
    template_list_file: str, template_dir: str, wavelength_grid: np.ndarray,
    n_components: int, out_dir: str,
    continuum_smooth_sigma_pix: float = 200.0,
) -> str:
    """Build SVD-reduced eigen-templates from a Tlist and write them out as
    an ordinary template set, ready to use as a new ``template_list_file``.

    A convenience wrapper around :func:`reduce_templates_svd`: reads every
    template in ``template_list_file``, resamples them onto
    ``wavelength_grid``, reduces to ``n_components`` eigen-templates, and
    writes each one as a 3-column (wavelength, flux, flux_err placeholder)
    ``.dat`` file in ``out_dir`` plus a matching ``Tlist``. The returned
    path is the new ``Tlist`` -- pass it (with ``out_dir``) as
    ``template_list_file``/``template_dir`` in a fresh ``FitConfig``, and
    remember to set ``template_w_bounds`` (see :func:`reduce_templates_svd`).

    Parameters
    ----------
    template_list_file, template_dir : str
        The original (unreduced) template list and directory, e.g.
        ``examples/data/muse/Tlist`` and its containing directory.
    wavelength_grid : ndarray
        Wavelength grid to resample every template onto before reduction
        (should cover the full range any fit using the result will need,
        not just one fit window -- typically the galaxy's full observed
        grid, ``wavemin_full + arange(n_pix) * step``).
    n_components : int
        Number of eigen-templates to keep.
    out_dir : str
        Directory to write the reduced template files and new Tlist into
        (created if it doesn't exist).
    continuum_smooth_sigma_pix : float, optional
        Gaussian smoothing width (pixels) used to estimate and divide out
        each raw template's own continuum shape before the SVD -- see the
        note above on why this matters. Default 200 pixels matches this
        package's standard stellar-continuum-normalization convention
        (e.g. notebook 02's own template setup).

    Returns
    -------
    str
        Path to the newly-written ``Tlist`` in ``out_dir``.
    """
    from pathlib import Path

    from .io import read_template_list, read_template_xy

    paths = read_template_list(template_list_file, template_dir)
    wavelength_grid = np.asarray(wavelength_grid, float)
    T = np.empty((len(wavelength_grid), len(paths)), float)
    for k, p in enumerate(paths):
        wave, flux, _err = read_template_xy(p)
        pos = flux > 0
        med = float(np.nanmedian(flux[pos])) if pos.any() else 1.0
        if med > 0:
            flux = flux / med
        # Continuum-normalize (divide by a heavily-smoothed version of itself,
        # matching the standard library-preparation convention -- see e.g.
        # examples/notebooks/02_realistic_mock_fit.ipynb's own template setup)
        # *before* the SVD. Without this, each raw physical-flux template's
        # own broadband SED shape (stellar temperature/brightness) dominates
        # the variance the SVD sees -- confirmed directly: on the 10-star
        # MUSE G/K-giant subset, the first *raw*-flux component alone
        # captures 96% of the variance, essentially all of it continuum
        # slope, none of it the absorption-line structure that actually
        # distinguishes templates kinematically. `fit_continuum=True` fits
        # away whatever overall shape the templates carry anyway, so that
        # variance is not just irrelevant here, it actively starves the
        # kept components of the line-shape diversity that matters.
        flux_smooth = convolve_gaussian_pixels(flux, continuum_smooth_sigma_pix)
        flux_smooth = np.where(flux_smooth > 0, flux_smooth, 1.0)
        flux = flux / flux_smooth
        tp, _outside = interp_template_tp_with_outside(wavelength_grid, wave, flux)
        T[:, k] = tp

    eigen_templates, explained_variance = reduce_templates_svd(T, n_components)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(eigen_templates.shape[1]):
        name = f"eigen_{i:02d}.dat"
        np.savetxt(
            out_dir_p / name,
            np.column_stack([wavelength_grid, eigen_templates[:, i],
                              np.full(len(wavelength_grid), 0.001)]),
            fmt="%12.4f  %14.8f  %12.8f",
        )
        names.append(name)
    tlist_path = out_dir_p / "Tlist"
    tlist_path.write_text("\n".join(names) + "\n")
    return str(tlist_path)
