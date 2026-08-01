"""Instrument/template line-spread-function (LSF) matching.

**Why this module exists.** `bias_correction`'s calibration and the whole
ablation ladder were built and validated using the package's original
33-star template library (`sco_workdir_example/.../muse_templates/`),
which -- based on its native 1.25 Angstrom/pixel sampling, exact
4750-9350 Angstrom range match to MUSE, and calibrated-flux-plus-error
format -- is almost certainly a set of real stars observed directly with
MUSE itself, not an external library. That means every fit done with it
so far has zero template/instrument LSF mismatch by construction: the
templates share the data's own instrumental resolution automatically.
That protection disappears the moment a general-purpose library (E-MILES,
or any other instrument's templates) is used instead, which is exactly
why this module has to exist before that integration goes further --
`continuum.py` already flagged "wavelength-dependent LSF matching is out
of scope" for the same underlying reason.

**The method (Gaussian quadrature-difference convolution).** If both the
template's native LSF and the target instrument's LSF are treated as
Gaussian with known FWHM as a function of wavelength, matching one to the
other is a single convolution with a Gaussian of

    sigma_match(lambda)^2 = sigma_target(lambda)^2 - sigma_template(lambda)^2

wherever this is positive (the template is the sharper of the two, the
ordinary case -- you can broaden a spectrum, you cannot sharpen one by
convolution). Where `sigma_target <= sigma_template`, no correction is
applied and the wavelength is flagged via `cannot_sharpen`; this is a
real limitation of any pre-fit LSF-matching approach, not a bug -- if it
happens over a wide, scientifically important range, that's a sign to
either widen the target instrument's effective resolution assumption or
accept the residual mismatch (see `cannot_sharpen` below) explicitly
rather than silently.

This follows the standard variable-width convolution recipe used by
`pPXF` (Cappellari 2017) and equivalent tools -- nothing novel here, this
module just makes it a reusable, instrument-agnostic utility rather than
something hand-rolled per script.

**A known, deliberately-not-yet-addressed gap: non-Gaussian LSF shape.**
Simon (2026, arXiv:2606.30129) shows that even when FWHM is matched,
genuine LSF *shape* mismatch (real instrument LSFs, including MUSE's, are
documented as measurably non-Gaussian/"boxy" rather than clean Gaussians)
can still bias h3/h4 by up to +/-0.1, and describes a more general
Fourier-domain matching method (public code:
github.com/dsimon45/LSF_Matching) for that case. Two reasons this
module does *not* implement that method yet: (1) it requires the
template library's own *measured* LSF shape, which the same paper notes
is "typically not measured" for stellar libraries including MILES/E-MILES
-- there is nothing to match against yet, only an assumed-Gaussian
approximation; (2) per this project's own working rule, don't build a
more complex mechanism before the simpler one has been tried and shown
insufficient. If FWHM-only matching leaves a real, measurable h3/h4 bias
on real data, this is the documented next step, not a hypothetical one.

**Instrument LSF built-ins.** `muse_lsf_fwhm` below is MUSE's own
published wavelength-dependent LSF FWHM curve, fit as a low-order
polynomial. Treat it as an approximation good to distinguish "obviously
matched" from "obviously mismatched" -- confirm against the specific
MUSE mode/pipeline version in use (WFM vs. NFM, pipeline data-reduction
version) before leaning on it for a precision correction.
"""
from __future__ import annotations

import numpy as np

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def fwhm_to_sigma(fwhm):
    """Convert a Gaussian FWHM to its standard deviation (same units)."""
    return np.asarray(fwhm, dtype=float) * _FWHM_TO_SIGMA


def muse_lsf_fwhm(wavelength):
    """MUSE's instrumental LSF FWHM (Angstrom) as a function of
    wavelength (Angstrom), via the published wavelength-dependent
    polynomial fit (roughly 3.0 Angstrom at 4800 Angstrom, a minimum of
    about 2.47 Angstrom near 7800-8500 Angstrom, rising again toward the
    red end). See this module's docstring for the caveats on precision.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    return 5.866e-8 * wavelength ** 2 - 9.187e-4 * wavelength + 6.040


def _as_fwhm_array(fwhm_or_func, wavelength):
    if callable(fwhm_or_func):
        return np.asarray(fwhm_or_func(wavelength), dtype=float)
    return np.full_like(wavelength, float(fwhm_or_func), dtype=float)


def variable_gaussian_convolve(wavelength, flux, sigma):
    """Convolve `flux` (sampled on a uniform `wavelength` grid) with a
    Gaussian kernel whose standard deviation `sigma` (Angstrom, same
    length as `wavelength`) varies per pixel. Pixels with `sigma <= 0`
    pass through unchanged (no-op, not an error) -- this is the expected
    case wherever `cannot_sharpen` is True in `match_gaussian_lsf`.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(wavelength)
    dx = wavelength[1] - wavelength[0]
    sigma_pix = sigma / dx

    max_sigma_pix = np.max(sigma_pix) if n > 0 else 0.0
    if max_sigma_pix <= 0:
        return flux.copy()

    half_width = int(np.ceil(max_sigma_pix * 4.0))
    offsets = np.arange(-half_width, half_width + 1)
    safe_sigma_pix = np.where(sigma_pix <= 0, np.inf, sigma_pix)
    kernel = np.exp(-0.5 * (offsets[None, :] / safe_sigma_pix[:, None]) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)

    flux_padded = np.pad(flux, half_width, mode="edge")
    out = np.zeros(n)
    for j, shift in enumerate(offsets):
        out += kernel[:, j] * flux_padded[half_width + shift: half_width + shift + n]

    passthrough = sigma_pix <= 0
    out[passthrough] = flux[passthrough]
    return out


def match_gaussian_lsf(wavelength, flux, template_fwhm, target_fwhm):
    """Convolve a template spectrum so its effective LSF FWHM goes from
    `template_fwhm` to `target_fwhm` (both a constant float or a
    callable of wavelength, in Angstrom), assuming both LSFs are
    Gaussian. See module docstring for the method and its limits.

    Returns `(matched_flux, cannot_sharpen)`: `cannot_sharpen` is a
    boolean array, True wherever `target_fwhm < template_fwhm` at that
    wavelength -- those pixels are passed through unmodified because a
    template cannot be sharpened by convolution (equal FWHM is not
    flagged: nothing needs correcting there). Inspect this before
    trusting the result if it is True over a wavelength range that
    matters for the fit.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    fwhm_tem = _as_fwhm_array(template_fwhm, wavelength)
    fwhm_tar = _as_fwhm_array(target_fwhm, wavelength)

    fwhm_diff_sq = fwhm_tar ** 2 - fwhm_tem ** 2
    cannot_sharpen = fwhm_diff_sq < 0
    fwhm_diff = np.sqrt(np.maximum(fwhm_diff_sq, 0.0))
    sigma_diff = fwhm_to_sigma(fwhm_diff)
    sigma_diff[cannot_sharpen] = 0.0

    matched = variable_gaussian_convolve(wavelength, flux, sigma_diff)
    return matched, cannot_sharpen
