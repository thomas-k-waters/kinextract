"""Forward-model construction: turn one or more fixed stellar templates
into the design-matrix stack a LOSVD fit solves against.

    f_model(lambda) = sum_k w_k * sum_j b_j * T_k(lambda shifted by v_j)

Doppler-shifting and LOSVD convolution are both linear in the template, so
for a fixed set of weights `w` the mixture's effective design matrix is
`sum_k w_k * A_stack[k]` -- see `kinextract.multi_template.fit_losvd`,
which optimizes `w` jointly with the LOSVD rather than requiring it fixed.
A single template is simply `build_template_stack` called with a length-1
list.
"""
from __future__ import annotations

import numpy as np

C_KMS = 299792.458  # speed of light, km/s


def build_velocity_grid(vmin: float, vmax: float, n_bins: int) -> np.ndarray:
    """Uniformly spaced LOSVD velocity-bin centers spanning [vmin, vmax]."""
    return np.linspace(vmin, vmax, n_bins)


def _lanczos_kernel(x: np.ndarray, a: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.sinc(x) * np.sinc(x / a)
    out[np.abs(x) >= a] = 0.0
    return out


def _shift_template_lanczos(wave_obs: np.ndarray, template_wave: np.ndarray,
                            template_flux: np.ndarray, v_kms: float, a: int) -> np.ndarray:
    """Band-limited (Lanczos-windowed sinc) alternative to plain linear
    interpolation -- see `shift_template`'s own `method` docstring for why
    this exists. Requires `template_wave` to be uniformly sampled (not
    checked here for speed; `shift_template` checks once per call)."""
    dx = template_wave[1] - template_wave[0]
    lam_rest = wave_obs / (1.0 + v_kms / C_KMS)
    idx = (lam_rest - template_wave[0]) / dx
    idx_floor = np.floor(idx).astype(int)
    n = len(template_wave)
    out = np.zeros_like(lam_rest)
    for t in range(-a + 1, a + 1):
        k = idx_floor + t
        valid = (k >= 0) & (k < n)
        w = _lanczos_kernel(idx - k, a)
        out[valid] += template_flux[k[valid]] * w[valid]
    out_of_range = (lam_rest < template_wave[0]) | (lam_rest > template_wave[-1])
    out[out_of_range] = 0.0
    return out


def shift_template(wave_obs: np.ndarray, template_wave: np.ndarray, template_flux: np.ndarray,
                   v_kms: float, method: str = "linear", lanczos_a: int = 3) -> np.ndarray:
    """Evaluate a template, Doppler-shifted by `v_kms`, on the observed
    wavelength grid. A star moving at velocity v contributes flux at
    `lambda_obs` that corresponds to the template's own rest-frame
    `lambda_obs / (1 + v/c)`. Out-of-coverage pixels are filled with zero.

    `method` : `"linear"` (default, unchanged from every version of this
        package before this parameter existed) uses `np.interp` --
        piecewise-linear interpolation. This has a real, measured defect:
        linear interpolation of a band-limited signal is not itself
        band-limited, and its error is periodic in the *fractional pixel
        shift*, with period equal to the template's own native sampling.
        For the shipped E-MILES grid (0.9 Angstrom/pix native sampling) at
        ~8500 Angstrom, that period is `c*0.9/8500 ~ 31.7 km/s` -- so the
        interpolation error itself varies systematically as a function of
        the (arbitrary, target-redshift-dependent) velocity offset between
        a real spectrum and the template grid, not just noise. Measured
        directly (`dev_notes/shift_template/FINDINGS.md`): max relative
        error swings 0.7-1.2% of peak amplitude across one 31.7 km/s
        period, and this leaks into recovered kinematics as a genuine,
        fractional-shift-dependent effective broadening (0-17 km/s on real
        MUSE data), invisible to synthetic-mock validation because mocks
        are conventionally built with this exact same design matrix (the
        interpolation "error" cancels against itself by construction).

        `"lanczos"` uses a windowed-sinc (Lanczos) kernel instead --
        requires `template_wave` to be uniformly sampled (checked; raises
        `ValueError` if not). Measured to remove the *periodic* component
        of the interpolation error (the part that varies systematically
        with velocity shift and is therefore the part that can masquerade
        as kinematic broadening) -- what remains is a smaller,
        phase-independent residual (real spectra are not perfectly
        band-limited, so no resampling kernel reproduces them exactly),
        which does not have the same shift-dependent-bias failure mode.
        Costs `2*lanczos_a` times the work of linear interpolation (still
        fully vectorized, no per-pixel Python loop) -- real but modest
        given this function's own cost is a small fraction of a fit's
        total time. Opt-in, not the default: changes the numerical output
        of every existing fit by a small amount, and hasn't yet been
        validated at the scale linear interpolation has (this package's
        entire existing calibration/validation history was built with
        `method="linear"`).
    `lanczos_a` : kernel half-width in native-template-pixel units, only
        used when `method="lanczos"`. Default 3 (a standard choice:
        strong stopband rejection without excessive cost or ringing).
    """
    if method == "linear":
        lam_rest = wave_obs / (1.0 + v_kms / C_KMS)
        return np.interp(lam_rest, template_wave, template_flux, left=0.0, right=0.0)
    elif method == "lanczos":
        dx = np.diff(template_wave)
        if not np.allclose(dx, dx[0], rtol=1e-6):
            raise ValueError(
                "shift_template: method='lanczos' requires a uniformly-sampled "
                "template_wave (band-limited resampling assumes fixed native "
                "spacing) -- resample your template library onto a uniform grid first."
            )
        return _shift_template_lanczos(wave_obs, template_wave, template_flux, v_kms, lanczos_a)
    else:
        raise ValueError(f"shift_template: unknown method={method!r}, expected 'linear' or 'lanczos'")


def build_template_stack(wave_obs: np.ndarray, template_wave: np.ndarray,
                          template_fluxes: list[np.ndarray], v_grid: np.ndarray,
                          method: str = "linear", lanczos_a: int = 3) -> np.ndarray:
    """Design-matrix stack, one `(n_pix, n_bins)` matrix per entry in
    `template_fluxes` (all sharing the same rest-frame `template_wave`
    sampling -- resample onto a common grid first if your library doesn't
    already share one); column `j` of each matrix is that template
    Doppler-shifted to `v_grid[j]`.

    `method`/`lanczos_a` : passed through to `shift_template` -- see its
    own docstring for the linear-interpolation defect this exists to fix,
    and why `"lanczos"` is opt-in rather than the default.

    Returns an array of shape `(n_templates, n_pix, n_bins)`, the input
    `A_stack` for `kinextract.multi_template.fit_losvd` (works for
    any `n_templates >= 1`, including a single template via a length-1
    `template_fluxes`).
    """
    n_pix = len(wave_obs)
    n_bins = len(v_grid)
    stack = np.empty((len(template_fluxes), n_pix, n_bins))
    for k, template_flux in enumerate(template_fluxes):
        for j, v in enumerate(v_grid):
            stack[k, :, j] = shift_template(wave_obs, template_wave, template_flux, v,
                                            method=method, lanczos_a=lanczos_a)
    return stack


def mask_wavelength_windows(wave: np.ndarray, *arrays: np.ndarray,
                             windows: list[tuple[float, float]]) -> tuple[np.ndarray, ...]:
    """Drop pixels inside any of `windows` (a list of `(wave_min, wave_max)`
    pairs, inclusive) from `wave` and every array passed alongside it (e.g.
    flux, per-pixel error) -- for excluding known bad-pixel/sky-residual/
    cosmic-ray wavelength ranges *before* building a design matrix or
    fitting, not as a fit-time weighting scheme. Deliberately simple and
    general: callers supply their own `windows` from whatever they've
    identified in their own data.

    Parameters
    ----------
    wave : wavelength array to filter and use as the reference for `windows`.
    *arrays : any number of same-length arrays (flux, errors, ...) to filter identically.
    windows : list of `(wave_min, wave_max)` pairs; a pixel is dropped if
        it falls inside *any* window (inclusive on both ends).

    Returns
    -------
    `(wave_filtered, *arrays_filtered)` -- same number and order of
    arrays as passed in, each with the masked pixels removed.
    """
    wave = np.asarray(wave, float)
    keep = np.ones(len(wave), dtype=bool)
    for wmin, wmax in windows:
        keep &= ~((wave >= wmin) & (wave <= wmax))
    return (wave[keep],) + tuple(np.asarray(a)[keep] for a in arrays)
