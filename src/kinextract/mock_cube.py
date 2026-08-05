"""Synthetic MUSE-like IFU cube generator: a Sersic-profile elliptical
galaxy with a known, spatially-constant LOSVD and template mixture,
independent Gaussian pixel noise, and S-N that declines with radius
purely because the galaxy's own surface brightness fades while the noise
floor stays fixed -- exactly what a real background/detector-noise-
dominated observation does, without needing a separate photon-noise
model.

**What this is for.** `binning.py`'s dartboard scheme and
`template_reduction.py`'s iterative template-selection loop both need a
target where the *right answer is known* to check that they behave
sensibly -- does the iterative reduction converge to a non-trivial
template subset (not collapse to a single template), and does chi^2 (or
recovered bias) actually plateau as more templates are retained, rather
than improving forever. Real data can't answer either question, since
there's no ground truth to check against.

**Deliberately minimal kinematics.** `w_true`/`b_true` are the same at
every spaxel -- this mock's job is exercising the *spatial binning and
template-selection* machinery, not re-litigating kinematic recovery
across S-N and sigma, which `01_basic_mock_fit.ipynb`/
`02_realistic_mock_fit.ipynb` already cover thoroughly. A future version
could add a rotation/dispersion gradient once that specific question
needs answering.

**Deliberately independent Gaussian noise.** Real MUSE pixel noise does
show measurable lag-1 correlation (see `noise_model.py`) -- modeling that
here would conflate two separate validation questions. This mock's own
noise model matches `01_basic_mock_fit.ipynb`/`02_realistic_mock_fit.ipynb`'s
own simple-noise convention, not `05_real_data_muse.ipynb`'s measured
correlated case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.modeling.functional_models import Sersic2D

from .config import FitConfig
from .emiles import load_emiles_templates
from .forward_model import build_template_stack
from .gh_moments import gauss_hermite_model, compute_losvd_moments
from .losvd import build_spline_truth
from .muse_cube import MuseCube

_DEFAULT_TEMPLATE_SELECT = [(1.0, -0.66), (5.0, 0.06), (10.0, 0.06)]
_DEFAULT_TRUE_W = np.array([0.2, 0.3, 0.5])  # young -> old, weighted toward the old population

WAVEMIN, STEP = 4750.0, 1.25  # MUSE pixel-sampling convention, matching every other notebook/mock in this package
WAVEFITMIN, WAVEFITMAX = 8400.0, 8800.0  # CaII triplet region -- this mock stays in-window, no full-range spectrum needed


def build_mock_cube(emiles_grid_path: str, ny: int = 50, nx: int = 50, pixel_scale: float = 0.2,
                     r_eff_arcsec: float = 3.0, sersic_n: float = 4.0, amplitude: float = 1.0,
                     noise_floor: float = 2.0,
                     true_v: float = 20.0, true_sigma: float = 80.0,
                     truth_kind: str = "spline", wiggle_scale: float = 0.15,
                     true_h3: float = 0.0, true_h4: float = 0.0,
                     template_select: list[tuple[float, float]] | None = None,
                     true_w: np.ndarray | None = None, seed: int = 0) -> MuseCube:
    """Build one synthetic MUSE-like cube: a circular (no PA/ellipticity)
    Sersic surface-brightness profile, modulating a single fixed
    `(true_w, LOSVD)` spectrum at every spaxel, plus independent Gaussian
    noise at a constant floor.

    Parameters
    ----------
    emiles_grid_path : path to the packed E-MILES grid (see `emiles.py`).
    ny, nx : spatial shape.
    pixel_scale : arcsec/pixel (MUSE WFM's own default sampling is
        0.2"/pixel).
    r_eff_arcsec, sersic_n, amplitude : `astropy.modeling.functional_models.
        Sersic2D` parameters -- `sersic_n=4` (de Vaucouleurs) is a
        typical elliptical-galaxy profile. `amplitude` is in the same
        arbitrary flux units as `noise_floor`; only their *ratio* (which
        sets the central S-N) is physically meaningful here.
    noise_floor : constant per-pixel-per-wavelength Gaussian sigma. Since
        the Sersic profile's own surface brightness declines with radius
        while this stays fixed, S-N declines with radius automatically.
    true_v, true_sigma : the LOSVD's location/scale, used at every spaxel
        (see module docstring for why this is spatially constant) --
        apply to both `truth_kind`s.
    truth_kind : `"spline"` (default) builds the LOSVD directly as
        `exp(spline)` via `build_spline_truth` -- no Gauss-Hermite step,
        so there is no gap between the generating parameters and the
        curve's own moments (see `build_spline_truth`'s docstring, and
        `dev_notes/validation_suite/FINDINGS.md`, for why that gap is
        large and exact once h3/h4 leave a narrow neighborhood of zero
        under the `"gh"` alternative). `"gh"` reproduces this function's
        original behavior (a truncated Gauss-Hermite series via
        `gauss_hermite_model`, using `true_h3`/`true_h4`) -- kept as an
        explicit opt-in for anything that specifically needs to validate
        against that parameterization (e.g. Gauss-Hermite-fitting code
        itself), not as a general-purpose mock truth going forward.
    wiggle_scale : only used when `truth_kind="spline"` -- the spline
        core's random log-space perturbation size (see
        `build_spline_truth`). `wiggle_scale=0.0` gives a pure Gaussian
        core (the spline-truth analog of `true_h3=true_h4=0`), so h3/h4
        recovery is still meaningful to check against a near-zero-shape
        truth.
    true_h3, true_h4 : only used when `truth_kind="gh"`; ignored (must be
        left at 0.0) when `truth_kind="spline"`, since a spline truth has
        no GH shape parameters to set.
    template_select, true_w : the injected template mixture (default:
        3 real E-MILES SSPs spanning young to old, weighted toward the
        old population, a plausible elliptical-galaxy mix). `true_w` is
        renormalized to sum to 1 internally.
    seed : noise-draw seed; also seeds the `truth_kind="spline"` wiggle
        (kept separate from the noise RNG stream internally, so changing
        `noise_floor` or re-drawing noise never perturbs the injected
        shape).

    Returns
    -------
    A `MuseCube` with `truth` populated: `w_true` (normalized), `b_true`
    (the LOSVD histogram), `v_grid`, `template_select`, `template_names`,
    `wave_fit` (== `cube.wave`), `truth_kind`, `truth_moments` (the
    curve's own `compute_losvd_moments`, the correct basis for scoring
    recovery bias against -- see `build_spline_truth`'s docstring for why
    this is not simply `(true_v, true_sigma, true_h3, true_h4)`), and the
    Sersic parameters used.
    """
    if truth_kind not in ("spline", "gh"):
        raise ValueError(f"unknown truth_kind={truth_kind!r}, expected 'spline' or 'gh'")
    if truth_kind == "spline" and (true_h3 != 0.0 or true_h4 != 0.0):
        raise ValueError("true_h3/true_h4 have no meaning for truth_kind='spline' (use wiggle_scale instead)")

    template_select = template_select if template_select is not None else _DEFAULT_TEMPLATE_SELECT
    true_w = np.asarray(true_w if true_w is not None else _DEFAULT_TRUE_W, dtype=float)
    true_w = true_w / true_w.sum()

    templates, template_wave, names, ages, metals = load_emiles_templates(emiles_grid_path, select=template_select)
    n_templates = len(templates)

    observed_wavelength = WAVEMIN + np.arange(int((10500 - WAVEMIN) / STEP)) * STEP
    wave_fit = observed_wavelength[(observed_wavelength >= WAVEFITMIN) & (observed_wavelength <= WAVEFITMAX)]

    cfg = FitConfig(vmin=true_v - 7 * true_sigma, vmax=true_v + 7 * true_sigma)
    v_grid = cfg.build_velocity_grid()
    if truth_kind == "spline":
        # a distinct RNG stream from the noise draw below, so re-seeding one never perturbs the other.
        wiggle_rng = np.random.default_rng(1_000_003 * seed + 7)
        b_true, _coeffs = build_spline_truth(v_grid, vel=true_v, sigma=true_sigma,
                                              wiggle_scale=wiggle_scale, seed=wiggle_rng)
    else:
        b_true = np.maximum(
            gauss_hermite_model(v_grid, amp=1.0, vel=true_v, sig=true_sigma, h3=true_h3, h4=true_h4), 0.0)
        b_true = b_true / np.trapezoid(b_true, v_grid)
    truth_moments = compute_losvd_moments(v_grid, b_true, fit_h3h4=True)

    # `b_true` above is unit-AREA (trapz) -- the correct convention for reporting moments,
    # and `build_spline_truth`'s own contract. But the noiseless-spectrum amplitude below is
    # compared against an absolute (not S/N-relative) `noise_floor`, so switching truth_kind
    # or the velocity grid's bin spacing must not silently rescale the mock's S/N regime.
    # Building the flux from a unit-SUM curve instead keeps that regime fixed regardless of
    # `truth_kind`, `true_sigma`, or `bin_spacing_kms` -- matching this function's original,
    # long-standing (pre-2026-07-29) `gauss_hermite_model` behavior exactly.
    b_true_flux = b_true / b_true.sum()

    A_stack_true = build_template_stack(wave_fit, template_wave, templates, v_grid)
    spectrum_shape = sum(true_w[k] * (A_stack_true[k] @ b_true_flux) for k in range(n_templates))  # (n_pix,)

    y, x = np.mgrid[0:ny, 0:nx]
    r_eff_pix = r_eff_arcsec / pixel_scale
    sersic = Sersic2D(amplitude=amplitude, r_eff=r_eff_pix, n=sersic_n, x_0=nx / 2.0, y_0=ny / 2.0, ellip=0.0, theta=0.0)
    surface_brightness = sersic(x, y)  # (ny, nx)

    flux_noiseless = surface_brightness[None, :, :] * spectrum_shape[:, None, None]  # (n_pix, ny, nx)
    rng = np.random.default_rng(seed)
    flux = flux_noiseless + rng.normal(0.0, noise_floor, size=flux_noiseless.shape)
    error = np.full_like(flux, noise_floor)

    truth = dict(
        w_true=true_w, b_true=b_true, v_grid=v_grid, template_select=template_select,
        template_names=names, template_ages=ages, template_metals=metals, wave_fit=wave_fit,
        truth_kind=truth_kind, wiggle_scale=wiggle_scale if truth_kind == "spline" else None,
        true_v=true_v, true_sigma=true_sigma, true_h3=true_h3, true_h4=true_h4,
        truth_moments=truth_moments,
        r_eff_arcsec=r_eff_arcsec, sersic_n=sersic_n, amplitude=amplitude, noise_floor=noise_floor,
        surface_brightness=surface_brightness,
    )
    return MuseCube(flux=flux, error=error, wave=wave_fit, pixel_scale=pixel_scale, wcs=None, truth=truth)
