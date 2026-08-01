"""Tests for kinextract.gh_moments.

See `gh_moments.py`'s own module docstring for the full story: a
core-windowed direct moment for V/sigma was validated against pure
Gaussian truth, looked like a clean win, then appeared to fail badly on
skewed (h3!=0) truth. Two distinct things were going on, and both matter
for how these tests are written:

1. A real bug: a *hard* window (b > threshold) is itself asymmetric
   about the true center for a skewed density, biasing the windowed
   moment. Fixed by using a smooth logistic taper on the same axis
   instead of a hard cutoff.
2. A real, separate subtlety this project's own comparisons must respect:
   for a Gauss-Hermite density with h3/h4 != 0, the *true* first/second
   moment of the distribution is NOT equal to the model's own `vel`/`sig`
   input parameters (those are constructed to decouple from h3/h4 only
   at leading order, not exactly) -- e.g. vel=0, h3=+0.1, sig=140 has a
   numerically-integrated true mean of about +20.6 km/s, not 0. Any test
   or benchmark comparing an estimator's accuracy on skewed truth MUST
   compare against the numerically-integrated true moment, not the raw
   input parameter, or it will manufacture a ~20 km/s "bias" out of
   nothing. `_true_moments_numeric` below exists for exactly this.

**Deliberately kept on `gauss_hermite_model` truth, not the package-wide
`build_spline_truth` default (2026-07-29).** This module tests
`gauss_hermite_model`/`fit_gauss_hermite`/`compute_losvd_moments`
themselves -- the GH parameterization and its documented moments gap ARE
the subject under test here, not an incidental mock-generation choice, so
there is nothing to swap.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract import (
    FitConfig, build_template_stack, gauss_hermite_model,
    fit_gauss_hermite, compute_losvd_moments, compute_direct_moments,
    shape_aware_taper_correction,
)
from kinextract.bias_correction import BiasCalibration
from kinextract.multi_template import fit_losvd


def _true_moments_numeric(vel, sig, h3, h4, vmin, vmax, n=400001):
    """Exact (to numerical precision) first/second moments of the actual
    continuous Gauss-Hermite density, via fine trapezoidal quadrature --
    NOT assumed equal to the `vel`/`sig` input parameters, which only
    approximate the true moments at leading order in h3/h4.
    """
    v = np.linspace(vmin, vmax, n)
    d = np.maximum(gauss_hermite_model(v, 1.0, vel, sig, h3, h4), 0.0)
    total = np.trapezoid(d, v)
    true_vel = np.trapezoid(d * v, v) / total
    true_sig = np.sqrt(np.trapezoid(d * (v - true_vel) ** 2, v) / total)
    return true_vel, true_sig


def _mock_case(vel, sig, h3, h4, nsig=7, snr=100.0, seed=0):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    ca_centers = [8498.02, 8542.09, 8662.14]
    ca_depths = [0.55, 0.70, 0.65]
    template_flux = np.ones(n_pix)
    for cen, depth in zip(ca_centers, ca_depths):
        template_flux -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)

    vmin, vmax = vel - nsig * sig, vel + nsig * sig
    cfg = FitConfig(vmin=vmin, vmax=vmax)
    v_grid = cfg.build_velocity_grid()
    b_truth = gauss_hermite_model(v_grid, amp=1.0, vel=vel, sig=sig, h3=h3, h4=h4)
    b_truth = np.maximum(b_truth, 0.0)
    b_truth /= b_truth.sum()

    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w, y_w = A_stack / errors[None, :, None], galaxy / errors

    result = fit_losvd(A_stack_w, y_w, cfg)
    return v_grid, result["b"], vmin, vmax


def test_compute_losvd_moments_matches_gh_fit_on_gaussian_truth():
    """Pure Gaussian (h3=h4=0) truth -- the true moment equals the input
    parameter exactly here, so this is a simple case either way.
    """
    vel, sig = 0.0, 140.0
    v_grid, b, *_ = _mock_case(vel, sig, 0.0, 0.0, seed=0)

    new = compute_losvd_moments(v_grid, b)
    old = fit_gauss_hermite(v_grid, b, fit_h3h4=True)

    assert abs(new["vherm"] - vel) < 10.0
    assert abs(old["vherm"] - vel) < 10.0
    assert abs(new["sherm"] - sig) < 15.0
    assert abs(old["sherm"] - sig) < 15.0


def test_gh_vel_parameter_is_not_the_true_mean_when_skewed():
    """Documents the subtlety every other test here has to account for:
    the Gauss-Hermite `vel` parameter is not the distribution's actual
    mean once h3 is non-negligible. If this ever starts failing, every
    other skewed-truth comparison in this file needs re-deriving, not
    just this one assertion.
    """
    true_vel, true_sig = _true_moments_numeric(0.0, 140.0, 0.1, 0.0, -980.0, 980.0)
    assert abs(true_vel - 0.0) > 15.0, "expected the input vel=0 parameter to diverge from the true mean"


def test_smooth_taper_fixes_the_skewed_case_a_hard_window_broke():
    """`compute_direct_moments`'s smooth logistic taper must stay
    accurate on skewed (h3!=0) truth -- compared against the true
    numerically-integrated moment, not the raw `vel` input parameter
    (see this module's own docstring for why that distinction matters).
    A *hard*-window version of this same function was confirmed to show
    a large bias *relative to the true moment* here; the fix was
    smoothing the window into a logistic taper, not abandoning direct
    moments. If this regresses, check `taper_width` wasn't set to
    something that behaves like a step function.
    """
    vel, sig, h3 = 0.0, 140.0, 0.1
    v_grid, b, vmin, vmax = _mock_case(vel, sig, h3, 0.0, seed=0)
    true_vel, true_sig = _true_moments_numeric(vel, sig, h3, 0.0, vmin, vmax)

    direct = compute_direct_moments(v_grid, b)
    assert abs(direct["vherm"] - true_vel) < 8.0
    assert abs(direct["sherm"] - true_sig) < 15.0


def test_direct_moments_beat_gh_fit_on_the_standard_validation_grid():
    """Confirms `compute_direct_moments` is not just "good enough" but a
    genuine improvement over `fit_gauss_hermite` at recovering the LOSVD's
    actual moments, averaged over this project's standard (V, sigma) grid
    plus moderate h3/h4 -- not a single cherry-picked case. Both
    estimators are scored against the true numerically-integrated moment,
    the physically meaningful quantity, not against `fit_gauss_hermite`'s
    own `vel`/`sig` input convention (which the previous test shows isn't
    the same thing once h3/h4 are non-negligible).
    """
    nsig = 7
    cases = [(-200, 50, 0.0, 0.0), (0, 140, 0.0, 0.0), (200, 300, 0.0, 0.0),
             (0, 140, 0.1, 0.0), (0, 140, -0.1, 0.0), (0, 140, 0.0, 0.1), (0, 140, 0.0, -0.1)]

    gh_errs, direct_errs = [], []
    for vel, sig, h3, h4 in cases:
        true_vel, true_sig = _true_moments_numeric(vel, sig, h3, h4, vel - nsig * sig, vel + nsig * sig)
        for seed in range(3):
            v_grid, b, *_ = _mock_case(vel, sig, h3, h4, nsig=nsig, seed=seed)
            gh = fit_gauss_hermite(v_grid, b, fit_h3h4=True)
            direct = compute_direct_moments(v_grid, b)
            gh_errs.append(abs(gh["vherm"] - true_vel) + abs(gh["sherm"] - true_sig))
            direct_errs.append(abs(direct["vherm"] - true_vel) + abs(direct["sherm"] - true_sig))

    assert np.mean(direct_errs) < np.mean(gh_errs)


def test_shape_aware_correction_is_exact_on_noiseless_truth():
    """The correction is a purely geometric self-consistency fix (build a
    reference curve with the given shape, measure the taper's own bias on
    it, undo it) -- so applied to a noiseless curve using its own true
    shape, it should recover the true moments essentially exactly. This
    is the mechanism-correctness check, independent of whether it helps
    on noisy fits (that's the next test).
    """
    for vel, sig, h3, h4 in [(0.0, 140.0, 0.0, 0.0), (0.0, 140.0, 0.1, 0.0), (0.0, 140.0, 0.0, 0.1)]:
        v_grid = np.linspace(vel - 15 * sig, vel + 15 * sig, 200000)
        b_truth = np.maximum(gauss_hermite_model(v_grid, 1.0, vel, sig, h3, h4), 0.0)
        direct = compute_direct_moments(v_grid, b_truth)
        true_vel, true_sig = _true_moments_numeric(vel, sig, h3, h4, v_grid.min(), v_grid.max())

        factor, v_shift = shape_aware_taper_correction(direct["vherm"], direct["sherm"], h3, h4)
        corrected_sig = direct["sherm"] * factor
        corrected_vel = direct["vherm"] + v_shift

        assert abs(corrected_sig - true_sig) < 0.05, f"shape=({h3},{h4}): corrected sigma should match truth closely"
        assert abs(corrected_vel - true_vel) < 0.05, f"shape=({h3},{h4}): corrected V should match truth closely"


def test_shape_aware_correction_improves_the_h4_case_it_was_built_for():
    """The case that motivated this: h4=+0.1 has the largest windowing
    bias in the standard validation grid (~-1.4% for a Gaussian-adjacent
    shape, confirmed in `shape_aware_taper_correction`'s own docstring).
    On noisy fits, the correction should measurably reduce that bias for
    this specific case -- this is the targeted fix working, not just the
    noiseless mechanism check above. (Not tested here: sigma=300, where
    this correction is documented to make things worse -- see that
    function's own docstring for why, and don't use it there yet.)
    """
    vel, sig, h4 = 0.0, 140.0, 0.1
    true_vel, true_sig = _true_moments_numeric(vel, sig, 0.0, h4, vel - 15 * sig, vel + 15 * sig)

    raw_errs, corrected_errs = [], []
    for seed in range(8):
        v_grid, b, *_ = _mock_case(vel, sig, 0.0, h4, seed=seed)
        raw = compute_losvd_moments(v_grid, b, shape_aware_correction=False)
        corrected = compute_losvd_moments(v_grid, b, shape_aware_correction=True)
        raw_errs.append(abs(raw["sherm"] - true_sig))
        corrected_errs.append(abs(corrected["sherm"] - true_sig))

    assert np.mean(corrected_errs) < np.mean(raw_errs), (
        f"expected the shape-aware correction to reduce mean |sigma error| for h4=+0.1, "
        f"got raw={np.mean(raw_errs):.2f} corrected={np.mean(corrected_errs):.2f}"
    )


def test_compute_losvd_moments_bias_correction_clips_negative_model():
    """Companion to `test_bias_correction.py`'s equivalent regression
    test -- `compute_losvd_moments(regularization_bias_correction=True)`
    is a second call site that reconstructs a Gauss-Hermite curve from
    the corrected (sherm, h4) and must clip it non-negative for the same
    reason (see `bias_correction.correct_losvd`'s docstring). Uses an
    extreme synthetic calibration to force the corrected h4 into the
    known-negative regime deterministically.
    """
    extreme_calib = BiasCalibration(
        recovered_sigma=np.array([50.0, 50.0, 200.0, 200.0]),
        snr=np.array([10.0, 200.0, 10.0, 200.0]),
        sig_bias=np.zeros(4),
        sig_bias_std=np.ones(4),
        h4_bias=np.array([0.2, 0.2, 0.2, 0.2]),
        h4_bias_std=np.full(4, 0.05),
        sigma_range=(50.0, 200.0),
        snr_range=(10.0, 200.0),
    )
    vel, sig = 0.0, 100.0
    v_grid, b, *_ = _mock_case(vel, sig, 0.0, 0.0, seed=0)
    out = compute_losvd_moments(v_grid, b, regularization_bias_correction=True, snr=100.0,
                                 bias_calibration=extreme_calib)
    assert out["h4"] == pytest.approx(out["h4_raw"] - 0.2)
    assert np.all(out["model"] >= 0.0), (
        "compute_losvd_moments's corrected model must be clipped non-negative -- a LOSVD is a "
        "probability density over velocity and cannot be negative anywhere"
    )
