"""Sanity checks for kinextract's LOSVD recovery -- does it recover
a known, clean synthetic Gaussian LOSVD from a well-conditioned linear
system, using the package's own current defaults (not values pinned to
an old configuration -- these should track `FitConfig`'s defaults, not
drift away from them).

Not a substitute for the full validation notebooks (real templates, real
noise, a full synthetic grid) -- just confirms the current defaults still
do what they're supposed to on an easy, self-contained case.
"""
from __future__ import annotations

import numpy as np

from kinextract import FitConfig, build_template_stack, build_spline_truth, fit_gauss_hermite
from kinextract.multi_template import fit_losvd


def _gaussian(v, vel, sig):
    return np.exp(-0.5 * ((v - vel) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)


def test_recovers_clean_gaussian():
    cfg = FitConfig(vmin=-300.0, vmax=300.0)
    v_grid = cfg.build_velocity_grid()
    true_v, true_sig = 10.0, 60.0
    true_b = _gaussian(v_grid, vel=true_v, sig=true_sig)
    true_b /= true_b.sum()

    n_pix = 400
    # A well-conditioned forward operator: a bank of smooth, overlapping
    # kernels (stand-in for template x LOSVD convolution), not raw noise,
    # so the recovered b is meaningfully comparable to true_b.
    centers = np.linspace(v_grid.min(), v_grid.max(), n_pix)
    A = np.array([_gaussian(v_grid, vel=c, sig=15.0) for c in centers])

    rng = np.random.default_rng(0)
    noiseless = A @ true_b
    noise_level = 0.01 * noiseless.std()
    y = noiseless + rng.normal(scale=noise_level, size=n_pix)
    err = np.full(n_pix, noise_level)
    A_w = A / err[:, None]
    y_w = y / err

    result = fit_losvd(A_w[np.newaxis], y_w, cfg)

    assert result["success"]
    assert np.all(result["b"] > 0.0)

    recovered_v = float(np.sum(v_grid * result["b"]) / np.sum(result["b"]))
    assert abs(recovered_v - true_v) < 10.0

    b_true_n = true_b / np.trapezoid(true_b, v_grid)
    b_fit_n = result["b"] / np.trapezoid(result["b"], v_grid)
    core = b_true_n > 0.01 * b_true_n.max()
    max_resid_pct = float(np.max(np.abs(b_fit_n[core] - b_true_n[core]) / b_true_n.max())) * 100
    assert max_resid_pct < 5.0


def test_knot_count_scales_with_grid_width():
    """A fixed n_interior_knots silently under-resolves the LOSVD once the
    velocity grid is widened for a broader target. The default
    (n_interior_knots=None) must auto-scale with vmin/vmax, not stay fixed.
    """
    narrow = FitConfig(vmin=-300.0, vmax=300.0)
    wide = FitConfig(vmin=-1600.0, vmax=1600.0)

    assert wide.resolved_n_interior_knots() > narrow.resolved_n_interior_knots()
    assert narrow.resolved_n_interior_knots() == narrow.min_interior_knots

    # An explicit override must still be honored, not silently replaced.
    overridden = FitConfig(vmin=-1600.0, vmax=1600.0, n_interior_knots=5)
    assert overridden.resolved_n_interior_knots() == 5


def test_n_bins_is_not_a_user_facing_field():
    """The velocity-grid point count is auto-derived, not a bare tunable
    field, and should scale with grid width the same way
    `resolved_n_interior_knots()` does.
    """
    assert not hasattr(FitConfig(), "n_bins")
    narrow = FitConfig(vmin=-300.0, vmax=300.0)
    wide = FitConfig(vmin=-1600.0, vmax=1600.0)
    assert wide.resolved_n_bins() > narrow.resolved_n_bins()
    assert narrow.resolved_n_bins() == narrow.min_bins
    assert len(narrow.build_velocity_grid()) == narrow.resolved_n_bins()


def _ca_mock(wavemin=4750.0, step=1.25, n_pix=3681, wavefitmin=8400.0, wavefitmax=8800.0):
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    ca_centers = [8498.02, 8542.09, 8662.14]
    ca_depths = [0.55, 0.70, 0.65]
    template_flux = np.ones(n_pix)
    for cen, depth in zip(ca_centers, ca_depths):
        template_flux -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)
    return wavelength, wave_fit, mask, template_flux


def test_multi_start_escapes_a_known_bad_local_minimum():
    """A specific noisy realization (CaII-triplet mock, V_true=80,
    sigma_true=240, seed=1) is a confirmed counter-example where the
    data-driven ridge start alone converges to a real, meaningfully worse
    local minimum while a flat start on the same data finds a much better
    one. Assert the package's actual default behavior (not a hand-rolled
    multi-start) lands in the good basin, not the bad one.
    """
    wavelength, wave_fit, mask, template_flux = _ca_mock()

    true_v, true_sig = 80.0, 240.0
    v_half = abs(true_v) + 5 * true_sig + 50.0
    cfg = FitConfig(vmin=-v_half, vmax=v_half, lam=10.0)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=true_v, sigma=true_sig, wiggle_scale=0.0, seed=0)

    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / 100.0
    rng = np.random.default_rng(1)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w, y_w = A_stack / errors[None, :, None], galaxy / errors

    result = fit_losvd(A_stack_w, y_w, cfg)
    gh = fit_gauss_hermite(v_grid, result["b"], fit_h3h4=True)

    assert abs(gh["vherm"] - true_v) < 10.0
    assert abs(gh["sherm"] - true_sig) < 15.0
    # losvd_objective_gap reflects fit_log_spline_map's own internal
    # multi-start (distinct from objective_gap, which is about the outer
    # w0_pool search and is None here since only one weight-vector start
    # is used) -- its presence confirms the LOSVD sub-fit's multi-start
    # actually ran and found more than one candidate to compare.
    assert result["losvd_objective_gap"] is not None


def test_curvature_penalty_scaled_by_velocity_spacing_removes_sigma_bias():
    """The curvature penalty is scaled by the grid's actual knot spacing
    (against `knot_spacing_kms`), so `lam` represents the same physical
    smoothing strength regardless of grid width. Confirm the mean sigma
    bias, averaged over several independent noise draws, is small relative
    to sigma at the default lam=2.0 -- not just that any single fit
    succeeds.
    """
    wavelength, wave_fit, mask, template_flux = _ca_mock()

    true_v, true_sig = 0.0, 140.0
    nsig = 7
    cfg = FitConfig(vmin=true_v - nsig * true_sig, vmax=true_v + nsig * true_sig)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=true_v, sigma=true_sig, wiggle_scale=0.0, seed=0)

    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / 100.0

    sigma_biases = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
        errors = np.full(len(wave_fit), noise_level)
        A_stack_w, y_w = A_stack / errors[None, :, None], galaxy / errors
        result = fit_losvd(A_stack_w, y_w, cfg)
        gh = fit_gauss_hermite(v_grid, result["b"], fit_h3h4=True)
        sigma_biases.append(gh["sherm"] - true_sig)

    # A single (V, sigma) case at ~10 seeds still carries real sampling
    # noise (per-seed std is ~3-5 km/s here) -- this threshold is set well
    # below what an unscaled penalty would show, not at a razor-thin exact
    # number.
    mean_bias = float(np.mean(sigma_biases))
    assert abs(mean_bias) < 5.0, f"mean sigma bias {mean_bias:.2f} km/s too large -- scaling regression?"
def test_non_finite_input_raises_instead_of_propagating_silently():
    """A single NaN or inf pixel in `y_w` (a very plausible real-data
    mistake -- a masked emission line or cosmic ray marked with NaN
    instead of excluded from the array) must fail loudly and specifically
    at the boundary, not propagate silently into the entire recovered `b`.
    """
    import pytest

    cfg = FitConfig(vmin=-980.0, vmax=980.0)
    v_grid = cfg.build_velocity_grid()
    true_b, _ = build_spline_truth(v_grid, vel=0.0, sigma=140.0, wiggle_scale=0.0, seed=0)
    n_pix = 200
    centers = np.linspace(v_grid.min(), v_grid.max(), n_pix)
    A = np.array([np.exp(-0.5 * ((v_grid - c) / 15.0) ** 2) for c in centers])
    y = A @ true_b
    err = np.full(n_pix, 0.01 * y.std())
    A_w, y_w = A / err[:, None], y / err
    A_stack_w = A_w[np.newaxis]

    y_w_nan = y_w.copy()
    y_w_nan[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_losvd(A_stack_w, y_w_nan, cfg)

    y_w_inf = y_w.copy()
    y_w_inf[10] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        fit_losvd(A_stack_w, y_w_inf, cfg)

    A_stack_w_nan = A_stack_w.copy()
    A_stack_w_nan[0, 5, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_losvd(A_stack_w_nan, y_w, cfg)


def test_narrow_grid_with_floored_knots_warns_about_effective_lam():
    """When the grid is narrow enough that `min_interior_knots` (not
    `knot_spacing_kms`) sets the knot count, the resulting knot spacing is
    much finer than the `knot_spacing_kms` reference, so the scaling
    amplifies `lam` far past its calibrated value. `FitConfig` should warn
    about this rather than let it pass silently. The validated NSIG=7
    convention does not trigger this warning even down to sigma~20 km/s.

    **The threshold is on each penalty order's own lam scale** (2026-07):
    an order-3 penalty is far weaker per unit lam and its validated grid
    tops out at 32768, so a single absolute constant -- calibrated when
    order 2 was the only option, where lam~2 is the sweet spot -- fired on
    every ordinary order-3 fit and called correct values "severe
    oversmoothing". Both orders are checked here so the threshold can't
    silently drift back to being meaningful for only one of them.
    """
    import warnings

    from kinextract.config import _reset_knot_spacing_warning_dedup

    narrow = dict(vmin=80 - 2 * 40, vmax=80 + 2 * 40)
    safe = dict(vmin=80 - 7 * 20, vmax=80 + 7 * 20)

    # This warning is deduped to once per process (see config.py's own
    # _KNOT_SPACING_WARNING_SHOWN docstring for why) -- reset before each
    # sub-case below so each is independently observable, the same way a
    # fresh process would see the first occurrence of each.

    # order 2: lam=10 is ~5x its own grid top (16) once amplified -> warns
    _reset_knot_spacing_warning_dedup()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FitConfig(penalty_order=2, lam=10.0, lam_selection="fixed", **narrow)
    assert any("effective lam" in str(w.message) for w in caught)

    # order 3: the SAME lam is an ordinary, well-regularized value on its
    # own scale, and must not warn...
    _reset_knot_spacing_warning_dedup()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FitConfig(penalty_order=3, lam=10.0, lam_selection="fixed", **narrow)
    assert not any("effective lam" in str(w.message) for w in caught)

    # ...but the check must still fire for a genuinely extreme order-3 value.
    _reset_knot_spacing_warning_dedup()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FitConfig(penalty_order=3, lam=1e6, lam_selection="fixed", **narrow)
    assert any("effective lam" in str(w.message) for w in caught)

    # The validated NSIG=7 convention stays clean at either order.
    for order in (2, 3):
        _reset_knot_spacing_warning_dedup()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FitConfig(penalty_order=order, **safe)
        assert not any("effective lam" in str(w.message) for w in caught)
