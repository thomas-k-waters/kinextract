"""dev_notes/VALIDATION_CAMPAIGN.md Stage 3: opt-in features tested *in combination*,
not just alone. Each opt-in feature (lam_wing, shape_aware_correction,
noise_scale) already has its own dedicated test(s) elsewhere confirming
it does what its docstring claims in isolation -- these tests instead
confirm the combinations don't crash, don't produce non-finite/negative
output, and that each feature's own known effect survives the other
feature(s) being active too (not necessarily unchanged in magnitude, but
not reversed or destroyed). Not a fresh statistical validation of each
combination at full power -- see dev_notes/VALIDATION_CAMPAIGN.md for the broader
grid campaign these targeted checks are a regression-safety complement to.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth,
    compute_losvd_moments, bootstrap_losvd, calibrate_noise_scale,
)
from kinextract.multi_template import fit_losvd


def _mock_single_template(vel, sig, snr=30.0, seed=0, lam_wing=0.0):
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

    cfg = FitConfig(vmin=vel - 7 * sig, vmax=vel + 7 * sig, lam_wing=lam_wing)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=seed)
    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w, y_w = A_stack / errors[None, :, None], galaxy / errors
    return A_stack_w, y_w, cfg, v_grid


def test_lam_wing_and_shape_aware_correction_compose_without_crashing():
    """Both opt-in fixes active at once on the exact low-S/N/low-sigma
    case each is individually validated on (see test_losvd.py's
    lam_wing test and test_gh_moments.py's shape_aware_correction
    tests) -- confirms they don't interact destructively (e.g. one
    undoing the other, or the pair producing non-finite output).
    """
    A_stack_w, y_w, cfg, v_grid = _mock_single_template(0.0, 50.0, snr=30.0, seed=1, lam_wing=3e5)
    result = fit_losvd(A_stack_w, y_w, cfg)
    m = compute_losvd_moments(v_grid, result["b"], shape_aware_correction=True)
    assert np.isfinite(m["vherm"]) and np.isfinite(m["sherm"])
    assert m["sherm"] > 0
    assert m["success"]


@pytest.mark.slow
def test_noise_scale_and_lam_wing_compose_in_bootstrap():
    """Injecting extra bootstrap-replicate noise (noise_scale) while
    lam_wing is also active in the underlying fit -- confirms the
    combination runs cleanly and lam_wing's own point-estimate behavior
    (checked via result0, which noise_scale must never touch) is
    unaffected by noise_scale being set.

    Genuinely expensive (`lam_wing`'s own wing-taper mechanism re-fits
    per bootstrap replicate, confirmed directly: ~230s for n_boot=6 alone,
    JIT compilation is not the driver -- `_neg_log_post` compiles only
    twice across all 6 replicates, not once per replicate) -- marked
    `slow` rather than trimmed, since the combination this test exists to
    check is real and worth keeping.

    `noise_scale=1.0` vs `4.0` (not `2.0`) deliberately -- found directly
    (2026-08-05): at n_boot=6, a 2.0 contrast's true effect on `sherm_std`
    (real, usually ~2-3x, confirmed across 5 independent seeds) is close
    enough to the sampling noise of a 6-replicate standard deviation that
    it reversed outright in 1/5 seeds tested, and reversed for real on a
    GitHub Actions Python 3.11 run with this test's own fixed seed (while
    passing locally on macOS/Python 3.13 with the same seed) -- a
    platform/library-version-sensitive coin flip, not a real regression.
    A wider contrast doesn't fix the small-n statistics, but makes the
    true effect large enough to reliably dominate that noise instead of
    re-seeding until it happens to pass.
    """
    A_stack_w, y_w, cfg, v_grid = _mock_single_template(0.0, 50.0, snr=30.0, seed=2, lam_wing=3e5)
    boot_ns1 = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=6, seed=5, noise_scale=1.0, reselect_lam=False)
    boot_ns2 = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=6, seed=5, noise_scale=4.0, reselect_lam=False)
    np.testing.assert_array_equal(boot_ns1["result0"]["b"], boot_ns2["result0"]["b"])
    assert np.all(np.isfinite(boot_ns2["b_boot"])) and np.all(boot_ns2["b_boot"] > 0)
    assert boot_ns2["sherm_std"] > boot_ns1["sherm_std"]


def test_noise_scale_and_shape_aware_correction_compose():
    """noise_scale only affects bootstrap replicate noise injection;
    shape_aware_correction only affects the moments computed from a
    given b(v). Confirms combining them (noise_scale active on the
    bootstrap, shape_aware_correction active on the resulting point
    estimate's own moments) still produces sane, finite output.
    """
    A_stack_w, y_w, cfg, v_grid = _mock_single_template(0.0, 60.0, snr=40.0, seed=3)
    result = fit_losvd(A_stack_w, y_w, cfg)
    A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
    scale = calibrate_noise_scale(A_eff_w, y_w, result["b"])
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=6, seed=7, noise_scale=scale, reselect_lam=False)
    m_sac = compute_losvd_moments(v_grid, boot["result0"]["b"], shape_aware_correction=True)
    assert np.isfinite(m_sac["sherm"]) and m_sac["sherm"] > 0
    assert np.isfinite(boot["sherm_std"]) and boot["sherm_std"] > 0


def test_multi_template_noise_scale_composition():
    """noise_scale on bootstrap_losvd with a genuine multi-template
    mixture, using the effective (weight-collapsed) design matrix for
    calibrate_noise_scale, exactly as its own docstring recommends.
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    ca_centers = [8498.02, 8542.09, 8662.14]

    def make_template(depths, width):
        t = np.ones(n_pix)
        for cen, d in zip(ca_centers, depths):
            t -= d * np.exp(-0.5 * ((wavelength - cen) / width) ** 2)
        return t

    templates = [make_template([0.30, 0.65, 0.35], 5.0), make_template([0.60, 0.25, 0.55], 3.0),
                 make_template([0.45, 0.45, 0.20], 7.0)]
    true_w = np.array([0.5, 0.3, 0.2])

    vel, sig = 0.0, 140.0
    cfg = FitConfig(vmin=vel - 7 * sig, vmax=vel + 7 * sig)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    galaxy_noiseless = sum(true_w[k] * (A_stack[k] @ b_truth) for k in range(3))
    noise_level = np.median(galaxy_noiseless) / 30.0
    rng = np.random.default_rng(9)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors

    result0 = fit_losvd(A_stack_w, y_w, cfg)
    A_eff0 = np.tensordot(result0["w"], A_stack_w, axes=([0], [0]))
    scale = calibrate_noise_scale(A_eff0, y_w, result0["b"])
    assert np.isfinite(scale) and scale > 0

    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=6, seed=11, noise_scale=scale, reselect_lam=False)
    assert np.all(np.isfinite(boot["w_boot"])) and np.all(boot["w_boot"] >= 0)
    assert np.isfinite(boot["sherm_std"]) and boot["sherm_std"] > 0


def test_all_opt_in_single_template_features_active_simultaneously():
    """Sanity only, not a bias claim: lam_wing + shape_aware_correction +
    noise_scale all active at once on one fit -- confirms nothing
    catastrophically interacts (crash, NaN, negative b, negative std).
    """
    A_stack_w, y_w, cfg, v_grid = _mock_single_template(0.0, 50.0, snr=30.0, seed=4, lam_wing=3e5)
    result0 = fit_losvd(A_stack_w, y_w, cfg)
    A_eff_w = np.tensordot(result0["w"], A_stack_w, axes=([0], [0]))
    scale = calibrate_noise_scale(A_eff_w, y_w, result0["b"])
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=6, seed=13, noise_scale=scale, reselect_lam=False)
    m = compute_losvd_moments(v_grid, boot["result0"]["b"], shape_aware_correction=True)

    assert np.all(np.isfinite(boot["b_boot"])) and np.all(boot["b_boot"] > 0)
    assert np.isfinite(boot["sherm_std"]) and boot["sherm_std"] > 0
    assert np.isfinite(m["vherm"]) and np.isfinite(m["sherm"]) and m["sherm"] > 0
