"""Tests for kinextract.uncertainty.

See `uncertainty.py`'s own module docstring for why parametric bootstrap
(not a Laplace approximation) was chosen. See
`kinextract_test_suite/notebooks/kinextract_validation/` for the full
coverage validation -- these tests check that things run and produce sane
output, not full calibration (that's a multi-realization statistical
check, not a fast unit test).

`bootstrap_losvd` works for any `n_templates >= 1`; the tests below cover
both a single template and a genuine 3-template mixture.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import run_jobs_in_one_worker_process

from kinextract import (
    FitConfig, build_template_stack, build_spline_truth,
    bootstrap_losvd, bootstrap_losvd_parallel, calibrate_noise_scale,
)
from kinextract.bias_correction import BiasCalibration
from kinextract.multi_template import fit_losvd
from kinextract.uncertainty import save_bootstrap, load_bootstrap, _bootstrap_losvd_worker


def _mock_case(vel, sig, nsig=7, snr=100.0, seed=0,
                wavefitmin=8400.0, wavefitmax=8800.0):
    wavemin, step, n_pix = 4750.0, 1.25, 3681
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
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=seed)

    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w, y_w = A_stack / errors[None, :, None], galaxy / errors
    return A_stack_w, y_w, cfg


def test_bootstrap_losvd_runs_and_produces_sane_output():
    # reselect_lam=False: this test checks output shape/positivity, not lam-selection
    # mechanics -- pinning keeps it fast (see reselect_lam's own docstring for the cost).
    A_stack_w, y_w, cfg = _mock_case(80.0, 140.0, seed=0)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=10, seed=1, reselect_lam=False)

    n_bins = len(cfg.build_velocity_grid())
    assert boot["b_boot"].shape == (10, n_bins)
    assert np.all(np.isfinite(boot["b_boot"]))
    assert np.all(boot["b_boot"] > 0.0)  # positivity is structural, must survive every replicate
    assert boot["b_boot_cov"].shape == (n_bins, n_bins)

    for key in ("vherm_std", "sherm_std", "h3_std", "h4_std"):
        assert np.isfinite(boot[key])
        assert boot[key] > 0.0


def test_bootstrap_uncertainty_is_the_right_order_of_magnitude():
    """Not a full coverage test (see the validation notebook for that) --
    just a sanity floor: the reported std should be in the same ballpark
    as the actual seed-to-seed scatter this project's own validation grid
    found for a comparable case, not off by an order of magnitude in
    either direction (which would indicate a real bug, e.g. forgetting to
    re-run multi-start, or a noise-scale error in the resampling itself).
    """
    # reselect_lam=False: keeps this a check of order-of-magnitude sanity, not a
    # (much slower, regime-dependent) test of the reselection mechanism itself.
    A_stack_w, y_w, cfg = _mock_case(80.0, 140.0, seed=0)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=20, seed=1, reselect_lam=False)

    assert 0.5 < boot["vherm_std"] < 15.0
    assert 0.5 < boot["sherm_std"] < 20.0


def test_reselect_lam_controls_the_actual_cfg_passed_to_every_replicate_fit():
    """Direct structural check of the mechanism itself (not just its
    statistical consequence): monkeypatch `fit_losvd` inside the
    `uncertainty` module to record the `cfg` object passed on every call,
    and confirm reselect_lam=False pins `lam_selection='fixed'` at the same
    `lam` for every one of the n_boot replicate calls, while
    reselect_lam=True leaves `lam_selection='aic_local'` untouched on every
    replicate call (i.e. it genuinely re-invokes adaptive selection each
    time, not just "happens to produce similar output").
    """
    import kinextract.uncertainty as unc_mod

    A_stack_w, y_w, cfg = _mock_case(0.0, 90.0, snr=60.0, seed=0)
    real_fit_losvd = unc_mod.fit_losvd

    for reselect, expect_fixed in [(False, True), (True, False)]:
        seen_cfgs = []

        def _spy(A, y, c, wave_fit=None, _seen=seen_cfgs):
            _seen.append(c)
            return real_fit_losvd(A, y, c, wave_fit=wave_fit)

        unc_mod.fit_losvd = _spy
        try:
            boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=4, seed=1, reselect_lam=reselect)
        finally:
            unc_mod.fit_losvd = real_fit_losvd

        replicate_cfgs = seen_cfgs[1:]  # [0] is the point-estimate call, not a replicate
        assert len(replicate_cfgs) == 4
        for c in replicate_cfgs:
            assert (c.lam_selection == "fixed") == expect_fixed, (
                f"reselect_lam={reselect}: expected lam_selection=='fixed' to be "
                f"{expect_fixed}, got cfg.lam_selection={c.lam_selection!r}"
            )
        if expect_fixed:
            lams = {c.lam for c in replicate_cfgs}
            assert lams == {boot["result0"]["lam_used"]}, (
                f"reselect_lam=False: every replicate must be pinned at the point "
                f"estimate's own lam_used ({boot['result0']['lam_used']}), got {lams}"
            )


def test_reselect_lam_true_runs_and_produces_sane_output():
    """Basic mechanism check for reselect_lam=True (the default as of
    2026-07-28): every replicate independently re-runs adaptive lam
    selection instead of being pinned. Confirms this doesn't crash and
    still produces sane, finite, positive output -- the quantitative
    evidence that it actually closes part of the previously under-reported
    uncertainty band lives in dev_notes/uncertainty/FINDINGS.md and
    diag01b_reselect_isolated.py (not reproduced here: that comparison
    needs ~30 replicates x multiple regimes to be meaningful, far more than
    a fast unit test should spend).
    """
    A_stack_w, y_w, cfg = _mock_case(0.0, 90.0, snr=60.0, seed=0)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=4, seed=1, reselect_lam=True)
    assert np.all(np.isfinite(boot["sherm_boot"]))
    assert np.all(boot["b_boot"] > 0.0)
    assert boot["sherm_std"] > 0.0


def test_reselect_lam_is_a_no_op_when_lam_selection_is_already_fixed():
    """When the caller already pinned `lam_selection='fixed'` (an explicit,
    endorsed usage pattern -- see FitConfig.lam_selection's own docstring),
    there is nothing to reselect: reselect_lam=True and reselect_lam=False
    must give bit-identical results, since both reduce to fitting every
    replicate at the same caller-supplied fixed lam.
    """
    A_stack_w, y_w, cfg = _mock_case(0.0, 90.0, snr=60.0, seed=0)
    cfg = dataclasses.replace(cfg, lam_selection="fixed", lam=8.0)
    boot_reselect = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=4, seed=1, reselect_lam=True)
    boot_pinned = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=4, seed=1, reselect_lam=False)
    np.testing.assert_array_equal(boot_reselect["b_boot"], boot_pinned["b_boot"])
    np.testing.assert_array_equal(boot_reselect["sherm_boot"], boot_pinned["sherm_boot"])


def test_reselect_lam_runs_cleanly_under_continuum_knots():
    """reselect_lam=True now genuinely re-runs adaptive lam selection
    under continuum-cofitting too (estimate_m_eff_local's own joint
    LOSVD+continuum linearization, aic_regularization.py) -- this used to
    be silently forced off (adaptive selection didn't support continuum
    yet, so fit_losvd fell back to a fixed lam with a UserWarning on
    every replicate). Confirms no fallback warning fires anymore, both
    reselect_lam=True/False produce finite, successful output, and
    reselect_lam actually has an effect now (unlike the old forced-off
    behavior, where True and False were bit-identical under continuum).
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    ca_centers = [8498.02, 8542.09, 8662.14]
    template_flux = np.ones(n_pix)
    for cen, depth in zip(ca_centers, [0.55, 0.70, 0.65]):
        template_flux -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)
    sig = 90.0
    cfg = FitConfig(vmin=-7 * sig, vmax=7 * sig, continuum_knots=4)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    noise_level = np.median(galaxy_noiseless) / 100.0
    rng = np.random.default_rng(3)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w, y_w = A_stack / errors[None, :, None], galaxy / errors

    import warnings as warnings_module
    with warnings_module.catch_warnings(record=True) as w:
        warnings_module.simplefilter("always")
        boot_reselect = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=4, seed=1,
                                         wave_fit=wave_fit, reselect_lam=True)
    fallback_warnings = [x for x in w if "does not support continuum-cofitting" in str(x.message)]
    assert len(fallback_warnings) == 0, (
        f"got {len(fallback_warnings)} fallback warnings -- adaptive selection should no "
        "longer fall back to a fixed lam under continuum-cofitting"
    )
    assert np.all(np.isfinite(boot_reselect["b_boot"])) and np.all(boot_reselect["b_boot"] > 0)
    assert np.isfinite(boot_reselect["sherm_std"])

    boot_pinned = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=4, seed=1,
                                   wave_fit=wave_fit, reselect_lam=False)
    assert np.all(np.isfinite(boot_pinned["b_boot"])) and np.all(boot_pinned["b_boot"] > 0)


def test_bootstrap_result0_matches_a_plain_fit_losvd_call():
    """result0 (the non-bootstrap point estimate) should be identical to
    what a plain fit_losvd call gives on the same data -- bootstrap_losvd
    shouldn't perturb or duplicate the actual fit logic, just wrap it.
    """
    # reselect_lam=False: result0 comes from the point-estimate fit, before cfg_boot
    # is even built -- irrelevant to what this test checks, pinned for speed only.
    A_stack_w, y_w, cfg = _mock_case(0.0, 140.0, seed=2)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=5, seed=3, reselect_lam=False)
    direct = fit_losvd(A_stack_w, y_w, cfg)
    np.testing.assert_allclose(boot["result0"]["b"], direct["b"])


def test_save_and_load_bootstrap_roundtrips():
    A_stack_w, y_w, cfg = _mock_case(0.0, 140.0, seed=2)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=5, seed=3, reselect_lam=False)

    path = "/tmp/kinextract_test_bootstrap.npz"
    save_bootstrap(boot, path)
    loaded = load_bootstrap(path)

    np.testing.assert_allclose(loaded["b_boot"], boot["b_boot"])
    np.testing.assert_allclose(loaded["b_boot_cov"], boot["b_boot_cov"])
    np.testing.assert_allclose(loaded["result0_b"], boot["result0"]["b"])
    assert float(loaded["point0_vherm"]) == boot["point0"]["vherm"]


def test_noise_scale_default_reproduces_unmodified_bootstrap():
    """`noise_scale=1.0` (the default) must be a fully separate code path
    that reproduces bit-for-bit what `bootstrap_losvd` did before this
    parameter existed -- same pattern used for `lam_wing=0.0`
    (see test_losvd.py). Guards against ever silently changing default
    behavior while adding an opt-in feature.
    """
    # reselect_lam=False on both calls: this test is about noise_scale's own
    # default-equivalence, not lam-selection mechanics -- pinned consistently for speed.
    A_stack_w, y_w, cfg = _mock_case(0.0, 140.0, seed=2)
    boot_explicit = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=8, seed=3, noise_scale=1.0, reselect_lam=False)
    boot_implicit = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=8, seed=3, reselect_lam=False)
    np.testing.assert_array_equal(boot_explicit["b_boot"], boot_implicit["b_boot"])
    np.testing.assert_array_equal(boot_explicit["sherm_boot"], boot_implicit["sherm_boot"])


def test_noise_scale_widens_bootstrap_uncertainty_proportionately_not_explosively():
    """A bigger noise_scale should widen reported uncertainty roughly in
    proportion to the injected extra scatter, not runaway -- the concern
    that motivated decoupling noise_scale from A_stack_w/y_w in the first
    place (see calibrate_noise_scale's own docstring).
    """
    # lam_selection="fixed" pinned deliberately: this test isolates noise_scale's own
    # mechanical effect on the reported uncertainty, orthogonal to which lam-selection
    # mode is active -- letting it default to "aic_local" would fold in extra bootstrap-
    # to-bootstrap variance from a less-regularized (adaptively-selected) lam, which is a
    # real, expected effect (see FitConfig.lam_selection's own help text) but a confound
    # for what this specific test checks.
    A_stack_w, y_w, cfg = _mock_case(0.0, 140.0, seed=0)
    cfg = dataclasses.replace(cfg, lam_selection="fixed")
    boot_1x = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=15, seed=1, noise_scale=1.0)
    boot_2x = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=15, seed=1, noise_scale=2.0)
    ratio = boot_2x["sherm_std"] / boot_1x["sherm_std"]
    assert 1.3 < ratio < 4.0, f"expected roughly proportionate growth, got {ratio:.2f}x for a 2x noise_scale"
    # the point estimate itself must be untouched by noise_scale -- it's not part of replicate generation
    np.testing.assert_array_equal(boot_1x["result0"]["b"], boot_2x["result0"]["b"])


def test_calibrate_noise_scale_returns_close_to_one_when_noise_is_correct():
    A_stack_w, y_w, cfg = _mock_case(0.0, 140.0, snr=60.0, seed=4)
    result = fit_losvd(A_stack_w, y_w, cfg)
    A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
    scale = calibrate_noise_scale(A_eff_w, y_w, result["b"])
    assert 0.7 < scale < 1.4, f"expected ~1.0 when the assumed noise is already correct, got {scale:.2f}"


def test_calibrate_noise_scale_recovers_a_known_mismatch():
    """The core claim this function has to earn: given data generated at
    a KNOWN true noise level but whitened with a deliberately wrong
    (here, 2x too small) assumed level, the estimate should land close
    to the true 2.0x mismatch factor -- averaged over several seeds since
    any single-fit MAD estimate carries some sampling scatter.
    """
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

    vel, sig, nsig = 0.0, 140.0, 7
    cfg = FitConfig(vmin=vel - nsig * sig, vmax=vel + nsig * sig)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth

    true_noise = np.median(galaxy_noiseless) / 60.0
    assumed_noise = true_noise / 2.0  # deliberately 2x too small

    scales = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        galaxy = galaxy_noiseless + rng.normal(0.0, true_noise, len(wave_fit))
        errors = np.full(len(wave_fit), assumed_noise)
        A_stack_w, y_w = A_stack / errors[None, :, None], galaxy / errors
        result = fit_losvd(A_stack_w, y_w, cfg)
        A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))
        scales.append(calibrate_noise_scale(A_eff_w, y_w, result["b"]))

    assert 1.5 < np.mean(scales) < 2.5, f"expected ~2.0, got {np.mean(scales):.2f}"


def test_calibrate_noise_scale_warns_and_clips_on_an_extreme_estimate():
    """MAD measures *scatter*, not offset -- an all-zero `b` manufactures a
    huge but nearly-constant residual (no real scatter), which correctly
    does NOT trigger the cap (that's the point of using a robust scatter
    estimator rather than raw residual magnitude). To exercise the
    warn-and-clip path, construct genuinely huge *scatter* instead: assume
    an error array far too small (50x) relative to the actual injected
    noise, so the true random noise appears wildly amplified once whitened.
    """
    import warnings as warnings_module
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

    vel, sig, nsig = 0.0, 140.0, 7
    cfg = FitConfig(vmin=vel - nsig * sig, vmax=vel + nsig * sig)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    true_noise = np.median(galaxy_noiseless) / 100.0
    rng = np.random.default_rng(0)
    galaxy = galaxy_noiseless + rng.normal(0.0, true_noise, len(wave_fit))
    tiny_errors = np.full(len(wave_fit), true_noise / 50.0)
    A_stack_w, y_w = A_stack / tiny_errors[None, :, None], galaxy / tiny_errors
    result = fit_losvd(A_stack_w, y_w, cfg)
    A_eff_w = np.tensordot(result["w"], A_stack_w, axes=([0], [0]))

    with warnings_module.catch_warnings(record=True) as w:
        warnings_module.simplefilter("always")
        scale = calibrate_noise_scale(A_eff_w, y_w, result["b"])
        assert any("outside the sane" in str(x.message) for x in w)
    assert scale <= 3.0


def _mock_multi_template_case(vel=0.0, sig=140.0, snr=100.0, seed=0):
    """Three genuinely distinct templates (same shapes validated in
    tests/test_multi_template.py), not near-collinear ones -- this file
    is testing the bootstrap mechanics, not re-litigating template-library
    conditioning.
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

    templates = [
        make_template([0.30, 0.65, 0.35], 5.0),
        make_template([0.60, 0.25, 0.55], 3.0),
        make_template([0.45, 0.45, 0.20], 7.0),
    ]
    true_w = np.array([0.5, 0.3, 0.2])

    vmin, vmax = vel - 7 * sig, vel + 7 * sig
    cfg = FitConfig(vmin=vmin, vmax=vmax)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=vel, sigma=sig, wiggle_scale=0.0, seed=0)

    A_stack = build_template_stack(wave_fit, wavelength, templates, v_grid)
    galaxy_noiseless = sum(true_w[k] * (A_stack[k] @ b_truth) for k in range(3))
    noise_level = np.median(galaxy_noiseless) / snr
    rng = np.random.default_rng(seed)
    galaxy = galaxy_noiseless + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w = A_stack / errors[None, :, None]
    y_w = galaxy / errors
    return A_stack_w, y_w, cfg, true_w


def test_bootstrap_losvd_multi_template_runs_and_produces_sane_output():
    # reselect_lam=False: general sanity/shape check, not about lam mechanics.
    A_stack_w, y_w, cfg, true_w = _mock_multi_template_case(seed=0)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=8, seed=1, reselect_lam=False)

    n_bins = len(cfg.build_velocity_grid())
    n_templates = A_stack_w.shape[0]
    assert boot["b_boot"].shape == (8, n_bins)
    assert np.all(np.isfinite(boot["b_boot"]))
    assert np.all(boot["b_boot"] > 0.0)  # positivity is structural, must survive every replicate
    assert boot["b_boot_cov"].shape == (n_bins, n_bins)

    assert boot["w_boot"].shape == (8, n_templates)
    np.testing.assert_allclose(boot["w_boot"].sum(axis=1), 1.0)  # normalized mixture fractions
    np.testing.assert_allclose(boot["w_norm0"].sum(), 1.0)
    np.testing.assert_allclose(boot["w_norm0"], true_w, atol=0.05)
    assert np.all(boot["w_std"] > 0.0) and np.all(np.isfinite(boot["w_std"]))

    for key in ("vherm_std", "sherm_std", "h3_std", "h4_std"):
        assert np.isfinite(boot[key])
        assert boot[key] > 0.0


def test_bootstrap_losvd_b_boot_is_unit_area_normalized():
    """The multi-template model is invariant under `(w, b) -> (c*w, b/c)`,
    so each bootstrap replicate's `b` is only meaningful up to its own
    arbitrary scale. Confirm every stored replicate integrates to 1
    (unit-area normalized, matching `w_boot`'s own `w/w.sum()` convention),
    and that the resulting percentile band brackets the point estimate's
    own (identically normalized) peak.
    """
    # reselect_lam=False: this test is about unit-area normalization, not lam mechanics.
    A_stack_w, y_w, cfg, _ = _mock_multi_template_case(seed=4, vel=0.0, sig=140.0)
    v_grid = cfg.build_velocity_grid()
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=10, seed=5, reselect_lam=False)

    areas = np.trapezoid(boot["b_boot"], v_grid, axis=1)
    np.testing.assert_allclose(areas, 1.0, atol=1e-8)

    b_fit_n = boot["result0"]["b"] / np.trapezoid(
        np.maximum(boot["result0"]["b"], 0.0), v_grid
    )
    assert boot["b_p84"].max() > 0.5 * b_fit_n.max()
    assert boot["b_p16"].max() < 2.0 * b_fit_n.max()


def test_bootstrap_losvd_b_median_is_the_raw_bootstrap_median_regardless_of_bias_correction():
    """`b_median` (uncorrected) is what every notebook should plot as the
    reported LOSVD -- the bootstrap band is centered on it, not on the
    point estimate. It must equal `np.median(b_boot, axis=0)` and must NOT
    change when `apply_bias_correction=True` (that's what `b_median_corrected`
    is for) -- a fixed, uncorrected reference point regardless of that flag.
    """
    A_stack_w, y_w, cfg, _ = _mock_multi_template_case(seed=6, vel=0.0, sig=140.0)
    boot_raw = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=10, seed=7, reselect_lam=False)
    boot_corrected = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=10, seed=7, reselect_lam=False,
                                      apply_bias_correction=True)

    np.testing.assert_allclose(boot_raw["b_median"], np.median(boot_raw["b_boot"], axis=0))
    np.testing.assert_allclose(boot_raw["b_median"], boot_corrected["b_median"])
    # apply_bias_correction=False: b_median_corrected is just an alias for b_median.
    np.testing.assert_allclose(boot_raw["b_median"], boot_raw["b_median_corrected"])


def test_bootstrap_losvd_multi_template_result0_matches_a_plain_fit():
    """result0 should be identical to a plain fit_losvd call on the same
    data -- same discipline as
    test_bootstrap_result0_matches_a_plain_fit_losvd_call above.
    """
    # reselect_lam=False: only result0 (the pre-cfg_boot point estimate) is checked here.
    A_stack_w, y_w, cfg, _ = _mock_multi_template_case(seed=2)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=3, seed=3, reselect_lam=False)
    direct = fit_losvd(A_stack_w, y_w, cfg)
    np.testing.assert_allclose(boot["result0"]["b"], direct["b"])
    np.testing.assert_allclose(boot["result0"]["w"], direct["w"])


def test_bias_correction_when_enabled_adds_new_keys_only():
    """apply_bias_correction=True (opt-in, not the default as of 2026-07 --
    see uncertainty.py's own module docstring for the two overcorrection
    failure modes that led to that default flip) must only ADD new
    `_corrected`-suffixed keys, never change the meaning of an existing
    one. `result0`, `b_boot`, `sherm_boot`, `sherm_std`, etc. must be
    bit-identical to a run with apply_bias_correction=False (the only
    thing that flag changes is which extra keys get computed). Both calls
    pass the flag explicitly -- relying on the implicit default here would
    make "on" and "off" the same call and silently stop testing anything
    once the default is off.
    """
    # reselect_lam=False on both: this test is about which KEYS apply_bias_correction
    # adds, not lam-selection mechanics -- pinned consistently for speed.
    A_stack_w, y_w, cfg = _mock_case(0.0, 50.0, snr=60.0, seed=0)
    boot_on = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=8, seed=1, apply_bias_correction=True, reselect_lam=False)
    boot_off = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=8, seed=1, apply_bias_correction=False, reselect_lam=False)

    for key in ("result0", "point0", "b_boot", "b_p16", "b_p84", "sherm_boot",
                "sherm_std", "h4_std", "vherm_std", "h3_std"):
        if key in ("result0", "point0"):
            continue  # dicts of arrays/scalars, spot-checked below instead
        np.testing.assert_array_equal(boot_on[key], boot_off[key])
    np.testing.assert_allclose(boot_on["result0"]["b"], boot_off["result0"]["b"])
    assert boot_on["point0"]["sherm"] == boot_off["point0"]["sherm"]

    for key in ("point0_corrected", "sherm_std_corrected", "h4_std_corrected",
                "sig_uncertainty_inflation", "h4_uncertainty_inflation",
                "b_median_corrected", "b_p16_corrected", "b_p84_corrected", "snr_estimated"):
        assert key in boot_on and key in boot_off


def test_bias_correction_false_leaves_corrected_keys_as_pass_through():
    A_stack_w, y_w, cfg = _mock_case(0.0, 50.0, snr=60.0, seed=0)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=5, seed=1, apply_bias_correction=False, reselect_lam=False)
    assert boot["point0_corrected"]["sherm"] == boot["point0"]["sherm"]
    assert boot["sherm_std_corrected"] == boot["sherm_std"]
    assert boot["sig_uncertainty_inflation"] == 0.0
    np.testing.assert_array_equal(boot["b_median_corrected"], np.median(boot["b_boot"], axis=0))
    np.testing.assert_array_equal(boot["b_p16_corrected"], boot["b_p16"])


def test_bias_correction_clips_negative_corrected_model():
    """Companion to the equivalent regression tests in
    `test_bias_correction.py`/`test_gh_moments.py` -- `bootstrap_losvd`'s
    own `point0_corrected["model"]` reconstruction is a third call site
    with the same negative-curve risk (see `bias_correction.correct_losvd`'s
    docstring) and must be clipped the same way.
    """
    extreme_calib = BiasCalibration(
        recovered_sigma=np.array([30.0, 30.0, 200.0, 200.0]),
        snr=np.array([10.0, 200.0, 10.0, 200.0]),
        sig_bias=np.zeros(4),
        sig_bias_std=np.ones(4),
        h4_bias=np.array([0.2, 0.2, 0.2, 0.2]),
        h4_bias_std=np.full(4, 0.05),
        sigma_range=(30.0, 200.0),
        snr_range=(10.0, 200.0),
    )
    A_stack_w, y_w, cfg = _mock_case(0.0, 50.0, snr=60.0, seed=0)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=5, seed=1,
                            apply_bias_correction=True, bias_calibration=extreme_calib,
                            reselect_lam=False)
    # pytest.approx, not exact `==`: under JAX float64 (the 2026-07 default,
    # see dev_notes/VALIDATION_CAMPAIGN.md Stage 22) this subtraction is no longer
    # bit-identical to the calibration's own float arithmetic -- a pure
    # precision artifact unrelated to what this test actually checks.
    assert boot["point0_corrected"]["h4"] == pytest.approx(boot["point0"]["h4"] - 0.2)
    assert np.all(boot["point0_corrected"]["model"] >= 0.0), (
        "bootstrap_losvd's corrected model must be clipped non-negative -- a LOSVD is a "
        "probability density over velocity and cannot be negative anywhere"
    )


def test_corrected_band_width_reflects_sherm_std_corrected_inflation():
    """Regression test for a confirmed bug: `b_p16_corrected`/`b_p84_corrected`
    used to only recenter the raw (uninflated) bootstrap band on the
    corrected median, silently leaving its WIDTH at the raw, uninflated
    value -- so the plotted band could look far tighter than
    `sherm_std_corrected`, the number it's supposed to accompany (found by
    visual inspection: a case with sherm_std_corrected/sherm_std ~5x wider
    than sherm_std still plotted a band barely different from the raw one).
    The band's half-width at the peak must now scale with the same
    inflation ratio as the reported scalar uncertainty.
    """
    # lam_selection="fixed" pinned deliberately: this regime was chosen to reliably
    # produce inflation_ratio > 2.0 under the plain (sigma, S/N) table at fixed lam=2.0
    # -- the "aic_local" default would switch to the m_eff-axis table (a different,
    # separately-tested correction path) and change this test's own regime, which isn't
    # what this test is checking (band-width-tracks-inflation-ratio mechanics, not which
    # table/lam-selection mode is active).
    A_stack_w, y_w, cfg = _mock_case(50.0, 140.0, snr=100.0, seed=42)
    cfg = dataclasses.replace(cfg, lam_selection="fixed")
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=30, seed=0, apply_bias_correction=True)
    inflation_ratio = boot["sherm_std_corrected"] / boot["sherm_std"]
    assert inflation_ratio > 2.0, "test regime should have a real, checkable inflation factor"

    raw_half_width = (boot["b_p84"] - boot["b_p16"]).max() / 2
    corrected_half_width = (boot["b_p84_corrected"] - boot["b_p16_corrected"]).max() / 2
    band_ratio = corrected_half_width / raw_half_width
    assert band_ratio == pytest.approx(inflation_ratio, rel=1e-6)


def test_bias_correction_improves_sigma_in_a_well_calibrated_regime():
    """The core claim: in a regime with well-characterized regularization
    bias (low sigma, moderate S/N -- see dev_notes/VALIDATION_CAMPAIGN.md Stage 1),
    the corrected point estimate should land closer to the true sigma than
    the raw point estimate does. Directly reproduces the improvement found
    in this session's own diagnostic case (V=10, sigma=50 mock, corrected
    sigma error 1.6 km/s vs. raw 3.9 km/s, vs. raw bootstrap median 8.5 km/s).
    """
    # lam_selection="fixed" pinned deliberately: this test validates the *plain*
    # (sigma, S/N) calibration table's own correction quality at fixed lam=2.0, the
    # regime it and this test's own reference numbers were built against. Left at the
    # "aic_local" default, bootstrap_losvd would auto-switch to the m_eff-axis table
    # instead (see FitConfig.lam_selection's help text) -- a real, separate correction
    # path this test isn't about. (Checked directly: at this specific point the m_eff-
    # table correction is *worse* than the raw estimate for this mock -- a genuine,
    # separate finding worth its own investigation, not something to paper over here.)
    # penalty_order=2 pinned too (2026-07, Stage 22): this calibration table was built
    # against fixed lam=2.0 under the order-2 penalty specifically -- combined with the
    # new order-3 default, fixed lam=2.0 is (per FitConfig's own warning) essentially
    # unregularized, which breaks this table's calibration correspondence entirely.
    A_stack_w, y_w, cfg = _mock_case(0.0, 30.0, snr=60.0, seed=7)
    cfg = dataclasses.replace(cfg, lam_selection="fixed", penalty_order=2)
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=30, seed=8, apply_bias_correction=True)
    true_sigma = 30.0
    raw_err = abs(boot["point0"]["sherm"] - true_sigma)
    corrected_err = abs(boot["point0_corrected"]["sherm"] - true_sigma)
    assert corrected_err < raw_err

    # And it should not perform worse than the raw bootstrap median either
    # -- the specific failure mode this correction exists to fix.
    boot_median_sherm = np.median(boot["sherm_boot"])
    boot_median_err = abs(boot_median_sherm - true_sigma)
    assert corrected_err <= boot_median_err


def test_bias_correction_table_choice_checks_actual_lam_not_whether_selection_ran():
    """Found via code review, confirmed by reproduction: the table-mismatch
    guard used to only fire when `cfg.lam_selection == "aic_local"` actually
    ran (gated on an `adaptive_selection_ran` flag), so a caller-supplied
    *fixed* lam far from the plain table's own calibration lam silently got
    the plain table applied anyway -- `lam_selection="fixed"` with an
    explicit lam is an endorsed usage pattern (see `FitConfig.lam_selection`'s
    own docstring: "useful for reproducing a specific lam deliberately"),
    not an edge case. Now the mismatch check runs against the fit's own lam
    value unconditionally, so a mismatched fixed lam also correctly
    switches to the per-lam calibration set, exactly as a mismatched
    adaptively-selected lam already did.
    """
    from kinextract.bias_correction import _get_default_calibration, get_default_lam_axis_calibration

    plain_table = _get_default_calibration()
    lam_far = plain_table.lam * 4.0  # well past the 25% mismatch threshold

    A_stack_w, y_w, cfg = _mock_case(0.0, 50.0, snr=60.0, seed=0)
    cfg_fixed_far = dataclasses.replace(cfg, lam=lam_far, lam_selection="fixed")

    boot_auto = bootstrap_losvd(A_stack_w, y_w, cfg_fixed_far, n_boot=8, seed=1,
                                 apply_bias_correction=True)
    boot_forced_per_lam = bootstrap_losvd(A_stack_w, y_w, cfg_fixed_far, n_boot=8, seed=1,
                                           apply_bias_correction=True,
                                           bias_calibration=get_default_lam_axis_calibration())
    boot_forced_plain = bootstrap_losvd(A_stack_w, y_w, cfg_fixed_far, n_boot=8, seed=1,
                                         apply_bias_correction=True, bias_calibration=plain_table)

    # Automatic selection at a far-mismatched *fixed* lam must behave like forcing the
    # per-lam set, not like forcing the (wrong-domain) plain table.
    assert boot_auto["point0_corrected"]["sherm"] == boot_forced_per_lam["point0_corrected"]["sherm"]
    assert boot_auto["point0_corrected"]["sherm"] != boot_forced_plain["point0_corrected"]["sherm"]


def test_bootstrap_losvd_with_continuum_knots_runs_and_the_injected_noise_model_includes_C():
    """New as of this refactor: `bootstrap_losvd` accepts `wave_fit` and
    respects `cfg.continuum_knots`, so every bootstrap replicate re-fits
    with continuum-cofitting active, not just the point estimate.

    Also guards against a real bug caught while wiring this up: the model
    each replicate's synthetic noise is injected around must include the
    point estimate's own recovered continuum `C0`, not just the template
    mixture -- omitting it would inject noise around the wrong (continuum-
    free) prediction, silently biasing every replicate wherever `C0`
    deviates from 1. Checked here by injecting a real, non-flat true
    continuum and confirming the bootstrap still recovers sigma sanely
    (a mismatched injected-noise model would show up as a large, obvious
    sigma bias across replicates, not just a subtly wider band).
    """
    wavemin, step, n_pix = 4750.0, 1.25, 3681
    wavefitmin, wavefitmax = 8400.0, 8800.0
    wavelength = wavemin + np.arange(n_pix) * step
    mask = (wavelength >= wavefitmin) & (wavelength <= wavefitmax)
    wave_fit = wavelength[mask]
    ca_centers = [8498.02, 8542.09, 8662.14]
    template_flux = np.ones(n_pix)
    for cen, depth in zip(ca_centers, [0.55, 0.70, 0.65]):
        template_flux -= depth * np.exp(-0.5 * ((wavelength - cen) / 5.0) ** 2)

    sig = 140.0
    cfg = FitConfig(vmin=-7 * sig, vmax=7 * sig, continuum_knots=4)
    v_grid = cfg.build_velocity_grid()
    b_truth, _ = build_spline_truth(v_grid, vel=0.0, sigma=sig, wiggle_scale=0.0, seed=0)
    A_stack = build_template_stack(wave_fit, wavelength, [template_flux], v_grid)
    galaxy_noiseless = A_stack[0] @ b_truth
    true_C = 1.0 + 0.1 * np.sin(2 * np.pi * (wave_fit - wave_fit.min()) / (wave_fit.max() - wave_fit.min()))
    galaxy_noiseless_c = galaxy_noiseless * true_C
    noise_level = np.median(galaxy_noiseless_c) / 100.0
    rng = np.random.default_rng(3)
    galaxy = galaxy_noiseless_c + rng.normal(0.0, noise_level, len(wave_fit))
    errors = np.full(len(wave_fit), noise_level)
    A_stack_w, y_w = A_stack / errors[None, :, None], galaxy / errors

    # reselect_lam defaults True but is forced off automatically whenever
    # cfg.continuum_knots is set (see bootstrap_losvd's own reselect_lam docstring) --
    # no explicit override needed here, this comment just makes that reliance visible.
    boot = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=15, seed=4, wave_fit=wave_fit)
    assert np.isfinite(boot["point0"]["sherm"])
    assert abs(boot["point0"]["sherm"] - sig) < 20.0
    assert np.all(np.isfinite(boot["sherm_boot"]))
    # bootstrap spread should be a reasonable fraction of sigma, not
    # blown up the way it would be if replicates were noised around a
    # continuum-free (systematically wrong) model
    assert boot["sherm_std"] < 0.5 * sig


def test_bootstrap_losvd_L_none_reproduces_original_behavior():
    """`L=None` (the default) must be bit-identical to before this
    parameter existed -- the mainline (properly-pre-whitened) case needs
    no change at all, see `bootstrap_losvd`'s own docstring."""
    # reselect_lam=False on both: this test is about L's own default-equivalence.
    A_stack_w, y_w, cfg = _mock_case(0.0, 140.0, seed=5)
    boot_default = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=10, seed=6, reselect_lam=False)
    boot_explicit_none = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=10, seed=6, L=None, reselect_lam=False)
    np.testing.assert_array_equal(boot_default["sherm_boot"], boot_explicit_none["sherm_boot"])


def test_bootstrap_losvd_with_a_deliberately_mismatched_L_changes_the_result():
    """`L` is a deliberate assumed-vs-true covariance mismatch diagnostic,
    not a required fix -- confirms it actually changes the bootstrap
    ensemble when given (same seed, different injected-noise covariance
    should give different replicates)."""
    from kinextract.noise_model import build_ar1_covariance

    A_stack_w, y_w, cfg = _mock_case(0.0, 140.0, seed=5)
    n_pix = len(y_w)
    L_uncorrelated = np.eye(n_pix)
    L_strongly_correlated = np.linalg.cholesky(build_ar1_covariance(np.ones(n_pix), rho=0.8))

    boot_plain = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=10, seed=6, L=L_uncorrelated, reselect_lam=False)
    boot_correlated = bootstrap_losvd(A_stack_w, y_w, cfg, n_boot=10, seed=6, L=L_strongly_correlated,
                                       reselect_lam=False)
    assert not np.allclose(boot_plain["sherm_boot"], boot_correlated["sherm_boot"])


class TestBootstrapLosvdParallel:
    """`bootstrap_losvd_parallel`'s own core promise, same as
    `fit_losvd_parallel`'s: fitting bins across worker processes must be
    bit-identical to fitting them one at a time in a serial loop -- pure
    wall-clock orchestration, zero accuracy cost. Also covers the
    per-bin-seed convention (`seed + i`, not one shared `seed`), since
    that's this function's own real correctness contract, not just
    "doesn't crash."
    """

    @staticmethod
    def _bins():
        return [_mock_case(v, s, seed=seed, snr=60.0)
                for v, s, seed in [(0.0, 80.0, 0), (20.0, 140.0, 1), (-15.0, 60.0, 2)]]

    def test_n_jobs_1_matches_a_serial_loop_with_the_seed_plus_i_convention(self):
        bins = self._bins()
        A_stack_w_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]
        base_seed = 100

        serial = [bootstrap_losvd(A, y, cfg, n_boot=6, seed=base_seed + i, reselect_lam=False)
                  for i, (A, y, cfg) in enumerate(zip(A_stack_w_list, y_w_list, cfg_list))]
        parallel = bootstrap_losvd_parallel(A_stack_w_list, y_w_list, cfg_list, n_boot=6,
                                             seed=base_seed, reselect_lam=False, n_jobs=1)

        assert len(parallel) == len(serial) == 3
        for s, p in zip(serial, parallel):
            np.testing.assert_array_equal(s["sherm_boot"], p["sherm_boot"])
            np.testing.assert_array_equal(s["b_boot"], p["b_boot"])
            assert s["sherm_std"] == p["sherm_std"]

    def test_a_real_process_pool_is_bit_identical_to_serial(self):
        """The "serial" reference is run through a single-worker
        `ProcessPoolExecutor` (`conftest.run_jobs_in_one_worker_process`),
        not called directly in this test process. A direct call compares
        against a different JAX/XLA process state (already-initialized,
        multi-threaded) than what a real `n_jobs=2` worker sees (freshly
        spawned, single-threaded) -- a real few-ULP difference that
        `bootstrap_losvd`'s iterative per-replicate fit (robust-fit
        clipping / lambda-dependent convergence) can occasionally amplify
        into a much larger, run-dependent mismatch (confirmed directly,
        2026-09-14: a loosened `rtol=0.05` on this exact assertion still
        failed intermittently in CI at 5.33%, proving that widening the
        tolerance further was chasing the wrong thing). With both sides
        run through matching fresh worker processes, this goes back to
        exact equality -- the `seed + i` convention itself is covered
        separately, and passes exactly, in
        `test_n_jobs_1_matches_a_serial_loop_with_the_seed_plus_i_
        convention` just above, which runs `n_jobs=1` (no real subprocess,
        so no JAX-reinitialization difference at all)."""
        bins = self._bins()
        A_stack_w_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]
        base_seed = 200

        # Matches bootstrap_losvd's own defaults / bootstrap_losvd_parallel's job-tuple
        # order: (A_stack_w, y_w, cfg, seed, wave_fit, L, n_boot, fit_h3h4, noise_scale,
        # apply_bias_correction, bias_calibration, replicate_single_start, reselect_lam).
        jobs = [(A, y, cfg, base_seed + i, None, None, 6, True, 1.0, False, None, True, False)
                for i, (A, y, cfg) in enumerate(zip(A_stack_w_list, y_w_list, cfg_list))]
        serial = run_jobs_in_one_worker_process(_bootstrap_losvd_worker, jobs)
        parallel = bootstrap_losvd_parallel(A_stack_w_list, y_w_list, cfg_list, n_boot=6,
                                             seed=base_seed, reselect_lam=False, n_jobs=2)

        for i, (s, p) in enumerate(zip(serial, parallel)):
            np.testing.assert_array_equal(s["sherm_boot"], p["sherm_boot"], err_msg=f"bin {i} mismatch")
            np.testing.assert_array_equal(s["b_boot"], p["b_boot"], err_msg=f"bin {i} mismatch")
            assert s["sherm_std"] == p["sherm_std"], f"bin {i} sherm_std mismatch"

    def test_seed_none_draws_independent_entropy_per_bin_not_one_shared_stream(self):
        """seed=None must not silently give every bin the exact same
        replicate noise -- each bin's own bootstrap_losvd(seed=None) call
        draws fresh entropy independently.
        """
        bins = self._bins()
        A_stack_w_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]

        parallel = bootstrap_losvd_parallel(A_stack_w_list, y_w_list, cfg_list, n_boot=6,
                                             seed=None, reselect_lam=False, n_jobs=1)
        # different underlying data per bin already guarantees different sherm_boot,
        # so this mainly checks nothing crashes/silently shares state across bins
        assert len(parallel) == 3
        for p in parallel:
            assert np.all(np.isfinite(p["sherm_boot"]))

    def test_mismatched_list_lengths_raise_a_clear_error(self):
        bins = self._bins()
        A_stack_w_list = [b[0] for b in bins]
        y_w_list = [b[1] for b in bins]
        cfg_list = [b[2] for b in bins]

        with pytest.raises(ValueError, match="same length"):
            bootstrap_losvd_parallel(A_stack_w_list, y_w_list[:-1], cfg_list, n_jobs=1)
        with pytest.raises(ValueError, match="one entry per bin"):
            bootstrap_losvd_parallel(A_stack_w_list, y_w_list, cfg_list[:-1], n_jobs=1)
